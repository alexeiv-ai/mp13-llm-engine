"""Pure version-2 toolbox definition planning and post-resolution grouping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mp13_engine.mp13_intrinsics_metadata import intrinsic_dependency_metadata, intrinsic_metadata

from .bundle_models import (
    ResolvedToolboxProfileSpec,
    SandboxProfileSpec,
    ToolboxAutoAssignmentRequestV2,
    ToolboxBundleAutoTool,
    ToolboxBundleFile,
    ToolboxBundleSpec,
    ToolboxBundleTool,
    ToolboxDefinitionSpec,
    ToolboxManualAssignmentRequestV2,
)
from .catalog import ToolboxEnvironmentTemplateSpec
from .dependency_analysis import (
    ToolboxResolvedDependencies,
    ToolboxTemplateSelection,
    analyze_toolbox_bundle_imports,
    resolve_toolbox_dependencies,
    select_toolbox_environment_template,
)
from .identity import custom_lock_digest, environment_identity


@dataclass(frozen=True)
class ToolboxDefinitionPlanDraft:
    definition: ToolboxDefinitionSpec
    profiles: tuple[ResolvedToolboxProfileSpec, ...]
    bundles: tuple[ToolboxBundleSpec, ...]
    custom_environment_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.to_dict(),
            "definition_revision": self.definition.revision,
            "profiles": [item.to_dict() for item in self.profiles],
            "bundles": [item.manifest_payload() for item in self.bundles],
            "custom_environment_count": self.custom_environment_count,
        }


@dataclass(frozen=True)
class _ResolvedMember:
    kind: str
    stable_key: str
    request: ToolboxAutoAssignmentRequestV2 | ToolboxManualAssignmentRequestV2 | None
    selection: ToolboxTemplateSelection
    dependencies: ToolboxResolvedDependencies
    environment_key: str
    custom_digest: str | None
    sandbox_policy: Mapping[str, Any]
    assigned_tool_keys: tuple[str, ...]
    intrinsic_names: tuple[str, ...] = ()
    include_intrinsic_guides: bool = False

    @property
    def resolved_import_roots(self) -> tuple[str, ...]:
        return tuple(sorted(item.import_root for item in self.dependencies.analysis.imports))


def _resolve_member(
    *,
    kind: str,
    stable_key: str,
    request: ToolboxAutoAssignmentRequestV2 | ToolboxManualAssignmentRequestV2 | None,
    files: Sequence[ToolboxBundleFile],
    dependency_mode: str,
    requested_template_id: str | None,
    declared_imports: Sequence[str],
    package_requirements: Sequence[str],
    templates: Sequence[ToolboxEnvironmentTemplateSpec],
    python_abi: str,
    platform: str,
    runtime_identity: Mapping[str, Any],
    sandbox_policy: Mapping[str, Any],
    assigned_tool_keys: Sequence[str],
    intrinsic_names: Sequence[str] = (),
    include_intrinsic_guides: bool = False,
) -> _ResolvedMember:
    analysis = analyze_toolbox_bundle_imports(files, declared_imports=declared_imports)
    dependencies = resolve_toolbox_dependencies(
        analysis,
        package_requirements=package_requirements,
    )
    allowed = (requested_template_id,) if requested_template_id else None
    selection = select_toolbox_environment_template(
        dependencies,
        templates,
        python_abi=python_abi,
        platform=platform,
        allowed_template_ids=allowed,
    )
    if dependency_mode == "template" and selection.mode != "template":
        raise ValueError("definition_explicit_template_incomplete")
    if dependency_mode == "custom" and selection.mode != "custom":
        raise ValueError("definition_custom_delta_empty")
    custom_digest = None
    if selection.mode == "custom":
        custom_digest = custom_lock_digest(
            {
                "base_template_lock_digest": selection.template.lock_digest,
                "distributions": [item.to_dict() for item in selection.custom_delta],
                "artifacts": [],
            }
        )
    environment_key = environment_identity(
        runtime_identity=dict(runtime_identity),
        template_lock_digest=selection.template.lock_digest,
        custom_lock_digest=custom_digest,
        isolation_policy={"version": selection.template.isolation_policy_version},
    )
    return _ResolvedMember(
        kind=kind,
        stable_key=stable_key,
        request=request,
        selection=selection,
        dependencies=dependencies,
        environment_key=environment_key,
        custom_digest=custom_digest,
        sandbox_policy=dict(sandbox_policy),
        assigned_tool_keys=tuple(assigned_tool_keys),
        intrinsic_names=tuple(intrinsic_names),
        include_intrinsic_guides=include_intrinsic_guides,
    )


def _validate_definition_namespace(definition: ToolboxDefinitionSpec) -> None:
    advertised: list[str] = [item.advertised_name for item in definition.auto_requests]
    advertised.extend(item.advertised_name for item in definition.manual_requests)
    for name in definition.intrinsics.names:
        metadata = intrinsic_metadata(name)
        advertised.append(metadata.name)
        if definition.intrinsics.include_guides and metadata.guide_name:
            advertised.append(metadata.guide_name)
    duplicates = sorted(name for name in set(advertised) if advertised.count(name) > 1)
    if duplicates:
        raise ValueError(f"toolbox_definition_duplicate_advertised_name:{duplicates[0]}")

    files: dict[str, str] = {}
    for request in (*definition.auto_requests, *definition.manual_requests):
        for file in request.files:
            key = file.normalized_path().casefold()
            existing = files.get(key)
            if existing is not None and existing != file.content:
                raise ValueError(f"toolbox_definition_file_conflict:{file.normalized_path()}")
            files[key] = file.content


def plan_toolbox_definition(
    definition: ToolboxDefinitionSpec | Mapping[str, Any],
    *,
    templates: Sequence[ToolboxEnvironmentTemplateSpec],
    python_abi: str,
    platform: str,
    runtime_identity: Mapping[str, Any],
) -> ToolboxDefinitionPlanDraft:
    """Resolve every request first, then group by environment and policy identity."""

    model = definition if isinstance(definition, ToolboxDefinitionSpec) else ToolboxDefinitionSpec.from_dict(definition)
    _validate_definition_namespace(model)
    members: list[_ResolvedMember] = []
    for request in model.auto_requests:
        members.append(
            _resolve_member(
                kind="auto",
                stable_key=request.stable_key,
                request=request,
                files=request.files,
                dependency_mode=request.dependency.mode,
                requested_template_id=request.dependency.template_id,
                declared_imports=request.dependency.declared_imports,
                package_requirements=request.dependency.package_requirements,
                templates=templates,
                python_abi=python_abi,
                platform=platform,
                runtime_identity=runtime_identity,
                sandbox_policy=request.sandbox_policy,
                assigned_tool_keys=(request.stable_key,),
            )
        )
    for request in model.manual_requests:
        members.append(
            _resolve_member(
                kind="manual",
                stable_key=request.stable_key,
                request=request,
                files=request.files,
                dependency_mode=request.dependency.mode,
                requested_template_id=request.dependency.template_id,
                declared_imports=request.dependency.declared_imports,
                package_requirements=request.dependency.package_requirements,
                templates=templates,
                python_abi=python_abi,
                platform=platform,
                runtime_identity=runtime_identity,
                sandbox_policy=request.sandbox_policy,
                assigned_tool_keys=(request.stable_key,),
            )
        )
    if model.intrinsics.names:
        metadata = intrinsic_dependency_metadata(model.intrinsics.names)
        assigned = [f"intrinsic:{name}" for name in metadata["intrinsics"]]
        if model.intrinsics.include_guides:
            assigned.extend(
                f"intrinsic:{intrinsic_metadata(name).guide_name}"
                for name in metadata["intrinsics"]
                if intrinsic_metadata(name).guide_name
            )
        members.append(
            _resolve_member(
                kind="intrinsic",
                stable_key="intrinsics",
                request=None,
                files=(),
                dependency_mode="auto",
                requested_template_id=None,
                declared_imports=metadata["import_roots"],
                package_requirements=metadata["package_requirements"],
                templates=templates,
                python_abi=python_abi,
                platform=platform,
                runtime_identity=runtime_identity,
                sandbox_policy=model.intrinsics.sandbox_policy,
                assigned_tool_keys=assigned,
                intrinsic_names=metadata["intrinsics"],
                include_intrinsic_guides=model.intrinsics.include_guides,
            )
        )

    grouped: dict[str, list[_ResolvedMember]] = {}
    for member in members:
        provisional = ResolvedToolboxProfileSpec(
            environment_key=member.environment_key,
            template_id=member.selection.template.template_id,
            template_lock_digest=member.selection.template.lock_digest,
            custom_resolved_lock_digest=member.custom_digest,
            sandbox_policy=member.sandbox_policy,
            assigned_tool_keys=member.assigned_tool_keys,
            resolved_import_roots=member.resolved_import_roots,
        )
        grouped.setdefault(provisional.profile_id, []).append(member)

    profiles: list[ResolvedToolboxProfileSpec] = []
    bundles: list[ToolboxBundleSpec] = []
    for profile_id, group in sorted(grouped.items()):
        first = group[0]
        profile = ResolvedToolboxProfileSpec(
            environment_key=first.environment_key,
            template_id=first.selection.template.template_id,
            template_lock_digest=first.selection.template.lock_digest,
            custom_resolved_lock_digest=first.custom_digest,
            sandbox_policy=first.sandbox_policy,
            assigned_tool_keys=tuple(key for member in group for key in member.assigned_tool_keys),
            resolved_import_roots=tuple(
                sorted({root for member in group for root in member.resolved_import_roots})
            ),
            profile_id=profile_id,
        )
        files_by_path: dict[str, ToolboxBundleFile] = {}
        auto_tools: list[ToolboxBundleAutoTool] = []
        manual_tools: list[ToolboxBundleTool] = []
        intrinsic_names: list[str] = []
        include_guides = False
        for member in group:
            if member.request is not None:
                for file in member.request.files:
                    files_by_path.setdefault(file.normalized_path(), file)
            if member.kind == "auto":
                request = member.request
                assert isinstance(request, ToolboxAutoAssignmentRequestV2)
                auto_tools.append(
                    ToolboxBundleAutoTool(
                        module_name=request.module_name,
                        callable_name=request.callable_name,
                        activate=request.activate,
                        hidden=request.hidden,
                        non_restartable=request.non_restartable,
                        guide_content=dict(request.guide_content) if request.guide_content is not None else None,
                        guide_description=request.guide_description,
                        callback_signature=dict(request.callback_signature) if request.callback_signature is not None else None,
                        concurrency=dict(request.concurrency) if request.concurrency is not None else None,
                    )
                )
            elif member.kind == "manual":
                request = member.request
                assert isinstance(request, ToolboxManualAssignmentRequestV2)
                manual_tools.append(
                    ToolboxBundleTool(
                        definition=dict(request.tool_definition),
                        entrypoint=f"{request.module_name}:{request.callable_name}",
                        hidden=request.hidden,
                        non_restartable=request.non_restartable,
                        callback_signature=dict(request.callback_signature) if request.callback_signature is not None else None,
                        concurrency=dict(request.concurrency) if request.concurrency is not None else None,
                    )
                )
            else:
                intrinsic_names.extend(member.intrinsic_names)
                include_guides = include_guides or member.include_intrinsic_guides
        safe_profile_id = profile.profile_id.removeprefix("sha256:")
        bundles.append(
            ToolboxBundleSpec(
                bundle_id=f"{model.toolbox_id}-{safe_profile_id}",
                toolbox_id=model.toolbox_id,
                sandbox_profile=SandboxProfileSpec(
                    profile_id=safe_profile_id,
                    required_imports=list(profile.resolved_import_roots),
                    sandbox_policy=dict(profile.sandbox_policy),
                ),
                files=list(files_by_path.values()),
                tools=manual_tools,
                auto_tools=auto_tools,
                with_intrinsics=bool(intrinsic_names),
                with_intrinsic_guides=include_guides,
                intrinsic_tool_names=sorted(set(intrinsic_names)),
                active_intrinsic_tool_names=sorted(set(intrinsic_names)),
                dependency_lock_hash=profile.effective_lock_digest,
                resolved_profile=profile,
            )
        )
        profiles.append(profile)
    return ToolboxDefinitionPlanDraft(
        definition=model,
        profiles=tuple(profiles),
        bundles=tuple(bundles),
        custom_environment_count=sum(1 for item in profiles if item.custom_resolved_lock_digest is not None),
    )


__all__ = ["ToolboxDefinitionPlanDraft", "plan_toolbox_definition"]
