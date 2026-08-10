"""Pure version-2 toolbox definition planning and post-resolution grouping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mp13_engine.mp13_intrinsics_metadata import intrinsic_dependency_metadata, intrinsic_metadata

from .bundle_models import (
    ResolvedToolboxProfileSpec,
    SandboxProfileSpec,
    ToolboxEnvironmentMutationSpec,
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
from .identity import custom_lock_digest, environment_identity, identity_digest, require_digest


@dataclass(frozen=True)
class ActiveToolboxProfileSnapshot:
    profile_id: str
    manifest_hash: str
    environment_key: str
    sandbox_policy_digest: str
    assigned_tool_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not str(self.profile_id or "").strip():
            raise ValueError("active_profile_id_required")
        manifest = str(self.manifest_hash or "").strip()
        if not (
            manifest.startswith("sha256:") and len(manifest) == 71
            or len(manifest) == 64 and all(character in "0123456789abcdef" for character in manifest)
        ):
            raise ValueError("active_profile_manifest_hash_invalid")
        require_digest(self.environment_key, label="active_profile_environment_key")
        require_digest(self.sandbox_policy_digest, label="active_profile_policy_digest")
        assigned = tuple(sorted(str(item or "").strip() for item in self.assigned_tool_keys))
        if not assigned or any(not item for item in assigned) or len(set(assigned)) != len(assigned):
            raise ValueError("active_profile_assigned_tool_keys_invalid")
        object.__setattr__(self, "profile_id", str(self.profile_id).strip())
        object.__setattr__(self, "manifest_hash", manifest)
        object.__setattr__(self, "assigned_tool_keys", assigned)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "manifest_hash": self.manifest_hash,
            "environment_key": self.environment_key,
            "sandbox_policy_digest": self.sandbox_policy_digest,
            "assigned_tool_keys": list(self.assigned_tool_keys),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveToolboxProfileSnapshot":
        row = dict(payload or {})
        fields = {"profile_id", "manifest_hash", "environment_key", "sandbox_policy_digest", "assigned_tool_keys"}
        if set(row) != fields:
            raise ValueError("active_profile_snapshot_fields_invalid")
        return cls(**{**row, "assigned_tool_keys": tuple(row["assigned_tool_keys"])})


def profile_snapshots_from_draft(
    draft: "ToolboxDefinitionPlanDraft",
) -> tuple[ActiveToolboxProfileSnapshot, ...]:
    snapshots: list[ActiveToolboxProfileSnapshot] = []
    for profile, bundle in zip(draft.profiles, draft.bundles, strict=True):
        manifest = bundle.manifest_payload()
        snapshots.append(
            ActiveToolboxProfileSnapshot(
                profile_id=profile.profile_id,
                manifest_hash=manifest["manifest_hash"],
                environment_key=profile.environment_key,
                sandbox_policy_digest=identity_digest(
                    "hosting.toolbox.sandbox_policy.v1", profile.sandbox_policy
                ),
                assigned_tool_keys=profile.assigned_tool_keys,
            )
        )
    return tuple(sorted(snapshots, key=lambda item: item.profile_id))


def classify_toolbox_profiles(
    draft: "ToolboxDefinitionPlanDraft",
    active_profiles: Sequence[ActiveToolboxProfileSnapshot | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Classify without staging by exact runtime identity and stable ownership."""

    proposed = list(profile_snapshots_from_draft(draft))
    active = sorted([
        item if isinstance(item, ActiveToolboxProfileSnapshot) else ActiveToolboxProfileSnapshot.from_dict(item)
        for item in active_profiles
    ], key=lambda item: item.profile_id)
    if len({item.profile_id for item in active}) != len(active):
        raise ValueError("active_profile_snapshot_duplicate")
    remaining_active = {item.profile_id: item for item in active}
    remaining_proposed = {item.profile_id: item for item in proposed}
    out: list[dict[str, Any]] = []

    for proposed_item in proposed:
        exact = next(
            (
                active_item for active_item in remaining_active.values()
                if active_item.manifest_hash == proposed_item.manifest_hash
                and active_item.environment_key == proposed_item.environment_key
                and active_item.sandbox_policy_digest == proposed_item.sandbox_policy_digest
            ),
            None,
        )
        if exact is None:
            continue
        out.append(
            {
                "classification": "reused",
                "active_profile_id": exact.profile_id,
                "proposed_profile_id": proposed_item.profile_id,
                "changed_fields": [],
            }
        )
        remaining_active.pop(exact.profile_id)
        remaining_proposed.pop(proposed_item.profile_id)

    for proposed_item in list(remaining_proposed.values()):
        proposed_keys = set(proposed_item.assigned_tool_keys)
        candidates = [
            (len(proposed_keys & set(active_item.assigned_tool_keys)), active_item.profile_id, active_item)
            for active_item in remaining_active.values()
        ]
        overlap, _profile_id, matched = max(candidates, default=(0, "", None), key=lambda item: (item[0], item[1]))
        if overlap <= 0 or matched is None:
            continue
        changed = [
            field for field in ("manifest_hash", "environment_key", "sandbox_policy_digest")
            if getattr(matched, field) != getattr(proposed_item, field)
        ]
        out.append(
            {
                "classification": "replaced",
                "active_profile_id": matched.profile_id,
                "proposed_profile_id": proposed_item.profile_id,
                "changed_fields": changed,
            }
        )
        remaining_active.pop(matched.profile_id)
        remaining_proposed.pop(proposed_item.profile_id)

    out.extend(
        {
            "classification": "added",
            "active_profile_id": None,
            "proposed_profile_id": item.profile_id,
            "changed_fields": [],
        }
        for item in remaining_proposed.values()
    )
    out.extend(
        {
            "classification": "removed",
            "active_profile_id": item.profile_id,
            "proposed_profile_id": None,
            "changed_fields": [],
        }
        for item in remaining_active.values()
    )
    order = {"reused": 0, "replaced": 1, "added": 2, "removed": 3}
    return tuple(
        sorted(
            out,
            key=lambda item: (
                order[item["classification"]],
                str(item["proposed_profile_id"] or ""),
                str(item["active_profile_id"] or ""),
            ),
        )
    )


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
class ToolboxEnvironmentConfirmationChoice:
    environment_id: str
    alternative_id: str
    accept_package_changes: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment_id",
            require_digest(self.environment_id, label="toolbox_confirmation_environment_id"),
        )
        object.__setattr__(
            self,
            "alternative_id",
            require_digest(self.alternative_id, label="toolbox_confirmation_alternative_id"),
        )
        if not isinstance(self.accept_package_changes, bool):
            raise ValueError("toolbox_confirmation_accept_boolean_required")

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ToolboxEnvironmentConfirmationChoice":
        row = dict(payload or {})
        if set(row) != {"environment_id", "alternative_id", "accept_package_changes"}:
            raise ValueError("toolbox_confirmation_choice_fields_invalid")
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "alternative_id": self.alternative_id,
            "accept_package_changes": self.accept_package_changes,
        }


@dataclass(frozen=True)
class ToolboxConfirmationReduction:
    effective_definition: ToolboxDefinitionSpec
    selected_alternatives: tuple[Mapping[str, str], ...]
    accepted_tool_keys: tuple[str, ...]
    skipped_tools: tuple[Mapping[str, Any], ...]
    preserved_active_tool_keys: tuple[str, ...]
    removed_tool_keys: tuple[str, ...]
    package_mutations: tuple[Mapping[str, Any], ...]
    dependency_approval_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_definition": self.effective_definition.to_dict(),
            "effective_definition_revision": self.effective_definition.revision,
            "selected_alternatives": [dict(item) for item in self.selected_alternatives],
            "accepted_tool_keys": list(self.accepted_tool_keys),
            "skipped_tools": [dict(item) for item in self.skipped_tools],
            "preserved_active_tool_keys": list(self.preserved_active_tool_keys),
            "removed_tool_keys": list(self.removed_tool_keys),
            "package_mutations": [dict(item) for item in self.package_mutations],
            "dependency_approval_required": self.dependency_approval_required,
        }


def _definition_items(
    definition: ToolboxDefinitionSpec,
) -> tuple[dict[str, tuple[str, Any]], dict[str, str]]:
    items: dict[str, tuple[str, Any]] = {}
    for request in definition.auto_requests:
        items[request.stable_key] = ("auto", request)
    for request in definition.manual_requests:
        items[request.stable_key] = ("manual", request)
    for name in definition.intrinsics.names:
        items[f"intrinsic:{name}"] = ("intrinsic", name)
    advertised: dict[str, str] = {}
    for key, (kind, value) in items.items():
        if kind == "auto":
            advertised[key] = value.advertised_name
        elif kind == "manual":
            advertised[key] = value.advertised_name
        else:
            advertised[key] = intrinsic_metadata(value).name
    return items, advertised


def reduce_toolbox_confirmation(
    *,
    active_definition: ToolboxDefinitionSpec | Mapping[str, Any],
    proposed_definition: ToolboxDefinitionSpec | Mapping[str, Any],
    environment_mutations: Sequence[ToolboxEnvironmentMutationSpec | Mapping[str, Any]],
    choices: Sequence[ToolboxEnvironmentConfirmationChoice | Mapping[str, Any]],
) -> ToolboxConfirmationReduction:
    """Reduce only offered choices into one pinned effective definition."""

    active = (
        active_definition
        if isinstance(active_definition, ToolboxDefinitionSpec)
        else ToolboxDefinitionSpec.from_dict(active_definition)
    )
    proposed = (
        proposed_definition
        if isinstance(proposed_definition, ToolboxDefinitionSpec)
        else ToolboxDefinitionSpec.from_dict(proposed_definition)
    )
    active_is_empty = (
        not active.auto_requests and not active.manual_requests and not active.intrinsics.names
    )
    expected_active_revision = None if active_is_empty else active.revision
    if (
        active.toolbox_id != proposed.toolbox_id
        or proposed.expected_revision != expected_active_revision
    ):
        raise ValueError("toolbox_confirmation_active_revision_mismatch")
    offers = tuple(
        item if isinstance(item, ToolboxEnvironmentMutationSpec)
        else ToolboxEnvironmentMutationSpec.from_dict(item)
        for item in environment_mutations
    )
    decisions = tuple(
        item if isinstance(item, ToolboxEnvironmentConfirmationChoice)
        else ToolboxEnvironmentConfirmationChoice.from_dict(item)
        for item in choices
    )
    if (
        not offers
        or len({item.environment_id for item in offers}) != len(offers)
        or len({item.environment_id for item in decisions}) != len(decisions)
        or {item.environment_id for item in offers} != {item.environment_id for item in decisions}
    ):
        raise ValueError("toolbox_confirmation_choices_incomplete")
    decision_by_environment = {item.environment_id: item for item in decisions}
    active_items, _active_advertised = _definition_items(active)
    proposed_items, _proposed_advertised = _definition_items(proposed)
    expected_changes: dict[str, str] = {}
    all_edges: dict[str, Any] = {}
    tool_environment: dict[str, str] = {}
    selected: list[dict[str, str]] = []
    declined_distributions: dict[str, tuple[str, ...]] = {}
    package_mutations: list[dict[str, Any]] = []
    approval_required = False
    for offer in offers:
        decision = decision_by_environment[offer.environment_id]
        alternative = next(
            (item for item in offer.alternatives if item.alternative_id == decision.alternative_id),
            None,
        )
        if alternative is None:
            raise ValueError("toolbox_confirmation_alternative_not_offered")
        selected.append(
            {
                "environment_id": offer.environment_id,
                "alternative_id": alternative.alternative_id,
            }
        )
        package_change_distributions = tuple(
            sorted(
                item.distribution
                for item in alternative.package_mutations
                if item.mutation in {"addition", "transition"}
            )
        )
        declined = not decision.accept_package_changes and bool(package_change_distributions)
        for mutation in alternative.package_mutations:
            if not declined or mutation.mutation == "removal":
                package_mutations.append(mutation.to_dict())
        if decision.accept_package_changes:
            approval_required = approval_required or offer.dependency_approval_required
        for mutation in offer.tool_mutations:
            if mutation.tool_key in expected_changes:
                raise ValueError("toolbox_confirmation_tool_offered_twice")
            expected_changes[mutation.tool_key] = mutation.change
            tool_environment[mutation.tool_key] = offer.environment_id
            if declined:
                declined_distributions[mutation.tool_key] = package_change_distributions
        for edge in offer.dependency_edges:
            all_edges[edge.tool_key] = edge

    actual_keys = set(active_items) | set(proposed_items)
    if set(expected_changes) != actual_keys or set(all_edges) != actual_keys:
        raise ValueError("toolbox_confirmation_offer_tools_incomplete")
    for key in sorted(actual_keys):
        before = active_items.get(key)
        after = proposed_items.get(key)
        actual = (
            "added" if before is None else
            "removed" if after is None else
            "unchanged" if before[0] == after[0] and (
                before[1] == after[1]
                if before[0] == "intrinsic"
                else before[1].to_dict() == after[1].to_dict()
            ) else "updated"
        )
        if expected_changes[key] != actual:
            raise ValueError("toolbox_confirmation_tool_change_mismatch")

    skipped: dict[str, dict[str, Any]] = {}
    preserved: set[str] = set()
    removed = {key for key, change in expected_changes.items() if change == "removed"}
    for key, distributions in declined_distributions.items():
        change = expected_changes[key]
        if change in {"added", "updated"}:
            skipped[key] = {
                "tool_key": key,
                "reason": "package_changes_declined",
                "affected_distributions": list(distributions),
                "environment_id": tool_environment[key],
            }
            if change == "updated":
                preserved.add(key)

    changed = True
    while changed:
        changed = False
        for key in sorted(actual_keys - removed - set(skipped)):
            missing = sorted(set(all_edges[key].required_tool_keys) & set(skipped))
            if not missing:
                continue
            skipped[key] = {
                "tool_key": key,
                "reason": "shared_environment_incomplete",
                "affected_distributions": [],
                "environment_id": tool_environment[key],
                "missing_tool_keys": missing,
            }
            if expected_changes[key] == "updated":
                preserved.add(key)
            changed = True

    effective_keys = (set(proposed_items) - set(skipped)) | preserved
    autos = []
    manuals = []
    intrinsic_names = []
    for key in sorted(effective_keys):
        source = active_items[key] if key in preserved else proposed_items[key]
        if source[0] == "auto":
            autos.append(source[1])
        elif source[0] == "manual":
            manuals.append(source[1])
        else:
            intrinsic_names.append(source[1])
    effective = ToolboxDefinitionSpec(
        toolbox_id=proposed.toolbox_id,
        expected_revision=proposed.expected_revision,
        auto_requests=tuple(autos),
        manual_requests=tuple(manuals),
        intrinsics=type(proposed.intrinsics)(
            names=tuple(intrinsic_names),
            include_guides=proposed.intrinsics.include_guides,
            sandbox_policy=proposed.intrinsics.sandbox_policy,
        ),
    )
    try:
        _validate_definition_namespace(effective)
    except ValueError as exc:
        raise ValueError("toolbox_confirmation_namespace_conflict") from exc
    accepted = tuple(sorted(set(proposed_items) - set(skipped)))
    unique_mutations = {
        (item["distribution"], item["mutation"], item["from_version"], item["to_version"]): item
        for item in package_mutations
    }
    return ToolboxConfirmationReduction(
        effective_definition=effective,
        selected_alternatives=tuple(sorted(selected, key=lambda item: item["environment_id"])),
        accepted_tool_keys=accepted,
        skipped_tools=tuple(skipped[key] for key in sorted(skipped)),
        preserved_active_tool_keys=tuple(sorted(preserved)),
        removed_tool_keys=tuple(sorted(removed)),
        package_mutations=tuple(
            unique_mutations[key] for key in sorted(unique_mutations)
        ),
        dependency_approval_required=approval_required,
    )


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


__all__ = [
    "ActiveToolboxProfileSnapshot",
    "ToolboxDefinitionPlanDraft",
    "classify_toolbox_profiles",
    "plan_toolbox_definition",
    "profile_snapshots_from_draft",
]
