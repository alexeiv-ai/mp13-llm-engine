from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from ..toolbox.builtin_resolver import AirgapBuiltinWheelResolver
from ..toolbox.bundle_models import ToolboxExactArtifactSpec
from ..toolbox.catalog import PHASE0_REVIEWED_IMPORT_CATALOG, ToolboxEnvironmentTemplateSpec, normalize_distribution_name
from ..toolbox.definition_planner import (
    ActiveToolboxEnvironmentResolution,
    ToolboxDefinitionPlanDraft,
    VerifiedToolboxResolutionCandidate,
)
from ..toolbox.dependency_analysis import analyze_toolbox_bundle_imports, resolve_toolbox_dependencies
from ..toolbox.host_project_config import ToolboxHostProjectConfiguration
from .toolbox_artifact_store import AtomicToolboxArtifactStore


class ConfiguredToolboxPlanResolver:
    """Resolve definition offers only from configured, verified CAS artifacts."""

    def __init__(
        self,
        *,
        configuration: ToolboxHostProjectConfiguration,
        artifact_store: AtomicToolboxArtifactStore,
        catalog_state: Mapping[str, Any],
    ) -> None:
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        self.configuration = configuration
        self.artifact_store = artifact_store
        self.catalog_state = dict(catalog_state or {})
        self._source_paths: dict[str, dict[str, Path]] = {}

    def _source(self, source_id: str):
        logical = str(source_id or "").strip()
        try:
            return next(item for item in self.configuration.sources if item.source_id == logical)
        except StopIteration as exc:
            raise ValueError("toolbox_plan_artifact_source_unconfigured") from exc

    def _paths(self, source_id: str) -> dict[str, Path]:
        logical = str(source_id or "").strip()
        if logical not in self._source_paths:
            self._source_paths[logical] = self.artifact_store.source_artifacts(logical)
        return self._source_paths[logical]

    def _entry(self, template_id: str) -> dict[str, Any]:
        active = dict(self.catalog_state.get("active") or {})
        digest = str(active.get(template_id) or "")
        entry = next(
            (
                dict(item)
                for item in list(self.catalog_state.get("entries") or [])
                if item.get("template_id") == template_id
                and item.get("template_digest") == digest
                and item.get("lifecycle") == "active"
            ),
            None,
        )
        if entry is None:
            raise ValueError("toolbox_plan_template_revision_unavailable")
        return entry

    @staticmethod
    def _artifact_roots(distribution: str, template: ToolboxEnvironmentTemplateSpec) -> tuple[str, ...]:
        rule = PHASE0_REVIEWED_IMPORT_CATALOG.for_distribution(distribution)
        roots = tuple(rule.import_roots) if rule is not None else ()
        if distribution == "mp13-engine":
            roots = tuple(sorted(set(roots) | {"hosting", "mp13_engine"}))
        return tuple(sorted(set(roots) & set(template.exposed_import_roots)))

    def _template_artifacts(
        self, template: ToolboxEnvironmentTemplateSpec, entry: Mapping[str, Any]
    ) -> tuple[ToolboxExactArtifactSpec, ...]:
        locked = {item.name: item.version for item in template.locked_distributions}
        artifacts: list[ToolboxExactArtifactSpec] = []
        for reference in list(entry.get("artifacts") or []):
            row = dict(reference or {})
            source_id = str(row.get("source_id") or "")
            filename = str(row.get("filename") or "")
            digest = str(row.get("sha256") or "")
            try:
                wheel_name, wheel_version, _build, wheel_tags = parse_wheel_filename(filename)
            except InvalidWheelFilename as exc:
                raise ValueError("toolbox_plan_template_artifact_invalid") from exc
            distribution = normalize_distribution_name(str(wheel_name))
            version = str(wheel_version)
            if locked.get(distribution) != version:
                raise ValueError("toolbox_plan_template_artifact_lock_mismatch")
            path = self._paths(source_id).get(filename)
            if path is None or not path.is_file() or self.artifact_store.object_path(digest) != path:
                raise ValueError("toolbox_plan_template_artifact_unavailable")
            artifacts.append(
                ToolboxExactArtifactSpec(
                    import_roots=self._artifact_roots(distribution, template),
                    distribution=distribution,
                    dependency_reason="template_runtime",
                    version=version,
                    wheel_filename=filename,
                    artifact_digest=digest,
                    compatibility_tags=tuple(sorted(str(item) for item in wheel_tags)),
                    provenance=(
                        f"{template.provenance.source}:{template.provenance.evidence_digest}"
                    ),
                    source_id=source_id,
                )
            )
        if {item.distribution for item in artifacts} != set(locked):
            raise ValueError("toolbox_plan_template_artifact_closure_incomplete")
        return tuple(sorted(artifacts, key=lambda item: item.distribution))

    @staticmethod
    def _requirement_text(distribution: str, extras: Sequence[str], constraint: str) -> str:
        suffix = f"[{','.join(sorted(extras))}]" if extras else ""
        return f"{distribution}{suffix}{constraint}"

    def _profile_requirements(
        self, draft: ToolboxDefinitionPlanDraft, tool_keys: set[str]
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        requirements: list[str] = []
        roots: dict[str, set[str]] = {}
        requests = {
            item.stable_key: item
            for item in (*draft.definition.auto_requests, *draft.definition.manual_requests)
        }
        for key in sorted(tool_keys):
            request = requests.get(key)
            if request is None:
                continue
            dependencies = resolve_toolbox_dependencies(
                analyze_toolbox_bundle_imports(
                    request.files, declared_imports=request.dependency.declared_imports
                ),
                package_requirements=request.dependency.package_requirements,
            )
            for item in dependencies.requirements:
                requirements.append(
                    self._requirement_text(item.distribution, item.extras, item.constraint)
                )
                roots.setdefault(item.distribution, set()).update(item.import_roots)
        return tuple(sorted(set(requirements))), {
            key: tuple(sorted(value)) for key, value in sorted(roots.items())
        }

    def _closure_artifacts(
        self,
        closure,
        *,
        template: ToolboxEnvironmentTemplateSpec,
        direct_roots: Mapping[str, tuple[str, ...]],
        base_distributions: set[str],
    ) -> tuple[ToolboxExactArtifactSpec, ...]:
        artifacts: list[ToolboxExactArtifactSpec] = []
        for item in closure.locked_artifacts:
            try:
                _name, _version, _build, wheel_tags = parse_wheel_filename(item.filename)
            except InvalidWheelFilename as exc:
                raise ValueError("toolbox_plan_resolved_artifact_invalid") from exc
            if item.distribution_name in direct_roots:
                reason = "direct"
                roots = direct_roots[item.distribution_name]
            elif item.distribution_name in base_distributions:
                reason = "template_runtime"
                roots = self._artifact_roots(item.distribution_name, template)
            else:
                reason = "transitive"
                rule = PHASE0_REVIEWED_IMPORT_CATALOG.for_distribution(
                    item.distribution_name
                )
                roots = tuple(rule.import_roots) if rule is not None else ()
            artifacts.append(
                ToolboxExactArtifactSpec(
                    import_roots=roots,
                    distribution=item.distribution_name,
                    dependency_reason=reason,
                    version=item.version,
                    wheel_filename=item.filename,
                    artifact_digest=item.sha256,
                    compatibility_tags=tuple(sorted(str(value) for value in wheel_tags)),
                    provenance=f"verified-cas:{item.source_id}",
                    source_id=item.source_id,
                )
            )
        return tuple(sorted(artifacts, key=lambda item: item.distribution))

    def candidates_for_draft(
        self, draft: ToolboxDefinitionPlanDraft
    ) -> tuple[VerifiedToolboxResolutionCandidate, ...]:
        candidates: list[VerifiedToolboxResolutionCandidate] = []
        entries = {profile.template_id: self._entry(profile.template_id) for profile in draft.profiles}
        templates = {
            template_id: ToolboxEnvironmentTemplateSpec.from_dict(entry["template"])
            for template_id, entry in entries.items()
        }
        for profile in draft.profiles:
            entry = entries[profile.template_id]
            template = templates[profile.template_id]
            base_artifacts = self._template_artifacts(template, entry)
            base_sources = {item.source_id for item in base_artifacts}
            if profile.custom_resolved_lock_digest is None:
                source_ids = tuple(sorted(base_sources))
                candidates.append(
                    VerifiedToolboxResolutionCandidate(
                        environment_id=profile.profile_id,
                        base_template_id=profile.template_id,
                        base_template_revision=entry["template_digest"],
                        source_ids=source_ids,
                        source_origins=tuple(
                            sorted(self._source(item).origin for item in source_ids)
                        ),
                        source_priority=min(self._source(item).priority for item in source_ids),
                        lock_digest=profile.template_lock_digest,
                        artifacts=base_artifacts,
                    )
                )
                continue
            custom_requirements, direct_roots = self._profile_requirements(
                draft, set(profile.assigned_tool_keys)
            )
            pins = tuple(
                f"{item.name}=={item.version}" for item in template.locked_distributions
            )
            requirements = tuple(sorted({*pins, *custom_requirements}))
            configured_ids = tuple(item.source_id for item in self.configuration.sources)
            source_sets = {
                tuple(sorted(base_sources | {source_id})) for source_id in configured_ids
            }
            source_sets.add(tuple(sorted(base_sources | set(configured_ids))))
            seen: set[tuple[str, tuple[str, ...]]] = set()
            for source_ids in sorted(source_sets, key=lambda item: (len(item), item)):
                verified = {source_id: self._paths(source_id) for source_id in source_ids}
                resolver = AirgapBuiltinWheelResolver(
                    self.configuration,
                    artifact_sources={},
                    verified_artifacts=verified,
                )
                try:
                    closure = resolver.resolve_requirements(
                        template_id=profile.template_id,
                        package_requirements=requirements,
                    )
                except RuntimeError:
                    continue
                artifacts = self._closure_artifacts(
                    closure,
                    template=template,
                    direct_roots=direct_roots,
                    base_distributions={item.name for item in template.locked_distributions},
                )
                actual_sources = tuple(sorted({item.source_id for item in artifacts}))
                identity = (
                    closure.lock_digest,
                    tuple(f"{item.source_id}:{item.artifact_digest}" for item in artifacts),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append(
                    VerifiedToolboxResolutionCandidate(
                        environment_id=profile.profile_id,
                        base_template_id=profile.template_id,
                        base_template_revision=entry["template_digest"],
                        source_ids=actual_sources,
                        source_origins=tuple(
                            sorted(self._source(item).origin for item in actual_sources)
                        ),
                        source_priority=min(
                            self._source(item).priority for item in actual_sources
                        ),
                        lock_digest=closure.lock_digest,
                        artifacts=artifacts,
                    )
                )
            if not any(item.environment_id == profile.profile_id for item in candidates):
                raise ValueError("toolbox_definition_exact_wheel_missing")
        return tuple(candidates)

    def active_environments(
        self, active_snapshot: Mapping[str, Any] | None
    ) -> tuple[ActiveToolboxEnvironmentResolution, ...]:
        snapshot = dict(active_snapshot or {})
        environments: list[ActiveToolboxEnvironmentResolution] = []
        for profile_id, raw in sorted(dict(snapshot.get("profiles") or {}).items()):
            profile = dict(dict(raw or {}).get("profile") or {})
            if profile.get("custom_resolved_lock_digest") is not None:
                from ..toolbox.hermetic_environment import ResolvedToolboxEnvironmentInput

                resolved = ResolvedToolboxEnvironmentInput.from_dict(
                    dict(dict(raw or {}).get("resolved_environment") or {})
                )
                template_entry = self._entry(str(profile.get("template_id") or ""))
                template = ToolboxEnvironmentTemplateSpec.from_dict(template_entry["template"])
                template_names = {item.name for item in template.locked_distributions}
                artifacts = []
                for item in resolved.locked_artifacts:
                    _name, _version, _build, tags = parse_wheel_filename(item.filename)
                    distribution = normalize_distribution_name(item.distribution_name)
                    rule = PHASE0_REVIEWED_IMPORT_CATALOG.for_distribution(distribution)
                    roots = tuple(
                        sorted(set(rule.import_roots) & set(resolved.resolved_import_roots))
                    ) if rule is not None else ()
                    artifacts.append(ToolboxExactArtifactSpec(
                        import_roots=roots,
                        distribution=distribution,
                        dependency_reason=(
                            "template_runtime" if distribution in template_names else "transitive"
                        ),
                        version=item.version,
                        wheel_filename=item.filename,
                        artifact_digest=item.sha256,
                        compatibility_tags=tuple(sorted(str(value) for value in tags)),
                        provenance="verified-cas:active-confirmation",
                        source_id=item.source_id,
                    ))
                source_ids = tuple(sorted({item.source_id for item in resolved.locked_artifacts}))
                environments.append(ActiveToolboxEnvironmentResolution(
                    environment_id=profile_id,
                    tool_keys=tuple(profile["assigned_tool_keys"]),
                    base_template_id=template.template_id,
                    base_template_revision=resolved.template_digest,
                    source_ids=source_ids,
                    source_origins=tuple(sorted(self._source(item).origin for item in source_ids)),
                    lock_digest=resolved.custom_resolved_lock_digest or resolved.complete_lock_digest,
                    artifacts=tuple(sorted(artifacts, key=lambda item: item.distribution)),
                ))
                continue
            template_id = str(profile.get("template_id") or "")
            entry = self._entry(template_id)
            template = ToolboxEnvironmentTemplateSpec.from_dict(entry["template"])
            artifacts = self._template_artifacts(template, entry)
            source_ids = tuple(sorted({item.source_id for item in artifacts}))
            environments.append(
                ActiveToolboxEnvironmentResolution(
                    environment_id=profile_id,
                    tool_keys=tuple(profile["assigned_tool_keys"]),
                    base_template_id=template_id,
                    base_template_revision=entry["template_digest"],
                    source_ids=source_ids,
                    source_origins=tuple(
                        sorted(self._source(item).origin for item in source_ids)
                    ),
                    lock_digest=str(profile.get("template_lock_digest") or ""),
                    artifacts=artifacts,
                )
            )
        return tuple(environments)


__all__ = ["ConfiguredToolboxPlanResolver"]
