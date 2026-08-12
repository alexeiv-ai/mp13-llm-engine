"""Sandboxed toolbox assignment orchestration."""
from __future__ import annotations

import sys
import uuid
from typing import Any, Dict, List, Optional, Sequence

from .bundle_models import (
    SandboxProfileSpec,
    ResolvedToolboxProfileSpec,
    ResolvedToolboxSandboxAssignment,
    ToolboxAutoAssignmentRequest,
    ToolboxBundleFile,
    ToolboxBundleSpec,
    ToolboxEnvironmentSpec,
    ToolboxManualAssignmentRequest,
    ToolboxSandboxAssignment,
)
from .environment import EnvironmentRuntimeAdapter
from .staging import ToolboxBundleStager
from .target import detect_current_toolbox_target
from ..sandbox.toolbox_runtime import HostedToolboxRuntimeBase
from ..service.toolbox_runtime_identity import runtime_binding_digest


class ToolboxSandboxOrchestrator:
    def __init__(
        self,
        *,
        service: Any,
        stager: ToolboxBundleStager,
        python_executable: Optional[str] = None,
    ) -> None:
        self.service = service
        self.stager = stager
        self.python_executable = str(python_executable or sys.executable)
        self.environment_manager = EnvironmentRuntimeAdapter(self.stager.hosting_root)
        self.runtime_base = HostedToolboxRuntimeBase()

    @staticmethod
    def _bundle_id(toolbox_id: str, profile: SandboxProfileSpec) -> str:
        return f"{str(toolbox_id or '').strip()}-{profile.normalized_profile_id()}"

    @staticmethod
    def _engine_id(toolbox_id: str, profile: SandboxProfileSpec, revision: str) -> str:
        return f"{str(toolbox_id or '').strip()}-{profile.normalized_profile_id()}-{str(revision or '')[:8]}"

    @staticmethod
    def _capabilities_for_profile(profile: SandboxProfileSpec) -> Dict[str, Any]:
        brokered = dict(dict(profile.sandbox_policy or {}).get("sandbox") or {}).get("brokered_io")
        return {
            "brokered_filesystem": bool(dict(brokered or {}).get("filesystem", False)),
            "brokered_http": bool(dict(brokered or {}).get("http", False)),
            "dynamic_reload": False,
        }

    @staticmethod
    def build_resolved_assignments(
        *,
        toolbox_id: str,
        profiles: Sequence[ResolvedToolboxProfileSpec],
        bundles: Sequence[ToolboxBundleSpec],
        profile_changes: Sequence[Dict[str, Any]],
    ) -> List[ResolvedToolboxSandboxAssignment]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id_required")
        if len(profiles) != len(bundles):
            raise ValueError("resolved_assignment_bundle_count_mismatch")
        profile_map = {item.profile_id: (item, bundle) for item, bundle in zip(profiles, bundles, strict=True)}
        if len(profile_map) != len(profiles):
            raise ValueError("resolved_assignment_profile_duplicate")
        changes = {
            str(item.get("proposed_profile_id") or "").strip(): dict(item)
            for item in profile_changes
            if str(item.get("proposed_profile_id") or "").strip()
        }
        if set(changes) != set(profile_map):
            raise ValueError("resolved_assignment_profile_changes_mismatch")
        assignments: List[ResolvedToolboxSandboxAssignment] = []
        for profile_id in sorted(profile_map):
            profile, bundle = profile_map[profile_id]
            change = changes[profile_id]
            assignments.append(
                ResolvedToolboxSandboxAssignment(
                    toolbox_id=tid,
                    resolved_profile=profile,
                    bundle_spec=bundle,
                    classification=str(change.get("classification") or ""),
                    active_profile_id=(
                        str(change.get("active_profile_id") or "").strip() or None
                    ),
                )
            )
        return assignments

    def spawn_resolved_assignments(
        self,
        *,
        toolbox_id: str,
        definition_revision: str,
        assignments: Sequence[ResolvedToolboxSandboxAssignment],
        resolved_environments: Optional[Dict[str, Any]] = None,
        worker_profile_class: str = "generic",
    ) -> List[ResolvedToolboxSandboxAssignment]:
        """Stage and spawn only added/replaced profiles as non-routable candidates."""

        tid = str(toolbox_id or "").strip()
        revision = str(definition_revision or "").strip()
        if not tid or not revision:
            raise ValueError("resolved_rollout_identity_required")
        out = list(assignments or [])
        current_target = detect_current_toolbox_target()
        pinned_environments = dict(resolved_environments or {})
        for item in out:
            if item.toolbox_id != tid:
                raise ValueError("resolved_assignment_toolbox_mismatch")
            if item.classification == "reused":
                continue
            profile = item.resolved_profile
            item.staged_bundle = self.stager.stage_bundle(item.bundle_spec)
            staged = item.staged_bundle
            bundle_revision = str(staged.manifest.get("bundle_revision") or "")
            logical_engine_id = f"{tid}-{profile.profile_id.removeprefix('sha256:')[:20]}-{bundle_revision[:8]}"
            # A semantic reapply must never reuse a prior runtime registration.
            # Keep the logical prefix for diagnostics, but make each concrete
            # candidate identity unique and non-replacing.
            engine_id = f"{logical_engine_id}-{uuid.uuid4().hex[:12]}"
            adoption_request_id = (
                f"toolbox:{tid}:{profile.profile_id.removeprefix('sha256:')[:24]}:"
                f"{revision.removeprefix('sha256:')[:24]}"
            )
            hermetic = self.service.materialize_toolbox_environment_for_bundle(
                files=list(staged.manifest.get("files") or []),
                python_abi=str(getattr(self.service, "_toolbox_required_python_abi", "") or "").strip()
                or current_target.python_abi,
                platform=str(getattr(self.service, "_toolbox_required_platform", "") or "").strip()
                or current_target.platform,
                declared_imports=profile.resolved_import_roots,
                intrinsic_names=list(staged.manifest.get("intrinsic_tool_names") or []),
                allowed_template_ids=(profile.template_id,),
                sandbox_policy=profile.sandbox_policy,
                resolved_environment=dict(pinned_environments.get(profile.profile_id) or {}),
            )
            if (
                hermetic.environment_key != profile.environment_key
                or (
                    getattr(hermetic.resolved, "custom_resolved_lock_digest", None)
                    or hermetic.resolved.complete_lock_digest
                ) != profile.effective_lock_digest
            ):
                raise RuntimeError("resolved_environment_identity_mismatch")
            generic_artifacts = []
            generic_dependencies = []
            for artifact in hermetic.resolved.locked_artifacts:
                imported = self.service._package_manager.import_verified_file(
                    source_id=artifact.source_id,
                    path=self.service._toolbox_artifact_store.object_path(artifact.sha256),
                    expected_digest=artifact.sha256,
                    actor_id="service:toolbox",
                    request_id=adoption_request_id,
                )
                generic_artifacts.append(imported)
                generic_dependencies.append({
                    "name": artifact.distribution_name,
                    "version": artifact.version,
                    "artifact_id": artifact.sha256,
                })
            generic_lock = self.service._package_manager.create_lock(
                lock_id=f"toolbox-{profile.profile_id.removeprefix('sha256:')[:32]}",
                revision=1,
                runtime_kind="python",
                platform=current_target.platform,
                artifacts=generic_artifacts,
                dependencies=generic_dependencies,
            )
            adopted = self.service._environment_manager.adopt_published(
                environment_id=hermetic.environment_key,
                consumer_kind="toolbox",
                consumer_id=tid,
                revision=1,
                template_id=profile.template_id,
                template_revision=1,
                package_lock_digest=str(generic_lock["lock_digest"]),
                runtime_kind="python",
                platform=current_target.platform,
                builder_id="python-environment-v1",
            )
            item.materialization_reference_id = str(adopted["reference"]["reference_id"])
            environment_spec = ToolboxEnvironmentSpec(
                venv_key=hermetic.environment_key,
                venv_path=hermetic.environment_root,
                python_executable=hermetic.python_executable,
                environment_name=profile.template_id,
                venv_lock_hash=profile.effective_lock_digest,
                toolbox_runtime_hash=hermetic.resolved.runtime_artifact_digest,
                intrinsics_profile_id="resolved",
                required_imports=list(profile.resolved_import_roots),
                dependency_lock_hash=profile.effective_lock_digest,
                environment_root_kind="environments",
                environment_consumer_kind="toolbox_executor",
            )
            registration_environment = self.runtime_base.registration_environment(
                environment=staged.registration_environment(environment_spec),
                toolbox_id=tid,
                sandbox_profile_id=profile.profile_id,
                bundle_revision=bundle_revision,
                sandbox_policy=dict(profile.sandbox_policy),
            )
            registration_environment.update(
                {
                    "environment_key": profile.environment_key,
                    "environment_reference": item.materialization_reference_id,
                    "verification_receipt_contract": "hosting.environment_receipt.v1",
                    "verification_state": "verified",
                }
            )
            registration_bundle = staged.registration_bundle()
            registration_bundle.update(
                {
                    "sandbox_profile_id": profile.profile_id,
                    "resolved_profile_id": profile.profile_id,
                    "definition_revision": revision,
                    "logical_engine_id": logical_engine_id,
                    "scratch_root": str((self.stager.hosting_root / "toolbox_scratch" / engine_id).resolve()),
                }
            )
            legacy_profile = SandboxProfileSpec(
                profile_id=profile.profile_id.removeprefix("sha256:"),
                required_imports=list(profile.resolved_import_roots),
                sandbox_policy=dict(profile.sandbox_policy),
            )
            item.registration = self.service.spawn(
                engine_id=engine_id,
                command=staged.worker_command(python_executable=hermetic.python_executable),
                env=staged.worker_env_with_startup_spec(
                    worker_id=engine_id,
                    sandbox_id=f"{tid}-{profile.profile_id.removeprefix('sha256:')[:20]}",
                    scratch_root=self.stager.hosting_root / "toolbox_scratch" / engine_id,
                    engines_state_file=self.service.engines_state_file,
                    control_state_file=self.service.control_state_file,
                    venv_path=hermetic.environment_root,
                    policy=dict(profile.sandbox_policy),
                ),
                worker_profile_class=worker_profile_class,
                sandbox_policy=dict(profile.sandbox_policy),
                executor_kind="toolbox_executor",
                bundle=registration_bundle,
                environment=registration_environment,
                tool_access=staged.registration_tool_access(),
                capabilities=self._capabilities_for_profile(legacy_profile),
                routing_state="candidate",
                runtime_id=engine_id,
                runtime_binding_digest=runtime_binding_digest(
                    toolbox_id=tid,
                    profile_id=profile.profile_id,
                    manifest_hash=str(registration_bundle.get("manifest_hash") or ""),
                    environment_reference=item.materialization_reference_id,
                    engine_id=engine_id,
                    definition_revision=revision,
                ),
            )
        return out

    def build_assignments(
        self,
        *,
        toolbox_id: str,
        requests: Sequence[ToolboxAutoAssignmentRequest],
        manual_requests: Optional[Sequence[ToolboxManualAssignmentRequest]] = None,
        intrinsic_tool_names: Optional[Sequence[str]] = None,
        intrinsic_profile: Optional[SandboxProfileSpec] = None,
        with_intrinsic_guides: bool = False,
    ) -> List[ToolboxSandboxAssignment]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id_required")
        grouped: Dict[str, Dict[str, Any]] = {}
        for request in list(requests or []):
            profile = request.sandbox_profile or SandboxProfileSpec()
            profile_key = profile.normalized_profile_id()
            row = grouped.setdefault(profile_key, {"profile": profile, "files": [], "auto_tools": [], "tools": []})
            row["files"].extend(list(request.files or []))
            row["auto_tools"].append(request.to_auto_tool())
        for request in list(manual_requests or []):
            profile = request.sandbox_profile or SandboxProfileSpec()
            profile_key = profile.normalized_profile_id()
            row = grouped.setdefault(profile_key, {"profile": profile, "files": [], "auto_tools": [], "tools": []})
            row["files"].extend(list(request.files or []))
            row["tools"].append(request.to_bundle_tool())
        out: List[ToolboxSandboxAssignment] = []
        for row in grouped.values():
            profile = row["profile"]
            file_map: Dict[str, ToolboxBundleFile] = {}
            for file_spec in list(row["files"] or []):
                file_map[file_spec.normalized_path()] = file_spec
            hidden_tool_names: List[str] = []
            for tool_spec in list(row["tools"] or []):
                if bool(getattr(tool_spec, "hidden", False)):
                    name = tool_spec.tool_name()
                    if name not in hidden_tool_names:
                        hidden_tool_names.append(name)
            for auto_spec in list(row["auto_tools"] or []):
                if bool(getattr(auto_spec, "hidden", False)):
                    name = auto_spec.tool_name()
                    if name not in hidden_tool_names:
                        hidden_tool_names.append(name)
            spec = ToolboxBundleSpec(
                bundle_id=self._bundle_id(tid, profile),
                toolbox_id=tid,
                sandbox_profile=profile,
                files=list(file_map.values()),
                tools=list(row["tools"] or []),
                auto_tools=list(row["auto_tools"] or []),
                hidden_tool_names=hidden_tool_names,
            )
            out.append(
                ToolboxSandboxAssignment(
                    toolbox_id=tid,
                    sandbox_profile=profile,
                    bundle_spec=spec,
                )
            )
        intrinsic_names = [str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()]
        if intrinsic_names:
            profile = intrinsic_profile or SandboxProfileSpec(profile_id="default")
            profile_id = profile.normalized_profile_id()
            existing = next((item for item in out if item.sandbox_profile.normalized_profile_id() == profile_id), None)
            if existing is None:
                existing = ToolboxSandboxAssignment(
                    toolbox_id=tid,
                    sandbox_profile=profile,
                    bundle_spec=ToolboxBundleSpec(
                        bundle_id=self._bundle_id(tid, profile),
                        toolbox_id=tid,
                        sandbox_profile=profile,
                    ),
                )
                out.append(existing)
            existing.bundle_spec.with_intrinsics = True
            existing.bundle_spec.with_intrinsic_guides = bool(with_intrinsic_guides)
            existing.bundle_spec.intrinsic_tool_names = intrinsic_names
            existing.bundle_spec.active_intrinsic_tool_names = intrinsic_names
        return sorted(out, key=lambda item: item.sandbox_profile.normalized_profile_id())

    def stage_assignments(
        self,
        *,
        toolbox_id: str,
        requests: Sequence[ToolboxAutoAssignmentRequest],
        manual_requests: Optional[Sequence[ToolboxManualAssignmentRequest]] = None,
        intrinsic_tool_names: Optional[Sequence[str]] = None,
        intrinsic_profile: Optional[SandboxProfileSpec] = None,
        with_intrinsic_guides: bool = False,
    ) -> List[ToolboxSandboxAssignment]:
        assignments = self.build_assignments(
            toolbox_id=toolbox_id,
            requests=requests,
            manual_requests=manual_requests,
            intrinsic_tool_names=intrinsic_tool_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
        )
        for item in assignments:
            item.staged_bundle = self.stager.stage_bundle(item.bundle_spec)
        return assignments

    def spawn_assignments(
        self,
        *,
        toolbox_id: str,
        requests: Sequence[ToolboxAutoAssignmentRequest],
        manual_requests: Optional[Sequence[ToolboxManualAssignmentRequest]] = None,
        intrinsic_tool_names: Optional[Sequence[str]] = None,
        intrinsic_profile: Optional[SandboxProfileSpec] = None,
        with_intrinsic_guides: bool = False,
        worker_profile_class: str = "generic",
    ) -> List[ToolboxSandboxAssignment]:
        assignments = self.stage_assignments(
            toolbox_id=toolbox_id,
            requests=requests,
            manual_requests=manual_requests,
            intrinsic_tool_names=intrinsic_tool_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
        )
        for item in assignments:
            if item.staged_bundle is None:
                raise RuntimeError("staged_bundle_required")
            staged = item.staged_bundle
            revision = str(staged.manifest.get("bundle_revision") or "")
            engine_id = self._engine_id(toolbox_id, item.sandbox_profile, revision)
            if getattr(self.service, "_hermetic_toolbox_environment_builder", None) is not None:
                current_target = detect_current_toolbox_target()
                python_abi = str(getattr(self.service, "_toolbox_required_python_abi", "") or "").strip()
                platform = str(getattr(self.service, "_toolbox_required_platform", "") or "").strip()
                python_abi = python_abi or current_target.python_abi
                platform = platform or current_target.platform
                hermetic = self.service.materialize_toolbox_environment_for_bundle(
                    files=list(staged.manifest.get("files") or []),
                    python_abi=python_abi,
                    platform=platform,
                    declared_imports=item.sandbox_profile.normalized_required_imports(),
                    intrinsic_names=list(staged.manifest.get("intrinsic_tool_names") or []),
                    sandbox_policy=dict(item.sandbox_profile.sandbox_policy or {}),
                )
                environment_spec = ToolboxEnvironmentSpec(
                    venv_key=hermetic.environment_key,
                    venv_path=hermetic.environment_root,
                    python_executable=hermetic.python_executable,
                    environment_name=hermetic.resolved.template_id,
                    environment_description_hash="",
                    venv_lock_hash=hermetic.resolved.complete_lock_digest,
                    toolbox_runtime_hash=hermetic.resolved.runtime_artifact_digest,
                    intrinsics_profile_id="resolved",
                    required_imports=list(hermetic.resolved.resolved_import_roots),
                    dependency_lock_hash=hermetic.resolved.complete_lock_digest,
                    environment_root_kind="environments",
                    environment_consumer_kind="toolbox_executor",
                )
            else:
                raise RuntimeError("hermetic_toolbox_environment_builder_required")
            registration_environment = self.runtime_base.registration_environment(
                environment=staged.registration_environment(environment_spec),
                toolbox_id=toolbox_id,
                sandbox_profile_id=item.sandbox_profile.normalized_profile_id(),
                bundle_revision=revision,
                sandbox_policy=dict(item.sandbox_profile.sandbox_policy or {}),
            )
            item.registration = self.service.spawn(
                engine_id=engine_id,
                command=staged.worker_command(
                    python_executable=environment_spec.python_executable or self.python_executable
                ),
                env=staged.worker_env_with_startup_spec(
                    worker_id=engine_id,
                    sandbox_id=f"{str(toolbox_id or '').strip()}-{item.sandbox_profile.normalized_profile_id()}",
                    scratch_root=self.stager.hosting_root / "toolbox_scratch" / engine_id,
                    engines_state_file=self.service.engines_state_file,
                    control_state_file=self.service.control_state_file,
                    venv_path=environment_spec.venv_path,
                    policy=dict(item.sandbox_profile.sandbox_policy or {}),
                ),
                worker_profile_class=worker_profile_class,
                sandbox_policy=dict(item.sandbox_profile.sandbox_policy or {}),
                executor_kind="toolbox_executor",
                bundle=staged.registration_bundle(),
                environment=registration_environment,
                tool_access=staged.registration_tool_access(),
                capabilities=self._capabilities_for_profile(item.sandbox_profile),
            )
        return assignments
