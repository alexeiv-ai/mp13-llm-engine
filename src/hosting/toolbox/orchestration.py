"""Sandboxed toolbox assignment orchestration."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence

from .bundle_models import (
    SandboxProfileSpec,
    ToolboxAutoAssignmentRequest,
    ToolboxBundleFile,
    ToolboxBundleSpec,
    ToolboxManualAssignmentRequest,
    ToolboxSandboxAssignment,
)
from .environment import ToolboxEnvironmentManager
from .staging import ToolboxBundleStager


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
        self.environment_manager = ToolboxEnvironmentManager(self.stager.hosting_root)

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
            environment_name = str(item.sandbox_profile.environment_name or "base").strip() or "base"
            environment_description = None
            if hasattr(self.service, "toolbox_environment_description_effective_get"):
                try:
                    environment_description = self.service.toolbox_environment_description_effective_get(environment_name)
                except Exception:
                    environment_description = None
            elif hasattr(self.service, "toolbox_environment_description_get"):
                try:
                    environment_description = self.service.toolbox_environment_description_get(environment_name)
                except Exception:
                    environment_description = None
            environment_spec = self.environment_manager.ensure_for_bundle(
                staged,
                environment_description=environment_description,
            )
            environment_spec.python_executable = self.environment_manager.runtime_python_executable(
                environment_spec,
                fallback_python_executable=self.python_executable,
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
                environment=staged.registration_environment(environment_spec),
                tool_access=staged.registration_tool_access(),
                capabilities=self._capabilities_for_profile(item.sandbox_profile),
            )
        return assignments
