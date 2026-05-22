"""Toolbox bundle, sandbox, assignment, and harness data models."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .common import _sha256_text, _stable_json


@dataclass
class ToolboxBundleFile:
    relative_path: str
    content: str

    def normalized_path(self) -> str:
        raw = str(self.relative_path or "").replace("\\", "/").strip("/")
        if not raw or raw.startswith("../") or "/../" in f"/{raw}/":
            raise ValueError("bundle_file_path_invalid")
        return raw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.normalized_path(),
            "content_sha256": _sha256_text(str(self.content or "")),
        }

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.normalized_path(),
            "content": str(self.content or ""),
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxBundleFile":
        row = dict(payload or {})
        return cls(
            relative_path=str(row.get("relative_path") or "").strip(),
            content=str(row.get("content") or ""),
        )


@dataclass
class ToolboxBundleTool:
    definition: Dict[str, Any]
    entrypoint: str
    hidden: bool = False
    non_restartable: bool = False
    callback_signature: Optional[Dict[str, Any]] = None

    def tool_name(self) -> str:
        fn = dict(self.definition.get("function") or {})
        name = str(fn.get("name") or "").strip()
        if not name:
            raise ValueError("tool_name_required")
        return name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.tool_name(),
            "definition": dict(self.definition or {}),
            "entrypoint": str(self.entrypoint or "").strip(),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "callback_signature": dict(self.callback_signature or {}) or None,
        }


@dataclass
class ToolboxBundleAutoTool:
    module_name: str
    callable_name: str
    activate: bool = True
    hidden: bool = False
    non_restartable: bool = False
    guide_content: Optional[Dict[str, List[str]]] = None
    guide_description: Optional[str] = None
    callback_signature: Optional[Dict[str, Any]] = None

    def normalized_module_name(self) -> str:
        raw = str(self.module_name or "").strip()
        if not raw:
            raise ValueError("auto_tool_module_name_required")
        return raw

    def normalized_callable_name(self) -> str:
        raw = str(self.callable_name or "").strip()
        if not raw:
            raise ValueError("auto_tool_callable_name_required")
        return raw

    def tool_name(self) -> str:
        return self.normalized_callable_name()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.tool_name(),
            "module_name": self.normalized_module_name(),
            "callable_name": self.normalized_callable_name(),
            "activate": bool(self.activate),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "guide_content": dict(self.guide_content or {}) or None,
            "guide_description": str(self.guide_description or "").strip() or None,
            "callback_signature": dict(self.callback_signature or {}) or None,
        }


@dataclass
class SandboxProfileSpec:
    profile_id: str = ""
    environment_name: str = ""
    required_imports: List[str] = field(default_factory=list)
    sandbox_policy: Dict[str, Any] = field(default_factory=dict)

    def normalized_profile_id(self) -> str:
        raw = str(self.profile_id or "").strip()
        if raw:
            return raw
        return f"profile-{self.profile_fingerprint()[:12]}"

    def normalized_required_imports(self) -> List[str]:
        imports: List[str] = []
        seen: set[str] = set()
        for item in list(self.required_imports or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                imports.append(name)
        return imports

    def profile_fingerprint(self) -> str:
        payload = {
            "environment_name": str(self.environment_name or "").strip() or "base",
            "required_imports": self.normalized_required_imports(),
            "sandbox_policy": dict(self.sandbox_policy or {}),
        }
        return _sha256_text(_stable_json(payload))

    def intrinsics_profile_id(self, intrinsic_tool_names: Sequence[Any]) -> str:
        names = {
            str(item or "").strip()
            for item in list(intrinsic_tool_names or [])
            if str(item or "").strip()
        }
        uses_calculator = bool(
            {"scriptable_calculator", "scriptable_calculator_guide"} & names
        )
        uses_symbolic = bool(
            {"symbolic_algebra", "symbolic_algebra_guide"} & names
        )
        if uses_calculator and uses_symbolic:
            return "calculator+symbolic_math"
        if uses_symbolic:
            return "symbolic_math"
        if uses_calculator:
            return "calculator"
        return "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.normalized_profile_id(),
            "environment_name": str(self.environment_name or "").strip() or "base",
            "required_imports": self.normalized_required_imports(),
            "sandbox_policy": dict(self.sandbox_policy or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SandboxProfileSpec":
        row = dict(payload or {})
        return cls(
            profile_id=str(row.get("profile_id") or "").strip(),
            environment_name=str(row.get("environment_name") or "base").strip() or "base",
            required_imports=[str(item or "").strip() for item in list(row.get("required_imports") or []) if str(item or "").strip()],
            sandbox_policy=dict(row.get("sandbox_policy") or {}),
        )


@dataclass
class ToolboxAutoAssignmentRequest:
    files: List[ToolboxBundleFile]
    module_name: str
    callable_name: str
    sandbox_profile: SandboxProfileSpec = field(default_factory=SandboxProfileSpec)
    activate: bool = True
    hidden: bool = False
    non_restartable: bool = False
    guide_content: Optional[Dict[str, List[str]]] = None
    guide_description: Optional[str] = None
    callback_signature: Optional[Dict[str, Any]] = None

    def to_auto_tool(self) -> ToolboxBundleAutoTool:
        return ToolboxBundleAutoTool(
            module_name=str(self.module_name or "").strip(),
            callable_name=str(self.callable_name or "").strip(),
            activate=bool(self.activate),
            hidden=bool(self.hidden),
            non_restartable=bool(self.non_restartable),
            guide_content=dict(self.guide_content or {}) or None,
            guide_description=str(self.guide_description or "").strip() or None,
            callback_signature=dict(self.callback_signature or {}) or None,
        )

    def stable_key(self) -> str:
        return f"{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}"

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "files": [item.to_runtime_dict() for item in list(self.files or [])],
            "module_name": str(self.module_name or "").strip(),
            "callable_name": str(self.callable_name or "").strip(),
            "sandbox_profile": self.sandbox_profile.to_dict(),
            "activate": bool(self.activate),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "guide_content": dict(self.guide_content or {}) or None,
            "guide_description": str(self.guide_description or "").strip() or None,
            "callback_signature": dict(self.callback_signature or {}) or None,
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxAutoAssignmentRequest":
        row = dict(payload or {})
        return cls(
            files=[ToolboxBundleFile.from_runtime_dict(dict(item or {})) for item in list(row.get("files") or [])],
            module_name=str(row.get("module_name") or "").strip(),
            callable_name=str(row.get("callable_name") or "").strip(),
            sandbox_profile=SandboxProfileSpec.from_dict(dict(row.get("sandbox_profile") or {})),
            activate=bool(row.get("activate", True)),
            hidden=bool(row.get("hidden", False)),
            non_restartable=bool(row.get("non_restartable", False)),
            guide_content=dict(row.get("guide_content") or {}) or None,
            guide_description=str(row.get("guide_description") or "").strip() or None,
            callback_signature=dict(row.get("callback_signature") or {}) or None,
        )


@dataclass
class ToolboxManualAssignmentRequest:
    files: List[ToolboxBundleFile]
    module_name: str
    callable_name: str
    tool_definition: Dict[str, Any]
    sandbox_profile: SandboxProfileSpec = field(default_factory=SandboxProfileSpec)
    hidden: bool = False
    non_restartable: bool = False
    callback_signature: Optional[Dict[str, Any]] = None

    def to_bundle_tool(self) -> ToolboxBundleTool:
        return ToolboxBundleTool(
            definition=dict(self.tool_definition or {}),
            entrypoint=f"{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}",
            hidden=bool(self.hidden),
            non_restartable=bool(self.non_restartable),
            callback_signature=dict(self.callback_signature or {}) or None,
        )

    def stable_key(self) -> str:
        return f"manual:{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}"

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "files": [item.to_runtime_dict() for item in list(self.files or [])],
            "module_name": str(self.module_name or "").strip(),
            "callable_name": str(self.callable_name or "").strip(),
            "tool_definition": dict(self.tool_definition or {}),
            "sandbox_profile": self.sandbox_profile.to_dict(),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "callback_signature": dict(self.callback_signature or {}) or None,
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxManualAssignmentRequest":
        row = dict(payload or {})
        return cls(
            files=[ToolboxBundleFile.from_runtime_dict(dict(item or {})) for item in list(row.get("files") or [])],
            module_name=str(row.get("module_name") or "").strip(),
            callable_name=str(row.get("callable_name") or "").strip(),
            tool_definition=dict(row.get("tool_definition") or {}),
            sandbox_profile=SandboxProfileSpec.from_dict(dict(row.get("sandbox_profile") or {})),
            hidden=bool(row.get("hidden", False)),
            non_restartable=bool(row.get("non_restartable", False)),
            callback_signature=dict(row.get("callback_signature") or {}) or None,
        )


@dataclass
class ToolboxSandboxAssignment:
    toolbox_id: str
    sandbox_profile: SandboxProfileSpec
    bundle_spec: "ToolboxBundleSpec"
    staged_bundle: Optional["StagedToolboxBundle"] = None
    registration: Optional[Dict[str, Any]] = None


@dataclass
class ToolboxBundleSpec:
    bundle_id: str
    toolbox_id: Optional[str] = None
    sandbox_profile: Optional[SandboxProfileSpec] = None
    files: List[ToolboxBundleFile] = field(default_factory=list)
    tools: List[ToolboxBundleTool] = field(default_factory=list)
    auto_tools: List[ToolboxBundleAutoTool] = field(default_factory=list)
    with_intrinsics: bool = False
    with_intrinsic_guides: bool = False
    intrinsic_tool_names: List[str] = field(default_factory=list)
    active_intrinsic_tool_names: List[str] = field(default_factory=list)
    hidden_intrinsic_tool_names: List[str] = field(default_factory=list)
    hidden_tool_names: List[str] = field(default_factory=list)
    dependency_lock_hash: Optional[str] = None

    def normalized_bundle_id(self) -> str:
        raw = str(self.bundle_id or "").strip()
        if not raw:
            raise ValueError("bundle_id_required")
        return raw

    def normalized_toolbox_id(self) -> str:
        raw = str(self.toolbox_id or "").strip()
        return raw or self.normalized_bundle_id()

    def normalized_intrinsic_tool_names(self) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(self.intrinsic_tool_names or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @staticmethod
    def _normalize_name_list(items: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(items or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def manifest_payload(self) -> Dict[str, Any]:
        bundle_id = self.normalized_bundle_id()
        toolbox_id = self.normalized_toolbox_id()
        sandbox_profile = (self.sandbox_profile or SandboxProfileSpec(profile_id="default")).to_dict()
        tools = [item.to_dict() for item in self.tools]
        auto_tools = [item.to_dict() for item in self.auto_tools]
        intrinsic_tool_names = self.normalized_intrinsic_tool_names()
        if not tools and not auto_tools and not intrinsic_tool_names:
            raise ValueError("bundle_tools_required")
        files = [item.to_dict() for item in self.files]
        active_intrinsic_tool_names = self._normalize_name_list(
            self.active_intrinsic_tool_names if self.active_intrinsic_tool_names else intrinsic_tool_names
        )
        hidden_intrinsic_tool_names = self._normalize_name_list(self.hidden_intrinsic_tool_names)
        hidden_tool_names = self._normalize_name_list(
            list(self.hidden_tool_names)
            + [item.tool_name() for item in list(self.tools or []) if bool(getattr(item, "hidden", False))]
            + [item.tool_name() for item in list(self.auto_tools or []) if bool(getattr(item, "hidden", False))]
        )
        manifest_input = {
            "bundle_id": bundle_id,
            "toolbox_id": toolbox_id,
            "sandbox_profile": sandbox_profile,
            "tools": tools,
            "auto_tools": auto_tools,
            "files": files,
            "with_intrinsics": bool(self.with_intrinsics or bool(intrinsic_tool_names)),
            "with_intrinsic_guides": bool(self.with_intrinsic_guides),
            "intrinsic_tool_names": intrinsic_tool_names,
            "active_intrinsic_tool_names": active_intrinsic_tool_names,
            "hidden_intrinsic_tool_names": hidden_intrinsic_tool_names,
            "hidden_tool_names": hidden_tool_names,
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
        }
        manifest_hash = _sha256_text(_stable_json(manifest_input))
        bundle_revision = manifest_hash[:16]
        return {
            "executor_kind": "toolbox_executor",
            "bundle_id": bundle_id,
            "toolbox_id": toolbox_id,
            "sandbox_profile": sandbox_profile,
            "bundle_revision": bundle_revision,
            "manifest_hash": manifest_hash,
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
            "tools": tools,
            "auto_tools": auto_tools,
            "files": files,
            "with_intrinsics": bool(self.with_intrinsics or bool(intrinsic_tool_names)),
            "with_intrinsic_guides": bool(self.with_intrinsic_guides),
            "intrinsic_tool_names": intrinsic_tool_names,
            "active_intrinsic_tool_names": active_intrinsic_tool_names,
            "hidden_intrinsic_tool_names": hidden_intrinsic_tool_names,
            "hidden_tool_names": hidden_tool_names,
        }


@dataclass
class ToolboxWorkerStartupSpec:
    worker_id: str
    sandbox_id: str
    toolbox_revision: str
    manifest_path: str
    scratch_root: str
    engines_state_file: Optional[str] = None
    control_state_file: Optional[str] = None
    venv_path: Optional[str] = None
    ipc_family: str = field(default_factory=lambda: "AF_PIPE" if os.name == "nt" else "AF_UNIX")
    ipc_address: str = ""
    auth_token_env: str = "MP13_ENGINE_HOST_TOKEN"
    execution_contract: str = "hosting.toolbox.worker.v1"
    callback_contract: str = "hosting.toolbox.callbacks.v1"
    policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        default_ipc_family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        return {
            "worker_id": str(self.worker_id or "").strip(),
            "sandbox_id": str(self.sandbox_id or "").strip(),
            "toolbox_revision": str(self.toolbox_revision or "").strip(),
            "manifest_path": str(self.manifest_path or "").strip(),
            "scratch_root": str(self.scratch_root or "").strip(),
            "engines_state_file": str(self.engines_state_file or "").strip() or None,
            "control_state_file": str(self.control_state_file or "").strip() or None,
            "venv_path": str(self.venv_path or "").strip() or None,
            "ipc_family": str(self.ipc_family or default_ipc_family).strip() or default_ipc_family,
            "ipc_address": str(self.ipc_address or "").strip(),
            "auth_token_env": str(self.auth_token_env or "MP13_ENGINE_HOST_TOKEN").strip() or "MP13_ENGINE_HOST_TOKEN",
            "execution_contract": str(self.execution_contract or "hosting.toolbox.worker.v1").strip() or "hosting.toolbox.worker.v1",
            "callback_contract": str(self.callback_contract or "hosting.toolbox.callbacks.v1").strip() or "hosting.toolbox.callbacks.v1",
            "policy": dict(self.policy or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToolboxWorkerStartupSpec":
        row = dict(payload or {})
        default_ipc_family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        return cls(
            worker_id=str(row.get("worker_id") or "").strip(),
            sandbox_id=str(row.get("sandbox_id") or "").strip(),
            toolbox_revision=str(row.get("toolbox_revision") or "").strip(),
            manifest_path=str(row.get("manifest_path") or "").strip(),
            scratch_root=str(row.get("scratch_root") or "").strip(),
            engines_state_file=str(row.get("engines_state_file") or "").strip() or None,
            control_state_file=str(row.get("control_state_file") or "").strip() or None,
            venv_path=str(row.get("venv_path") or "").strip() or None,
            ipc_family=str(row.get("ipc_family") or default_ipc_family).strip() or default_ipc_family,
            ipc_address=str(row.get("ipc_address") or "").strip(),
            auth_token_env=str(row.get("auth_token_env") or "MP13_ENGINE_HOST_TOKEN").strip() or "MP13_ENGINE_HOST_TOKEN",
            execution_contract=str(row.get("execution_contract") or "hosting.toolbox.worker.v1").strip() or "hosting.toolbox.worker.v1",
            callback_contract=str(row.get("callback_contract") or "hosting.toolbox.callbacks.v1").strip() or "hosting.toolbox.callbacks.v1",
            policy=dict(row.get("policy") or {}),
        )

    def write_json(self, path: Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


@dataclass
class ToolboxEnvironmentSpec:
    venv_key: str
    venv_path: str
    python_executable: str = ""
    environment_name: str = "base"
    environment_description_hash: str = ""
    venv_lock_hash: Optional[str] = None
    toolbox_runtime_hash: str = "toolbox-executor-v1"
    intrinsics_profile_id: str = "none"
    required_imports: List[str] = field(default_factory=list)
    dependency_lock_hash: Optional[str] = None
    environment_root_kind: str = "toolbox_venvs"
    environment_consumer_kind: str = "toolbox_executor"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venv_key": str(self.venv_key or "").strip(),
            "venv_path": str(self.venv_path or "").strip(),
            "python_executable": str(self.python_executable or "").strip(),
            "environment_name": str(self.environment_name or "base").strip() or "base",
            "environment_description_hash": str(self.environment_description_hash or "").strip() or None,
            "venv_lock_hash": str(self.venv_lock_hash or "").strip() or None,
            "toolbox_runtime_hash": str(self.toolbox_runtime_hash or "toolbox-executor-v1").strip() or "toolbox-executor-v1",
            "intrinsics_profile_id": str(self.intrinsics_profile_id or "none").strip() or "none",
            "required_imports": [str(item or "").strip() for item in list(self.required_imports or []) if str(item or "").strip()],
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
            "environment_root_kind": str(self.environment_root_kind or "toolbox_venvs").strip() or "toolbox_venvs",
            "environment_consumer_kind": str(self.environment_consumer_kind or "toolbox_executor").strip() or "toolbox_executor",
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToolboxEnvironmentSpec":
        row = dict(payload or {})
        return cls(
            venv_key=str(row.get("venv_key") or "").strip(),
            venv_path=str(row.get("venv_path") or "").strip(),
            python_executable=str(row.get("python_executable") or "").strip(),
            environment_name=str(row.get("environment_name") or "base").strip() or "base",
            environment_description_hash=str(row.get("environment_description_hash") or "").strip() or None,
            venv_lock_hash=str(row.get("venv_lock_hash") or "").strip() or None,
            toolbox_runtime_hash=str(row.get("toolbox_runtime_hash") or "toolbox-executor-v1").strip() or "toolbox-executor-v1",
            intrinsics_profile_id=str(row.get("intrinsics_profile_id") or "none").strip() or "none",
            required_imports=[str(item or "").strip() for item in list(row.get("required_imports") or []) if str(item or "").strip()],
            dependency_lock_hash=str(row.get("dependency_lock_hash") or "").strip() or None,
            environment_root_kind=str(row.get("environment_root_kind") or "toolbox_venvs").strip() or "toolbox_venvs",
            environment_consumer_kind=str(row.get("environment_consumer_kind") or "toolbox_executor").strip() or "toolbox_executor",
        )

@dataclass
class ToolboxHarnessConfig:
    mode: str = "native"
    sandbox_toolbox_id: Optional[str] = None
    sandbox_engine_ids: List[str] = field(default_factory=list)
    sandbox_selection: str = "round_robin"
