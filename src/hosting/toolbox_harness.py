from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import os
import sys
import tempfile
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mp13_engine.mp13_config import ToolCall
from mp13_engine.mp13_toolbox import Toolbox


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        }


@dataclass
class ToolboxBundleAutoTool:
    module_name: str
    callable_name: str
    activate: bool = True
    guide_content: Optional[Dict[str, List[str]]] = None
    guide_description: Optional[str] = None

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
            "guide_content": dict(self.guide_content or {}) or None,
            "guide_description": str(self.guide_description or "").strip() or None,
        }


@dataclass
class SandboxProfileSpec:
    profile_id: str = ""
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
            "required_imports": self.normalized_required_imports(),
            "sandbox_policy": dict(self.sandbox_policy or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SandboxProfileSpec":
        row = dict(payload or {})
        return cls(
            profile_id=str(row.get("profile_id") or "").strip(),
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
    guide_content: Optional[Dict[str, List[str]]] = None
    guide_description: Optional[str] = None

    def to_auto_tool(self) -> ToolboxBundleAutoTool:
        return ToolboxBundleAutoTool(
            module_name=str(self.module_name or "").strip(),
            callable_name=str(self.callable_name or "").strip(),
            activate=bool(self.activate),
            guide_content=dict(self.guide_content or {}) or None,
            guide_description=str(self.guide_description or "").strip() or None,
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
            "guide_content": dict(self.guide_content or {}) or None,
            "guide_description": str(self.guide_description or "").strip() or None,
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
            guide_content=dict(row.get("guide_content") or {}) or None,
            guide_description=str(row.get("guide_description") or "").strip() or None,
        )


@dataclass
class ToolboxManualAssignmentRequest:
    files: List[ToolboxBundleFile]
    module_name: str
    callable_name: str
    tool_definition: Dict[str, Any]
    sandbox_profile: SandboxProfileSpec = field(default_factory=SandboxProfileSpec)

    def to_bundle_tool(self) -> ToolboxBundleTool:
        return ToolboxBundleTool(
            definition=dict(self.tool_definition or {}),
            entrypoint=f"{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}",
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
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
        }
        manifest_hash = _sha256_text(_stable_json(manifest_input))
        bundle_revision = manifest_hash[:16]
        return {
            "executor_kind": "toolbox_executor_v1",
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
    ipc_family: str = "AF_PIPE"
    ipc_address: str = ""
    auth_token_env: str = "MP13_ENGINE_HOST_TOKEN"
    execution_contract: str = "hosting.toolbox.worker.v1"
    callback_contract: str = "hosting.toolbox.callbacks.v1"
    policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": str(self.worker_id or "").strip(),
            "sandbox_id": str(self.sandbox_id or "").strip(),
            "toolbox_revision": str(self.toolbox_revision or "").strip(),
            "manifest_path": str(self.manifest_path or "").strip(),
            "scratch_root": str(self.scratch_root or "").strip(),
            "engines_state_file": str(self.engines_state_file or "").strip() or None,
            "control_state_file": str(self.control_state_file or "").strip() or None,
            "venv_path": str(self.venv_path or "").strip() or None,
            "ipc_family": str(self.ipc_family or "AF_PIPE").strip() or "AF_PIPE",
            "ipc_address": str(self.ipc_address or "").strip(),
            "auth_token_env": str(self.auth_token_env or "MP13_ENGINE_HOST_TOKEN").strip() or "MP13_ENGINE_HOST_TOKEN",
            "execution_contract": str(self.execution_contract or "hosting.toolbox.worker.v1").strip() or "hosting.toolbox.worker.v1",
            "callback_contract": str(self.callback_contract or "hosting.toolbox.callbacks.v1").strip() or "hosting.toolbox.callbacks.v1",
            "policy": dict(self.policy or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToolboxWorkerStartupSpec":
        row = dict(payload or {})
        return cls(
            worker_id=str(row.get("worker_id") or "").strip(),
            sandbox_id=str(row.get("sandbox_id") or "").strip(),
            toolbox_revision=str(row.get("toolbox_revision") or "").strip(),
            manifest_path=str(row.get("manifest_path") or "").strip(),
            scratch_root=str(row.get("scratch_root") or "").strip(),
            engines_state_file=str(row.get("engines_state_file") or "").strip() or None,
            control_state_file=str(row.get("control_state_file") or "").strip() or None,
            venv_path=str(row.get("venv_path") or "").strip() or None,
            ipc_family=str(row.get("ipc_family") or "AF_PIPE").strip() or "AF_PIPE",
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
    venv_lock_hash: Optional[str] = None
    toolbox_runtime_hash: str = "toolbox-executor-v1"
    intrinsics_profile_id: str = "none"
    required_imports: List[str] = field(default_factory=list)
    dependency_lock_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venv_key": str(self.venv_key or "").strip(),
            "venv_path": str(self.venv_path or "").strip(),
            "python_executable": str(self.python_executable or "").strip(),
            "venv_lock_hash": str(self.venv_lock_hash or "").strip() or None,
            "toolbox_runtime_hash": str(self.toolbox_runtime_hash or "toolbox-executor-v1").strip() or "toolbox-executor-v1",
            "intrinsics_profile_id": str(self.intrinsics_profile_id or "none").strip() or "none",
            "required_imports": [str(item or "").strip() for item in list(self.required_imports or []) if str(item or "").strip()],
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToolboxEnvironmentSpec":
        row = dict(payload or {})
        return cls(
            venv_key=str(row.get("venv_key") or "").strip(),
            venv_path=str(row.get("venv_path") or "").strip(),
            python_executable=str(row.get("python_executable") or "").strip(),
            venv_lock_hash=str(row.get("venv_lock_hash") or "").strip() or None,
            toolbox_runtime_hash=str(row.get("toolbox_runtime_hash") or "toolbox-executor-v1").strip() or "toolbox-executor-v1",
            intrinsics_profile_id=str(row.get("intrinsics_profile_id") or "none").strip() or "none",
            required_imports=[str(item or "").strip() for item in list(row.get("required_imports") or []) if str(item or "").strip()],
            dependency_lock_hash=str(row.get("dependency_lock_hash") or "").strip() or None,
        )


class ToolboxEnvironmentManager:
    def __init__(self, hosting_root: Path):
        self.hosting_root = Path(hosting_root).expanduser().resolve()
        self.environments_root = (self.hosting_root / "toolbox_venvs").resolve()

    @staticmethod
    def _fingerprint_payload(payload: Dict[str, Any]) -> str:
        return _sha256_text(_stable_json(payload))

    def environment_spec_for_bundle(self, staged: "StagedToolboxBundle") -> ToolboxEnvironmentSpec:
        manifest = dict(staged.manifest or {})
        sandbox_profile = SandboxProfileSpec.from_dict(dict(manifest.get("sandbox_profile") or {}))
        intrinsic_tool_names = list(manifest.get("intrinsic_tool_names") or [])
        intrinsics_profile_id = sandbox_profile.intrinsics_profile_id(intrinsic_tool_names)
        dependency_lock_hash = str(manifest.get("dependency_lock_hash") or "").strip() or None
        required_imports = sandbox_profile.normalized_required_imports()
        toolbox_runtime_hash = "toolbox-executor-v1"
        venv_key = self._fingerprint_payload(
            {
                "toolbox_runtime_hash": toolbox_runtime_hash,
                "intrinsics_profile_id": intrinsics_profile_id,
                "required_imports": required_imports,
                "dependency_lock_hash": dependency_lock_hash,
            }
        )[:16]
        venv_root = (self.environments_root / venv_key).resolve()
        venv_path = str(venv_root)
        venv_lock_hash = dependency_lock_hash or self._fingerprint_payload(
            {
                "intrinsics_profile_id": intrinsics_profile_id,
                "required_imports": required_imports,
            }
        )[:16]
        return ToolboxEnvironmentSpec(
            venv_key=venv_key,
            venv_path=venv_path,
            python_executable=str(self.python_executable_path(venv_root)),
            venv_lock_hash=venv_lock_hash,
            toolbox_runtime_hash=toolbox_runtime_hash,
            intrinsics_profile_id=intrinsics_profile_id,
            required_imports=required_imports,
            dependency_lock_hash=dependency_lock_hash,
        )

    @staticmethod
    def python_executable_path(venv_root: Path) -> Path:
        base = Path(venv_root).expanduser().resolve()
        if os.name == "nt":
            return base / "Scripts" / "python.exe"
        return base / "bin" / "python"

    def ensure_environment(self, spec: ToolboxEnvironmentSpec) -> ToolboxEnvironmentSpec:
        target = Path(spec.venv_path).expanduser().resolve()
        if not (target / "pyvenv.cfg").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reuse the current interpreter's site packages for now so sandbox worker
            # execution remains functional before locked dependency installs are added.
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(str(target))
        spec.python_executable = str(self.python_executable_path(target))
        metadata_path = target / "environment.json"
        metadata_path.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return spec

    def ensure_for_bundle(self, staged: "StagedToolboxBundle") -> ToolboxEnvironmentSpec:
        return self.ensure_environment(self.environment_spec_for_bundle(staged))


@dataclass
class StagedToolboxBundle:
    bundle_root: Path
    manifest_path: Path
    manifest: Dict[str, Any]

    def registration_bundle(self) -> Dict[str, Any]:
        return {
            "bundle_id": str(self.manifest.get("bundle_id") or ""),
            "toolbox_id": str(self.manifest.get("toolbox_id") or self.manifest.get("bundle_id") or ""),
            "sandbox_profile_id": str(dict(self.manifest.get("sandbox_profile") or {}).get("profile_id") or "default"),
            "bundle_revision": str(self.manifest.get("bundle_revision") or ""),
            "manifest_hash": str(self.manifest.get("manifest_hash") or ""),
            "bundle_root": str(self.bundle_root),
            "manifest_path": str(self.manifest_path),
        }

    def registration_environment(self, environment_spec: Optional[ToolboxEnvironmentSpec] = None) -> Dict[str, Any]:
        spec = environment_spec
        if spec is None:
            spec = ToolboxEnvironmentManager(self.bundle_root.parents[2]).environment_spec_for_bundle(self)
        return {
            "venv_key": spec.venv_key,
            "venv_path": spec.venv_path,
            "python_executable": spec.python_executable,
            "venv_lock_hash": spec.venv_lock_hash,
            "venv_mutable": False,
            "toolbox_runtime_hash": spec.toolbox_runtime_hash,
            "intrinsics_profile_id": spec.intrinsics_profile_id,
            "required_imports": list(spec.required_imports or []),
            "dependency_lock_hash": spec.dependency_lock_hash,
        }

    def registration_tool_access(self) -> Dict[str, Any]:
        tool_names = [str(item.get("name") or "").strip() for item in list(self.manifest.get("tools") or [])]
        tool_names = [name for name in tool_names if name]
        auto_tool_names = [str(item.get("name") or "").strip() for item in list(self.manifest.get("auto_tools") or [])]
        for name in auto_tool_names:
            if name and name not in tool_names:
                tool_names.append(name)
        active_intrinsic_names = [
            str(item or "").strip()
            for item in list(self.manifest.get("active_intrinsic_tool_names") or [])
            if str(item or "").strip()
        ]
        hidden_intrinsic_names = {
            str(item or "").strip()
            for item in list(self.manifest.get("hidden_intrinsic_tool_names") or [])
            if str(item or "").strip()
        }
        allowed_tool_names = list(tool_names)
        for name in active_intrinsic_names:
            if name not in allowed_tool_names:
                allowed_tool_names.append(name)
        advertised_tool_names = [name for name in allowed_tool_names if name not in hidden_intrinsic_names]
        sandbox_profile_id = str(dict(self.manifest.get("sandbox_profile") or {}).get("profile_id") or "default")
        return {
            "allowed_tool_names": allowed_tool_names,
            "advertised_tool_names": advertised_tool_names,
            "tool_routes": {
                name: {
                    "toolbox_id": str(self.manifest.get("toolbox_id") or self.manifest.get("bundle_id") or ""),
                    "sandbox_profile_id": sandbox_profile_id,
                }
                for name in allowed_tool_names
            },
        }

    def worker_command(self, *, python_executable: Optional[str] = None) -> List[str]:
        return [
            str(python_executable or sys.executable),
            "-m",
            "hosting.toolbox_executor_ipc",
        ]

    def worker_startup_spec(
        self,
        *,
        worker_id: str,
        sandbox_id: Optional[str] = None,
        scratch_root: Optional[Path] = None,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
        venv_path: Optional[str] = None,
        ipc_family: str = "AF_PIPE",
        ipc_address: str = "",
        policy: Optional[Dict[str, Any]] = None,
    ) -> ToolboxWorkerStartupSpec:
        scratch = Path(scratch_root or (self.bundle_root / "scratch")).expanduser().resolve()
        return ToolboxWorkerStartupSpec(
            worker_id=str(worker_id or "").strip(),
            sandbox_id=str(sandbox_id or worker_id or "").strip(),
            toolbox_revision=str(self.manifest.get("bundle_revision") or "").strip(),
            manifest_path=str(self.manifest_path),
            scratch_root=str(scratch),
            engines_state_file=str(Path(engines_state_file).expanduser().resolve()) if engines_state_file else None,
            control_state_file=str(Path(control_state_file).expanduser().resolve()) if control_state_file else None,
            venv_path=str(venv_path or "").strip() or None,
            ipc_family=str(ipc_family or "AF_PIPE").strip() or "AF_PIPE",
            ipc_address=str(ipc_address or "").strip(),
            policy=dict(policy or {}),
        )

    def worker_env(self, *, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        src_root = str(Path(__file__).resolve().parents[1])
        env = {str(k): str(v) for k, v in dict(extra_env or {}).items()}
        env["MP13_TOOLBOX_MANIFEST_PATH"] = str(self.manifest_path)
        current_py = str(env.get("PYTHONPATH") or "")
        paths = [p for p in current_py.split(os.pathsep) if p] if current_py else []
        if src_root not in paths:
            env["PYTHONPATH"] = src_root if not current_py else f"{src_root}{os.pathsep}{current_py}"
        return env

    def worker_env_with_startup_spec(
        self,
        *,
        worker_id: str,
        sandbox_id: Optional[str] = None,
        scratch_root: Optional[Path] = None,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
        venv_path: Optional[str] = None,
        ipc_family: str = "AF_PIPE",
        ipc_address: str = "",
        policy: Optional[Dict[str, Any]] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        env = self.worker_env(extra_env=extra_env)
        spec = self.worker_startup_spec(
            worker_id=worker_id,
            sandbox_id=sandbox_id,
            scratch_root=scratch_root,
            engines_state_file=engines_state_file,
            control_state_file=control_state_file,
            venv_path=venv_path,
            ipc_family=ipc_family,
            ipc_address=ipc_address,
            policy=policy,
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"mp13-toolbox-startup-{spec.worker_id or 'worker'}-",
            suffix=".json",
            dir=str(self.bundle_root),
        )
        os.close(fd)
        spec.write_json(Path(tmp_name))
        env["MP13_TOOLBOX_WORKER_SPEC_PATH"] = str(Path(tmp_name).resolve())
        return env


class ToolboxBundleStager:
    def __init__(self, hosting_root: Path):
        self.hosting_root = Path(hosting_root).expanduser().resolve()

    def stage_bundle(self, spec: ToolboxBundleSpec) -> StagedToolboxBundle:
        manifest = spec.manifest_payload()
        bundle_root = (
            self.hosting_root
            / "toolbox_bundles"
            / str(manifest["bundle_id"])
            / str(manifest["bundle_revision"])
        ).resolve()
        files_root = (bundle_root / "files").resolve()
        files_root.mkdir(parents=True, exist_ok=True)
        for file_spec in spec.files:
            rel = file_spec.normalized_path()
            target = (files_root / rel).resolve()
            if files_root not in target.parents and target != files_root:
                raise ValueError("bundle_file_path_invalid")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(file_spec.content or ""), encoding="utf-8")
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return StagedToolboxBundle(bundle_root=bundle_root, manifest_path=manifest_path, manifest=manifest)


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
            spec = ToolboxBundleSpec(
                bundle_id=self._bundle_id(tid, profile),
                toolbox_id=tid,
                sandbox_profile=profile,
                files=list(file_map.values()),
                tools=list(row["tools"] or []),
                auto_tools=list(row["auto_tools"] or []),
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
            environment_spec = self.environment_manager.ensure_for_bundle(staged)
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
                executor_kind="toolbox_executor_v1",
                bundle=staged.registration_bundle(),
                environment=staged.registration_environment(environment_spec),
                tool_access=staged.registration_tool_access(),
                capabilities=self._capabilities_for_profile(item.sandbox_profile),
            )
        return assignments


@dataclass
class ToolboxHarnessConfig:
    mode: str = "native"
    sandbox_toolbox_id: Optional[str] = None
    sandbox_engine_ids: List[str] = field(default_factory=list)
    sandbox_selection: str = "round_robin"


class ToolboxExecutionHarness:
    def __init__(
        self,
        *,
        config: Optional[ToolboxHarnessConfig] = None,
        native_toolbox: Optional[Toolbox] = None,
        control_channel: Optional[Any] = None,
    ) -> None:
        self.config = config or ToolboxHarnessConfig()
        self.native_toolbox = native_toolbox
        self.control_channel = control_channel
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()

    async def describe(self) -> Dict[str, Any]:
        mode = str(self.config.mode or "native").strip().lower()
        if mode == "native":
            if self.native_toolbox is None:
                raise RuntimeError("native_toolbox_not_configured")
            names = sorted(list(self.native_toolbox._registered_tool_names()))
            return {
                "mode": "native",
                "executor_kind": "native_toolbox",
                "tool_names": names,
                "parallel_execution": {
                    "async_within_executor": True,
                    "sandbox_pool": False,
                },
            }
        engine_id = await self._select_engine_id()
        toolbox_id = str(self.config.sandbox_toolbox_id or "").strip()
        if toolbox_id:
            result = await asyncio.to_thread(self.control_channel.toolbox_describe, toolbox_id=toolbox_id)
        else:
            result = await asyncio.to_thread(self.control_channel.toolbox_describe, engine_id=engine_id)
        out = dict(result or {})
        out.setdefault("mode", "sandbox")
        out.setdefault(
            "parallel_execution",
            {
                "async_within_executor": True,
                "sandbox_pool": len(self.config.sandbox_engine_ids) > 1,
            },
        )
        return out

    async def execute_calls(
        self,
        tool_calls: Sequence[ToolCall | Dict[str, Any]],
        *,
        parallel: bool = True,
        timeout_seconds: float = 30.0,
        native_execute_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[ToolCall]:
        calls = [item if isinstance(item, ToolCall) else ToolCall.from_dict(dict(item or {})) for item in list(tool_calls or [])]
        if not calls:
            return []
        if not parallel:
            out: List[ToolCall] = []
            for call in calls:
                out.append(
                    await self._execute_one(
                        call,
                        timeout_seconds=timeout_seconds,
                        native_execute_kwargs=dict(native_execute_kwargs or {}),
                    )
                )
            return out
        tasks = [
            self._execute_one(
                call,
                timeout_seconds=timeout_seconds,
                native_execute_kwargs=dict(native_execute_kwargs or {}),
            )
            for call in calls
        ]
        return list(await asyncio.gather(*tasks))

    async def _execute_one(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float,
        native_execute_kwargs: Dict[str, Any],
    ) -> ToolCall:
        mode = str(self.config.mode or "native").strip().lower()
        if mode == "native":
            if self.native_toolbox is None:
                raise RuntimeError("native_toolbox_not_configured")
            result = await self.native_toolbox.execute(call, **dict(native_execute_kwargs or {}))
            if result is not None:
                call.result = result
            return call
        engine_id = await self._select_engine_id()
        toolbox_id = str(self.config.sandbox_toolbox_id or "").strip()
        if toolbox_id:
            rpc_out = await asyncio.to_thread(
                self.control_channel.toolbox_execute,
                toolbox_id=toolbox_id,
                tool_call=call.to_dict(),
                timeout_seconds=float(timeout_seconds or 30.0),
            )
        else:
            rpc_out = await asyncio.to_thread(
                self.control_channel.toolbox_execute,
                engine_id=engine_id,
                tool_call=call.to_dict(),
                timeout_seconds=float(timeout_seconds or 30.0),
            )
        payload = dict(rpc_out or {})
        tool_out = dict(payload.get("tool_call") or {})
        return ToolCall.from_dict(tool_out) if tool_out else call

    async def _select_engine_id(self) -> str:
        if str(self.config.sandbox_toolbox_id or "").strip():
            return ""
        if self.control_channel is None:
            raise RuntimeError("control_channel_not_configured")
        engine_ids = [str(item or "").strip() for item in list(self.config.sandbox_engine_ids or []) if str(item or "").strip()]
        if not engine_ids:
            raise RuntimeError("sandbox_engine_ids_required")
        if len(engine_ids) == 1 or str(self.config.sandbox_selection or "round_robin").strip().lower() != "round_robin":
            return engine_ids[0]
        async with self._rr_lock:
            engine_id = engine_ids[self._rr_index % len(engine_ids)]
            self._rr_index = (self._rr_index + 1) % max(1, len(engine_ids))
            return engine_id


class SandboxedToolboxFacade:
    def __init__(
        self,
        *,
        toolbox_id: str,
        host: Any,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> None:
        self.toolbox_id = str(toolbox_id or "").strip()
        if not self.toolbox_id:
            raise ValueError("toolbox_id_required")
        self.host = host
        self.python_executable = str(python_executable or "").strip() or None
        self.worker_profile_class = str(worker_profile_class or "generic").strip() or "generic"

    def register_auto_callable(
        self,
        *,
        relative_path: str,
        content: str,
        module_name: str,
        callable_name: str,
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        request = {
            "files": [
                ToolboxBundleFile(
                    relative_path=str(relative_path or "").strip(),
                    content=str(content or ""),
                ).to_runtime_dict()
            ],
            "module_name": str(module_name or "").strip(),
            "callable_name": str(callable_name or "").strip(),
            "sandbox_profile": SandboxProfileSpec(
                required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                sandbox_policy=dict(sandbox_policy or {}),
            ).to_dict(),
            "activate": bool(activate),
            "guide_content": dict(guide_content or {}) or None,
            "guide_description": str(guide_description or "").strip() or None,
        }
        return dict(
            self.host.toolbox_register_auto(
                toolbox_id=self.toolbox_id,
                requests=[request],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def register_python_callable(
        self,
        implementation: Any,
        *,
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        return self.register_auto_callable(
            relative_path=source_file.name,
            content=source_file.read_text(encoding="utf-8"),
            module_name=module_name,
            callable_name=callable_name,
            required_imports=required_imports,
            sandbox_policy=sandbox_policy,
            activate=activate,
            guide_content=guide_content,
            guide_description=guide_description,
        )

    def register_manual_tool(
        self,
        tool_definition: Dict[str, Any],
        implementation: Any,
        *,
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        return dict(
            self.host.toolbox_register_manual(
                toolbox_id=self.toolbox_id,
                requests=[
                    {
                        "files": [
                            ToolboxBundleFile(
                                relative_path=source_file.name,
                                content=source_file.read_text(encoding="utf-8"),
                            ).to_runtime_dict()
                        ],
                        "module_name": module_name,
                        "callable_name": callable_name,
                        "tool_definition": dict(tool_definition or {}),
                        "sandbox_profile": SandboxProfileSpec(
                            required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                            sandbox_policy=dict(sandbox_policy or {}),
                        ).to_dict(),
                    }
                ],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def unregister_manual_tool(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        key = f"manual:{str(module_name or '').strip()}:{str(callable_name or '').strip()}"
        return dict(
            self.host.toolbox_unregister_manual(
                toolbox_id=self.toolbox_id,
                tool_keys=[key],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def unregister_auto_callable(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        key = f"{str(module_name or '').strip()}:{str(callable_name or '').strip()}"
        return dict(
            self.host.toolbox_unregister_auto(
                toolbox_id=self.toolbox_id,
                tool_keys=[key],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def register_intrinsic_tools(
        self,
        intrinsic_tool_names: Sequence[str],
        *,
        include_guides: bool = False,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_register_intrinsics(
                toolbox_id=self.toolbox_id,
                intrinsic_tool_names=[str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
                include_guides=bool(include_guides),
                sandbox_profile=SandboxProfileSpec(sandbox_policy=dict(sandbox_policy or {})).to_dict() if sandbox_policy else None,
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def unregister_intrinsic_tools(
        self,
        intrinsic_tool_names: Sequence[str],
        *,
        include_guides: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_unregister_intrinsics(
                toolbox_id=self.toolbox_id,
                intrinsic_tool_names=[str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
                include_guides=bool(include_guides),
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def describe(self, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_describe(
                toolbox_id=self.toolbox_id,
                timeout_seconds=float(timeout_seconds or 10.0),
            )
            or {}
        )

    def execute(
        self,
        *,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_execute(
                toolbox_id=self.toolbox_id,
                tool_call={
                    "name": str(tool_name or "").strip(),
                    "arguments": dict(arguments or {}),
                },
                timeout_seconds=float(timeout_seconds or 30.0),
            )
            or {}
        )


def load_toolbox_from_manifest(manifest_path: Path) -> tuple[Toolbox, Dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("toolbox_manifest_invalid")
    bundle_root = manifest_file.parent
    files_root = (bundle_root / "files").resolve()
    if str(files_root) not in sys.path:
        sys.path.insert(0, str(files_root))
    intrinsic_tool_names = [
        str(item or "").strip()
        for item in list(manifest.get("intrinsic_tool_names") or [])
        if str(item or "").strip()
    ]
    toolbox = Toolbox()
    if intrinsic_tool_names:
        ok, msg = toolbox.add_tool_callable(
            intrinsic_tool_names,
            is_intrinsic=True,
            include_guides=bool(manifest.get("with_intrinsic_guides", False)),
            activate=True,
        )
        if not ok:
            raise ValueError(str(msg or "intrinsic_registration_failed"))
        active_intrinsic = [
            str(item or "").strip()
            for item in list(manifest.get("active_intrinsic_tool_names") or [])
            if str(item or "").strip()
        ]
        hidden_intrinsic = [
            str(item or "").strip()
            for item in list(manifest.get("hidden_intrinsic_tool_names") or [])
            if str(item or "").strip()
        ]
        if active_intrinsic:
            toolbox.active_intrinsic_tool_names = [
                name for name in active_intrinsic if name in toolbox.intrinsic_tools
            ]
        if hidden_intrinsic:
            toolbox.hidden_intrinsic_tool_names = [
                name for name in hidden_intrinsic if name in toolbox.intrinsic_tools
            ]
    for item in list(manifest.get("auto_tools") or []):
        auto_meta = dict(item or {})
        module_name = str(auto_meta.get("module_name") or "").strip()
        callable_name = str(auto_meta.get("callable_name") or "").strip()
        if not module_name:
            raise ValueError("auto_tool_module_name_required")
        if not callable_name:
            raise ValueError("auto_tool_callable_name_required")
        module = importlib.import_module(module_name)
        ok, msg = toolbox.add_tool_callable(
            callable_name,
            search_scope=dict(vars(module)),
            activate=bool(auto_meta.get("activate", True)),
            guide_content=dict(auto_meta.get("guide_content") or {}) or None,
            guide_description=str(auto_meta.get("guide_description") or "").strip() or None,
        )
        if not ok:
            raise ValueError(str(msg or "auto_tool_registration_failed"))
    for item in list(manifest.get("tools") or []):
        tool_meta = dict(item or {})
        entrypoint = str(tool_meta.get("entrypoint") or "").strip()
        if ":" not in entrypoint:
            raise ValueError(f"tool_entrypoint_invalid:{entrypoint}")
        module_name, attr_name = entrypoint.split(":", 1)
        module = importlib.import_module(module_name)
        implementation = getattr(module, attr_name)
        ok, msg = toolbox.add_tool_external(
            tool_definition=dict(tool_meta.get("definition") or {}),
            implementation=implementation,
            activate=True,
            allow_override=False,
        )
        if not ok:
            raise ValueError(str(msg or "tool_registration_failed"))
    return toolbox, manifest
