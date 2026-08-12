"""Toolbox bundle staging and worker startup helpers."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bundle_models import ToolboxBundleSpec, ToolboxWorkerStartupSpec
from .environment import EnvironmentRuntimeAdapter, ToolboxEnvironmentSpec


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
            spec = EnvironmentRuntimeAdapter(self.bundle_root.parents[2]).environment_spec_for_bundle(self)
        return {
            "venv_key": spec.venv_key,
            "venv_path": spec.venv_path,
            "python_executable": spec.python_executable,
            "environment_name": spec.environment_name,
            "environment_description_hash": spec.environment_description_hash,
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
        hidden_tool_names = {
            str(item or "").strip()
            for item in list(self.manifest.get("hidden_tool_names") or [])
            if str(item or "").strip()
        }
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
        hidden_allowed_tool_names = [name for name in allowed_tool_names if name in hidden_tool_names or name in hidden_intrinsic_names]
        advertised_tool_names = [name for name in allowed_tool_names if name not in set(hidden_allowed_tool_names)]
        sandbox_profile_id = str(dict(self.manifest.get("sandbox_profile") or {}).get("profile_id") or "default")
        return {
            "allowed_tool_names": allowed_tool_names,
            "advertised_tool_names": advertised_tool_names,
            "hidden_allowed_tool_names": hidden_allowed_tool_names,
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
        ipc_family: Optional[str] = None,
        ipc_address: str = "",
        policy: Optional[Dict[str, Any]] = None,
    ) -> ToolboxWorkerStartupSpec:
        scratch = Path(scratch_root or (self.bundle_root / "scratch")).expanduser().resolve()
        default_ipc_family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        return ToolboxWorkerStartupSpec(
            worker_id=str(worker_id or "").strip(),
            sandbox_id=str(sandbox_id or worker_id or "").strip(),
            toolbox_revision=str(self.manifest.get("bundle_revision") or "").strip(),
            manifest_path=str(self.manifest_path),
            scratch_root=str(scratch),
            engines_state_file=str(Path(engines_state_file).expanduser().resolve()) if engines_state_file else None,
            control_state_file=str(Path(control_state_file).expanduser().resolve()) if control_state_file else None,
            venv_path=str(venv_path or "").strip() or None,
            ipc_family=str(ipc_family or default_ipc_family).strip() or default_ipc_family,
            ipc_address=str(ipc_address or "").strip(),
            policy=dict(policy or {}),
        )

    def worker_env(self, *, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        src_root = str(Path(__file__).resolve().parents[2])
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
        ipc_family: Optional[str] = None,
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
