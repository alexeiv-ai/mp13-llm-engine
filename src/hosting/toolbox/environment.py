"""Toolbox environment description and installation management."""
from __future__ import annotations

import json
import os
import subprocess
import time
import venv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .._process_utils import hidden_subprocess_kwargs
from .common import _sha256_text, _stable_json
from .bundle_models import SandboxProfileSpec, ToolboxEnvironmentSpec


class ToolboxEnvironmentManager:
    def __init__(self, hosting_root: Path):
        self.hosting_root = Path(hosting_root).expanduser().resolve()
        self.environments_root = (self.hosting_root / "toolbox_venvs").resolve()

    @staticmethod
    def normalize_environment_description(
        payload: Optional[Dict[str, Any]],
        *,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        row = dict(payload or {})
        env_name = str(name or row.get("name") or "base").strip() or "base"
        base_env_name = str(row.get("base_env_name") or ("base" if env_name != "base" else "")).strip()
        extra_packages: List[str] = []
        seen: set[str] = set()
        for item in list(row.get("extra_packages") or []):
            pkg = str(item or "").strip()
            if pkg and pkg not in seen:
                seen.add(pkg)
                extra_packages.append(pkg)
        return {
            "name": env_name,
            "base_env_name": base_env_name or None,
            "extra_packages": extra_packages,
            "allow_online_install": bool(row.get("allow_online_install", False)),
        }

    @classmethod
    def environment_description_hash(cls, payload: Optional[Dict[str, Any]], *, name: Optional[str] = None) -> str:
        normalized = cls.normalize_environment_description(payload, name=name)
        return cls._fingerprint_payload(normalized)[:16]

    @classmethod
    def resolve_environment_description(
        cls,
        payload_by_name: Dict[str, Dict[str, Any]],
        *,
        name: str,
    ) -> Dict[str, Any]:
        env_name = str(name or "base").strip() or "base"
        seen_stack: set[str] = set()
        lineage: List[str] = []
        merged_packages: List[str] = []
        merged_seen: set[str] = set()
        allow_online_install = False

        current = env_name
        while current:
            normalized = cls.normalize_environment_description(
                dict(payload_by_name.get(current) or {}),
                name=current,
            )
            if current in seen_stack:
                raise ValueError(f"environment description cycle detected at '{current}'")
            seen_stack.add(current)
            lineage.append(current)
            for item in list(normalized.get("extra_packages") or []):
                pkg = str(item or "").strip()
                if pkg and pkg not in merged_seen:
                    merged_seen.add(pkg)
                    merged_packages.append(pkg)
            allow_online_install = bool(allow_online_install or normalized.get("allow_online_install", False))
            base_env_name = str(normalized.get("base_env_name") or "").strip()
            current = base_env_name if base_env_name and base_env_name != normalized["name"] else ""

        direct = cls.normalize_environment_description(dict(payload_by_name.get(env_name) or {}), name=env_name)
        return {
            "name": env_name,
            "base_env_name": direct.get("base_env_name"),
            "extra_packages": list(direct.get("extra_packages") or []),
            "allow_online_install": bool(direct.get("allow_online_install", False)),
            "effective_extra_packages": merged_packages,
            "effective_allow_online_install": allow_online_install,
            "lineage": lineage,
        }

    @staticmethod
    def _fingerprint_payload(payload: Dict[str, Any]) -> str:
        return _sha256_text(_stable_json(payload))

    def environment_spec_for_bundle(
        self,
        staged: "StagedToolboxBundle",
        *,
        environment_description: Optional[Dict[str, Any]] = None,
    ) -> ToolboxEnvironmentSpec:
        manifest = dict(staged.manifest or {})
        sandbox_profile = SandboxProfileSpec.from_dict(dict(manifest.get("sandbox_profile") or {}))
        intrinsic_tool_names = list(manifest.get("intrinsic_tool_names") or [])
        intrinsics_profile_id = sandbox_profile.intrinsics_profile_id(intrinsic_tool_names)
        dependency_lock_hash = str(manifest.get("dependency_lock_hash") or "").strip() or None
        required_imports = sandbox_profile.normalized_required_imports()
        toolbox_runtime_hash = "toolbox-executor-v1"
        environment_name = str(sandbox_profile.environment_name or "base").strip() or "base"
        input_desc = dict(environment_description or {})
        raw_desc = self.normalize_environment_description(input_desc, name=environment_name)
        effective_extra_packages = [
            str(item or "").strip()
            for item in list(input_desc.get("effective_extra_packages") or raw_desc.get("extra_packages") or [])
            if str(item or "").strip()
        ]
        env_desc = {
            "name": environment_name,
            "base_env_name": raw_desc.get("base_env_name"),
            "extra_packages": effective_extra_packages,
            "allow_online_install": bool(
                input_desc.get("effective_allow_online_install", raw_desc.get("allow_online_install", False))
            ),
        }
        env_desc_hash = self.environment_description_hash(env_desc, name=environment_name)
        venv_key = self._fingerprint_payload(
            {
                "toolbox_runtime_hash": toolbox_runtime_hash,
                "environment_name": environment_name,
                "environment_description_hash": env_desc_hash,
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
            environment_name=environment_name,
            environment_description_hash=env_desc_hash,
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
        metadata = self.read_environment_metadata(spec) if metadata_path.exists() else {}
        metadata.update(spec.to_dict())
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return spec

    def runtime_python_executable(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        fallback_python_executable: Optional[str] = None,
    ) -> str:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        env_python = str(ensured.python_executable or self.python_executable_path(env_root)).strip()
        fallback_python = str(fallback_python_executable or "").strip()
        if not fallback_python:
            return env_python
        metadata = self.read_environment_metadata(ensured)
        install_execution_status = str(dict(metadata.get("install_execution") or {}).get("status") or "").strip().lower()
        receipt_verification_status = str(
            dict(metadata.get("install_receipt_verification") or {}).get("status") or ""
        ).strip().lower()
        if install_execution_status == "ok" and receipt_verification_status == "ok":
            return env_python
        return fallback_python

    @staticmethod
    def _unique_names(items: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(items or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @staticmethod
    def _normalize_package_name(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in raw:
                raw = raw.split(sep, 1)[0]
                break
        raw = raw.strip()
        return raw.lower()

    @classmethod
    def _install_plan_hash(cls, install_plan: Dict[str, Any]) -> str:
        payload = {
            "planned_packages": cls._unique_names(install_plan.get("planned_packages") or []),
            "requirements_relpath": str(install_plan.get("requirements_relpath") or "").strip() or "requirements-planned.txt",
        }
        return cls._fingerprint_payload(payload)[:16]

    @classmethod
    def _resolved_install_lock_hash(
        cls,
        spec: ToolboxEnvironmentSpec,
        *,
        resolved_packages: Sequence[Any],
        source_install_plan_hash: str,
        requirements_relpath: str = "requirements-resolved.txt",
    ) -> str:
        payload = {
            "venv_key": spec.venv_key,
            "environment_name": spec.environment_name,
            "environment_description_hash": spec.environment_description_hash,
            "resolved_packages": cls._unique_names(resolved_packages or []),
            "source_install_plan_hash": str(source_install_plan_hash or "").strip() or None,
            "requirements_relpath": str(requirements_relpath or "").strip() or "requirements-resolved.txt",
            "toolbox_runtime_hash": spec.toolbox_runtime_hash,
            "intrinsics_profile_id": spec.intrinsics_profile_id,
            "dependency_lock_hash": spec.dependency_lock_hash,
            "venv_lock_hash": spec.venv_lock_hash,
        }
        return cls._fingerprint_payload(payload)[:16]

    @classmethod
    def _resolved_packages_from_report(cls, report: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(dict(report or {}).get("install") or []):
            row = dict(item or {})
            metadata = dict(row.get("metadata") or {})
            name = str(metadata.get("name") or "").strip()
            version = str(metadata.get("version") or "").strip()
            if not name:
                continue
            pinned = f"{name}=={version}" if version else name
            key = pinned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(pinned)
        return out

    def read_environment_metadata(self, spec: ToolboxEnvironmentSpec) -> Dict[str, Any]:
        metadata_path = Path(spec.venv_path).expanduser().resolve() / "environment.json"
        if not metadata_path.exists():
            return dict(spec.to_dict())
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return dict(payload or {}) if isinstance(payload, dict) else dict(spec.to_dict())
        except Exception:
            return dict(spec.to_dict())

    def realize_environment(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        environment_description: Optional[Dict[str, Any]] = None,
        required_packages: Optional[Sequence[str]] = None,
        missing_packages: Optional[Sequence[str]] = None,
        toolbox_id: Optional[str] = None,
        sandbox_profile_id: Optional[str] = None,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        effective_desc_input = dict(environment_description or {})
        effective_desc = {
            "name": str(effective_desc_input.get("name") or ensured.environment_name or "base").strip() or "base",
            "base_env_name": effective_desc_input.get("base_env_name"),
            "effective_extra_packages": self._unique_names(
                effective_desc_input.get("effective_extra_packages")
                or effective_desc_input.get("extra_packages")
                or []
            ),
            "effective_allow_online_install": bool(
                effective_desc_input.get(
                    "effective_allow_online_install",
                    effective_desc_input.get("allow_online_install", False),
                )
            ),
            "lineage": [str(item or "").strip() for item in list(effective_desc_input.get("lineage") or []) if str(item or "").strip()],
        }
        required = self._unique_names(required_packages or ensured.required_imports)
        missing = self._unique_names(missing_packages or [])
        planned = self._unique_names(list(effective_desc["effective_extra_packages"]) + list(required))
        provenance_payload = {
            "toolbox_id": str(toolbox_id or "").strip() or None,
            "sandbox_profile_id": str(sandbox_profile_id or "").strip() or None,
            "venv_key": ensured.venv_key,
            "environment_name": ensured.environment_name,
            "environment_description_hash": ensured.environment_description_hash,
            "required_packages": required,
            "effective_extra_packages": list(effective_desc["effective_extra_packages"]),
            "planned_packages": planned,
            "missing_packages": missing,
            "allow_online_install": bool(effective_desc["effective_allow_online_install"]),
            "tool_keys": self._unique_names(tool_keys or []),
            "dependency_lock_hash": ensured.dependency_lock_hash,
            "venv_lock_hash": ensured.venv_lock_hash,
            "toolbox_runtime_hash": ensured.toolbox_runtime_hash,
            "intrinsics_profile_id": ensured.intrinsics_profile_id,
        }
        provenance_hash = self._fingerprint_payload(provenance_payload)[:16]
        realization = {
            "mode": "metadata_only",
            "status": "planned",
            "provenance_hash": provenance_hash,
            "realized_at": time.time(),
            "required_packages": required,
            "effective_extra_packages": list(effective_desc["effective_extra_packages"]),
            "planned_packages": planned,
            "missing_packages": missing,
            "allow_online_install": bool(effective_desc["effective_allow_online_install"]),
            "environment_lineage": list(effective_desc["lineage"]),
        }
        metadata = self.read_environment_metadata(ensured)
        metadata.update(ensured.to_dict())
        metadata["realization"] = realization
        metadata_path = Path(ensured.venv_path).expanduser().resolve() / "environment.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def prepare_install_plan(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        environment_description: Optional[Dict[str, Any]] = None,
        required_packages: Optional[Sequence[str]] = None,
        missing_packages: Optional[Sequence[str]] = None,
        toolbox_id: Optional[str] = None,
        sandbox_profile_id: Optional[str] = None,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        metadata = self.realize_environment(
            spec,
            environment_description=environment_description,
            required_packages=required_packages,
            missing_packages=missing_packages,
            toolbox_id=toolbox_id,
            sandbox_profile_id=sandbox_profile_id,
            tool_keys=tool_keys,
        )
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        realization = dict(metadata.get("realization") or {})
        planned_packages = self._unique_names(realization.get("planned_packages") or [])
        requirements_relpath = "requirements-planned.txt"
        requirements_path = env_root / requirements_relpath
        requirements_body = "".join(f"{pkg}\n" for pkg in planned_packages)
        requirements_path.write_text(requirements_body, encoding="utf-8")
        install_command = [
            str(ensured.python_executable or self.python_executable_path(env_root)),
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
        install_plan = {
            "mode": "plan_only",
            "requirements_path": str(requirements_path),
            "requirements_relpath": requirements_relpath,
            "planned_packages": planned_packages,
            "missing_packages": self._unique_names(realization.get("missing_packages") or []),
            "can_execute_online": bool(realization.get("allow_online_install", False)),
            "install_command": install_command,
            "generated_at": time.time(),
        }
        metadata["install_plan"] = install_plan
        metadata_path = env_root / "environment.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def lock_install_plan(self, spec: ToolboxEnvironmentSpec) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.read_environment_metadata(ensured)
        install_plan = dict(metadata.get("install_plan") or {})
        if not install_plan:
            raise ValueError("install_plan_missing")
        planned_packages = self._unique_names(install_plan.get("planned_packages") or [])
        requirements_relpath = "requirements-locked.txt"
        requirements_path = env_root / requirements_relpath
        requirements_body = "".join(f"{pkg}\n" for pkg in planned_packages)
        requirements_path.write_text(requirements_body, encoding="utf-8")
        lock_payload = {
            "venv_key": ensured.venv_key,
            "environment_name": ensured.environment_name,
            "environment_description_hash": ensured.environment_description_hash,
            "planned_packages": planned_packages,
            "requirements_relpath": requirements_relpath,
            "toolbox_runtime_hash": ensured.toolbox_runtime_hash,
            "intrinsics_profile_id": ensured.intrinsics_profile_id,
            "dependency_lock_hash": ensured.dependency_lock_hash,
            "venv_lock_hash": ensured.venv_lock_hash,
        }
        install_lock_hash = self._fingerprint_payload(lock_payload)[:16]
        locked_plan = {
            "status": "locked",
            "locked_at": time.time(),
            "install_lock_hash": install_lock_hash,
            "planned_packages": planned_packages,
            "requirements_path": str(requirements_path),
            "requirements_relpath": requirements_relpath,
        }
        metadata["install_lock"] = locked_plan
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def resolve_install_lock(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        allow_resolution: bool = False,
    ) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.read_environment_metadata(ensured)
        install_plan = dict(metadata.get("install_plan") or {})
        if not install_plan:
            raise ValueError("install_plan_missing")
        install_lock = dict(metadata.get("install_lock") or {})
        if not install_lock:
            resolution = {
                "status": "blocked",
                "resolved_at": time.time(),
                "reason": "install_lock_required",
            }
            metadata["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        verification_meta = self.verify_install_lock(ensured)
        verification = dict(verification_meta.get("install_lock_verification") or {})
        if str(verification.get("status") or "").strip().lower() != "ok":
            resolution = {
                "status": "blocked",
                "resolved_at": time.time(),
                "reason": str(verification.get("reason") or "install_lock_invalid"),
                "install_lock_hash": str(verification.get("install_lock_hash") or "").strip() or None,
                "expected_install_lock_hash": str(verification.get("expected_install_lock_hash") or "").strip() or None,
            }
            verification_meta["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(verification_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return verification_meta
        metadata = verification_meta
        install_plan = dict(metadata.get("install_plan") or {})
        planned_packages = self._unique_names(install_plan.get("planned_packages") or [])
        if not planned_packages:
            resolution = {
                "status": "noop",
                "resolved_at": time.time(),
                "reason": "no_planned_packages",
            }
            metadata["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not allow_resolution:
            resolution = {
                "status": "blocked",
                "resolved_at": time.time(),
                "reason": "resolution_not_enabled",
            }
            metadata["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not bool(install_plan.get("can_execute_online", False)):
            resolution = {
                "status": "blocked",
                "resolved_at": time.time(),
                "reason": "online_resolution_not_allowed",
            }
            metadata["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        requirements_path = Path(str(install_plan.get("requirements_path") or "")).expanduser().resolve()
        if not requirements_path.exists():
            raise ValueError("install_plan_requirements_missing")
        report_relpath = "install-resolution-report.json"
        report_path = env_root / report_relpath
        command = [
            str(ensured.python_executable or self.python_executable_path(env_root)),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--report",
            str(report_path),
            "-r",
            str(requirements_path),
        ]
        result = subprocess.run(  # noqa: S603
            command,
            cwd=str(env_root),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            **hidden_subprocess_kwargs(),
        )
        resolution = {
            "status": "ok" if int(result.returncode or 0) == 0 else "failed",
            "resolved_at": time.time(),
            "returncode": int(result.returncode or 0),
            "stdout": str(result.stdout or ""),
            "stderr": str(result.stderr or ""),
            "command": command,
            "report_path": str(report_path),
            "report_relpath": report_relpath,
            "source_install_plan_hash": self._install_plan_hash(install_plan),
        }
        metadata["install_resolution"] = resolution
        if resolution["status"] == "ok" and report_path.exists():
            report_text = report_path.read_text(encoding="utf-8")
            report = json.loads(report_text)
            report_hash = _sha256_text(report_text)
            resolved_packages = self._resolved_packages_from_report(dict(report or {}))
            resolved_relpath = "requirements-resolved.txt"
            resolved_path = env_root / resolved_relpath
            resolved_path.write_text("".join(f"{pkg}\n" for pkg in resolved_packages), encoding="utf-8")
            resolved_lock_payload = {
                "venv_key": ensured.venv_key,
                "environment_name": ensured.environment_name,
                "environment_description_hash": ensured.environment_description_hash,
                "resolved_packages": resolved_packages,
                "source_install_plan_hash": resolution["source_install_plan_hash"],
                "requirements_relpath": resolved_relpath,
                "toolbox_runtime_hash": ensured.toolbox_runtime_hash,
                "intrinsics_profile_id": ensured.intrinsics_profile_id,
                "dependency_lock_hash": ensured.dependency_lock_hash,
                "venv_lock_hash": ensured.venv_lock_hash,
            }
            resolved_lock_hash = self._fingerprint_payload(resolved_lock_payload)[:16]
            metadata["resolved_install_lock"] = {
                "status": "locked",
                "locked_at": time.time(),
                "resolved_lock_hash": resolved_lock_hash,
                "resolved_packages": resolved_packages,
                "requirements_path": str(resolved_path),
                "requirements_relpath": resolved_relpath,
                "report_path": str(report_path),
                "report_relpath": report_relpath,
                "report_sha256": report_hash,
                "source_install_plan_hash": resolution["source_install_plan_hash"],
            }
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def verify_install_lock(self, spec: ToolboxEnvironmentSpec) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.read_environment_metadata(ensured)
        install_plan = dict(metadata.get("install_plan") or {})
        install_lock = dict(metadata.get("install_lock") or {})
        if not install_plan:
            raise ValueError("install_plan_missing")
        if not install_lock:
            verification = {
                "status": "missing",
                "verified_at": time.time(),
                "reason": "install_lock_missing",
            }
            metadata["install_lock_verification"] = verification
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        planned_packages = self._unique_names(install_plan.get("planned_packages") or [])
        expected_requirements_relpath = "requirements-locked.txt"
        expected_payload = {
            "venv_key": ensured.venv_key,
            "environment_name": ensured.environment_name,
            "environment_description_hash": ensured.environment_description_hash,
            "planned_packages": planned_packages,
            "requirements_relpath": expected_requirements_relpath,
            "toolbox_runtime_hash": ensured.toolbox_runtime_hash,
            "intrinsics_profile_id": ensured.intrinsics_profile_id,
            "dependency_lock_hash": ensured.dependency_lock_hash,
            "venv_lock_hash": ensured.venv_lock_hash,
        }
        expected_lock_hash = self._fingerprint_payload(expected_payload)[:16]
        lock_hash = str(install_lock.get("install_lock_hash") or "").strip()
        requirements_path = Path(
            str(install_lock.get("requirements_path") or (env_root / expected_requirements_relpath))
        ).expanduser().resolve()
        requirements_ok = requirements_path.exists()
        status = "ok"
        reason = None
        if not requirements_ok:
            status = "stale"
            reason = "locked_requirements_missing"
        elif lock_hash != expected_lock_hash:
            status = "stale"
            reason = "install_lock_hash_mismatch"
        resolved_install_lock = dict(metadata.get("resolved_install_lock") or {})
        resolved_lock_hash = str(resolved_install_lock.get("resolved_lock_hash") or "").strip()
        expected_resolved_lock_hash = None
        resolved_requirements_path = None
        resolved_report_path = None
        resolved_report_hash = str(resolved_install_lock.get("report_sha256") or "").strip()
        expected_resolved_report_hash = None
        resolved_reason = None
        resolved_status = "missing"
        if resolved_install_lock:
            expected_plan_hash = self._install_plan_hash(install_plan)
            source_plan_hash = str(resolved_install_lock.get("source_install_plan_hash") or "").strip()
            expected_resolved_relpath = (
                str(resolved_install_lock.get("requirements_relpath") or "").strip() or "requirements-resolved.txt"
            )
            expected_resolved_lock_hash = self._resolved_install_lock_hash(
                ensured,
                resolved_packages=resolved_install_lock.get("resolved_packages") or [],
                source_install_plan_hash=expected_plan_hash,
                requirements_relpath=expected_resolved_relpath,
            )
            resolved_requirements_path = Path(
                str(resolved_install_lock.get("requirements_path") or (env_root / expected_resolved_relpath))
            ).expanduser().resolve()
            expected_report_relpath = (
                str(resolved_install_lock.get("report_relpath") or "").strip() or "install-resolution-report.json"
            )
            resolved_report_path = Path(
                str(resolved_install_lock.get("report_path") or (env_root / expected_report_relpath))
            ).expanduser().resolve()
            resolved_status = "ok"
            if source_plan_hash != expected_plan_hash:
                resolved_status = "stale"
                resolved_reason = "resolved_lock_plan_hash_mismatch"
            elif not resolved_requirements_path.exists():
                resolved_status = "stale"
                resolved_reason = "resolved_lock_requirements_missing"
            elif not resolved_report_path.exists():
                resolved_status = "stale"
                resolved_reason = "resolved_lock_report_missing"
            else:
                expected_resolved_report_hash = _sha256_text(resolved_report_path.read_text(encoding="utf-8"))
                if resolved_report_hash and resolved_report_hash != expected_resolved_report_hash:
                    resolved_status = "stale"
                    resolved_reason = "resolved_lock_report_hash_mismatch"
            if resolved_status == "ok" and resolved_lock_hash != expected_resolved_lock_hash:
                resolved_status = "stale"
                resolved_reason = "resolved_lock_hash_mismatch"
            if resolved_status != "ok":
                status = "stale"
                reason = resolved_reason
        verification = {
            "status": status,
            "verified_at": time.time(),
            "install_lock_hash": lock_hash or None,
            "expected_install_lock_hash": expected_lock_hash,
            "requirements_path": str(requirements_path),
            "reason": reason,
            "resolved_lock_status": resolved_status,
            "resolved_lock_hash": resolved_lock_hash or None,
            "expected_resolved_lock_hash": expected_resolved_lock_hash,
            "resolved_requirements_path": str(resolved_requirements_path) if resolved_requirements_path else None,
            "resolved_report_path": str(resolved_report_path) if resolved_report_path else None,
            "resolved_report_sha256": resolved_report_hash or None,
            "expected_resolved_report_sha256": expected_resolved_report_hash,
            "resolved_reason": resolved_reason,
        }
        metadata["install_lock_verification"] = verification
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def execute_install_plan(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        allow_execution: bool = False,
    ) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.read_environment_metadata(ensured)
        install_plan = dict(metadata.get("install_plan") or {})
        if not install_plan:
            raise ValueError("install_plan_missing")
        planned_packages = self._unique_names(install_plan.get("planned_packages") or [])
        if not planned_packages:
            execution = {
                "status": "noop",
                "executed": False,
                "executed_at": time.time(),
                "reason": "no_planned_packages",
            }
            metadata["install_execution"] = execution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not allow_execution:
            execution = {
                "status": "blocked",
                "executed": False,
                "executed_at": time.time(),
                "reason": "execution_not_enabled",
            }
            metadata["install_execution"] = execution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not bool(install_plan.get("can_execute_online", False)):
            execution = {
                "status": "blocked",
                "executed": False,
                "executed_at": time.time(),
                "reason": "online_install_not_allowed",
            }
            metadata["install_execution"] = execution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        verification_meta = self.verify_install_lock(ensured)
        verification = dict(verification_meta.get("install_lock_verification") or {})
        if str(verification.get("status") or "") != "ok":
            verification_reason = str(verification.get("reason") or "").strip()
            execution = {
                "status": "blocked",
                "executed": False,
                "executed_at": time.time(),
                "reason": (
                    "install_lock_required"
                    if verification_reason in {"", "install_lock_missing"}
                    else verification_reason
                ),
                "install_lock_hash": str(verification.get("install_lock_hash") or "").strip() or None,
                "expected_install_lock_hash": str(verification.get("expected_install_lock_hash") or "").strip() or None,
            }
            metadata = self.read_environment_metadata(ensured)
            metadata["install_execution"] = execution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        metadata = verification_meta
        install_lock = dict(metadata.get("install_lock") or {})
        resolved_install_lock = dict(metadata.get("resolved_install_lock") or {})
        requirements_path = str(
            install_lock.get("requirements_path")
            or install_plan.get("requirements_path")
            or ""
        ).strip()
        command = [str(item or "").strip() for item in list(install_plan.get("install_command") or []) if str(item or "").strip()]
        if not command:
            raise ValueError("install_command_missing")
        resolved_lock_hash = None
        if resolved_install_lock:
            expected_plan_hash = self._install_plan_hash(install_plan)
            source_plan_hash = str(resolved_install_lock.get("source_install_plan_hash") or "").strip()
            resolved_requirements_path = Path(
                str(resolved_install_lock.get("requirements_path") or "")
            ).expanduser().resolve()
            if source_plan_hash != expected_plan_hash:
                execution = {
                    "status": "blocked",
                    "executed": False,
                    "executed_at": time.time(),
                    "reason": "resolved_lock_plan_hash_mismatch",
                    "resolved_lock_hash": str(resolved_install_lock.get("resolved_lock_hash") or "").strip() or None,
                    "source_install_plan_hash": source_plan_hash or None,
                    "expected_install_plan_hash": expected_plan_hash,
                }
                metadata["install_execution"] = execution
                (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                return metadata
            if resolved_requirements_path.exists():
                requirements_path = str(resolved_requirements_path)
                resolved_lock_hash = str(resolved_install_lock.get("resolved_lock_hash") or "").strip() or None
        if requirements_path:
            command = command[:-1] + [requirements_path]
        result = subprocess.run(  # noqa: S603
            command,
            cwd=str(env_root),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            **hidden_subprocess_kwargs(),
        )
        execution = {
            "status": "ok" if int(result.returncode or 0) == 0 else "failed",
            "executed": True,
            "executed_at": time.time(),
            "returncode": int(result.returncode or 0),
            "stdout": str(result.stdout or ""),
            "stderr": str(result.stderr or ""),
            "command": command,
            "install_lock_hash": str(install_lock.get("install_lock_hash") or "").strip() or None,
            "resolved_lock_hash": resolved_lock_hash,
        }
        metadata["install_execution"] = execution
        if execution["status"] == "ok":
            freeze_cmd = [
                str(ensured.python_executable or self.python_executable_path(env_root)),
                "-m",
                "pip",
                "freeze",
            ]
            try:
                freeze_result = subprocess.run(  # noqa: S603
                    freeze_cmd,
                    cwd=str(env_root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    **hidden_subprocess_kwargs(),
                )
                freeze_output = str(freeze_result.stdout or "")
                lines = [
                    line.strip()
                    for line in freeze_output.splitlines()
                    if str(line or "").strip()
                ]
                receipt_payload = {
                    "status": "ok" if int(freeze_result.returncode or 0) == 0 else "failed",
                    "captured_at": time.time(),
                    "returncode": int(freeze_result.returncode or 0),
                    "command": freeze_cmd,
                    "packages": lines,
                    "packages_hash": self._fingerprint_payload({"packages": lines})[:16],
                    "stderr": str(freeze_result.stderr or ""),
                }
            except Exception as exc:
                receipt_payload = {
                    "status": "failed",
                    "captured_at": time.time(),
                    "command": freeze_cmd,
                    "packages": [],
                    "packages_hash": None,
                    "stderr": str(exc),
                }
            metadata["install_receipt"] = receipt_payload
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            metadata = self.verify_install_receipt(ensured)
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def verify_install_receipt(self, spec: ToolboxEnvironmentSpec) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.verify_install_lock(ensured)
        install_lock = dict(metadata.get("install_lock") or {})
        resolved_install_lock = dict(metadata.get("resolved_install_lock") or {})
        install_receipt = dict(metadata.get("install_receipt") or {})
        lock_verification = dict(metadata.get("install_lock_verification") or {})
        if not install_lock and not resolved_install_lock:
            verification = {
                "status": "missing",
                "verified_at": time.time(),
                "reason": "install_lock_missing",
            }
            metadata["install_receipt_verification"] = verification
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not install_receipt:
            verification = {
                "status": "missing",
                "verified_at": time.time(),
                "reason": "install_receipt_missing",
            }
            metadata["install_receipt_verification"] = verification
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if str(lock_verification.get("status") or "").strip() not in {"ok", "missing"}:
            verification = {
                "status": "stale",
                "verified_at": time.time(),
                "reason": str(lock_verification.get("reason") or "install_lock_invalid"),
                "lock_verification_status": str(lock_verification.get("status") or "").strip() or None,
                "lock_source": "resolved_install_lock" if resolved_install_lock else "install_lock",
            }
            metadata["install_receipt_verification"] = verification
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        locked_source = list(resolved_install_lock.get("resolved_packages") or []) or list(install_lock.get("planned_packages") or [])
        locked_names = {
            self._normalize_package_name(item)
            for item in locked_source
            if self._normalize_package_name(item)
        }
        observed_names = {
            self._normalize_package_name(item)
            for item in list(install_receipt.get("packages") or [])
            if self._normalize_package_name(item)
        }
        missing = sorted(name for name in locked_names if name not in observed_names)
        status = "ok" if not missing else "mismatch"
        verification = {
            "status": status,
            "verified_at": time.time(),
            "locked_package_names": sorted(locked_names),
            "observed_package_names": sorted(observed_names),
            "missing_package_names": missing,
            "lock_source": "resolved_install_lock" if resolved_install_lock else "install_lock",
        }
        metadata["install_receipt_verification"] = verification
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def ensure_for_bundle(
        self,
        staged: "StagedToolboxBundle",
        *,
        environment_description: Optional[Dict[str, Any]] = None,
    ) -> ToolboxEnvironmentSpec:
        return self.ensure_environment(self.environment_spec_for_bundle(staged, environment_description=environment_description))
