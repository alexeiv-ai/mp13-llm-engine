"""Workflow-facing Python runtime environment helpers.

This wraps the existing toolbox environment manager with workflow/runtime
terminology so new workflow APIs do not need to expose toolbox IDs or tool keys.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..model_runtime_contract import reject_model_runtime_selection
from ..toolbox.environment import WorkflowEnvironmentAdapter
from ..toolbox.bundle_models import ToolboxEnvironmentSpec
from .process_base import HostedProcessSandboxBase
from .runtime_base import HostedEnvironmentKeySpec, HostedRuntimeIdentity, stable_hash


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _pins(policy: Optional[Dict[str, Any]]) -> Dict[str, str]:
    return {
        _clean(key): _clean(value)
        for key, value in dict(dict(policy or {}).get("package_pins") or {}).items()
        if _clean(key) and _clean(value)
    }


def _imports(policy: Optional[Dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(dict(policy or {}).get("import_allowlist") or []):
        name = _clean(item)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _pinned_packages(policy: Optional[Dict[str, Any]]) -> list[str]:
    return [f"{name}=={version}" for name, version in sorted(_pins(policy).items())]


def _uv_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = dict(policy or {})
    uv = dict(row.get("uv") or {}) if isinstance(row.get("uv"), dict) else {}
    pyproject = _clean(uv.get("pyproject_toml") or row.get("pyproject_toml"))
    uv_lock = _clean(uv.get("uv_lock") or row.get("uv_lock"))
    groups = []
    seen: set[str] = set()
    for item in list(uv.get("dependency_groups") or row.get("dependency_groups") or []):
        value = _clean(item)
        if value and value not in seen:
            seen.add(value)
            groups.append(value)
    enabled = bool(uv or pyproject or uv_lock or groups or row.get("uv_enabled"))
    return {
        "enabled": enabled,
        "uv_executable": _clean(uv.get("uv_executable") or row.get("uv_executable")) or "uv",
        "pyproject_toml": pyproject or None,
        "uv_lock": uv_lock or None,
        "dependency_groups": groups,
    }


def _uv_intent_hash(policy: Optional[Dict[str, Any]]) -> Optional[str]:
    uv = _uv_policy(policy)
    if not uv["enabled"]:
        return None
    explicit = _clean(dict(policy or {}).get("dependency_lock_hash"))
    return explicit or stable_hash({"uv": uv})


def _uv_availability(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    uv = _uv_policy(policy)
    executable = _clean(uv.get("uv_executable")) or "uv"
    path = shutil.which(executable)
    version = None
    if path:
        try:
            proc = subprocess.run(
                [path, "--version"],
                text=True,
                capture_output=True,
                timeout=5.0,
                check=False,
            )
            version = (proc.stdout or proc.stderr or "").strip() or None
        except Exception:
            version = None
    return {
        **uv,
        "available": bool(path),
        "resolved_executable": path,
        "version": version,
    }


def _uv_python_executable(env_root: Path) -> Path:
    if sys.platform == "win32":
        return env_root / ".venv" / "Scripts" / "python.exe"
    return env_root / ".venv" / "bin" / "python"


def _runtime_hash(default_hash: str, policy: Optional[Dict[str, Any]]) -> str:
    row = dict(policy or {})
    explicit = _clean(row.get("runtime_hash"))
    if explicit:
        return explicit
    executable = (
        _clean(row.get("python_executable"))
        or _clean(row.get("bootstrap_python_executable"))
        or _clean(row.get("fallback_python_executable"))
    )
    if executable:
        return f"{_clean(default_hash) or 'workflow-python-v1'}:{stable_hash({'python_executable': executable})[:16]}"
    return _clean(default_hash) or "workflow-python-v1"


class HostedPythonRuntimeBase(HostedProcessSandboxBase):
    """Internal Python runtime base above the language-neutral process base."""

    sandbox_kind = "workflow_python"

    def __init__(self, hosting_root: Path):
        super().__init__()
        self.hosting_root = Path(hosting_root).expanduser().resolve()
        self.environment_manager = WorkflowEnvironmentAdapter(self.hosting_root)

    def environment_key_spec(
        self,
        *,
        environment_name: str,
        profile: str,
        python_policy: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        runtime_hash: str = "workflow-python-v1",
        runtime_version: Optional[str] = None,
    ) -> HostedEnvironmentKeySpec:
        reject_model_runtime_selection(
            {
                "environment_name": environment_name,
                "profile": profile,
                "python": dict(python_policy or {}),
            }
        )
        runtime = HostedRuntimeIdentity(
            runtime_kind="workflow_python",
            profile=_clean(profile) or "helper",
            runtime_hash=_clean(runtime_hash) or "workflow-python-v1",
            runtime_version=_clean(runtime_version) or None,
        )
        return HostedEnvironmentKeySpec(
            environment_name=_clean(environment_name) or "workflow-python-helper",
            runtime=runtime,
            sandbox_policy=dict(sandbox_policy or {}),
            required_imports=_imports(python_policy),
            package_pins=_pins(python_policy),
            dependency_lock_hash=_uv_intent_hash(python_policy),
        )


class HostedPythonRuntimeManager(HostedPythonRuntimeBase):
    """Internal Python environment adapter for workflow/runtime APIs."""

    def environment_spec(
        self,
        *,
        environment_name: str = "workflow-python-helper",
        profile: str = "helper",
        python_policy: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        env_name = _clean(environment_name) or "workflow-python-helper"
        spec = self.environment_manager.workflow_python_helper_environment_spec(
            policy=dict(python_policy or {}),
            environment_name=env_name,
        )
        key_spec = self.environment_key_spec(
            environment_name=env_name,
            profile=profile,
            python_policy=python_policy,
            sandbox_policy=sandbox_policy,
            runtime_hash=_runtime_hash(spec.toolbox_runtime_hash, python_policy),
        )
        uv = _uv_availability(python_policy)
        env = spec.to_dict()
        env["uv"] = uv
        return {
            "status": "ok",
            "environment": env,
            "environment_key": key_spec.short_key(),
            "environment_key_full": key_spec.full_key(),
            "environment_identity": key_spec.to_dict(),
        }

    def _spec_from_request(
        self,
        *,
        environment_name: str,
        python_policy: Optional[Dict[str, Any]] = None,
    ) -> ToolboxEnvironmentSpec:
        policy = dict(python_policy or {})
        uv_dependency_hash = _uv_intent_hash(policy)
        if uv_dependency_hash and not _clean(policy.get("dependency_lock_hash")):
            policy["dependency_lock_hash"] = uv_dependency_hash
        return self.environment_manager.workflow_python_helper_environment_spec(
            policy=policy,
            environment_name=_clean(environment_name) or "workflow-python-helper",
        )

    def _environment_description(self, spec: ToolboxEnvironmentSpec, *, python_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        pinned = _pinned_packages(python_policy)
        uv = _uv_policy(python_policy)
        return {
            "name": spec.environment_name,
            "extra_packages": pinned,
            "effective_extra_packages": pinned,
            "allow_online_install": False,
            "effective_allow_online_install": False,
            "lineage": [spec.environment_name],
            "uv": uv,
        }

    def _install_status_summary(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        row = dict(metadata or {})
        plan = dict(row.get("install_plan") or {})
        lock = dict(row.get("install_lock") or {})
        lock_verification = dict(row.get("install_lock_verification") or {})
        execution = dict(row.get("install_execution") or {})
        receipt = dict(row.get("install_receipt") or {})
        receipt_verification = dict(row.get("install_receipt_verification") or {})
        resolved_lock = dict(row.get("resolved_install_lock") or {})
        uv_plan = dict(row.get("uv_install_plan") or {})
        uv_lock = dict(row.get("uv_install_lock") or {})
        uv_lock_verification = dict(row.get("uv_install_lock_verification") or {})
        uv_execution = dict(row.get("uv_install_execution") or {})
        uv_receipt = dict(row.get("uv_install_receipt") or {})
        uv_receipt_verification = dict(row.get("uv_install_receipt_verification") or {})
        return {
            "install_plan_status": str(plan.get("status") or ("planned" if plan else "missing")),
            "install_lock_status": str(lock.get("status") or "missing"),
            "install_lock_verification_status": str(lock_verification.get("status") or "not_checked"),
            "install_execution_status": str(execution.get("status") or "not_executed"),
            "install_receipt_status": str(receipt.get("status") or "missing"),
            "install_receipt_verification_status": str(receipt_verification.get("status") or "not_checked"),
            "resolved_lock_status": str(resolved_lock.get("status") or "missing"),
            "install_lock_hash": str(lock.get("install_lock_hash") or lock_verification.get("install_lock_hash") or "").strip() or None,
            "resolved_lock_hash": str(resolved_lock.get("resolved_lock_hash") or lock_verification.get("resolved_lock_hash") or "").strip() or None,
            "receipt_packages_hash": str(receipt.get("packages_hash") or "").strip() or None,
            "uv_install_plan_status": str(uv_plan.get("status") or "missing"),
            "uv_plan_hash": str(uv_plan.get("plan_hash") or "").strip() or None,
            "uv_install_lock_status": str(uv_lock.get("status") or "missing"),
            "uv_install_lock_verification_status": str(uv_lock_verification.get("status") or "not_checked"),
            "uv_install_execution_status": str(uv_execution.get("status") or "not_executed"),
            "uv_install_receipt_status": str(uv_receipt.get("status") or "missing"),
            "uv_install_receipt_verification_status": str(uv_receipt_verification.get("status") or "not_checked"),
            "uv_lock_hash": str(uv_lock.get("uv_lock_hash") or uv_lock_verification.get("uv_lock_hash") or "").strip() or None,
            "uv_receipt_hash": str(uv_receipt.get("uv_receipt_hash") or "").strip() or None,
            "reason": str(
                uv_receipt_verification.get("reason")
                or uv_execution.get("reason")
                or uv_lock_verification.get("reason")
                or receipt_verification.get("reason")
                or execution.get("reason")
                or lock_verification.get("reason")
                or ""
            ).strip()
            or None,
        }

    def _with_install_summary(self, *, status: str, environment: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": status,
            "environment": dict(environment or {}),
            "metadata": dict(metadata or {}),
            "install_status": self._install_status_summary(metadata),
        }

    def _spec_and_metadata(self, environment: Dict[str, Any]) -> tuple[ToolboxEnvironmentSpec, Path, Dict[str, Any]]:
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        ensured = self.environment_manager.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        return ensured, env_root, self.environment_manager.read_environment_metadata(ensured)

    @staticmethod
    def _write_metadata(env_root: Path, metadata: Dict[str, Any]) -> None:
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _environment_with_uv(spec: ToolboxEnvironmentSpec, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        env = spec.to_dict()
        uv = dict(dict(metadata or {}).get("uv_install_plan") or {}).get("uv")
        if isinstance(uv, dict):
            env["uv"] = dict(uv)
        return env

    def _lock_uv_plan(self, *, spec: ToolboxEnvironmentSpec, env_root: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        uv_plan = dict(metadata.get("uv_install_plan") or {})
        if not uv_plan:
            return metadata
        lock_payload = {
            "venv_key": str(spec.venv_key or ""),
            "environment_name": str(spec.environment_name or ""),
            "uv_plan_hash": str(uv_plan.get("plan_hash") or ""),
            "pyproject_toml": uv_plan.get("pyproject_toml"),
            "uv_lock": uv_plan.get("uv_lock"),
            "dependency_groups": list(uv_plan.get("dependency_groups") or []),
        }
        lock_hash = stable_hash(lock_payload)[:16]
        metadata["uv_install_lock"] = {
            "status": "locked",
            "locked_at": time.time(),
            "uv_lock_hash": lock_hash,
            "source_uv_plan_hash": str(uv_plan.get("plan_hash") or ""),
            "dependency_groups": list(uv_plan.get("dependency_groups") or []),
        }
        self._write_metadata(env_root, metadata)
        return metadata

    def _verify_uv_lock(self, *, env_root: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        uv_plan = dict(metadata.get("uv_install_plan") or {})
        uv_lock = dict(metadata.get("uv_install_lock") or {})
        if not uv_plan:
            return metadata
        status = "ok"
        reason = None
        expected = stable_hash(
            {
                "venv_key": str(metadata.get("venv_key") or ""),
                "environment_name": str(metadata.get("environment_name") or ""),
                "uv_plan_hash": str(uv_plan.get("plan_hash") or ""),
                "pyproject_toml": uv_plan.get("pyproject_toml"),
                "uv_lock": uv_plan.get("uv_lock"),
                "dependency_groups": list(uv_plan.get("dependency_groups") or []),
            }
        )[:16]
        actual = str(uv_lock.get("uv_lock_hash") or "").strip()
        if not uv_lock:
            status = "missing"
            reason = "uv_install_lock_missing"
        elif actual != expected:
            status = "stale"
            reason = "uv_install_lock_hash_mismatch"
        metadata["uv_install_lock_verification"] = {
            "status": status,
            "verified_at": time.time(),
            "uv_lock_hash": actual or None,
            "expected_uv_lock_hash": expected,
            "reason": reason,
        }
        self._write_metadata(env_root, metadata)
        return metadata

    def _execute_uv_install(self, *, spec: ToolboxEnvironmentSpec, env_root: Path, metadata: Dict[str, Any], allow_execution: bool) -> Dict[str, Any]:
        uv_plan = dict(metadata.get("uv_install_plan") or {})
        if not uv_plan:
            return metadata
        uv = dict(uv_plan.get("uv") or {})
        if not allow_execution:
            metadata["uv_install_execution"] = {
                "status": "blocked",
                "executed": False,
                "reason": "execution_not_enabled",
                "executed_at": time.time(),
            }
            self._write_metadata(env_root, metadata)
            return metadata
        verification = dict(self._verify_uv_lock(env_root=env_root, metadata=metadata).get("uv_install_lock_verification") or {})
        if str(verification.get("status") or "") != "ok":
            metadata = self.environment_manager.read_environment_metadata(spec)
            metadata["uv_install_execution"] = {
                "status": "blocked",
                "executed": False,
                "reason": str(verification.get("reason") or "uv_install_lock_invalid"),
                "executed_at": time.time(),
            }
            self._write_metadata(env_root, metadata)
            return metadata
        if not bool(uv.get("available")) or not str(uv.get("resolved_executable") or "").strip():
            metadata["uv_install_execution"] = {
                "status": "blocked",
                "executed": False,
                "reason": "uv_missing",
                "executed_at": time.time(),
            }
            self._write_metadata(env_root, metadata)
            return metadata
        pyproject = str(uv_plan.get("pyproject_toml") or "").strip()
        uv_lock = str(uv_plan.get("uv_lock") or "").strip()
        if pyproject:
            (env_root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        if uv_lock:
            (env_root / "uv.lock").write_text(uv_lock, encoding="utf-8")
        command = [str(uv.get("resolved_executable")), "sync", "--project", str(env_root), "--frozen"]
        for group in list(uv_plan.get("dependency_groups") or []):
            command.extend(["--group", str(group)])
        result = subprocess.run(
            command,
            cwd=str(env_root),
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        uv_python = str(_uv_python_executable(env_root))
        metadata["uv_install_execution"] = {
            "status": "ok" if int(result.returncode or 0) == 0 else "failed",
            "executed": True,
            "executed_at": time.time(),
            "returncode": int(result.returncode or 0),
            "stdout": str(result.stdout or ""),
            "stderr": str(result.stderr or ""),
            "command": command,
            "uv_python_executable": uv_python,
            "uv_lock_hash": str(dict(metadata.get("uv_install_lock") or {}).get("uv_lock_hash") or "").strip() or None,
        }
        if int(result.returncode or 0) == 0:
            receipt_payload = {
                "status": "ok",
                "captured_at": time.time(),
                "uv_python_executable": uv_python,
                "uv_lock_hash": str(dict(metadata.get("uv_install_lock") or {}).get("uv_lock_hash") or "").strip() or None,
                "uv_plan_hash": str(uv_plan.get("plan_hash") or "").strip() or None,
                "dependency_groups": list(uv_plan.get("dependency_groups") or []),
            }
            receipt_payload["uv_receipt_hash"] = stable_hash(receipt_payload)[:16]
            metadata["uv_install_receipt"] = receipt_payload
        self._write_metadata(env_root, metadata)
        return self._verify_uv_receipt(env_root=env_root, metadata=metadata)

    def _verify_uv_receipt(self, *, env_root: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        uv_plan = dict(metadata.get("uv_install_plan") or {})
        if not uv_plan:
            return metadata
        receipt = dict(metadata.get("uv_install_receipt") or {})
        lock_verification = dict(metadata.get("uv_install_lock_verification") or {})
        status = "ok"
        reason = None
        if str(lock_verification.get("status") or "") != "ok":
            status = "stale"
            reason = str(lock_verification.get("reason") or "uv_install_lock_invalid")
        elif not receipt:
            status = "missing"
            reason = "uv_install_receipt_missing"
        elif str(receipt.get("uv_plan_hash") or "") != str(uv_plan.get("plan_hash") or ""):
            status = "stale"
            reason = "uv_receipt_plan_hash_mismatch"
        elif str(receipt.get("uv_lock_hash") or "") != str(dict(metadata.get("uv_install_lock") or {}).get("uv_lock_hash") or ""):
            status = "stale"
            reason = "uv_receipt_lock_hash_mismatch"
        metadata["uv_install_receipt_verification"] = {
            "status": status,
            "verified_at": time.time(),
            "reason": reason,
            "uv_receipt_hash": str(receipt.get("uv_receipt_hash") or "").strip() or None,
            "uv_python_executable": str(receipt.get("uv_python_executable") or "").strip() or None,
        }
        self._write_metadata(env_root, metadata)
        return metadata

    def realize_environment(
        self,
        *,
        environment_name: str = "workflow-python-helper",
        python_policy: Optional[Dict[str, Any]] = None,
        package_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec = self._spec_from_request(environment_name=environment_name, python_policy=python_policy)
        metadata = self.environment_manager.realize_environment(
            spec,
            environment_description=self._environment_description(spec, python_policy=python_policy),
            required_packages=_pinned_packages(python_policy),
            missing_packages=[],
            toolbox_id=_clean(package_id) or "workflow_python",
            sandbox_profile_id=_clean(workflow_id) or "workflow_python",
            tool_keys=[],
        )
        env = spec.to_dict()
        env["uv"] = _uv_availability(python_policy)
        return self._with_install_summary(status="ok", environment=env, metadata=metadata)

    def prepare_install(
        self,
        *,
        environment_name: str = "workflow-python-helper",
        python_policy: Optional[Dict[str, Any]] = None,
        package_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec = self._spec_from_request(environment_name=environment_name, python_policy=python_policy)
        metadata = self.environment_manager.prepare_install_plan(
            spec,
            environment_description=self._environment_description(spec, python_policy=python_policy),
            required_packages=_pinned_packages(python_policy),
            missing_packages=[],
            toolbox_id=_clean(package_id) or "workflow_python",
            sandbox_profile_id=_clean(workflow_id) or "workflow_python",
            tool_keys=[],
        )
        uv = _uv_availability(python_policy)
        if bool(uv.get("enabled")):
            metadata["uv_install_plan"] = {
                "status": "planned" if bool(uv.get("available")) else "missing_uv",
                "tool": "uv",
                "uv": uv,
                "pyproject_toml": uv.get("pyproject_toml"),
                "uv_lock": uv.get("uv_lock"),
                "dependency_groups": list(uv.get("dependency_groups") or []),
                "allow_execution": False,
                "plan_hash": stable_hash(
                    {
                        "tool": "uv",
                        "pyproject_toml": uv.get("pyproject_toml"),
                        "uv_lock": uv.get("uv_lock"),
                        "dependency_groups": list(uv.get("dependency_groups") or []),
                    }
                ),
            }
            self._write_metadata(Path(spec.venv_path).expanduser().resolve(), metadata)
        return self._with_install_summary(status="ok", environment=self._environment_with_uv(spec, metadata), metadata=metadata)

    def lock_install(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        reject_model_runtime_selection(environment)
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.lock_install_plan(spec)
        ensured, env_root, metadata = self._spec_and_metadata(environment)
        metadata = self._lock_uv_plan(spec=ensured, env_root=env_root, metadata=metadata)
        return self._with_install_summary(status="ok", environment=self._environment_with_uv(ensured, metadata), metadata=metadata)

    def verify_install_lock(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        reject_model_runtime_selection(environment)
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.verify_install_lock(spec)
        ensured, env_root, metadata = self._spec_and_metadata(environment)
        metadata = self._verify_uv_lock(env_root=env_root, metadata=metadata)
        return self._with_install_summary(status="ok", environment=self._environment_with_uv(ensured, metadata), metadata=metadata)

    def resolve_install_lock(self, *, environment: Dict[str, Any], allow_resolution: bool = False) -> Dict[str, Any]:
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.resolve_install_lock(spec, allow_resolution=bool(allow_resolution))
        return self._with_install_summary(status="ok", environment=spec.to_dict(), metadata=metadata)

    def execute_install(self, *, environment: Dict[str, Any], allow_execution: bool = False) -> Dict[str, Any]:
        reject_model_runtime_selection(environment)
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.execute_install_plan(spec, allow_execution=bool(allow_execution))
        ensured, env_root, metadata = self._spec_and_metadata(environment)
        metadata = self._execute_uv_install(spec=ensured, env_root=env_root, metadata=metadata, allow_execution=bool(allow_execution))
        return self._with_install_summary(status="ok", environment=self._environment_with_uv(ensured, metadata), metadata=metadata)

    def verify_install_receipt(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        reject_model_runtime_selection(environment)
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.verify_install_receipt(spec)
        ensured, env_root, metadata = self._spec_and_metadata(environment)
        metadata = self._verify_uv_lock(env_root=env_root, metadata=metadata)
        metadata = self._verify_uv_receipt(env_root=env_root, metadata=metadata)
        return self._with_install_summary(status="ok", environment=self._environment_with_uv(ensured, metadata), metadata=metadata)

    def select_runtime_python(
        self,
        *,
        environment: Dict[str, Any],
        bootstrap_python_executable: Optional[str] = None,
        fallback_python_executable: Optional[str] = None,
    ) -> Dict[str, Any]:
        reject_model_runtime_selection(environment)
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.read_environment_metadata(spec)
        uv_receipt_verification = dict(metadata.get("uv_install_receipt_verification") or {})
        uv_receipt = dict(metadata.get("uv_install_receipt") or {})
        uv_python = str(uv_receipt.get("uv_python_executable") or "").strip()
        if str(uv_receipt_verification.get("status") or "") == "ok" and uv_python:
            return {
                "status": "ok",
                "environment": spec.to_dict(),
                "python_executable": uv_python,
                "python_source": "uv",
            }
        executable = self.environment_manager.runtime_python_executable(
            spec,
            bootstrap_python_executable=bootstrap_python_executable,
            fallback_python_executable=fallback_python_executable,
        )
        bootstrap = _clean(bootstrap_python_executable) or _clean(fallback_python_executable)
        return {
            "status": "ok",
            "environment": spec.to_dict(),
            "python_executable": executable,
            "python_source": "bootstrap" if bootstrap and executable == bootstrap else "venv",
        }

    def gc_runtime_environments(
        self,
        *,
        referenced_environment_keys: Optional[Sequence[str]] = None,
        referenced_environment_paths: Optional[Sequence[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        root = self.environment_manager.runtime_environments_root
        referenced_keys = {
            _clean(item)
            for item in list(referenced_environment_keys or [])
            if _clean(item)
        }
        referenced_paths: set[str] = set()
        for raw in list(referenced_environment_paths or []):
            value = _clean(raw)
            if not value:
                continue
            try:
                resolved = Path(value).expanduser().resolve()
            except Exception:
                continue
            try:
                if resolved == root or root in resolved.parents:
                    referenced_paths.add(str(resolved))
            except Exception:
                continue
        stale: list[str] = []
        removed: list[str] = []
        if root.exists():
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                try:
                    resolved_child = str(child.expanduser().resolve())
                except Exception:
                    resolved_child = ""
                if child.name in referenced_keys or (resolved_child and resolved_child in referenced_paths):
                    continue
                stale.append(child.name)
                if not dry_run:
                    shutil.rmtree(child, ignore_errors=True)
                    removed.append(child.name)
        return {
            "status": "ok",
            "environment_root_kind": "environments",
            "environment_root": str(root),
            "dry_run": bool(dry_run),
            "referenced_environment_keys": sorted(referenced_keys),
            "stale_environment_keys": sorted(stale),
            "removed_environment_keys": sorted(removed),
        }


__all__ = ["HostedPythonRuntimeBase", "HostedPythonRuntimeManager"]
