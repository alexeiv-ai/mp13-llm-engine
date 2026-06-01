"""Workflow-facing Python runtime environment helpers.

This wraps the existing toolbox environment manager with workflow/runtime
terminology so new workflow APIs do not need to expose toolbox IDs or tool keys.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..toolbox.environment import ToolboxEnvironmentManager
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
        self.environment_manager = ToolboxEnvironmentManager(self.hosting_root)

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
            dependency_lock_hash=None,
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
        return {
            "status": "ok",
            "environment": spec.to_dict(),
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
        return self.environment_manager.workflow_python_helper_environment_spec(
            policy=dict(python_policy or {}),
            environment_name=_clean(environment_name) or "workflow-python-helper",
        )

    def _environment_description(self, spec: ToolboxEnvironmentSpec, *, python_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        pinned = _pinned_packages(python_policy)
        return {
            "name": spec.environment_name,
            "extra_packages": pinned,
            "effective_extra_packages": pinned,
            "allow_online_install": False,
            "effective_allow_online_install": False,
            "lineage": [spec.environment_name],
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
            "reason": str(
                receipt_verification.get("reason")
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
        return self._with_install_summary(status="ok", environment=spec.to_dict(), metadata=metadata)

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
        return self._with_install_summary(status="ok", environment=spec.to_dict(), metadata=metadata)

    def lock_install(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.lock_install_plan(spec)
        return self._with_install_summary(status="ok", environment=spec.to_dict(), metadata=metadata)

    def verify_install_lock(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.verify_install_lock(spec)
        return self._with_install_summary(status="ok", environment=spec.to_dict(), metadata=metadata)

    def resolve_install_lock(self, *, environment: Dict[str, Any], allow_resolution: bool = False) -> Dict[str, Any]:
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.resolve_install_lock(spec, allow_resolution=bool(allow_resolution))
        return self._with_install_summary(status="ok", environment=spec.to_dict(), metadata=metadata)

    def execute_install(self, *, environment: Dict[str, Any], allow_execution: bool = False) -> Dict[str, Any]:
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.execute_install_plan(spec, allow_execution=bool(allow_execution))
        return self._with_install_summary(status="ok", environment=spec.to_dict(), metadata=metadata)

    def verify_install_receipt(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        metadata = self.environment_manager.verify_install_receipt(spec)
        return self._with_install_summary(status="ok", environment=spec.to_dict(), metadata=metadata)

    def select_runtime_python(
        self,
        *,
        environment: Dict[str, Any],
        bootstrap_python_executable: Optional[str] = None,
        fallback_python_executable: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec = ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
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
            "environment_root_kind": "runtime_envs",
            "environment_root": str(root),
            "dry_run": bool(dry_run),
            "referenced_environment_keys": sorted(referenced_keys),
            "stale_environment_keys": sorted(stale),
            "removed_environment_keys": sorted(removed),
        }


__all__ = ["HostedPythonRuntimeBase", "HostedPythonRuntimeManager"]
