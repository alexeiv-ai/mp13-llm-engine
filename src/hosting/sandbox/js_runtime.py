"""Internal JS runtime base for hosted workflow node execution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..environments import EnvironmentManager, EnvironmentRequest
from .process_base import HostedProcessSandboxBase
from .runtime_base import HostedEnvironmentKeySpec, HostedRuntimeIdentity


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _imports(policy: Optional[Dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    row = dict(policy or {})
    for item in list(row.get("required_imports") or row.get("allowed_host_modules") or []):
        name = _clean(item)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _pins(policy: Optional[Dict[str, Any]]) -> Dict[str, str]:
    return {
        _clean(key): _clean(value)
        for key, value in dict(dict(policy or {}).get("package_pins") or {}).items()
        if _clean(key) and _clean(value)
    }


class HostedJsRuntimeBase(HostedProcessSandboxBase):
    """Thin QuickJS runtime base above the language-neutral process base."""

    sandbox_kind = "workflow_js"

    def __init__(self, hosting_root: Path, *, shared_environment_manager: EnvironmentManager | None = None):
        super().__init__()
        self.hosting_root = Path(hosting_root).expanduser().resolve()
        self.shared_environment_manager = shared_environment_manager

    def acquire_shared_environment(self, request: EnvironmentRequest) -> Dict[str, Any]:
        if self.shared_environment_manager is None:
            raise RuntimeError("shared_environment_manager_required")
        if request.consumer_kind != "workflow_js_node":
            raise ValueError("workflow_js_consumer_kind_invalid")
        return self.shared_environment_manager.ensure(request)

    def release_shared_environment(self, *, reference_id: str) -> Dict[str, Any]:
        if self.shared_environment_manager is None:
            raise RuntimeError("shared_environment_manager_required")
        return self.shared_environment_manager.release(reference_id=reference_id)

    def environment_key_spec(
        self,
        *,
        environment_name: str,
        profile: str = "node",
        node_policy: Optional[Dict[str, Any]] = None,
        javascript_policy: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        runtime_hash: Optional[str] = None,
        runtime_version: Optional[str] = None,
    ) -> HostedEnvironmentKeySpec:
        node = dict(node_policy or {})
        js = {**node, **dict(javascript_policy or {})}
        effective_runtime_hash = (
            _clean(runtime_hash)
            or _clean(js.get("runtime_hash"))
            or _clean(js.get("quickjs_runtime_hash"))
            or _clean(js.get("binding"))
            or "quickjs-default"
        )
        runtime = HostedRuntimeIdentity(
            runtime_kind="workflow_js",
            profile=_clean(profile) or "node",
            runtime_hash=effective_runtime_hash,
            runtime_version=_clean(runtime_version) or _clean(js.get("runtime_version")) or _clean(js.get("quickjs_version")) or None,
            capability_profile="workflow_js_node",
        )
        return HostedEnvironmentKeySpec(
            environment_name=_clean(environment_name) or "workflow-js-node",
            runtime=runtime,
            sandbox_policy=dict(sandbox_policy or {}),
            required_imports=_imports(js),
            package_pins=_pins(js),
            dependency_lock_hash=_clean(js.get("dependency_lock_hash") or js.get("bundle_hash")) or None,
        )

    def environment_spec(
        self,
        *,
        environment_name: str = "workflow-js-node",
        profile: str = "node",
        node_policy: Optional[Dict[str, Any]] = None,
        javascript_policy: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = self.environment_key_spec(
            environment_name=environment_name,
            profile=profile,
            node_policy=node_policy,
            javascript_policy=javascript_policy,
            sandbox_policy=sandbox_policy,
        )
        env = spec.to_dict()
        return {
            "status": "ok",
            **env,
            "environment": {
                "environment_key": env["environment_key"],
                "environment_key_full": env["environment_key_full"],
                "environment_name": env["environment_name"],
                "environment_root_kind": "environments",
                "environment_consumer_kind": "workflow_js_node",
                "workflow_runtime_kind": "workflow_js",
                "workflow_profile": _clean(profile) or "node",
                "runtime_hash": env["runtime"]["runtime_hash"],
                "sandbox_policy_hash": env["sandbox_policy_hash"],
                "required_imports": list(env["required_imports"]),
                "package_pins": dict(env["package_pins"]),
                "dependency_lock_hash": env.get("dependency_lock_hash"),
                "install_status": "not_applicable",
            },
        }


__all__ = ["HostedJsRuntimeBase"]
