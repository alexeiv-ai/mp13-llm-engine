"""Internal JS runtime base for hosted workflow helper compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .process_base import HostedProcessSandboxBase
from .runtime_base import HostedEnvironmentKeySpec, HostedRuntimeIdentity


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _imports(policy: Optional[Dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(dict(policy or {}).get("required_imports") or []):
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
    """Thin JS runtime base above the language-neutral process base."""

    sandbox_kind = "workflow_js"

    def __init__(self, hosting_root: Path):
        super().__init__()
        self.hosting_root = Path(hosting_root).expanduser().resolve()

    def environment_key_spec(
        self,
        *,
        environment_name: str,
        profile: str = "helper",
        node_policy: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        runtime_hash: Optional[str] = None,
        runtime_version: Optional[str] = None,
    ) -> HostedEnvironmentKeySpec:
        node = dict(node_policy or {})
        effective_runtime_hash = (
            _clean(runtime_hash)
            or _clean(node.get("runtime_hash"))
            or _clean(node.get("node_executable"))
            or "node-default"
        )
        runtime = HostedRuntimeIdentity(
            runtime_kind="workflow_js",
            profile=_clean(profile) or "helper",
            runtime_hash=effective_runtime_hash,
            runtime_version=_clean(runtime_version) or _clean(node.get("runtime_version")) or None,
            capability_profile="workflow_js_helper",
        )
        return HostedEnvironmentKeySpec(
            environment_name=_clean(environment_name) or "workflow-js-helper",
            runtime=runtime,
            sandbox_policy=dict(sandbox_policy or {}),
            required_imports=_imports(node),
            package_pins=_pins(node),
            dependency_lock_hash=_clean(node.get("dependency_lock_hash")) or None,
        )

    def environment_spec(
        self,
        *,
        environment_name: str = "workflow-js-helper",
        profile: str = "helper",
        node_policy: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = self.environment_key_spec(
            environment_name=environment_name,
            profile=profile,
            node_policy=node_policy,
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
                "environment_root_kind": "runtime_envs",
                "environment_consumer_kind": "workflow_js_helper",
                "workflow_runtime_kind": "workflow_js",
                "workflow_profile": _clean(profile) or "helper",
                "runtime_hash": env["runtime"]["runtime_hash"],
                "sandbox_policy_hash": env["sandbox_policy_hash"],
                "required_imports": list(env["required_imports"]),
                "package_pins": dict(env["package_pins"]),
                "dependency_lock_hash": env.get("dependency_lock_hash"),
                "install_status": "not_applicable",
            },
        }


__all__ = ["HostedJsRuntimeBase"]
