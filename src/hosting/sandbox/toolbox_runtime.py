"""Internal toolbox runtime base.

Toolbox remains a concrete public sandbox kind with its own staged bundles,
tool routing, callbacks, and brokered I/O semantics.  This module only maps
toolbox worker identity onto the shared hosted process/runtime metadata shape.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..toolbox.bundle_models import ToolboxEnvironmentSpec
from .process_base import HostedProcessSandboxBase
from .runtime_base import HostedEnvironmentKeySpec, HostedRuntimeIdentity, stable_hash


def _clean(value: Any) -> str:
    return str(value or "").strip()


class HostedToolboxRuntimeBase(HostedProcessSandboxBase):
    """Internal base for toolbox executor registrations.

    It deliberately does not own toolbox environment realization.  Existing
    toolbox lifecycle code keeps using ``ToolboxEnvironmentManager`` and
    ``toolbox_venvs``; this base adds shared environment-key and registration
    metadata so toolbox workers can later move onto common pool accounting.
    """

    sandbox_kind = "toolbox_executor"

    def environment_key_spec(
        self,
        *,
        toolbox_id: str,
        sandbox_profile_id: str,
        bundle_revision: str,
        environment: ToolboxEnvironmentSpec | Dict[str, Any],
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> HostedEnvironmentKeySpec:
        spec = environment if isinstance(environment, ToolboxEnvironmentSpec) else ToolboxEnvironmentSpec.from_dict(dict(environment or {}))
        profile_id = _clean(sandbox_profile_id) or "default"
        runtime_hash = stable_hash(
            {
                "toolbox_runtime_hash": _clean(spec.toolbox_runtime_hash) or "toolbox-executor-v1",
                "toolbox_id": _clean(toolbox_id),
                "sandbox_profile_id": profile_id,
                "bundle_revision": _clean(bundle_revision),
                "venv_key": _clean(spec.venv_key),
                "intrinsics_profile_id": _clean(spec.intrinsics_profile_id) or "none",
            }
        )
        runtime = HostedRuntimeIdentity(
            runtime_kind="toolbox_executor",
            profile=profile_id,
            runtime_hash=f"toolbox-executor-v1:{runtime_hash[:16]}",
            capability_profile=_clean(spec.intrinsics_profile_id) or "none",
        )
        return HostedEnvironmentKeySpec(
            environment_name=_clean(spec.environment_name) or "base",
            runtime=runtime,
            sandbox_policy=dict(sandbox_policy or {}),
            required_imports=list(spec.required_imports or []),
            package_pins={},
            dependency_lock_hash=(
                _clean(spec.dependency_lock_hash)
                or _clean(spec.venv_lock_hash)
                or _clean(spec.environment_description_hash)
                or None
            ),
        )

    def registration_environment(
        self,
        *,
        environment: Dict[str, Any],
        toolbox_id: str,
        sandbox_profile_id: str,
        bundle_revision: str,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base = dict(environment or {})
        spec = ToolboxEnvironmentSpec.from_dict(base)
        key_spec = self.environment_key_spec(
            toolbox_id=toolbox_id,
            sandbox_profile_id=sandbox_profile_id,
            bundle_revision=bundle_revision,
            environment=spec,
            sandbox_policy=sandbox_policy,
        )
        identity = key_spec.to_dict()
        return {
            **base,
            "environment_key": identity["environment_key"],
            "environment_key_full": identity["environment_key_full"],
            "environment_identity": identity,
            "environment_root_kind": _clean(spec.environment_root_kind) or "toolbox_venvs",
            "environment_consumer_kind": _clean(spec.environment_consumer_kind) or "toolbox_executor",
        }


__all__ = ["HostedToolboxRuntimeBase"]
