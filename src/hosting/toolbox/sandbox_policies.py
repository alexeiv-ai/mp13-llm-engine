"""Host-owned sandbox policies referenced by toolbox configuration."""
from __future__ import annotations

import copy
from typing import Any

from ..sandbox.policy import WorkerSandboxPolicy


_COMPUTE_ONLY_POLICY = {
    "policy_id": "compute-only",
    "sandbox_required": True,
    "filesystem_read_roots": [],
    "filesystem_write_roots": [],
    "artifact_roots": [],
    "network": False,
    "subprocess": False,
    "brokered_io": {"filesystem": False, "http": False, "subprocess": False},
    "host_api_permissions": [],
}


def compute_only_sandbox_policy() -> dict[str, Any]:
    return copy.deepcopy(_COMPUTE_ONLY_POLICY)


def compute_only_worker_policy() -> WorkerSandboxPolicy:
    policy = WorkerSandboxPolicy.from_mapping(
        {
            "sandbox": {
                "enabled": True,
                "profile": "compute-only",
                "filesystem": {"default_access": "deny", "rules": []},
                "artifact_roots": {},
                "process": {"allow_subprocess": False, "inherit_parent_handles": False},
                "network": {"mode": "disabled"},
                "brokered_io": {"filesystem": False, "http": False, "subprocess": False},
            }
        }
    )
    if (
        not policy.enabled
        or policy.filesystem_rules
        or policy.artifact_roots
        or policy.process.allow_subprocess
        or policy.process.inherit_parent_handles
        or policy.network.mode != "disabled"
        or policy.brokered_io.filesystem
        or policy.brokered_io.http
        or policy.brokered_io.subprocess
    ):
        raise ValueError("compute_only_policy_invalid")
    return policy


__all__ = ["compute_only_sandbox_policy", "compute_only_worker_policy"]
