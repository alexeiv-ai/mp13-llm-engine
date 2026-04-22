"""Stable API for explicit elevated transport host setup."""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .hosting_config_cli import (
    _admin_setup_elevation_command,
    _admin_setup_platform,
    _admin_setup_script,
    _write_admin_setup_script,
)


@dataclass(frozen=True)
class TransportAdminSetupRequest:
    enable_ssh_service: bool = True
    enable_firewall: bool = False
    enable_user_linger: bool = False
    target_user: str = ""
    execute: bool = False


class _Args:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.admin_setup_enable_ssh_service = bool(data.get("enable_ssh_service", True))
        self.admin_setup_enable_firewall = bool(data.get("enable_firewall", False))
        self.admin_setup_enable_user_linger = bool(data.get("enable_user_linger", False))
        self.admin_setup_target_user = str(data.get("target_user") or "").strip()


def _data(request: TransportAdminSetupRequest | Dict[str, Any]) -> Dict[str, Any]:
    return asdict(request) if isinstance(request, TransportAdminSetupRequest) else dict(request or {})


def plan_transport_admin_setup(request: TransportAdminSetupRequest | Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _data(request or {})
    platform_name = _admin_setup_platform()
    script, suffix, followups = _admin_setup_script(_Args(data), platform_name=platform_name)
    return {
        "status": "dry_run",
        "action": "transport_admin_setup_plan",
        "platform": platform_name,
        "script_suffix": suffix,
        "script": script,
        "followups": followups,
        "execute": False,
    }


def execute_transport_admin_setup(request: TransportAdminSetupRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    if not bool(data.get("execute", False)):
        raise PermissionError("execute=True is required for elevated transport admin setup")
    plan = plan_transport_admin_setup(data)
    script_path = _write_admin_setup_script(str(plan.get("script") or ""), str(plan.get("script_suffix") or ".sh"))
    command, method = _admin_setup_elevation_command(script_path, platform_name=str(plan.get("platform") or ""))
    completed = subprocess.run(command, check=False)
    return {
        **plan,
        "status": "ok" if int(completed.returncode) == 0 else "elevation_failed",
        "action": "transport_admin_setup_execute",
        "execute": True,
        "script_file": str(script_path),
        "elevation_method": method,
        "returncode": int(completed.returncode),
    }


__all__ = [
    "TransportAdminSetupRequest",
    "plan_transport_admin_setup",
    "execute_transport_admin_setup",
]
