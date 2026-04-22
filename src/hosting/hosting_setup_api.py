"""Stable host-local hosting setup API.

This is the integration contract for backend bootstrap/materialization code.
It is host-local only: callers must be running on the machine whose hosting
configuration files are being inspected or changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from . import hosting_config_cli as _cli


@dataclass(frozen=True)
class LocalHostingSetupRequest:
    host_local: bool = True
    default_config_dir: Optional[Path] = None
    hosting_root: Optional[Path] = None
    mode: str = "local_only"
    endpoint_mode: str = "exclusive"
    lifecycle_profile: str = "detached_user_process"
    require_auth: Optional[bool] = None
    usage_intent: str = "single_admin"
    key_action: str = "replace"
    key_source: str = "generate"
    admin_key_id: str = "admin-main"
    admin_public_key: str = ""
    admin_public_key_file: str = ""
    generated_key_passphrase: str = ""
    client_secret_password: str = ""
    permission_action: str = "none"
    print_private_key_handoff: bool = False
    confirm_reset: bool = False


def _data(request: LocalHostingSetupRequest | Dict[str, Any] | None) -> Dict[str, Any]:
    if request is None:
        return {}
    return asdict(request) if isinstance(request, LocalHostingSetupRequest) else dict(request or {})


def _args(request: LocalHostingSetupRequest | Dict[str, Any] | None) -> Any:
    data = _data(request)
    args = _cli._build_parser().parse_args([])
    args.interactive = False
    args.json_output = False
    args.no_interactive = True
    for key, value in data.items():
        if value is None:
            continue
        if key == "hosting_root":
            if not data.get("default_config_dir"):
                setattr(args, "default_config_dir", str(Path(value).expanduser().resolve().parent))
            continue
        if isinstance(value, Path):
            value = str(value)
        setattr(args, key, value)
    return args


def _require_host_local(data: Dict[str, Any]) -> None:
    if not bool(data.get("host_local", True)):
        raise PermissionError("hosting setup API is host-local only")


def plan_local_hosting_setup(request: LocalHostingSetupRequest | Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    args = _args(data)
    paths = _cli._resolve_paths(args, create_dirs=False)
    mode = _cli._normalize_mode(str(getattr(args, "mode", "") or "local_only"), "local_only")
    endpoint_mode = _cli._normalize_endpoint_mode(str(getattr(args, "endpoint_mode", "") or "exclusive"), "exclusive")
    lifecycle_profile = _cli._normalize_lifecycle_profile(
        str(getattr(args, "lifecycle_profile", "") or "detached_user_process"),
        "detached_user_process",
    )
    require_auth = _cli._safe_require_auth(
        connectivity_mode=mode,
        endpoint_mode=endpoint_mode,
        requested=getattr(args, "require_auth", None),
    )
    summary = _cli._summarize_existing_config(
        control_state_path=paths["control_state_path"],
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
    )
    probe = _cli._probe_current_files(
        control_state_path=paths["control_state_path"],
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
        mappings_file=paths["mappings_file"],
        bootstrap_state_file=paths["bootstrap_state_file"],
        audit_file=paths["audit_file"],
    )
    return {
        "status": "planned",
        "host_local": True,
        "would_write": False,
        "hosting_root": str(paths["hosting_root"]),
        "control_state_file": str(paths["control_state_path"]),
        "access_control_file": str(paths["access_file"]),
        "keys_file": str(paths["keys_file"]),
        "connectivity_mode": mode,
        "endpoint_mode_default": endpoint_mode,
        "lifecycle_profile": lifecycle_profile,
        "require_auth": require_auth,
        "key_action": str(getattr(args, "key_action", "") or "replace"),
        "key_source": str(getattr(args, "key_source", "") or "generate"),
        "admin_key_id": str(getattr(args, "admin_key_id", "") or "admin-main"),
        "current_summary": summary,
        "current_state": _cli._classify_config_state(summary, probe),
    }


def apply_local_hosting_setup(request: LocalHostingSetupRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    return _cli.run_setup(_args(data))


def inspect_local_hosting_setup(request: LocalHostingSetupRequest | Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    args = _args(data)
    return {
        "status": "ok",
        "host_local": True,
        "setup": _cli.run_status(args),
        "doctor": _cli.run_doctor(args),
    }


def reset_local_hosting_setup(request: LocalHostingSetupRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    if not bool(data.get("confirm_reset", False)):
        raise PermissionError("confirm_reset=True is required to reset hosting setup")
    args = _args(data)
    paths = _cli._resolve_paths(args, create_dirs=False)
    return _cli._reset_access_configuration(paths)


__all__ = [
    "LocalHostingSetupRequest",
    "plan_local_hosting_setup",
    "apply_local_hosting_setup",
    "inspect_local_hosting_setup",
    "reset_local_hosting_setup",
]
