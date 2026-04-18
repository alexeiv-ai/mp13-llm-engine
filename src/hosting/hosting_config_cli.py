"""
Interactive/non-interactive hosting access setup and reconfiguration utility.

Usage examples:
  python -m hosting.hosting_config_cli
  py hosting_config.py
  python -m hosting.hosting_config_cli --no-interactive --mode local_only --key-source import --admin-key-id admin-main --admin-public-key-file C:\\keys\\admin.pub
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if __package__ in {None, ""}:
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    from hosting.engine_host_service import EngineHostService, VALID_AUTH_ROLES
else:
    from .engine_host_service import EngineHostService, VALID_AUTH_ROLES


VALID_CONNECTIVITY_MODES = {"local_only", "ssh_tunnel_only", "truly_remote"}
VALID_ENDPOINT_MODES = {"exclusive", "shared"}
VALID_LIFECYCLE_PROFILES = {
    "foreground_terminal_bound",
    "detached_user_process",
    "service_managed",
}
VALID_KEY_SOURCES = {"generate", "import"}
VALID_IMPORT_SOURCES = {"file", "inline"}
VALID_COLOR_SCHEMES = {"dark", "light"}


CONNECTIVITY_INTENT_GUIDANCE: Dict[str, Dict[str, str]] = {
    "local_only": {
        "intent": "Single host usage with no off-host clients.",
        "provides": "Lowest setup overhead. Optional no-auth is possible only in strict safe profile.",
        "precautions": "Keep loopback-only bind and exclusive endpoint mode when auth is disabled.",
    },
    "ssh_tunnel_only": {
        "intent": "Remote operators connect through SSH tunnel while daemon stays loopback-bound.",
        "provides": "Remote reachability without direct non-loopback daemon exposure.",
        "precautions": "Require auth, maintain SSH key hygiene, and verify tunnel endpoint controls.",
    },
    "truly_remote": {
        "intent": "Persistent direct/proxied remote serving for multiple remote clients.",
        "provides": "Full remote operations with role separation and explicit ingress controls.",
        "precautions": "Require auth, enforce strict role boundaries, and apply firewall/proxy policy.",
    },
}


_COLOR_SCHEME = "dark"
_ANSI_ENABLED = False
_COLOR_TOKENS: Dict[str, str] = {}


def _enable_ansi_if_supported() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.name != "nt":
        return bool(getattr(sys.stdout, "isatty", lambda: False)())
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        if handle == 0:
            return False
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        if kernel32.SetConsoleMode(handle, mode.value | 0x0004) == 0:
            return False
        return True
    except Exception:
        return False


def _set_color_scheme(scheme: str) -> None:
    global _COLOR_SCHEME, _ANSI_ENABLED, _COLOR_TOKENS
    _COLOR_SCHEME = scheme if scheme in VALID_COLOR_SCHEMES else "dark"
    _ANSI_ENABLED = _enable_ansi_if_supported()
    if not _ANSI_ENABLED:
        _COLOR_TOKENS = {k: "" for k in {"reset", "title", "label", "value", "muted", "good", "warn", "bad", "accent"}}
        return
    if _COLOR_SCHEME == "light":
        _COLOR_TOKENS = {
            "reset": "\033[0m",
            "title": "\033[1;34m",
            "label": "\033[1;30m",
            "value": "\033[0;30m",
            "muted": "\033[0;90m",
            "good": "\033[0;32m",
            "warn": "\033[0;33m",
            "bad": "\033[0;31m",
            "accent": "\033[0;36m",
        }
    else:
        _COLOR_TOKENS = {
            "reset": "\033[0m",
            "title": "\033[1;96m",
            "label": "\033[1;37m",
            "value": "\033[0;97m",
            "muted": "\033[0;90m",
            "good": "\033[0;92m",
            "warn": "\033[0;93m",
            "bad": "\033[0;91m",
            "accent": "\033[0;96m",
        }


def _c(kind: str, text: Any) -> str:
    raw = str(text)
    if not _ANSI_ENABLED:
        return raw
    return f"{_COLOR_TOKENS.get(kind, '')}{raw}{_COLOR_TOKENS.get('reset', '')}"


def _print_title(text: str) -> None:
    print(f"\n{_c('title', text)}")


def _print_rule(char: str = "-", width: int = 72) -> None:
    print(_c("muted", char * width))


def _kv_rows(rows: list[Tuple[str, Any]], *, indent: str = "  ") -> None:
    width = max((len(str(label)) for label, _ in rows), default=0)
    for label, value in rows:
        print(f"{indent}{_c('label', str(label).ljust(width))} : {_c('value', value)}")


def _status_text(value: bool) -> str:
    return _c("good", "yes") if value else _c("muted", "no")


def _default_paths() -> Tuple[Path, Path]:
    try:
        from mp13_engine.mp13_config_paths import (  # type: ignore
            get_default_config_dir,
            get_hosting_control_state_path,
        )

        config_dir = Path(get_default_config_dir()).expanduser().resolve()
        control_state = Path(get_hosting_control_state_path()).expanduser().resolve()
        return config_dir, control_state
    except Exception:
        config_dir = (Path.home() / ".mp13-llm").expanduser().resolve()
        control_state = (config_dir / "hosting" / "access_control.json").resolve()
        return config_dir, control_state


def _hosting_root(default_config_dir: Path) -> Path:
    return (default_config_dir / "hosting").resolve()


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if not path.exists():
            return dict(default)
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else dict(default)
    except Exception:
        return dict(default)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _normalize_mode(value: str, default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in VALID_CONNECTIVITY_MODES else default


def _normalize_endpoint_mode(value: str, default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in VALID_ENDPOINT_MODES else default


def _normalize_lifecycle_profile(value: str, default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in VALID_LIFECYCLE_PROFILES else default


def _bool_prompt(question: str, default: bool) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(question + suffix).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"y", "yes", "1", "true"}


def _prompt_choice(question: str, valid: set[str], default: str) -> str:
    raw = input(f"{question} ({', '.join(sorted(valid))}) [{default}]: ").strip().lower()
    if not raw:
        return default
    return raw if raw in valid else default


def _prompt_menu(question: str, options: Dict[str, str], default: str) -> str:
    _print_rule("=")
    print(_c("title", question))
    for key, label in options.items():
        print(f"  {_c('accent', key + ')')} {_c('value', label)}")
    _print_rule("-")
    raw = input(f"Select [{default}]: ").strip().lower()
    if not raw:
        return default
    return raw if raw in options else default


def _wizard_choice_prompt(
    *,
    title: str,
    valid: set[str],
    current: str,
    allow_skip: bool = True,
) -> Tuple[str, str]:
    print(_c("title", title))
    print(f"  {_c('muted', 'options:')} {_c('value', ', '.join(sorted(valid)))}")
    nav = " prev=p, skip=s, enter=keep/current" if allow_skip else " prev=p, enter=keep/current"
    raw = input(f"  current={current}; choose value ({nav}): ").strip().lower()
    if raw in {"p", "prev"}:
        return "prev", current
    if allow_skip and raw in {"s", "skip"}:
        return "skip", current
    if not raw:
        return "next", current
    if raw in valid:
        return "next", raw
    print(f"  invalid choice '{raw}', keeping current value")
    return "next", current


def _wizard_bool_prompt(*, title: str, current: bool, allow_skip: bool = True) -> Tuple[str, bool]:
    nav = " prev=p, skip=s, enter=keep/current" if allow_skip else " prev=p, enter=keep/current"
    raw = input(f"{title} current={'yes' if current else 'no'} ({nav}): ").strip().lower()
    if raw in {"p", "prev"}:
        return "prev", current
    if allow_skip and raw in {"s", "skip"}:
        return "skip", current
    if not raw:
        return "next", current
    if raw in {"y", "yes", "1", "true"}:
        return "next", True
    if raw in {"n", "no", "0", "false"}:
        return "next", False
    print(f"  invalid boolean '{raw}', keeping current value")
    return "next", current


def _wizard_text_prompt(
    *,
    title: str,
    current: str,
    allow_skip: bool = True,
) -> Tuple[str, str]:
    nav = " prev=p, skip=s, enter=keep/current" if allow_skip else " prev=p, enter=keep/current"
    raw = input(f"{title} current={current} ({nav}): ").strip()
    if raw.lower() in {"p", "prev"}:
        return "prev", current
    if allow_skip and raw.lower() in {"s", "skip"}:
        return "skip", current
    if not raw:
        return "next", current
    return "next", raw


def _detect_bootstrap_admin_key_id(access_payload: Dict[str, Any]) -> str:
    if "control_config" in access_payload and isinstance(access_payload.get("control_config"), dict):
        access_payload = dict(access_payload.get("control_config") or {})
    candidate = str(access_payload.get("bootstrap_admin_key_id") or "").strip()
    return candidate


def _detect_admin_key_id(*, access_payload: Dict[str, Any], keys: Dict[str, Any]) -> str:
    bootstrap = _detect_bootstrap_admin_key_id(access_payload)
    if bootstrap:
        return bootstrap
    admin_ids = sorted(
        str(key_id).strip()
        for key_id, meta in dict(keys or {}).items()
        if str(key_id).strip() and str((meta or {}).get("role") or "").strip().lower() == "admin"
    )
    if len(admin_ids) == 1:
        return admin_ids[0]
    if "admin-main" in admin_ids:
        return "admin-main"
    return admin_ids[0] if admin_ids else "admin-main"


def _summarize_existing_config(
    *,
    control_state_path: Path,
    access_file: Path,
    keys_file: Path,
) -> Dict[str, Any]:
    access_exists = access_file.exists()
    keys_exists = keys_file.exists()
    summary: Dict[str, Any] = {
        "exists": False,
        "connectivity_mode": "local_only",
        "endpoint_mode_default": "exclusive",
        "lifecycle_profile": "detached_user_process",
        "require_auth": True,
        "admin_key_id": "admin-main",
        "admin_key_count": 0,
    }
    access_payload = _read_json(access_file, {})
    if "control_config" in access_payload and isinstance(access_payload.get("control_config"), dict):
        access_payload = dict(access_payload.get("control_config") or {})
    keys_payload = _read_json(keys_file, {"keys": {}})
    keys = dict(keys_payload.get("keys") or {})
    summary["admin_key_count"] = len([k for _, k in keys.items() if str((k or {}).get("role") or "") == "admin"])
    summary["admin_key_id"] = _detect_admin_key_id(access_payload=access_payload, keys=keys)
    if access_payload:
        ap = dict(access_payload.get("access_profile") or {})
        summary["connectivity_mode"] = _normalize_mode(
            str(ap.get("connectivity_mode") or summary["connectivity_mode"]),
            summary["connectivity_mode"],
        )
        summary["endpoint_mode_default"] = _normalize_endpoint_mode(
            str(access_payload.get("endpoint_mode_default") or summary["endpoint_mode_default"]),
            summary["endpoint_mode_default"],
        )
        summary["lifecycle_profile"] = _normalize_lifecycle_profile(
            str(access_payload.get("lifecycle_profile") or summary["lifecycle_profile"]),
            summary["lifecycle_profile"],
        )
        summary["require_auth"] = bool(access_payload.get("require_auth", summary["require_auth"]))
        summary["exists"] = bool(access_exists or keys_exists)
    try:
        svc = EngineHostService(control_state_file=control_state_path)
        cfg = dict(svc.get_control_config() or {})
        ap = dict(cfg.get("access_profile") or {})
        auth = dict(cfg.get("auth") or {})
        cfg_keys = dict(auth.get("keys") or {})
        summary["connectivity_mode"] = _normalize_mode(
            str(ap.get("connectivity_mode") or summary["connectivity_mode"]),
            summary["connectivity_mode"],
        )
        summary["endpoint_mode_default"] = _normalize_endpoint_mode(
            str(cfg.get("endpoint_mode_default") or summary["endpoint_mode_default"]),
            summary["endpoint_mode_default"],
        )
        summary["lifecycle_profile"] = _normalize_lifecycle_profile(
            str(cfg.get("lifecycle_profile") or summary["lifecycle_profile"]),
            summary["lifecycle_profile"],
        )
        summary["require_auth"] = bool(cfg.get("require_auth", summary["require_auth"]))
        summary["admin_key_count"] = len(
            [k for _, k in cfg_keys.items() if str((k or {}).get("role") or "").strip().lower() == "admin"]
        ) or summary["admin_key_count"]
        summary["admin_key_id"] = _detect_admin_key_id(access_payload=cfg, keys=cfg_keys or keys)
        summary["exists"] = bool(access_exists or keys_exists or summary["admin_key_count"])
    except Exception:
        pass
    if not (access_exists or keys_exists or int(summary.get("admin_key_count") or 0) > 0):
        summary["exists"] = False
    return summary


def _probe_current_files(
    *,
    control_state_path: Path,
    access_file: Path,
    keys_file: Path,
    mappings_file: Path,
    bootstrap_state_file: Path,
    audit_file: Path,
) -> Dict[str, Any]:
    access_payload = _read_json(access_file, {})
    keys_payload = _read_json(keys_file, {"keys": {}})
    mapping_payload = _read_json(mappings_file, {"clients": []})
    bootstrap_payload = _read_json(bootstrap_state_file, {})
    keys = dict(keys_payload.get("keys") or {})
    admin_key_ids = sorted(
        str(key_id).strip()
        for key_id, meta in keys.items()
        if str(key_id).strip() and str((meta or {}).get("role") or "").strip().lower() == "admin"
    )
    clients = list(mapping_payload.get("clients") or [])
    bootstrap_setup = dict(bootstrap_payload.get("setup") or {})
    return {
        "control_state_path": str(control_state_path),
        "hosting_root_path": str(access_file.parent),
        "access_exists": access_file.exists(),
        "keys_exists": keys_file.exists(),
        "mapping_exists": mappings_file.exists(),
        "bootstrap_exists": bootstrap_state_file.exists(),
        "audit_exists": audit_file.exists(),
        "bootstrap_admin_key_id": _detect_bootstrap_admin_key_id(access_payload),
        "admin_key_ids": admin_key_ids,
        "client_count": len(clients),
        "setup_scope": str(bootstrap_setup.get("setup_scope") or ""),
        "setup_key_action": str(bootstrap_setup.get("key_action") or ""),
        "setup_permission_action": str(bootstrap_setup.get("permission_action") or ""),
    }


def _classify_config_state(summary: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
    managed_file_flags = [
        bool(probe.get("access_exists")),
        bool(probe.get("keys_exists")),
        bool(probe.get("mapping_exists")),
        bool(probe.get("bootstrap_exists")),
        bool(probe.get("audit_exists")),
    ]
    managed_files_present = sum(1 for flag in managed_file_flags if flag)
    admin_key_count = int(summary.get("admin_key_count") or 0)
    bootstrap_admin_key_id = str(probe.get("bootstrap_admin_key_id") or "").strip()
    if managed_files_present == 0 and admin_key_count == 0 and not bootstrap_admin_key_id:
        return {
            "code": "clean",
            "label": "Not configured yet",
            "configured": False,
            "details": "No hosting access files or admin keys were detected.",
        }
    if admin_key_count == 0:
        return {
            "code": "partial",
            "label": "Partially configured",
            "configured": False,
            "details": "Some hosting files exist, but no admin key is registered yet.",
        }
    return {
        "code": "configured",
        "label": "Configured",
        "configured": True,
        "details": "Hosting access files and at least one admin key were detected.",
    }


def _admin_key_metadata(keys_file: Path, admin_key_id: str) -> Dict[str, Any]:
    payload = _read_json(keys_file, {"keys": {}})
    row = dict(dict(payload.get("keys") or {}).get(str(admin_key_id or "").strip()) or {})
    if not row:
        return {}
    key_origin = str(row.get("key_origin") or row.get("key_source") or "imported").strip().lower()
    public_key_source = str(row.get("public_key_source") or key_origin or "unknown").strip()
    private_key_storage = str(row.get("private_key_storage") or "").strip()
    private_key_export_path = str(row.get("private_key_export_path") or "").strip()
    warning = str(row.get("private_key_warning") or "").strip()
    if not private_key_storage:
        if str(row.get("private_key_openssh") or "").strip():
            private_key_storage = "embedded_keyring"
        elif key_origin == "generated":
            private_key_storage = "unknown_generated_location"
        else:
            private_key_storage = "not_managed"
    export_exists = bool(private_key_export_path and Path(private_key_export_path).exists())
    if private_key_storage == "embedded_keyring" and not warning:
        warning = "Generated private key is still embedded in keys.json; export/move it or rotate it."
    if private_key_storage == "exported_file" and private_key_export_path and not export_exists:
        warning = f"Expected exported private key file is missing: {private_key_export_path}"
    return {
        "key_origin": key_origin,
        "public_key_source": public_key_source,
        "private_key_storage": private_key_storage,
        "private_key_export_path": private_key_export_path or None,
        "private_key_export_exists": export_exists if private_key_export_path else None,
        "private_key_warning": warning or None,
    }


def _print_current_probe(summary: Dict[str, Any], probe: Dict[str, Any], state: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Current config snapshot")
    _kv_rows(
        [
            ("status", state.get("label")),
            ("configured", _status_text(bool(state.get("configured")))),
            ("connectivity_mode", summary.get("connectivity_mode")),
            ("endpoint_mode_default", summary.get("endpoint_mode_default")),
            ("lifecycle_profile", summary.get("lifecycle_profile")),
            ("require_auth", _status_text(bool(summary.get("require_auth")))),
            ("inferred_admin_key_id", summary.get("admin_key_id")),
            ("admin_key_entries", summary.get("admin_key_count")),
        ]
    )
    _print_rule("-")
    _print_title("Config probes")
    _kv_rows(
        [
            ("control_state_file", probe.get("control_state_path")),
            ("access_control_present", _status_text(bool(probe.get("access_exists")))),
            ("keys_present", _status_text(bool(probe.get("keys_exists")))),
            ("client_map_present", _status_text(bool(probe.get("mapping_exists")))),
            ("bootstrap_state_present", _status_text(bool(probe.get("bootstrap_exists")))),
            ("setup_audit_present", _status_text(bool(probe.get("audit_exists")))),
            ("bootstrap_admin_key_id", probe.get("bootstrap_admin_key_id") or "n/a"),
        ]
    )
    admin_ids = ", ".join(list(probe.get("admin_key_ids") or [])) or "none"
    _kv_rows([("admin_key_ids", admin_ids), ("client_mapping_rows", probe.get("client_count"))])
    if str(probe.get("setup_scope") or "").strip():
        _kv_rows([("previous_setup_scope", probe.get("setup_scope"))])
    if str(probe.get("setup_key_action") or "").strip():
        _kv_rows([("previous_key_action", probe.get("setup_key_action"))])
    if str(probe.get("setup_permission_action") or "").strip():
        _kv_rows([("previous_permission_action", probe.get("setup_permission_action"))])
    _print_rule("=")


def _print_wizard_home(summary: Dict[str, Any], probe: Dict[str, Any], state: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("=== Hosting Access Wizard ===")
    _kv_rows(
        [
            ("status", state.get("label")),
            ("summary", state.get("details")),
            ("hosting_root", probe.get("hosting_root_path")),
            ("connectivity_mode", summary.get("connectivity_mode")),
            ("endpoint_mode", summary.get("endpoint_mode_default")),
            ("lifecycle_profile", summary.get("lifecycle_profile")),
        ]
    )
    if bool(state.get("configured")):
        _kv_rows(
            [
                ("admin_key", summary.get("admin_key_id")),
                ("require_auth", _status_text(bool(summary.get("require_auth")))),
            ]
        )
    else:
        _kv_rows([("admin_key", "not configured")])
    _print_rule("=")


def _print_doctor_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Doctor checks")
    for row in list(result.get("checks") or []):
        ok = bool(row.get("ok"))
        status = _c("good", "ok") if ok else _c("bad", "issue")
        details = dict(row.get("details") or {})
        suffix = f" {json.dumps(details, ensure_ascii=False)}" if details else ""
        print(f"  [{status}] {_c('label', row.get('check'))}{_c('muted', suffix)}")
    _print_rule("-")
    _print_title("Doctor summary")
    _kv_rows([("status", result.get("status")), ("issues_count", result.get("issues_count"))])
    _print_rule("=")


def _resolve_import_source(
    *,
    interactive: bool,
    current_file: str,
    current_inline: str,
) -> Tuple[str, str]:
    public_key_file = str(current_file or "").strip()
    public_key_inline = str(current_inline or "").strip()
    if not interactive:
        return public_key_file, public_key_inline
    import_source_default = "file" if public_key_file else "inline" if public_key_inline else "file"
    print("\n[Group: Public key import]")
    cmd, import_source = _wizard_choice_prompt(
        title="Import source",
        valid=VALID_IMPORT_SOURCES,
        current=import_source_default,
        allow_skip=False,
    )
    if cmd == "next" and import_source == "file":
        _, value = _wizard_text_prompt(
            title="Admin public key file path",
            current=public_key_file or "<required>",
            allow_skip=False,
        )
        public_key_file = "" if value == "<required>" else str(value).strip()
        public_key_inline = ""
    elif cmd == "next":
        _, value = _wizard_text_prompt(
            title="Paste admin public key",
            current=public_key_inline or "<required>",
            allow_skip=False,
        )
        public_key_inline = "" if value == "<required>" else str(value).strip()
        public_key_file = ""
    return public_key_file, public_key_inline


def _print_intent_guidance(mode: str) -> None:
    _print_rule("-")
    g = dict(CONNECTIVITY_INTENT_GUIDANCE.get(mode) or {})
    _print_title(f"Intent `{mode}`")
    _kv_rows(
        [
            ("usage", str(g.get("intent") or "n/a")),
            ("value", str(g.get("provides") or "n/a")),
            ("precautions", str(g.get("precautions") or "n/a")),
        ]
    )


def _print_status_report(result: Dict[str, Any]) -> None:
    summary = dict(result.get("summary") or {})
    probe = dict(result.get("probe") or {})
    state = dict(result.get("state") or {})
    key_meta = dict(result.get("admin_key_metadata") or {})
    _print_wizard_home(summary, probe, state)
    _print_rule("-")
    rows: list[Tuple[str, Any]] = [
        ("control_state_file", result.get("control_state_file")),
        ("access_control_file", result.get("access_control_file")),
        ("keys_file", result.get("keys_file")),
        ("admin_key_count", summary.get("admin_key_count")),
    ]
    if key_meta:
        rows.extend(
            [
                ("admin_key_origin", key_meta.get("key_origin") or "unknown"),
                ("admin_public_key_source", key_meta.get("public_key_source") or "unknown"),
                ("admin_private_key_storage", key_meta.get("private_key_storage") or "unknown"),
            ]
        )
        if key_meta.get("private_key_export_path"):
            rows.append(("admin_private_key_path", key_meta.get("private_key_export_path")))
            rows.append(
                (
                    "admin_private_key_path_exists",
                    _status_text(bool(key_meta.get("private_key_export_exists"))),
                )
            )
        if key_meta.get("private_key_warning"):
            rows.append(("admin_key_warning", key_meta.get("private_key_warning")))
    _kv_rows(rows)


def _print_setup_result_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Resulting config")
    _kv_rows(
        [
            ("status", result.get("status")),
            ("connectivity_mode", result.get("connectivity_mode")),
            ("endpoint_mode_default", result.get("endpoint_mode_default")),
            ("lifecycle_profile", result.get("lifecycle_profile")),
            ("require_auth", _status_text(bool(result.get("require_auth")))),
            ("admin_key_id", result.get("admin_key_id")),
            ("admin_key_origin", result.get("admin_key_origin") or "unknown"),
            ("admin_public_key_source", result.get("admin_public_key_source") or "unknown"),
            ("admin_private_key_storage", result.get("admin_private_key_storage") or "unknown"),
            ("setup_scope", result.get("setup_scope")),
            ("key_action", result.get("key_action")),
            ("permission_action", result.get("permission_action")),
        ]
    )
    if result.get("admin_private_key_path"):
        _kv_rows([("admin_private_key_path", result.get("admin_private_key_path"))])
    if result.get("admin_private_key_warning"):
        _kv_rows([("admin_key_warning", result.get("admin_private_key_warning"))])
    _print_rule("-")
    _print_title("Changes applied")
    changes = list(result.get("changes") or [])
    if not changes:
        print(f"  {_c('muted', 'No config changes detected.')}")
    else:
        for item in changes:
            print(f"  {_c('accent', '-')} {_c('value', item)}")
    _print_rule("=")


def _print_key_list_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("RBAC keys")
    rows = list(result.get("keys") or [])
    if not rows:
        print(f"  {_c('muted', 'No keys configured.')}")
        _print_rule("=")
        return
    for row in rows:
        scopes: list[str] = []
        configs = list(row.get("allowed_configs") or [])
        engines = list(row.get("allowed_engines") or [])
        if configs:
            scopes.append(f"configs={','.join(configs)}")
        if engines:
            scopes.append(f"engines={','.join(engines)}")
        _kv_rows(
            [
                ("key_id", row.get("key_id")),
                ("role", row.get("role")),
                ("auth_method", row.get("auth_method")),
                ("disabled", _status_text(bool(row.get("disabled")))),
                ("scope", ", ".join(scopes) or "default"),
            ]
        )
        _print_rule("-")
    _print_rule("=")


def _print_key_change_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("RBAC change")
    _kv_rows(
        [
            ("action", result.get("action")),
            ("key_id", result.get("key_id")),
            ("role", result.get("role") or "n/a"),
            ("auth_method", result.get("auth_method") or "n/a"),
            ("disabled", _status_text(bool(result.get("disabled"))) if "disabled" in result else "n/a"),
        ]
    )
    if list(result.get("allowed_configs") or []):
        _kv_rows([("allowed_configs", ", ".join(list(result.get("allowed_configs") or [])))])
    if list(result.get("allowed_engines") or []):
        _kv_rows([("allowed_engines", ", ".join(list(result.get("allowed_engines") or [])))])
    if "revoked" in result:
        _kv_rows(
            [
                ("revoked", _status_text(bool(result.get("revoked")))),
                ("revoked_sessions", result.get("revoked_sessions")),
            ]
        )
    _print_rule("=")


def _print_sessions_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Active sessions")
    sessions = list(result.get("sessions") or [])
    if not sessions:
        print(f"  {_c('muted', 'No active sessions.')}")
        _print_rule("=")
        return
    for row in sessions:
        binding = dict(row.get("ssh_binding") or {})
        binding_text = ""
        if binding:
            binding_text = f"{binding.get('target') or ''} {binding.get('key_fingerprint') or ''}".strip()
        _kv_rows(
            [
                ("token_preview", row.get("token_preview")),
                ("key_id", row.get("key_id")),
                ("role", row.get("role")),
                ("scope", row.get("scope")),
                ("ttl_remaining_seconds", row.get("ttl_remaining_seconds")),
                ("allowed_configs", ", ".join(list(row.get("allowed_configs") or [])) or "default"),
                ("allowed_engines", ", ".join(list(row.get("allowed_engines") or [])) or "default"),
                ("ssh_binding", binding_text or "none"),
            ]
        )
        _print_rule("-")
    _kv_rows(
        [
            ("count", result.get("count")),
            ("sessions_count", result.get("sessions_count")),
            ("has_more", _status_text(bool(result.get("has_more")))),
        ]
    )
    _print_rule("=")


def _print_tokens_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Issued runtime tokens")
    tokens = list(result.get("tokens") or [])
    if not tokens:
        print(f"  {_c('muted', 'No issued runtime tokens.')}")
        _print_rule("=")
        return
    for row in tokens:
        identity = row.get("engine_id") or row.get("resource_key") or ""
        _kv_rows(
            [
                ("kind", row.get("kind")),
                ("token_preview", row.get("token_preview")),
                ("identity", identity),
                ("backend_id", row.get("backend_id") or "n/a"),
                ("issued_at", row.get("issued_at")),
            ]
        )
        _print_rule("-")
    _kv_rows(
        [
            ("count", result.get("count")),
            ("total_count", result.get("total_count")),
            ("has_more", _status_text(bool(result.get("has_more")))),
        ]
    )
    _print_rule("=")


def _print_audit_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Auth audit")
    events = list(result.get("events") or [])
    if not events:
        print(f"  {_c('muted', 'No auth audit events.')}")
        _print_rule("=")
        return
    for row in events:
        _kv_rows(
            [
                ("timestamp", row.get("timestamp")),
                ("event_type", row.get("event_type")),
                ("result", row.get("result")),
                ("actor_key_id", row.get("actor_key_id") or "n/a"),
                ("target_key_id", row.get("target_key_id") or "n/a"),
                ("target_token_preview", row.get("target_token_preview") or "n/a"),
            ]
        )
        details = dict(row.get("details") or {})
        if details:
            _kv_rows([("details", json.dumps(details, ensure_ascii=False, sort_keys=True))])
        _print_rule("-")
    _kv_rows(
        [
            ("count", result.get("count")),
            ("total_count", result.get("total_count")),
            ("has_more", _status_text(bool(result.get("has_more")))),
        ]
    )
    _print_rule("=")


def _apply_permission_hardening(paths: list[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"attempted": [], "errors": []}
    for p in paths:
        try:
            if p.exists():
                mode = 0o700 if p.is_dir() else 0o600
                p.chmod(mode)
                out["attempted"].append({"path": str(p), "mode": oct(mode)})
        except Exception as exc:
            out["errors"].append({"path": str(p), "error": str(exc)})
    return out


def _ensure_dirs(hosting_root: Path) -> Dict[str, Path]:
    paths = {
        "root": hosting_root,
        "keyring": hosting_root / "keyring",
        "audit": hosting_root / "audit",
        "state": hosting_root / "state",
        "bootstrap": hosting_root / "bootstrap",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _import_public_key(*, public_key_file: Optional[str], public_key_inline: Optional[str]) -> str:
    if public_key_inline:
        return str(public_key_inline).strip()
    if public_key_file:
        p = Path(public_key_file).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"public key file not found: {p}")
        return str(p.read_text(encoding="utf-8")).strip()
    raise ValueError("public key is required (provide --admin-public-key-file or --admin-public-key)")


def _generate_keypair(
    *,
    key_id: str,
    passphrase: Optional[str],
) -> Tuple[str, str]:
    def _run_ssh_keygen(dest_private: Path) -> None:
        cmd = [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-C",
            key_id,
            "-f",
            str(dest_private),
            "-N",
            str(passphrase or ""),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30.0)  # noqa: S603
        if int(proc.returncode) != 0:
            stderr = str(proc.stderr or "").strip()
            raise RuntimeError(f"ssh-keygen failed: {stderr or 'unknown error'}")

    private_text = ""
    public_text = ""
    tmpdir = Path(tempfile.mkdtemp(prefix="hosting_keygen_")).resolve()
    try:
        tmp_private = (tmpdir / f"{key_id}_ed25519").resolve()
        tmp_public = Path(str(tmp_private) + ".pub")
        _run_ssh_keygen(tmp_private)
        if not tmp_private.exists() or not tmp_public.exists():
            raise RuntimeError("ssh-keygen did not produce expected key files")
        private_text = str(tmp_private.read_text(encoding="utf-8")).strip()
        public_text = str(tmp_public.read_text(encoding="utf-8")).strip()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if not private_text or not public_text:
        raise RuntimeError("failed to generate importable key material")
    return private_text, public_text


def _build_access_control_payload(
    *,
    connectivity_mode: str,
    endpoint_mode: str,
    lifecycle_profile: str,
    require_auth: bool,
    admin_key_id: str,
    admin_key_origin: str,
) -> Dict[str, Any]:
    now = time.time()
    return {
        "version": 1,
        "updated_at": now,
        "access_profile": {"connectivity_mode": connectivity_mode},
        "endpoint_mode_default": endpoint_mode,
        "lifecycle_profile": lifecycle_profile,
        "require_auth": bool(require_auth),
        "bootstrap_admin_key_id": admin_key_id,
        "bootstrap_admin_key_origin": admin_key_origin,
    }


def _store_importable_key_record(
    *,
    keys_file: Path,
    key_id: str,
    role: str,
    auth_method: str,
    public_key: str,
    private_key_openssh: Optional[str] = None,
    key_source: Optional[str] = None,
    key_origin: Optional[str] = None,
    public_key_source: Optional[str] = None,
    private_key_storage: Optional[str] = None,
    private_key_export_path: Optional[str] = None,
    private_key_warning: Optional[str] = None,
) -> None:
    payload = _read_json(keys_file, {"version": 1, "keys": {}})
    keys = dict(payload.get("keys") or {})
    existing = dict(keys.get(key_id) or {})
    row = {
        "role": str(role or "").strip(),
        "auth_method": str(auth_method or "").strip(),
        "public_key": str(public_key or "").strip(),
    }
    if private_key_openssh:
        row["private_key_openssh"] = str(private_key_openssh).strip()
    if key_source:
        row["key_source"] = str(key_source).strip()
    if key_origin:
        row["key_origin"] = str(key_origin).strip()
    if public_key_source:
        row["public_key_source"] = str(public_key_source).strip()
    if private_key_storage:
        row["private_key_storage"] = str(private_key_storage).strip()
    if private_key_export_path:
        row["private_key_export_path"] = str(private_key_export_path).strip()
    if private_key_warning:
        row["private_key_warning"] = str(private_key_warning).strip()
    preserved = {
        str(k): v
        for k, v in existing.items()
        if str(k)
        not in {
            "role",
            "auth_method",
            "public_key",
            "private_key_openssh",
            "key_source",
            "key_origin",
            "public_key_source",
            "private_key_storage",
            "private_key_export_path",
            "private_key_warning",
        }
    }
    keys[str(key_id)] = preserved | row
    payload["version"] = 1
    payload["updated_at"] = time.time()
    payload["keys"] = keys
    _write_json(keys_file, payload)


def _safe_require_auth(
    *,
    connectivity_mode: str,
    endpoint_mode: str,
    requested: Optional[bool],
) -> bool:
    if requested is None:
        return True
    val = bool(requested)
    if val:
        return True
    # Safe-only profile for unauth mode.
    if connectivity_mode == "local_only" and endpoint_mode == "exclusive":
        return False
    raise ValueError(
        "require_auth=false is only allowed for local_only connectivity with exclusive endpoint mode"
    )


def _write_audit_event(audit_file: Path, event: Dict[str, Any]) -> None:
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event or {}), ensure_ascii=False)
    with audit_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _migrate_legacy_key_files(
    *,
    default_config_dir: Path,
    hosting_root: Path,
    audit_file: Path,
    migrations_file: Path,
) -> Dict[str, Any]:
    candidates = [
        (default_config_dir / "backend" / "host_auth_keys.json").resolve(),
        (default_config_dir / "backend" / "engine_host_auth_keys.json").resolve(),
        (hosting_root / "keys.json").resolve(),
    ]
    migrated: list[Dict[str, Any]] = []
    for source in candidates:
        if not source.exists() or not source.is_file():
            continue
        target = Path(str(source) + ".migrated")
        if target.exists():
            idx = 1
            while True:
                alt = Path(str(source) + f".migrated.{idx}")
                if not alt.exists():
                    target = alt
                    break
                idx += 1
        source.rename(target)
        evt = {
            "timestamp": time.time(),
            "event": "legacy_key_file_renamed",
            "source": str(source),
            "target": str(target),
        }
        migrated.append(evt)
        _write_audit_event(audit_file, evt)
    if migrated:
        payload = _read_json(migrations_file, {"version": 1, "migrations": []})
        rows = list(payload.get("migrations") or [])
        rows.extend(migrated)
        payload["version"] = 1
        payload["updated_at"] = time.time()
        payload["migrations"] = rows
        _write_json(migrations_file, payload)
    return {"migrated_count": len(migrated), "migrated": migrated}


def _resolve_paths(args: argparse.Namespace, *, create_dirs: bool = False) -> Dict[str, Path]:
    default_config_dir, default_control_state_path = _default_paths()
    if str(args.default_config_dir or "").strip():
        default_config_dir = Path(str(args.default_config_dir)).expanduser().resolve()
    hosting_root = _hosting_root(default_config_dir)
    control_state_path = (hosting_root / "access_control.json").resolve()
    if str(args.control_state_file or "").strip():
        control_state_path = Path(str(args.control_state_file)).expanduser().resolve()
    dirs = _ensure_dirs(hosting_root) if create_dirs else {
        "root": hosting_root,
        "keyring": hosting_root / "keyring",
        "audit": hosting_root / "audit",
        "state": hosting_root / "state",
        "bootstrap": hosting_root / "bootstrap",
    }
    return {
        "default_config_dir": default_config_dir,
        "control_state_path": control_state_path,
        "hosting_root": hosting_root,
        "access_file": dirs["root"] / "access_control.json",
        "keys_file": dirs["keyring"] / "keys.json",
        "mappings_file": dirs["bootstrap"] / "client_key_map.json",
        "bootstrap_state_file": dirs["bootstrap"] / "bootstrap_state.json",
        "audit_file": dirs["audit"] / "setup_audit.jsonl",
        "migrations_file": dirs["keyring"] / "migrations.json",
    }


def run_status(args: argparse.Namespace) -> Dict[str, Any]:
    paths = _resolve_paths(args, create_dirs=False)
    summary = _summarize_existing_config(
        control_state_path=paths["control_state_path"],
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
    )
    probe = _probe_current_files(
        control_state_path=paths["control_state_path"],
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
        mappings_file=paths["mappings_file"],
        bootstrap_state_file=paths["bootstrap_state_file"],
        audit_file=paths["audit_file"],
    )
    state = _classify_config_state(summary, probe)
    summary["exists"] = bool(state.get("configured"))
    key_meta = _admin_key_metadata(paths["keys_file"], str(summary.get("admin_key_id") or ""))
    return {
        "status": "ok",
        "state": state,
        "summary": summary,
        "probe": probe,
        "admin_key_metadata": key_meta,
        "control_state_file": str(paths["control_state_path"]),
        "access_control_file": str(paths["access_file"]),
        "keys_file": str(paths["keys_file"]),
    }


def run_rbac(args: argparse.Namespace) -> Dict[str, Any]:
    paths = _resolve_paths(args, create_dirs=False)
    svc = EngineHostService(control_state_file=paths["control_state_path"])
    if bool(args.list_keys):
        return {"status": "ok", "action": "list_keys", "keys": svc.auth_list_keys()}
    if bool(args.list_sessions):
        return {
            "status": "ok",
            "action": "list_sessions",
            **svc.auth_list_sessions(
                key_id=str(args.session_key_id or "").strip() or None,
                scope=str(args.session_scope or "").strip() or None,
                role=str(args.session_role or "").strip() or None,
                token_preview_contains=str(args.token_preview_contains or "").strip() or None,
                limit=int(args.limit or 100),
                offset=int(args.offset or 0),
            ),
        }
    if bool(args.list_issued_tokens):
        return {
            "status": "ok",
            "action": "list_issued_tokens",
            **svc.auth_list_issued_tokens(
                engine_id=str(args.engine_id or "").strip() or None,
                resource_kind=str(args.resource_kind or "").strip() or None,
                resource_id=str(args.resource_id or "").strip() or None,
                backend_id=str(args.backend_id or "").strip() or None,
                token_preview_contains=str(args.token_preview_contains or "").strip() or None,
                limit=int(args.limit or 100),
                offset=int(args.offset or 0),
            ),
        }
    if bool(args.list_auth_audit):
        return {
            "status": "ok",
            "action": "list_auth_audit",
            **svc.auth_list_audit_events(
                event_type=str(args.audit_event_type or "").strip() or None,
                actor_key_id=str(args.audit_actor_key_id or "").strip() or None,
                target_key_id=str(args.audit_target_key_id or "").strip() or None,
                result=str(args.audit_result or "").strip() or None,
                limit=int(args.limit or 100),
                offset=int(args.offset or 0),
            ),
        }
    if str(args.revoke_session or "").strip():
        out = svc.auth_revoke_session(str(args.revoke_session).strip())
        return {"status": "ok", "action": "revoke_session", **out}
    if str(args.revoke_key_id or "").strip():
        out = svc.auth_revoke_key(str(args.revoke_key_id).strip())
        return {"status": "ok", "action": "revoke_key", **out}
    if bool(args.upsert_key):
        key_id = str(args.key_id or args.admin_key_id or "").strip()
        if not key_id:
            raise ValueError("--upsert-key requires --key-id")
        role = str(args.key_role or "").strip().lower()
        if role not in VALID_AUTH_ROLES:
            raise ValueError(f"--key-role must be one of: {', '.join(sorted(VALID_AUTH_ROLES))}")
        auth_method = str(args.auth_method or "public_key").strip().lower()
        public_key = ""
        if auth_method == "public_key":
            public_key = _import_public_key(
                public_key_file=str(args.public_key_file or "").strip(),
                public_key_inline=str(args.public_key_inline or "").strip(),
            )
        key_secret = str(args.key_secret or "").strip()
        allowed_configs = _split_csv(str(args.allowed_configs or ""))
        allowed_engines = _split_csv(str(args.allowed_engines or ""))
        out = svc.auth_upsert_key(
            key_id=key_id,
            role=role,
            auth_method=auth_method,
            public_key=public_key,
            key_secret=key_secret,
            allowed_configs=allowed_configs or None,
            allowed_engines=allowed_engines or None,
            disabled=bool(args.disable_key),
        )
        return {"status": "ok", "action": "upsert_key", **out}
    raise ValueError("No RBAC action selected")


def run_setup(args: argparse.Namespace) -> Dict[str, Any]:
    paths = _resolve_paths(args, create_dirs=True)
    default_config_dir = paths["default_config_dir"]
    control_state_path = paths["control_state_path"]
    hosting_root = paths["hosting_root"]
    access_file = paths["access_file"]
    keys_file = paths["keys_file"]
    mappings_file = paths["mappings_file"]
    bootstrap_state_file = paths["bootstrap_state_file"]
    audit_file = paths["audit_file"]
    migrations_file = paths["migrations_file"]

    interactive = bool(args.interactive)
    mode = _normalize_mode(args.mode, "local_only")
    endpoint_mode = _normalize_endpoint_mode(args.endpoint_mode, "exclusive")
    lifecycle_profile = _normalize_lifecycle_profile(args.lifecycle_profile, "detached_user_process")
    key_source = str(args.key_source or "").strip().lower() or "import"
    if key_source not in VALID_KEY_SOURCES:
        key_source = "import"
    admin_key_id = str(args.admin_key_id or "").strip() or "admin-main"
    key_action = "replace"
    permission_action = "none"
    setup_scope = "fresh_setup"
    setup_notes: list[str] = []
    permission_result: Dict[str, Any] = {"attempted": [], "errors": []}
    admin_public_key_file_value = str(args.admin_public_key_file or "").strip()
    admin_public_key_inline_value = str(args.admin_public_key or "").strip()

    existing_summary = _summarize_existing_config(
        control_state_path=control_state_path,
        access_file=access_file,
        keys_file=keys_file,
    )
    before_summary = dict(existing_summary)
    current_probe = _probe_current_files(
        control_state_path=control_state_path,
        access_file=access_file,
        keys_file=keys_file,
        mappings_file=mappings_file,
        bootstrap_state_file=bootstrap_state_file,
        audit_file=audit_file,
    )
    config_state = _classify_config_state(existing_summary, current_probe)
    existing_summary["exists"] = bool(config_state.get("configured"))

    if interactive:
        assumed_intent = "local_only" if str(config_state.get("code")) == "clean" else _normalize_mode(
            existing_summary.get("connectivity_mode", mode),
            mode,
        )
        _print_wizard_home(existing_summary, current_probe, config_state)

        mode = assumed_intent
        endpoint_mode = _normalize_endpoint_mode(
            str(existing_summary.get("endpoint_mode_default") or endpoint_mode),
            endpoint_mode,
        )
        lifecycle_profile = _normalize_lifecycle_profile(
            str(existing_summary.get("lifecycle_profile") or lifecycle_profile),
            lifecycle_profile,
        )
        require_auth_seed = bool(existing_summary.get("require_auth", True))
        admin_key_id = str(existing_summary.get("admin_key_id") or admin_key_id)
        if bool(config_state.get("configured")):
            key_action = "keep_existing"

        while True:
            operator_choice = _prompt_menu(
                "\nMain menu:",
                {
                    "1": "Start guided setup",
                    "2": "Review detailed status",
                    "3": "Run diagnostics",
                    "4": "Exit without changes",
                },
                "1",
            )
            if operator_choice == "1":
                break
            if operator_choice == "2":
                _print_current_probe(existing_summary, current_probe, config_state)
                continue
            if operator_choice == "3":
                _print_doctor_report(run_doctor(args))
                continue
            if operator_choice == "4":
                raise RuntimeError("interactive setup cancelled by user")

        workflow_choice = _prompt_menu(
            "\nChoose workflow path:",
            {
                "1": "Keep current connectivity intent",
                "2": "Choose a different connectivity intent",
            },
            "1" if bool(config_state.get("configured")) else "2",
        )
        if workflow_choice == "2":
            setup_scope = "full_reconfigure_new_intent"
            print("\nConnectivity intent")
            for k in sorted(VALID_CONNECTIVITY_MODES):
                _print_intent_guidance(k)
            mode = _prompt_choice("Connectivity mode", VALID_CONNECTIVITY_MODES, mode)
            setup_notes.append("Full reconfigure selected: all grouped steps reviewed.")
        else:
            setup_scope = "adjust_within_intent"
            setup_notes.append("Within-intent adjustment selected.")

        grouped_steps = [
            "endpoint_mode",
            "lifecycle_profile",
            "require_auth",
            "key_action",
            "key_source",
            "admin_key_id",
            "permission_action",
        ]
        step_idx = 0
        current_require_auth = _safe_require_auth(
            connectivity_mode=mode,
            endpoint_mode=endpoint_mode,
            requested=args.require_auth if args.require_auth is not None else require_auth_seed,
        )
        print("\nConfiguration steps")
        print("Use Enter to keep the current value, `p` for previous, `s` to skip.")
        while step_idx < len(grouped_steps):
            step = grouped_steps[step_idx]
            if step == "endpoint_mode":
                print("\n[Access]")
                cmd, val = _wizard_choice_prompt(
                    title="Step 1: Endpoint mode",
                    valid=VALID_ENDPOINT_MODES,
                    current=endpoint_mode,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    endpoint_mode = val
                    current_require_auth = _safe_require_auth(
                        connectivity_mode=mode,
                        endpoint_mode=endpoint_mode,
                        requested=current_require_auth,
                    )
                step_idx += 1
                continue
            if step == "lifecycle_profile":
                cmd, val = _wizard_choice_prompt(
                    title="Step 2: Lifecycle profile",
                    valid=VALID_LIFECYCLE_PROFILES,
                    current=lifecycle_profile,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    lifecycle_profile = val
                step_idx += 1
                continue
            if step == "require_auth":
                print(
                    "Step 3: Require auth\n"
                    "  - value: protects multi-user and remote/tunnel workflows.\n"
                    "  - constraint: no-auth allowed only for local_only + exclusive."
                )
                cmd, val = _wizard_bool_prompt(
                    title="Enable require_auth?",
                    current=current_require_auth,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    current_require_auth = _safe_require_auth(
                        connectivity_mode=mode,
                        endpoint_mode=endpoint_mode,
                        requested=val,
                    )
                step_idx += 1
                continue
            if step == "key_action":
                print("\n[Keys]")
                cmd, val = _wizard_choice_prompt(
                    title="Step 4: Key handling action",
                    valid={"keep_existing", "replace"},
                    current=key_action,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    key_action = val
                step_idx += 1
                continue
            if step == "key_source":
                if key_action == "keep_existing":
                    step_idx += 1
                    continue
                cmd, val = _wizard_choice_prompt(
                    title="Step 5: Key source for replacement",
                    valid=VALID_KEY_SOURCES,
                    current=key_source,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    key_source = val
                step_idx += 1
                continue
            if step == "admin_key_id":
                cmd, val = _wizard_text_prompt(
                    title="Step 6: Admin key_id",
                    current=admin_key_id,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next" and str(val).strip():
                    admin_key_id = str(val).strip()
                    if key_action == "keep_existing":
                        existing_keys = dict(_read_json(keys_file, {"keys": {}}).get("keys") or {})
                        existing_row = dict(existing_keys.get(admin_key_id) or {})
                        if existing_row:
                            admin_public_key_file_value = ""
                            admin_public_key_inline_value = str(existing_row.get("public_key") or "").strip()
                step_idx += 1
                continue
            if step == "permission_action":
                print("\n[Permissions]")
                print("  - none: keep filesystem permissions unchanged")
                print("  - tighten: best-effort chmod on Hosting folders/files")
                cmd, val = _wizard_choice_prompt(
                    title="Step 7: Permission action",
                    valid={"none", "tighten"},
                    current=permission_action,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    permission_action = val
                step_idx += 1
                continue

        require_auth = current_require_auth
        if key_action != "keep_existing" and key_source == "import":
            admin_public_key_file_value, admin_public_key_inline_value = _resolve_import_source(
                interactive=interactive,
                current_file=admin_public_key_file_value,
                current_inline=admin_public_key_inline_value,
            )
        print("\nPlanned result:")
        print(f"  - workflow: {setup_scope}")
        print(f"  - connectivity_mode: {mode}")
        print(f"  - endpoint_mode_default: {endpoint_mode}")
        print(f"  - lifecycle_profile: {lifecycle_profile}")
        print(f"  - require_auth: {require_auth}")
        print(f"  - key_action: {key_action}")
        if key_action != "keep_existing":
            print(f"  - key_source: {key_source}")
            if key_source == "import":
                import_from = admin_public_key_file_value or "<inline public key>"
                print(f"  - import_source: {import_from}")
        print(f"  - admin_key_id: {admin_key_id}")
        print(f"  - permission_action: {permission_action}")
        _print_intent_guidance(mode)
        if not _bool_prompt("Apply this configuration now?", True):
            raise RuntimeError("interactive setup cancelled by user")
    else:
        require_auth = _safe_require_auth(
            connectivity_mode=mode,
            endpoint_mode=endpoint_mode,
            requested=args.require_auth,
        )

    migration_result = _migrate_legacy_key_files(
        default_config_dir=default_config_dir,
        hosting_root=hosting_root,
        audit_file=audit_file,
        migrations_file=migrations_file,
    )

    admin_public_key = ""
    admin_private_key_text: Optional[str] = None
    admin_public_key_path: Optional[Path] = None
    export_private = bool(args.export_private_key)
    export_private_path = (
        Path(str(args.export_private_key_path)).expanduser().resolve()
        if str(args.export_private_key_path or "").strip()
        else None
    )
    key_origin = "imported"
    public_key_source = "existing_keyring" if key_action == "keep_existing" else "inline"
    private_key_storage = "not_managed"
    private_key_warning: Optional[str] = None

    if key_action == "keep_existing":
        keyring_existing = _read_json(keys_file, {"keys": {}})
        existing_keys = dict(keyring_existing.get("keys") or {})
        row = dict(existing_keys.get(admin_key_id) or {})
        admin_public_key = str(row.get("public_key") or "").strip()
        if not admin_public_key:
            raise ValueError(
                f"key_action=keep_existing requested but key_id={admin_key_id} has no existing public key"
            )
        key_source = "import"
        key_origin = str(row.get("key_origin") or row.get("key_source") or "imported").strip().lower() or "imported"
        public_key_source = str(row.get("public_key_source") or "existing_keyring").strip() or "existing_keyring"
        private_key_storage = str(row.get("private_key_storage") or "").strip() or (
            "embedded_keyring" if str(row.get("private_key_openssh") or "").strip() else "not_managed"
        )
        private_key_warning = str(row.get("private_key_warning") or "").strip() or None
        export_private_path = (
            Path(str(row.get("private_key_export_path"))).expanduser().resolve()
            if str(row.get("private_key_export_path") or "").strip()
            else export_private_path
        )
    else:
        if key_source == "generate":
            key_origin = "generated"
            public_key_source = "generated"
            passphrase = str(args.generated_key_passphrase or "")
            if interactive and not args.generated_key_passphrase:
                if _bool_prompt("Protect generated private key with passphrase?", False):
                    passphrase = input("Passphrase: ")
            generated_private, generated_public = _generate_keypair(
                key_id=admin_key_id,
                passphrase=passphrase or None,
            )
            admin_private_key_text = generated_private
            admin_public_key = str(generated_public).strip()
            if interactive:
                export_private = _bool_prompt("Export generated private key for client use?", export_private)
            if export_private and export_private_path is not None:
                export_private_path.parent.mkdir(parents=True, exist_ok=True)
                export_private_path.write_text(str(generated_private), encoding="utf-8")
                private_key_storage = "exported_file"
            else:
                private_key_storage = "embedded_keyring"
                private_key_warning = (
                    "Generated private key remains embedded in hosting key metadata. "
                    "Export/move it to a managed location or rotate it."
                )
        else:
            key_origin = "imported"
            public_key_source = "file" if admin_public_key_file_value else "inline"
            admin_public_key = _import_public_key(
                public_key_file=admin_public_key_file_value,
                public_key_inline=admin_public_key_inline_value,
            )

    svc = EngineHostService(control_state_file=control_state_path)
    _ = svc.auth_upsert_key(
        key_id=admin_key_id,
        auth_method="public_key",
        public_key=admin_public_key,
        role="admin",
        disabled=False,
    )
    _ = svc.set_control_config(
        require_auth=require_auth,
        access_profile={"connectivity_mode": mode},
        endpoint_mode_default=endpoint_mode,
        lifecycle_profile=lifecycle_profile,
    )
    _store_importable_key_record(
        keys_file=keys_file,
        key_id=admin_key_id,
        role="admin",
        auth_method="public_key",
        public_key=admin_public_key,
        private_key_openssh=admin_private_key_text,
        key_source=key_source,
        key_origin=key_origin,
        public_key_source=public_key_source,
        private_key_storage=private_key_storage,
        private_key_export_path=str(export_private_path) if export_private_path else None,
        private_key_warning=private_key_warning,
    )

    if permission_action == "tighten":
        permission_result = _apply_permission_hardening(
            [
                dirs["root"],
                dirs["keyring"],
                dirs["audit"],
                dirs["state"],
                dirs["bootstrap"],
                access_file,
                keys_file,
                mappings_file,
                bootstrap_state_file,
                audit_file,
            ]
        )

    _write_json(
        mappings_file,
        {
            "version": 1,
            "updated_at": time.time(),
            "clients": [
                {
                    "client_id": "default-admin-client",
                    "key_id": admin_key_id,
                    "role": "admin",
                    "engine_host_session_scope": "control",
                    "engine_host_session_ttl_seconds": 900,
                    "connectivity_mode": mode,
                    "notes": [
                        "Set engine_host_key_id in client profile/config",
                        "Issue short-lived session token for runtime access",
                    ],
                }
            ],
        },
    )
    _write_json(
        bootstrap_state_file,
        {
            "version": 1,
            "updated_at": time.time(),
            "setup": {
                "setup_scope": setup_scope,
                "setup_notes": setup_notes,
                "connectivity_mode": mode,
                "endpoint_mode_default": endpoint_mode,
                "lifecycle_profile": lifecycle_profile,
                "require_auth": require_auth,
                "admin_key_id": admin_key_id,
                "key_source": key_source,
                "key_action": key_action,
                "permission_action": permission_action,
            },
            "files": {
                "control_state_file": str(control_state_path),
                "access_control_file": str(access_file),
                "keys_file": str(keys_file),
                "client_mapping_file": str(mappings_file),
                "audit_file": str(audit_file),
            },
            "legacy_migration": migration_result,
        },
    )
    _write_audit_event(
        audit_file,
        {
            "timestamp": time.time(),
            "event": "hosting_config_applied",
            "connectivity_mode": mode,
            "endpoint_mode_default": endpoint_mode,
            "lifecycle_profile": lifecycle_profile,
            "require_auth": require_auth,
            "admin_key_id": admin_key_id,
            "key_source": key_source,
        },
    )
    after_summary = _summarize_existing_config(
        control_state_path=control_state_path,
        access_file=access_file,
        keys_file=keys_file,
    )
    changes: list[str] = []
    tracked = [
        ("connectivity_mode", "connectivity_mode"),
        ("endpoint_mode_default", "endpoint_mode_default"),
        ("lifecycle_profile", "lifecycle_profile"),
        ("require_auth", "require_auth"),
        ("admin_key_id", "admin_key_id"),
        ("admin_key_count", "admin_key_count"),
    ]
    for key, label in tracked:
        before = before_summary.get(key)
        after = after_summary.get(key)
        if before != after:
            changes.append(f"{label}: {before!r} -> {after!r}")
    if bool(migration_result.get("migrated_count")):
        changes.append(f"legacy key files migrated: {migration_result.get('migrated_count')}")
    if bool(export_private and export_private_path):
        changes.append(f"generated private key exported to {export_private_path}")
    if permission_action == "tighten":
        changes.append("permission hardening attempted on hosting directories/files")
    return {
        "status": "ok",
        "hosting_root": str(hosting_root),
        "control_state_file": str(control_state_path),
        "access_control_file": str(access_file),
        "keys_file": str(keys_file),
        "client_mapping_file": str(mappings_file),
        "bootstrap_state_file": str(bootstrap_state_file),
        "connectivity_mode": mode,
        "endpoint_mode_default": endpoint_mode,
        "lifecycle_profile": lifecycle_profile,
        "require_auth": require_auth,
        "admin_key_id": admin_key_id,
        "legacy_migration": migration_result,
        "admin_public_key_path": str(admin_public_key_path) if admin_public_key_path else None,
        "admin_private_key_path": None,
        "private_key_exported": bool(export_private),
        "private_key_export_path": str(export_private_path) if export_private_path else None,
        "setup_scope": setup_scope,
        "key_action": key_action,
        "permission_action": permission_action,
        "permission_result": permission_result,
        "changes": changes,
        "admin_key_origin": key_origin,
        "admin_public_key_source": public_key_source,
        "admin_private_key_storage": private_key_storage,
        "admin_private_key_path": str(export_private_path) if export_private_path else None,
        "admin_private_key_warning": private_key_warning,
    }


def run_doctor(args: argparse.Namespace) -> Dict[str, Any]:
    paths = _resolve_paths(args, create_dirs=False)
    default_config_dir = paths["default_config_dir"]
    control_state_path = paths["control_state_path"]
    hosting_root = paths["hosting_root"]
    issues: list[Dict[str, Any]] = []
    checks: list[Dict[str, Any]] = []

    def _record(
        name: str,
        ok: bool,
        details: Optional[Dict[str, Any]] = None,
        *,
        blocking: bool = True,
    ) -> None:
        entry = {"check": name, "ok": bool(ok), "details": dict(details or {})}
        checks.append(entry)
        if (not ok) and bool(blocking):
            issues.append(entry)

    try:
        proc = subprocess.run(  # noqa: S603
            ["ssh-keygen", "-h"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        _record("ssh_dependency", proc.returncode in {0, 1})
    except Exception as exc:
        _record("ssh_dependency", False, {"error": str(exc)})

    _record("default_config_dir_exists", default_config_dir.exists(), {"path": str(default_config_dir)})
    _record("hosting_root_exists", hosting_root.exists(), {"path": str(hosting_root)})
    _record("control_state_exists", control_state_path.exists(), {"path": str(control_state_path)})

    # Write-check in hosting root if present.
    if hosting_root.exists():
        probe = hosting_root / ".doctor_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            _record("hosting_root_writable", True, {"path": str(hosting_root)})
        except Exception as exc:
            _record("hosting_root_writable", False, {"path": str(hosting_root), "error": str(exc)})
    else:
        _record("hosting_root_writable", False, {"path": str(hosting_root), "error": "missing_directory"})

    # Readiness probe for Windows/mapped-path keygen behavior.
    # This check is non-blocking for baseline setup because key import remains valid,
    # but it must be reviewed before rotation-heavy hardening work.
    key_probe_dir = (hosting_root / "keyring").resolve()
    key_probe_private = (key_probe_dir / ".doctor_keygen_probe_ed25519").resolve()
    key_probe_public = Path(str(key_probe_private) + ".pub")
    key_probe_details: Dict[str, Any] = {"path": str(key_probe_dir), "blocking": False}
    key_probe_ok = False
    try:
        key_probe_dir.mkdir(parents=True, exist_ok=True)
        key_probe_private.unlink(missing_ok=True)
        key_probe_public.unlink(missing_ok=True)
        probe = subprocess.run(  # noqa: S603
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-C",
                "doctor-probe",
                "-f",
                str(key_probe_private),
                "-N",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        key_probe_details["returncode"] = int(probe.returncode)
        if int(probe.returncode) == 0 and key_probe_private.exists() and key_probe_public.exists():
            key_probe_ok = True
        else:
            key_probe_details["stderr"] = str(probe.stderr or "").strip()
    except Exception as exc:
        key_probe_details["error"] = str(exc)
    finally:
        key_probe_private.unlink(missing_ok=True)
        key_probe_public.unlink(missing_ok=True)
    _record("ssh_keygen_host_path_probe", key_probe_ok, key_probe_details, blocking=False)

    try:
        svc = EngineHostService(control_state_file=control_state_path)
        cfg = svc.get_control_config()
        _record("control_config_readable", True, {"require_auth": bool(cfg.get("require_auth", False))})
        try:
            svc.assert_runtime_policy_safe()
            _record("runtime_policy_safe", True)
        except Exception as exc:
            _record("runtime_policy_safe", False, {"error": str(exc)})
    except Exception as exc:
        _record("control_config_readable", False, {"error": str(exc)})

    return {
        "status": "ok" if not issues else "issues_found",
        "issues_count": len(issues),
        "checks": checks,
        "issues": issues,
        "default_config_dir": str(default_config_dir),
        "hosting_root": str(hosting_root),
        "control_state_file": str(control_state_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Configure hosting access and keyring state")
    p.add_argument("--default-config-dir", default="", help="Override default config root directory")
    p.add_argument("--control-state-file", default="", help="Override engine host control state JSON path")
    p.add_argument(
        "--color-scheme",
        default="dark",
        choices=sorted(VALID_COLOR_SCHEMES),
        help="Terminal color scheme for interactive output",
    )
    p.add_argument("--status", action="store_true", help="Print current hosting access status and exit")
    p.add_argument("--doctor", action="store_true", help="Run diagnostics without mutating configuration")
    p.add_argument("--json-output", action="store_true", help="Also emit machine-readable JSON result")
    p.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=True,
        help="Run interactive setup wizard (default)",
    )
    p.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable wizard and use flags only",
    )
    p.add_argument("--mode", default="local_only", choices=sorted(VALID_CONNECTIVITY_MODES))
    p.add_argument("--endpoint-mode", default="exclusive", choices=sorted(VALID_ENDPOINT_MODES))
    p.add_argument("--lifecycle-profile", default="detached_user_process", choices=sorted(VALID_LIFECYCLE_PROFILES))
    p.add_argument(
        "--require-auth",
        dest="require_auth",
        action="store_true",
        default=None,
        help="Enable daemon auth requirement",
    )
    p.add_argument(
        "--no-require-auth",
        dest="require_auth",
        action="store_false",
        help="Disable daemon auth requirement (safe-profile only)",
    )
    p.add_argument("--list-keys", action="store_true", help="List configured RBAC keys and exit")
    p.add_argument("--list-sessions", action="store_true", help="List active auth sessions and exit")
    p.add_argument("--list-issued-tokens", action="store_true", help="List issued runtime tokens and exit")
    p.add_argument("--list-auth-audit", action="store_true", help="List auth audit events and exit")
    p.add_argument("--upsert-key", action="store_true", help="Create or update one RBAC key and exit")
    p.add_argument("--revoke-key-id", default="", help="Revoke one RBAC key_id and its sessions, then exit")
    p.add_argument("--revoke-session", default="", help="Revoke one session token and exit")
    p.add_argument("--key-id", default="", help="RBAC key_id for --upsert-key")
    p.add_argument("--key-role", default="", choices=sorted(VALID_AUTH_ROLES), help="RBAC role for --upsert-key")
    p.add_argument(
        "--auth-method",
        default="public_key",
        choices=["public_key", "shared_secret"],
        help="Authentication method for --upsert-key",
    )
    p.add_argument("--public-key-file", default="", help="Public key file for --upsert-key")
    p.add_argument("--public-key-inline", default="", help="Inline public key for --upsert-key")
    p.add_argument("--key-secret", default="", help="Shared secret for --upsert-key when auth-method=shared_secret")
    p.add_argument("--allowed-configs", default="", help="Comma-separated config selectors for config_editor keys")
    p.add_argument("--allowed-engines", default="", help="Comma-separated engine ids for traffic-capable keys")
    p.add_argument("--disable-key", action="store_true", default=False, help="Create/update the RBAC key as disabled")
    p.add_argument("--session-key-id", default="", help="Filter --list-sessions by key_id")
    p.add_argument("--session-scope", default="", help="Filter --list-sessions by scope")
    p.add_argument("--session-role", default="", help="Filter --list-sessions by role")
    p.add_argument("--token-preview-contains", default="", help="Filter session/token listings by token preview text")
    p.add_argument("--engine-id", default="", help="Filter --list-issued-tokens by engine id")
    p.add_argument("--resource-kind", default="", help="Filter --list-issued-tokens by resource kind")
    p.add_argument("--resource-id", default="", help="Filter --list-issued-tokens by resource id")
    p.add_argument("--backend-id", default="", help="Filter --list-issued-tokens by backend id")
    p.add_argument("--audit-event-type", default="", help="Filter --list-auth-audit by event type")
    p.add_argument("--audit-actor-key-id", default="", help="Filter --list-auth-audit by actor key id")
    p.add_argument("--audit-target-key-id", default="", help="Filter --list-auth-audit by target key id")
    p.add_argument("--audit-result", default="", help="Filter --list-auth-audit by result")
    p.add_argument("--limit", type=int, default=100, help="List command page size")
    p.add_argument("--offset", type=int, default=0, help="List command page offset")
    p.add_argument("--key-source", default="import", choices=sorted(VALID_KEY_SOURCES))
    p.add_argument("--admin-key-id", default="admin-main")
    p.add_argument("--admin-public-key-file", default="")
    p.add_argument("--admin-public-key", default="")
    p.add_argument(
        "--generated-key-passphrase",
        dest="generated_key_passphrase",
        default="",
        help="Passphrase for a newly generated private key when key-source=generate",
    )
    p.add_argument(
        "--key-passphrase",
        dest="generated_key_passphrase",
        default="",
        help="Deprecated alias for --generated-key-passphrase",
    )
    p.add_argument("--export-private-key", action="store_true", default=False)
    p.add_argument("--export-private-key-path", default="")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _set_color_scheme(str(args.color_scheme or "dark").strip().lower())
    try:
        if bool(args.status):
            result = run_status(args)
            _print_status_report(result)
        elif bool(args.doctor):
            result = run_doctor(args)
            _print_doctor_report(result)
        elif bool(
            args.list_keys
            or args.list_sessions
            or args.list_issued_tokens
            or args.list_auth_audit
            or args.upsert_key
            or str(args.revoke_key_id or "").strip()
            or str(args.revoke_session or "").strip()
        ):
            result = run_rbac(args)
            if str(result.get("action")) == "list_keys":
                _print_key_list_report(result)
            elif str(result.get("action")) == "list_sessions":
                _print_sessions_report(result)
            elif str(result.get("action")) == "list_issued_tokens":
                _print_tokens_report(result)
            elif str(result.get("action")) == "list_auth_audit":
                _print_audit_report(result)
            else:
                _print_key_change_report(result)
        else:
            result = run_setup(args)
            if not bool(args.interactive):
                _print_setup_result_report(result)
        if bool(args.json_output):
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        if bool(getattr(args, "json_output", False)):
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            _print_rule("=")
            _print_title("Error")
            _kv_rows([("message", str(exc))])
            _print_rule("=")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
