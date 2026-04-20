"""
Interactive/non-interactive hosting access setup and reconfiguration utility.

Usage examples:
  python -m hosting.hosting_config_cli
  py hosting_config.py
  python -m hosting.hosting_config_cli --no-interactive --mode local_only --key-source import --admin-key-id admin-main --admin-public-key-file C:\\keys\\admin.pub
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
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
    from hosting.service.host_service import EngineHostService, VALID_AUTH_ROLES
    from hosting.client_realm import (
        FileSecretStore,
        append_client_audit_event,
        discover_exported_private_keys,
        ensure_client_realm_dirs,
        get_default_client_realm_root,
        handoff_exported_private_key_file,
        normalize_pasted_private_key,
        purge_exported_private_key_file,
        read_client_access,
        read_client_profile,
        secret_record_path,
    )
    from hosting.transport_bootstrap import (
        DEFAULT_TRANSPORT_AUTHORIZED_KEY_COMMAND,
        _protect_openssh_private_key,
        import_transport_bootstrap_bundle,
        install_transport_authorized_key,
        make_transport_bootstrap_bundle,
        provision_client_ssh_artifacts,
        read_transport_bootstrap_bundle,
        validate_client_transport_profile,
        write_transport_bootstrap_bundle,
    )
else:
    from .service.host_service import EngineHostService, VALID_AUTH_ROLES
    from .client_realm import (
        FileSecretStore,
        append_client_audit_event,
        discover_exported_private_keys,
        ensure_client_realm_dirs,
        get_default_client_realm_root,
        handoff_exported_private_key_file,
        normalize_pasted_private_key,
        purge_exported_private_key_file,
        read_client_access,
        read_client_profile,
        secret_record_path,
    )
    from .transport_bootstrap import (
        DEFAULT_TRANSPORT_AUTHORIZED_KEY_COMMAND,
        _protect_openssh_private_key,
        import_transport_bootstrap_bundle,
        install_transport_authorized_key,
        make_transport_bootstrap_bundle,
        provision_client_ssh_artifacts,
        read_transport_bootstrap_bundle,
        validate_client_transport_profile,
        write_transport_bootstrap_bundle,
    )


VALID_CONNECTIVITY_MODES = {"local_only", "ssh_tunnel_only", "truly_remote"}
VALID_ENDPOINT_MODES = {"exclusive", "shared"}
VALID_USAGE_INTENTS = {"single_admin", "role_split", "multi_user"}
VALID_CONTEXT_CONSUMERS = {"local_experiment", "local_backend", "ssh_relay", "remote_backend"}
VALID_CONTEXT_LIFECYCLES = {"single_exclusive", "reconnect_shared"}
VALID_CONTEXT_CREDENTIALS = {"ssh_keys", "password_local", "no_auth_local"}
VALID_ADMIN_CAPABILITIES = {"no_admin_available", "admin_available_interactive", "admin_managed_externally"}
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
        "intent": "Same box/user account with no off-host clients.",
        "provides": "Lowest setup overhead. Optional no-auth is possible only in strict safe profile.",
        "script_checks": "No-auth is accepted only for local_only + exclusive; shared endpoints require auth.",
    },
    "ssh_tunnel_only": {
        "intent": "Remote operators connect through SSH transport bootstrap.",
        "provides": "Remote reachability through explicit SSH/session controls.",
        "script_checks": "Auth is required; SSH binding and role checks are enforced by hosting policy.",
    },
    "truly_remote": {
        "intent": "Persistent direct/proxied remote serving for multiple remote clients.",
        "provides": "Full remote operations with role separation and explicit ingress controls.",
        "script_checks": "Auth is required; role boundaries are enforced by hosting policy.",
    },
}

USAGE_INTENT_GUIDANCE: Dict[str, Dict[str, str]] = {
    "single_admin": {
        "label": "Single user, same as admin",
        "hint": "one operator; one key/password, or local no-auth when safe",
        "projection": "Projects to local-only, exclusive access, and minimal key management.",
    },
    "role_split": {
        "label": "Many roles",
        "hint": "separate admin and user access keys",
        "projection": "Projects to authenticated access; setup creates/administers bootstrap admin first.",
    },
    "multi_user": {
        "label": "Multi-user",
        "hint": "more users, granular roles, more keys/passwords",
        "projection": "Projects to authenticated shared access; users can be managed later by admin tooling/GUI.",
    },
}

OPTION_HINTS: Dict[str, str] = {
    "exclusive": "one controlled endpoint; safest for local/no-auth",
    "shared": "multiple clients may share access; auth required",
    "foreground_terminal_bound": "stops when the terminal/session ends",
    "detached_user_process": "continues under the current user account",
    "service_managed": "managed by service/supervisor integration",
    "local_only": "same box/user account; no off-host clients",
    "ssh_tunnel_only": "remote clients enter through SSH tunnel",
    "truly_remote": "direct/proxied remote clients; strongest policy needed",
    "keep_existing": "reuse the current registered key",
    "replace": "replace/register the setup key",
    "generate": "create a new keypair locally",
    "import": "use an existing public key",
    "none": "leave filesystem permissions unchanged",
    "tighten": "best-effort private permissions on hosting files",
    "yes": "require key/session authentication",
    "no": "only allowed for local_only + exclusive",
    "local_experiment": "skip setup; no access files are written, reset, or deleted",
    "local_backend": "hosting consumer runs on same box/user account",
    "ssh_relay": "consumer reaches hosting through SSH relay/tunnel; SSH keys required",
    "remote_backend": "consumer reaches hosting over direct/proxied remote network; SSH keys required",
    "single_exclusive": "consumer death/disconnect stops hosting daemon and all created children",
    "reconnect_shared": "daemon should survive consumer reconnects",
    "ssh_keys": "most secure baseline; private key can be passphrase-protected",
    "password_local": "local_only shared-secret convenience; cannot issue remote sessions",
    "no_auth_local": "only for local single-user exclusive access",
    "shared_secret": "local_only session issuance only; remote modes require public-key challenge",
    "public_key": "works for local and remote; required for SSH relay/truly remote",
    "no_admin_available": "use user-scoped SSH setup only",
    "admin_available_interactive": "can approve elevated setup prompts; password is not stored",
    "admin_managed_externally": "generate instructions for an administrator or infrastructure tool",
}

OPTION_ORDER: Dict[str, int] = {
    "local_experiment": 10,
    "local_backend": 20,
    "ssh_relay": 30,
    "remote_backend": 40,
    "single_exclusive": 10,
    "reconnect_shared": 20,
    "single_admin": 10,
    "role_split": 20,
    "multi_user": 30,
    "ssh_keys": 10,
    "password_local": 20,
    "no_auth_local": 30,
    "no_admin_available": 10,
    "admin_available_interactive": 20,
    "admin_managed_externally": 30,
    "local_only": 10,
    "ssh_tunnel_only": 20,
    "truly_remote": 30,
    "exclusive": 10,
    "shared": 20,
}


_COLOR_SCHEME = "dark"
_ANSI_ENABLED = False
_COLOR_TOKENS: Dict[str, str] = {}
_PENDING_STAGED_SETUP: Dict[str, Any] = {}


class UserCancelled(RuntimeError):
    """Raised when an interactive user intentionally exits the setup flow."""

    def __init__(self, message: str = "cancelled by user", *, via_keyboard: bool = False) -> None:
        super().__init__(message)
        self.via_keyboard = bool(via_keyboard)


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
        _COLOR_TOKENS = {k: "" for k in {"reset", "title", "label", "value", "muted", "good", "warn", "bad", "accent", "rule"}}
        return
    if _COLOR_SCHEME == "light":
        _COLOR_TOKENS = {
            "reset": "\033[0m",
            "title": "\033[1;34m",
            "label": "\033[1;30m",
            "value": "\033[0;30m",
            "muted": "\033[0;90m",
            "good": "\033[0;32m",
            "warn": "\033[0;35m",
            "bad": "\033[1;31m",
            "accent": "\033[0;36m",
            "rule": "\033[0;33m",
        }
    else:
        _COLOR_TOKENS = {
            "reset": "\033[0m",
            "title": "\033[1;96m",
            "label": "\033[1;37m",
            "value": "\033[0;97m",
            "muted": "\033[0;90m",
            "good": "\033[0;92m",
            "warn": "\033[0;95m",
            "bad": "\033[1;91m",
            "accent": "\033[0;96m",
            "rule": "\033[0;93m",
        }


def _c(kind: str, text: Any) -> str:
    raw = str(text)
    if not _ANSI_ENABLED:
        return raw
    return f"{_COLOR_TOKENS.get(kind, '')}{raw}{_COLOR_TOKENS.get('reset', '')}"


def _print_title(text: str) -> None:
    print(f"\n{_c('title', text)}")


def _print_rule(char: str = "-", width: int = 72) -> None:
    if char == "=":
        char = "."
    kind = "rule" if char in {"=", "."} else "muted"
    print(_c(kind, char * width))


def _print_block(title: str, *, kind: str = "accent", width: int = 78) -> None:
    label = title.strip()
    if not label:
        return
    print(f"\n{_c(kind, label)}")
    _print_rule(".", width=width)


def _recommended_action(summary: Dict[str, Any], state: Dict[str, Any]) -> str:
    code = str(state.get("code") or "").strip()
    if code == "clean":
        return "Press Enter to configure hosting access now, or press q to leave this machine unconfigured."
    if code == "missing_control_state":
        return "Press Enter to repair the partial setup, or choose reset later to archive partial access files."
    if code == "partial":
        return "Press Enter to finish setup and register an admin key."
    if code == "blocked_remote_bootstrap":
        return "Press Enter to add an admin key before enabling remote-capable access."
    if bool(state.get("configured")):
        mode = str(summary.get("connectivity_mode") or "local_only")
        return f"Configuration is usable. Run diagnostics after changing keys, profiles, or {mode} access settings."
    return "Run diagnostics, then press Enter from the main menu to repair setup."


def _format_state_banner(summary: Dict[str, Any], state: Dict[str, Any]) -> str:
    label = str(state.get("label") or "Unknown")
    details = str(state.get("details") or "").strip()
    return f"{label}. {details}" if details else label


def _kv_rows(rows: list[Tuple[str, Any]], *, indent: str = "  ", min_width: int = 24) -> None:
    width = max([min_width, *[len(str(label)) for label, _ in rows]])
    for label, value in rows:
        print(f"{indent}{_c('label', str(label).ljust(width))} : {_c('value', value)}")


def _print_recommendations(recommendations: list[str]) -> None:
    rows = [str(item).strip() for item in recommendations if str(item).strip()]
    if not rows:
        return
    _print_rule("-")
    _print_title("Recommended Next Actions")
    for idx, item in enumerate(rows, start=1):
        print(f"  {_badge('warn', str(idx))} {_c('value', item)}")


def _staged_setup_rows(
    *,
    setup_scope: str,
    usage_intent: str = "",
    mode: str,
    endpoint_mode: str,
    lifecycle_profile: str,
    require_auth: bool,
    key_action: str,
    key_source: str,
    admin_key_id: str,
    permission_action: str,
    admin_public_key_file: str = "",
    admin_public_key_inline: str = "",
) -> list[Tuple[str, Any]]:
    rows: list[Tuple[str, Any]] = [
        ("workflow", setup_scope),
        ("hosting_usage", _option_label(usage_intent) if usage_intent else "n/a"),
        ("connectivity_mode", mode),
        ("endpoint_mode_default", endpoint_mode),
        ("lifecycle_profile", lifecycle_profile),
        ("require_auth", "yes" if require_auth else "no"),
        ("key_action", key_action),
    ]
    if key_action != "keep_existing":
        rows.append(("key_source", key_source))
        if key_source == "import":
            rows.append(("import_source", admin_public_key_file or ("<inline public key>" if admin_public_key_inline else "<not provided>")))
    rows.extend(
        [
            ("admin_key_id", admin_key_id),
            ("permission_action", permission_action),
        ]
    )
    return rows


def _print_staged_setup(
    *,
    setup_scope: str,
    usage_intent: str = "",
    mode: str,
    endpoint_mode: str,
    lifecycle_profile: str,
    require_auth: bool,
    key_action: str,
    key_source: str,
    admin_key_id: str,
    permission_action: str,
    admin_public_key_file: str = "",
    admin_public_key_inline: str = "",
) -> None:
    _print_block("Staged Setup Changes", kind="warn")
    _kv_rows(
        _staged_setup_rows(
            setup_scope=setup_scope,
            usage_intent=usage_intent,
            mode=mode,
            endpoint_mode=endpoint_mode,
            lifecycle_profile=lifecycle_profile,
            require_auth=require_auth,
            key_action=key_action,
            key_source=key_source,
            admin_key_id=admin_key_id,
            permission_action=permission_action,
            admin_public_key_file=admin_public_key_file,
            admin_public_key_inline=admin_public_key_inline,
        )
    )


def _set_pending_staged_setup(**kwargs: Any) -> None:
    _PENDING_STAGED_SETUP.clear()
    _PENDING_STAGED_SETUP.update(dict(kwargs))


def _clear_pending_staged_setup() -> None:
    _PENDING_STAGED_SETUP.clear()


def _has_pending_staged_setup() -> bool:
    return bool(_PENDING_STAGED_SETUP)


def _print_pending_staged_setup() -> None:
    if not _PENDING_STAGED_SETUP:
        return
    _print_staged_setup(**_PENDING_STAGED_SETUP)


def _plain_yes_no(question: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        raw = input(question + suffix).strip().lower()
    except KeyboardInterrupt:
        return False
    if not raw:
        return bool(default)
    return raw in {"y", "yes", "1", "true"}


def _pending_staged_setup_args(args: argparse.Namespace) -> argparse.Namespace:
    staged = dict(_PENDING_STAGED_SETUP)
    save_args = argparse.Namespace(**vars(args))
    save_args.interactive = False
    save_args.setup_scope = str(staged.get("setup_scope") or "server")
    save_args.usage_intent = str(staged.get("usage_intent") or "")
    save_args.mode = str(staged.get("mode") or "local_only")
    save_args.endpoint_mode = str(staged.get("endpoint_mode") or "exclusive")
    save_args.lifecycle_profile = str(staged.get("lifecycle_profile") or "detached_user_process")
    save_args.require_auth = bool(staged.get("require_auth"))
    save_args.key_action = str(staged.get("key_action") or "replace")
    save_args.key_source = str(staged.get("key_source") or "generate")
    save_args.admin_key_id = str(staged.get("admin_key_id") or "admin-main")
    save_args.permission_action = str(staged.get("permission_action") or "none")
    save_args.admin_public_key_file = str(staged.get("admin_public_key_file") or "")
    save_args.admin_public_key = str(staged.get("admin_public_key_inline") or "")
    return save_args


def _print_staged_setup_dropped(*, via_keyboard: bool) -> None:
    _print_block("Cancelled", kind="warn")
    if _has_pending_staged_setup():
        _print_pending_staged_setup()
        _kv_rows([("result", "Staged setup changes were dropped.")])
    else:
        _kv_rows([("result", "No further action requested.")])
    change_note = "Ctrl+C quits immediately and does not apply staged setup changes." if via_keyboard else "No setup changes were written."
    _kv_rows([("changes", change_note)])


def _save_pending_staged_setup(args: argparse.Namespace) -> Dict[str, Any]:
    save_args = _pending_staged_setup_args(args)
    _clear_pending_staged_setup()
    return run_setup(save_args)


def _rbac_action_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    action_args = argparse.Namespace(**vars(args))
    for flag in (
        "list_keys",
        "list_sessions",
        "list_issued_tokens",
        "list_auth_audit",
        "upsert_key",
    ):
        setattr(action_args, flag, False)
    action_args.revoke_key_id = ""
    action_args.revoke_session = ""
    for key, value in overrides.items():
        setattr(action_args, key, value)
    return action_args


def _status_text(value: bool) -> str:
    return _c("good", "yes") if value else _c("muted", "no")


def _badge(kind: str, text: str) -> str:
    palette = {
        "ok": "good",
        "issue": "bad",
        "warn": "warn",
        "info": "accent",
    }
    return _c(palette.get(kind, "value"), f"[{text}]")


def _compact_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _doctor_guidance(name: str, ok: bool, details: Dict[str, Any]) -> Dict[str, str]:
    error = str(details.get("error") or "").strip()
    if name == "ssh_dependency":
        if ok:
            return {
                "impact": "OpenSSH key tooling is reachable for key generation and public-key derivation.",
                "recommendation": "No action needed.",
            }
        return {
            "root_cause": error or "The `ssh-keygen` executable was not reachable or did not respond.",
            "impact": "Generated-key setup and key import helpers may fail or hang.",
            "recommendation": "Install OpenSSH client tools, ensure `ssh-keygen` is on PATH, then rerun diagnostics.",
        }
    if name == "control_state_exists":
        path = str(details.get("path") or "").strip()
        access_artifacts_present = bool(details.get("access_artifacts_present"))
        if ok:
            return {
                "impact": "Hosting access control state exists.",
                "recommendation": "No action needed.",
            }
        if not access_artifacts_present:
            return {
                "root_cause": f"Hosting access is not configured; expected control state file is missing: {path}",
                "impact": "This is expected before first setup, or when this machine is intentionally left unconfigured.",
                "recommendation": "Use `Configure hosting now` to create access files, or quit the wizard to leave hosting unchanged.",
            }
        return {
            "root_cause": f"Hosting access is partially configured; expected control state file is missing: {path}",
            "impact": "Existing key/bootstrap/audit artifacts may be ignored, and authenticated access is not reliable.",
            "recommendation": "Use `Configure hosting now` to repair access, or choose `Reset to unconfigured` to archive partial access files.",
        }
    if name == "hosting_root_exists":
        if ok:
            return {"impact": "Hosting root directory exists.", "recommendation": "No action needed."}
        return {
            "root_cause": "The hosting configuration directory does not exist yet.",
            "impact": "No access/keyring/audit state can be found.",
            "recommendation": "Run guided setup to create the hosting directory and initial access files.",
        }
    if name == "hosting_root_writable":
        if ok:
            return {"impact": "Hosting root is writable by the current process.", "recommendation": "No action needed."}
        if error == "missing_directory":
            return {
                "root_cause": "The hosting root does not exist yet, so writability could not be checked directly.",
                "impact": "This is expected before first setup, but setup must be able to create the directory.",
                "recommendation": "Run guided setup. If directory creation fails, choose a writable `--default-config-dir`.",
            }
        return {
            "root_cause": error or "The current process cannot write to the hosting root.",
            "impact": "Setup, migration, audit, and key storage updates may fail.",
            "recommendation": "Fix filesystem permissions or choose a writable `--default-config-dir`.",
        }
    if name == "zero_key_remote_bootstrap_policy":
        if ok:
            return {"impact": "Zero-key bootstrap policy is safe for the current connectivity mode.", "recommendation": "No action needed."}
        return {
            "root_cause": str(details.get("error") or "Remote-capable mode has auth enabled but no configured keys."),
            "impact": "Remote clients cannot authenticate and zero-key bootstrap is intentionally denied remotely.",
            "recommendation": "Provision an admin public key locally before enabling remote-capable connectivity.",
        }
    if name == "runtime_policy_safe":
        if ok:
            return {"impact": "Current runtime auth policy passes safety checks.", "recommendation": "No action needed."}
        return {
            "root_cause": error or "Runtime auth policy violates hosting safety constraints.",
            "impact": "Daemon startup or control-config updates may be denied.",
            "recommendation": "Use guided setup to restore a safe profile, or inspect `access_control.json` before retrying.",
        }
    if name == "admin_client_secret_encrypted":
        encryption = str(details.get("encryption") or "unknown")
        if ok:
            return {
                "impact": "Generated client private key is stored in encrypted client-realm form.",
                "recommendation": "Keep the client secret password in user custody; it is not persisted by hosting.",
            }
        return {
            "root_cause": f"Client-realm private key secret is stored with encryption={encryption}.",
            "impact": "A local file disclosure can reveal the private key more directly.",
            "recommendation": "Recreate or re-import the client key with `--client-secret-password`, or migrate to encrypted storage.",
        }
    if name == "admin_client_secret_present":
        if ok:
            return {"impact": "Referenced admin client secret record exists.", "recommendation": "No action needed."}
        return {
            "root_cause": "Keyring metadata references a client-realm secret file that is missing.",
            "impact": "The client may be unable to materialize or use the generated private key.",
            "recommendation": "Restore the secret file, re-import the private key, or rotate the admin key.",
        }
    if name == "admin_exported_private_key_custody":
        export_exists = bool(details.get("exists"))
        export_purged = bool(details.get("purged_after_adoption"))
        export_purged_without_adoption = bool(details.get("purged_without_adoption"))
        if ok:
            return {
                "impact": "Exported private key file has been handed off into a client realm and marked purged.",
                "recommendation": "No action needed.",
            }
        if export_purged_without_adoption:
            return {
                "root_cause": "Generated admin private key export was purged without recorded client-realm hand-off.",
                "impact": "The generated admin private key may be unavailable if no other copy exists.",
                "recommendation": str(
                    details.get("recommendation")
                    or "Verify another private-key copy exists or rotate the admin key."
                ),
            }
        if export_exists:
            return {
                "root_cause": "A generated admin private key still exists as a loose exported file.",
                "impact": "That file can authenticate as the admin key if copied or exposed.",
                "recommendation": str(
                    details.get("recommendation")
                    or "Hand it off into the consumer client realm, then purge the loose exported key file."
                ),
            }
        if not export_purged:
            return {
                "root_cause": "Generated admin private key metadata points to an exported file that is missing.",
                "impact": "A consumer may not be able to import or use the generated admin key.",
                "recommendation": str(
                    details.get("recommendation")
                    or "Restore the exported file, import another private key into the consumer realm, or rotate the admin key."
                ),
            }
    if name == "client_transport_profiles_integrity":
        invalid = list(details.get("invalid_profiles") or [])
        if ok:
            return {"impact": "Client transport profiles reference existing key and host-pin files.", "recommendation": "No action needed."}
        return {
            "root_cause": f"{len(invalid)} client transport profile(s) reference missing or inconsistent files.",
            "impact": "Affected remote profiles may fail strict SSH validation or connection setup.",
            "recommendation": "Re-import the transport bootstrap bundle or repair the missing known_hosts/secret files.",
        }
    if name == "ssh_keygen_host_path_probe":
        if ok:
            return {"impact": "OpenSSH can write generated keys under the hosting keyring path.", "recommendation": "No action needed."}
        return {
            "root_cause": error or str(details.get("stderr") or "OpenSSH key generation failed in the hosting keyring path."),
            "impact": "Generated-key setup may fail on this filesystem/path.",
            "recommendation": "Use imported public keys or choose a config directory on a filesystem supported by OpenSSH.",
        }
    if ok:
        return {"impact": "Check passed.", "recommendation": "No action needed."}
    return {
        "root_cause": error or "The check failed.",
        "impact": "Hosting access setup may be incomplete or unsafe.",
        "recommendation": "Review the check details and rerun diagnostics after fixing the underlying condition.",
    }


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


def _client_realm_root(default_config_dir: Path, realm: str = "default") -> Path:
    return get_default_client_realm_root(default_config_dir=default_config_dir, realm=realm)


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


def _normalize_usage_intent(value: str, default: str = "single_admin") -> str:
    v = str(value or "").strip().lower()
    return v if v in VALID_USAGE_INTENTS else default


def _project_usage_intent(intent: str) -> Dict[str, Any]:
    if intent == "role_split":
        return {
            "mode": "ssh_tunnel_only",
            "endpoint_mode": "exclusive",
            "require_auth": True,
            "key_action": "replace",
            "permission_action": "tighten",
            "note": "Setup provisions the bootstrap admin key; add operator/user keys later from admin tooling.",
        }
    if intent == "multi_user":
        return {
            "mode": "truly_remote",
            "endpoint_mode": "shared",
            "require_auth": True,
            "key_action": "replace",
            "permission_action": "tighten",
            "note": "Setup enables shared authenticated access; manage additional users and roles after bootstrap.",
        }
    return {
        "mode": "local_only",
        "endpoint_mode": "exclusive",
        "require_auth": False,
        "key_action": "replace",
        "permission_action": "none",
        "note": "Same user is the operator/admin; local no-auth is allowed only for local_only + exclusive.",
    }


def _normalize_context_value(value: Any, valid: set[str], default: str) -> str:
    raw = str(value or "").strip()
    return raw if raw in valid else default


def _infer_setup_context_defaults(
    *,
    summary: Dict[str, Any],
    probe: Dict[str, Any],
    default_usage_intent: str,
) -> Dict[str, str]:
    setup_context = dict(probe.get("setup_context") or {})
    connectivity_mode = _normalize_mode(str(summary.get("connectivity_mode") or "local_only"), "local_only")
    endpoint_mode = _normalize_endpoint_mode(str(summary.get("endpoint_mode_default") or "exclusive"), "exclusive")
    lifecycle_profile = _normalize_lifecycle_profile(
        str(summary.get("lifecycle_profile") or "detached_user_process"),
        "detached_user_process",
    )
    require_auth = bool(summary.get("require_auth"))
    setup_scope = str(probe.get("setup_scope") or "").strip()
    usage_default = setup_scope if setup_scope in VALID_USAGE_INTENTS else default_usage_intent
    usage_default = _normalize_usage_intent(
        str(setup_context.get("access") or setup_context.get("usage_intent") or usage_default),
        default_usage_intent,
    )

    consumer_default = {
        "local_only": "local_backend",
        "ssh_tunnel_only": "ssh_relay",
        "truly_remote": "remote_backend",
    }.get(connectivity_mode, "local_backend")
    access_artifacts_present = any(
        bool(probe.get(name))
        for name in ("access_exists", "keys_exists", "mapping_exists", "bootstrap_exists", "audit_exists")
    ) or int(summary.get("admin_key_count") or 0) > 0
    if not access_artifacts_present:
        consumer_default = "local_experiment"
    lifecycle_default = (
        "reconnect_shared"
        if endpoint_mode == "shared" or lifecycle_profile in {"detached_user_process", "service_managed"}
        else "single_exclusive"
    )
    credentials_default = "ssh_keys" if require_auth else "no_auth_local"
    return {
        "consumer": _normalize_context_value(
            setup_context.get("consumer"),
            VALID_CONTEXT_CONSUMERS,
            consumer_default,
        ),
        "lifecycle": _normalize_context_value(
            setup_context.get("lifecycle"),
            VALID_CONTEXT_LIFECYCLES,
            lifecycle_default,
        ),
        "access": _normalize_usage_intent(str(setup_context.get("access") or usage_default), usage_default),
        "credentials": _normalize_context_value(
            setup_context.get("credentials"),
            VALID_CONTEXT_CREDENTIALS,
            credentials_default,
        ),
        "admin_capability": _normalize_context_value(
            setup_context.get("admin_capability"),
            VALID_ADMIN_CAPABILITIES,
            "no_admin_available",
        ),
    }


def _setup_context_from_config(
    *,
    base: Dict[str, str],
    usage_intent: str,
    connectivity_mode: str,
    endpoint_mode: str,
    lifecycle_profile: str,
    require_auth: bool,
    key_source: str,
) -> Dict[str, str]:
    mode = _normalize_mode(connectivity_mode, "local_only")
    endpoint = _normalize_endpoint_mode(endpoint_mode, "exclusive")
    lifecycle = _normalize_lifecycle_profile(lifecycle_profile, "detached_user_process")
    context = dict(base or {})
    context["consumer"] = {
        "local_only": "local_backend",
        "ssh_tunnel_only": "ssh_relay",
        "truly_remote": "remote_backend",
    }.get(mode, "local_backend")
    context["lifecycle"] = (
        "reconnect_shared"
        if endpoint == "shared" or lifecycle in {"detached_user_process", "service_managed"}
        else "single_exclusive"
    )
    context["access"] = _normalize_usage_intent(usage_intent, "single_admin")
    if not bool(require_auth):
        context["credentials"] = "no_auth_local"
    elif str(key_source or "").strip().lower() == "generate":
        context["credentials"] = "ssh_keys"
    else:
        context["credentials"] = _normalize_context_value(
            context.get("credentials"),
            VALID_CONTEXT_CREDENTIALS,
            "ssh_keys",
        )
        if mode != "local_only" and context["credentials"] != "ssh_keys":
            context["credentials"] = "ssh_keys"
    context["admin_capability"] = _normalize_context_value(
        context.get("admin_capability"),
        VALID_ADMIN_CAPABILITIES,
        "no_admin_available",
    )
    return context


def _collect_setup_context(default_usage_intent: str, defaults: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    _print_title("Hosting Usage Context")
    print(_c("muted", "Answer these first so setup can suggest a safe default configuration."))
    defaults = dict(defaults or {})
    context: Dict[str, str] = {}
    step = 0
    while step < 5:
        if step == 0:
            value = _prompt_choice(
                "Who consumes hosting?",
                VALID_CONTEXT_CONSUMERS,
                context.get("consumer", defaults.get("consumer", "local_experiment")),
                allow_back=True,
            )
            if value == "back":
                return {"action": "back"}
            context["consumer"] = value
            if value == "local_experiment":
                context["lifecycle"] = "single_exclusive"
                context["access"] = "single_admin"
                context["credentials"] = "no_auth_local"
                context["admin_capability"] = "no_admin_available"
                return context
            step += 1
            continue
        if step == 1:
            default = context.get(
                "lifecycle",
                defaults.get(
                    "lifecycle",
                    "single_exclusive" if context.get("consumer") == "local_backend" else "reconnect_shared",
                ),
            )
            value = _prompt_choice(
                "What should happen when the consumer disconnects?",
                VALID_CONTEXT_LIFECYCLES,
                default,
                allow_back=True,
            )
            if value == "back":
                step -= 1
                continue
            context["lifecycle"] = value
            step += 1
            continue
        if step == 2:
            value = _prompt_choice(
                "How many access roles/users are expected?",
                VALID_USAGE_INTENTS,
                context.get("access", defaults.get("access", default_usage_intent)),
                allow_back=True,
            )
            if value == "back":
                step -= 1
                continue
            context["access"] = value
            step += 1
            continue
        if step == 3:
            consumer = context.get("consumer", "local_backend")
            lifecycle = context.get("lifecycle", "single_exclusive")
            access = context.get("access", "single_admin")
            disabled_credentials: set[str] = set()
            if not (consumer == "local_backend" and lifecycle == "single_exclusive" and access == "single_admin"):
                disabled_credentials.add("no_auth_local")
            if consumer in {"ssh_relay", "remote_backend"}:
                disabled_credentials.add("password_local")
            credential_default = (
                "no_auth_local"
                if "no_auth_local" not in disabled_credentials
                else "ssh_keys"
            )
            credential_default = defaults.get("credentials", credential_default)
            if credential_default in disabled_credentials:
                credential_default = "ssh_keys"
            value = _prompt_choice(
                "Preferred credential style?",
                VALID_CONTEXT_CREDENTIALS,
                context.get("credentials", credential_default),
                disabled=disabled_credentials,
                allow_back=True,
            )
            if value == "back":
                step -= 1
                continue
            context["credentials"] = value
            if context.get("consumer") not in {"ssh_relay", "remote_backend"}:
                context["admin_capability"] = "no_admin_available"
                break
            step += 1
            continue
        if step == 4:
            value = _prompt_choice(
                "Can setup perform administrator/root changes on the target host?",
                VALID_ADMIN_CAPABILITIES,
                context.get("admin_capability", defaults.get("admin_capability", "no_admin_available")),
                allow_back=True,
            )
            if value == "back":
                step -= 1
                continue
            context["admin_capability"] = value
            step += 1
    return {
        "consumer": context.get("consumer", "local_backend"),
        "lifecycle": context.get("lifecycle", defaults.get("lifecycle", "single_exclusive")),
        "access": context.get("access", defaults.get("access", default_usage_intent)),
        "credentials": context.get("credentials", defaults.get("credentials", "ssh_keys")),
        "admin_capability": context.get("admin_capability", defaults.get("admin_capability", "no_admin_available")),
    }


def _suggest_auto_configuration(context: Dict[str, str]) -> Dict[str, Any]:
    consumer = str(context.get("consumer") or "local_backend")
    lifecycle = str(context.get("lifecycle") or "single_exclusive")
    access = _normalize_usage_intent(context.get("access") or "single_admin")
    credentials = str(context.get("credentials") or "ssh_keys")
    admin_capability = str(context.get("admin_capability") or "no_admin_available")
    if admin_capability not in VALID_ADMIN_CAPABILITIES:
        admin_capability = "no_admin_available"

    if consumer == "local_experiment":
        return {
            "usage_intent": "single_admin",
            "mode": "local_only",
            "endpoint_mode": "exclusive",
            "require_auth": False,
            "key_source": "import",
            "key_action": "keep_existing",
            "permission_action": "none",
            "lifecycle_profile": "foreground_terminal_bound",
            "leave_unconfigured": True,
            "followups": [
                "Selecting this writes nothing and leaves any existing hosting access files unchanged.",
                "Choose a backend/SSH/remote consumer instead when you want this wizard to create access files.",
            ],
        }

    usage_intent = access
    mode = "local_only"
    if consumer == "ssh_relay":
        mode = "ssh_tunnel_only"
    elif consumer == "remote_backend":
        mode = "truly_remote"
    if mode != "local_only" and credentials in {"password_local", "no_auth_local"}:
        credentials = "ssh_keys"

    endpoint_mode = "shared" if lifecycle == "reconnect_shared" or access == "multi_user" or mode == "truly_remote" else "exclusive"
    require_auth = not (mode == "local_only" and endpoint_mode == "exclusive" and access == "single_admin" and credentials == "no_auth_local")
    key_source = "generate" if credentials in {"ssh_keys", "password_local"} else "import"
    permission_action = "tighten" if require_auth or mode != "local_only" else "none"
    lifecycle_profile = "detached_user_process" if endpoint_mode == "shared" else "foreground_terminal_bound"

    followups: list[str] = []
    if credentials == "password_local":
        followups.append("Shared-secret/password session issuance is local_only; ssh_tunnel_only/truly_remote require public-key challenge.")
    if key_source == "generate":
        followups.append(
            "If you do not export the generated private key now, setup stores it in this machine's default client realm and prints an export/import handoff command."
        )
    if mode in {"ssh_tunnel_only", "truly_remote"}:
        followups.append("Run SSH transport hardening to install a forced-command transport key, pin the host key, and validate strict SSH.")
        if admin_capability == "no_admin_available":
            followups.append("Use user-scoped SSH setup only; service, firewall, and machine-wide sshd changes require an administrator.")
        elif admin_capability == "admin_available_interactive":
            followups.append("Administrator/root changes can be offered through explicit elevated steps; setup must not store the password.")
        else:
            followups.append("Generate administrator instructions for SSH service, firewall, and daemon auto-start changes, then rerun diagnostics.")
    if access in {"role_split", "multi_user"}:
        followups.append("After bootstrap, add/edit user and role keys from the hosting consumer admin UI or RBAC tooling.")
    if endpoint_mode == "exclusive":
        followups.append("Exclusive mode stops hosting-created child processes when the single consumer disconnects.")
    else:
        followups.append("Shared mode keeps the detached hosting daemon alive so consumers can reconnect.")

    return {
        "usage_intent": usage_intent,
        "mode": mode,
        "endpoint_mode": endpoint_mode,
        "require_auth": require_auth,
        "key_source": key_source,
        "key_action": "replace",
        "permission_action": permission_action,
        "lifecycle_profile": lifecycle_profile,
        "admin_capability": admin_capability,
        "followups": followups,
    }


def _print_auto_configuration(context: Dict[str, str], suggestion: Dict[str, Any]) -> None:
    if bool(suggestion.get("leave_unconfigured")):
        _print_title("No Access Setup Selected")
        _kv_rows(
            [
                ("consumer", _option_label(str(context.get("consumer") or ""))),
                ("action", "leave hosting access configuration unchanged"),
                ("files", "no access files are written, reset, or deleted"),
            ]
        )
        followups = [str(item) for item in list(suggestion.get("followups") or []) if str(item).strip()]
        if followups:
            _print_recommendations(followups)
        return
    _print_title("Suggested Auto Configuration")
    _kv_rows(
        [
            ("consumer", _option_label(str(context.get("consumer") or ""))),
            ("consumer_lifecycle", _option_label(str(context.get("lifecycle") or ""))),
            ("access_model", _option_label(str(context.get("access") or ""))),
            ("credentials", _option_label(str(context.get("credentials") or ""))),
            ("admin_capability", _option_label(str(context.get("admin_capability") or "no_admin_available"))),
            ("usage_intent", _option_label(str(suggestion.get("usage_intent") or ""))),
            ("clients_connectivity", suggestion.get("mode")),
            ("endpoint_mode", suggestion.get("endpoint_mode")),
            ("lifecycle_profile", suggestion.get("lifecycle_profile")),
            ("require_auth", "yes" if bool(suggestion.get("require_auth")) else "no"),
            ("key_source", suggestion.get("key_source")),
            ("permission_action", suggestion.get("permission_action")),
        ]
    )
    followups = [str(item) for item in list(suggestion.get("followups") or []) if str(item).strip()]
    if followups:
        _print_recommendations(followups)


def _input_or_quit(prompt: str, *, lower: bool = False) -> str:
    try:
        raw = input(prompt).strip()
    except KeyboardInterrupt as exc:
        raise UserCancelled("cancelled by user", via_keyboard=True) from exc
    value = raw.lower() if lower else raw
    if value in {"q", "quit", "exit"}:
        raise UserCancelled("cancelled by user")
    return value


def _secret_input_or_quit(prompt: str) -> str:
    try:
        return getpass.getpass(prompt).strip()
    except KeyboardInterrupt as exc:
        raise UserCancelled("cancelled by user", via_keyboard=True) from exc


def _option_label(value: str) -> str:
    if value in USAGE_INTENT_GUIDANCE:
        return str(USAGE_INTENT_GUIDANCE[value].get("label") or value)
    labels = {
        "local_experiment": "Skip access setup for now",
        "local_backend": "Same box backend consumer",
        "ssh_relay": "SSH relay/tunnel consumer",
        "remote_backend": "Remote backend consumer",
        "single_exclusive": "Single exclusive consumer",
        "reconnect_shared": "Reconnectable/shared daemon",
        "ssh_keys": "SSH keys",
        "password_local": "Local password convenience",
        "no_auth_local": "No auth, local only",
        "no_admin_available": "No admin/root access",
        "admin_available_interactive": "Admin/root available",
        "admin_managed_externally": "Admin managed externally",
        "apply": "Use suggested configuration",
        "customize": "Customize configuration",
        "leave_unconfigured": "Leave hosting unchanged",
        "reset_unconfigured": "Reset to unconfigured",
    }
    if value in labels:
        return labels[value]
    return value


def _option_hint(value: str, explicit: str = "") -> str:
    if explicit:
        return str(explicit).strip()
    if value in USAGE_INTENT_GUIDANCE:
        return str(USAGE_INTENT_GUIDANCE[value].get("hint") or "").strip()
    return str(OPTION_HINTS.get(value) or "").strip()


def _ordered_options(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (OPTION_ORDER.get(value, 1000), value))


def _print_prompt_help(*, allow_back: bool = True, allow_changes: bool = True) -> None:
    controls = ["Enter=default/keep"]
    if allow_back:
        controls.append("b=back")
    if allow_changes:
        controls.append("c=changes")
    controls.append("q=quit")
    print(f"  {_c('muted', ' | '.join(controls))}")


def _print_options(
    options: list[Tuple[str, str, str]],
    *,
    default: str,
    label_width: int = 34,
    disabled: Optional[set[str]] = None,
) -> Dict[str, str]:
    index: Dict[str, str] = {}
    disabled_values = {str(value) for value in (disabled or set())}
    for idx, (value, label, hint) in enumerate(options, start=1):
        is_default = value == default
        is_disabled = value in disabled_values
        marker = f" {_c('rule', '(*)')}" if is_default else ""
        marker_width = 4 if is_default else 0
        label_text = f"{str(label).ljust(max(0, label_width - marker_width))}{marker}"
        number = f"{idx}."
        if not is_disabled:
            index[str(idx)] = value
            index[value.lower()] = value
        right = hint
        if is_disabled:
            right = f"{right} disabled/incompatible".strip()
        label_kind = "muted" if is_disabled else "value"
        print(f"  {_c('accent', number.ljust(4))} {_c(label_kind, label_text)} {_c('muted', right)}")
    return index


def _print_changes_or_empty() -> None:
    if _has_pending_staged_setup():
        _print_pending_staged_setup()
    else:
        _kv_rows([("staged_changes", "none")])


def _bool_prompt(question: str, default: bool) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = _input_or_quit(question + suffix, lower=True)
    if not raw:
        return bool(default)
    return raw in {"y", "yes", "1", "true"}


def _prompt_choice(
    question: str,
    valid: set[str],
    default: str,
    *,
    hints: Optional[Dict[str, str]] = None,
    disabled: Optional[set[str]] = None,
    allow_back: bool = False,
    allow_changes: bool = True,
) -> str:
    disabled_values = {str(value) for value in (disabled or set())}
    while True:
        _print_title(question)
        option_rows = [
            (value, _option_label(value), _option_hint(value, (hints or {}).get(value, "")))
            for value in _ordered_options(valid)
        ]
        index = _print_options(option_rows, default=default, disabled=disabled_values)
        _print_prompt_help(allow_back=allow_back, allow_changes=allow_changes)
        raw = _input_or_quit(_c("rule", f"Select [{_option_label(default)}]: "), lower=True)
        if raw == "c" and allow_changes:
            _print_changes_or_empty()
            continue
        if raw == "b" and allow_back:
            return "back"
        if not raw:
            return default
        if raw in index:
            return index[raw]
        print(f"  {_c('warn', 'invalid or disabled choice')} {_c('muted', raw)}")


def _prompt_menu(
    question: str,
    options: Dict[str, Any],
    default: str,
    *,
    allow_back: bool = False,
    allow_changes: bool = True,
) -> str:
    while True:
        _print_block(question.strip(": \n") or "Menu", kind="accent")
        normalized: list[Tuple[str, str, str]] = []
        for key, item in options.items():
            if isinstance(item, tuple):
                label = str(item[0])
                hint = str(item[1]) if len(item) > 1 else ""
            else:
                label = str(item)
                hint = ""
            normalized.append((str(key), label, hint))
        index = _print_options(normalized, default=default, label_width=30)
        _print_prompt_help(allow_back=allow_back, allow_changes=allow_changes)
        _print_rule(".", width=78)
        raw = _input_or_quit(_c("rule", f"Select [{_option_label(default)}]: "), lower=True)
        if raw == "c" and allow_changes:
            _print_changes_or_empty()
            return "changes"
        if raw == "b" and allow_back:
            return "back"
        if not raw:
            return default
        if raw in index:
            return index[raw]
        print(f"  {_c('warn', 'invalid choice')} {_c('muted', raw)}")


def _wizard_choice_prompt(
    *,
    title: str,
    valid: set[str],
    current: str,
    allow_skip: bool = True,
) -> Tuple[str, str]:
    while True:
        print(_c("title", title))
        option_rows = [(value, _option_label(value), _option_hint(value)) for value in _ordered_options(valid)]
        index = _print_options(option_rows, default=current)
        _print_prompt_help(allow_back=True, allow_changes=True)
        raw = _input_or_quit(f"  current={_option_label(current)}; select [{_option_label(current)}]: ", lower=True)
        if raw in {"p", "prev", "b", "back"}:
            return "prev", current
        if raw == "c":
            _print_changes_or_empty()
            continue
        if not raw:
            return "next", current
        if raw in index:
            return "next", index[raw]
        print(f"  {_c('warn', 'invalid choice')} {_c('muted', raw)}")


def _wizard_bool_prompt(*, title: str, current: bool, allow_skip: bool = True) -> Tuple[str, bool]:
    current_value = "yes" if current else "no"
    while True:
        print(_c("title", title))
        index = _print_options(
            [
                ("yes", "Yes", _option_hint("yes")),
                ("no", "No", _option_hint("no")),
            ],
            default=current_value,
        )
        _print_prompt_help(allow_back=True, allow_changes=True)
        raw = _input_or_quit(f"  current={_option_label(current_value)}; select [{_option_label(current_value)}]: ", lower=True)
        if raw in {"p", "prev", "b", "back"}:
            return "prev", current
        if raw == "c":
            _print_changes_or_empty()
            continue
        if not raw:
            return "next", current
        value = index.get(raw, raw)
        if value in {"yes", "y", "1", "true"}:
            return "next", True
        if value in {"no", "n", "0", "false"}:
            return "next", False
        print(f"  {_c('warn', 'invalid boolean')} {_c('muted', raw)}")


def _wizard_text_prompt(
    *,
    title: str,
    current: str,
    allow_skip: bool = True,
) -> Tuple[str, str]:
    _print_title(title)
    _print_prompt_help(allow_back=True, allow_changes=True)
    raw = _input_or_quit(f"  current={current}; enter value [{current}]: ")
    if raw.lower() in {"p", "prev"}:
        return "prev", current
    if raw.lower() in {"b", "back"}:
        return "prev", current
    if raw.lower() == "c":
        _print_changes_or_empty()
        return _wizard_text_prompt(title=title, current=current, allow_skip=allow_skip)
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
    bootstrap_context = dict(bootstrap_setup.get("setup_context") or {})
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
        "setup_context": bootstrap_context,
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
    connectivity_mode = _normalize_mode(str(summary.get("connectivity_mode") or "local_only"), "local_only")
    require_auth = bool(summary.get("require_auth"))
    if managed_files_present == 0 and admin_key_count == 0 and not bootstrap_admin_key_id:
        return {
            "code": "clean",
            "label": "Not configured yet",
            "configured": False,
            "details": "No hosting access files or admin keys were detected.",
        }
    if admin_key_count == 0 and require_auth and connectivity_mode != "local_only":
        return {
            "code": "blocked_remote_bootstrap",
            "label": "Blocked setup state",
            "configured": False,
            "details": "Remote-capable auth is enabled, but no admin key is provisioned yet. Pre-provision a key locally before remote use.",
        }
    if not bool(probe.get("access_exists")):
        return {
            "code": "missing_control_state",
            "label": "Partially configured",
            "configured": False,
            "details": "Admin keys or setup files were detected, but the access control state file is missing.",
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
    hosting_root = keys_file.parent.parent.resolve()
    default_config_dir = hosting_root.parent.resolve()
    secret_id = str(row.get("private_key_secret_id") or "").strip()
    secret_realm = str(row.get("private_key_secret_realm") or "default").strip() or "default"
    secret_path = None
    secret_exists = None
    secret_encryption = None
    secret_protection = str(row.get("private_key_protection") or "").strip() or None
    if secret_id:
        secret_path = secret_record_path(_client_realm_root(default_config_dir, secret_realm), secret_id)
        secret_exists = secret_path.exists()
        if secret_exists:
            try:
                secret_payload = json.loads(secret_path.read_text(encoding="utf-8"))
                secret_encryption = str(secret_payload.get("encryption") or "").strip() or None
                metadata = dict(secret_payload.get("metadata") or {})
                secret_protection = str(metadata.get("private_key_protection") or secret_protection or "").strip() or None
            except Exception:
                secret_encryption = None
    key_origin = str(row.get("key_origin") or row.get("key_source") or "imported").strip().lower()
    public_key_source = str(row.get("public_key_source") or key_origin or "unknown").strip()
    private_key_storage = str(row.get("private_key_storage") or "").strip()
    private_key_export_path = str(row.get("private_key_export_path") or "").strip()
    private_key_export_purged_at = row.get("private_key_export_purged_at")
    private_key_export_purged_without_adoption_at = row.get("private_key_export_purged_without_adoption_at")
    private_key_adopted_realm_root = str(row.get("private_key_adopted_client_realm_root") or "").strip()
    private_key_adopted_secret_id = str(row.get("private_key_adopted_secret_id") or "").strip()
    warning = str(row.get("private_key_warning") or "").strip()
    if not private_key_storage:
        if str(row.get("private_key_openssh") or "").strip():
            private_key_storage = "embedded_keyring"
        elif secret_id:
            private_key_storage = "client_realm_secret"
        elif key_origin == "generated":
            private_key_storage = "unknown_generated_location"
        else:
            private_key_storage = "not_managed"
    export_exists = bool(private_key_export_path and Path(private_key_export_path).exists())
    if private_key_storage == "embedded_keyring" and not warning:
        warning = "Generated private key is still embedded in keys.json; export/move it or rotate it."
    if (
        private_key_storage == "exported_file"
        and private_key_export_path
        and not export_exists
        and not private_key_export_purged_at
        and not private_key_export_purged_without_adoption_at
    ):
        warning = f"Expected exported private key file is missing: {private_key_export_path}"
    if private_key_storage == "client_realm_secret" and secret_id and not bool(secret_exists):
        warning = f"Expected client realm secret record is missing: {secret_path}"
    return {
        "key_origin": key_origin,
        "public_key_source": public_key_source,
        "private_key_storage": private_key_storage,
        "private_key_export_path": private_key_export_path or None,
        "private_key_export_exists": export_exists if private_key_export_path else None,
        "private_key_export_purged_at": private_key_export_purged_at,
        "private_key_export_purged_without_adoption_at": private_key_export_purged_without_adoption_at,
        "private_key_adopted_client_realm_root": private_key_adopted_realm_root or None,
        "private_key_adopted_secret_id": private_key_adopted_secret_id or None,
        "private_key_secret_id": secret_id or None,
        "private_key_secret_realm": secret_realm if secret_id else None,
        "private_key_secret_path": str(secret_path) if secret_path else None,
        "private_key_secret_exists": secret_exists if secret_id else None,
        "private_key_secret_encryption": secret_encryption if secret_id else None,
        "private_key_protection": secret_protection,
        "private_key_warning": warning or None,
    }


def _print_current_probe(
    summary: Dict[str, Any],
    probe: Dict[str, Any],
    state: Dict[str, Any],
    key_meta: Optional[Dict[str, Any]] = None,
) -> None:
    _print_block("Current Configuration", kind="accent")
    state_kind = "ok" if bool(state.get("configured")) else "warn"
    print(f"  {_badge(state_kind, str(state.get('label') or 'unknown'))} {_c('value', _format_state_banner(summary, state))}")
    _print_rule(".", width=78)
    _kv_rows(
        [
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
    _print_title("File Probes")
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
    key_meta = dict(key_meta or {})
    if key_meta:
        _print_rule("-")
        _print_title("Admin Key Provenance")
        rows: list[Tuple[str, Any]] = [
            ("admin_key_id", summary.get("admin_key_id")),
            ("admin_key_origin", key_meta.get("key_origin") or "unknown"),
            ("admin_public_key_source", key_meta.get("public_key_source") or "unknown"),
            ("admin_private_key_storage", key_meta.get("private_key_storage") or "unknown"),
        ]
        if key_meta.get("private_key_export_path"):
            rows.extend(
                [
                    ("admin_private_key_path", key_meta.get("private_key_export_path")),
                    ("admin_private_key_path_exists", _status_text(bool(key_meta.get("private_key_export_exists")))),
                ]
            )
        if key_meta.get("private_key_export_purged_at"):
            rows.append(("admin_private_key_export_purged", _status_text(True)))
        if key_meta.get("private_key_export_purged_without_adoption_at"):
            rows.append(("admin_private_key_export_purged_without_adoption", _status_text(True)))
        if key_meta.get("private_key_adopted_client_realm_root"):
            rows.append(("admin_private_key_adopted_realm_root", key_meta.get("private_key_adopted_client_realm_root")))
        if key_meta.get("private_key_secret_id"):
            rows.append(("admin_private_key_secret_id", key_meta.get("private_key_secret_id")))
        if key_meta.get("private_key_secret_path"):
            rows.extend(
                [
                    ("admin_private_key_secret_path", key_meta.get("private_key_secret_path")),
                    ("admin_private_key_secret_exists", _status_text(bool(key_meta.get("private_key_secret_exists")))),
                ]
            )
        if key_meta.get("private_key_secret_encryption"):
            rows.append(("admin_private_key_secret_encryption", key_meta.get("private_key_secret_encryption")))
        if key_meta.get("private_key_protection"):
            rows.append(("admin_private_key_protection", key_meta.get("private_key_protection")))
        if key_meta.get("private_key_warning"):
            rows.append(("admin_key_warning", key_meta.get("private_key_warning")))
        _kv_rows(rows)
    _print_recommendations([_recommended_action(summary, state)])
    _print_rule("=")


def _print_wizard_home(summary: Dict[str, Any], probe: Dict[str, Any], state: Dict[str, Any]) -> None:
    _print_block("Hosting Access Wizard", kind="accent")
    state_kind = "ok" if bool(state.get("configured")) else "warn"
    print(f"  {_badge(state_kind, str(state.get('label') or 'unknown'))} {_c('value', _format_state_banner(summary, state))}")
    _print_rule(".", width=78)
    _kv_rows(
        [
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
    elif int(summary.get("admin_key_count") or 0) > 0:
        _kv_rows(
            [
                ("admin_key", f"detected: {summary.get('admin_key_id')}"),
                ("admin_key_entries", summary.get("admin_key_count")),
                ("require_auth", _status_text(bool(summary.get("require_auth")))),
            ]
        )
    else:
        _kv_rows([("admin_key", "not configured")])
    _print_recommendations([_recommended_action(summary, state)])
    _print_rule("=", width=78)


def _print_doctor_report(result: Dict[str, Any]) -> None:
    _print_block("Hosting Doctor", kind="accent")
    checks = list(result.get("checks") or [])
    issues = [dict(row or {}) for row in checks if not bool((row or {}).get("ok")) and bool((row or {}).get("blocking", True))]
    warnings = [dict(row or {}) for row in checks if not bool((row or {}).get("ok")) and not bool((row or {}).get("blocking", True))]
    status_kind = "issue" if issues else "warn" if warnings else "ok"
    headline = "Must fix issues before relying on this hosting configuration." if issues else (
        "Review warnings before production use." if warnings else "No blocking issues or warnings detected."
    )
    print(f"  {_badge(status_kind, str(result.get('status') or 'unknown'))} {_c('value', headline)}")
    _kv_rows(
        [
            ("blocking_issues", len(issues)),
            ("warnings", len(warnings)),
            ("checks_total", len(checks)),
        ]
    )
    if not issues and not warnings:
        _print_rule(".", width=78)
    _print_rule("-")
    _print_title("All Checks")
    check_name_width = max([34, *[len(str((row or {}).get("check") or "")) for row in checks]])
    for row in checks:
        ok = bool(row.get("ok"))
        raw_status = "ok" if ok else ("issue" if bool(row.get("blocking", True)) else "warn")
        status = _c({"ok": "good", "issue": "bad", "warn": "warn"}.get(raw_status, "value"), f"[{raw_status}]".ljust(8))
        details = dict(row.get("details") or {})
        compact_details = {
            k: v
            for k, v in details.items()
            if k not in {"root_cause", "impact", "recommendation"}
        }
        suffix = f" {_compact_json(compact_details)}" if compact_details else ""
        check_name = _c("label", str(row.get("check") or "").ljust(check_name_width))
        print(f"  {status} {check_name} {_c('muted', suffix)}")
    _print_rule("-")
    _print_title("Doctor summary")
    _kv_rows(
        [
            ("status", result.get("status")),
            ("issues_count", result.get("issues_count")),
            ("warnings_count", len(warnings)),
        ]
    )
    for group_title, group_rows, group_kind in (
        ("Must Fix", issues, "issue"),
        ("Warnings To Review", warnings, "warn"),
    ):
        if not group_rows:
            continue
        _print_rule("-")
        _print_title(group_title)
        for row in group_rows:
            details = dict(row.get("details") or {})
            print(f"  {_badge(group_kind, group_kind)} {_c('label', str(row.get('check') or 'check'))}")
            for key in ("root_cause", "impact"):
                value = str(details.get(key) or "").strip()
                if value:
                    print(f"    {_c('label', key.ljust(14))} : {_c('value', value)}")
            compact_details = {
                k: v
                for k, v in details.items()
                if k not in {"root_cause", "impact", "recommendation"}
            }
            if compact_details:
                print(f"    {_c('label', 'details'.ljust(14))} : {_c('muted', _compact_json(compact_details))}")
    recommendations: list[str] = []
    for row in [*issues, *warnings]:
        rec = str(dict(row.get("details") or {}).get("recommendation") or "").strip()
        if rec and rec not in recommendations:
            recommendations.append(rec)
    _print_recommendations(recommendations)
    _print_rule("=", width=78)


def _doctor_followup_action(result: Dict[str, Any]) -> str:
    issues = [dict(row or {}) for row in list(result.get("issues") or [])]
    issue_names = {str(row.get("check") or "") for row in issues}
    if "control_state_exists" in issue_names or "hosting_root_exists" in issue_names:
        if _plain_yes_no("Start guided setup now?", True):
            return "setup"
    return ""


def _interactive_rbac_menu(args: argparse.Namespace) -> None:
    while True:
        action = _prompt_menu(
            "RBAC Key Management",
            {
                "list_keys": ("List RBAC keys", "show key ids, roles, auth methods, disabled state, and scopes"),
                "revoke_key": ("Revoke RBAC key", "remove one key id and revoke its sessions"),
                "list_exported_private_keys": ("List exported private keys", "show generated private-key files still tracked by setup"),
                "export_client_private_key": ("Export stored private key", "write a client-realm private key to a file for handoff"),
                "handoff_exported_private_key": ("Hand off exported private key", "store a local exported key in this client realm and optionally delete the file"),
                "purge_exported_private_key": ("Purge exported private key", "delete a tracked exported key file without importing it"),
                "list_sessions": ("List auth sessions", "show active session tokens by key id and role"),
                "list_auth_audit": ("List auth audit", "show recent RBAC/session audit events"),
            },
            "list_keys",
            allow_back=True,
        )
        if action == "back":
            return
        if action == "changes":
            continue
        if action == "list_keys":
            _print_key_list_report(run_rbac(_rbac_action_args(args, list_keys=True)))
            continue
        if action == "list_sessions":
            _print_sessions_report(run_rbac(_rbac_action_args(args, list_sessions=True)))
            continue
        if action == "list_auth_audit":
            _print_audit_report(run_rbac(_rbac_action_args(args, list_auth_audit=True)))
            continue
        if action == "list_exported_private_keys":
            _print_client_key_report(run_client_keys(_rbac_action_args(args, client_list_exported_keys=True)))
            continue
        if action == "export_client_private_key":
            keys_result = run_client_keys(_rbac_action_args(args, client_list_keys=True))
            _print_client_key_report(keys_result)
            key_ids = {
                str(key_id or "").strip()
                for key_id, row in dict(keys_result.get("keys") or {}).items()
                if str(key_id or "").strip() and str(dict(row or {}).get("private_key_secret_id") or "").strip()
            }
            if not key_ids:
                continue
            key_id = _input_or_quit("Client-realm private key id to export [blank=back]: ")
            if not key_id:
                continue
            if key_id not in key_ids:
                print(f"  {_c('warn', 'unknown key_id')} {_c('muted', key_id)}")
                continue
            default_export_path = Path(_resolve_paths(args, create_dirs=False)["hosting_root"]) / "keyring" / f"{key_id}.private"
            export_path_raw = _input_or_quit(f"Private key export path [{default_export_path}]: ")
            export_path = str(Path(export_path_raw).expanduser().resolve() if export_path_raw else default_export_path)
            _print_client_key_report(
                run_client_keys(
                    _rbac_action_args(
                        args,
                        client_export_key=True,
                        client_key_id=key_id,
                        client_export_key_path=export_path,
                    )
                )
            )
            _print_recommendations(
                ["Import this file into the target consumer realm, then delete the loose exported private-key file."]
            )
            continue
        if action == "handoff_exported_private_key":
            exported = run_client_keys(_rbac_action_args(args, client_list_exported_keys=True))
            _print_client_key_report(exported)
            rows = [
                dict(row or {})
                for row in list(exported.get("exported_keys") or [])
                if str((row or {}).get("key_id") or "").strip() and bool((row or {}).get("private_key_export_exists"))
            ]
            key_ids = {str(row.get("key_id") or "").strip() for row in rows}
            if not key_ids:
                continue
            key_id = _input_or_quit("Exported private key id to hand off [blank=back]: ")
            if not key_id:
                continue
            if key_id not in key_ids:
                print(f"  {_c('warn', 'unknown key_id')} {_c('muted', key_id)}")
                continue
            delete_source = _plain_yes_no("Delete the loose exported private-key file after hand-off?", True)
            handoff_args = _rbac_action_args(
                args,
                client_handoff_exported_key=True,
                client_key_id=key_id,
                client_delete_exported_key_file=delete_source,
            )
            _print_client_key_report(run_client_keys(handoff_args))
            continue
        if action == "purge_exported_private_key":
            exported = run_client_keys(_rbac_action_args(args, client_list_exported_keys=True))
            _print_client_key_report(exported)
            rows = [
                dict(row or {})
                for row in list(exported.get("exported_keys") or [])
                if str((row or {}).get("key_id") or "").strip() and bool((row or {}).get("private_key_export_exists"))
            ]
            key_ids = {str(row.get("key_id") or "").strip() for row in rows}
            if not key_ids:
                continue
            key_id = _input_or_quit("Exported private key id to purge [blank=back]: ")
            if not key_id:
                continue
            if key_id not in key_ids:
                print(f"  {_c('warn', 'unknown key_id')} {_c('muted', key_id)}")
                continue
            if not _plain_yes_no(
                "Purge this exported private-key file? This can lose the only private-key copy if it was not imported elsewhere.",
                False,
            ):
                continue
            _print_client_key_report(run_client_keys(_rbac_action_args(args, client_purge_exported_key=True, client_key_id=key_id)))
            continue
        if action == "revoke_key":
            keys_result = run_rbac(_rbac_action_args(args, list_keys=True))
            _print_key_list_report(keys_result)
            key_ids = {
                str(row.get("key_id") or "").strip()
                for row in list(keys_result.get("keys") or [])
                if str(row.get("key_id") or "").strip()
            }
            if not key_ids:
                continue
            key_id = _input_or_quit("RBAC key id to revoke [blank=back]: ")
            if not key_id:
                continue
            if key_id not in key_ids:
                print(f"  {_c('warn', 'unknown key_id')} {_c('muted', key_id)}")
                continue
            if not _plain_yes_no(f"Revoke RBAC key `{key_id}` and all of its sessions?", False):
                continue
            _print_key_change_report(run_rbac(_rbac_action_args(args, revoke_key_id=key_id)))


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


def _print_intent_guidance(mode: str, *, require_auth: bool, endpoint_mode: str) -> None:
    _print_rule("-")
    g = dict(CONNECTIVITY_INTENT_GUIDANCE.get(mode) or {})
    script_checks = str(g.get("script_checks") or "n/a")
    if mode == "local_only" and require_auth:
        script_checks = "Auth is enabled for this local profile; shared endpoints require auth."
    elif mode == "local_only" and endpoint_mode == "exclusive":
        script_checks = "No-auth is accepted only because this profile is local_only + exclusive."
    _print_title(f"Clients Connectivity `{mode}`")
    _kv_rows(
        [
            ("usage", str(g.get("intent") or "n/a")),
            ("value", str(g.get("provides") or "n/a")),
            ("script_checks", script_checks),
        ]
    )


def _print_status_report(result: Dict[str, Any]) -> None:
    summary = dict(result.get("summary") or {})
    probe = dict(result.get("probe") or {})
    state = dict(result.get("state") or {})
    key_meta = dict(result.get("admin_key_metadata") or {})
    _print_block("Hosting Status", kind="accent")
    state_kind = "ok" if bool(state.get("configured")) else "warn"
    print(f"  {_badge(state_kind, str(state.get('label') or 'unknown'))} {_c('value', _format_state_banner(summary, state))}")
    _print_rule(".", width=78)
    _kv_rows(
        [
            ("connectivity_mode", summary.get("connectivity_mode")),
            ("endpoint_mode", summary.get("endpoint_mode_default")),
            ("lifecycle_profile", summary.get("lifecycle_profile")),
            ("require_auth", _status_text(bool(summary.get("require_auth")))),
            (
                "admin_key",
                summary.get("admin_key_id") if int(summary.get("admin_key_count") or 0) > 0 else "not configured",
            ),
        ]
    )
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
        if key_meta.get("private_key_export_purged_at"):
            rows.append(("admin_private_key_export_purged", _status_text(True)))
        if key_meta.get("private_key_export_purged_without_adoption_at"):
            rows.append(("admin_private_key_export_purged_without_adoption", _status_text(True)))
        if key_meta.get("private_key_adopted_client_realm_root"):
            rows.append(("admin_private_key_adopted_realm_root", key_meta.get("private_key_adopted_client_realm_root")))
        if key_meta.get("private_key_secret_id"):
            rows.append(("admin_private_key_secret_id", key_meta.get("private_key_secret_id")))
        if key_meta.get("private_key_secret_path"):
            rows.append(("admin_private_key_secret_path", key_meta.get("private_key_secret_path")))
        if key_meta.get("private_key_secret_encryption"):
            rows.append(("admin_private_key_secret_encryption", key_meta.get("private_key_secret_encryption")))
        if key_meta.get("private_key_secret_path"):
            rows.append(
                (
                    "admin_private_key_secret_exists",
                    _status_text(bool(key_meta.get("private_key_secret_exists"))),
                )
            )
        if key_meta.get("private_key_warning"):
            rows.append(("admin_key_warning", key_meta.get("private_key_warning")))
    _kv_rows(rows)
    _print_recommendations([_recommended_action(summary, state)])


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
    if result.get("admin_private_key_secret_id"):
        _kv_rows([("admin_private_key_secret_id", result.get("admin_private_key_secret_id"))])
    if result.get("admin_private_key_secret_path"):
        _kv_rows([("admin_private_key_secret_path", result.get("admin_private_key_secret_path"))])
    if result.get("admin_private_key_secret_encryption"):
        _kv_rows([("admin_private_key_secret_encryption", result.get("admin_private_key_secret_encryption"))])
    if result.get("admin_private_key_protection"):
        _kv_rows([("admin_private_key_protection", result.get("admin_private_key_protection"))])
    if result.get("admin_private_key_export_command"):
        _kv_rows([("admin_private_key_export_command", result.get("admin_private_key_export_command"))])
    if result.get("admin_private_key_handoff"):
        _kv_rows([("admin_private_key_handoff", result.get("admin_private_key_handoff"))])
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


def _print_transport_bootstrap_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Transport bootstrap")
    rows: list[Tuple[str, Any]] = [
        ("status", result.get("status")),
        ("action", result.get("action")),
    ]
    for key in (
        "bundle_file",
        "profile_name",
        "profile_path",
        "known_hosts_file",
        "secret_id",
        "secret_path",
        "client_realm_root",
        "target",
        "transport_key_id",
        "ssh_alias",
        "ssh_config_file",
        "identity_file",
        "authorized_keys_file",
        "forced_command",
        "restrict_options",
        "rbac_key_id",
        "rbac_role",
        "admin_capability",
        "ssh_command",
        "validation_status",
        "ssh_probe_ran",
        "ssh_probe_returncode",
        "marker",
    ):
        if key in result and result.get(key) is not None:
            rows.append((key, result.get(key)))
    _kv_rows(rows)
    followups = [str(item).strip() for item in list(result.get("followups") or []) if str(item).strip()]
    if followups:
        _print_recommendations(followups)
    _print_rule("=")


def _print_admin_setup_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Admin setup")
    rows: list[Tuple[str, Any]] = [
        ("status", result.get("status")),
        ("action", result.get("action")),
        ("platform", result.get("platform")),
        ("execute", _status_text(bool(result.get("execute")))),
    ]
    for key in (
        "script_file",
        "elevation_method",
        "returncode",
        "ssh_service",
        "firewall",
        "user_linger",
    ):
        if key in result and result.get(key) is not None:
            rows.append((key, result.get(key)))
    _kv_rows(rows)
    if result.get("script"):
        _print_rule("-")
        print(str(result.get("script")))
    followups = [str(item).strip() for item in list(result.get("followups") or []) if str(item).strip()]
    if followups:
        _print_recommendations(followups)
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


def _print_client_key_report(result: Dict[str, Any]) -> None:
    _print_rule("=")
    _print_title("Client keys")
    action = str(result.get("action") or "")
    if action == "client_list_exported_keys":
        rows = list(result.get("exported_keys") or [])
        if not rows:
            print(f"  {_c('muted', 'No exported private key file references were found.')}")
        else:
            for row in rows:
                item = dict(row or {})
                _kv_rows(
                    [
                        ("key_id", item.get("key_id")),
                        ("role", item.get("role") or "n/a"),
                        ("private_key_path", item.get("private_key_export_path") or "n/a"),
                        ("private_key_path_exists", _status_text(bool(item.get("private_key_export_exists")))),
                    ]
                )
                if bool(item.get("private_key_export_exists")):
                    _print_recommendations(
                        [
                            "Hand this key off into the consumer client realm, then delete the loose exported private-key file."
                        ]
                    )
                _print_rule("-")
    elif action == "client_list_keys":
        rows = dict(result.get("keys") or {})
        if not rows:
            print(f"  {_c('muted', 'No client keys.')}")
        else:
            for key_id, row in sorted(rows.items()):
                item = dict(row or {})
                _kv_rows(
                    [
                        ("key_id", key_id),
                        ("role", item.get("role") or "n/a"),
                        ("public_key_source", item.get("public_key_source") or "unknown"),
                        ("private_key_storage", item.get("private_key_storage") or "unknown"),
                        ("private_key_secret_id", item.get("private_key_secret_id") or "n/a"),
                    ]
                )
                _print_rule("-")
    else:
        _kv_rows(
            [
                ("status", result.get("status")),
                ("action", result.get("action")),
                ("key_id", result.get("key_id")),
                ("tag", result.get("tag")),
                ("secret_id", result.get("secret_id") or "n/a"),
                ("secret_encryption", result.get("secret_encryption") or "n/a"),
                ("export_path", result.get("export_path") or "n/a"),
                ("source_export_path", result.get("source_export_path") or "n/a"),
                ("deleted_source_file", _status_text(bool(result.get("deleted_source_file")))),
            ]
        )
        if result.get("warning"):
            _print_recommendations([str(result.get("warning"))])
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


def _derive_public_key_from_private(private_key_text: str) -> str:
    tmpdir = Path(tempfile.mkdtemp(prefix="hosting_pubderive_")).resolve()
    try:
        tmp_private = (tmpdir / "derived_ed25519").resolve()
        tmp_private.write_text(str(private_key_text or "").strip() + "\n", encoding="utf-8")
        try:
            tmp_private.chmod(0o600)
        except Exception:
            pass
        proc = subprocess.run(  # noqa: S603
            ["ssh-keygen", "-y", "-f", str(tmp_private)],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if int(proc.returncode) != 0:
            raise RuntimeError(str(proc.stderr or "").strip() or "ssh-keygen -y failed")
        public_text = str(proc.stdout or "").strip()
        if not public_text:
            raise RuntimeError("ssh-keygen -y returned empty public key")
        return public_text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    private_key_secret_id: Optional[str] = None,
    private_key_secret_realm: Optional[str] = None,
    private_key_protection: Optional[str] = None,
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
    if private_key_secret_id:
        row["private_key_secret_id"] = str(private_key_secret_id).strip()
    if private_key_secret_realm:
        row["private_key_secret_realm"] = str(private_key_secret_realm).strip()
    if private_key_protection:
        row["private_key_protection"] = str(private_key_protection).strip()
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
            "private_key_secret_id",
            "private_key_secret_realm",
            "private_key_protection",
            "private_key_warning",
        }
    }
    keys[str(key_id)] = preserved | row
    payload["version"] = 1
    payload["updated_at"] = time.time()
    payload["keys"] = keys
    _write_json(keys_file, payload)


def _source_exported_keys_file(args: argparse.Namespace) -> Path:
    raw = str(getattr(args, "client_exported_keys_file", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    paths = _resolve_paths(args, create_dirs=False)
    return Path(paths["keys_file"]).expanduser().resolve()


def _mark_exported_key_adopted(
    *,
    keys_file: Path,
    key_id: str,
    client_realm_root: Path,
    secret_id: str,
    delete_source_file: bool,
) -> None:
    payload = _read_json(keys_file, {"version": 1, "keys": {}})
    keys = dict(payload.get("keys") or {})
    row = dict(keys.get(key_id) or {})
    if not row:
        return
    row["private_key_adopted_client_realm_root"] = str(client_realm_root)
    row["private_key_adopted_secret_id"] = str(secret_id)
    row["private_key_adopted_at"] = time.time()
    if delete_source_file:
        row["private_key_export_purged_at"] = time.time()
    keys[key_id] = row
    payload["version"] = max(1, int(payload.get("version") or 1))
    payload["updated_at"] = time.time()
    payload["keys"] = keys
    _write_json(keys_file, payload)


def _mark_exported_key_purged_without_adoption(
    *,
    keys_file: Path,
    key_id: str,
) -> None:
    payload = _read_json(keys_file, {"version": 1, "keys": {}})
    keys = dict(payload.get("keys") or {})
    row = dict(keys.get(key_id) or {})
    if not row:
        return
    row["private_key_export_purged_without_adoption_at"] = time.time()
    keys[key_id] = row
    payload["version"] = max(1, int(payload.get("version") or 1))
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


def _reset_access_configuration(paths: Dict[str, Path]) -> Dict[str, Any]:
    hosting_root = paths["hosting_root"]
    archive_root = hosting_root / "archive" / ("access_reset_" + time.strftime("%Y%m%d_%H%M%S"))
    if archive_root.exists():
        suffix = 1
        while (hosting_root / "archive" / f"{archive_root.name}_{suffix}").exists():
            suffix += 1
        archive_root = hosting_root / "archive" / f"{archive_root.name}_{suffix}"
    candidates = [
        paths["access_file"],
        paths["keys_file"],
        paths["mappings_file"],
        paths["bootstrap_state_file"],
        paths["audit_file"],
        paths["migrations_file"],
        hosting_root / "state" / "sessions.json",
        hosting_root / "state" / "challenges.json",
        hosting_root / "audit" / "auth_audit.json",
    ]
    archived: list[Dict[str, str]] = []
    for source in candidates:
        source = source.expanduser().resolve()
        if not source.exists() or not source.is_file():
            continue
        relative = source.relative_to(hosting_root) if source.is_relative_to(hosting_root) else Path(source.name)
        target = (archive_root / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        archived.append({"source": str(source), "archive": str(target)})
    manifest = {
        "version": 1,
        "timestamp": time.time(),
        "event": "hosting_access_reset_to_unconfigured",
        "hosting_root": str(hosting_root),
        "archived": archived,
        "notes": [
            "Active hosting access files were archived, not deleted.",
            "Exported private key files and client-realm private-key secrets are not removed by this reset.",
        ],
    }
    if archived:
        archive_root.mkdir(parents=True, exist_ok=True)
        _write_json(archive_root / "reset_manifest.json", manifest)
    return {
        "status": "ok",
        "action": "reset_unconfigured",
        "hosting_root": str(hosting_root),
        "archive_dir": str(archive_root) if archived else None,
        "archived_count": len(archived),
        "archived": archived,
        "message": "Hosting access configuration was reset to unconfigured by archiving active access files.",
        "private_key_note": "Exported private key files and client-realm private-key secrets were not removed.",
    }


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


def _resolve_client_realm_root(args: argparse.Namespace) -> Path:
    default_config_dir, _ = _default_paths()
    if str(args.default_config_dir or "").strip():
        default_config_dir = Path(str(args.default_config_dir)).expanduser().resolve()
    if str(getattr(args, "client_realm_root", "") or "").strip():
        return Path(str(args.client_realm_root)).expanduser().resolve()
    realm = str(getattr(args, "client_realm", "") or "default").strip() or "default"
    return _client_realm_root(default_config_dir, realm)


def _read_text_or_file(*, inline_value: str, file_value: str, field_name: str) -> str:
    inline = str(inline_value or "").strip()
    if inline:
        return inline
    file_raw = str(file_value or "").strip()
    if file_raw:
        return Path(file_raw).expanduser().resolve().read_text(encoding="utf-8").strip()
    raise ValueError(f"{field_name} is required")


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


def run_transport_bootstrap(args: argparse.Namespace) -> Dict[str, Any]:
    action_harden = bool(getattr(args, "transport_harden_ssh", False))
    action_export = bool(getattr(args, "transport_export_bootstrap", False))
    action_import = bool(getattr(args, "transport_import_bootstrap", False))
    action_validate = bool(getattr(args, "transport_validate_profile", False))
    action_provision = bool(getattr(args, "transport_provision_ssh_artifacts", False))
    action_install_authorized = bool(getattr(args, "transport_install_authorized_key", False))
    selected_count = sum(
        1
        for flag in (
            action_harden,
            action_export,
            action_import,
            action_validate,
            action_provision,
            action_install_authorized,
        )
        if flag
    )
    if selected_count > 1:
        raise ValueError(
            "Choose only one transport action"
        )
    if selected_count == 0:
        raise ValueError("No transport bootstrap action selected")
    def _install_authorized_and_register(public_key: str) -> Dict[str, Any]:
        auth_file_raw = str(getattr(args, "ssh_authorized_keys_file", "") or "").strip()
        auth_file = (
            Path(auth_file_raw).expanduser().resolve()
            if auth_file_raw
            else (Path.home() / ".ssh" / "authorized_keys").resolve()
        )
        unrestricted = bool(getattr(args, "ssh_authorized_key_unrestricted", False))
        install_result = install_transport_authorized_key(
            transport_public_key=public_key,
            authorized_keys_file=auth_file,
            transport_key_id=str(args.transport_key_id or "").strip(),
            forced_command=""
            if unrestricted
            else str(
                getattr(args, "ssh_authorized_key_command", DEFAULT_TRANSPORT_AUTHORIZED_KEY_COMMAND)
                or ""
            ).strip(),
            restrict_options=not unrestricted,
        )
        paths = _resolve_paths(args, create_dirs=True)
        svc = EngineHostService(control_state_file=paths["control_state_path"])
        key_result = svc.auth_upsert_key(
            key_id=str(install_result.get("transport_key_id") or "transport"),
            role="transport",
            auth_method="public_key",
            public_key=public_key,
        )
        return {
            **install_result,
            "rbac_key_id": key_result.get("key_id"),
            "rbac_role": key_result.get("role"),
        }

    if action_harden:
        target = str(args.transport_target or "").strip()
        if not target:
            raise ValueError("--transport-target is required for --transport-harden-ssh")
        transport_key_id = str(args.transport_key_id or "").strip()
        if not transport_key_id:
            raise ValueError("--transport-key-id is required for --transport-harden-ssh")
        public_key = _read_text_or_file(
            inline_value=str(args.transport_public_key_inline or ""),
            file_value=str(args.transport_public_key_file or ""),
            field_name="transport public key",
        )
        private_key = _read_text_or_file(
            inline_value=str(args.transport_private_key_inline or ""),
            file_value=str(args.transport_private_key_file or ""),
            field_name="transport private key",
        )
        known_hosts_line = _read_text_or_file(
            inline_value=str(args.ssh_known_hosts_line or ""),
            file_value=str(args.ssh_known_hosts_file or ""),
            field_name="ssh known_hosts line",
        )
        profile_name = str(args.transport_profile_name or "").strip() or transport_key_id
        realm = str(getattr(args, "client_realm", "") or "default").strip() or "default"
        client_realm_root = _resolve_client_realm_root(args)
        bundle = make_transport_bootstrap_bundle(
            target=target,
            ssh_known_hosts_line=known_hosts_line,
            transport_key_id=transport_key_id,
            transport_public_key=public_key,
            transport_private_key_openssh=private_key,
            bundle_password=str(getattr(args, "bootstrap_password", "") or ""),
            control_ssh_fingerprint=str(args.control_ssh_fingerprint or "").strip(),
            profile_name=profile_name,
        )
        imported = import_transport_bootstrap_bundle(
            bundle=bundle,
            client_realm_root=client_realm_root,
            realm=realm,
            profile_name=profile_name,
            overwrite_profile=bool(getattr(args, "overwrite_profile", False)),
            bundle_password=str(getattr(args, "bootstrap_password", "") or ""),
            secret_password=str(getattr(args, "client_secret_password", "") or ""),
        )
        provisioned = provision_client_ssh_artifacts(
            client_realm_root=client_realm_root,
            profile_name=profile_name,
            realm=realm,
            ssh_alias=str(getattr(args, "ssh_config_alias", "") or "").strip(),
            secret_password=str(getattr(args, "client_secret_password", "") or ""),
            overwrite=True,
        )
        installed = _install_authorized_and_register(public_key)
        validated = validate_client_transport_profile(
            client_realm_root=client_realm_root,
            profile_name=profile_name,
            realm=realm,
            run_ssh=not bool(getattr(args, "validation_no_ssh_run", False)),
            ssh_bin=str(getattr(args, "validation_ssh_bin", "") or "ssh").strip() or "ssh",
            remote_command=str(getattr(args, "validation_remote_command", "") or "exit 0").strip() or "exit 0",
            timeout_seconds=float(getattr(args, "validation_timeout_seconds", 15.0) or 15.0),
            secret_password=str(getattr(args, "client_secret_password", "") or ""),
        )
        validation_status = str(validated.get("status") or "ok")
        admin_capability = str(getattr(args, "admin_capability", "") or "no_admin_available").strip()
        if admin_capability not in VALID_ADMIN_CAPABILITIES:
            admin_capability = "no_admin_available"
        followups: list[str] = []
        if admin_capability == "no_admin_available":
            followups.append("User-scoped SSH transport was hardened; machine-wide SSH service, firewall, and service-managed daemon setup still require an administrator.")
        elif admin_capability == "admin_available_interactive":
            followups.append("Run platform-specific elevated SSH service/firewall/daemon setup only through an explicit UAC/sudo/polkit prompt.")
        else:
            followups.append("Provide the generated forced-command authorized_keys entry and strict client profile details to the administrator, then rerun diagnostics.")
        return {
            "status": "ok" if validation_status == "ok" else validation_status,
            "action": "transport_harden_ssh",
            "client_realm_root": str(client_realm_root),
            "admin_capability": admin_capability,
            "profile_name": profile_name,
            "target": target,
            "transport_key_id": transport_key_id,
            "profile_path": imported.get("profile_path"),
            "known_hosts_file": imported.get("known_hosts_file"),
            "secret_id": imported.get("secret_id"),
            "secret_path": imported.get("secret_path"),
            "secret_encryption": imported.get("secret_encryption"),
            "ssh_alias": provisioned.get("ssh_alias"),
            "ssh_config_file": provisioned.get("ssh_config_file"),
            "identity_file": provisioned.get("identity_file"),
            "ssh_command": provisioned.get("ssh_command"),
            "authorized_keys_file": installed.get("authorized_keys_file"),
            "forced_command": installed.get("forced_command"),
            "restrict_options": installed.get("restrict_options"),
            "rbac_key_id": installed.get("rbac_key_id"),
            "rbac_role": installed.get("rbac_role"),
            "marker": installed.get("marker"),
            "validation_status": validation_status,
            "ssh_probe_ran": bool(validated.get("ssh_probe_ran")),
            "ssh_probe_returncode": validated.get("ssh_probe_returncode"),
            "followups": followups,
        }
    if action_export:
        target = str(args.transport_target or "").strip()
        if not target:
            raise ValueError("--transport-target is required for --transport-export-bootstrap")
        transport_key_id = str(args.transport_key_id or "").strip()
        if not transport_key_id:
            raise ValueError("--transport-key-id is required for --transport-export-bootstrap")
        bundle_file_raw = str(args.bootstrap_bundle_file or "").strip()
        if not bundle_file_raw:
            raise ValueError("--bootstrap-bundle-file is required for --transport-export-bootstrap")
        transport_public_key = _read_text_or_file(
            inline_value=str(args.transport_public_key_inline or ""),
            file_value=str(args.transport_public_key_file or ""),
            field_name="transport public key",
        )
        transport_private_key = _read_text_or_file(
            inline_value=str(args.transport_private_key_inline or ""),
            file_value=str(args.transport_private_key_file or ""),
            field_name="transport private key",
        )
        known_hosts_line = _read_text_or_file(
            inline_value=str(args.ssh_known_hosts_line or ""),
            file_value=str(args.ssh_known_hosts_file or ""),
            field_name="ssh known_hosts line",
        )
        bundle = make_transport_bootstrap_bundle(
            target=target,
            ssh_known_hosts_line=known_hosts_line,
            transport_key_id=transport_key_id,
            transport_public_key=transport_public_key,
            transport_private_key_openssh=transport_private_key,
            bundle_password=str(getattr(args, "bootstrap_password", "") or ""),
            control_ssh_fingerprint=str(args.control_ssh_fingerprint or "").strip(),
            profile_name=str(args.transport_profile_name or "").strip(),
        )
        bundle_path = write_transport_bootstrap_bundle(
            bundle,
            Path(bundle_file_raw).expanduser().resolve(),
        )
        return {
            "status": "ok",
            "action": "transport_export_bootstrap",
            "bundle_file": str(bundle_path),
            "target": target,
            "transport_key_id": transport_key_id,
        }
    if action_install_authorized:
        public_key = _read_text_or_file(
            inline_value=str(args.transport_public_key_inline or ""),
            file_value=str(args.transport_public_key_file or ""),
            field_name="transport public key",
        )
        result = _install_authorized_and_register(public_key)
        return {
            "status": str(result.get("status") or "ok"),
            "action": "transport_install_authorized_key",
            **result,
        }
    client_realm_root = _resolve_client_realm_root(args)
    realm = str(getattr(args, "client_realm", "") or "default").strip() or "default"
    if action_import:
        bundle_file_raw = str(args.bootstrap_bundle_file or "").strip()
        if not bundle_file_raw:
            raise ValueError("--bootstrap-bundle-file is required for --transport-import-bootstrap")
        bundle = read_transport_bootstrap_bundle(Path(bundle_file_raw).expanduser().resolve())
        result = import_transport_bootstrap_bundle(
            bundle=bundle,
            client_realm_root=client_realm_root,
            realm=realm,
            profile_name=str(args.transport_profile_name or "").strip() or None,
            overwrite_profile=bool(getattr(args, "overwrite_profile", False)),
            bundle_password=str(getattr(args, "bootstrap_password", "") or ""),
            secret_password=str(getattr(args, "client_secret_password", "") or ""),
        )
        profile = read_client_profile(client_realm_root, str(result.get("profile_name") or ""))
        return {
            "status": "ok",
            "action": "transport_import_bootstrap",
            "bundle_file": str(Path(bundle_file_raw).expanduser().resolve()),
            "client_realm_root": str(client_realm_root),
            "profile_name": result.get("profile_name"),
            "profile_path": result.get("profile_path"),
            "known_hosts_file": result.get("known_hosts_file"),
            "secret_id": result.get("secret_id"),
            "secret_path": result.get("secret_path"),
            "secret_encryption": result.get("secret_encryption"),
            "private_key_protection": result.get("private_key_protection"),
            "target": dict(profile.get("profile") or {}).get("engine_host_ssh_target"),
            "transport_key_id": dict(profile.get("profile") or {}).get("transport_key_id"),
        }
    profile_name = str(args.transport_profile_name or "").strip()
    if action_provision:
        if not profile_name:
            raise ValueError("--transport-profile-name is required for --transport-provision-ssh-artifacts")
        result = provision_client_ssh_artifacts(
            client_realm_root=client_realm_root,
            profile_name=profile_name,
            realm=realm,
            ssh_alias=str(getattr(args, "ssh_config_alias", "") or "").strip(),
            secret_password=str(getattr(args, "client_secret_password", "") or ""),
            overwrite=bool(getattr(args, "overwrite_ssh_config", False)),
        )
        return {
            "status": str(result.get("status") or "ok"),
            "action": "transport_provision_ssh_artifacts",
            "client_realm_root": str(client_realm_root),
            **result,
        }
    if not profile_name:
        raise ValueError("--transport-profile-name is required for --transport-validate-profile")
    result = validate_client_transport_profile(
        client_realm_root=client_realm_root,
        profile_name=profile_name,
        realm=realm,
        run_ssh=not bool(getattr(args, "validation_no_ssh_run", False)),
        ssh_bin=str(getattr(args, "validation_ssh_bin", "") or "ssh").strip() or "ssh",
        remote_command=str(getattr(args, "validation_remote_command", "") or "exit 0").strip() or "exit 0",
        timeout_seconds=float(getattr(args, "validation_timeout_seconds", 15.0) or 15.0),
        secret_password=str(getattr(args, "client_secret_password", "") or ""),
    )
    return {
        "status": str(result.get("status") or "ok"),
        "action": "transport_validate_profile",
        "client_realm_root": str(client_realm_root),
        **result,
    }


def _admin_setup_platform() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "unix"


def _admin_setup_script(args: argparse.Namespace, *, platform_name: str) -> Tuple[str, str, list[str]]:
    enable_ssh = bool(getattr(args, "admin_setup_enable_ssh_service", True))
    enable_firewall = bool(getattr(args, "admin_setup_enable_firewall", False))
    enable_linger = bool(getattr(args, "admin_setup_enable_user_linger", False))
    target_user = str(getattr(args, "admin_setup_target_user", "") or "").strip()
    followups: list[str] = []
    if platform_name == "windows":
        lines = [
            "$ErrorActionPreference = 'Stop'",
            "Write-Host 'mp13 hosting admin setup: Windows OpenSSH/service checks'",
        ]
        if enable_ssh:
            lines.extend(
                [
                    "$cap = Get-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0' -ErrorAction SilentlyContinue",
                    "if ($cap -and $cap.State -ne 'Installed') { Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0' }",
                    "Set-Service -Name sshd -StartupType Automatic",
                    "Start-Service sshd",
                ]
            )
        if enable_firewall:
            lines.extend(
                [
                    "$rule = Get-NetFirewallRule -Name 'mp13-hosting-sshd' -ErrorAction SilentlyContinue",
                    "if (-not $rule) { New-NetFirewallRule -Name 'mp13-hosting-sshd' -DisplayName 'mp13 Hosting OpenSSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 }",
                ]
            )
        else:
            followups.append("Firewall rule was not requested; remote SSH may still be blocked by Windows Firewall or network policy.")
        if enable_linger:
            followups.append("Windows user daemon auto-start is not configured here; use Task Scheduler or service-managed hosting setup.")
        lines.append("Write-Host 'mp13 hosting admin setup complete'")
        return "\r\n".join(lines) + "\r\n", ".ps1", followups

    target_user_expr = shlex.quote(target_user) if target_user else "${SUDO_USER:-$USER}"
    lines = [
        "#!/bin/sh",
        "set -eu",
        "echo 'mp13 hosting admin setup: SSH/service checks'",
    ]
    if enable_ssh:
        lines.extend(
            [
                "if command -v systemctl >/dev/null 2>&1; then",
                "  if systemctl list-unit-files ssh.service >/dev/null 2>&1; then",
                "    systemctl enable --now ssh.service",
                "  elif systemctl list-unit-files sshd.service >/dev/null 2>&1; then",
                "    systemctl enable --now sshd.service",
                "  else",
                "    echo 'No ssh.service or sshd.service unit was found; install/enable OpenSSH server using the platform package manager.'",
                "  fi",
                "elif command -v service >/dev/null 2>&1; then",
                "  service ssh start 2>/dev/null || service sshd start 2>/dev/null || echo 'Could not start ssh/sshd through service(8).'",
                "else",
                "  echo 'No supported service manager was found; enable OpenSSH server manually.'",
                "fi",
            ]
        )
    if enable_firewall:
        lines.extend(
            [
                "if command -v ufw >/dev/null 2>&1; then",
                "  ufw allow OpenSSH || ufw allow 22/tcp",
                "elif command -v firewall-cmd >/dev/null 2>&1; then",
                "  firewall-cmd --add-service=ssh --permanent",
                "  firewall-cmd --reload",
                "else",
                "  echo 'No supported firewall helper was found; allow TCP/22 manually if remote SSH is blocked.'",
                "fi",
            ]
        )
    else:
        followups.append("Firewall changes were not requested; remote SSH may still be blocked by host or network policy.")
    if enable_linger:
        lines.extend(
            [
                "if command -v loginctl >/dev/null 2>&1; then",
                f"  loginctl enable-linger {target_user_expr}",
                "else",
                "  echo 'loginctl is unavailable; configure user daemon persistence manually for this platform.'",
                "fi",
            ]
        )
    elif platform_name == "macos":
        followups.append("macOS daemon auto-start is not configured here; use a LaunchAgent or service-managed hosting setup.")
    else:
        followups.append("User daemon linger was not requested; detached user daemons may stop after logout on some systemd hosts.")
    lines.append("echo 'mp13 hosting admin setup complete'")
    return "\n".join(lines) + "\n", ".sh", followups


def _write_admin_setup_script(script: str, suffix: str) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "mp13-hosting-admin-setup"
    temp_dir.mkdir(parents=True, exist_ok=True)
    script_path = (temp_dir / f"admin_setup_{int(time.time())}{suffix}").resolve()
    script_path.write_text(script, encoding="utf-8")
    try:
        script_path.chmod(0o700)
    except Exception:
        pass
    return script_path


def _admin_setup_elevation_command(script_path: Path, *, platform_name: str) -> Tuple[list[str], str]:
    if platform_name == "windows":
        quoted_path = str(script_path).replace("'", "''")
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Start-Process -FilePath PowerShell "
            f"-ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','{quoted_path}' "
            "-Verb RunAs -Wait",
        ], "windows_uac"
    if platform_name == "macos":
        quoted = shlex.quote(str(script_path))
        return [
            "osascript",
            "-e",
            f'do shell script "/bin/sh {quoted}" with administrator privileges',
        ], "macos_authorization"
    if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and shutil.which("pkexec"):
        return ["pkexec", "/bin/sh", str(script_path)], "pkexec"
    if shutil.which("sudo"):
        return ["sudo", "/bin/sh", str(script_path)], "sudo"
    raise RuntimeError("No supported elevation tool found. Install pkexec/sudo or run the generated script as root.")


def run_transport_admin_setup(args: argparse.Namespace) -> Dict[str, Any]:
    platform_name = _admin_setup_platform()
    script, suffix, followups = _admin_setup_script(args, platform_name=platform_name)
    execute = bool(getattr(args, "admin_setup_execute", False))
    script_path: Optional[Path] = None
    result: Dict[str, Any] = {
        "status": "dry_run",
        "action": "transport_admin_setup",
        "platform": platform_name,
        "execute": execute,
        "ssh_service": bool(getattr(args, "admin_setup_enable_ssh_service", True)),
        "firewall": bool(getattr(args, "admin_setup_enable_firewall", False)),
        "user_linger": bool(getattr(args, "admin_setup_enable_user_linger", False)),
        "script": script,
        "followups": followups,
    }
    if not execute:
        return result
    script_path = _write_admin_setup_script(script, suffix)
    command, method = _admin_setup_elevation_command(script_path, platform_name=platform_name)
    completed = subprocess.run(command, check=False)
    result.update(
        {
            "status": "ok" if int(completed.returncode) == 0 else "elevation_failed",
            "script_file": str(script_path),
            "elevation_method": method,
            "returncode": int(completed.returncode),
        }
    )
    return result


def _interactive_admin_setup_followup(args: argparse.Namespace, suggestion: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mode = str(suggestion.get("mode") or "")
    if mode not in {"ssh_tunnel_only", "truly_remote"}:
        return None
    admin_capability = str(suggestion.get("admin_capability") or "no_admin_available")
    if admin_capability == "no_admin_available":
        return None
    while True:
        action = _prompt_menu(
            "SSH Admin Setup",
            {
                "skip": ("Skip admin setup", "continue with hosting config only"),
                "generate": ("Generate admin setup script", "dry-run script and instructions"),
                "execute": ("Run elevated admin setup now", "launch platform-native UAC/sudo/pkexec prompt"),
            },
            "generate" if admin_capability == "admin_managed_externally" else "skip",
            allow_back=True,
        )
        if action == "changes":
            continue
        if action in {"back", "skip"}:
            return None
        admin_args = argparse.Namespace(**vars(args))
        admin_args.admin_setup_execute = action == "execute"
        admin_args.transport_admin_setup = True
        if mode == "ssh_tunnel_only":
            admin_args.admin_setup_enable_user_linger = True
        result = run_transport_admin_setup(admin_args)
        _print_admin_setup_report(result)
        return result


def run_client_keys(args: argparse.Namespace) -> Dict[str, Any]:
    action_list = bool(getattr(args, "client_list_keys", False))
    action_generate = bool(getattr(args, "client_generate_key", False))
    action_import = bool(getattr(args, "client_import_key", False))
    action_export = bool(getattr(args, "client_export_key", False))
    action_list_exported = bool(getattr(args, "client_list_exported_keys", False))
    action_handoff_exported = bool(getattr(args, "client_handoff_exported_key", False) or getattr(args, "client_adopt_exported_key", False))
    action_purge_exported = bool(getattr(args, "client_purge_exported_key", False))
    selected_count = sum(
        1
        for flag in (
            action_list,
            action_generate,
            action_import,
            action_export,
            action_list_exported,
            action_handoff_exported,
            action_purge_exported,
        )
        if flag
    )
    if selected_count != 1:
        raise ValueError(
            "Choose exactly one client key action"
        )
    client_realm_root = _resolve_client_realm_root(args)
    realm = str(getattr(args, "client_realm", "") or "default").strip() or "default"
    layout = ensure_client_realm_dirs(client_realm_root)
    keys_file = layout["keys"]
    if action_list_exported:
        source_keys_file = _source_exported_keys_file(args)
        rows = discover_exported_private_keys(keys_file=source_keys_file)
        return {
            "status": "ok",
            "action": "client_list_exported_keys",
            "client_realm_root": str(client_realm_root),
            "realm": realm,
            "source_keys_file": str(source_keys_file),
            "exported_keys": rows,
        }
    if action_list:
        payload = _read_json(keys_file, {"keys": {}})
        return {
            "status": "ok",
            "action": "client_list_keys",
            "client_realm_root": str(client_realm_root),
            "realm": realm,
            "keys": dict(payload.get("keys") or {}),
        }
    key_id = str(getattr(args, "client_key_id", "") or "").strip()
    if not key_id:
        raise ValueError("--client-key-id is required")
    tag = str(getattr(args, "client_key_tag", "") or "rbac_private_key").strip() or "rbac_private_key"
    if tag not in {"rbac_private_key", "transport_private_key"}:
        raise ValueError("--client-key-tag must be rbac_private_key or transport_private_key")
    if action_purge_exported:
        source_keys_file = _source_exported_keys_file(args)
        purged = purge_exported_private_key_file(keys_file=source_keys_file, key_id=key_id)
        _mark_exported_key_purged_without_adoption(keys_file=source_keys_file, key_id=key_id)
        return {
            "status": "ok",
            "action": "client_purge_exported_key",
            "client_realm_root": str(client_realm_root),
            "realm": realm,
            "key_id": key_id,
            "source_keys_file": str(source_keys_file),
            "source_export_path": purged.get("source_export_path"),
            "deleted_source_file": bool(purged.get("deleted_source_file")),
            "warning": purged.get("warning"),
        }
    if action_handoff_exported:
        source_keys_file = _source_exported_keys_file(args)
        delete_source = bool(getattr(args, "client_delete_exported_key_file", False))
        stored = handoff_exported_private_key_file(
            client_realm_root,
            keys_file=source_keys_file,
            realm=realm,
            key_id=key_id,
            tag=tag,
            delete_source_file=delete_source,
        )
        _mark_exported_key_adopted(
            keys_file=source_keys_file,
            key_id=key_id,
            client_realm_root=client_realm_root,
            secret_id=str(stored.get("secret_id") or ""),
            delete_source_file=delete_source,
        )
        audit_path = append_client_audit_event(
            client_realm_root,
            event_type="client_key_handoff_exported",
            realm=realm,
            payload={
                "key_id": key_id,
                "tag": tag,
                "source_keys_file": str(source_keys_file),
                "source_export_path": stored.get("source_export_path"),
                "deleted_source_file": delete_source,
                "secret_id": stored.get("secret_id"),
            },
        )
        return {
            "status": "ok",
            "action": "client_handoff_exported_key",
            "client_realm_root": str(client_realm_root),
            "realm": realm,
            "key_id": key_id,
            "tag": tag,
            "source_keys_file": str(source_keys_file),
            "source_export_path": stored.get("source_export_path"),
            "deleted_source_file": delete_source,
            "secret_id": stored.get("secret_id"),
            "secret_path": stored.get("secret_path"),
            "keys_file": stored.get("keys_file"),
            "audit_path": str(audit_path),
        }
    if action_export:
        row = dict(_read_json(keys_file, {"keys": {}}).get("keys", {}).get(key_id) or {})
        secret_id = str(row.get("private_key_secret_id") or "").strip()
        if not secret_id:
            raise ValueError(f"client key {key_id!r} does not reference a client-realm secret")
        export_path_raw = str(getattr(args, "client_export_key_path", "") or "").strip()
        if not export_path_raw:
            raise ValueError("--client-export-key-path is required for --client-export-key")
        export_path = Path(export_path_raw).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        store = FileSecretStore(client_realm_root, realm=realm)
        export_path.write_text(
            str(store.get_secret_payload(secret_id, password=str(getattr(args, "client_secret_password", "") or "")) or ""),
            encoding="utf-8",
        )
        try:
            export_path.chmod(0o600)
        except Exception:
            pass
        audit_path = append_client_audit_event(
            client_realm_root,
            event_type="client_key_export",
            realm=realm,
            payload={"key_id": key_id, "tag": tag, "path": str(export_path)},
        )
        return {
            "status": "ok",
            "action": "client_export_key",
            "client_realm_root": str(client_realm_root),
            "realm": realm,
            "key_id": key_id,
            "tag": tag,
            "export_path": str(export_path),
            "audit_path": str(audit_path),
        }

    client_secret_password = str(getattr(args, "client_secret_password", "") or "")
    if action_generate:
        protection_passphrase = str(getattr(args, "generated_key_passphrase", "") or "") or client_secret_password
        private_key_text, public_key_text = _generate_keypair(
            key_id=key_id,
            passphrase=protection_passphrase or None,
        )
        source = "client_generate"
    else:
        inline_private_key = normalize_pasted_private_key(str(getattr(args, "client_private_key", "") or ""))
        try:
            setattr(args, "client_private_key", "")
        except Exception:
            pass
        private_key_text = inline_private_key or _read_text_or_file(
            inline_value="",
            file_value=str(getattr(args, "client_private_key_file", "") or ""),
            field_name="client private key",
        )
        private_key_text = normalize_pasted_private_key(private_key_text)
        public_key_text = _read_text_or_file(
            inline_value=str(getattr(args, "client_public_key_inline", "") or ""),
            file_value=str(getattr(args, "client_public_key_file", "") or ""),
            field_name="client public key",
        ) if (str(getattr(args, "client_public_key_inline", "") or "").strip() or str(getattr(args, "client_public_key_file", "") or "").strip()) else _derive_public_key_from_private(private_key_text)
        if client_secret_password:
            private_key_text = _protect_openssh_private_key(
                private_key_text,
                new_passphrase=client_secret_password,
            )
        source = "client_import"

    secret_store = FileSecretStore(client_realm_root, realm=realm)
    private_key_protection = "openssh_passphrase" if client_secret_password or (action_generate and protection_passphrase) else "none"
    secret_record = secret_store.put_secret(
        tag=tag,
        payload=private_key_text,
        secret_id=f"{tag.replace('_private_key', '')}-{key_id}-private",
        metadata={
            "key_id": key_id,
            "tag": tag,
            "source": source,
            "private_key_format": "openssh",
            "private_key_protection": private_key_protection,
        },
        encryption="none",
    )
    role = "transport" if tag == "transport_private_key" else "admin"
    _store_importable_key_record(
        keys_file=keys_file,
        key_id=key_id,
        role=role,
        auth_method="public_key",
        public_key=public_key_text,
        key_origin="generated" if action_generate else "imported",
        key_source="generate" if action_generate else "import",
        public_key_source="generated" if action_generate else ("file" if str(getattr(args, "client_public_key_file", "") or "").strip() else "derived_or_inline"),
        private_key_storage="client_realm_secret",
        private_key_secret_id=secret_record.secret_id,
        private_key_secret_realm=realm,
        private_key_protection=private_key_protection,
    )
    audit_path = append_client_audit_event(
        client_realm_root,
        event_type="client_key_generate" if action_generate else "client_key_import",
        realm=realm,
        payload={
            "key_id": key_id,
            "tag": tag,
            "secret_id": secret_record.secret_id,
            "encryption": secret_record.encryption,
            "private_key_protection": private_key_protection,
        },
    )
    return {
        "status": "ok",
        "action": "client_generate_key" if action_generate else "client_import_key",
        "client_realm_root": str(client_realm_root),
        "realm": realm,
        "key_id": key_id,
        "tag": tag,
        "public_key": public_key_text,
        "secret_id": secret_record.secret_id,
        "secret_path": str(secret_record_path(client_realm_root, secret_record.secret_id)),
        "secret_encryption": secret_record.encryption,
        "private_key_protection": private_key_protection,
        "keys_file": str(keys_file),
        "audit_path": str(audit_path),
    }


def run_setup(args: argparse.Namespace) -> Dict[str, Any]:
    interactive = bool(args.interactive)
    paths = _resolve_paths(args, create_dirs=not interactive)
    default_config_dir = paths["default_config_dir"]
    control_state_path = paths["control_state_path"]
    hosting_root = paths["hosting_root"]
    access_file = paths["access_file"]
    keys_file = paths["keys_file"]
    mappings_file = paths["mappings_file"]
    bootstrap_state_file = paths["bootstrap_state_file"]
    audit_file = paths["audit_file"]
    migrations_file = paths["migrations_file"]
    dirs = {
        "root": hosting_root,
        "keyring": keys_file.parent,
        "audit": audit_file.parent,
        "state": hosting_root / "state",
        "bootstrap": mappings_file.parent,
    }

    mode = _normalize_mode(args.mode, "local_only")
    endpoint_mode = _normalize_endpoint_mode(args.endpoint_mode, "exclusive")
    lifecycle_profile = _normalize_lifecycle_profile(args.lifecycle_profile, "detached_user_process")
    key_source = str(args.key_source or "").strip().lower() or "import"
    if key_source not in VALID_KEY_SOURCES:
        key_source = "import"
    admin_key_id = str(args.admin_key_id or "").strip() or "admin-main"
    key_action = str(getattr(args, "key_action", "") or "replace").strip().lower()
    if key_action not in {"keep_existing", "replace"}:
        key_action = "replace"
    permission_action = str(getattr(args, "permission_action", "") or "none").strip().lower()
    if permission_action not in {"none", "tighten"}:
        permission_action = "none"
    setup_scope = "fresh_setup"
    usage_intent = _normalize_usage_intent(getattr(args, "usage_intent", "") or "single_admin")
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
    existing_key_meta = _admin_key_metadata(keys_file, str(existing_summary.get("admin_key_id") or ""))
    setup_context_defaults = _infer_setup_context_defaults(
        summary=existing_summary,
        probe=current_probe,
        default_usage_intent=usage_intent,
    )
    setup_context: Dict[str, str] = dict(setup_context_defaults)

    if interactive:
        _clear_pending_staged_setup()
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

        def _run_main_menu() -> None:
            while True:
                operator_choice = _prompt_menu(
                    "Hosting Access Main Menu",
                    {
                        "1": ("Configure hosting now", "guided setup using usage/access projections"),
                        "2": ("Review status details", "show current files, keys, and config state"),
                        "3": ("Run doctor diagnostics", "validate setup and suggest fixes"),
                        "4": ("Manage RBAC keys", "list/revoke access keys, hand off or purge exported private keys, and review auth sessions/audit"),
                    },
                    "1",
                    allow_back=False,
                )
                if operator_choice == "1":
                    return
                if operator_choice == "2":
                    _print_current_probe(existing_summary, current_probe, config_state, existing_key_meta)
                    continue
                if operator_choice == "3":
                    doctor_result = run_doctor(args)
                    _print_doctor_report(doctor_result)
                    if _doctor_followup_action(doctor_result) == "setup":
                        return
                    continue
                if operator_choice == "4":
                    _interactive_rbac_menu(args)
                    continue
                if operator_choice == "changes":
                    continue

        _run_main_menu()

        auto_applied = False
        current_projection_auth = require_auth_seed
        while True:
            context = _collect_setup_context(usage_intent, setup_context_defaults)
            if str(context.get("action") or "") == "back":
                _run_main_menu()
                continue
            setup_context = dict(context)
            suggestion = _suggest_auto_configuration(context)
            _print_auto_configuration(context, suggestion)
            default_action = "leave_unconfigured" if bool(suggestion.get("leave_unconfigured")) else "apply"
            if bool(suggestion.get("leave_unconfigured")):
                suggested_options = {
                    "leave_unconfigured": ("Leave hosting unchanged", "no access files are written, reset, or deleted"),
                }
            else:
                suggested_options = {
                    "apply": ("Apply suggested configuration", "skip field-by-field review and continue to final confirmation"),
                    "customize": ("Customize configuration", "start from the suggested intent and edit choices"),
                }
            reset_available = any(
                bool(current_probe.get(name))
                for name in ("access_exists", "keys_exists", "mapping_exists", "bootstrap_exists", "audit_exists")
            ) or int(existing_summary.get("admin_key_count") or 0) > 0
            if bool(suggestion.get("leave_unconfigured")) and reset_available:
                suggested_options["reset_unconfigured"] = (
                    "Reset to unconfigured",
                    "archive active access files; exported private keys and client secrets are not removed",
                )
            suggestion_action = _prompt_menu(
                "Suggested Action",
                suggested_options,
                default_action,
                allow_back=True,
            )
            if suggestion_action == "back":
                _run_main_menu()
                continue
            if suggestion_action == "changes":
                continue
            if suggestion_action == "leave_unconfigured" or (
                suggestion_action == "apply" and bool(suggestion.get("leave_unconfigured"))
            ):
                _clear_pending_staged_setup()
                _print_title("No Changes Written")
                _kv_rows(
                    [
                        ("result", "Hosting access configuration was left unchanged."),
                        ("reason", "Skip access setup for now."),
                        ("hosting_root", str(hosting_root)),
                    ]
                )
                return {
                    "status": "skipped",
                    "action": "leave_unconfigured",
                    "reason": "local_experiment",
                    "message": "Hosting access configuration was left unchanged for local experimentation.",
                    "hosting_root": str(hosting_root),
                    "followups": list(suggestion.get("followups") or []),
                }
            if suggestion_action == "reset_unconfigured":
                if not _plain_yes_no(
                    "Archive active hosting access files and reset this setup to unconfigured?",
                    False,
                ):
                    continue
                reset_result = _reset_access_configuration(paths)
                _print_title("Reset Complete")
                _kv_rows(
                    [
                        ("result", reset_result.get("message")),
                        ("archived_files", reset_result.get("archived_count")),
                        ("archive_dir", reset_result.get("archive_dir") or "n/a"),
                        ("private_key_note", reset_result.get("private_key_note")),
                    ]
                )
                return reset_result

            usage_intent = _normalize_usage_intent(str(suggestion.get("usage_intent") or usage_intent), usage_intent)
            setup_scope = usage_intent
            mode = _normalize_mode(str(suggestion.get("mode") or mode), mode)
            endpoint_mode = _normalize_endpoint_mode(str(suggestion.get("endpoint_mode") or endpoint_mode), endpoint_mode)
            lifecycle_profile = _normalize_lifecycle_profile(
                str(suggestion.get("lifecycle_profile") or lifecycle_profile),
                lifecycle_profile,
            )
            current_projection_auth = bool(suggestion.get("require_auth"))
            key_source = str(suggestion.get("key_source") or key_source)
            key_action = str(suggestion.get("key_action") or key_action)
            permission_action = str(suggestion.get("permission_action") or permission_action)
            setup_notes.extend([str(item) for item in list(suggestion.get("followups") or []) if str(item).strip()])
            admin_setup_result = _interactive_admin_setup_followup(args, suggestion)
            if admin_setup_result is not None:
                setup_notes.append(
                    "SSH admin setup follow-up: "
                    + str(admin_setup_result.get("status") or "unknown")
                )
            if suggestion_action == "apply":
                auto_applied = True
                break

            while True:
                usage_choice = _prompt_menu(
                    "Hosting Access",
                    {
                        "single_admin": (
                            str(USAGE_INTENT_GUIDANCE["single_admin"]["label"]),
                            str(USAGE_INTENT_GUIDANCE["single_admin"]["hint"]),
                        ),
                        "role_split": (
                            str(USAGE_INTENT_GUIDANCE["role_split"]["label"]),
                            str(USAGE_INTENT_GUIDANCE["role_split"]["hint"]),
                        ),
                        "multi_user": (
                            str(USAGE_INTENT_GUIDANCE["multi_user"]["label"]),
                            str(USAGE_INTENT_GUIDANCE["multi_user"]["hint"]),
                        ),
                    },
                    usage_intent,
                    allow_back=True,
                )
                if usage_choice == "back":
                    _run_main_menu()
                    continue
                if usage_choice == "changes":
                    continue
                usage_intent = _normalize_usage_intent(usage_choice, usage_intent)
                break

            projection = _project_usage_intent(usage_intent)
            setup_scope = usage_intent
            mode = str(projection.get("mode") or mode)
            endpoint_mode = str(projection.get("endpoint_mode") or endpoint_mode)
            current_projection_auth = bool(projection.get("require_auth"))
            key_action = str(projection.get("key_action") or key_action)
            permission_action = str(projection.get("permission_action") or permission_action)

            _print_title("Hosting Access Projection")
            _kv_rows(
                [
                    ("usage_intent", _option_label(usage_intent)),
                    ("clients_connectivity", mode),
                    ("endpoint_mode", endpoint_mode),
                    ("require_auth", "yes" if current_projection_auth else "no"),
                    ("keys", str(projection.get("note") or "")),
                ]
            )
            mode_choice = _prompt_choice(
                "Clients Connectivity",
                VALID_CONNECTIVITY_MODES,
                mode,
                allow_back=True,
            )
            if mode_choice == "back":
                continue
            mode = _normalize_mode(mode_choice, mode)
            setup_notes.append(str(projection.get("note") or ""))
            break

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
            requested=args.require_auth if args.require_auth is not None else current_projection_auth,
        )
        def _stage_current() -> None:
            _set_pending_staged_setup(
                setup_scope=setup_scope,
                usage_intent=usage_intent,
                mode=mode,
                endpoint_mode=endpoint_mode,
                lifecycle_profile=lifecycle_profile,
                require_auth=current_require_auth,
                key_action=key_action,
                key_source=key_source,
                admin_key_id=admin_key_id,
                permission_action=permission_action,
                admin_public_key_file=admin_public_key_file_value,
                admin_public_key_inline=admin_public_key_inline_value,
            )

        _stage_current()
        if auto_applied:
            _print_title("Review Suggested Configuration")
            _print_pending_staged_setup()
        if not auto_applied:
            print("\nConfiguration steps")
            print(_c("muted", "Use Enter to keep the current value, `b` for back, `c` for staged changes, `q` to quit."))
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
                        _stage_current()
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
                        _stage_current()
                    step_idx += 1
                    continue
                if step == "require_auth":
                    _print_title("Step 3: Require auth")
                    _kv_rows(
                        [
                            ("value", "protects multi-user and remote/tunnel workflows"),
                            ("constraint", "no-auth allowed only for local_only + exclusive"),
                        ]
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
                        _stage_current()
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
                        _stage_current()
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
                        _stage_current()
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
                        _stage_current()
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
                        _stage_current()
                    step_idx += 1
                    continue

        require_auth = current_require_auth
        if key_action != "keep_existing" and key_source == "import":
            admin_public_key_file_value, admin_public_key_inline_value = _resolve_import_source(
                interactive=interactive,
                current_file=admin_public_key_file_value,
                current_inline=admin_public_key_inline_value,
            )
            _stage_current()
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
        _print_intent_guidance(mode, require_auth=require_auth, endpoint_mode=endpoint_mode)
        if not _bool_prompt("Apply this configuration now?", True):
            _clear_pending_staged_setup()
            raise UserCancelled("cancelled by user")
        _ensure_dirs(hosting_root)
        _clear_pending_staged_setup()
    else:
        _clear_pending_staged_setup()
        require_auth = _safe_require_auth(
            connectivity_mode=mode,
            endpoint_mode=endpoint_mode,
            requested=args.require_auth,
        )

    admin_public_key = ""
    admin_private_key_text: Optional[str] = None
    admin_public_key_path: Optional[Path] = None
    admin_private_key_secret_id: Optional[str] = None
    admin_private_key_secret_realm: Optional[str] = None
    admin_private_key_secret_path: Optional[Path] = None
    admin_private_key_secret_encryption: Optional[str] = None
    admin_private_key_export_command: Optional[str] = None
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
    admin_private_key_protection = "none"

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
        admin_private_key_protection = str(row.get("private_key_protection") or "").strip() or "none"
        admin_private_key_secret_id = str(row.get("private_key_secret_id") or "").strip() or None
        admin_private_key_secret_realm = str(row.get("private_key_secret_realm") or "default").strip() or None
        if admin_private_key_secret_id:
            admin_private_key_secret_path = secret_record_path(
                _client_realm_root(default_config_dir, admin_private_key_secret_realm or "default"),
                admin_private_key_secret_id,
            )
            if admin_private_key_secret_path.exists():
                try:
                    secret_payload = json.loads(admin_private_key_secret_path.read_text(encoding="utf-8"))
                    admin_private_key_secret_encryption = str(secret_payload.get("encryption") or "").strip() or None
                    secret_meta = dict(secret_payload.get("metadata") or {})
                    admin_private_key_protection = str(
                        secret_meta.get("private_key_protection") or admin_private_key_protection or "none"
                    ).strip() or "none"
                except Exception:
                    admin_private_key_secret_encryption = None
        export_private_path = (
            Path(str(row.get("private_key_export_path"))).expanduser().resolve()
            if str(row.get("private_key_export_path") or "").strip()
            else export_private_path
        )
    else:
        if key_source == "generate":
            key_origin = "generated"
            public_key_source = "generated"
            client_secret_password = str(getattr(args, "client_secret_password", "") or "")
            passphrase = str(args.generated_key_passphrase or "") or client_secret_password
            if interactive and not args.generated_key_passphrase:
                if _bool_prompt("Protect generated private key with passphrase?", False):
                    passphrase = _secret_input_or_quit("Passphrase: ")
            admin_private_key_protection = "openssh_passphrase" if passphrase else "none"
            generated_private, generated_public = _generate_keypair(
                key_id=admin_key_id,
                passphrase=passphrase or None,
            )
            admin_private_key_text = generated_private
            admin_public_key = str(generated_public).strip()
            if interactive:
                print("")
                _print_title("Generated Admin Private Key")
                _kv_rows(
                    [
                        ("export_now", "write a private-key file that can be imported into a consumer realm"),
                        (
                            "store_for_later",
                            "keep it in this machine's default client realm and print the later export/import command",
                        ),
                    ]
                )
                export_private = _bool_prompt("Export generated private key to a file now?", export_private)
                if export_private and export_private_path is None:
                    default_export_path = hosting_root / "keyring" / f"{admin_key_id}.private"
                    export_path_raw = _input_or_quit(f"Private key export path [{default_export_path}]: ")
                    export_private_path = Path(export_path_raw).expanduser().resolve() if export_path_raw else default_export_path
            if export_private and export_private_path is not None:
                export_private_path.parent.mkdir(parents=True, exist_ok=True)
                export_private_path.write_text(str(generated_private), encoding="utf-8")
                private_key_storage = "exported_file"
            else:
                admin_private_key_secret_realm = "default"
                client_realm_root = _client_realm_root(default_config_dir, admin_private_key_secret_realm)
                secret_store = FileSecretStore(client_realm_root, realm=admin_private_key_secret_realm)
                secret_record = secret_store.put_secret(
                    tag="rbac_private_key",
                    payload=str(generated_private),
                    secret_id=f"rbac-{admin_key_id}-private",
                    metadata={
                        "key_id": admin_key_id,
                        "role": "admin",
                        "auth_method": "public_key",
                        "source": "hosting_config_generate",
                        "private_key_format": "openssh",
                        "private_key_protection": admin_private_key_protection,
                    },
                    encryption="none",
                )
                admin_private_key_secret_id = secret_record.secret_id
                admin_private_key_secret_path = secret_record_path(client_realm_root, secret_record.secret_id)
                admin_private_key_secret_encryption = str(secret_record.encryption)
                private_key_storage = "client_realm_secret"
                private_key_warning = None
                client_keys_file = ensure_client_realm_dirs(client_realm_root)["keys"]
                _store_importable_key_record(
                    keys_file=client_keys_file,
                    key_id=admin_key_id,
                    role="admin",
                    auth_method="public_key",
                    public_key=admin_public_key,
                    key_source=key_source,
                    key_origin=key_origin,
                    public_key_source=public_key_source,
                    private_key_storage=private_key_storage,
                    private_key_secret_id=admin_private_key_secret_id,
                    private_key_secret_realm=admin_private_key_secret_realm,
                    private_key_protection=admin_private_key_protection,
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
        private_key_openssh=(
            admin_private_key_text
            if private_key_storage == "embedded_keyring"
            else None
        ),
        key_source=key_source,
        key_origin=key_origin,
        public_key_source=public_key_source,
        private_key_storage=private_key_storage,
        private_key_export_path=str(export_private_path) if export_private_path else None,
        private_key_secret_id=admin_private_key_secret_id,
        private_key_secret_realm=admin_private_key_secret_realm,
        private_key_protection=admin_private_key_protection if private_key_storage == "client_realm_secret" else None,
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
                *(
                    [(_client_realm_root(default_config_dir, admin_private_key_secret_realm or "default"))]
                    if admin_private_key_secret_id
                    else []
                ),
                *(
                    [admin_private_key_secret_path]
                    if admin_private_key_secret_path is not None
                    else []
                ),
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
    final_setup_context = _setup_context_from_config(
        base=setup_context,
        usage_intent=usage_intent,
        connectivity_mode=mode,
        endpoint_mode=endpoint_mode,
        lifecycle_profile=lifecycle_profile,
        require_auth=require_auth,
        key_source=key_source,
    )
    _write_json(
        bootstrap_state_file,
        {
            "version": 1,
            "updated_at": time.time(),
            "setup": {
                "setup_scope": setup_scope,
                "setup_context": final_setup_context,
                "setup_notes": setup_notes,
                "connectivity_mode": mode,
                "endpoint_mode_default": endpoint_mode,
                "lifecycle_profile": lifecycle_profile,
                "require_auth": require_auth,
                "admin_key_id": admin_key_id,
                "key_source": key_source,
                "key_action": key_action,
                "permission_action": permission_action,
                "admin_private_key_storage": private_key_storage,
                "admin_private_key_export_path": str(export_private_path) if export_private_path else None,
                "admin_private_key_secret_id": admin_private_key_secret_id,
                "admin_private_key_secret_realm": admin_private_key_secret_realm,
                "admin_private_key_secret_encryption": admin_private_key_secret_encryption,
                "admin_private_key_protection": admin_private_key_protection,
            },
            "files": {
                "control_state_file": str(control_state_path),
                "access_control_file": str(access_file),
                "keys_file": str(keys_file),
                "client_mapping_file": str(mappings_file),
                "audit_file": str(audit_file),
                "client_realm_root": (
                    str(_client_realm_root(default_config_dir, admin_private_key_secret_realm or "default"))
                    if admin_private_key_secret_id
                    else None
                ),
                "client_realm_secret_path": str(admin_private_key_secret_path) if admin_private_key_secret_path else None,
            },
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
            "key_action": key_action,
            "admin_key_origin": key_origin,
            "admin_public_key_source": public_key_source,
            "admin_private_key_storage": private_key_storage,
            "admin_private_key_export_path": str(export_private_path) if export_private_path else None,
            "admin_private_key_secret_id": admin_private_key_secret_id,
            "admin_private_key_secret_realm": admin_private_key_secret_realm,
            "admin_private_key_secret_path": str(admin_private_key_secret_path) if admin_private_key_secret_path else None,
            "admin_private_key_secret_encryption": admin_private_key_secret_encryption,
            "admin_private_key_protection": admin_private_key_protection,
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
    if bool(export_private and export_private_path):
        changes.append(f"generated private key exported to {export_private_path}")
    if bool(admin_private_key_secret_id and admin_private_key_secret_path):
        changes.append(f"generated private key stored in client realm secret {admin_private_key_secret_path}")
        setup_client_realm_root = _client_realm_root(default_config_dir, admin_private_key_secret_realm or "default")
        admin_private_key_export_command = (
            "python -m hosting.hosting_config_cli --client-export-key "
            f"--client-key-id {admin_key_id} "
            f"--client-realm {admin_private_key_secret_realm or 'default'} "
            f"--client-realm-root {shlex.quote(str(setup_client_realm_root))} "
            "--client-export-key-path <private-key-file>"
        )
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
        "admin_public_key_path": str(admin_public_key_path) if admin_public_key_path else None,
        "admin_private_key_path": None,
        "private_key_exported": bool(export_private),
        "private_key_export_path": str(export_private_path) if export_private_path else None,
        "setup_scope": setup_scope,
        "setup_context": final_setup_context,
        "key_action": key_action,
        "permission_action": permission_action,
        "permission_result": permission_result,
        "changes": changes,
        "admin_key_origin": key_origin,
        "admin_public_key_source": public_key_source,
        "admin_private_key_storage": private_key_storage,
        "admin_private_key_path": str(export_private_path) if export_private_path else None,
        "admin_private_key_secret_id": admin_private_key_secret_id,
        "admin_private_key_secret_path": str(admin_private_key_secret_path) if admin_private_key_secret_path else None,
        "admin_private_key_secret_encryption": admin_private_key_secret_encryption,
        "admin_private_key_protection": admin_private_key_protection,
        "admin_private_key_export_command": admin_private_key_export_command,
        "admin_private_key_handoff": (
            "Run the export command on this setup machine, transfer the private-key file if needed, "
            "then import it into the consumer realm with --client-import-key."
            if admin_private_key_export_command
            else None
        ),
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
        enriched_details = dict(details or {})
        guidance = _doctor_guidance(name, bool(ok), enriched_details)
        for key, value in guidance.items():
            if str(value or "").strip() and key not in enriched_details:
                enriched_details[key] = value
        entry = {
            "check": name,
            "ok": bool(ok),
            "blocking": bool(blocking),
            "details": enriched_details,
        }
        checks.append(entry)
        if (not ok) and bool(blocking):
            issues.append(entry)

    try:
        proc = subprocess.run(  # noqa: S603
            ["ssh-keygen", "-?"],
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
    keys_payload = _read_json(paths["keys_file"], {"keys": {}})
    admin_key_count = len(
        [
            k
            for _, k in dict(keys_payload.get("keys") or {}).items()
            if str((k or {}).get("role") or "").strip().lower() == "admin"
        ]
    )
    access_artifacts_present = any(
        p.exists()
        for p in (
            paths["keys_file"],
            paths["mappings_file"],
            paths["bootstrap_state_file"],
            paths["audit_file"],
        )
    ) or admin_key_count > 0
    _record(
        "control_state_exists",
        control_state_path.exists(),
        {
            "path": str(control_state_path),
            "access_artifacts_present": access_artifacts_present,
            "admin_key_count": admin_key_count,
        },
        blocking=access_artifacts_present,
    )

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
        _record(
            "hosting_root_writable",
            False,
            {"path": str(hosting_root), "error": "missing_directory"},
            blocking=False,
        )

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
        _record(
            "control_config_readable",
            True,
            {
                "require_auth": bool(cfg.get("require_auth", False)),
                "source": "access_control_file" if control_state_path.exists() else "service_defaults_no_control_state_file",
            },
        )
        connectivity_mode = _normalize_mode(
            str(dict(cfg.get("access_profile") or {}).get("connectivity_mode") or "local_only"),
            "local_only",
        )
        keys_count = int(cfg.get("keys_count") or 0)
        require_auth = bool(cfg.get("require_auth", False))
        zero_key_remote_ok = not (require_auth and connectivity_mode != "local_only" and keys_count == 0)
        zero_key_remote_details = {
            "require_auth": require_auth,
            "connectivity_mode": connectivity_mode,
            "keys_count": keys_count,
        }
        if not zero_key_remote_ok:
            zero_key_remote_details["error"] = "Remote-capable mode requires a pre-provisioned key before auth can work."
        _record("zero_key_remote_bootstrap_policy", zero_key_remote_ok, zero_key_remote_details)
        try:
            svc.assert_runtime_policy_safe()
            _record("runtime_policy_safe", True)
        except Exception as exc:
            _record("runtime_policy_safe", False, {"error": str(exc)})
    except Exception as exc:
        _record("control_config_readable", False, {"error": str(exc)})

    summary = _summarize_existing_config(
        control_state_path=control_state_path,
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
    )
    admin_key_id = str(summary.get("admin_key_id") or "").strip()
    key_meta = _admin_key_metadata(paths["keys_file"], admin_key_id) if admin_key_id else {}
    if key_meta:
        storage = str(key_meta.get("private_key_storage") or "")
        if storage == "client_realm_secret":
            secret_exists = bool(key_meta.get("private_key_secret_exists"))
            _record(
                "admin_client_secret_present",
                secret_exists,
                {
                    "key_id": admin_key_id,
                    "secret_id": key_meta.get("private_key_secret_id"),
                    "secret_path": key_meta.get("private_key_secret_path"),
                },
            )
            protection = str(key_meta.get("private_key_protection") or "").strip() or "none"
            _record(
                "admin_client_secret_storage_recorded",
                True,
                {
                    "key_id": admin_key_id,
                    "secret_id": key_meta.get("private_key_secret_id"),
                    "private_key_protection": protection,
                },
            )
        elif storage == "exported_file":
            export_exists = bool(key_meta.get("private_key_export_exists"))
            export_purged = bool(key_meta.get("private_key_export_purged_at"))
            export_purged_without_adoption = bool(key_meta.get("private_key_export_purged_without_adoption_at"))
            export_ok = bool(export_purged)
            export_blocking = bool((not export_exists) and (not export_purged) and (not export_purged_without_adoption))
            _record(
                "admin_exported_private_key_custody",
                export_ok,
                {
                    "key_id": admin_key_id,
                    "path": key_meta.get("private_key_export_path"),
                    "exists": export_exists,
                    "purged_after_adoption": export_purged,
                    "purged_without_adoption": export_purged_without_adoption,
                    "adopted_client_realm_root": key_meta.get("private_key_adopted_client_realm_root"),
                    "recommendation": (
                        "Hand off the exported private key into a consumer client realm, then purge the loose exported key file."
                        if export_exists
                        else (
                            "No action needed; exported private key file was marked purged after client-realm hand-off."
                            if export_purged
                            else (
                                "Verify another private-key copy exists or rotate the admin key; the exported file was purged without recorded hand-off."
                                if export_purged_without_adoption
                                else "Restore the exported private key file, import another private key into the consumer realm, or rotate the admin key."
                            )
                        )
                    ),
                },
                blocking=export_blocking,
            )
        elif storage == "embedded_keyring":
            _record(
                "admin_embedded_private_key_legacy",
                False,
                {"key_id": admin_key_id, "storage": storage},
                blocking=False,
            )

    try:
        client_access = read_client_access(get_default_client_realm_root(default_config_dir=default_config_dir))
        profiles = dict(client_access.get("client_access", {}).get("profiles") or {})
        _record(
            "client_realm_access_readable",
            True,
            {"profiles_count": len(profiles)},
            blocking=False,
        )
        bad_profiles: list[Dict[str, Any]] = []
        for name, row in sorted(profiles.items()):
            profile_row = dict(row or {})
            known_hosts_file = str(profile_row.get("ssh_known_hosts_file") or "").strip()
            secret_id = str(profile_row.get("control_ssh_key_secret_id") or "").strip()
            if known_hosts_file and not Path(known_hosts_file).expanduser().resolve().exists():
                bad_profiles.append(
                    {"profile_name": name, "error": "missing_known_hosts_file", "path": known_hosts_file}
                )
            if secret_id:
                secret_path = secret_record_path(
                    get_default_client_realm_root(default_config_dir=default_config_dir),
                    secret_id,
                )
                if not secret_path.exists():
                    bad_profiles.append(
                        {"profile_name": name, "error": "missing_secret_record", "path": str(secret_path)}
                    )
        _record(
            "client_transport_profiles_integrity",
            not bad_profiles,
            {"invalid_profiles": bad_profiles, "profiles_count": len(profiles)},
        )
    except Exception as exc:
        _record("client_realm_access_readable", False, {"error": str(exc)}, blocking=False)

    transport_key_id = str(getattr(args, "transport_key_id", "") or "").strip()
    auth_file_raw = str(getattr(args, "ssh_authorized_keys_file", "") or "").strip()
    if transport_key_id or auth_file_raw:
        auth_file = (
            Path(auth_file_raw).expanduser().resolve()
            if auth_file_raw
            else (Path.home() / ".ssh" / "authorized_keys").resolve()
        )
        block_text = ""
        public_key = str(getattr(args, "transport_public_key_inline", "") or "").strip()
        if not public_key and str(getattr(args, "transport_public_key_file", "") or "").strip():
            try:
                public_key = Path(str(args.transport_public_key_file)).expanduser().resolve().read_text(encoding="utf-8").strip()
            except Exception:
                public_key = ""
        if auth_file.exists():
            lines = auth_file.read_text(encoding="utf-8").splitlines()
            begin = f"# BEGIN mp13-hosting-transport {transport_key_id or 'transport'}"
            end = f"# END mp13-hosting-transport {transport_key_id or 'transport'}"
            in_block = False
            block_lines: list[str] = []
            for line in lines:
                if line.strip() == begin:
                    in_block = True
                    block_lines.append(line)
                    continue
                if in_block:
                    block_lines.append(line)
                    if line.strip() == end:
                        break
            block_text = "\n".join(block_lines)
        _record(
            "transport_authorized_key_present",
            bool(block_text and (not public_key or public_key in block_text)),
            {
                "authorized_keys_file": str(auth_file),
                "transport_key_id": transport_key_id or "transport",
                "public_key_checked": bool(public_key),
            },
        )
        hardened = all(
            item in block_text
            for item in (
                'command="',
                "no-pty",
                "no-agent-forwarding",
                "no-X11-forwarding",
                "no-port-forwarding",
            )
        )
        _record(
            "transport_authorized_key_hardened",
            hardened,
            {
                "authorized_keys_file": str(auth_file),
                "transport_key_id": transport_key_id or "transport",
            },
        )
        if transport_key_id:
            keys_payload = _read_json(paths["keys_file"], {"keys": {}})
            key_meta = dict(dict(keys_payload.get("keys") or {}).get(transport_key_id) or {})
            rbac_ok = (
                str(key_meta.get("role") or "") == "transport"
                and str(key_meta.get("auth_method") or "") == "public_key"
                and not bool(key_meta.get("disabled", False))
            )
            _record(
                "transport_rbac_registered",
                rbac_ok,
                {
                    "transport_key_id": transport_key_id,
                    "role": key_meta.get("role"),
                    "auth_method": key_meta.get("auth_method"),
                    "disabled": bool(key_meta.get("disabled", False)) if key_meta else None,
                },
            )
            if public_key:
                _record(
                    "transport_rbac_matches_ssh",
                    str(key_meta.get("public_key") or "").strip() == public_key,
                    {
                        "transport_key_id": transport_key_id,
                        "public_key_checked": True,
                    },
                )

    warnings = [dict(row or {}) for row in checks if not bool((row or {}).get("ok")) and not bool((row or {}).get("blocking", True))]
    return {
        "status": "issues_found" if issues else "warnings_found" if warnings else "ok",
        "issues_count": len(issues),
        "warnings_count": len(warnings),
        "checks": checks,
        "issues": issues,
        "warnings": warnings,
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
    p.add_argument("--usage-intent", default="single_admin", choices=sorted(VALID_USAGE_INTENTS), help=argparse.SUPPRESS)
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
    p.add_argument("--client-list-keys", action="store_true", help="List client-realm private-key metadata and exit")
    p.add_argument("--client-list-exported-keys", action="store_true", help="List exported private-key file references from a keyring")
    p.add_argument("--client-generate-key", action="store_true", help="Generate a client-realm private key and metadata record")
    p.add_argument("--client-import-key", action="store_true", help="Import a client private key into the client realm")
    p.add_argument("--client-handoff-exported-key", action="store_true", help="Move a local exported private-key file into the client realm")
    p.add_argument("--client-adopt-exported-key", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--client-purge-exported-key", action="store_true", help="Delete a tracked exported private-key file without importing it")
    p.add_argument("--client-export-key", action="store_true", help="Export a client-realm private key to a file")
    p.add_argument("--revoke-key-id", default="", help="Revoke one RBAC key_id and its sessions, then exit")
    p.add_argument("--revoke-session", default="", help="Revoke one session token and exit")
    p.add_argument("--key-id", default="", help="RBAC key_id for --upsert-key")
    p.add_argument("--client-key-id", default="", help="Client-realm key id for local key lifecycle commands")
    p.add_argument("--client-key-tag", default="rbac_private_key", help="Client secret tag: rbac_private_key or transport_private_key")
    p.add_argument("--client-private-key-file", default="", help="Private key file for --client-import-key")
    p.add_argument("--client-private-key", default="", help="Inline private key text for --client-import-key; cleared from args after read")
    p.add_argument("--client-public-key-file", default="", help="Public key file for --client-import-key")
    p.add_argument("--client-public-key-inline", default="", help="Inline public key for --client-import-key")
    p.add_argument("--client-export-key-path", default="", help="Output file for --client-export-key")
    p.add_argument("--client-exported-keys-file", default="", help="Source keyring for --client-list-exported-keys/--client-handoff-exported-key")
    p.add_argument("--client-delete-exported-key-file", action="store_true", help="Delete source exported private-key file after --client-handoff-exported-key")
    p.add_argument("--key-role", default="", choices=sorted(VALID_AUTH_ROLES), help="RBAC role for --upsert-key")
    p.add_argument(
        "--auth-method",
        default="public_key",
        choices=["public_key", "shared_secret"],
        help="Authentication method for --upsert-key. shared_secret can issue sessions only in local_only mode; remote modes require public_key challenge.",
    )
    p.add_argument("--public-key-file", default="", help="Public key file for --upsert-key")
    p.add_argument("--public-key-inline", default="", help="Inline public key for --upsert-key")
    p.add_argument("--key-secret", default="", help="Shared secret for local_only --upsert-key when auth-method=shared_secret")
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
    p.add_argument("--key-action", default="replace", choices=["keep_existing", "replace"], help=argparse.SUPPRESS)
    p.add_argument("--permission-action", default="none", choices=["none", "tighten"], help=argparse.SUPPRESS)
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
    p.add_argument("--client-realm", default="default", help="Client realm name for client-local secret/profile operations")
    p.add_argument("--client-realm-root", default="", help="Override client realm root path")
    p.add_argument("--transport-harden-ssh", action="store_true", help="Provision and validate hardened SSH transport mutual-auth artifacts")
    p.add_argument("--transport-admin-setup", action="store_true", help="Generate or execute elevated SSH service/firewall setup")
    p.add_argument("--transport-export-bootstrap", action="store_true", help="Export a transport bootstrap bundle and exit")
    p.add_argument("--transport-import-bootstrap", action="store_true", help="Import a transport bootstrap bundle into the client realm and exit")
    p.add_argument("--transport-validate-profile", action="store_true", help="Validate an imported transport client profile and exit")
    p.add_argument("--transport-provision-ssh-artifacts", action="store_true", help="Materialize a transport profile key and write a realm-local SSH config snippet")
    p.add_argument("--transport-install-authorized-key", action="store_true", help="Install a transport public key into a user-scoped authorized_keys file")
    p.add_argument("--bootstrap-bundle-file", default="", help="Transport bootstrap bundle file path for export/import")
    p.add_argument("--transport-target", default="", help="SSH target for transport bootstrap export")
    p.add_argument("--transport-key-id", default="", help="Transport key id for transport bootstrap export")
    p.add_argument("--transport-public-key-file", default="", help="Transport public key file for transport bootstrap export")
    p.add_argument("--transport-public-key-inline", default="", help="Inline transport public key for transport bootstrap export")
    p.add_argument("--transport-private-key-file", default="", help="Transport private key file for transport bootstrap export")
    p.add_argument("--transport-private-key-inline", default="", help="Inline transport private key for transport bootstrap export")
    p.add_argument("--ssh-known-hosts-file", default="", help="Known hosts line file for transport bootstrap export")
    p.add_argument("--ssh-known-hosts-line", default="", help="Inline known hosts line for transport bootstrap export")
    p.add_argument("--transport-profile-name", default="", help="Suggested/imported client profile name for transport bootstrap")
    p.add_argument("--control-ssh-fingerprint", default="", help="Optional SSH host fingerprint metadata for transport bootstrap")
    p.add_argument("--overwrite-profile", action="store_true", default=False, help="Allow transport bootstrap import to overwrite an existing profile")
    p.add_argument("--bootstrap-password", default="", help="Password for encrypting or decrypting transport bootstrap private-key payloads")
    p.add_argument("--client-secret-password", default="", help="Password for storing or materializing encrypted client-realm secret records")
    p.add_argument("--validation-no-ssh-run", action="store_true", default=False, help="Validate transport profile files/settings without running SSH")
    p.add_argument("--validation-ssh-bin", default="ssh", help="SSH binary used for transport profile validation")
    p.add_argument("--validation-remote-command", default="exit 0", help="Remote command for transport profile SSH validation")
    p.add_argument("--validation-timeout-seconds", type=float, default=15.0, help="Timeout for transport profile SSH validation")
    p.add_argument("--ssh-config-alias", default="", help="Host alias for --transport-provision-ssh-artifacts")
    p.add_argument("--overwrite-ssh-config", action="store_true", default=False, help="Overwrite existing realm-local SSH config snippet")
    p.add_argument("--ssh-authorized-keys-file", default="", help="authorized_keys path for --transport-install-authorized-key; defaults to ~/.ssh/authorized_keys")
    p.add_argument(
        "--ssh-authorized-key-command",
        default=DEFAULT_TRANSPORT_AUTHORIZED_KEY_COMMAND,
        help="Forced command for --transport-install-authorized-key",
    )
    p.add_argument(
        "--ssh-authorized-key-unrestricted",
        action="store_true",
        default=False,
        help="Install transport key without forced command or SSH option restrictions",
    )
    p.add_argument(
        "--admin-capability",
        default="no_admin_available",
        choices=sorted(VALID_ADMIN_CAPABILITIES),
        help="Target-host admin/root availability for SSH transport recommendations",
    )
    p.add_argument("--admin-setup-execute", action="store_true", default=False, help="Execute admin setup through platform-native elevation")
    p.add_argument("--admin-setup-enable-ssh-service", action="store_true", default=True, help="Enable/start the platform SSH server service")
    p.add_argument("--admin-setup-no-ssh-service", dest="admin_setup_enable_ssh_service", action="store_false", help="Do not enable/start the SSH server service")
    p.add_argument("--admin-setup-enable-firewall", action="store_true", default=False, help="Add SSH firewall allowance where a supported firewall helper exists")
    p.add_argument("--admin-setup-enable-user-linger", action="store_true", default=False, help="Enable systemd user linger where supported")
    p.add_argument("--admin-setup-target-user", default="", help="User account for user-linger setup; defaults to invoking user")
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
        elif bool(
            args.client_list_keys
            or args.client_list_exported_keys
            or args.client_generate_key
            or args.client_import_key
            or args.client_handoff_exported_key
            or args.client_adopt_exported_key
            or args.client_purge_exported_key
            or args.client_export_key
        ):
            result = run_client_keys(args)
            _print_client_key_report(result)
        elif bool(
            args.transport_admin_setup
        ):
            result = run_transport_admin_setup(args)
            _print_admin_setup_report(result)
        elif bool(
            args.transport_export_bootstrap
            or args.transport_harden_ssh
            or args.transport_import_bootstrap
            or args.transport_validate_profile
            or args.transport_provision_ssh_artifacts
            or args.transport_install_authorized_key
        ):
            result = run_transport_bootstrap(args)
            _print_transport_bootstrap_report(result)
        else:
            result = run_setup(args)
            if not bool(args.interactive):
                _print_setup_result_report(result)
        if bool(args.json_output):
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        if isinstance(exc, UserCancelled):
            if bool(getattr(args, "json_output", False)):
                print(json.dumps({"ok": False, "cancelled": True, "error": str(exc)}, ensure_ascii=False))
            else:
                via_keyboard = bool(getattr(exc, "via_keyboard", False))
                if via_keyboard:
                    _print_staged_setup_dropped(via_keyboard=True)
                    _clear_pending_staged_setup()
                elif _has_pending_staged_setup():
                    _print_pending_staged_setup()
                    if _plain_yes_no("Save these staged setup changes now?", False):
                        result = _save_pending_staged_setup(args)
                        _print_setup_result_report(result)
                        if bool(args.json_output):
                            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
                    else:
                        _print_staged_setup_dropped(via_keyboard=False)
                        _clear_pending_staged_setup()
                else:
                    _print_staged_setup_dropped(via_keyboard=False)
            return 0
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
