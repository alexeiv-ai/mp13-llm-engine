from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .hosting_config_cli import (
    _c,
    _print_title,
    _print_block,
    _prompt_menu,
    _kv_rows,
    _set_color_scheme,
    UserCancelled,
)
from .engine_host_channel import EngineHostControlChannel
from .transport_bootstrap import _protect_windows_private_key_path

_TOKEN_UNSET = object()


def _arg_value(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _control_channel_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "engine_host_daemon_auto_bootstrap": False,
        "engine_host_daemon_pid_file": str(_arg_value(args, "pid_file") or "") or None,
        "engine_host_state_file": str(_arg_value(args, "engines_state_file") or "") or None,
        "engine_host_control_state_file": str(_arg_value(args, "control_state_file") or "") or None,
    }
    for attr in (
        "engine_host_cmd",
        "engine_host_remote_cmd",
        "engine_host_ssh_target",
        "control_endpoint",
        "control_ssh_key",
        "control_ssh_fingerprint",
        "ssh_known_hosts_line",
        "engine_host_client_profile",
        "engine_host_client_realm",
        "engine_host_client_realm_root",
        "engine_host_client_secret_password",
        "engine_host_timeout_seconds",
        "engine_host_daemon_port",
        "engine_host_daemon_log_file",
        "engine_host_session_scope",
        "engine_host_session_ttl_seconds",
        "engine_host_bind_session_to_ssh",
    ):
        value = _arg_value(args, attr, None)
        if value not in (None, ""):
            settings[attr] = value
    return settings


def _control_channel(args: argparse.Namespace, session_token: object = _TOKEN_UNSET) -> EngineHostControlChannel:
    channel = getattr(args, "_interactive_control_channel", None)
    if not isinstance(channel, EngineHostControlChannel):
        channel = EngineHostControlChannel(_control_channel_settings(args))
        setattr(args, "_interactive_control_channel", channel)
    if session_token is not _TOKEN_UNSET:
        channel.set_session_token(session_token if isinstance(session_token, str) else None)
    return channel


def _raise_interactive_api_error(exc: Exception) -> None:
    msg = str(exc)
    if (
        "session_token_required" in msg
        or "missing_or_invalid_session_token" in msg
        or "auth_failed" in msg
        or "invalid_session" in msg
        or "session_expired" in msg
    ):
        raise PermissionError("session_token_required") from exc
    raise exc


def _api_invoke(args: argparse.Namespace, cmd: str, payload: dict, session_token: Optional[str] = None) -> Any:
    channel = _control_channel(args, session_token=session_token)
    try:
        return channel.invoke_control_command(str(cmd or "").strip(), dict(payload or {}))
    except Exception as exc:
        _raise_interactive_api_error(exc)


def _offline_service(args: argparse.Namespace):
    from .service.host_service import EngineHostService

    return EngineHostService(
        engines_state_file=_arg_value(args, "engines_state_file", None),
        control_state_file=_arg_value(args, "control_state_file", None),
    )


def _offline_local_invoke(args: argparse.Namespace, cmd: str, payload: dict, session_token: Optional[str] = None) -> Any:
    payload_copy = dict(payload or {})
    if session_token:
        payload_copy["session_token"] = session_token
    svc = _offline_service(args)
    svc.authorize_command(cmd, payload_copy)
    if cmd == "discover-running":
        return svc.discover_running()
    if cmd == "host-metrics":
        return svc.get_host_metrics()
    if cmd == "auth-list-sessions":
        return svc.auth_list_sessions()
    if cmd == "auth-begin-challenge":
        return svc.auth_begin_challenge(
            key_id=str(payload_copy.get("key_id") or ""),
            scope=str(payload_copy.get("scope") or "control"),
            ttl_seconds=int(payload_copy.get("ttl_seconds") or 120),
            config_paths=list(payload_copy.get("config_paths") or []),
            engine_ids=list(payload_copy.get("engine_ids") or []),
            ssh_binding=dict(payload_copy.get("ssh_binding") or {}),
        )
    if cmd == "auth-complete-challenge":
        return svc.auth_complete_challenge(
            challenge_id=str(payload_copy.get("challenge_id") or ""),
            signature_ssh=str(payload_copy.get("signature_ssh") or ""),
            presented_ssh_binding=dict(payload_copy.get("_ssh_session_binding") or {}),
        )
    raise RuntimeError(f"Offline local fallback is not available for {cmd}")


def _can_use_offline_local_fallback(args: argparse.Namespace, session_token: Optional[str] = None) -> bool:
    return _target_mode(args) != "ssh" and not _is_daemon_running(args, session_token=session_token)


def _print_offline_auth_required() -> None:
    print(_c('warn', "  Daemon is stopped. This offline read is protected by hosting auth."))
    print(_c('muted', "  Authenticate locally with an admin private key to continue."))


def _offline_read_unavailable(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "session_token_required" in msg
        or "missing_or_invalid_session_token" in msg
        or "auth_failed" in msg
        or "invalid_session" in msg
        or "session_expired" in msg
    )


def _offline_local_read_with_auth(
    args: argparse.Namespace,
    cmd: str,
    payload: dict,
    session_token: Optional[str],
) -> tuple[Optional[Any], Optional[str]]:
    try:
        return _offline_local_invoke(args, cmd, payload, session_token=session_token), session_token
    except PermissionError as exc:
        if not _offline_read_unavailable(exc):
            raise
        _print_offline_auth_required()
        token = _local_authenticate(args)
        if not token:
            print(_c('bad', "Command failed: Authentication required."))
            return None, session_token
        return _offline_local_invoke(args, cmd, payload, session_token=token), token


def _is_daemon_running(args: argparse.Namespace, session_token: Optional[str] = None) -> bool:
    channel = _control_channel(args, session_token=session_token)
    if str(channel.get_target().get("mode") or "local") == "ssh":
        try:
            _api_invoke(args, "host-metrics", {}, session_token=session_token)
            return True
        except PermissionError:
            return True
        except Exception:
            return False
    status = channel.get_daemon_status()
    return bool(status.get("alive") or status.get("reachable"))


def _sandbox_enabled(info: Dict[str, Any]) -> bool:
    summary = dict(info.get("sandbox") or {})
    if "enabled" in summary:
        return bool(summary.get("enabled"))
    policy = dict(info.get("sandbox_policy") or {})
    nested = dict(policy.get("sandbox") or {})
    return bool(policy.get("enabled") or nested.get("enabled"))


def _operator_resource_state(info: Dict[str, Any]) -> str:
    state = str(info.get("state") or "").strip()
    if state:
        return state
    if bool(info.get("alive")):
        if "reachable" in info and not bool(info.get("reachable")):
            return "unreachable"
        return "running"
    if "alive" in info:
        return "stopped"
    return "unknown"


def _operator_resource_kind(info: Dict[str, Any]) -> str:
    kind = str(info.get("kind") or "").strip()
    if kind:
        return kind
    executor_kind = str(info.get("executor_kind") or "").strip()
    worker_class = str(info.get("worker_profile_class") or "").strip().lower()
    command_text = " ".join(str(x) for x in list(info.get("command") or [])).lower()
    env = {str(k): str(v) for k, v in dict(info.get("env") or {}).items()}
    sandbox_enabled = _sandbox_enabled(info)
    is_toolbox = (
        executor_kind == "toolbox_executor"
        or "hosting.toolbox_executor_ipc" in command_text
        or "MP13_TOOLBOX_EXECUTOR_ENGINE_ID" in env
        or isinstance(info.get("tool_access"), dict)
    )
    if is_toolbox:
        return "tools sandbox" if sandbox_enabled else "tools worker"
    is_model = (
        worker_class == "model"
        or "MP13_MODEL_PATH" in env
        or "hosting.engine_worker_ipc" in command_text
    )
    if is_model:
        return "sandboxed model instance" if sandbox_enabled else "model instance"
    if worker_class == "generic":
        return "sandboxed worker" if sandbox_enabled else "generic worker"
    return "sandboxed worker" if sandbox_enabled else "worker"


def _target_mode(args: argparse.Namespace) -> str:
    return str(_control_channel(args).get_target().get("mode") or "local")


def _key_id_from_secret_id(secret_id: str) -> str:
    sid = str(secret_id or "").strip()
    for prefix in ("rbac-", "transport-"):
        if sid.startswith(prefix):
            sid = sid[len(prefix):]
            break
    for suffix in ("-private", ".private"):
        if sid.endswith(suffix):
            sid = sid[: -len(suffix)]
            break
    return sid.strip()


def _extract_key_id_from_private_key_json(payload: Dict[str, Any]) -> Optional[str]:
    data = dict(payload or {})
    metadata = dict(data.get("metadata") or {})
    for source in (metadata, data):
        key_id = str(source.get("key_id") or source.get("admin_key_id") or "").strip()
        if key_id:
            return key_id
    derived = _key_id_from_secret_id(str(data.get("secret_id") or ""))
    return derived or None


def _obtain_session_token(
    args: argparse.Namespace,
    *,
    invoke_fn: Optional[Callable[[str, dict], Any]] = None,
) -> Optional[str]:
    print(f"\n{_c('warn', 'Authentication required. Please provide an admin private key.')}")
    print(_c('muted', "You can paste the private key content, a JSON SecretRecord blob (end with an empty line), or provide a file path."))
    lines = []
    while True:
        try:
            line = input("> ")
        except (KeyboardInterrupt, EOFError):
            return None
        if not line and not lines:
            continue
        if not line and lines:
            break
        lines.append(line)

    input_text = "\n".join(lines).strip()
    if not input_text:
        return None

    pk_file_path = None
    pk_text = input_text
    is_json = False
    json_payload: Optional[Dict[str, Any]] = None

    # Check if it's a file path
    possible_path = lines[0].strip()
    if possible_path.startswith('"') and possible_path.endswith('"'):
        possible_path = possible_path[1:-1]
    elif possible_path.startswith("'") and possible_path.endswith("'"):
        possible_path = possible_path[1:-1]

    if len(lines) == 1 and os.path.isfile(os.path.expanduser(possible_path)):
        try:
            pk_file_path = Path(os.path.expanduser(possible_path)).resolve()
            pk_text = pk_file_path.read_text(encoding="utf-8").strip()
            input_text = pk_text  # Update input_text so metadata extraction works if it's JSON
        except Exception as e:
            print(_c('bad', f"Failed to read file: {e}"))
            return None
    else:
        pk_text = input_text
    # Check if it's a JSON SecretRecord
    if pk_text.startswith("{") and pk_text.endswith("}"):
        try:
            payload = json.loads(pk_text)
            if "payload" in payload and "secret_id" in payload:
                json_payload = dict(payload)
                pk_text = payload["payload"].strip()
                # Check for nested formatting
                if pk_text.startswith('-----BEGIN') and '\\n' in pk_text:
                    pk_text = pk_text.replace('\\n', '\n')
                is_json = True
            else:
                 print(_c('bad', "JSON provided does not look like a valid SecretRecord (missing 'payload' or 'secret_id')."))
                 return None
        except json.JSONDecodeError:
            # Not JSON, assume it's raw key text
            pass

    if pk_text.startswith('-----BEGIN') and '\\n' in pk_text:
        pk_text = pk_text.replace('\\n', '\n')
            
    pk_text = pk_text.strip() + "\n"
    
    if not pk_text.strip():
        return None

    tmpdir = Path(tempfile.mkdtemp(prefix="host_cli_auth_")).resolve()
    try:
        _protect_windows_private_key_path(tmpdir)
        
        if pk_file_path and not is_json:
            pk_file = pk_file_path
        else:
            pk_file = tmpdir / "private_key"
            pk_file.write_text(pk_text, encoding="utf-8")
            _protect_windows_private_key_path(pk_file)

        key_id = "admin-main"
        key_id_from_json = False
        # If the user pasted a JSON blob, it might have metadata with the true key_id
        if json_payload is not None:
            parsed_key_id = _extract_key_id_from_private_key_json(json_payload)
            if parsed_key_id:
                key_id = parsed_key_id
                key_id_from_json = True
        
        if not key_id_from_json:
            try:
                print(_c('muted', f"Could not determine Key ID from input. Defaulting to '{key_id}'."))
                key_id_input = input(f"Key ID [{key_id}]: ").strip()
                if key_id_input:
                    key_id = key_id_input
            except (KeyboardInterrupt, EOFError):
                return None

        invoke = invoke_fn or (lambda cmd, payload: _api_invoke(args, cmd, payload))
        chal_res = invoke("auth-begin-challenge", {"key_id": key_id, "scope": "control"})
        challenge_id = chal_res.get("challenge_id")
        challenge_text = chal_res.get("challenge")
        if not challenge_id or not challenge_text:
            print(_c('bad', "Failed to get challenge from daemon."))
            return None

        chal_file = tmpdir / "challenge.txt"
        chal_file.write_text(challenge_text, encoding="utf-8")
        _protect_windows_private_key_path(chal_file)
        
        # We drop capture_output here so ssh-keygen can prompt for the passphrase interactively if needed.
        # It needs direct access to the console to prompt securely if SSH_ASKPASS isn't set.
        print("Signing challenge... (You may be prompted for your passphrase)")
        proc = subprocess.run(
            ["ssh-keygen", "-Y", "sign", "-f", str(pk_file), "-n", "engine-host-auth", str(chal_file)],
            check=False
        )
        if proc.returncode != 0:
            print(_c('bad', f"Failed to sign challenge (ssh-keygen exited with {proc.returncode})."))
            return None
            
        sig_file = tmpdir / "challenge.txt.sig"
        if not sig_file.exists():
             print(_c('bad', "Signature file was not created."))
             return None
             
        sig_text = sig_file.read_text(encoding="utf-8")
        
        comp_res = invoke("auth-complete-challenge", {
            "challenge_id": challenge_id,
            "signature_ssh": sig_text
        })
        
        token = comp_res.get("token")
        if token:
            print(_c('good', "Authenticated successfully."))
            return token
        else:
            print(_c('bad', "Authentication failed: no token returned."))
            return None
    except Exception as e:
        print(_c('bad', f"Authentication error: {e}"))
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def _get_token_preview(token: str) -> str:
    tok = str(token or "").strip()
    if not tok: return ""
    if len(tok) <= 15:
        return tok[: max(1, len(tok) // 2)] + "..."
    return f"{tok[:8]}...{tok[-4:]}"


def _local_recovery_header() -> None:
    _print_block("Local Recovery/Auth Tools")
    print(_c('warn', "  These actions read or edit local hosting control state directly."))
    print(_c('muted', "  They are local-only, not daemon RPC, and do not apply to remote targets."))
    print()


def _show_local_auth_status(args: argparse.Namespace) -> None:
    _local_recovery_header()
    svc = _offline_service(args)
    status = svc.auth_status()
    _kv_rows(
        [
            ("require_auth", status.get("require_auth")),
            ("keys_count", status.get("keys_count")),
            ("sessions_count", status.get("sessions_count")),
            ("challenges_count", status.get("challenges_count")),
            ("roles", ", ".join(list(status.get("roles") or []))),
            ("control_state_file", status.get("control_state_file") or getattr(svc, "control_state_file", "")),
        ]
    )


def _list_local_auth_keys(args: argparse.Namespace) -> None:
    _local_recovery_header()
    rows = _offline_service(args).auth_list_keys()
    if not rows:
        print("  No local auth keys registered.")
        return
    for row in rows:
        disabled = "disabled" if bool(row.get("disabled")) else "enabled"
        print(
            f"  - {_c('accent', row.get('key_id'))} "
            f"role={_c('value', row.get('role'))} "
            f"method={row.get('auth_method')} "
            f"{_c('warn' if disabled == 'disabled' else 'good', disabled)}"
        )


def _list_local_sessions(args: argparse.Namespace) -> None:
    _local_recovery_header()
    _print_sessions(_offline_service(args).auth_list_sessions(), session_token=None)


def _select_local_session(args: argparse.Namespace) -> Optional[str]:
    sessions = list(dict(_offline_service(args).auth_list_sessions() or {}).get("sessions") or [])
    if not sessions:
        print("  No local sessions to select.")
        return None
    opts: Dict[str, tuple[str, str]] = {}
    for sess in sessions:
        token = str(sess.get("token_preview") or sess.get("token_prefix") or "").strip()
        if not token:
            continue
        key_id = str(sess.get("key_id") or "<unknown>")
        scope = str(sess.get("scope") or "")
        role = str(sess.get("role") or "")
        hint = " ".join(part for part in (f"key={key_id}", f"scope={scope}" if scope else "", f"role={role}" if role else "") if part)
        opts[token] = (f"Session {token}", hint)
    if not opts:
        print("  No selectable local sessions found.")
        return None
    choice = _prompt_menu("Select Local Session", opts, "b", allow_back=True, allow_changes=False)
    return None if choice in {"b", "back"} else choice


def _select_local_key(args: argparse.Namespace) -> Optional[str]:
    rows = list(_offline_service(args).auth_list_keys() or [])
    if not rows:
        print("  No local auth keys to select.")
        return None
    opts: Dict[str, tuple[str, str]] = {}
    for row in rows:
        key_id = str(row.get("key_id") or "").strip()
        if not key_id:
            continue
        disabled = "disabled" if bool(row.get("disabled")) else "enabled"
        role = str(row.get("role") or "")
        method = str(row.get("auth_method") or "")
        opts[key_id] = (f"Key {key_id}", " ".join(part for part in (f"role={role}" if role else "", f"method={method}" if method else "", disabled) if part))
    if not opts:
        print("  No selectable local auth keys found.")
        return None
    choice = _prompt_menu("Select Local Auth Key", opts, "b", allow_back=True, allow_changes=False)
    return None if choice in {"b", "back"} else choice


def _confirm_local_mutation(prompt: str) -> bool:
    try:
        value = input(f"{prompt} Type LOCAL to confirm: ").strip()
    except (KeyboardInterrupt, EOFError):
        return False
    return value == "LOCAL"


def _local_authenticate(args: argparse.Namespace) -> Optional[str]:
    _local_recovery_header()
    token = _obtain_session_token(
        args,
        invoke_fn=lambda cmd, payload: _offline_local_invoke(args, cmd, payload),
    )
    if token:
        print(_c('good', f"Local session token acquired: {_get_token_preview(token)}"))
    return token


def _revoke_local_session(args: argparse.Namespace) -> None:
    _local_recovery_header()
    token = _select_local_session(args)
    if not token:
        return
    if not _confirm_local_mutation("Revoke this local session?"):
        print("  Cancelled.")
        return
    out = _offline_service(args).auth_revoke_session(token)
    _kv_rows([("revoked", out.get("revoked")), ("token", _get_token_preview(str(out.get("token") or token)))])


def _revoke_local_key(args: argparse.Namespace) -> None:
    _local_recovery_header()
    key_id = _select_local_key(args)
    if not key_id:
        return
    if not _confirm_local_mutation(f"Revoke local key {key_id!r}?"):
        print("  Cancelled.")
        return
    out = _offline_service(args).auth_revoke_key(key_id)
    _kv_rows([("key_id", out.get("key_id") or key_id), ("revoked", out.get("revoked"))])


def _clear_local_auth_keys_sessions(args: argparse.Namespace) -> None:
    _local_recovery_header()
    print(_c('warn', "  This does not reset hosting to unconfigured."))
    print(_c('muted', "  It stops the local daemon if possible, then clears only saved auth keys, sessions, and pending challenges."))
    print(_c('muted', "  Access policy such as require_auth, endpoint mode, lifecycle profile, and setup artifacts are kept."))
    print()
    if not _confirm_local_mutation("Clear local auth keys, sessions, and challenges?"):
        print("  Cancelled.")
        return
    result = _control_channel(args).reset_hosting_access()
    _kv_rows(
        [
            ("status", result.get("status")),
            ("daemon_stop", dict(result.get("daemon_stop") or {}).get("status")),
            ("cleared_keys", dict(result.get("auth_reset") or {}).get("cleared_keys")),
            ("cleared_sessions", dict(result.get("auth_reset") or {}).get("cleared_sessions")),
            ("cleared_challenges", dict(result.get("auth_reset") or {}).get("cleared_challenges")),
        ]
    )


def _force_stop_local_daemon(args: argparse.Namespace) -> None:
    _local_recovery_header()
    print(_c('warn', "  This forcibly stops registered workers, then terminates the local daemon PID if it is still alive."))
    print(_c('muted', "  Use this when the daemon is wedged/unreachable or an old daemon blocks startup."))
    print()
    if not _confirm_local_mutation("Force stop local daemon and registered workers?"):
        print("  Cancelled.")
        return
    result = _control_channel(args).force_stop_daemon(stop_workers=True)
    workers = dict(result.get("worker_shutdown") or {})
    term = dict(result.get("daemon_terminate") or {})
    graceful = dict(result.get("graceful_stop") or {})
    _kv_rows(
        [
            ("status", result.get("status")),
            ("workers_attempted", workers.get("attempted")),
            ("workers_stopped", workers.get("stopped")),
            ("workers_failed", workers.get("failed")),
            ("graceful_stop", graceful.get("status")),
            ("daemon_terminate", term.get("status")),
            ("daemon_pid", term.get("pid")),
        ]
    )


def _force_restart_local_daemon(args: argparse.Namespace) -> None:
    _local_recovery_header()
    print(_c('warn', "  This forcibly stops registered workers and the local daemon, then starts a fresh daemon."))
    print(_c('muted', "  Use this only when the existing daemon is stale, unreachable, or blocking startup."))
    print()
    if not _confirm_local_mutation("Force restart local daemon and registered workers?"):
        print("  Cancelled.")
        return
    result = _control_channel(args).force_restart_daemon(wait_ready_seconds=8.0)
    stop = dict(result.get("force_stop") or {})
    start = dict(result.get("start") or {})
    workers = dict(stop.get("worker_shutdown") or {})
    _kv_rows(
        [
            ("status", result.get("status")),
            ("workers_attempted", workers.get("attempted")),
            ("workers_stopped", workers.get("stopped")),
            ("workers_failed", workers.get("failed")),
            ("started_pid", start.get("pid")),
            ("started_port", start.get("port")),
            ("reachable", start.get("reachable") or start.get("alive") or start.get("already_running")),
            ("error", start.get("error") or start.get("reachability_error")),
        ]
    )


def _local_recovery_menu(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    if _target_mode(args) == "ssh":
        _print_block("Local Recovery/Auth Tools")
        print(_c('warn', "  Local recovery tools are not available for remote targets."))
        return session_token
    while True:
        opts = {
            "a": ("Show local auth status", ""),
            "u": ("Authenticate locally with admin private key", ""),
            "s": ("List local sessions", ""),
            "r": ("Revoke local session", ""),
            "k": ("List local auth keys", ""),
            "x": ("Revoke local auth key", ""),
            "z": ("Clear local auth keys/sessions", ""),
            "f": ("Force stop daemon and workers", ""),
            "n": ("Force restart daemon and workers", ""),
            "d": ("Start daemon after recovery", ""),
        }
        choice = _prompt_menu("Local Recovery/Auth Tools", opts, "b", allow_back=True, allow_changes=False)
        if choice in {"b", "back"}:
            return session_token
        if choice == "a":
            _show_local_auth_status(args)
        elif choice == "u":
            token = _local_authenticate(args)
            if token:
                session_token = token
        elif choice == "s":
            _list_local_sessions(args)
        elif choice == "r":
            _revoke_local_session(args)
        elif choice == "k":
            _list_local_auth_keys(args)
        elif choice == "x":
            _revoke_local_key(args)
        elif choice == "z":
            _clear_local_auth_keys_sessions(args)
            session_token = None
        elif choice == "f":
            _force_stop_local_daemon(args)
            session_token = None
        elif choice == "n":
            _force_restart_local_daemon(args)
            session_token = None
        elif choice == "d":
            _start_daemon(args)


def run_interactive_mode(args: argparse.Namespace) -> int:
    scheme = getattr(args, "color_scheme", "dark")
    _set_color_scheme(scheme)
    
    from . import hosting_config_cli as hc
    if scheme == "light":
        hc._COLOR_TOKENS.update({
            "title": "\033[1;35m", # Magenta
            "accent": "\033[0;35m",
            "rule": "\033[0;35m",
        })
    else:
        hc._COLOR_TOKENS.update({
            "title": "\033[1;95m", # Light Magenta
            "accent": "\033[0;95m",
            "rule": "\033[0;95m",
        })

    session_token = None
    last_status_c = None
    first_run = True

    try:
        while True:
            try:
                target_mode = _target_mode(args)
                daemon_status: Dict[str, Any] = {}
                if target_mode == "ssh":
                    daemon_up = _is_daemon_running(args, session_token=session_token)
                else:
                    try:
                        daemon_status = _control_channel(args, session_token=session_token).get_daemon_status()
                        daemon_up = bool(daemon_status.get("alive") or daemon_status.get("reachable"))
                    except Exception:
                        daemon_up = False
                status_c = _c("good", "Running") if daemon_up else _c("muted", "Stopped")
                auth_value = daemon_status.get("require_auth")
                
                # Print a more informative summary if daemon is running
                if daemon_up:
                    status_parts = []
                    pid = daemon_status.get("pid")
                    if pid:
                        status_parts.append(f"PID: {pid}")
                    if auth_value is not None:
                        status_parts.append(f"Auth: {'required' if bool(auth_value) else 'not required'}")
                    try:
                        res = _api_invoke(args, "host-metrics", {}, session_token=session_token)
                        if not pid and res.get("pid"):
                            status_parts.append(f"PID: {res.get('pid')}")
                        if auth_value is None and "require_auth" in res:
                            status_parts.append(f"Auth: {'required' if bool(res.get('require_auth')) else 'not required'}")
                        cpu = res.get("process_cpu_percent", 0.0)
                        mem = res.get("process_memory_mb", 0.0)
                        engines = res.get("engines_count", 0)
                        status_parts.extend([f"CPU: {cpu}%", f"Mem: {mem}MB", f"Engines: {engines}"])
                    except PermissionError as pe:
                        if "session_token_required" in str(pe):
                            if auth_value is None:
                                status_parts.append(_c("warn", "Auth required"))
                    except Exception:
                        pass
                    if status_parts:
                        status_c += f" ({', '.join(status_parts)})"
                elif auth_value is not None:
                    status_c += f" (Auth: {'required' if bool(auth_value) else 'not required'})"

                if first_run or status_c != last_status_c:
                    print()
                    _print_title("Engine Host Interactive Control")
                    _kv_rows([("Daemon Status", status_c)])
                    last_status_c = status_c
                    first_run = False
                
                lifecycle_label = "Restart remote daemon" if target_mode == "ssh" else ("Start daemon" if not daemon_up else "Stop daemon")
                opts = {
                    "l": ("List loaded engines and sandboxes", ""),
                    "d": ("Engine/Sandbox details", ""),
                    "m": ("Print daemon metrics", ""),
                    "c": ("List connected consumers", ""),
                    "k": ("Kill/Disconnect resource", ""),
                    "s": (lifecycle_label, ""),
                    "r": ("Local recovery/auth tools", "" if target_mode != "ssh" else "local only"),
                }
                choice = _prompt_menu("Main Menu", opts, "q", allow_changes=False)
                if choice == "q":                    return 0

                try:
                    if choice == "l":
                        session_token = _list_engines(args, session_token)
                    elif choice == "d":
                        session_token = _engine_details(args, session_token)
                    elif choice == "m":
                        session_token = _show_metrics(args, session_token)
                    elif choice == "c":
                        session_token = _list_consumers(args, session_token)
                    elif choice == "k":
                        session_token = _kill_resource(args, session_token)
                    elif choice == "s":
                        if target_mode == "ssh":
                            _start_daemon(args)
                        elif daemon_up:
                            _stop_daemon(args)
                        else:
                            _start_daemon(args)
                    elif choice == "r":
                        session_token = _local_recovery_menu(args, session_token)
                except PermissionError as pe:
                    if "session_token_required" in str(pe):
                        if _target_mode(args) != "ssh" and not _is_daemon_running(args, session_token=session_token):
                            print(_c('warn', "Daemon is stopped. Start the daemon before authenticating or running protected commands."))
                            time.sleep(1)
                            continue
                        token = _obtain_session_token(args)
                        if token:
                            session_token = token
                            print(_c('good', "Please try your command again now that you are authenticated."))
                            time.sleep(1)
                        else:
                            print(_c('bad', "Command failed: Authentication required."))
                            time.sleep(1)
                    else:
                        raise pe
            except (KeyboardInterrupt, EOFError):
                return 0
            except UserCancelled as exc:
                if getattr(exc, "via_keyboard", False):
                    return 0
                return 0
            except Exception as exc:
                print(f"\n{_c('bad', 'Error:')} {exc}")
                time.sleep(1)
    finally:
        if session_token:
            try:
                _api_invoke(args, "auth-revoke-session", {"token": session_token}, session_token=session_token)
            except Exception:
                pass


def _get_engines_dict(res: Any) -> Dict[str, dict]:
    if isinstance(res, dict) and "engines" in res:
        engines_data = res.get("engines")
        if isinstance(engines_data, dict):
            # Verify it's a dict of dicts
            return {k: v for k, v in engines_data.items() if isinstance(v, dict)}
        elif isinstance(engines_data, list):
             return {str(e.get("engine_id", f"unknown-{i}")): e for i, e in enumerate(engines_data) if isinstance(e, dict)}
    elif isinstance(res, list):
        return {str(e.get("engine_id", f"unknown-{i}")): e for i, e in enumerate(res) if isinstance(e, dict)}
    return {}


def _print_sessions(res: Dict[str, Any], session_token: Optional[str]) -> None:
    sessions = list(dict(res or {}).get("sessions") or [])
    cli_preview = _get_token_preview(session_token) if session_token else None

    filtered = []
    for s in sessions:
        tok = s.get("token_preview") or s.get("token_prefix") or ""
        if cli_preview and tok == cli_preview:
            continue
        filtered.append(s)

    if not filtered:
        if sessions:
            print("  No active sessions/consumers (excluding this CLI).")
        else:
            print("  No active sessions/consumers.")
        return

    for sess in filtered:
        tok = sess.get("token_preview") or sess.get("token_prefix") or "<unknown>"
        key_id = sess.get("key_id", "<unknown>")
        scope = sess.get("scope", "")
        print(f"  - Session [{_c('accent', tok)}] Key: {_c('value', key_id)} Scope: {scope}")

        ttl = sess.get("ttl_remaining_seconds")
        if ttl is not None:
            print(f"    Expires in: {ttl} seconds")
        elif "expires_at" in sess and sess["expires_at"] > 0:
            print(f"    Expires at: {sess['expires_at']}")

        role = sess.get("role")
        if role:
            print(f"    Role: {role}")

        allowed_configs = sess.get("allowed_configs")
        if allowed_configs:
            print(f"    Allowed Configs: {', '.join(allowed_configs)}")

        allowed_engines = sess.get("allowed_engines")
        if allowed_engines:
            print(f"    Allowed Engines: {', '.join(allowed_engines)}")

        ssh_binding = sess.get("ssh_binding", {})
        if ssh_binding:
            target = ssh_binding.get("target") or "<any>"
            fp = ssh_binding.get("key_fingerprint") or "<any>"
            print(f"    SSH Binding: Target={target}, Fingerprint={fp}")

        claims = sess.get("claims", {})
        if claims:
            for ck, cv in claims.items():
                print(f"    {ck}: {cv}")


def _list_engines(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Loaded Engines & Sandboxes")
    try:
        if _can_use_offline_local_fallback(args, session_token=session_token):
            res, session_token = _offline_local_read_with_auth(args, "discover-running", {}, session_token)
            if res is None:
                return session_token
            print(_c('warn', "  Note: Daemon is currently stopped. Showing offline fallback state."))
            print()
        else:
            res = _api_invoke(args, "discover-running", {}, session_token=session_token)
        engines = _get_engines_dict(res)
        if not engines:
            print("  No engines or sandboxes currently loaded.")
            return session_token
        for eid, info in engines.items():
            state = _operator_resource_state(info)
            kind = _operator_resource_kind(info)
            status_color = "good" if state == "running" else ("warn" if state in {"spawning", "unreachable"} else "muted")
            details = []
            if info.get("pid"):
                details.append(f"pid={info.get('pid')}")
            if "reachable" in info:
                details.append(f"reachable={'yes' if bool(info.get('reachable')) else 'no'}")
            suffix = f" {' '.join(details)}" if details else ""
            print(f"  - {_c('accent', eid)} [{_c(status_color, state)}] ({_c('value', kind)}){suffix}")
            loaded_models = [dict(item or {}) for item in list(info.get("loaded_models") or []) if isinstance(item, dict)]
            config_bindings = [dict(item or {}) for item in list(info.get("config_bindings") or []) if isinstance(item, dict)]
            if loaded_models:
                for model in loaded_models:
                    mid = str(model.get("model_instance_id") or model.get("engine_id") or "").strip()
                    mpath = str(model.get("model_path") or model.get("canonical_model_path") or "").strip()
                    print(f"    Model: {_c('accent', mid)}" + (f" {_c('muted', mpath)}" if mpath else ""))
                    for binding in config_bindings:
                        if str(binding.get("model_instance_id") or "").strip() != mid:
                            continue
                        bid = str(binding.get("engine_id") or "").strip()
                        cpath = str(binding.get("config_path") or binding.get("canonical_config_path") or "").strip()
                        print(f"      Binding: {_c('value', bid)}" + (f" {_c('muted', cpath)}" if cpath else ""))
            
            if _sandbox_enabled(info):
                sandbox = dict(info.get("sandbox") or {})
                profile = sandbox.get("profile") or dict(dict(info.get("sandbox_policy") or {}).get("sandbox") or {}).get("profile")
                bits = [f"profile={profile}"] if profile else []
                if sandbox.get("network_mode"):
                    bits.append(f"network={sandbox.get('network_mode')}")
                print(f"    Sandbox: {_c('good', 'enabled')}" + (f" {' '.join(bits)}" if bits else ""))
        return session_token
    except PermissionError:
        raise
    except Exception as e:
        print(_c('bad', f"Failed to list: {e}"))
        raise e


def _engine_details(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Resource Details")
    try:
        offline = _can_use_offline_local_fallback(args, session_token=session_token)
        if offline:
            res, session_token = _offline_local_read_with_auth(args, "discover-running", {}, session_token)
            if res is None:
                return session_token
            print(_c('warn', "  Note: Daemon is currently stopped. Showing offline fallback state."))
            print()
        else:
            res = _api_invoke(args, "discover-running", {}, session_token=session_token)
        engines = _get_engines_dict(res)
        if not engines:
            print("  No engines or sandboxes available.")
            return session_token
            
        opts = {eid: (f"Details for {eid}", "") for eid in engines.keys()}
        choice = _prompt_menu("Select Resource", opts, "b", allow_back=True, allow_changes=False)
        if choice in ("b", "back"):
            return session_token
            
        info = engines[choice]
        _kv_rows([
            ("ID", choice),
            ("State", _operator_resource_state(info)),
            ("Kind", _operator_resource_kind(info)),
            ("Pid", info.get("pid")),
        ])
        
        sandbox_policy = info.get("sandbox_policy", {})
        if sandbox_policy:
            print("\nSandbox Policy:")
            for k, v in sandbox_policy.items():
                print(f"  {k}: {v}")
                
        print("\nRaw State Info:")
        for k, v in info.items():
            if k not in ("state", "kind", "pid", "sandbox_policy"):
                print(f"  {k}: {v}")
        return session_token
    except PermissionError:
        raise
    except Exception as e:
        print(_c('bad', f"Error: {e}"))
        raise e


def _show_metrics(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Daemon Metrics")
    try:
        offline = _can_use_offline_local_fallback(args, session_token=session_token)
        if offline:
            res, session_token = _offline_local_read_with_auth(args, "host-metrics", {}, session_token)
            if res is None:
                return session_token
            print(_c('warn', "  Note: Daemon is currently stopped. Showing offline fallback state."))
            print(_c('muted', "  (PID shown belongs to the current CLI process, live network/proxy stats are N/A)"))
            print()
        else:
            res = _api_invoke(args, "host-metrics", {}, session_token=session_token)
            
        for metric, value in res.items():
            if isinstance(value, (dict, list)):
                formatted_val = "\n" + json.dumps(value, indent=2)
                # indent the json block
                formatted_val = formatted_val.replace("\n", "\n    ")
                _kv_rows([(metric, formatted_val)])
            else:
                _kv_rows([(metric, str(value))])
        return session_token
    except PermissionError:
        raise
    except Exception as e:
        print(_c('bad', f"Error fetching metrics (daemon may not be running?): {e}"))
        raise e


def _list_consumers(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Connected Consumers & Sessions")
    try:
        offline = _can_use_offline_local_fallback(args, session_token=session_token)
        if offline:
            res, session_token = _offline_local_read_with_auth(args, "auth-list-sessions", {}, session_token)
            if res is None:
                return session_token
            print(_c('warn', "  Note: Daemon is currently stopped. Showing offline fallback state."))
            print(_c('muted', "  (Session status may be stale.)"))
            print()
        else:
            res = _api_invoke(args, "auth-list-sessions", {}, session_token=session_token)
        _print_sessions(res, session_token)
        return session_token
    except PermissionError:
        raise
    except Exception as e:
        print(_c('bad', f"Error listing consumers: {e}"))
        raise e


def _kill_resource(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Kill Resource")
    try:
        if _can_use_offline_local_fallback(args, session_token=session_token):
            print(_c('warn', "  Daemon is stopped. Kill/disconnect actions require a running daemon."))
            return session_token
        opts = {
            "u": ("Unload Model Binding", ""),
            "e": ("Stop Worker/Sandbox", ""),
            "c": ("Disconnect Consumer (Revoke Session)", ""),
        }
        ch = _prompt_menu("What to kill?", opts, "b", allow_back=True, allow_changes=False)
        if ch in ("b", "back"): return session_token
        
        if ch == "u":
            res = _api_invoke(args, "discover-running", {}, session_token=session_token)
            engines = _get_engines_dict(res)
            model_opts = {}
            for wid, info in engines.items():
                bindings = [dict(item or {}) for item in list(info.get("config_bindings") or []) if isinstance(item, dict)]
                if bindings:
                    for binding in bindings:
                        eid = str(binding.get("engine_id") or "").strip()
                        if eid:
                            model_opts[eid] = (f"Unload {eid} from worker {wid}", "")
                elif _operator_resource_kind(info).endswith("model instance"):
                    model_opts[wid] = (f"Unload {wid}", "")
            if not model_opts:
                print("  No model bindings to unload.")
                return session_token
            ech = _prompt_menu("Select Model Binding", model_opts, "b", allow_back=True, allow_changes=False)
            if ech in ("b", "back"): return session_token
            print(f"Unloading {ech}...")
            _api_invoke(args, "unload-model", {"engine_id": ech}, session_token=session_token)
            print(_c('good', "Unload requested."))

        elif ch == "e":
            res = _api_invoke(args, "discover-running", {}, session_token=session_token)
            engines = _get_engines_dict(res)
            if not engines:
                print("  No workers to stop.")
                return session_token
            eopts = {
                str(info.get("worker_id") or eid): (f"Stop {info.get('worker_id') or eid}", "")
                for eid, info in engines.items()
            }
            ech = _prompt_menu("Select Worker", eopts, "b", allow_back=True, allow_changes=False)
            if ech in ("b", "back"): return session_token
            print(f"Stopping {ech}...")
            _api_invoke(args, "shutdown", {"engine_id": ech}, session_token=session_token)
            print(_c('good', "Shutdown signal sent."))
            
        elif ch == "c":
            res = _api_invoke(args, "auth-list-sessions", {}, session_token=session_token)
            sessions = res.get("sessions", [])
            cli_preview = _get_token_preview(session_token) if session_token else None
                
            sopts = {}
            for s in sessions:
                tok = s.get("token_preview") or s.get("token_prefix")
                if cli_preview and tok == cli_preview:
                    continue
                if tok:
                    sopts[tok] = (f"Revoke session [{tok}] (Key: {s.get('key_id', '<unknown>')})", "")
                    
            if not sopts:
                if sessions:
                    print("  No active sessions to disconnect (excluding this CLI).")
                else:
                    print("  No active sessions to disconnect.")
                return session_token
                
            sch = _prompt_menu("Select Session Preview", sopts, "b", allow_back=True, allow_changes=False)
            if sch in ("b", "back"): return session_token
            
            print(f"Revoking session...")
            # Pass the token_preview. The API auth_revoke_session needs to support matching by preview.
            _api_invoke(args, "auth-revoke-session", {"token": sch}, session_token=session_token)
            print(_c('good', "Session revoked."))
        return session_token
    except PermissionError:
        raise
            
    except Exception as e:
        print(_c('bad', f"Error: {e}"))
        raise e


def _start_daemon(args: argparse.Namespace) -> None:
    channel = _control_channel(args)
    if str(channel.get_target().get("mode") or "local") == "ssh":
        print("Restarting remote daemon...")
        result = channel.restart_remote_daemon()
        if result.get("started"):
            print(_c('good', "Remote daemon restart requested."))
        else:
            print(_c('bad', f"Remote daemon restart failed: {result.get('error') or 'unknown error'}"))
        return
    print("Starting daemon in background...")
    result = channel.bootstrap_daemon(wait_ready_seconds=8.0)
    if result.get("blocked_by_unreachable_pid"):
        print(_c('bad', f"Daemon start blocked: {result.get('error') or result.get('reachability_error') or 'existing daemon is unreachable'}"))
        policy = dict(result.get("auto_recovery_policy") or {})
        if policy:
            print(_c('muted', f"Recovery policy: endpoint={policy.get('endpoint_mode_default') or 'unknown'}, lifecycle={policy.get('lifecycle_profile') or 'unknown'}"))
        print(_c('muted', "Use Local recovery/auth tools -> Force restart daemon and workers if this shared/detached daemon is stale."))
        return
    if result.get("auto_recovery_attempted"):
        print(_c('warn', "Recovered an unreachable exclusive/foreground local daemon before starting a fresh one."))
    if result.get("alive") or result.get("reachable") or result.get("already_running"):
        print(_c('good', "Daemon started."))
    else:
        print(_c('bad', f"Daemon start did not become ready: {result.get('reachability_error') or result.get('error') or 'unknown error'}"))


def _stop_daemon(args: argparse.Namespace) -> None:
    channel = _control_channel(args)
    if str(channel.get_target().get("mode") or "local") == "ssh":
        print(_c('warn', "Stopping remote daemons is not supported by the interactive control channel."))
        return
    print("Stopping daemon...")
    if not _is_daemon_running(args):
        print("Daemon is not running.")
        return
    result = channel.stop_daemon()
    status = str(result.get("status") or "")
    if status in {"shutdown_sent", "not_running"}:
        print(_c('good', "Daemon stop signal sent."))
    else:
        print(_c('bad', f"Failed to stop: {result.get('error') or status or 'unknown error'}"))
