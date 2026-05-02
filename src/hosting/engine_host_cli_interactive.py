from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

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


def _control_channel(args: argparse.Namespace, session_token: Optional[str] = None) -> EngineHostControlChannel:
    channel = getattr(args, "_interactive_control_channel", None)
    if not isinstance(channel, EngineHostControlChannel):
        channel = EngineHostControlChannel(_control_channel_settings(args))
        setattr(args, "_interactive_control_channel", channel)
    channel.set_session_token(session_token)
    return channel


def _raise_interactive_api_error(exc: Exception) -> None:
    msg = str(exc)
    if "session_token_required" in msg or "missing_or_invalid_session_token" in msg:
        raise PermissionError("session_token_required") from exc
    raise exc


def _api_invoke(args: argparse.Namespace, cmd: str, payload: dict, session_token: Optional[str] = None) -> Any:
    channel = _control_channel(args, session_token=session_token)
    try:
        return channel.invoke_control_command(str(cmd or "").strip(), dict(payload or {}))
    except Exception as exc:
        _raise_interactive_api_error(exc)


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
    return bool(status.get("alive") or status.get("reachable") or status.get("pid_alive"))


def _target_mode(args: argparse.Namespace) -> str:
    return str(_control_channel(args).get_target().get("mode") or "local")


def _obtain_session_token(args: argparse.Namespace) -> Optional[str]:
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
        if input_text.startswith("{") and input_text.endswith("}"):
            try:
                payload = json.loads(input_text)
                if "metadata" in payload and "key_id" in payload["metadata"]:
                    key_id = payload["metadata"]["key_id"]
                    key_id_from_json = True
            except Exception:
                pass
        
        if not key_id_from_json:
            try:
                print(_c('muted', f"Could not determine Key ID from input. Defaulting to '{key_id}'."))
                key_id_input = input(f"Key ID [{key_id}]: ").strip()
                if key_id_input:
                    key_id = key_id_input
            except (KeyboardInterrupt, EOFError):
                return None

        chal_res = _api_invoke(args, "auth-begin-challenge", {"key_id": key_id, "scope": "control"})
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
        
        comp_res = _api_invoke(args, "auth-complete-challenge", {
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
                daemon_up = _is_daemon_running(args, session_token=session_token)
                status_c = _c("good", "Running") if daemon_up else _c("muted", "Stopped")
                
                # Print a more informative summary if daemon is running
                if daemon_up:
                    try:
                        res = _api_invoke(args, "host-metrics", {}, session_token=session_token)
                        cpu = res.get("process_cpu_percent", 0.0)
                        mem = res.get("process_memory_mb", 0.0)
                        engines = res.get("engines_count", 0)
                        status_c += f" (CPU: {cpu}%, Mem: {mem}MB, Engines: {engines})"
                    except PermissionError as pe:
                        if "session_token_required" in str(pe):
                            status_c += " " + _c("warn", "(Auth required)")
                    except Exception:
                        pass

                if first_run or status_c != last_status_c:
                    print()
                    _print_title("Engine Host Interactive Control")
                    _kv_rows([("Daemon Status", status_c)])
                    last_status_c = status_c
                    first_run = False
                
                target_mode = _target_mode(args)
                lifecycle_label = "Restart remote daemon" if target_mode == "ssh" else ("Start daemon" if not daemon_up else "Stop daemon")
                opts = {
                    "l": ("List loaded engines and sandboxes", ""),
                    "d": ("Engine/Sandbox details", ""),
                    "m": ("Print daemon metrics", ""),
                    "c": ("List connected consumers", ""),
                    "k": ("Kill/Disconnect resource", ""),
                    "s": (lifecycle_label, ""),
                }
                choice = _prompt_menu("Main Menu", opts, "q", allow_changes=False)
                if choice == "q":                    return 0

                try:
                    if choice == "l":
                        _list_engines(args, session_token)
                    elif choice == "d":
                        _engine_details(args, session_token)
                    elif choice == "m":
                        _show_metrics(args, session_token)
                    elif choice == "c":
                        _list_consumers(args, session_token)
                    elif choice == "k":
                        _kill_resource(args, session_token)
                    elif choice == "s":
                        if target_mode == "ssh":
                            _start_daemon(args)
                        elif daemon_up:
                            _stop_daemon(args)
                        else:
                            _start_daemon(args)
                except PermissionError as pe:
                    if "session_token_required" in str(pe):
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

def _list_engines(args: argparse.Namespace, session_token: Optional[str]) -> None:
    _print_block("Loaded Engines & Sandboxes")
    try:
        res = _api_invoke(args, "discover-running", {}, session_token=session_token)
        engines = _get_engines_dict(res)
        if not engines:
            print("  No engines or sandboxes currently loaded.")
            return
        for eid, info in engines.items():
            state = info.get("state", "unknown")
            kind = info.get("kind", "unknown")
            status_color = "good" if state == "running" else ("warn" if state == "spawning" else "muted")
            print(f"  - {_c('accent', eid)} [{_c(status_color, state)}] ({_c('value', kind)})")
            
            sandbox_policy = info.get("sandbox_policy", {})
            if sandbox_policy and sandbox_policy.get("enabled"):
                print(f"    Sandbox: {_c('good', 'enabled')} root={sandbox_policy.get('root_id')}")
    except Exception as e:
        print(_c('bad', f"Failed to list: {e}"))
        raise e


def _engine_details(args: argparse.Namespace, session_token: Optional[str]) -> None:
    _print_block("Resource Details")
    try:
        res = _api_invoke(args, "discover-running", {}, session_token=session_token)
        engines = _get_engines_dict(res)
        if not engines:
            print("  No engines or sandboxes available.")
            return
            
        opts = {eid: (f"Details for {eid}", "") for eid in engines.keys()}
        choice = _prompt_menu("Select Resource", opts, "b", allow_back=True, allow_changes=False)
        if choice in ("b", "back"):
            return
            
        info = engines[choice]
        _kv_rows([
            ("ID", choice),
            ("State", info.get("state")),
            ("Kind", info.get("kind")),
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
    except Exception as e:
        print(_c('bad', f"Error: {e}"))
        raise e


def _show_metrics(args: argparse.Namespace, session_token: Optional[str]) -> None:
    _print_block("Daemon Metrics")
    try:
        res = _api_invoke(args, "host-metrics", {}, session_token=session_token)
        if not _is_daemon_running(args, session_token=session_token):
            print(_c('warn', "  Note: Daemon is currently stopped. Showing offline fallback state."))
            print(_c('muted', "  (PID shown belongs to the current CLI process, live network/proxy stats are N/A)"))
            print()
            
        for metric, value in res.items():
            if isinstance(value, (dict, list)):
                formatted_val = "\n" + json.dumps(value, indent=2)
                # indent the json block
                formatted_val = formatted_val.replace("\n", "\n    ")
                _kv_rows([(metric, formatted_val)])
            else:
                _kv_rows([(metric, str(value))])
    except Exception as e:
        print(_c('bad', f"Error fetching metrics (daemon may not be running?): {e}"))
        raise e


def _list_consumers(args: argparse.Namespace, session_token: Optional[str]) -> None:
    _print_block("Connected Consumers & Sessions")
    try:
        res = _api_invoke(args, "auth-list-sessions", {}, session_token=session_token)
        if not _is_daemon_running(args, session_token=session_token):
            print(_c('warn', "  Note: Daemon is currently stopped. Showing offline fallback state."))
            print(_c('muted', "  (Session status may be stale.)"))
            print()
            
        sessions = res.get("sessions", [])
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
            
            # Show TTL or connection time
            ttl = sess.get("ttl_remaining_seconds")
            if ttl is not None:
                print(f"    Expires in: {ttl} seconds")
            elif "expires_at" in sess and sess["expires_at"] > 0:
                print(f"    Expires at: {sess['expires_at']}")
            
            # Show role and allowed scope details
            role = sess.get("role")
            if role:
                print(f"    Role: {role}")
            
            allowed_configs = sess.get("allowed_configs")
            if allowed_configs:
                print(f"    Allowed Configs: {', '.join(allowed_configs)}")
                
            allowed_engines = sess.get("allowed_engines")
            if allowed_engines:
                print(f"    Allowed Engines: {', '.join(allowed_engines)}")
            
            # Show SSH binding
            ssh_binding = sess.get("ssh_binding", {})
            if ssh_binding:
                target = ssh_binding.get("target") or "<any>"
                fp = ssh_binding.get("key_fingerprint") or "<any>"
                print(f"    SSH Binding: Target={target}, Fingerprint={fp}")
                
            # Legacy claims fallback
            claims = sess.get("claims", {})
            if claims:
                for ck, cv in claims.items():
                    print(f"    {ck}: {cv}")
    except Exception as e:
        print(_c('bad', f"Error listing consumers: {e}"))
        raise e


def _kill_resource(args: argparse.Namespace, session_token: Optional[str]) -> None:
    _print_block("Kill Resource")
    try:
        opts = {
            "e": ("Kill Engine/Sandbox", ""),
            "c": ("Disconnect Consumer (Revoke Session)", ""),
        }
        ch = _prompt_menu("What to kill?", opts, "b", allow_back=True, allow_changes=False)
        if ch in ("b", "back"): return
        
        if ch == "e":
            res = _api_invoke(args, "discover-running", {}, session_token=session_token)
            engines = _get_engines_dict(res)
            if not engines:
                print("  No engines to kill.")
                return
            eopts = {eid: (f"Kill {eid}", "") for eid in engines.keys()}
            ech = _prompt_menu("Select Engine", eopts, "b", allow_back=True, allow_changes=False)
            if ech in ("b", "back"): return
            print(f"Killing {ech}...")
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
                return
                
            sch = _prompt_menu("Select Session Preview", sopts, "b", allow_back=True, allow_changes=False)
            if sch in ("b", "back"): return
            
            print(f"Revoking session...")
            # Pass the token_preview. The API auth_revoke_session needs to support matching by preview.
            _api_invoke(args, "auth-revoke-session", {"token": sch}, session_token=session_token)
            print(_c('good', "Session revoked."))
            
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
