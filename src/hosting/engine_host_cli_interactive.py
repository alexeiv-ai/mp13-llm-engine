from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List

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
_SESSION_RENEW_CHECK_INTERVAL_SECONDS = 30
_SESSION_RENEW_MIN_TTL_SECONDS = 180


def _arg_value(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _control_channel_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "engine_host_daemon_auto_bootstrap": False,
        "engine_host_daemon_pid_file": str(_arg_value(args, "pid_file") or "") or None,
        "engine_host_state_file": str(_arg_value(args, "engines_state_file") or "") or None,
        "engine_host_control_state_file": str(_arg_value(args, "control_state_file") or "") or None,
        # Interactive menus require explicit Authenticate selection; do not let
        # profile-provided shared secrets auto-mint sessions for protected actions.
        "engine_host_key_secret": "",
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
        "engine_host_key_id",
        "engine_host_session_token",
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
    code = str(getattr(exc, "code", "") or "").strip()
    details = dict(getattr(exc, "details", {}) or {})
    auth_code = code or str(details.get("reason") or "").strip()
    if auth_code in {
        "session_token_required",
        "missing_or_invalid_session_token",
        "invalid_session",
        "session_expired",
        "session_not_found",
        "invalid_token",
        "expired_token",
    }:
        raise PermissionError(auth_code) from exc
    if auth_code:
        raise RuntimeError(auth_code) from exc
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
    channel = (
        _control_channel(args)
        if session_token is None
        else _control_channel(args, session_token=session_token)
    )
    try:
        return channel.invoke_control_command(str(cmd or "").strip(), dict(payload or {}))
    except Exception as exc:
        _raise_interactive_api_error(exc)


def _auth_api_invoke(args: argparse.Namespace, cmd: str, payload: dict) -> Any:
    channel = EngineHostControlChannel(_control_channel_settings(args))
    channel.set_session_token(None)
    try:
        return channel.invoke_control_command(str(cmd or "").strip(), dict(payload or {}))
    except Exception as exc:
        _raise_interactive_api_error(exc)


def _active_session_token(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    if bool(getattr(args, "_interactive_session_token_invalid", False)):
        setattr(args, "_interactive_session_token_invalid", False)
        _control_channel(args, session_token=None)
        return None
    worker_token = str(getattr(args, "_interactive_session_token", "") or "").strip()
    if worker_token:
        session_token = worker_token
    channel = _control_channel(args)
    get_token = getattr(channel, "get_session_token", None)
    if not callable(get_token):
        return session_token
    current = get_token()
    return current if current else session_token


def _set_interactive_session_token(args: argparse.Namespace, token: Optional[str]) -> Optional[str]:
    tok = str(token or "").strip() or None
    setattr(args, "_interactive_session_token", tok or "")
    setattr(args, "_interactive_session_token_invalid", False)
    _control_channel(args, session_token=tok)
    return tok


def _background_session_renew_loop(args: argparse.Namespace, stop_event: threading.Event) -> None:
    channel = EngineHostControlChannel(_control_channel_settings(args))
    while not stop_event.wait(_SESSION_RENEW_CHECK_INTERVAL_SECONDS):
        token = str(getattr(args, "_interactive_session_token", "") or "").strip()
        if not token:
            continue
        channel.set_session_token(token)
        try:
            status = channel.get_daemon_status()
            daemon_up = bool(dict(status or {}).get("alive") or dict(status or {}).get("reachable"))
        except Exception:
            daemon_up = False
        if not daemon_up:
            setattr(args, "_interactive_daemon_stopped_notice", True)
            return
        try:
            validation = channel.invoke_control_command(
                "auth-validate-session",
                {"token": token, "scope": "control"},
            )
        except Exception:
            continue
        data = dict(validation or {})
        if not bool(data.get("valid", False)):
            reason = str(data.get("reason") or "").strip()
            if reason in {
                "missing_or_invalid_session_token",
                "session_revoked",
                "session_expired",
                "invalid_session",
                "expired_token",
            }:
                setattr(args, "_interactive_session_token", "")
                setattr(args, "_interactive_session_token_invalid", True)
                channel.set_session_token(None)
            continue
        try:
            ttl_remaining = int(data.get("ttl_remaining_seconds"))
        except Exception:
            continue
        if ttl_remaining > _SESSION_RENEW_MIN_TTL_SECONDS:
            continue
        try:
            try:
                configured_ttl = int(_control_channel_settings(args).get("engine_host_session_ttl_seconds") or 900)
            except Exception:
                configured_ttl = 900
            renewed = channel.invoke_control_command(
                "auth-renew-session",
                {
                    "token": token,
                    "scope": "control",
                    "ttl_seconds": configured_ttl,
                },
            )
            if renewed:
                setattr(args, "_interactive_daemon_stopped_notice", False)
        except Exception:
            continue


def _ensure_session_renewer(args: argparse.Namespace) -> None:
    thread = getattr(args, "_interactive_session_renew_thread", None)
    if isinstance(thread, threading.Thread) and thread.is_alive():
        return
    stop_event = threading.Event()
    setattr(args, "_interactive_session_renew_stop", stop_event)
    thread = threading.Thread(
        target=_background_session_renew_loop,
        args=(args, stop_event),
        name="engine-host-interactive-session-renew",
        daemon=True,
    )
    setattr(args, "_interactive_session_renew_thread", thread)
    thread.start()


def _stop_session_renewer(args: argparse.Namespace) -> None:
    stop_event = getattr(args, "_interactive_session_renew_stop", None)
    if isinstance(stop_event, threading.Event):
        stop_event.set()
    thread = getattr(args, "_interactive_session_renew_thread", None)
    if isinstance(thread, threading.Thread) and thread.is_alive():
        thread.join(timeout=1.0)


def _renew_session_token_if_needed(
    args: argparse.Namespace,
    session_token: Optional[str],
    *,
    daemon_up: bool,
) -> Optional[str]:
    if not session_token or not daemon_up:
        return session_token
    now = time.time()
    last_check = float(getattr(args, "_interactive_session_renew_checked_at", 0.0) or 0.0)
    if now - last_check < _SESSION_RENEW_CHECK_INTERVAL_SECONDS:
        return session_token
    setattr(args, "_interactive_session_renew_checked_at", now)

    try:
        validation = _api_invoke(
            args,
            "auth-validate-session",
            {"token": session_token, "scope": "control"},
            session_token=session_token,
        )
    except Exception:
        return session_token
    data = dict(validation or {})
    if not bool(data.get("valid", False)):
        reason = str(data.get("reason") or "").strip()
        if reason in {
            "missing_or_invalid_session_token",
            "session_revoked",
            "session_expired",
            "invalid_session",
            "expired_token",
        }:
            _control_channel(args, session_token=None)
            return None
        return session_token

    ttl_remaining = data.get("ttl_remaining_seconds")
    try:
        ttl_remaining_int = int(ttl_remaining) if ttl_remaining is not None else None
    except Exception:
        ttl_remaining_int = None
    if ttl_remaining_int is None or ttl_remaining_int > _SESSION_RENEW_MIN_TTL_SECONDS:
        return session_token

    try:
        _default_key_id, _configured_secret, configured_ttl = _configured_auth_defaults(args)
        renewed = _api_invoke(
            args,
            "auth-renew-session",
            {
                "token": session_token,
                "scope": "control",
                "ttl_seconds": configured_ttl,
            },
            session_token=session_token,
        )
        expires_at = float(dict(renewed or {}).get("expires_at") or 0.0)
        set_meta = getattr(_control_channel(args, session_token=session_token), "_set_session_token_meta", None)
        if callable(set_meta):
            set_meta(
                {
                    "auth_method": str(dict(renewed or {}).get("auth_method") or data.get("auth_method") or ""),
                    "key_id": str(dict(renewed or {}).get("key_id") or data.get("key_id") or ""),
                    "scope": str(dict(renewed or {}).get("scope") or data.get("scope") or "control"),
                    "expires_at": expires_at,
                    "ssh_binding": dict(dict(renewed or {}).get("ssh_binding") or data.get("ssh_binding") or {}),
                }
            )
        return session_token
    except PermissionError:
        return session_token
    except Exception:
        return session_token


def _session_identity_from_list(res: Dict[str, Any], session_token: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not session_token:
        return None, None
    cli_preview = _get_token_preview(session_token)
    for sess in list(dict(res or {}).get("sessions") or []):
        tok = str(sess.get("token_preview") or sess.get("token_prefix") or "").strip()
        if tok != cli_preview:
            continue
        key_id = str(sess.get("key_id") or "").strip() or None
        role = str(sess.get("role") or "").strip() or None
        return key_id, role
    return None, None


def _lookup_current_session_identity(args: argparse.Namespace, session_token: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not session_token:
        return None, None
    try:
        res = _api_invoke(args, "auth-list-sessions", {}, session_token=session_token)
    except Exception:
        return None, None
    return _session_identity_from_list(res if isinstance(res, dict) else {}, session_token)


def _reachability_summary(info: Dict[str, Any]) -> Optional[str]:
    if bool(info.get("reachable", False)):
        return None
    reachability = dict(info.get("reachability") or {})
    error = str(reachability.get("error") or "").strip()
    if error:
        if "worker IPC endpoint is unavailable" in error:
            if str(info.get("state") or "").strip().lower() == "spawning":
                return "Worker is still starting; its IPC endpoint is not available yet."
            return "IPC endpoint unavailable; worker exited, failed to start its IPC server, or this is a stale PID registration."
        return error
    if bool(info.get("alive", False)):
        return "PID exists, but the worker did not answer the hosting IPC health probe."
    return None


def _worker_status_summary(
    args: argparse.Namespace,
    session_token: Optional[str],
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        res = dict(metrics or _api_invoke(args, "host-metrics", {}, session_token=session_token) or {})
    except Exception:
        return {}
    summary = dict(res.get("resource_summary") or {})
    if not summary:
        return {}
    return {
        "workers_count": int(summary.get("workers_count") or 0),
        "worker_cpu_percent": summary.get("worker_cpu_percent"),
        "worker_memory_mb": summary.get("worker_memory_mb"),
        "worker_gpu_vram_mb": summary.get("worker_gpu_vram_mb"),
        "worker_gpu_vram_pending": bool(summary.get("worker_gpu_vram_pending")),
    }


def _format_percent_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "N/A"


def _format_mb_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f}MB"
    except Exception:
        return "N/A"


def _format_gb_from_mb_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) / 1024.0:.1f}GB"
    except Exception:
        return "N/A"


def _format_gb_from_mb_or_pending(value: Any, *, pending: bool = False) -> str:
    if value is None and pending:
        return "pending"
    return _format_gb_from_mb_or_na(value)


def _python_runtime_rows(metrics: Optional[Dict[str, Any]]) -> list[tuple[str, str]]:
    data = dict(metrics or {})
    daemon_python = str(data.get("daemon_python_executable") or "").strip()
    engine_python = str(data.get("engine_python_executable") or "").strip()
    env_python = str(data.get("mp13_engine_python_env") or "").strip()
    rows: list[tuple[str, str]] = []
    if daemon_python:
        rows.append(("Daemon Python", _c("muted", daemon_python)))
    if engine_python:
        source = f"MP13_ENGINE_PYTHON={env_python}" if env_python else "MP13_ENGINE_PYTHON unset; using daemon Python"
        rows.append(("Engine Python", _c("muted", f"{engine_python} ({source})")))
    elif env_python:
        rows.append(("Engine Python", _c("muted", f"unresolved (MP13_ENGINE_PYTHON={env_python})")))
    return rows


def _resource_bits(resources: Dict[str, Any]) -> list[str]:
    res = dict(resources or {})
    bits: list[str] = []
    if "cpu_percent" in res:
        bits.append(f"cpu={_format_percent_or_na(res.get('cpu_percent'))}")
    if "memory_mb" in res:
        bits.append(f"rss={_format_mb_or_na(res.get('memory_mb'))}")
    if "gpu_vram_mb" in res:
        bits.append(f"vram={_format_gb_from_mb_or_pending(res.get('gpu_vram_mb'), pending=bool(res.get('gpu_vram_pending')))}")
    if "workflow_js_node_process_count" in res:
        bits.append(
            "js_nodes="
            f"{int(res.get('workflow_js_active_node_process_count') or 0)}/"
            f"{int(res.get('workflow_js_node_process_count') or 0)}/"
            f"{int(res.get('workflow_js_capacity') or 0)}"
        )
    if res.get("workflow_js_node_cpu_percent") is not None:
        bits.append(f"js_cpu={_format_percent_or_na(res.get('workflow_js_node_cpu_percent'))}")
    if res.get("workflow_js_node_memory_mb") is not None:
        bits.append(f"js_rss={_format_mb_or_na(res.get('workflow_js_node_memory_mb'))}")
    return bits


def _read_json_file(path: str) -> Dict[str, Any]:
    raw = str(path or "").strip()
    if not raw:
        return {}
    try:
        p = Path(raw).expanduser()
        if not p.exists() or not p.is_file():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return dict(data or {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _configured_model_path_from_config_row(row: Dict[str, Any]) -> Optional[str]:
    cfg = _read_json_file(str(row.get("path") or ""))
    if not cfg:
        return None
    engine_params = cfg.get("engine_params") if isinstance(cfg.get("engine_params"), dict) else {}
    value = (
        engine_params.get("base_model_path")
        or cfg.get("base_model_path")
        or cfg.get("model")
        or cfg.get("base_model_name_or_path")
    )
    text = str(value or "").strip()
    return text or None


def _config_uses_generic_worker(row: Dict[str, Any]) -> bool:
    cfg = _read_json_file(str(row.get("path") or ""))
    if not cfg:
        return False
    hosting_cfg = cfg.get("hosting") if isinstance(cfg.get("hosting"), dict) else {}
    marker = str(
        cfg.get("worker_kind")
        or cfg.get("worker_type")
        or hosting_cfg.get("worker_kind")
        or hosting_cfg.get("worker_type")
        or ""
    ).strip().lower()
    if marker in {"generic", "non_model", "worker", "generic_worker"}:
        return True
    spawn_cfg = cfg.get("spawn") if isinstance(cfg.get("spawn"), dict) else {}
    return bool(
        (isinstance(cfg.get("worker_command"), list) and cfg.get("worker_command"))
        or (isinstance(spawn_cfg.get("command"), list) and spawn_cfg.get("command"))
    )


def _config_selector(row: Dict[str, Any]) -> str:
    name = str(row.get("name") or "").strip()
    if name and name != "default":
        return name
    return "default"


def _print_progress_snapshot(snapshot: Dict[str, Any], *, last_text: str = "") -> str:
    percent = snapshot.get("progress_percent")
    text = str(snapshot.get("progress_text") or "").strip()
    if not text:
        events = [dict(item or {}) for item in list(snapshot.get("progress_events") or []) if isinstance(item, dict)]
        if events:
            text = str(events[-1].get("message") or events[-1].get("stage") or "").strip()
    if isinstance(percent, int):
        pct = max(0, min(100, int(percent)))
        filled = int(round(pct / 5.0))
        bar = "#" * filled + "." * (20 - filled)
        line = f"  Progress: [{bar}] {pct:3d}%"
        if text:
            line += f"  {text}"
    elif text:
        line = f"  Progress: [{'.' * 20}]   0%  {text}"
    else:
        line = f"  Progress: [{'.' * 20}]   0%  {snapshot.get('status') or 'running'}"
    if line != last_text:
        print(_c("muted", line))
        progress_error = str(snapshot.get("progress_error") or "").strip()
        if progress_error:
            print(_c("bad", f"  Worker log error: {progress_error}"))
        return line
    return last_text


def _operation_failure_message(snapshot: Dict[str, Any]) -> str:
    snap = dict(snapshot or {})
    result = dict(snap.get("result") or {}) if isinstance(snap.get("result"), dict) else {}
    candidates = [
        result.get("message"),
        result.get("reason"),
        snap.get("error"),
        snap.get("error_code"),
        result.get("status"),
        snap.get("status"),
    ]
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return "unknown error"


def _print_operation_diagnostics(snapshot: Dict[str, Any]) -> None:
    result = dict(snapshot.get("result") or {}) if isinstance(snapshot.get("result"), dict) else {}
    reason = str(result.get("reason") or "").strip()
    message = str(result.get("message") or "").strip()
    if reason:
        print(_c("muted", f"  Reason: {reason}"))
    if message and message != reason:
        print(_c("muted", f"  Detail: {message}"))
    events = [
        dict(item or {})
        for item in list(result.get("progress_events") or snapshot.get("progress_events") or [])
        if isinstance(item, dict)
    ]
    failed_events = [event for event in events if str(event.get("status") or "").lower() == "failed"]
    if failed_events:
        last = failed_events[-1]
        stage = str(last.get("stage") or "").strip()
        msg = str(last.get("message") or "").strip()
        print(_c("muted", f"  Failed stage: {stage or 'unknown'}" + (f" - {msg}" if msg else "")))
    managed = result.get("managed_engine") if isinstance(result.get("managed_engine"), dict) else {}
    log_path = str(dict(managed or {}).get("log_path") or dict(snapshot.get("diagnostics") or {}).get("log_path") or "").strip()
    if log_path:
        print(_c("muted", f"  Worker log: {log_path}"))


def _inference_text_and_metrics(rpc_result: Dict[str, Any]) -> tuple[str, Dict[str, Any], list[Dict[str, Any]]]:
    payload = dict(rpc_result.get("result") or {}) if isinstance(rpc_result.get("result"), dict) else {}
    chunks = [dict(item or {}) for item in list(payload.get("stream") or []) if isinstance(item, dict)]
    final_text = ""
    streamed_parts: list[str] = []
    metrics: Dict[str, Any] = {}
    metric_keys = {
        "input_tokens",
        "output_tokens",
        "generation_duration_sec",
        "tokens_per_second",
        "time_to_first_token_sec",
        "total_input_tokens",
        "total_output_tokens",
        "total_generation_duration_sec",
        "overall_tps",
        "avg_time_to_first_token_sec",
        "mem_allocated",
        "mem_reserved",
        "cache_metric",
        "cache_warming",
        "cache_queued",
        "was_truncated",
        "was_canceled",
        "had_error",
    }
    for chunk in chunks:
        chunk_text = str(chunk.get("chunk_text") or "")
        if chunk_text:
            streamed_parts.append(chunk_text)
        response_text = str(chunk.get("response_text") or "")
        if response_text:
            final_text = response_text
        for key in metric_keys:
            if key in chunk and chunk.get(key) is not None:
                metrics[key] = chunk.get(key)
        if str(chunk.get("error") or "").strip():
            metrics["error"] = str(chunk.get("error") or "").strip()
    if not final_text:
        final_text = "".join(streamed_parts)
    return final_text, metrics, chunks


def _print_inference_metrics(metrics: Dict[str, Any], *, observed_latency_sec: float) -> None:
    reported = dict(metrics or {})
    reported_duration = reported.get("total_generation_duration_sec")
    if reported_duration is None:
        reported_duration = reported.get("generation_duration_sec")
    rows: list[tuple[str, Any]] = [
        ("observed_e2e_sec", f"{observed_latency_sec:.3f}"),
    ]
    try:
        if reported_duration is not None:
            reported_float = float(reported_duration)
            rows.append(("reported_generation_sec", f"{reported_float:.3f}"))
            rows.append(("observed_minus_reported_sec", f"{observed_latency_sec - reported_float:.3f}"))
    except Exception:
        rows.append(("reported_generation_sec", reported_duration))
    for key in [
        "avg_time_to_first_token_sec",
        "time_to_first_token_sec",
        "total_input_tokens",
        "total_output_tokens",
        "input_tokens",
        "output_tokens",
        "overall_tps",
        "tokens_per_second",
        "mem_allocated",
        "mem_reserved",
        "cache_metric",
        "cache_warming",
        "cache_queued",
        "was_truncated",
        "had_error",
        "error",
    ]:
        if key in reported:
            rows.append((key, reported.get(key)))
    _kv_rows(rows, min_width=30)


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
        return svc.get_host_metrics(session_token=session_token)
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
    print(_c('muted', "  Start the daemon and choose Authenticate from the main menu, or use Local recovery/auth tools for explicit local state access."))


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
        if token:
            return _offline_local_invoke(args, cmd, payload, session_token=token), token
        return None, session_token


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
    is_workflow_python_helper = (
        executor_kind == "workflow_python_helper"
        or "hosting.workflow_python_helper_ipc" in command_text
        or "MP13_WORKFLOW_PYTHON_HELPER_CAPACITY" in env
    )
    if is_workflow_python_helper:
        return "workflow python sandbox" if sandbox_enabled else "workflow python worker"
    is_workflow_js_node = executor_kind == "workflow_js_node" or "hosting.workflow_js_node_worker_ipc" in command_text
    if is_workflow_js_node:
        return "workflow js node sandbox" if sandbox_enabled else "workflow js node worker"
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
    for prefix in ("rbac-", "backend_rbac-", "backend-rbac-", "transport-"):
        if sid.startswith(prefix):
            sid = sid[len(prefix):]
            break
    for suffix in ("-private", ".private"):
        if sid.endswith(suffix):
            sid = sid[: -len(suffix)]
            break
    return sid.strip()


def _private_key_input_path(value: str) -> tuple[Optional[Path], Optional[str]]:
    raw = str(value or "").strip()
    if not raw:
        return None, None
    unquoted = raw
    while len(unquoted) >= 2 and unquoted[0] == unquoted[-1] and unquoted[0] in {"'", '"'}:
        unquoted = unquoted[1:-1].strip()
    sanitized = unquoted.replace('"', "")
    recovered = ""
    for idx in range(1, max(1, len(sanitized) - 2)):
        if sanitized[idx] in {"\\", "/"} and sanitized[idx + 1].isalpha() and sanitized[idx + 2] == ":":
            recovered = sanitized[idx + 1 :]
    candidates = []
    for candidate in (unquoted, recovered, sanitized):
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        try:
            path = Path(os.path.expanduser(candidate)).resolve()
        except Exception:
            continue
        if path.is_file():
            return path, None
    path_like = (
        "\\" in raw
        or "/" in raw
        or raw.startswith("~")
        or raw.lower().endswith((".json", ".pem", ".key"))
        or (len(raw) >= 2 and raw[1] == ":")
    )
    if path_like:
        shown = sanitized if sanitized != raw else raw
        return None, f"Private key input looks like a file path, but no file was found: {shown}"
    return None, None


def _extract_key_id_from_private_key_json(payload: Dict[str, Any]) -> Optional[str]:
    data = dict(payload or {})
    metadata = dict(data.get("metadata") or {})
    for source in (metadata, data):
        key_id = str(source.get("key_id") or source.get("admin_key_id") or "").strip()
        if key_id:
            return key_id
    derived = _key_id_from_secret_id(str(data.get("secret_id") or ""))
    return derived or None


def _print_invalid_challenge_signature_diagnostic(
    args: argparse.Namespace,
    *,
    key_id: str,
    pk_file_path: Optional[Path],
    json_payload: Optional[Dict[str, Any]],
) -> None:
    print(_c("bad", "Authentication failed: private key signature does not match the registered public key."))
    print(_c("muted", f"  Key ID used: {key_id or '<unknown>'}"))
    if pk_file_path is not None:
        print(_c("muted", f"  Private key input: {pk_file_path}"))
    secret_id = ""
    if isinstance(json_payload, dict):
        secret_id = str(json_payload.get("secret_id") or "").strip()
        if secret_id:
            print(_c("muted", f"  SecretRecord id: {secret_id}"))
    try:
        keyring_path = pk_file_path.parent.parent / "keyring" / "keys.json" if pk_file_path is not None else None
        if keyring_path is not None and keyring_path.exists():
            keyring = json.loads(keyring_path.read_text(encoding="utf-8"))
            key_row = dict(dict(keyring.get("keys") or {}).get(str(key_id or "").strip()) or {})
            expected_secret_id = str(key_row.get("private_key_secret_id") or "").strip()
            if expected_secret_id and secret_id and expected_secret_id != secret_id:
                print(_c("warn", f"  Client realm keyring maps this key id to SecretRecord {expected_secret_id}, not {secret_id}."))
    except Exception:
        pass
    if _target_mode(args) == "ssh":
        return
    try:
        status = _offline_service(args).auth_status()
    except Exception:
        return
    for row in list(dict(status or {}).get("local_private_key_custody") or []):
        item = dict(row or {})
        if str(item.get("key_id") or "").strip() != str(key_id or "").strip():
            continue
        expected = str(item.get("private_key_secret_path") or item.get("private_key_export_path") or "").strip()
        exists = item.get("private_key_secret_exists")
        warning = str(item.get("private_key_warning") or "").strip()
        if expected:
            print(_c("muted", f"  Registered key custody path: {expected}"))
        if exists is False:
            print(_c("warn", "  Registered key custody file is missing."))
        if warning:
            print(_c("warn", f"  Custody warning: {warning}"))
        if expected and pk_file_path is not None and str(Path(expected).resolve()).lower() != str(pk_file_path.resolve()).lower():
            print(_c("warn", "  The file you supplied is not the registered custody path for this key."))
        break
    print(_c("muted", "  Use the private key that was generated/imported with the current hosting access setup, or re-register this public key."))


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

    # Check if it's a file path. Windows paths sometimes get pasted with a quoted
    # filename component; normalize that before falling back to raw key text.
    pk_input_path = None
    if len(lines) == 1:
        pk_input_path, path_error = _private_key_input_path(lines[0])
        if path_error:
            print(_c('bad', path_error))
            return None

    if pk_input_path is not None:
        try:
            pk_file_path = pk_input_path
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

        invoke = invoke_fn or (lambda cmd, payload: _auth_api_invoke(args, cmd, payload))
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
        if "invalid_challenge_signature" in str(e):
            _print_invalid_challenge_signature_diagnostic(
                args,
                key_id=locals().get("key_id", ""),
                pk_file_path=locals().get("pk_file_path"),
                json_payload=locals().get("json_payload"),
            )
        else:
            print(_c('bad', f"Authentication error: {e}"))
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _configured_auth_defaults(args: argparse.Namespace) -> tuple[str, str, int]:
    try:
        settings = dict(_control_channel(args).control_settings or {})
    except Exception:
        settings = {}
    key_id = str(settings.get("engine_host_key_id") or "").strip() or "admin-main"
    key_secret = str(settings.get("engine_host_key_secret") or "").strip()
    try:
        ttl = int(settings.get("engine_host_session_ttl_seconds") or 900)
    except Exception:
        ttl = 900
    return key_id, key_secret, ttl


def _obtain_shared_secret_session_token(args: argparse.Namespace) -> Optional[str]:
    if _target_mode(args) == "ssh":
        print(_c("bad", "Shared key password authentication is local-only. Use private-key challenge auth for SSH targets."))
        return None

    default_key_id, configured_secret, ttl = _configured_auth_defaults(args)
    print(f"\n{_c('warn', 'Authenticate with local shared key password.')}")
    print(_c("muted", "This uses auth-issue-session and is accepted only by local-only shared-secret keys."))
    try:
        key_id_raw = input(f"Key ID [{default_key_id}]: ").strip()
        key_id = key_id_raw or default_key_id
        if configured_secret:
            secret = getpass.getpass("Shared key password [configured secret if empty]: ")
            if not secret:
                secret = configured_secret
        else:
            secret = getpass.getpass("Shared key password: ")
    except (KeyboardInterrupt, EOFError):
        return None

    if not key_id or not secret:
        print(_c("bad", "Key ID and shared key password are required."))
        return None

    try:
        res = _api_invoke(
            args,
            "auth-issue-session",
            {
                "key_id": key_id,
                "key_secret": secret,
                "scope": "control",
                "ttl_seconds": ttl,
            },
        )
    except PermissionError as exc:
        print(_c("bad", f"Shared key authentication failed: {exc}"))
        return None
    except Exception as exc:
        print(_c("bad", f"Shared key authentication error: {exc}"))
        return None

    token = str(dict(res or {}).get("token") or "").strip() if isinstance(res, dict) else ""
    if not token:
        print(_c("bad", "Authentication failed: no token returned."))
        return None
    print(_c("good", "Authenticated successfully."))
    return token


def _authenticate_interactive(args: argparse.Namespace) -> Optional[str]:
    if _target_mode(args) == "ssh":
        return _obtain_session_token(args)

    opts = {
        "key": ("Admin private key challenge", ""),
        "shared": ("Shared key password", "local only"),
    }
    choice = _prompt_menu("Authenticate", opts, "key", allow_back=True, allow_changes=False, enter_hint="key")
    if choice in ("b", "back"):
        return None
    if choice == "shared":
        return _obtain_shared_secret_session_token(args)
    if choice == "key":
        return _obtain_session_token(args)
    return None

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
                session_token = _set_interactive_session_token(args, token)
                _ensure_session_renewer(args)
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
            session_token = _set_interactive_session_token(args, None)
        elif choice == "f":
            _force_stop_local_daemon(args)
            session_token = _set_interactive_session_token(args, None)
        elif choice == "n":
            _force_restart_local_daemon(args)
            session_token = _set_interactive_session_token(args, None)
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

    try:
        while True:
            try:
                session_token = _active_session_token(args, session_token)
                target_mode = _target_mode(args)
                daemon_status: Dict[str, Any] = {}
                if target_mode == "ssh":
                    daemon_up = _is_daemon_running(args, session_token=session_token)
                else:
                    try:
                        channel = (
                            _control_channel(args)
                            if session_token is None
                            else _control_channel(args, session_token=session_token)
                        )
                        daemon_status = channel.get_daemon_status()
                        daemon_up = bool(daemon_status.get("alive") or daemon_status.get("reachable"))
                    except Exception:
                        daemon_up = False
                if session_token and daemon_up:
                    _ensure_session_renewer(args)
                session_token = _renew_session_token_if_needed(args, session_token, daemon_up=daemon_up)
                status_c = _c("good", "Running") if daemon_up else _c("muted", "Stopped")
                auth_status = daemon_status.get("auth_status") or {}
                auth_value = daemon_status.get("require_auth") if "require_auth" in daemon_status else auth_status.get("require_auth")
                caller_key = auth_status.get("caller_key_id")
                caller_role = auth_status.get("caller_role")
                python_runtime_rows: list[tuple[str, str]] = []
                
                # Print a more informative summary if daemon is running
                if daemon_up:
                    status_parts = []
                    pid = daemon_status.get("pid")
                    if pid:
                        status_parts.append(f"PID: {pid}")
                    try:
                        res = _api_invoke(args, "host-metrics", {}, session_token=session_token)
                        session_token = _active_session_token(args, session_token)
                        python_runtime_rows = _python_runtime_rows(res if isinstance(res, dict) else None)
                        if not pid and res.get("pid"):
                            status_parts.append(f"PID: {res.get('pid')}")
                        if "require_auth" in res and auth_value is None:
                            auth_value = res.get("require_auth")
                        res_auth_status = res.get("auth_status") or {}
                        if res_auth_status.get("caller_key_id"):
                            caller_key = res_auth_status.get("caller_key_id")
                            caller_role = res_auth_status.get("caller_role")
                        if session_token and not caller_key:
                            caller_key, caller_role = _lookup_current_session_identity(args, session_token)
                        worker_status = _worker_status_summary(args, session_token, metrics=res if isinstance(res, dict) else None)
                        workers = worker_status.get("workers_count")
                        worker_cpu = worker_status.get("worker_cpu_percent")
                        worker_mem = worker_status.get("worker_memory_mb")
                        worker_vram = worker_status.get("worker_gpu_vram_mb")
                        worker_vram_pending = bool(worker_status.get("worker_gpu_vram_pending"))
                        
                        if caller_key and caller_role:
                            status_parts.append(f"Auth: {caller_key} ({caller_role})")
                        elif auth_value is not None:
                            status_parts.append(f"Auth: {'required' if bool(auth_value) else 'not required'}")
                        if workers is not None:
                            status_parts.extend([
                                f"Workers: {workers}",
                                f"Worker CPU: {_format_percent_or_na(worker_cpu)}",
                                f"Worker RSS: {_format_mb_or_na(worker_mem)}",
                                f"Worker VRAM: {_format_gb_from_mb_or_pending(worker_vram, pending=worker_vram_pending)}",
                            ])
                    except PermissionError as pe:
                        if caller_key and caller_role:
                            status_parts.append(f"Auth: {caller_key} ({caller_role})")
                        elif auth_value is not None:
                            status_parts.append(f"Auth: {'required' if bool(auth_value) else 'not required'}")
                        if "session_token_required" in str(pe) and auth_value is None and not caller_key:
                            status_parts.append(_c("warn", "Auth required"))
                    except Exception:
                        if caller_key and caller_role:
                            status_parts.append(f"Auth: {caller_key} ({caller_role})")
                        elif auth_value is not None:
                            status_parts.append(f"Auth: {'required' if bool(auth_value) else 'not required'}")
                    if status_parts:
                        status_c += f" ({', '.join(status_parts)})"
                else:
                    if caller_key and caller_role:
                        status_c += f" (Auth: {caller_key} ({caller_role}))"
                    elif session_token:
                        status_c += " (Auth refresh paused; daemon stopped)"
                    elif auth_value is not None:
                        status_c += f" (Auth: {'required' if bool(auth_value) else 'not required'})"

                print()
                _print_title("Engine Host Interactive Control")
                rows = [("Daemon", status_c)]
                rows.extend(python_runtime_rows)
                _kv_rows(rows, min_width = 6)
                
                lifecycle_label = "Restart remote daemon" if target_mode == "ssh" else ("Start daemon" if not daemon_up else "Stop daemon")
                opts: Dict[str, tuple[str, str]] = {}
                opts.update({
                    "l": ("List loaded engines and sandboxes", ""),
                    "o": ("Load engine from config", ""),
                    "d": ("Engine/Sandbox details", ""),
                    "j": ("Manage workflow runtimes", ""),
                    "m": ("Print daemon metrics", ""),
                    "t": ("Test loaded model prompt", ""),
                    "c": ("List live consumers", ""),
                    "a": ("List auth sessions", ""),
                    "k": ("Kill/Disconnect resource", ""),
                    "s": (lifecycle_label, ""),
                    "r": ("Local recovery/auth tools", "" if target_mode != "ssh" else "local only"),
                })
                if daemon_up and bool(auth_value):
                    auth_label = "Re-authenticate / switch identity" if (caller_key or session_token) else "Authenticate"
                    auth_hint = f"{caller_key} ({caller_role})" if caller_key and caller_role else ""
                    opts["auth"] = (auth_label, auth_hint)
                choice = _prompt_menu("Main Menu", opts, "refresh", allow_changes=False, enter_hint="refresh")
                if choice == "q":
                    return 0
                if choice == "refresh":
                    continue

                try:
                    if choice == "auth":
                        token = _authenticate_interactive(args)
                        if token:
                            session_token = _set_interactive_session_token(args, token)
                            _ensure_session_renewer(args)
                            print(_c("good", "Please try your command now that you are authenticated."))
                            time.sleep(1)
                    elif choice == "l":
                        session_token = _list_engines(args, session_token)
                    elif choice == "o":
                        session_token = _load_engine(args, session_token)
                    elif choice == "d":
                        session_token = _engine_details(args, session_token)
                    elif choice == "j":
                        session_token = _manage_workflow_runtimes(args, session_token)
                    elif choice == "m":
                        session_token = _show_metrics(args, session_token)
                    elif choice == "t":
                        session_token = _test_loaded_model(args, session_token)
                    elif choice == "c":
                        session_token = _list_live_consumers(args, session_token)
                    elif choice == "a":
                        session_token = _list_auth_sessions(args, session_token)
                    elif choice == "k":
                        session_token = _kill_resource(args, session_token)
                    elif choice == "s":
                        if target_mode == "ssh":
                            _start_daemon(args)
                        elif daemon_up:
                            _stop_daemon(args)
                            _stop_session_renewer(args)
                        else:
                            _start_daemon(args)
                            if session_token:
                                _ensure_session_renewer(args)
                    elif choice == "r":
                        session_token = _local_recovery_menu(args, session_token)
                except PermissionError as pe:
                    if "session_token_required" in str(pe):
                        if _target_mode(args) != "ssh" and not _is_daemon_running(args, session_token=session_token):
                            print(_c('warn', "Daemon is stopped. Start the daemon before authenticating or running protected commands."))
                            time.sleep(1)
                            continue
                        print(_c("bad", "Command requires authentication. Choose Authenticate from the main menu first."))
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
        _stop_session_renewer(args)
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

    if not sessions:
        print("  No active auth sessions.")
        return

    print(_c('muted', "  Auth sessions are issued tokens; they are not a live socket/process count."))
    print()
    for sess in sessions:
        tok = sess.get("token_preview") or sess.get("token_prefix") or "<unknown>"
        key_id = sess.get("key_id", "<unknown>")
        scope = sess.get("scope", "")
        is_current_cli = bool(cli_preview and tok == cli_preview)
        marker = f" {_c('good', '(this interactive CLI)')}" if is_current_cli else ""
        print(f"  - Session [{_c('accent', tok)}] Key: {_c('value', key_id)} Scope: {scope}{marker}")

        ttl = sess.get("ttl_remaining_seconds")
        if ttl is not None:
            print(f"    Expires in: {ttl} seconds")
        elif "expires_at" in sess and sess["expires_at"] > 0:
            print(f"    Expires at: {sess['expires_at']}")
        issued_at = float(sess.get("issued_at") or 0.0)
        if issued_at > 0:
            age = max(0, int(time.time() - issued_at))
            print(f"    Issued: {age} seconds ago")

        role = sess.get("role")
        if role:
            print(f"    Role: {role}")

        if is_current_cli:
            print("    Consumer: interactive CLI")

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


def _print_live_consumers(res: Dict[str, Any], session_token: Optional[str]) -> None:
    rows = [dict(item or {}) for item in list(dict(res or {}).get("connections") or []) if isinstance(item, dict)]
    cli_preview = _get_token_preview(session_token) if session_token else None
    if not rows:
        print("  No live authenticated consumer connections.")
        return

    for row in rows:
        cid = str(row.get("connection_id") or "<unknown>")
        transport = str(row.get("transport") or "unknown")
        peer = str(row.get("peer_host") or "local")
        pid = row.get("pid")
        kind = str(row.get("consumer_kind") or dict(row.get("process") or {}).get("consumer_kind") or "").strip()
        previews = [str(x) for x in list(row.get("session_token_previews") or []) if str(x or "").strip()]
        is_current_cli = bool(cli_preview and cli_preview in previews)
        marker = f" {_c('good', '(this interactive CLI)')}" if is_current_cli else ""
        details = [f"Transport: {transport}", f"Peer: {peer}"]
        if pid:
            details.append(f"PID: {pid}")
        if kind:
            details.append(f"Kind: {kind}")
        print(f"  - Connection [{_c('accent', cid[:12])}] {' '.join(details)}{marker}")
        process = dict(row.get("process") or {})
        if process.get("name") or process.get("parent_pid"):
            proc_bits = []
            if process.get("name"):
                proc_bits.append(f"name={process.get('name')}")
            if process.get("parent_pid"):
                proc_bits.append(f"ppid={process.get('parent_pid')}")
            print(f"    Process: {', '.join(proc_bits)}")
        actors = [str(x) for x in list(row.get("actor_ids") or []) if str(x or "").strip()]
        if actors:
            print(f"    Actors: {', '.join(actors)}")
        if previews:
            print(f"    Session Tokens: {', '.join(previews)}")
        age = row.get("age_seconds")
        idle = row.get("idle_seconds")
        bits = []
        if age is not None:
            bits.append(f"age={age}s")
        if idle is not None:
            bits.append(f"idle={idle}s")
        command_count = row.get("command_count")
        if command_count is not None:
            bits.append(f"commands={command_count}")
        if row.get("last_command"):
            bits.append(f"last={row.get('last_command')}")
        if bits:
            print(f"    Activity: {', '.join(bits)}")

    actors = [dict(item or {}) for item in list(dict(res or {}).get("actors") or []) if isinstance(item, dict)]
    if actors:
        print()
        print("  Actor connection counts:")
        for row in actors:
            print(f"    - {row.get('actor_id')}: {row.get('connection_count')}")


def _workflow_environment_key(info: Dict[str, Any]) -> str:
    env = dict(dict(info or {}).get("environment") or {})
    caps = dict(dict(info or {}).get("capabilities") or {})
    return str(env.get("environment_key") or caps.get("environment_key") or "").strip()


def _workflow_pool_active_request_ids(metrics: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for request_id in list(metrics.get("active_request_ids") or []):
        rid = str(request_id or "").strip()
        if rid and rid not in seen:
            seen.add(rid)
            out.append(rid)
    for worker in [dict(row or {}) for row in list(metrics.get("workers") or []) if isinstance(row, dict)]:
        for request_id in list(worker.get("active_request_ids") or []):
            rid = str(request_id or "").strip()
            if rid and rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out


def _print_workflow_request_status(result: Dict[str, Any]) -> None:
    data = dict(result or {})
    request = dict(data.get("request") or {})
    rows = [
        ("Environment Key", data.get("environment_key") or request.get("environment_key") or "<unknown>"),
        ("Request ID", request.get("request_id") or data.get("request_id") or "<unknown>"),
        ("Request Status", request.get("status") or data.get("status") or "<unknown>"),
        ("Reason", request.get("reason") or data.get("reason") or ""),
        ("Worker", request.get("worker_engine_id") or data.get("engine_id") or ""),
        ("Stream Events", request.get("stream_event_count")),
    ]
    latest = dict(request.get("latest_progress") or {})
    if latest:
        rows.append(("Latest Progress", latest.get("message") or latest.get("stage") or latest.get("type") or latest))
    _kv_rows(rows)


def _print_workflow_stream_events(result: Dict[str, Any]) -> None:
    data = dict(result or {})
    events = [dict(row or {}) for row in list(data.get("normalized_events") or []) if isinstance(row, dict)]
    if not events:
        batch = dict(data.get("batch") or {})
        context = dict(batch.get("context") or {})
        base = dict(batch.get("base") or {})
        sequence_base = int(base.get("sequence") or 0)
        timestamp_base = int(base.get("timestamp_ms") or 0)
        for index, frame in enumerate(list(batch.get("frames") or [])):
            if not isinstance(frame, dict):
                continue
            row = {**context, **dict(frame or {})}
            row.setdefault("sequence", sequence_base + index)
            row.setdefault("timestamp_ms", timestamp_base + int(row.get("dt_ms") or 0))
            events.append(row)
    if not events:
        print("  No events available.")
        return
    print("Events:")
    for event in events:
        payload = dict(event.get("payload") or {}) if isinstance(event.get("payload"), dict) else dict(event)
        summary = ""
        kind = str(event.get("kind") or event.get("type") or "").strip()
        if kind == "stream_loss":
            summary = str(event.get("loss") or {}).strip()
        elif kind == "error":
            error = dict(payload.get("error") or {})
            summary = str(error.get("code") or error.get("message") or "").strip()
        elif kind == "log":
            logs = dict(payload.get("logs") or {})
            summary = str(logs.get("summary") or "").strip() or f"limit={logs.get('output_limit_bytes')}"
        elif kind in {"progress", "metric"}:
            summary = str(payload.get("message") or payload.get("stage") or payload).strip()
        elif kind == "artifact":
            bits = [
                str(payload.get("name") or "").strip(),
                str(payload.get("artifact_kind") or payload.get("kind_label") or "").strip(),
                str(payload.get("ref") or "").strip(),
                str(payload.get("filename") or "").strip(),
            ]
            summary = " ".join(bit for bit in bits if bit)
            if payload.get("size_bytes") is not None:
                summary = f"{summary} size={payload.get('size_bytes')}".strip()
        elif kind in {"result", "done", "canceled", "started"}:
            summary = str(payload.get("status") or payload.get("request_id") or "").strip()
        print(f"  - {kind}: {summary}")


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
            session_token = _active_session_token(args, session_token)
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
            details.extend(_resource_bits(dict(info.get("process_resources") or {})))
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
            reachability_note = _reachability_summary(info)
            if reachability_note:
                print(f"    Reachability: {_c('warn', reachability_note)}")
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
            session_token = _active_session_token(args, session_token)
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
        resource_bits = _resource_bits(dict(info.get("process_resources") or {}))
        if resource_bits:
            _kv_rows([
                ("Resources", ", ".join(resource_bits)),
            ])
        js_resources = dict(info.get("process_resources") or {})
        if "workflow_js_node_process_count" in js_resources:
            pids = [str(x) for x in list(js_resources.get("workflow_js_node_pids") or []) if str(x or "").strip()]
            _kv_rows([
                ("JS Capacity", js_resources.get("workflow_js_capacity")),
                ("JS Active Calls", js_resources.get("workflow_js_active_calls")),
                ("JS Available Slots", js_resources.get("workflow_js_available_slots")),
                ("JS Node Processes", (
                    f"active={js_resources.get('workflow_js_active_node_process_count')}, "
                    f"idle={js_resources.get('workflow_js_idle_node_process_count')}, "
                    f"total={js_resources.get('workflow_js_node_process_count')}"
                )),
                ("JS Node PIDs", ", ".join(pids) if pids else "<none>"),
                ("JS Active Requests", ", ".join(str(x) for x in list(js_resources.get("workflow_js_active_request_ids") or [])) or "<none>"),
                ("JS Node CPU", _format_percent_or_na(js_resources.get("workflow_js_node_cpu_percent"))),
                ("JS Node RSS", _format_mb_or_na(js_resources.get("workflow_js_node_memory_mb"))),
                ("JS Max Requests/Node", js_resources.get("workflow_js_max_requests_per_node")),
            ])
            node_rows = [dict(row or {}) for row in list(js_resources.get("workflow_js_node_processes") or []) if isinstance(row, dict)]
            if node_rows:
                print()
                print("JS Node Pool:")
                for node in node_rows:
                    metrics = dict(node.get("resources") or {})
                    bits = [
                        f"pid={node.get('pid')}",
                        f"busy={'yes' if bool(node.get('busy')) else 'no'}",
                        f"requests={node.get('request_count')}",
                    ]
                    active_request_id = str(node.get("active_request_id") or "").strip()
                    if active_request_id:
                        bits.append(f"request={active_request_id}")
                    if metrics.get("cpu_percent") is not None:
                        bits.append(f"cpu={_format_percent_or_na(metrics.get('cpu_percent'))}")
                    if metrics.get("memory_mb") is not None:
                        bits.append(f"rss={_format_mb_or_na(metrics.get('memory_mb'))}")
                    print(f"  - {', '.join(bits)}")
        if "workflow_python_process_count" in js_resources:
            pids = [str(x) for x in list(js_resources.get("workflow_python_pids") or []) if str(x or "").strip()]
            _kv_rows([
                ("Python Capacity", js_resources.get("workflow_python_capacity")),
                ("Python Active Calls", js_resources.get("workflow_python_active_calls")),
                ("Python Available Slots", js_resources.get("workflow_python_available_slots")),
                ("Python Processes", (
                    f"active={js_resources.get('workflow_python_active_process_count')}, "
                    f"idle={js_resources.get('workflow_python_idle_process_count')}, "
                    f"total={js_resources.get('workflow_python_process_count')}"
                )),
                ("Python PIDs", ", ".join(pids) if pids else "<none>"),
                ("Python Active Requests", ", ".join(str(x) for x in list(js_resources.get("workflow_python_active_request_ids") or [])) or "<none>"),
                ("Python CPU", _format_percent_or_na(js_resources.get("workflow_python_cpu_percent"))),
                ("Python RSS", _format_mb_or_na(js_resources.get("workflow_python_memory_mb"))),
            ])
            proc_rows = [dict(row or {}) for row in list(js_resources.get("workflow_python_processes") or []) if isinstance(row, dict)]
            if proc_rows:
                print()
                print("Python Pool:")
                for proc in proc_rows:
                    metrics = dict(proc.get("resources") or {})
                    bits = [
                        f"pid={proc.get('pid')}",
                        f"busy={'yes' if bool(proc.get('busy')) else 'no'}",
                        f"requests={proc.get('request_count')}",
                    ]
                    active_request_id = str(proc.get("active_request_id") or "").strip()
                    if active_request_id:
                        bits.append(f"request={active_request_id}")
                    if metrics.get("cpu_percent") is not None:
                        bits.append(f"cpu={_format_percent_or_na(metrics.get('cpu_percent'))}")
                    if metrics.get("memory_mb") is not None:
                        bits.append(f"rss={_format_mb_or_na(metrics.get('memory_mb'))}")
                    print(f"  - {', '.join(bits)}")
        reachability_note = _reachability_summary(info)
        if reachability_note:
            print()
            _kv_rows([
                ("Reachability", reachability_note),
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


def _manage_workflow_runtimes(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Workflow Runtimes")
    try:
        if _can_use_offline_local_fallback(args, session_token=session_token):
            print(_c('warn', "  Daemon is stopped. Workflow runtime management requires a running daemon."))
            return session_token
        res = _api_invoke(args, "discover-running", {}, session_token=session_token)
        session_token = _active_session_token(args, session_token)
        engines = _get_engines_dict(res)
        helpers = {
            eid: info
            for eid, info in engines.items()
            if str(dict(info or {}).get("executor_kind") or "").strip() in {"workflow_python_helper", "workflow_js_node"}
        }
        if not helpers:
            print("  No workflow runtimes are loaded.")
            return session_token
        opts = {}
        for eid, info in helpers.items():
            resources = dict(dict(info or {}).get("process_resources") or {})
            executor = str(dict(info or {}).get("executor_kind") or "").strip()
            is_js = executor == "workflow_js_node"
            prefix = "workflow_js" if is_js else "workflow_python"
            cap = resources.get(f"{prefix}_capacity")
            active = (
                resources.get("workflow_js_active_calls")
                if is_js
                else resources.get("workflow_helper_pool_active_process_count")
            )
            total = (
                resources.get("workflow_js_node_process_count")
                if is_js
                else resources.get("workflow_helper_pool_process_count")
            )
            label = "JS node" if is_js else "Python helper"
            env_key = _workflow_environment_key(dict(info or {}))
            hint_bits = [label]
            if env_key:
                hint_bits.append(f"env={env_key[:16]}")
            if cap is not None:
                hint_bits.append(f"capacity={cap}")
            if active is not None:
                hint_bits.append(f"active={active}")
            if total is not None:
                hint_bits.append(f"total={total}")
            hint = " ".join(hint_bits)
            opts[eid] = (f"Manage {eid}", hint)
        choice = _prompt_menu("Select Workflow Runtime", opts, "b", allow_back=True, allow_changes=False)
        if choice in ("b", "back"):
            return session_token
        selected_info = dict(helpers.get(choice) or {})
        selected_executor = str(selected_info.get("executor_kind") or "").strip()
        is_python = selected_executor == "workflow_python_helper"
        runtime_profile = "helper" if is_python else "node"
        runtime_kind = "workflow-python" if is_python else "workflow-js"
        environment_key = _workflow_environment_key(selected_info)
        command_prefix = runtime_kind
        helper_label = "Workflow Python helper runtime" if is_python else "Workflow JS node runtime"
        while True:
            resource_payload = {
                "engine_id": choice,
                "profile": runtime_profile,
            }
            if environment_key:
                resource_payload["environment_key"] = environment_key
            resources = dict(
                _api_invoke(
                    args,
                    f"{command_prefix}-resources",
                    resource_payload,
                    session_token=session_token,
                )
                or {}
            )
            session_token = _active_session_token(args, session_token)
            workflow_pool = dict(resources.get("workflow_pool") or {})
            workflow_pool_metrics = dict(workflow_pool.get("metrics") or {})
            pool = dict(resources.get("pool") or resources.get("node_pool") or resources.get("node_runtime") or {})
            node_rows = [
                dict(row or {})
                for row in list(pool.get("processes") or pool.get("node_processes") or [])
                if isinstance(row, dict)
            ]
            active_request_ids = [
                str(row.get("active_request_id") or row.get("request_id") or "").strip()
                for row in node_rows
                if str(row.get("active_request_id") or row.get("request_id") or "").strip()
            ]
            if not active_request_ids:
                active_request_ids = [
                    str(x or "").strip()
                    for x in list(pool.get("active_request_ids") or [])
                    if str(x or "").strip()
                ]
            if workflow_pool_metrics:
                active_request_ids = _workflow_pool_active_request_ids(workflow_pool_metrics)
            capacity_value = (
                workflow_pool_metrics.get("desired_capacity")
                if workflow_pool_metrics
                else resources.get("capacity") or pool.get("capacity")
            )
            active_value = (
                workflow_pool_metrics.get("active_calls")
                if workflow_pool_metrics
                else resources.get("active_calls")
            )
            available_value = (
                workflow_pool_metrics.get("available_slots")
                if workflow_pool_metrics
                else resources.get("available_slots")
            )
            _kv_rows([
                ("Engine ID", choice),
                ("Environment Key", environment_key or resources.get("environment_key") or "<legacy helper>"),
                ("Capacity", capacity_value),
                ("Active Calls", active_value),
                ("Available Slots", available_value),
                ("Processes", (
                    f"active={pool.get('active_process_count') if pool.get('active_process_count') is not None else pool.get('active_node_process_count') if pool.get('active_node_process_count') is not None else pool.get('active_count')}, "
                    f"idle={pool.get('idle_process_count') if pool.get('idle_process_count') is not None else pool.get('idle_node_process_count')}, "
                    f"total={pool.get('process_count') if pool.get('process_count') is not None else pool.get('node_process_count') if pool.get('node_process_count') is not None else len(node_rows)}"
                )),
                ("CPU", _format_percent_or_na(pool.get("cpu_percent") if pool.get("cpu_percent") is not None else pool.get("node_cpu_percent") if pool.get("node_cpu_percent") is not None else resources.get("node_cpu_percent") if not is_python else resources.get("python_cpu_percent"))),
                ("RSS", _format_mb_or_na(pool.get("memory_mb") if pool.get("memory_mb") is not None else pool.get("node_memory_mb") if pool.get("node_memory_mb") is not None else resources.get("node_memory_mb") if not is_python else resources.get("python_memory_mb"))),
                ("Active Requests", ", ".join(active_request_ids) or "<none>"),
            ])
            if workflow_pool_metrics:
                _kv_rows([
                    ("Pool ID", workflow_pool.get("pool_id")),
                    ("Pool Workers", workflow_pool_metrics.get("worker_count")),
                    ("Pool Saturation", workflow_pool_metrics.get("saturation_count")),
                    ("Pool Canceled", workflow_pool_metrics.get("cancellation_count")),
                    ("Pool Errors", workflow_pool_metrics.get("error_count")),
                ])
                recent = [
                    dict(row or {})
                    for row in list(workflow_pool_metrics.get("recent_requests") or [])
                    if isinstance(row, dict)
                ]
                if recent:
                    print()
                    print("Recent Requests:")
                    for row in recent[-5:]:
                        bits = [
                            f"id={row.get('request_id')}",
                            f"status={row.get('status')}",
                            f"lifetime_ms={row.get('lifetime_ms')}",
                        ]
                        if row.get("reason"):
                            bits.append(f"reason={row.get('reason')}")
                        print(f"  - {', '.join(bits)}")
            if node_rows:
                print()
                print("Process Pool:")
                for node in node_rows:
                    metrics = dict(node.get("resources") or {})
                    bits = [
                        f"pid={node.get('pid')}",
                        f"busy={'yes' if bool(node.get('busy')) else 'no'}",
                        f"requests={node.get('request_count')}",
                    ]
                    active_request_id = str(node.get("active_request_id") or node.get("request_id") or "").strip()
                    if active_request_id:
                        bits.append(f"request={active_request_id}")
                    if metrics.get("cpu_percent") is not None:
                        bits.append(f"cpu={_format_percent_or_na(metrics.get('cpu_percent'))}")
                    if metrics.get("memory_mb") is not None:
                        bits.append(f"rss={_format_mb_or_na(metrics.get('memory_mb'))}")
                    print(f"  - {', '.join(bits)}")
            action_opts = {
                "s": ("Set capacity", ""),
                "r": ("Refresh", ""),
                "e": ("Ensure workflow runtime", f"Use {runtime_profile} profile"),
            }
            if active_request_ids:
                action_opts["c"] = ("Cancel request", "Kill the child process currently running a request")
            status_prefix = "workflow-python" if is_python else "workflow-js"
            can_use_runtime_status = bool(environment_key)
            if can_use_runtime_status:
                action_opts["i"] = ("Inspect request", "Show request lifetime/progress by environment key")
            action_opts["v"] = ("Receive stream events", f"Read {command_prefix} stream events by stream id")
            action = _prompt_menu("Workflow Runtime Action", action_opts, "b", allow_back=True, allow_changes=False)
            if action in ("b", "back"):
                return session_token
            if action == "r":
                continue
            if action == "e":
                payload = {
                    "profile": runtime_profile,
                    "engine_id": choice,
                    "capacity": int(capacity_value or 1),
                }
                if environment_key:
                    payload["environment_key"] = environment_key
                out = _api_invoke(args, f"{command_prefix}-ensure", payload, session_token=session_token)
                session_token = _active_session_token(args, session_token)
                result = dict(out or {})
                environment_key = str(result.get("environment_key") or environment_key or "").strip()
                print(_c("good", f"{helper_label} ensured for {choice}."))
                continue
            if action == "s":
                raw = input("New capacity [leave blank to keep]: ").strip()
                if not raw:
                    continue
                try:
                    capacity = max(1, min(int(raw), 256))
                except Exception:
                    print(_c("bad", "Capacity must be an integer from 1 to 256."))
                    continue
                out = _api_invoke(
                    args,
                    f"{command_prefix}-set-capacity",
                    {
                        "engine_id": choice,
                        "profile": runtime_profile,
                        **({"environment_key": environment_key} if environment_key else {}),
                        "capacity": capacity,
                    },
                    session_token=session_token,
                )
                session_token = _active_session_token(args, session_token)
                result = dict(out or {})
                print(_c("good", f"{helper_label} capacity is now {result.get('capacity', capacity)}."))
                continue
            if action == "c":
                default_request = active_request_ids[0] if active_request_ids else ""
                prompt = f"Request id to cancel [{default_request}]: " if default_request else "Request id to cancel: "
                request_id = input(prompt).strip() or default_request
                if not request_id:
                    print(_c("bad", "Request id is required."))
                    continue
                out = _api_invoke(
                    args,
                    f"{command_prefix}-cancel-request",
                    {
                        "engine_id": choice,
                        "profile": runtime_profile,
                        **({"environment_key": environment_key} if environment_key else {}),
                        "request_id": request_id,
                    },
                    session_token=session_token,
                )
                session_token = _active_session_token(args, session_token)
                result = dict(out or {})
                if bool(result.get("canceled")):
                    print(_c("good", f"Canceled workflow runtime request {request_id}."))
                else:
                    print(_c("warn", f"Request was not active: {request_id} ({result.get('reason') or 'not_found'})."))
                continue
            if action == "i":
                if not can_use_runtime_status:
                    print(_c("bad", "Request status requires an environment key."))
                    continue
                default_request = active_request_ids[0] if active_request_ids else ""
                prompt = f"Request id [{default_request}]: " if default_request else "Request id: "
                request_id = input(prompt).strip() or default_request
                if not request_id:
                    print(_c("bad", "Request id is required."))
                    continue
                out = _api_invoke(
                    args,
                    f"{status_prefix}-request-status",
                    {
                        "engine_id": choice,
                        "profile": runtime_profile,
                        "environment_key": environment_key,
                        "request_id": request_id,
                    },
                    session_token=session_token,
                )
                session_token = _active_session_token(args, session_token)
                _print_workflow_request_status(dict(out or {}))
                continue
            if action == "v":
                stream_id = input("Stream id: ").strip()
                if not stream_id:
                    print(_c("bad", "Stream id is required."))
                    continue
                raw_limit = input("Max events [20]: ").strip()
                try:
                    max_items = max(1, min(int(raw_limit or "20"), 100))
                except Exception:
                    print(_c("bad", "Max events must be an integer from 1 to 100."))
                    continue
                out = _api_invoke(
                    args,
                    f"{command_prefix}-event-subscribe",
                    {"stream_id": stream_id, "max_items": max_items},
                    session_token=session_token,
                )
                session_token = _active_session_token(args, session_token)
                _print_workflow_stream_events(dict(out or {}))
                continue
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
            session_token = _active_session_token(args, session_token)
            
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


def _test_loaded_model(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Test Loaded Model")
    try:
        res = _api_invoke(args, "discover-running", {}, session_token=session_token)
        session_token = _active_session_token(args, session_token)
        engines = _get_engines_dict(res)
        opts: Dict[str, tuple[str, str]] = {}
        for eid, info in engines.items():
            if not bool(info.get("reachable", False)):
                continue
            loaded_models = [dict(item or {}) for item in list(info.get("loaded_models") or []) if isinstance(item, dict)]
            for model in loaded_models:
                mid = str(model.get("model_instance_id") or model.get("engine_id") or eid).strip()
                mpath = str(model.get("model_path") or model.get("canonical_model_path") or "").strip()
                if mid:
                    opts[mid] = (f"Test {mid}", mpath)
        if not opts:
            print("  No reachable loaded model instances to test.")
            return session_token
        engine_id = _prompt_menu("Select Model", opts, "b", allow_back=True, allow_changes=False, enter_hint="back")
        if engine_id in {"b", "back"}:
            return session_token
        prompt = input("Prompt [Say hello in one short sentence.]: ").strip() or "Say hello in one short sentence."
        max_tokens_raw = input("Max new tokens [64]: ").strip()
        try:
            max_new_tokens = max(1, min(4096, int(max_tokens_raw or "64")))
        except Exception:
            max_new_tokens = 64
        params = {
            "messages_list": [[{"role": "user", "content": prompt}]],
            "stream": True,
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            },
        }
        print(_c("muted", "Sending prompt through hosting proxy-rpc-call..."))
        started = time.perf_counter()
        rpc_result = dict(_api_invoke(
            args,
            "proxy-rpc-call",
            {
                "engine_id": engine_id,
                "method": "run-inference",
                "params": params,
                "timeout_seconds": 300.0,
            },
            session_token=session_token,
        ) or {})
        observed_latency = time.perf_counter() - started
        session_token = _active_session_token(args, session_token)
        response_text, reported_metrics, chunks = _inference_text_and_metrics(rpc_result)
        errors = [str(chunk.get("error") or "").strip() for chunk in chunks if str(chunk.get("error") or "").strip()]
        if errors:
            print(_c("bad", "Model returned error:"))
            for error in errors:
                print(_c("bad", f"  {error}"))
        print()
        print(_c("accent", "Response"))
        print(_c("value", response_text or "<empty response>"))
        print()
        print(_c("accent", "Metrics"))
        _print_inference_metrics(reported_metrics, observed_latency_sec=observed_latency)
        return session_token
    except PermissionError:
        raise
    except Exception as e:
        print(_c("bad", f"Error testing model: {e}"))
        raise e


def _list_live_consumers(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Live Consumer Connections")
    if _can_use_offline_local_fallback(args, session_token=session_token):
        print(_c('warn', "  Daemon is stopped. Live consumers are only available from a running daemon."))
        return session_token
    try:
        res = _api_invoke(args, "list-live-consumers", {}, session_token=session_token)
        session_token = _active_session_token(args, session_token)
        _print_live_consumers(res if isinstance(res, dict) else {}, session_token)
        return session_token
    except PermissionError:
        raise
    except Exception as e:
        print(_c('bad', f"Error listing live consumers: {e}"))
        raise e


def _list_auth_sessions(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Issued Auth Sessions")
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
            session_token = _active_session_token(args, session_token)
        _print_sessions(res, session_token)
        return session_token
    except PermissionError:
        raise
    except Exception as e:
        print(_c('bad', f"Error listing auth sessions: {e}"))
        raise e


def _list_consumers(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    return _list_auth_sessions(args, session_token)


def _load_engine(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Load Engine")
    if _can_use_offline_local_fallback(args, session_token=session_token):
        print(_c('warn', "  Daemon is stopped. Loading an engine requires a running daemon."))
        return session_token
    try:
        while True:
            configs_raw = _api_invoke(args, "list-configs", {}, session_token=session_token)
            session_token = _active_session_token(args, session_token)
            configs = [dict(item or {}) for item in list(configs_raw or []) if isinstance(item, dict)]
            if not configs:
                print("  No hosted configs found.")
                return session_token
            opts: Dict[str, tuple[str, str]] = {}
            by_key: Dict[str, Dict[str, Any]] = {}
            for idx, row in enumerate(configs, start=1):
                key = str(idx)
                name = str(row.get("name") or _config_selector(row))
                hint_bits = []
                if row.get("is_default"):
                    hint_bits.append("default")
                reason = str(row.get("connect_reason") or "").strip()
                if reason:
                    hint_bits.append(reason)
                by_key[key] = row
                opts[key] = (name, " ".join(hint_bits))
            choice = _prompt_menu("Select Config", opts, "b", allow_back=True, allow_changes=False, enter_hint="back")
            if choice in {"b", "back"}:
                return session_token
            selected = by_key.get(choice)
            if not selected:
                continue
            config_selector = _config_selector(selected)

            while True:
                model_path = _configured_model_path_from_config_row(selected)
                generic_worker = _config_uses_generic_worker(selected)
                used_model_menu = False
                if not model_path and not generic_worker:
                    try:
                        models_raw = _api_invoke(args, "models-from-config", {"config_path": config_selector}, session_token=session_token)
                        session_token = _active_session_token(args, session_token)
                    except Exception:
                        models_raw = []
                    models = [dict(item or {}) for item in list(models_raw or []) if isinstance(item, dict)]
                    model_opts: Dict[str, tuple[str, str]] = {}
                    model_by_key: Dict[str, str] = {}
                    for idx, row in enumerate(models, start=1):
                        key = str(idx)
                        path = str(row.get("path") or "").strip()
                        if not path:
                            continue
                        model_by_key[key] = path
                        model_opts[key] = (str(row.get("name") or Path(path).name), path)
                    if model_opts:
                        used_model_menu = True
                        model_opts["p"] = ("Enter model path", "")
                        model_choice = _prompt_menu("Select Model", model_opts, "b", allow_back=True, allow_changes=False, enter_hint="back")
                        if model_choice in {"b", "back"}:
                            break
                        if model_choice == "p":
                            model_path = input("Model path: ").strip()
                        else:
                            model_path = model_by_key.get(model_choice, "")
                    else:
                        print(_c('warn', "  Selected config does not specify a model path."))
                        model_path = input("Model path: ").strip()
                    if not str(model_path or "").strip():
                        print(_c('bad', "Model path is required."))
                        if used_model_menu:
                            continue
                        break

                force_new = False
                target_worker_id = ""
                if not generic_worker:
                    existing_raw = _api_invoke(args, "discover-running", {}, session_token=session_token)
                    session_token = _active_session_token(args, session_token)
                    existing = _get_engines_dict(existing_raw)
                    load_opts: Dict[str, tuple[str, str]] = {}
                    model_by_key: Dict[str, str] = {}
                    target_worker_by_key: Dict[str, str] = {}
                    idx = 1
                    for eid, info in existing.items():
                        loaded_models = [dict(item or {}) for item in list(info.get("loaded_models") or []) if isinstance(item, dict)]
                        for model in loaded_models:
                            mpath = str(model.get("model_path") or model.get("canonical_model_path") or "").strip()
                            mid = str(model.get("model_instance_id") or model.get("engine_id") or eid).strip()
                            if not mpath:
                                continue
                            key = str(idx)
                            idx += 1
                            model_by_key[key] = mpath
                            load_opts[key] = (f"Use running {mid}", mpath)
                        if not loaded_models and _operator_resource_kind(info).endswith("model instance"):
                            worker_id = str(info.get("worker_id") or eid).strip()
                            if worker_id:
                                key = str(idx)
                                idx += 1
                                target_worker_by_key[key] = worker_id
                                load_opts[key] = (f"Load into idle worker {worker_id}", str(model_path))
                    if load_opts:
                        load_opts["auto"] = ("Auto: reuse compatible or create new", str(model_path))
                        load_opts["new"] = ("Force new engine instance", str(model_path))
                        load_choice = _prompt_menu(
                            "Load Target",
                            load_opts,
                            "auto",
                            allow_back=True,
                            allow_changes=False,
                            enter_hint="auto",
                        )
                        if load_choice in {"b", "back"}:
                            if used_model_menu:
                                continue
                            break
                        force_new = load_choice == "new"
                        if load_choice in model_by_key:
                            model_path = model_by_key[load_choice]
                            force_new = False
                        target_worker_id = target_worker_by_key.get(load_choice, "")
                        if target_worker_id:
                            force_new = False
                    else:
                        print(_c("muted", "  No compatible running model workers found; creating a new engine instance."))

                payload = {
                    "config_path": config_selector,
                    "model_path": model_path or None,
                    "force_new_worker": force_new,
                }
                if not generic_worker and str(target_worker_id or "").strip():
                    payload["target_worker_id"] = str(target_worker_id or "").strip()
                if session_token:
                    payload["session_token"] = session_token
                print(_c("muted", "Starting load operation..."))
                started = _api_invoke(args, "op-start", {"command": "connect-from-config", "payload": payload}, session_token=session_token)
                session_token = _active_session_token(args, session_token)
                op_id = str(dict(started or {}).get("operation_id") or "").strip()
                if not op_id:
                    print(_c("bad", "Load did not return an operation id."))
                    return session_token
                last_line = ""
                while True:
                    status = _api_invoke(args, "op-status", {"operation_id": op_id}, session_token=session_token)
                    session_token = _active_session_token(args, session_token)
                    snap = dict(status or {})
                    last_line = _print_progress_snapshot(snap, last_text=last_line)
                    if bool(snap.get("done", False)):
                        result = dict(snap.get("result") or {}) if isinstance(snap.get("result"), dict) else {}
                        op_status = str(snap.get("status") or "").lower()
                        result_status = str(result.get("status") or "").lower()
                        failed = bool(snap.get("error")) or op_status == "failed" or result_status in {"failed", "error"}
                        if not failed:
                            result = dict(snap.get("result") or {})
                            final_status = str(result.get("status") or snap.get("status") or "completed")
                            print(_c("good", f"Load finished: {final_status}"))
                        else:
                            print(_c("bad", f"Load failed: {_operation_failure_message(snap)}"))
                            _print_operation_diagnostics(snap)
                        return session_token
                    time.sleep(1.0)
                break

            continue
    except PermissionError:
        raise
    except KeyboardInterrupt:
        print()
        print(_c("warn", "Load progress display interrupted; operation may still be running."))
        return session_token
    except Exception as e:
        print(_c('bad', f"Error loading engine: {e}"))
        raise e


def _kill_resource(args: argparse.Namespace, session_token: Optional[str]) -> Optional[str]:
    _print_block("Kill Resource")
    try:
        if _can_use_offline_local_fallback(args, session_token=session_token):
            print(_c('warn', "  Daemon is stopped. Kill/disconnect actions require a running daemon."))
            return session_token
        while True:
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
                    continue
                ech = _prompt_menu("Select Model Binding", model_opts, "b", allow_back=True, allow_changes=False)
                if ech in ("b", "back"): continue
                print(f"Unloading {ech}...")
                unload_result = dict(_api_invoke(
                    args,
                    "unload-model",
                    {"engine_id": ech, "timeout_seconds": 120.0},
                    session_token=session_token,
                ) or {})
                session_token = _active_session_token(args, session_token)
                status = str(unload_result.get("status") or "unknown").strip()
                if status not in {"unloaded", "not_found"}:
                    print(_c("bad", f"Unload failed: {status}"))
                    message = str(unload_result.get("message") or unload_result.get("error") or "").strip()
                    if message:
                        print(_c("muted", f"  Detail: {message}"))
                    continue
                worker_result = dict(unload_result.get("worker") or {}) if isinstance(unload_result.get("worker"), dict) else {}
                worker_status = str(worker_result.get("status") or "").strip()
                if worker_status and worker_status != "ok":
                    print(_c("warn", f"Worker unload returned: {worker_status}"))
                verify_raw = _api_invoke(args, "discover-running", {}, session_token=session_token)
                session_token = _active_session_token(args, session_token)
                verify_engines = _get_engines_dict(verify_raw)
                still_bound = False
                for _, info in verify_engines.items():
                    bindings = [dict(item or {}) for item in list(info.get("config_bindings") or []) if isinstance(item, dict)]
                    models = [dict(item or {}) for item in list(info.get("loaded_models") or []) if isinstance(item, dict)]
                    if any(str(binding.get("engine_id") or "").strip() == ech for binding in bindings):
                        still_bound = True
                    if any(str(model.get("model_instance_id") or model.get("engine_id") or "").strip() == ech for model in models):
                        still_bound = True
                if still_bound:
                    print(_c("bad", "Unload completed but daemon still reports the model binding."))
                else:
                    remaining = unload_result.get("remaining_model_count")
                    suffix = f" Remaining models on worker: {remaining}." if remaining is not None else ""
                    print(_c("good", f"Unload completed.{suffix}"))

            elif ch == "e":
                res = _api_invoke(args, "discover-running", {}, session_token=session_token)
                engines = _get_engines_dict(res)
                if not engines:
                    print("  No workers to stop.")
                    continue
                eopts = {
                    str(info.get("worker_id") or eid): (f"Stop {info.get('worker_id') or eid}", "")
                    for eid, info in engines.items()
                }
                ech = _prompt_menu("Select Worker", eopts, "b", allow_back=True, allow_changes=False)
                if ech in ("b", "back"): continue
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
                    continue
                    
                sch = _prompt_menu("Select Session Preview", sopts, "b", allow_back=True, allow_changes=False)
                if sch in ("b", "back"): continue
                
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
