"""
Terminal-friendly command interface for engine host lifecycle/control.

Modes:
  --daemon              Start long-lived daemon server (foreground)
  --daemon --background Start daemon detached in background
  --daemon-http         Start HTTP ingress daemon (foreground)
  --daemon-http --background Start HTTP ingress daemon detached in background
  --relay               Bridge stdin/stdout to local daemon IPC (SSH channel)
  --relay-wrapper       SSH forced-command wrapper: start detached daemon if allowed, then relay
  <subcommand>          Short-lived: send one command to running daemon (or direct fallback)

Usage examples:
  python -m hosting.engine_host_cli --daemon
  python -m hosting.engine_host_cli --daemon --background
  python -m hosting.engine_host_cli --daemon-http
  python -m hosting.engine_host_cli --daemon-http --background
  python -m hosting.engine_host_cli --relay
  python -m hosting.engine_host_cli --relay-wrapper
  python -m hosting.engine_host_cli discover-running
  python -m hosting.engine_host_cli spawn --payload-stdin < payload.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in {None, ""}:
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    __package__ = "hosting"

from .service.host_service import EngineHostService


EXAMPLES_BY_COMMAND = {
    "discover-running": [
        "python -m hosting.engine_host_cli discover-running",
        "python -m hosting.engine_host_cli --engines-state-file C:\\tmp\\managed_engines.json discover-running",
    ],
    "spawn": [
        "@'{\"engine_id\":\"worker1\",\"command\":[\"python\",\"-m\",\"hosting.engine_worker_ipc\"]}'@ | python -m hosting.engine_host_cli --payload-stdin spawn",
    ],
    "shutdown": [
        "@'{\"engine_id\":\"worker1\"}'@ | python -m hosting.engine_host_cli --payload-stdin shutdown",
    ],
    "ensure-running": [
        "@'{\"engine_id\":\"worker1\"}'@ | python -m hosting.engine_host_cli --payload-stdin ensure-running",
    ],
    "connect-from-config": [
        "@'{\"config_path\":\"default\",\"engine_id\":\"worker_cfg\"}'@ | python -m hosting.engine_host_cli --payload-stdin connect-from-config",
        "@'{\"config_path\":\"default\",\"engine_id\":\"worker_cfg\",\"model_path\":\"C:\\\\models\\\\granite-3.3-2b-instruct\"}'@ | python -m hosting.engine_host_cli --payload-stdin connect-from-config",
    ],
    "list-configs": [
        "python -m hosting.engine_host_cli list-configs",
    ],
    "create-config": [
        "@'{\"name\":\"local_worker\",\"config\":{\"engine_params\":{\"base_model_path\":\"C:\\\\models\\\\granite-3.3-2b-instruct\"}}}'@ | python -m hosting.engine_host_cli --payload-stdin create-config",
    ],
    "claim-engine": [
        "@'{\"engine_id\":\"worker1\",\"backend_id\":\"backend:abc123\",\"exclusive\":false}'@ | python -m hosting.engine_host_cli --payload-stdin claim-engine",
    ],
    "issue-token": [
        "@'{\"engine_id\":\"worker1\",\"backend_id\":\"backend:abc123\"}'@ | python -m hosting.engine_host_cli --payload-stdin issue-token",
    ],
    "inspect-capabilities": [
        "@'{\"engine_id\":\"worker1\"}'@ | python -m hosting.engine_host_cli --payload-stdin inspect-capabilities",
    ],
    "logs-tail": [
        "@'{\"engine_id\":\"worker1\",\"lines\":100}'@ | python -m hosting.engine_host_cli --payload-stdin logs-tail",
    ],
    "logs-follow": [
        "@'{\"engine_id\":\"worker1\",\"cursor\":0,\"max_bytes\":65536}'@ | python -m hosting.engine_host_cli --payload-stdin logs-follow",
    ],
    "sandbox-fs-list": [
        "@'{\"engine_id\":\"worker1\",\"root_id\":\"rw\",\"relative_path\":\"nested\"}'@ | python -m hosting.engine_host_cli --payload-stdin sandbox-fs-list",
    ],
    "sandbox-http-fetch": [
        "@'{\"engine_id\":\"worker1\",\"url\":\"https://example.com/api/test\",\"method\":\"GET\"}'@ | python -m hosting.engine_host_cli --payload-stdin sandbox-http-fetch",
    ],
    "toolbox-describe": [
        "'{\"engine_id\":\"toolbox1\"}' | python -m hosting.engine_host_cli --payload-stdin toolbox-describe",
    ],
    "toolbox-gate": [
        "'{\"toolbox_id\":\"toolbox-demo\",\"tool_name\":\"hello_tool\"}' | python -m hosting.engine_host_cli --payload-stdin toolbox-gate",
    ],
    "toolbox-execute": [
        "'{\"engine_id\":\"toolbox1\",\"execution_request_id\":\"request-1\",\"tool_call\":{\"name\":\"hello_tool\",\"arguments\":{\"name\":\"Sam\"}}}' | python -m hosting.engine_host_cli --payload-stdin toolbox-execute",
    ],
    "hosted-operation-status": [
        "Get-Content operation-ref.json | python -m hosting.engine_host_cli --payload-stdin hosted-operation-status",
    ],
    "hosted-operation-result": [
        "Get-Content operation-ref.json | python -m hosting.engine_host_cli --payload-stdin hosted-operation-result",
    ],
    "hosted-operation-cancel": [
        "'{\"ref\":{...},\"reason\":\"workspace_unload\"}' | python -m hosting.engine_host_cli --payload-stdin hosted-operation-cancel",
    ],
    "hosting-receipt-ledger-cutover": [
        "'{\"acknowledge_replay_window_clear\":true}' | python -m hosting.engine_host_cli --payload-stdin hosting-receipt-ledger-cutover",
    ],
    "toolbox-state-archive-v1": [
        "Get-Content toolbox-state-archive-v1.json | python -m hosting.engine_host_cli --payload-stdin toolbox-state-archive-v1",
    ],
    "toolbox-gc": [
        "python -m hosting.engine_host_cli toolbox-gc",
    ],
    "toolbox-template-list": [
        "python -m hosting.engine_host_cli toolbox-template-list",
    ],
    "toolbox-template-describe": [
        "'{\"template_id\":\"core\"}' | python -m hosting.engine_host_cli --payload-stdin toolbox-template-describe",
    ],
    "toolbox-template-publish": [
        "Get-Content template-publish.json | python -m hosting.engine_host_cli --payload-stdin toolbox-template-publish",
    ],
    "toolbox-template-deprecate": [
        "Get-Content template-lifecycle.json | python -m hosting.engine_host_cli --payload-stdin toolbox-template-deprecate",
    ],
    "toolbox-template-revoke": [
        "Get-Content template-lifecycle.json | python -m hosting.engine_host_cli --payload-stdin toolbox-template-revoke",
    ],
    "toolbox-template-prewarm": [
        "Get-Content template-prewarm.json | python -m hosting.engine_host_cli --payload-stdin toolbox-template-prewarm",
    ],
    "toolbox-references": [
        "python -m hosting.engine_host_cli toolbox-references",
    ],
    "toolbox-consistency": [
        "python -m hosting.engine_host_cli toolbox-consistency",
    ],
    "toolbox-review-snapshot": [
        "python -m hosting.engine_host_cli toolbox-review-snapshot",
        "'{\"toolbox_ids\":[\"toolbox-demo\"]}' | python -m hosting.engine_host_cli --payload-stdin toolbox-review-snapshot",
    ],
    "toolbox-repair": [
        "python -m hosting.engine_host_cli toolbox-repair",
        "'{\"toolbox_ids\":[\"toolbox-demo\"],\"only_inconsistent\":false}' | python -m hosting.engine_host_cli --payload-stdin toolbox-repair",
    ],
    "toolbox-reconcile": [
        "python -m hosting.engine_host_cli toolbox-reconcile",
        "'{\"toolbox_ids\":[\"toolbox-demo\"],\"only_inconsistent\":false}' | python -m hosting.engine_host_cli --payload-stdin toolbox-reconcile",
    ],
    "auth-upsert-key": [
        "@'{\"key_id\":\"admin-key\",\"key_secret\":\"change_me\",\"role\":\"admin\"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key",
        "@'{\"key_id\":\"worker-key\",\"key_secret\":\"change_me\",\"role\":\"worker_user\",\"allowed_engines\":[\"worker1\",\"worker2\"]}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key",
        "@'{\"key_id\":\"admin-pub\",\"auth_method\":\"public_key\",\"public_key\":\"ssh-ed25519 AAAA...\",\"role\":\"admin\"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key",
    ],
    "auth-issue-session": [
        "@'{\"key_id\":\"admin-key\",\"key_secret\":\"change_me\",\"scope\":\"control\"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-issue-session",
    ],
    "auth-status": [
        "python -m hosting.engine_host_cli auth-status",
    ],
    "hosting-setup-status": [
        "python -m hosting.engine_host_cli hosting-setup-status",
    ],
    "model-runtime-status": [
        "python -m hosting.engine_host_cli model-runtime-status",
    ],
    "hosting-secure-state-status": [
        "python -m hosting.engine_host_cli hosting-secure-state-status",
    ],
    "list-live-consumers": [
        "python -m hosting.engine_host_cli list-live-consumers --session-token <control_token>",
    ],
    "auth-begin-challenge": [
        "@'{\"key_id\":\"admin-pub\",\"scope\":\"control\"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-begin-challenge",
    ],
    "auth-complete-challenge": [
        "@'{\"challenge_id\":\"<id>\",\"signature_ssh\":\"-----BEGIN SSH SIGNATURE-----...\"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-complete-challenge",
    ],
    "auth-list-sessions": [
        "python -m hosting.engine_host_cli auth-list-sessions",
    ],
    "auth-list-issued-tokens": [
        "python -m hosting.engine_host_cli auth-list-issued-tokens",
    ],
    "auth-audit-list": [
        "python -m hosting.engine_host_cli auth-audit-list",
    ],
    "proxy-request": [
        "@'{\"engine_id\":\"worker1\",\"method\":\"GET\",\"path\":\"/health\",\"session_token\":\"<traffic_session_token>\"}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-request",
    ],
    "proxy-rpc-call": [
        "@'{\"engine_id\":\"worker1\",\"method\":\"rpc.describe\",\"params\":{},\"session_token\":\"<traffic_session_token>\"}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-call",
    ],
    "proxy-rpc-open": [
        "@'{\"engine_id\":\"worker1\",\"method\":\"run-inference\",\"params\":{\"messages_list\":[[{\"role\":\"user\",\"content\":\"hello\"}]],\"stream\":true},\"request_id\":\"req-1\"}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-open",
    ],
    "proxy-rpc-send": [
        "@'{\"engine_id\":\"worker1\",\"stream_id\":\"<stream_id>\",\"message\":{\"action\":\"cancel\",\"request_id\":\"req-1\"}}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-send",
    ],
    "proxy-rpc-recv": [
        "@'{\"engine_id\":\"worker1\",\"stream_id\":\"<stream_id>\",\"timeout_seconds\":2.0,\"max_items\":64}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-recv",
    ],
    "proxy-rpc-close": [
        "@'{\"engine_id\":\"worker1\",\"stream_id\":\"<stream_id>\"}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-close",
    ],
    "proxy-stream-open": [
        "@'{\"engine_id\":\"worker1\",\"tool\":\"run-inference\",\"arguments\":{\"messages_list\":[[{\"role\":\"user\",\"content\":\"hello\"}]],\"stream\":true}}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-stream-open",
    ],
    "proxy-stream-send": [
        "@'{\"engine_id\":\"worker1\",\"stream_id\":\"<stream_id>\",\"message\":{\"action\":\"cancel\"}}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-stream-send",
    ],
    "proxy-stream-recv": [
        "@'{\"engine_id\":\"worker1\",\"stream_id\":\"<stream_id>\",\"timeout_seconds\":2.0,\"max_items\":64}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-stream-recv",
    ],
    "proxy-stream-close": [
        "@'{\"engine_id\":\"worker1\",\"stream_id\":\"<stream_id>\"}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-stream-close",
    ],
    "host-metrics": [
        "python -m hosting.engine_host_cli host-metrics",
    ],
    "set-endpoint-mode-override": [
        "@'{\"mode\":\"exclusive\",\"session_token\":\"<control_token>\"}'@ | python -m hosting.engine_host_cli --payload-stdin set-endpoint-mode-override",
        "@'{\"mode\":\"default\",\"session_token\":\"<control_token>\"}'@ | python -m hosting.engine_host_cli --payload-stdin set-endpoint-mode-override",
    ],
    "get-endpoint-mode-effective": [
        "@'{\"session_token\":\"<control_token>\"}'@ | python -m hosting.engine_host_cli --payload-stdin get-endpoint-mode-effective",
    ],
    "get-lifecycle-policy-effective": [
        "@'{\"session_token\":\"<control_token>\"}'@ | python -m hosting.engine_host_cli --payload-stdin get-lifecycle-policy-effective",
    ],
    "reset-hosting-access": [
        "python -m hosting.engine_host_cli reset-hosting-access",
    ],
    "op-start": [
        "@'{\"command\":\"connect-from-config\",\"payload\":{\"config_path\":\"default\",\"engine_id\":\"worker_cfg\"}}'@ | python -m hosting.engine_host_cli --payload-stdin op-start",
    ],
    "op-status": [
        "@'{\"operation_id\":\"<operation_id>\"}'@ | python -m hosting.engine_host_cli --payload-stdin op-status",
    ],
    "op-cancel": [
        "@'{\"operation_id\":\"<operation_id>\",\"reason\":\"user_requested\"}'@ | python -m hosting.engine_host_cli --payload-stdin op-cancel",
    ],
}


def _examples_text(command: str = "") -> str:
    cmd = str(command or "").strip()
    if cmd:
        rows = EXAMPLES_BY_COMMAND.get(cmd)
        if not rows:
            keys = ", ".join(sorted(EXAMPLES_BY_COMMAND.keys()))
            return f"No examples for '{cmd}'. Available: {keys}"
        lines = [f"Examples for '{cmd}':"]
        lines.extend([f"  {x}" for x in rows])
        return "\n".join(lines)
    lines = ["Lifecycle and control examples:"]
    for key in sorted(EXAMPLES_BY_COMMAND.keys()):
        lines.append(f"- {key}")
        for row in EXAMPLES_BY_COMMAND[key]:
            lines.append(f"  {row}")
    return "\n".join(lines)


def _load_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(getattr(args, "payload_stdin", False)):
        raw = sys.stdin.read()
        if not str(raw or "").strip():
            return {}
        payload = json.loads(raw)
        return dict(payload or {}) if isinstance(payload, dict) else {}
    payload_raw = str(getattr(args, "payload_json", "") or "").strip()
    if payload_raw:
        payload = json.loads(payload_raw)
        return dict(payload or {}) if isinstance(payload, dict) else {}
    return {}


def _print_ok(result: Any) -> None:
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))


def _print_error(message: Any) -> None:
    if hasattr(message, "to_error_payload"):
        payload = dict(getattr(message, "to_error_payload")() or {})
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(payload.get("error") or "unknown_error"),
                    "error_code": str(payload.get("error_code") or ""),
                    "error_details": dict(payload.get("error_details") or {}),
                },
                ensure_ascii=False,
            )
        )
        return
    code = str(getattr(message, "code", "") or "").strip()
    details = dict(getattr(message, "details", {}) or {})
    payload = {"ok": False, "error": str(message or "unknown_error")}
    if code:
        payload["error_code"] = code
    if details:
        payload["error_details"] = details
    print(json.dumps(payload, ensure_ascii=False))


class RelayStartupError(RuntimeError):
    """Structured error emitted by SSH relay auto-start before daemon relay exists."""

    def __init__(self, message: str, *, code: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(str(message or code or "relay_startup_failed"))
        self.code = str(code or "relay_startup_failed").strip()
        self.details = dict(details or {})

    def to_error_payload(self) -> Dict[str, Any]:
        return {
            "error": str(self),
            "error_code": self.code,
            "error_details": dict(self.details or {}),
        }


def _error_payload(exc: BaseException) -> Dict[str, Any]:
    if hasattr(exc, "to_error_payload"):
        payload = dict(getattr(exc, "to_error_payload")() or {})
        return {
            "error": str(payload.get("error") or exc or "unknown_error"),
            "error_code": str(payload.get("error_code") or "").strip(),
            "error_details": dict(payload.get("error_details") or {}),
        }
    return {
        "error": str(exc or "unknown_error"),
        "error_code": str(getattr(exc, "code", "") or "").strip(),
        "error_details": dict(getattr(exc, "details", {}) or {}),
    }


def _request_seq(req: Any, default: int = -1) -> int:
    if not isinstance(req, dict):
        return int(default)
    try:
        return int(req.get("seq") or 0)
    except Exception:
        return int(default)


# ---------------------------------------------------------------------------
# Argument extraction helpers for pre-parse mode flags
# ---------------------------------------------------------------------------

def _extract_int_arg(argv: list, flag: str, default: int) -> int:
    """Extract a single-value int flag from raw argv without full argparse."""
    try:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return int(argv[idx + 1])
    except (ValueError, IndexError):
        pass
    return default


def _extract_path_arg(argv: list, flag: str, default: Optional[Path]) -> Optional[Path]:
    """Extract a single-value Path flag from raw argv without full argparse."""
    try:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return Path(argv[idx + 1])
    except (ValueError, IndexError):
        pass
    return default


def _extract_str_arg(argv: list, flag: str, default: Optional[str]) -> Optional[str]:
    """Extract a single-value string flag from raw argv without full argparse."""
    try:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    except (ValueError, IndexError):
        pass
    return default


def _setup_file_logging(log_file: Optional[str]) -> None:
    """Configure root logger to write to a file and redirect stdout/stderr."""
    if not log_file:
        return
    import logging
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)-20s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
    root_logger.addHandler(handler)

    class _StreamToLogger:
        def __init__(self, logger_instance, level):
            self.logger = logger_instance
            self.level = level

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                if line:  # Avoid logging empty lines
                    self.logger.log(self.level, line.rstrip())

        def flush(self):
            pass

    sys.stdout = _StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

    logging.info("=" * 80)
    logging.info("Logging started. stdout and stderr are now redirected to this file.")
    logging.info("Python: %s", sys.version)
    logging.info("Platform: %s", sys.platform)
    logging.info("CLI args: %s", sys.argv)
    logging.info("=" * 80)


# ---------------------------------------------------------------------------
# Relay mode: bridge stdin/stdout to local daemon control channel
# ---------------------------------------------------------------------------

def _run_relay(pid_file: Optional[Path] = None, port: int = 0) -> None:
    """Bridge stdin/stdout JSON lines through the local daemon control channel."""
    from .engine_host_connection import LocalSocketConnection

    conn = LocalSocketConnection(
        port=int(port or 0),
        pid_file=pid_file,
        timeout=10.0,
        max_reconnect_attempts=1,
    )
    try:
        for line in sys.stdin.buffer:
            stripped = line.strip()
            if not stripped:
                continue
            req: Dict[str, Any] = {}
            try:
                req = json.loads(stripped.decode("utf-8", errors="replace"))
                seq = _request_seq(req, 0)
                cmd = str(req.get("cmd") or "").strip()
                payload = dict(req.get("payload") or {})
                result = conn.invoke(cmd, payload)
                resp = {"seq": seq, "ok": True, "result": result}
            except Exception as exc:
                err = _error_payload(exc)
                resp = {
                    "seq": _request_seq(req, -1),
                    "ok": False,
                    "error": err["error"],
                }
                if err["error_code"]:
                    resp["error_code"] = err["error_code"]
                if err["error_details"]:
                    resp["error_details"] = err["error_details"]
            sys.stdout.buffer.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        conn.close()


def _run_relay_error_loop(exc: BaseException) -> None:
    """Return one structured relay error for each client request until SSH stdin closes."""
    err = _error_payload(exc)
    try:
        for line in sys.stdin.buffer:
            stripped = line.strip()
            if not stripped:
                continue
            seq = -1
            try:
                req = json.loads(stripped.decode("utf-8", errors="replace"))
                seq = _request_seq(req, -1)
            except Exception:
                pass
            resp: Dict[str, Any] = {
                "seq": seq,
                "ok": False,
                "error": err["error"],
            }
            if err["error_code"]:
                resp["error_code"] = err["error_code"]
            if err["error_details"]:
                resp["error_details"] = err["error_details"]
            sys.stdout.buffer.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()
    except (BrokenPipeError, EOFError, OSError):
        pass


def _relay_port(pid_file: Optional[Path], port: int) -> int:
    from .daemon import DEFAULT_DAEMON_PORT, DaemonPidFile

    if port:
        return int(port)
    pid_info = DaemonPidFile(pid_file)
    return int(pid_info.get_port() or DEFAULT_DAEMON_PORT)


def _relay_daemon_reachable(*, pid_file: Optional[Path], port: int) -> bool:
    from .engine_host_connection import LocalSocketConnection

    conn = LocalSocketConnection(
        port=int(port or 0),
        pid_file=pid_file,
        timeout=3.0,
        max_reconnect_attempts=1,
    )
    try:
        return bool(conn.is_alive())
    finally:
        conn.close()


def _validate_relay_autostart_policy(control_state_file: Optional[Path]) -> Dict[str, Any]:
    svc = EngineHostService(control_state_file=control_state_file)
    cfg = dict(svc.get_control_config() or {})
    access_profile = dict(cfg.get("access_profile") or {})
    connectivity_mode = str(access_profile.get("connectivity_mode") or "local_only").strip().lower()
    lifecycle_profile = str(cfg.get("lifecycle_profile") or "detached_user_process").strip().lower()
    require_auth = bool(cfg.get("require_auth", False))
    keys_count = int(cfg.get("keys_count") or 0)
    details = {
        "connectivity_mode": connectivity_mode,
        "lifecycle_profile": lifecycle_profile,
        "require_auth": require_auth,
        "keys_count": keys_count,
        "control_state_file": str(Path(control_state_file).expanduser()) if control_state_file else None,
    }
    if connectivity_mode not in {"ssh_tunnel_only", "truly_remote"}:
        raise RelayStartupError(
            "SSH relay auto-start is disabled because hosting connectivity is not remote-enabled",
            code="relay_autostart_requires_remote_connectivity",
            details=details,
        )
    if not require_auth:
        raise RelayStartupError(
            "SSH relay auto-start is disabled because hosting require_auth is false",
            code="relay_autostart_requires_auth",
            details=details,
        )
    if keys_count <= 0:
        raise RelayStartupError(
            "SSH relay auto-start is disabled because no hosting auth keys are registered",
            code="relay_autostart_requires_registered_keys",
            details=details,
        )
    if lifecycle_profile != "detached_user_process":
        raise RelayStartupError(
            "SSH relay auto-start is only supported for detached_user_process lifecycle",
            code="relay_autostart_requires_detached_user_process",
            details=details,
        )
    return details


def _ensure_relay_daemon_ready(
    *,
    pid_file: Optional[Path],
    port: int,
    control_state_file: Optional[Path],
    engines_state_file: Optional[Path] = None,
    wait_ready_seconds: float = 8.0,
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    from .daemon import start_daemon_background

    resolved_port = _relay_port(pid_file, int(port or 0))
    if _relay_daemon_reachable(pid_file=pid_file, port=resolved_port):
        return {"status": "already_running", "port": resolved_port}
    policy = _validate_relay_autostart_policy(control_state_file)
    try:
        started = dict(
            start_daemon_background(
                port=resolved_port,
                pid_file=pid_file,
                log_file=log_file,
                engines_state_file=engines_state_file,
                control_state_file=control_state_file,
                wait_ready_seconds=float(wait_ready_seconds or 8.0),
            )
            or {}
        )
        started["status"] = "started"
        started.setdefault("policy", policy)
        return started
    except Exception as exc:
        raise RelayStartupError(
            "SSH relay auto-start failed to start the detached hosting daemon",
            code="relay_autostart_start_failed",
            details={**policy, "cause": str(exc)},
        ) from exc


def _run_relay_wrapper(
    *,
    pid_file: Optional[Path],
    port: int,
    control_state_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    wait_ready_seconds: float = 8.0,
    log_file: Optional[Path] = None,
) -> None:
    try:
        ready = _ensure_relay_daemon_ready(
            pid_file=pid_file,
            port=port,
            control_state_file=control_state_file,
            engines_state_file=engines_state_file,
            wait_ready_seconds=wait_ready_seconds,
            log_file=log_file,
        )
    except Exception as exc:
        _run_relay_error_loop(exc)
        return
    ready_port = int(ready.get("port") or port or _relay_port(pid_file, 0))
    _run_relay(pid_file=pid_file, port=ready_port)


# ---------------------------------------------------------------------------
# Short-lived client: send one command to running daemon, print response
# ---------------------------------------------------------------------------

def _try_daemon_invoke(
    cmd: str,
    payload: Dict[str, Any],
    *,
    pid_file: Optional[Path] = None,
) -> bool:
    """
    Try to send a command to the local running daemon.
    Prints JSON response and returns True on success.
    Returns False if daemon not found or not reachable.
    """
    from .daemon import DaemonPidFile
    from .engine_host_connection import LocalSocketConnection

    pid_info = DaemonPidFile(pid_file)
    if not pid_info.is_alive():
        return False
    port = pid_info.get_port()
    if not port:
        return False
    try:
        conn = LocalSocketConnection(
            port=port,
            pid_file=pid_info.path,
            timeout=10.0,
            max_reconnect_attempts=1,
        )
        result = conn.invoke(cmd, payload)
        conn.close()
        _print_ok(result)
        return True
    except Exception:
        return False


def _channel_settings_from_args(args: argparse.Namespace, *, auto_bootstrap: bool = False) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "engine_host_daemon_auto_bootstrap": bool(auto_bootstrap),
        "engine_host_daemon_pid_file": str(getattr(args, "pid_file", "") or "") or None,
        "engine_host_state_file": str(getattr(args, "engines_state_file", "") or "") or None,
        "engine_host_control_state_file": str(getattr(args, "control_state_file", "") or "") or None,
    }
    for attr in (
        "engine_host_ssh_target",
        "control_endpoint",
        "control_ssh_key",
        "control_ssh_fingerprint",
        "ssh_known_hosts_line",
        "engine_host_remote_cmd",
        "engine_host_client_profile",
        "engine_host_client_realm",
        "engine_host_client_realm_root",
        "engine_host_client_secret_password",
        "engine_host_timeout_seconds",
        "engine_host_daemon_port",
        "engine_host_key_id",
        "engine_host_key_secret",
        "engine_host_session_token",
        "engine_host_session_scope",
        "engine_host_session_ttl_seconds",
        "engine_host_bind_session_to_ssh",
    ):
        value = getattr(args, attr, None)
        if value not in (None, ""):
            settings[attr] = value
    return settings


def _has_explicit_channel_target(args: argparse.Namespace) -> bool:
    return any(
        str(getattr(args, attr, "") or "").strip()
        for attr in (
            "engine_host_ssh_target",
            "control_endpoint",
            "engine_host_client_profile",
            "engine_host_client_realm_root",
        )
    )


def _payload_with_cli_selectors(args: argparse.Namespace, payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    for attr in ("engine_id", "resource_kind", "resource_id"):
        value = str(getattr(args, attr, "") or "").strip()
        if value:
            out.setdefault(attr, value)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Engine host control CLI")
    p.add_argument("--engines-state-file", type=Path, default=None)
    p.add_argument("--control-state-file", type=Path, default=None)
    p.add_argument("--pid-file", type=Path, default=None, help="Daemon PID file path (for daemon client mode)")
    p.add_argument(
        "--ssh-target",
        "--remote-target",
        "--engine-host-ssh-target",
        dest="engine_host_ssh_target",
        default="",
        help="Remote host target for SSH relay control, for example user@example-host",
    )
    p.add_argument(
        "--control-endpoint",
        default="",
        help="Control endpoint override; ssh:// or user@host values imply SSH relay mode",
    )
    p.add_argument(
        "--control-ssh-key",
        default="",
        help="Private key file used for SSH relay control",
    )
    p.add_argument(
        "--control-ssh-fingerprint",
        default="",
        help="Expected SSH control key fingerprint used when binding issued sessions",
    )
    p.add_argument(
        "--ssh-known-hosts-line",
        default="",
        help="Pinned SSH known_hosts line for strict remote host verification",
    )
    p.add_argument(
        "--engine-host-remote-cmd",
        default="",
        help="Remote base command for engine_host_cli; relay mode appends/uses --relay-wrapper as needed",
    )
    p.add_argument("--client-profile", dest="engine_host_client_profile", default="", help="Client-realm profile name")
    p.add_argument("--client-realm", dest="engine_host_client_realm", default="", help="Client-realm name")
    p.add_argument("--client-realm-root", dest="engine_host_client_realm_root", default="", help="Client-realm root")
    p.add_argument(
        "--client-secret-password",
        dest="engine_host_client_secret_password",
        default="",
        help="Password for client-realm secret materialization when needed",
    )
    p.add_argument("--session-token", dest="engine_host_session_token", default="", help="Existing daemon session token")
    p.add_argument("--timeout-seconds", dest="engine_host_timeout_seconds", type=float, default=0.0, help="Control command timeout")
    p.add_argument("--payload-stdin", action="store_true")
    p.add_argument("--payload-json", type=str, default="")
    p.add_argument(
        "--examples",
        nargs="?",
        const="",
        default=None,
        metavar="COMMAND",
        help="Print lifecycle/control usage examples (optionally for one command) and exit",
    )
    p.add_argument(
        "--color-scheme",
        default="dark",
        choices=["dark", "light"],
        help="Terminal color scheme for interactive output",
    )
    p.add_argument("--interactive", action="store_true", help="Launch interactive control menu")
    sp = p.add_subparsers(dest="command", required=False)

    for name in [
        "discover-running",
        "spawn",
        "workflow-js-environment-spec",
        "workflow-js-ensure",
        "workflow-js-execute",
        "workflow-js-action-describe",
        "workflow-js-action-execute",
        "workflow-js-instance-create",
        "workflow-js-instance-execute",
        "workflow-js-instance-close",
        "workflow-js-instance-list",
        "workflow-js-resources",
        "workflow-js-set-capacity",
        "workflow-js-stream-open",
        "workflow-js-event-subscribe",
        "workflow-js-stream-send",
        "workflow-js-stream-close",
        "workflow-python-environment-spec",
        "workflow-python-prepare-environment",
        "workflow-python-lock-environment",
        "workflow-python-verify-environment",
        "workflow-python-install-environment",
        "workflow-python-verify-install-receipt",
        "sandbox-state-snapshot",
        "sandbox-state-restore",
        "workflow-artifact-recovery-inspect",
        "workflow-artifact-recovery-claim",
        "workflow-artifact-recovery-cleanup",
        "workflow-python-ensure",
        "workflow-python-execute",
        "workflow-python-action-describe",
        "workflow-python-action-execute",
        "workflow-python-instance-create",
        "workflow-python-instance-execute",
        "workflow-python-instance-close",
        "workflow-python-instance-list",
        "workflow-python-resources",
        "workflow-python-set-capacity",
        "workflow-python-stream-open",
        "workflow-python-event-subscribe",
        "workflow-python-stream-send",
        "workflow-python-stream-close",
        "get-registration",
        "shutdown",
        "ensure-running",
        "unload-model",
        "remove-registration",
        "claim-engine",
        "claim-endpoint",
        "claim-status",
        "issue-token",
        "validate-token",
        "claim-resource",
        "resource-claim-status",
        "issue-resource-token",
        "validate-resource-token",
        "list-configs",
        "create-config",
        "models-from-config",
        "connect-from-config",
        "inspect-capabilities",
        "logs-tail",
        "logs-follow",
        "sandbox-fs-list",
        "sandbox-fs-read-text",
        "sandbox-fs-write-text",
        "sandbox-fs-mkdir",
        "sandbox-fs-stat",
        "sandbox-http-fetch",
        "toolbox-describe",
        "toolbox-gate",
        "toolbox-execute",
        "hosted-operation-status",
        "hosted-operation-result",
        "hosted-operation-cancel",
        "hosting-receipt-ledger-cutover",
        "toolbox-state-archive-v1",
        "toolbox-gc",
        "toolbox-template-list",
        "toolbox-template-describe",
        "toolbox-template-publish",
        "toolbox-template-deprecate",
        "toolbox-template-revoke",
        "toolbox-template-prewarm",
        "toolbox-references",
        "toolbox-consistency",
        "toolbox-review-snapshot",
        "toolbox-repair",
        "toolbox-reconcile",
        "get-control-config",
        "set-control-config",
        "auth-status",
        "daemon-status",
        "hosting-setup-status",
        "model-runtime-status",
        "hosting-secure-state-status",
        "auth-list-keys",
        "auth-list-sessions",
        "list-live-consumers",
        "auth-list-issued-tokens",
        "auth-audit-list",
        "auth-validate-session",
        "auth-renew-session",
        "auth-upsert-key",
        "auth-revoke-key",
        "auth-issue-session",
        "auth-begin-challenge",
        "auth-complete-challenge",
        "auth-revoke-session",
        "host-capability-session-register",
        "host-capability-session-list",
        "host-capability-session-close",
        "host-capability-audit-list",
        "proxy-request",
        "proxy-rpc-call",
        "proxy-rpc-open",
        "proxy-rpc-send",
        "proxy-rpc-recv",
        "proxy-rpc-close",
        "proxy-stream-open",
        "proxy-stream-send",
        "proxy-stream-recv",
        "proxy-stream-close",
        "host-metrics",
        "set-endpoint-mode-override",
        "get-endpoint-mode-effective",
        "get-lifecycle-policy-effective",
        "reset-hosting-access",
        "force-stop-daemon",
        "force-restart-daemon",
        "op-start",
        "op-status",
        "op-cancel",
    ]:
        cp = sp.add_parser(name)
        cp.add_argument("--engine-id", type=str, default="")
        cp.add_argument("--resource-kind", type=str, default="")
        cp.add_argument("--resource-id", type=str, default="")
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    argv = list(argv) if argv is not None else list(sys.argv[1:])

    # ------------------------------------------------------------------
    # Mode 1: --daemon  →  start long-lived daemon server
    # ------------------------------------------------------------------
    if "--daemon" in argv:
        from .daemon import (
            DEFAULT_DAEMON_PORT,
            run_daemon_foreground,
            start_daemon_background,
        )

        port = _extract_int_arg(argv, "--port", DEFAULT_DAEMON_PORT)
        runtime_profile = _extract_str_arg(argv, "--runtime-profile", "foreground_terminal_bound")
        pid_file = _extract_path_arg(argv, "--pid-file", None)
        engines_state = _extract_path_arg(argv, "--engines-state-file", None)
        control_state = _extract_path_arg(argv, "--control-state-file", None)
        background = "--background" in argv
        log_file_str = _extract_str_arg(argv, "--log-file", None)
        from .daemon.diagnostics import daemon_report_path_for_control_state, install_daemon_crash_report

        crash_report_path = install_daemon_crash_report(daemon_report_path_for_control_state(control_state))
        _setup_file_logging(log_file_str)
        import logging

        logging.info("Daemon crash report path: %s", crash_report_path)

        if background:
            try:
                result = start_daemon_background(
                    port=port,
                    pid_file=pid_file,
                    log_file=Path(log_file_str) if log_file_str else None,
                    engines_state_file=engines_state,
                    control_state_file=control_state,
                )
                _print_ok(result)
                return 0
            except Exception as exc:
                _print_error(str(exc))
                return 1
        else:
            run_daemon_foreground(
                port=port,
                pid_file=pid_file,
                engines_state_file=engines_state,
                control_state_file=control_state,
                runtime_profile=str(runtime_profile or "foreground_terminal_bound"),
            )
            return 0

    # ------------------------------------------------------------------
    # Mode 1b: --daemon-http  →  start HTTP ingress daemon
    # ------------------------------------------------------------------
    if "--daemon-http" in argv:
        from .daemon import (
            DEFAULT_HTTP_INGRESS_PORT,
            run_http_ingress_foreground,
            start_http_ingress_background,
        )

        port = _extract_int_arg(argv, "--http-port", DEFAULT_HTTP_INGRESS_PORT)
        pid_file = _extract_path_arg(argv, "--pid-file", None)
        engines_state = _extract_path_arg(argv, "--engines-state-file", None)
        control_state = _extract_path_arg(argv, "--control-state-file", None)
        background = "--background" in argv
        log_file_str = _extract_str_arg(argv, "--log-file", None)
        from .daemon.diagnostics import daemon_report_path_for_control_state, install_daemon_crash_report

        crash_report_path = install_daemon_crash_report(daemon_report_path_for_control_state(control_state))
        _setup_file_logging(log_file_str)
        import logging

        logging.info("Daemon crash report path: %s", crash_report_path)

        if background:
            try:
                result = start_http_ingress_background(
                    port=port,
                    pid_file=pid_file,
                    log_file=Path(log_file_str) if log_file_str else None,
                    engines_state_file=engines_state,
                    control_state_file=control_state,
                )
                _print_ok(result)
                return 0
            except Exception as exc:
                _print_error(str(exc))
                return 1
        else:
            run_http_ingress_foreground(
                port=port,
                pid_file=pid_file,
                engines_state_file=engines_state,
                control_state_file=control_state,
            )
            return 0

    # ------------------------------------------------------------------
    # Mode 2: --relay-wrapper  →  SSH forced-command auto-start + relay
    # ------------------------------------------------------------------
    if "--relay-wrapper" in argv:
        port = _extract_int_arg(argv, "--port", 0)
        pid_file = _extract_path_arg(argv, "--pid-file", None)
        engines_state = _extract_path_arg(argv, "--engines-state-file", None)
        control_state = _extract_path_arg(argv, "--control-state-file", None)
        log_file = _extract_path_arg(argv, "--log-file", None)
        wait_raw = _extract_str_arg(argv, "--wait-ready-seconds", "8.0")
        try:
            wait_ready_seconds = float(wait_raw or "8.0")
        except ValueError:
            wait_ready_seconds = 8.0
        try:
            _run_relay_wrapper(
                pid_file=pid_file,
                port=port,
                engines_state_file=engines_state,
                control_state_file=control_state,
                wait_ready_seconds=wait_ready_seconds,
                log_file=log_file,
            )
            return 0
        except Exception as exc:
            _print_error(exc)
            return 1

    # ------------------------------------------------------------------
    # Mode 2b: --relay  →  bridge stdin/stdout to local daemon control channel
    # ------------------------------------------------------------------
    if "--relay" in argv:
        from .daemon import DEFAULT_DAEMON_PORT, DaemonPidFile

        port = _extract_int_arg(argv, "--port", 0)
        pid_file = _extract_path_arg(argv, "--pid-file", None)
        if not port:
            pid_info = DaemonPidFile(pid_file)
            port = pid_info.get_port() or DEFAULT_DAEMON_PORT
        try:
            _run_relay(pid_file=pid_file, port=port)
            return 0
        except Exception as exc:
            _print_error(str(exc))
            return 1

    # ------------------------------------------------------------------
    # Mode 2c: --hosting-config  →  run setup/reconfiguration wizard/tool
    # ------------------------------------------------------------------
    if "--hosting-config" in argv:
        from .hosting_config_cli import main as hosting_config_main

        forwarded = [a for a in argv if a != "--hosting-config"]
        return int(hosting_config_main(forwarded))

    # ------------------------------------------------------------------
    # Mode 3: subcommand  →  parse normally; try daemon first, then direct
    # ------------------------------------------------------------------
    parser = _build_parser()
    if "--examples" in argv:
        ex_idx = argv.index("--examples")
        ex_cmd = ""
        if ex_idx + 1 < len(argv) and not str(argv[ex_idx + 1]).startswith("-"):
            ex_cmd = str(argv[ex_idx + 1])
        print(_examples_text(ex_cmd))
        return 0
    args = parser.parse_args(argv)

    if bool(getattr(args, "interactive", False)):
        from .engine_host_cli_interactive import run_interactive_mode
        return run_interactive_mode(args)

    if not args.command:
        parser.print_help()
        return 1

    payload = _load_payload(args)
    cmd_name = str(args.command or "").strip()
    effective_payload = _payload_with_cli_selectors(args, payload)

    # Local-only recovery helpers. Intentionally bypass daemon RPC/auth surfaces.
    if cmd_name in {"reset-hosting-access", "force-stop-daemon", "force-restart-daemon"}:
        if _has_explicit_channel_target(args):
            print(json.dumps({"ok": False, "error": f"{cmd_name} is local-only"}, ensure_ascii=False))
            return 2
        from .engine_host_channel import EngineHostControlChannel

        ch = EngineHostControlChannel(
            _channel_settings_from_args(args, auto_bootstrap=False)
        )
        if cmd_name == "reset-hosting-access":
            _print_ok(ch.reset_hosting_access())
        elif cmd_name == "force-stop-daemon":
            _print_ok(ch.force_stop_daemon(stop_workers=True))
        else:
            _print_ok(ch.force_restart_daemon())
        return 0

    if cmd_name == "toolbox-state-archive-v1" and _has_explicit_channel_target(args):
        print(json.dumps({"ok": False, "error": "toolbox_state_archive_v1_local_only"}, ensure_ascii=False))
        return 2

    if _has_explicit_channel_target(args):
        from .engine_host_channel import EngineHostControlChannel

        ch = EngineHostControlChannel(_channel_settings_from_args(args, auto_bootstrap=False))
        try:
            _print_ok(ch.invoke_control_command(cmd_name, effective_payload))
            return 0
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2

    # Try sending the command to the running daemon first
    pid_file_arg = getattr(args, "pid_file", None)
    if cmd_name and cmd_name != "toolbox-state-archive-v1" and _try_daemon_invoke(cmd_name, effective_payload, pid_file=pid_file_arg):
        return 0

    if cmd_name == "toolbox-template-prewarm":
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "template_prewarm_requires_daemon",
                    "error_code": "template_prewarm_requires_daemon",
                },
                ensure_ascii=False,
            )
        )
        return 2

    # Fallback: direct EngineHostService call (original behavior)
    svc = EngineHostService(
        engines_state_file=args.engines_state_file,
        control_state_file=args.control_state_file,
    )
    try:
        cmd = str(args.command or "").strip()
        payload = effective_payload
        svc.authorize_command(cmd, payload)
        if cmd == "discover-running":
            _print_ok(svc.discover_running())
            return 0
        if cmd == "spawn":
            _print_ok(
                svc.spawn(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    command=list(payload.get("command") or []),
                    cwd=payload.get("cwd"),
                    env=dict(payload.get("env") or {}),
                    worker_profile_class=payload.get("worker_profile_class"),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}),
                    executor_kind=payload.get("executor_kind"),
                    bundle=dict(payload.get("bundle") or {}),
                    environment=dict(payload.get("environment") or {}),
                    tool_access=dict(payload.get("tool_access") or {}),
                    capabilities=dict(payload.get("capabilities") or {}),
                )
            )
            return 0
        if cmd == "workflow-js-environment-spec":
            _print_ok(
                svc.workflow_js_environment_spec(
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-js-ensure":
            _print_ok(
                svc.ensure_workflow_js(
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
                )
            )
            return 0
        if cmd == "workflow-js-resources":
            _print_ok(
                svc.workflow_js_resources(
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-js-execute":
            _print_ok(
                svc.execute_workflow_js(
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-js-action-describe":
            _print_ok(
                svc.workflow_js_action_describe(
                    request=dict(payload.get("request") or {}),
                    include_hidden=bool(payload.get("include_hidden", False)),
                    dynamic=bool(payload.get("dynamic", False)),
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                    instance_id=str(payload.get("instance_id") or "").strip() or None,
                )
            )
            return 0
        if cmd == "workflow-js-action-execute":
            _print_ok(
                svc.execute_workflow_js_action(
                    action_name=str(payload.get("action_name") or ""),
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-js-instance-create":
            _print_ok(
                svc.workflow_js_instance_create(
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                    instance_id=str(payload.get("instance_id") or "").strip() or None,
                    replace=bool(payload.get("replace", False)),
                )
            )
            return 0
        if cmd == "workflow-js-instance-execute":
            _print_ok(
                svc.workflow_js_instance_execute(
                    instance_id=str(payload.get("instance_id") or ""),
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-js-instance-close":
            _print_ok(
                svc.workflow_js_instance_close(
                    instance_id=str(payload.get("instance_id") or ""),
                    reason=str(payload.get("reason") or "client_requested"),
                )
            )
            return 0
        if cmd == "workflow-js-instance-list":
            _print_ok(svc.workflow_js_instance_list())
            return 0
        if cmd == "workflow-js-set-capacity":
            _print_ok(
                svc.set_workflow_js_capacity(
                    profile=str(payload.get("profile") or "node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    capacity=int(payload.get("capacity") or 1),
                )
            )
            return 0
        if cmd == "workflow-js-stream-open":
            _print_ok(
                svc.workflow_js_stream_open(
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    node=dict(payload.get("node") or {}),
                    javascript=dict(payload.get("javascript") or {}),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                    capacity=int(payload.get("capacity") or 1),
                )
            )
            return 0
        if cmd == "workflow-js-event-subscribe":
            _print_ok(
                svc.workflow_js_event_subscribe(
                    stream_id=str(payload.get("stream_id") or ""),
                    max_items=int(payload.get("max_items") or 64),
                )
            )
            return 0
        if cmd == "workflow-js-stream-send":
            _print_ok(
                svc.workflow_js_stream_send(
                    stream_id=str(payload.get("stream_id") or ""),
                    message=dict(payload.get("message") or {}),
                )
            )
            return 0
        if cmd == "workflow-js-stream-close":
            _print_ok(svc.workflow_js_stream_close(stream_id=str(payload.get("stream_id") or "")))
            return 0
        if cmd == "workflow-python-environment-spec":
            _print_ok(
                svc.workflow_python_environment_spec(
                    profile=str(payload.get("profile") or "helper"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                    python=dict(payload.get("python") or {}),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-python-prepare-environment":
            _print_ok(
                svc.workflow_python_prepare_environment(
                    environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                    python=dict(payload.get("python") or {}),
                    package_id=str(payload.get("package_id") or "").strip() or None,
                    workflow_id=str(payload.get("workflow_id") or "").strip() or None,
                )
            )
            return 0
        if cmd == "workflow-python-lock-environment":
            _print_ok(svc.workflow_python_lock_environment(environment=dict(payload.get("environment") or {})))
            return 0
        if cmd == "workflow-python-verify-environment":
            _print_ok(svc.workflow_python_verify_environment(environment=dict(payload.get("environment") or {})))
            return 0
        if cmd == "workflow-python-install-environment":
            _print_ok(
                svc.workflow_python_install_environment(
                    environment=dict(payload.get("environment") or {}),
                    allow_execution=bool(payload.get("allow_execution", False)),
                )
            )
            return 0
        if cmd == "workflow-python-verify-install-receipt":
            _print_ok(svc.workflow_python_verify_install_receipt(environment=dict(payload.get("environment") or {})))
            return 0
        if cmd == "sandbox-state-snapshot":
            _print_ok(
                svc.sandbox_state_snapshot(
                    scope=str(payload.get("scope") or ""),
                    workflow_id=str(payload.get("workflow_id") or ""),
                    instance_id=str(payload.get("instance_id") or ""),
                    request_id=str(payload.get("request_id") or ""),
                    prefix=str(payload.get("prefix") or ""),
                )
            )
            return 0
        if cmd == "sandbox-state-restore":
            _print_ok(
                svc.sandbox_state_restore(
                    snapshot=dict(payload.get("snapshot") or {}),
                    scope=str(payload.get("scope") or ""),
                    workflow_id=str(payload.get("workflow_id") or ""),
                    instance_id=str(payload.get("instance_id") or ""),
                    request_id=str(payload.get("request_id") or ""),
                    mode=str(payload.get("mode") or "merge"),
                )
            )
            return 0
        if cmd == "workflow-artifact-recovery-inspect":
            _print_ok(
                svc.workflow_artifact_recovery_inspect(
                    request_id=str(payload.get("request_id") or ""),
                    names=list(payload.get("names") or []),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-artifact-recovery-claim":
            _print_ok(
                svc.workflow_artifact_recovery_claim(
                    request_id=str(payload.get("request_id") or ""),
                    names=list(payload.get("names") or []),
                    target_id=str(payload.get("target_id") or ""),
                    instance_id=str(payload.get("instance_id") or ""),
                    patch_absolute_paths=bool(payload.get("patch_absolute_paths", False)),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-artifact-recovery-cleanup":
            _print_ok(
                svc.workflow_artifact_recovery_cleanup(
                    request_id=str(payload.get("request_id") or ""),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-python-ensure":
            _print_ok(
                svc.ensure_workflow_python(
                    profile=str(payload.get("profile") or "helper"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    python=dict(payload.get("python") or {}),
                    python_executable=payload.get("python_executable"),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
                )
            )
            return 0
        if cmd == "workflow-python-execute":
            _print_ok(
                svc.execute_workflow_python(
                    profile=str(payload.get("profile") or "helper"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-python-action-describe":
            _print_ok(
                svc.workflow_python_action_describe(
                    request=dict(payload.get("request") or {}),
                    include_hidden=bool(payload.get("include_hidden", False)),
                    dynamic=bool(payload.get("dynamic", False)),
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                    instance_id=str(payload.get("instance_id") or "").strip() or None,
                )
            )
            return 0
        if cmd == "workflow-python-action-execute":
            _print_ok(
                svc.execute_workflow_python_action(
                    action_name=str(payload.get("action_name") or ""),
                    profile=str(payload.get("profile") or "helper"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-python-instance-create":
            _print_ok(
                svc.workflow_python_instance_create(
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                    instance_id=str(payload.get("instance_id") or "").strip() or None,
                    replace=bool(payload.get("replace", False)),
                )
            )
            return 0
        if cmd == "workflow-python-instance-execute":
            _print_ok(
                svc.workflow_python_instance_execute(
                    instance_id=str(payload.get("instance_id") or ""),
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    capacity=int(payload.get("capacity") or 1),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-python-instance-close":
            _print_ok(
                svc.workflow_python_instance_close(
                    instance_id=str(payload.get("instance_id") or ""),
                    reason=str(payload.get("reason") or "client_requested"),
                )
            )
            return 0
        if cmd == "workflow-python-instance-list":
            _print_ok(svc.workflow_python_instance_list())
            return 0
        if cmd == "workflow-python-resources":
            _print_ok(
                svc.workflow_python_resources(
                    profile=str(payload.get("profile") or "helper"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    python=dict(payload.get("python") or {}),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                )
            )
            return 0
        if cmd == "workflow-python-set-capacity":
            _print_ok(
                svc.set_workflow_python_capacity(
                    profile=str(payload.get("profile") or "helper"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    capacity=int(payload.get("capacity") or 1),
                )
            )
            return 0
        if cmd == "workflow-python-stream-open":
            _print_ok(
                svc.workflow_python_stream_open(
                    profile=str(payload.get("profile") or "node"),
                    environment_name=str(payload.get("environment_name") or "workflow-python-node"),
                    environment_key=str(payload.get("environment_key") or "").strip() or None,
                    engine_id=str(payload.get("engine_id") or args.engine_id or "").strip() or None,
                    request=dict(payload.get("request") or {}),
                    python=dict(payload.get("python") or {}),
                    sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                    capacity=int(payload.get("capacity") or 1),
                )
            )
            return 0
        if cmd == "workflow-python-event-subscribe":
            _print_ok(
                svc.workflow_python_event_subscribe(
                    stream_id=str(payload.get("stream_id") or ""),
                    max_items=int(payload.get("max_items") or 64),
                )
            )
            return 0
        if cmd == "workflow-python-stream-send":
            _print_ok(
                svc.workflow_python_stream_send(
                    stream_id=str(payload.get("stream_id") or ""),
                    message=dict(payload.get("message") or {}),
                )
            )
            return 0
        if cmd == "workflow-python-stream-close":
            _print_ok(svc.workflow_python_stream_close(stream_id=str(payload.get("stream_id") or "")))
            return 0
        if cmd == "get-registration":
            _print_ok(svc.get_registration(str(payload.get("engine_id") or args.engine_id)))
            return 0
        if cmd == "shutdown":
            _print_ok(
                svc.shutdown(
                    str(payload.get("engine_id") or args.engine_id),
                    timeout_seconds=float(payload.get("timeout_seconds") or 8.0),
                )
            )
            return 0
        if cmd == "ensure-running":
            _print_ok(svc.ensure_running(str(payload.get("engine_id") or args.engine_id)))
            return 0
        if cmd == "unload-model":
            _print_ok(
                svc.unload_model(
                    str(payload.get("engine_id") or args.engine_id),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                    shutdown_all=bool(payload.get("shutdown_all", False)),
                )
            )
            return 0
        if cmd == "remove-registration":
            _print_ok(svc.remove_registration(str(payload.get("engine_id") or args.engine_id)))
            return 0
        if cmd == "claim-engine":
            _print_ok(
                svc.claim_engine(
                    str(payload.get("engine_id") or args.engine_id),
                    backend_id=payload.get("backend_id"),
                    exclusive=payload.get("exclusive"),
                    force_override=bool(payload.get("force_override", False)),
                    force_override_reason=payload.get("force_override_reason"),
                    force_override_emergency=bool(payload.get("force_override_emergency", False)),
                    actor_id=payload.get("_claim_actor_id"),
                    peer_host=payload.get("_daemon_peer_host"),
                )
            )
            return 0
        if cmd == "claim-endpoint":
            _print_ok(
                svc.claim_endpoint(
                    backend_id=payload.get("backend_id"),
                    exclusive=payload.get("exclusive"),
                    force_override=bool(payload.get("force_override", False)),
                    force_override_reason=payload.get("force_override_reason"),
                    force_override_emergency=bool(payload.get("force_override_emergency", False)),
                    actor_id=payload.get("_claim_actor_id"),
                    peer_host=payload.get("_daemon_peer_host"),
                )
            )
            return 0
        if cmd == "claim-status":
            _print_ok(svc.get_claim_status(str(payload.get("engine_id") or args.engine_id)))
            return 0
        if cmd == "issue-token":
            _print_ok(
                svc.issue_token(
                    str(payload.get("engine_id") or args.engine_id),
                    backend_id=payload.get("backend_id"),
                )
            )
            return 0
        if cmd == "validate-token":
            _print_ok(
                svc.validate_token(
                    str(payload.get("engine_id") or args.engine_id),
                    str(payload.get("token") or ""),
                )
            )
            return 0
        if cmd == "claim-resource":
            _print_ok(
                svc.claim_resource(
                    str(payload.get("resource_kind") or args.resource_kind),
                    str(payload.get("resource_id") or args.resource_id),
                    backend_id=payload.get("backend_id"),
                    exclusive=payload.get("exclusive"),
                    force_override=bool(payload.get("force_override", False)),
                    force_override_reason=payload.get("force_override_reason"),
                    force_override_emergency=bool(payload.get("force_override_emergency", False)),
                    actor_id=payload.get("_claim_actor_id"),
                    peer_host=payload.get("_daemon_peer_host"),
                )
            )
            return 0
        if cmd == "resource-claim-status":
            _print_ok(
                svc.get_resource_claim_status(
                    str(payload.get("resource_kind") or args.resource_kind),
                    str(payload.get("resource_id") or args.resource_id),
                )
            )
            return 0
        if cmd == "issue-resource-token":
            _print_ok(
                svc.issue_resource_token(
                    str(payload.get("resource_kind") or args.resource_kind),
                    str(payload.get("resource_id") or args.resource_id),
                    backend_id=payload.get("backend_id"),
                )
            )
            return 0
        if cmd == "validate-resource-token":
            _print_ok(
                svc.validate_resource_token(
                    str(payload.get("resource_kind") or args.resource_kind),
                    str(payload.get("resource_id") or args.resource_id),
                    str(payload.get("token") or ""),
                )
            )
            return 0
        if cmd == "list-configs":
            _print_ok(svc.list_engine_configs())
            return 0
        if cmd == "create-config":
            _print_ok(
                svc.create_engine_config(
                    name=str(payload.get("name") or "engine_config"),
                    config=dict(payload.get("config") or {}),
                    overwrite=bool(payload.get("overwrite", False)),
                )
            )
            return 0
        if cmd == "models-from-config":
            _print_ok(svc.models_from_config(str(payload.get("config_path") or "default")))
            return 0
        if cmd == "connect-from-config":
            _print_ok(
                svc.connect_from_config(
                    config_path=str(payload.get("config_path") or "default"),
                    engine_id=payload.get("engine_id"),
                    model_path=payload.get("model_path"),
                    force_new_worker=bool(payload.get("force_new_worker", False)),
                    launch_policy=payload.get("launch_policy"),
                    target_worker_id=payload.get("target_worker_id"),
                )
            )
            return 0
        if cmd == "inspect-capabilities":
            _print_ok(
                svc.inspect_engine_capabilities(
                    str(payload.get("engine_id") or args.engine_id),
                    "",
                )
            )
            return 0
        if cmd == "logs-tail":
            _print_ok(
                svc.logs_tail(
                    str(payload.get("engine_id") or args.engine_id),
                    lines=int(payload.get("lines") or 200),
                    max_bytes=int(payload.get("max_bytes") or 65536),
                )
            )
            return 0
        if cmd == "logs-follow":
            _print_ok(
                svc.logs_follow(
                    str(payload.get("engine_id") or args.engine_id),
                    cursor=int(payload.get("cursor") or 0),
                    max_bytes=int(payload.get("max_bytes") or 65536),
                    max_lines=int(payload.get("max_lines") or 500),
                )
            )
            return 0
        if cmd == "sandbox-fs-list":
            _print_ok(
                svc.sandbox_fs_list(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    root_id=str(payload.get("root_id") or ""),
                    relative_path=payload.get("relative_path"),
                )
            )
            return 0
        if cmd == "sandbox-fs-read-text":
            _print_ok(
                svc.sandbox_fs_read_text(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    root_id=str(payload.get("root_id") or ""),
                    relative_path=str(payload.get("relative_path") or ""),
                    encoding=str(payload.get("encoding") or "utf-8"),
                )
            )
            return 0
        if cmd == "sandbox-fs-write-text":
            _print_ok(
                svc.sandbox_fs_write_text(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    root_id=str(payload.get("root_id") or ""),
                    relative_path=str(payload.get("relative_path") or ""),
                    text=str(payload.get("text") or ""),
                    encoding=str(payload.get("encoding") or "utf-8"),
                    create_parents=bool(payload.get("create_parents", True)),
                )
            )
            return 0
        if cmd == "sandbox-fs-mkdir":
            _print_ok(
                svc.sandbox_fs_mkdir(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    root_id=str(payload.get("root_id") or ""),
                    relative_path=str(payload.get("relative_path") or ""),
                    parents=bool(payload.get("parents", True)),
                    exist_ok=bool(payload.get("exist_ok", True)),
                )
            )
            return 0
        if cmd == "sandbox-fs-stat":
            _print_ok(
                svc.sandbox_fs_stat(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    root_id=str(payload.get("root_id") or ""),
                    relative_path=payload.get("relative_path"),
                )
            )
            return 0
        if cmd == "sandbox-http-fetch":
            _print_ok(
                svc.sandbox_http_fetch(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    url=str(payload.get("url") or ""),
                    method=str(payload.get("method") or "GET"),
                    headers=dict(payload.get("headers") or {}),
                    body_b64=str(payload.get("body_b64") or ""),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                    max_response_bytes=int(payload.get("max_response_bytes") or 1024 * 1024),
                )
            )
            return 0
        if cmd == "toolbox-describe":
            _print_ok(
                svc.toolbox_describe(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    timeout_seconds=float(payload.get("timeout_seconds") or 10.0),
                )
            )
            return 0
        if cmd == "toolbox-gate":
            _print_ok(
                svc.toolbox_gate(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    tool_name=str(payload.get("tool_name") or ""),
                    tools_view=dict(payload.get("tools_view") or {}) if isinstance(payload.get("tools_view"), dict) else None,
                )
            )
            return 0
        if cmd == "toolbox-execute":
            _print_ok(
                svc.toolbox_execute(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    tool_call=dict(payload.get("tool_call") or {}),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                    tools_view=dict(payload.get("tools_view") or {}) if isinstance(payload.get("tools_view"), dict) else None,
                    execution_request_id=str(payload.get("execution_request_id") or ""),
                )
            )
            return 0
        if cmd == "hosted-operation-status":
            _print_ok(
                svc.hosted_operation_status(
                    ref=dict(payload.get("ref") or payload),
                )
            )
            return 0
        if cmd == "hosted-operation-result":
            _print_ok(
                svc.hosted_operation_result(
                    ref=dict(payload.get("ref") or payload),
                )
            )
            return 0
        if cmd == "hosted-operation-cancel":
            _print_ok(
                svc.hosted_operation_cancel(
                    ref=dict(payload.get("ref") or {}),
                    reason=str(payload.get("reason") or "client_requested"),
                    timeout_seconds=float(payload.get("timeout_seconds") or 8.0),
                    respawn=bool(payload.get("respawn", True)),
                )
            )
            return 0
        if cmd == "hosting-receipt-ledger-cutover":
            _print_ok(
                svc.hosting_receipt_ledger_cutover(
                    acknowledge_replay_window_clear=bool(payload.get("acknowledge_replay_window_clear", False)),
                )
            )
            return 0
        if cmd == "toolbox-state-archive-v1":
            _print_ok(
                svc.toolbox_state_archive_v1(
                    hosting_root=str(payload.get("hosting_root") or ""),
                    expected_state_sha256=str(payload.get("expected_state_sha256") or ""),
                    acknowledge_version_1_archive=bool(
                        payload.get("acknowledge_version_1_archive", False)
                    ),
                )
            )
            return 0
        if cmd == "toolbox-gc":
            _print_ok(svc.toolbox_gc())
            return 0
        if cmd == "toolbox-template-list":
            _print_ok(svc.toolbox_template_list())
            return 0
        if cmd == "toolbox-template-describe":
            _print_ok(
                svc.toolbox_template_describe(
                    template_id=str(payload.get("template_id") or ""),
                    template_digest=str(payload.get("template_digest") or "").strip() or None,
                )
            )
            return 0
        if cmd == "toolbox-template-publish":
            _print_ok(
                svc.toolbox_template_publish(
                    template=dict(payload.get("template") or {}),
                    artifact_references=[dict(item or {}) for item in list(payload.get("artifact_references") or [])],
                    manifest_signature=str(payload.get("manifest_signature") or ""),
                    activate=payload.get("activate", False),
                )
            )
            return 0
        if cmd == "toolbox-template-deprecate":
            _print_ok(
                svc.toolbox_template_deprecate(
                    template_id=str(payload.get("template_id") or ""),
                    template_digest=str(payload.get("template_digest") or ""),
                )
            )
            return 0
        if cmd == "toolbox-template-revoke":
            _print_ok(
                svc.toolbox_template_revoke(
                    template_id=str(payload.get("template_id") or ""),
                    template_digest=str(payload.get("template_digest") or ""),
                )
            )
            return 0
        if cmd == "toolbox-template-prewarm":
            _print_ok(
                svc.toolbox_template_prewarm(
                    template_id=str(payload.get("template_id") or ""),
                    template_digest=str(payload.get("template_digest") or "").strip() or None,
                    python_abi=str(payload.get("python_abi") or ""),
                    platform=str(payload.get("platform") or ""),
                    request_id=str(payload.get("request_id") or ""),
                )
            )
            return 0
        if cmd == "toolbox-references":
            _print_ok(svc.toolbox_references())
            return 0
        if cmd == "toolbox-consistency":
            _print_ok(svc.toolbox_consistency())
            return 0
        if cmd == "toolbox-review-snapshot":
            _print_ok(
                svc.toolbox_review_snapshot(
                    toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()],
                )
            )
            return 0
        if cmd == "toolbox-repair":
            _print_ok(
                svc.toolbox_repair(
                    toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()],
                    only_inconsistent=bool(payload.get("only_inconsistent", True)),
                    details=bool(payload.get("details", False)),
                )
            )
            return 0
        if cmd == "toolbox-reconcile":
            _print_ok(
                svc.toolbox_reconcile(
                    toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()],
                    only_inconsistent=bool(payload.get("only_inconsistent", True)),
                    details=bool(payload.get("details", False)),
                )
            )
            return 0
        if cmd == "toolbox-register-auto":
            _print_ok(
                svc.toolbox_register_auto(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    requests=[dict(item or {}) for item in list(payload.get("requests") or [])],
                    python_executable=str(payload.get("python_executable") or "").strip() or None,
                    worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
                )
            )
            return 0
        if cmd == "toolbox-unregister-auto":
            _print_ok(
                svc.toolbox_unregister_auto(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()],
                    python_executable=str(payload.get("python_executable") or "").strip() or None,
                    worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
                )
            )
            return 0
        if cmd == "toolbox-register-intrinsics":
            _print_ok(
                svc.toolbox_register_intrinsics(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    intrinsic_tool_names=[str(item or "").strip() for item in list(payload.get("intrinsic_tool_names") or []) if str(item or "").strip()],
                    include_guides=bool(payload.get("include_guides", False)),
                    sandbox_profile=dict(payload.get("sandbox_profile") or {}) or None,
                    python_executable=str(payload.get("python_executable") or "").strip() or None,
                    worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
                )
            )
            return 0
        if cmd == "toolbox-unregister-intrinsics":
            _print_ok(
                svc.toolbox_unregister_intrinsics(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    intrinsic_tool_names=[str(item or "").strip() for item in list(payload.get("intrinsic_tool_names") or []) if str(item or "").strip()],
                    include_guides=bool(payload.get("include_guides", False)),
                    python_executable=str(payload.get("python_executable") or "").strip() or None,
                    worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
                )
            )
            return 0
        if cmd == "toolbox-register-manual":
            _print_ok(
                svc.toolbox_register_manual(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    requests=[dict(item or {}) for item in list(payload.get("requests") or [])],
                    python_executable=str(payload.get("python_executable") or "").strip() or None,
                    worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
                )
            )
            return 0
        if cmd == "toolbox-unregister-manual":
            _print_ok(
                svc.toolbox_unregister_manual(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()],
                    python_executable=str(payload.get("python_executable") or "").strip() or None,
                    worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
                )
            )
            return 0
        if cmd == "toolbox-environment-list":
            _print_ok(svc.toolbox_environment_description_list())
            return 0
        if cmd == "toolbox-environment-upsert":
            _print_ok(
                svc.toolbox_environment_description_upsert(
                    name=str(payload.get("name") or ""),
                    base_env_name=str(payload.get("base_env_name") or "").strip() or None,
                    extra_packages=[str(item or "").strip() for item in list(payload.get("extra_packages") or []) if str(item or "").strip()],
                    allow_online_install=bool(payload.get("allow_online_install", False)),
                )
            )
            return 0
        if cmd == "toolbox-environment-clone":
            _print_ok(
                svc.toolbox_environment_description_clone(
                    source_name=str(payload.get("source_name") or ""),
                    target_name=str(payload.get("target_name") or ""),
                    extra_packages=[str(item or "").strip() for item in list(payload.get("extra_packages") or []) if str(item or "").strip()] if payload.get("extra_packages") is not None else None,
                    allow_online_install=payload.get("allow_online_install"),
                )
            )
            return 0
        if cmd == "toolbox-environment-resolve":
            _print_ok(
                svc.toolbox_environment_resolve_requirements(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    environment_name=str(payload.get("environment_name") or "base"),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                )
            )
            return 0
        if cmd == "toolbox-environment-apply":
            _print_ok(
                svc.toolbox_environment_apply(
                    environment_name=str(payload.get("environment_name") or "base"),
                    toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()] or None,
                )
            )
            return 0
        if cmd == "toolbox-environment-realize":
            _print_ok(
                svc.toolbox_environment_realize(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    environment_name=str(payload.get("environment_name") or "base"),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                )
            )
            return 0
        if cmd == "toolbox-environment-sync":
            _print_ok(
                svc.toolbox_environment_sync_description(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    source_environment_name=str(payload.get("source_environment_name") or "base"),
                    target_environment_name=str(payload.get("target_environment_name") or "").strip() or None,
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                    apply=bool(payload.get("apply", False)),
                    realize=bool(payload.get("realize", False)),
                )
            )
            return 0
        if cmd == "toolbox-environment-prepare-install":
            _print_ok(
                svc.toolbox_environment_prepare_install(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    environment_name=str(payload.get("environment_name") or "base"),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                )
            )
            return 0
        if cmd == "toolbox-environment-lock-install":
            _print_ok(
                svc.toolbox_environment_lock_install(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    environment_name=str(payload.get("environment_name") or "base"),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                )
            )
            return 0
        if cmd == "toolbox-environment-resolve-install-lock":
            _print_ok(
                svc.toolbox_environment_resolve_install_lock(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    environment_name=str(payload.get("environment_name") or "base"),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                    allow_resolution=bool(payload.get("allow_resolution", False)),
                )
            )
            return 0
        if cmd == "toolbox-environment-verify-install-lock":
            _print_ok(
                svc.toolbox_environment_verify_install_lock(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    environment_name=str(payload.get("environment_name") or "base"),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                )
            )
            return 0
        if cmd == "toolbox-environment-verify-install-receipt":
            _print_ok(
                svc.toolbox_environment_verify_install_receipt(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    environment_name=str(payload.get("environment_name") or "base"),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                )
            )
            return 0
        if cmd == "toolbox-environment-execute-install":
            _print_ok(
                svc.toolbox_environment_execute_install(
                    toolbox_id=str(payload.get("toolbox_id") or ""),
                    environment_name=str(payload.get("environment_name") or "base"),
                    tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                    allow_execution=bool(payload.get("allow_execution", False)),
                )
            )
            return 0
        if cmd == "proxy-request":
            _print_ok(
                svc.proxy_request(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    method=str(payload.get("method") or "GET"),
                    path=str(payload.get("path") or "/"),
                    query=str(payload.get("query") or ""),
                    headers=dict(payload.get("headers") or {}),
                    body_b64=str(payload.get("body_b64") or ""),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                    max_response_bytes=int(payload.get("max_response_bytes") or 1024 * 1024),
                )
            )
            return 0
        if cmd == "proxy-rpc-call":
            _print_ok(
                svc.proxy_rpc_call(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    method=str(payload.get("method") or ""),
                    params=dict(payload.get("params") or {}),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                )
            )
            return 0
        if cmd == "proxy-rpc-open":
            _print_ok(
                svc.proxy_rpc_open(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    method=str(payload.get("method") or ""),
                    params=dict(payload.get("params") or {}),
                    request_id=str(payload.get("request_id") or ""),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                )
            )
            return 0
        if cmd == "proxy-rpc-send":
            _print_ok(
                svc.proxy_rpc_send(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    stream_id=str(payload.get("stream_id") or ""),
                    message=dict(payload.get("message") or {}),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                )
            )
            return 0
        if cmd == "proxy-rpc-recv":
            _print_ok(
                svc.proxy_rpc_recv(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    stream_id=str(payload.get("stream_id") or ""),
                    timeout_seconds=float(payload.get("timeout_seconds") or 2.0),
                    max_items=int(payload.get("max_items") or 64),
                )
            )
            return 0
        if cmd == "proxy-rpc-close":
            _print_ok(
                svc.proxy_rpc_close(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    stream_id=str(payload.get("stream_id") or ""),
                    timeout_seconds=float(payload.get("timeout_seconds") or 10.0),
                )
            )
            return 0
        if cmd == "proxy-stream-open":
            _print_ok(
                svc.proxy_stream_open(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    tool=str(payload.get("tool") or "run-inference"),
                    arguments=dict(payload.get("arguments") or {}),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                )
            )
            return 0
        if cmd == "proxy-stream-send":
            _print_ok(
                svc.proxy_stream_send(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    stream_id=str(payload.get("stream_id") or ""),
                    message=dict(payload.get("message") or {}),
                    timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                )
            )
            return 0
        if cmd == "proxy-stream-recv":
            _print_ok(
                svc.proxy_stream_recv(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    stream_id=str(payload.get("stream_id") or ""),
                    timeout_seconds=float(payload.get("timeout_seconds") or 2.0),
                    max_items=int(payload.get("max_items") or 64),
                )
            )
            return 0
        if cmd == "proxy-stream-close":
            _print_ok(
                svc.proxy_stream_close(
                    engine_id=str(payload.get("engine_id") or args.engine_id),
                    stream_id=str(payload.get("stream_id") or ""),
                    timeout_seconds=float(payload.get("timeout_seconds") or 10.0),
                )
            )
            return 0
        if cmd == "host-metrics":
            _print_ok(svc.get_host_metrics())
            return 0
        if cmd in {"set-endpoint-mode-override", "get-endpoint-mode-effective"}:
            _print_error(f"{cmd} requires a running daemon")
            return 1
        if cmd == "get-control-config":
            _print_ok(svc.get_control_config())
            return 0
        if cmd == "set-control-config":
            _print_ok(
                svc.set_control_config(
                    ssh_key=payload.get("ssh_key"),
                    require_auth=payload.get("require_auth"),
                    access_profile=dict(payload.get("access_profile") or {}),
                    endpoint_mode_default=payload.get("endpoint_mode_default"),
                    lifecycle_profile=payload.get("lifecycle_profile"),
                    lifecycle_policy=dict(payload.get("lifecycle_policy") or {}),
                    traffic_policy=dict(payload.get("traffic_policy") or {}),
                    engine_traffic_policies=dict(payload.get("engine_traffic_policies") or {}),
                    claim_acl_policy=dict(payload.get("claim_acl_policy") or {}),
                )
            )
            return 0
        if cmd == "get-lifecycle-policy-effective":
            _print_ok(svc.get_lifecycle_policy_effective())
            return 0
        if cmd == "auth-status":
            _print_ok(svc.auth_status())
            return 0
        if cmd == "daemon-status":
            _print_error("daemon-status requires a running daemon")
            return 1
        if cmd == "hosting-setup-status":
            _print_ok(svc.hosting_setup_summary())
            return 0
        if cmd == "model-runtime-status":
            _print_ok(svc.model_runtime_status())
            return 0
        if cmd == "hosting-secure-state-status":
            _print_ok(svc.hosting_secure_state_status())
            return 0
        if cmd == "auth-list-keys":
            _print_ok(svc.auth_list_keys())
            return 0
        if cmd == "auth-list-sessions":
            _print_ok(
                svc.auth_list_sessions(
                    key_id=payload.get("key_id"),
                    scope=payload.get("scope"),
                    role=payload.get("role"),
                    token_preview_contains=payload.get("token_preview_contains"),
                    limit=int(payload.get("limit") or 100),
                    offset=int(payload.get("offset") or 0),
                )
            )
            return 0
        if cmd == "list-live-consumers":
            _print_error("list-live-consumers requires a running daemon")
            return 1
        if cmd == "auth-list-issued-tokens":
            _print_ok(
                svc.auth_list_issued_tokens(
                    engine_id=payload.get("engine_id"),
                    resource_kind=payload.get("resource_kind"),
                    resource_id=payload.get("resource_id"),
                    backend_id=payload.get("backend_id"),
                    token_preview_contains=payload.get("token_preview_contains"),
                    limit=int(payload.get("limit") or 100),
                    offset=int(payload.get("offset") or 0),
                )
            )
            return 0
        if cmd == "auth-audit-list":
            _print_ok(
                svc.auth_list_audit_events(
                    event_type=payload.get("event_type"),
                    actor_key_id=payload.get("actor_key_id"),
                    target_key_id=payload.get("target_key_id"),
                    result=payload.get("result"),
                    limit=int(payload.get("limit") or 100),
                    offset=int(payload.get("offset") or 0),
                )
            )
            return 0
        if cmd == "host-capability-audit-list":
            _print_ok(
                svc.host_capability_audit_list(
                    workflow_id=payload.get("workflow_id"),
                    instance_id=payload.get("instance_id"),
                    request_id=payload.get("request_id"),
                    provider_id=payload.get("provider_id"),
                    method=payload.get("method"),
                    approval_id=payload.get("approval_id"),
                    since=payload.get("since"),
                    until=payload.get("until"),
                    limit=int(payload.get("limit") or 100),
                    offset=int(payload.get("offset") or 0),
                )
            )
            return 0
        if cmd == "auth-validate-session":
            _print_ok(
                svc.auth_validate_session(
                    token=str(payload.get("token") or payload.get("session_token") or ""),
                    scope=str(payload.get("scope") or "control"),
                    expected_key_id=payload.get("expected_key_id") or payload.get("key_id"),
                    check_ssh_binding=bool(payload.get("check_ssh_binding", True)),
                    presented_ssh_binding=dict(payload.get("_ssh_session_binding") or payload.get("ssh_binding") or {}),
                )
            )
            return 0
        if cmd == "auth-renew-session":
            _print_ok(
                svc.auth_renew_session(
                    token=str(payload.get("token") or payload.get("session_token") or ""),
                    scope=str(payload.get("scope") or "control"),
                    ttl_seconds=int(payload.get("ttl_seconds") or 900),
                    presented_ssh_binding=dict(payload.get("_ssh_session_binding") or payload.get("ssh_binding") or {}),
                )
            )
            return 0
        if cmd == "auth-upsert-key":
            _print_ok(
                svc.auth_upsert_key(
                    key_id=str(payload.get("key_id") or ""),
                    key_secret=str(payload.get("key_secret") or ""),
                    role=str(payload.get("role") or ""),
                    auth_method=str(payload.get("auth_method") or "shared_secret"),
                    public_key=str(payload.get("public_key") or ""),
                    allowed_configs=list(payload.get("allowed_configs") or []),
                    allowed_engines=list(payload.get("allowed_engines") or []),
                    disabled=bool(payload.get("disabled", False)),
                )
            )
            return 0
        if cmd == "auth-revoke-key":
            _print_ok(svc.auth_revoke_key(str(payload.get("key_id") or "")))
            return 0
        if cmd == "auth-issue-session":
            _print_ok(
                svc.auth_issue_session(
                    key_id=str(payload.get("key_id") or ""),
                    key_secret=str(payload.get("key_secret") or ""),
                    scope=str(payload.get("scope") or "control"),
                    ttl_seconds=int(payload.get("ttl_seconds") or 900),
                    config_paths=list(payload.get("config_paths") or []),
                    engine_ids=list(payload.get("engine_ids") or []),
                    ssh_binding=dict(payload.get("ssh_binding") or {}),
                )
            )
            return 0
        if cmd == "auth-begin-challenge":
            _print_ok(
                svc.auth_begin_challenge(
                    key_id=str(payload.get("key_id") or ""),
                    scope=str(payload.get("scope") or "control"),
                    ttl_seconds=int(payload.get("ttl_seconds") or 120),
                    config_paths=list(payload.get("config_paths") or []),
                    engine_ids=list(payload.get("engine_ids") or []),
                    ssh_binding=dict(payload.get("ssh_binding") or {}),
                )
            )
            return 0
        if cmd == "auth-complete-challenge":
            _print_ok(
                svc.auth_complete_challenge(
                    challenge_id=str(payload.get("challenge_id") or ""),
                    signature_ssh=str(payload.get("signature_ssh") or ""),
                    presented_ssh_binding=dict(payload.get("_ssh_session_binding") or {}),
                )
            )
            return 0
        if cmd == "auth-revoke-session":
            _print_ok(svc.auth_revoke_session(str(payload.get("token") or "")))
            return 0
        if cmd in {"op-start", "op-status"}:
            _print_error(f"{cmd} requires a running daemon")
            return 1
        _print_error(f"Unknown command '{cmd}'")
        return 2
    except PermissionError as e:
        _print_error(f"auth_failed: {e}")
        return 1
    except Exception as e:
        _print_error(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
