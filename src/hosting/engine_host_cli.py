"""
Terminal-friendly command interface for engine host lifecycle/control.

Modes:
  --daemon              Start long-lived daemon server (foreground)
  --daemon --background Start daemon detached in background
  --daemon-http         Start HTTP ingress daemon (foreground)
  --daemon-http --background Start HTTP ingress daemon detached in background
  --relay               Bridge stdin/stdout to local daemon TCP socket (SSH channel)
  <subcommand>          Short-lived: send one command to running daemon (or direct fallback)

Usage examples:
  python -m hosting.engine_host_cli --daemon
  python -m hosting.engine_host_cli --daemon --background
  python -m hosting.engine_host_cli --daemon-http
  python -m hosting.engine_host_cli --daemon-http --background
  python -m hosting.engine_host_cli --relay
  python -m hosting.engine_host_cli discover-running
  python -m hosting.engine_host_cli spawn --payload-stdin < payload.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .engine_host_service import EngineHostService


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
    "auth-upsert-key": [
        "@'{\"key_id\":\"admin-key\",\"key_secret\":\"change_me\",\"role\":\"management\"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key",
        "@'{\"key_id\":\"traffic-key\",\"key_secret\":\"change_me\",\"role\":\"traffic\",\"allowed_engines\":[\"worker1\",\"worker2\"]}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key",
        "@'{\"key_id\":\"admin-pub\",\"auth_method\":\"public_key\",\"public_key\":\"ssh-ed25519 AAAA...\",\"role\":\"management\"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key",
    ],
    "auth-issue-session": [
        "@'{\"key_id\":\"admin-key\",\"key_secret\":\"change_me\",\"scope\":\"control\"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-issue-session",
    ],
    "auth-status": [
        "python -m hosting.engine_host_cli auth-status",
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


def _print_error(message: str) -> None:
    print(json.dumps({"ok": False, "error": str(message or "unknown_error")}, ensure_ascii=False))


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
    import time
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
# Relay mode: bridge stdin/stdout to local daemon TCP socket
# ---------------------------------------------------------------------------

def _run_relay(port: int) -> None:
    """Bridge sys.stdin.buffer -> daemon TCP socket, daemon -> sys.stdout.buffer."""
    import socket as _socket

    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.settimeout(10.0)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(None)

    sock_file = sock.makefile("rb")

    def _reader() -> None:
        try:
            for line in sock_file:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        except Exception:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        for line in sys.stdin.buffer:
            stripped = line.strip()
            if not stripped:
                continue
            sock.sendall(stripped + b"\n")
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass


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
    from .engine_host_daemon import DaemonPidFile
    from .engine_host_connection import LocalSocketConnection

    pid_info = DaemonPidFile(pid_file)
    if not pid_info.is_alive():
        return False
    port = pid_info.get_port()
    if not port:
        return False
    try:
        conn = LocalSocketConnection(port=port, timeout=10.0, max_reconnect_attempts=1)
        result = conn.invoke(cmd, payload)
        conn.close()
        _print_ok(result)
        return True
    except Exception:
        return False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Engine host control CLI")
    p.add_argument("--engines-state-file", type=Path, default=None)
    p.add_argument("--control-state-file", type=Path, default=None)
    p.add_argument("--pid-file", type=Path, default=None, help="Daemon PID file path (for daemon client mode)")
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
    sp = p.add_subparsers(dest="command", required=True)

    for name in [
        "discover-running",
        "spawn",
        "get-registration",
        "shutdown",
        "ensure-running",
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
        "get-control-config",
        "set-control-config",
        "auth-status",
        "auth-list-keys",
        "auth-list-sessions",
        "auth-list-issued-tokens",
        "auth-upsert-key",
        "auth-revoke-key",
        "auth-issue-session",
        "auth-begin-challenge",
        "auth-complete-challenge",
        "auth-revoke-session",
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
        from .engine_host_daemon import (
            DEFAULT_DAEMON_PORT,
            run_daemon_foreground,
            start_daemon_background,
        )

        log_file_str = _extract_str_arg(argv, "--log-file", None)
        _setup_file_logging(log_file_str)

        port = _extract_int_arg(argv, "--port", DEFAULT_DAEMON_PORT)
        pid_file = _extract_path_arg(argv, "--pid-file", None)
        engines_state = _extract_path_arg(argv, "--engines-state-file", None)
        control_state = _extract_path_arg(argv, "--control-state-file", None)
        background = "--background" in argv

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
            )
            return 0

    # ------------------------------------------------------------------
    # Mode 1b: --daemon-http  →  start HTTP ingress daemon
    # ------------------------------------------------------------------
    if "--daemon-http" in argv:
        from .engine_host_daemon import (
            DEFAULT_HTTP_INGRESS_PORT,
            run_http_ingress_foreground,
            start_http_ingress_background,
        )

        log_file_str = _extract_str_arg(argv, "--log-file", None)
        _setup_file_logging(log_file_str)

        port = _extract_int_arg(argv, "--http-port", DEFAULT_HTTP_INGRESS_PORT)
        pid_file = _extract_path_arg(argv, "--pid-file", None)
        engines_state = _extract_path_arg(argv, "--engines-state-file", None)
        control_state = _extract_path_arg(argv, "--control-state-file", None)
        background = "--background" in argv

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
    # Mode 2: --relay  →  bridge stdin/stdout to local daemon TCP socket
    # ------------------------------------------------------------------
    if "--relay" in argv:
        from .engine_host_daemon import DEFAULT_DAEMON_PORT, DaemonPidFile

        port = _extract_int_arg(argv, "--port", 0)
        pid_file = _extract_path_arg(argv, "--pid-file", None)
        if not port:
            pid_info = DaemonPidFile(pid_file)
            port = pid_info.get_port() or DEFAULT_DAEMON_PORT
        try:
            _run_relay(port=port)
            return 0
        except Exception as exc:
            _print_error(str(exc))
            return 1

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
    payload = _load_payload(args)

    # Try sending the command to the running daemon first
    pid_file_arg = getattr(args, "pid_file", None)
    cmd_name = str(args.command or "").strip()
    if cmd_name and _try_daemon_invoke(cmd_name, payload, pid_file=pid_file_arg):
        return 0

    # Fallback: direct EngineHostService call (original behavior)
    svc = EngineHostService(
        engines_state_file=args.engines_state_file,
        control_state_file=args.control_state_file,
    )
    try:
        cmd = str(args.command or "").strip()
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
                )
            )
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
        if cmd == "remove-registration":
            _print_ok(svc.remove_registration(str(payload.get("engine_id") or args.engine_id)))
            return 0
        if cmd == "claim-engine":
            _print_ok(
                svc.claim_engine(
                    str(payload.get("engine_id") or args.engine_id),
                    backend_id=payload.get("backend_id"),
                    exclusive=bool(payload.get("exclusive", False)),
                    force_override=bool(payload.get("force_override", False)),
                    actor_id=payload.get("_claim_actor_id"),
                    peer_host=payload.get("_daemon_peer_host"),
                )
            )
            return 0
        if cmd == "claim-endpoint":
            _print_ok(
                svc.claim_endpoint(
                    backend_id=payload.get("backend_id"),
                    exclusive=bool(payload.get("exclusive", False)),
                    force_override=bool(payload.get("force_override", False)),
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
                    exclusive=bool(payload.get("exclusive", False)),
                    force_override=bool(payload.get("force_override", False)),
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
        if cmd == "get-control-config":
            _print_ok(svc.get_control_config())
            return 0
        if cmd == "set-control-config":
            _print_ok(
                svc.set_control_config(
                    ssh_key=payload.get("ssh_key"),
                    require_auth=payload.get("require_auth"),
                    traffic_policy=dict(payload.get("traffic_policy") or {}),
                    engine_traffic_policies=dict(payload.get("engine_traffic_policies") or {}),
                    claim_acl_policy=dict(payload.get("claim_acl_policy") or {}),
                )
            )
            return 0
        if cmd == "auth-status":
            _print_ok(svc.auth_status())
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
