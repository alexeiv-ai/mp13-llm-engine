"""
Long-lived daemon server for engine host control.

Start in foreground:
  python -m hosting.engine_host_cli --daemon

Start detached in background:
  python -m hosting.engine_host_cli --daemon --background

The daemon binds to 127.0.0.1:<port> (default 19876) and accepts persistent
client connections using line-delimited JSON:

  Request:  {"seq": N, "cmd": "discover-running", "payload": {}}\n
  Response: {"seq": N, "ok": true, "result": [...]}\n
  Error:    {"seq": N, "ok": false, "error": "message"}\n

Built-in commands:
  __ping__     -> {"seq": N, "ok": true, "result": "pong"}
  __shutdown__ -> requires {"shutdown_token": "..."} in payload; stops daemon
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DAEMON_PORT = 19876


def _default_state_dir() -> Path:
    try:
        from mp13_engine.mp13_config_paths import get_default_config_dir  # type: ignore
        return (Path(get_default_config_dir()) / "backend").expanduser().resolve()
    except Exception:
        return (Path.home() / ".mp13-llm" / "backend").expanduser().resolve()


def _default_pid_file() -> Path:
    return _default_state_dir() / "host_daemon.pid"


class DaemonPidFile:
    """Read/write the daemon PID file used for discovery by CLI and channel."""

    def __init__(self, path: Optional[Path] = None):
        self.path = (Path(path) if path else _default_pid_file()).expanduser().resolve()

    def write(self, *, pid: int, port: int, shutdown_token: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "pid": int(pid),
            "port": int(port),
            "started_at": time.time(),
            "shutdown_token": str(shutdown_token),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return dict(raw) if isinstance(raw, dict) else None
        except Exception:
            return None

    def remove(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            p = int(pid or 0)
            if p <= 0:
                return False
            os.kill(p, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    def is_alive(self) -> bool:
        info = self.read()
        if not info:
            return False
        return self._pid_alive(int(info.get("pid") or 0))

    def get_port(self) -> Optional[int]:
        info = self.read()
        if not info:
            return None
        port = int(info.get("port") or 0)
        return port if port > 0 else None

    def get_shutdown_token(self) -> Optional[str]:
        info = self.read()
        if not info:
            return None
        return str(info.get("shutdown_token") or "").strip() or None


class EngineHostDaemon:
    """
    Asyncio TCP server that routes line-delimited JSON requests to EngineHostService.

    Usage::

        daemon = EngineHostDaemon(port=19876)
        asyncio.run(daemon.run())  # blocks until __shutdown__ or SIGINT
    """

    def __init__(
        self,
        *,
        port: int = DEFAULT_DAEMON_PORT,
        pid_file: Optional[Path] = None,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
    ):
        from .engine_host_service import EngineHostService
        self.port = int(port or DEFAULT_DAEMON_PORT)
        self.pid_file = DaemonPidFile(pid_file)
        self.shutdown_token = secrets.token_urlsafe(24)
        self.svc = EngineHostService(
            engines_state_file=engines_state_file,
            control_state_file=control_state_file,
        )
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def run(self) -> None:
        """Start server, write PID file, run until stop event, clean up."""
        self._stop_event = asyncio.Event()
        self.pid_file.write(pid=os.getpid(), port=self.port, shutdown_token=self.shutdown_token)
        logger.info("EngineHostDaemon starting on 127.0.0.1:%d", self.port)
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                "127.0.0.1",
                self.port,
                limit=2 ** 20,
            )
            async with self._server:
                await self._stop_event.wait()
        finally:
            self.pid_file.remove()
            logger.info("EngineHostDaemon stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.debug("Client connected: %s", peer)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=300.0)
                except asyncio.TimeoutError:
                    break
                if not line:
                    break
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                response = await self._dispatch(raw)
                writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
                # Stop serving this client after __shutdown__ is accepted
                if response.get("result") == "shutting_down" and response.get("ok"):
                    break
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            logger.warning("Client error %s: %s", peer, exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug("Client disconnected: %s", peer)

    async def _dispatch(self, raw_line: str) -> Dict[str, Any]:
        try:
            req = json.loads(raw_line)
        except Exception:
            return {"seq": -1, "ok": False, "error": "parse_error"}
        seq = int(req.get("seq") or 0)
        cmd = str(req.get("cmd") or "").strip()
        payload = dict(req.get("payload") or {})

        if cmd == "__ping__":
            return {"seq": seq, "ok": True, "result": "pong"}

        if cmd == "__shutdown__":
            token = str(payload.get("shutdown_token") or "")
            if token and token == self.shutdown_token:
                assert self._stop_event is not None
                self._stop_event.set()
                return {"seq": seq, "ok": True, "result": "shutting_down"}
            return {"seq": seq, "ok": False, "error": "invalid_shutdown_token"}

        try:
            result = await asyncio.to_thread(self._call_service, cmd, payload)
            return {"seq": seq, "ok": True, "result": result}
        except Exception as exc:
            return {"seq": seq, "ok": False, "error": str(exc)}

    def _call_service(self, cmd: str, payload: Dict[str, Any]) -> Any:
        """Synchronous dispatch to EngineHostService (runs in thread pool)."""
        svc = self.svc
        if cmd == "discover-running":
            return svc.discover_running()
        if cmd == "spawn":
            return svc.spawn(
                engine_id=str(payload.get("engine_id") or ""),
                command=list(payload.get("command") or []),
                cwd=payload.get("cwd"),
                env=dict(payload.get("env") or {}),
                endpoint=payload.get("endpoint"),
            )
        if cmd == "get-registration":
            return svc.get_registration(str(payload.get("engine_id") or ""))
        if cmd == "shutdown":
            return svc.shutdown(
                str(payload.get("engine_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 8.0),
            )
        if cmd == "ensure-running":
            return svc.ensure_running(str(payload.get("engine_id") or ""))
        if cmd == "remove-registration":
            return svc.remove_registration(str(payload.get("engine_id") or ""))
        if cmd == "claim-engine":
            return svc.claim_engine(
                str(payload.get("engine_id") or ""),
                backend_id=payload.get("backend_id"),
                exclusive=bool(payload.get("exclusive", False)),
            )
        if cmd == "claim-endpoint":
            return svc.claim_endpoint(
                backend_id=payload.get("backend_id"),
                exclusive=bool(payload.get("exclusive", False)),
            )
        if cmd == "claim-status":
            return svc.get_claim_status(str(payload.get("engine_id") or ""))
        if cmd == "issue-token":
            return svc.issue_token(
                str(payload.get("engine_id") or ""),
                backend_id=payload.get("backend_id"),
            )
        if cmd == "validate-token":
            return svc.validate_token(
                str(payload.get("engine_id") or ""),
                str(payload.get("token") or ""),
            )
        if cmd == "claim-resource":
            return svc.claim_resource(
                str(payload.get("resource_kind") or ""),
                str(payload.get("resource_id") or ""),
                backend_id=payload.get("backend_id"),
                exclusive=bool(payload.get("exclusive", False)),
            )
        if cmd == "resource-claim-status":
            return svc.get_resource_claim_status(
                str(payload.get("resource_kind") or ""),
                str(payload.get("resource_id") or ""),
            )
        if cmd == "issue-resource-token":
            return svc.issue_resource_token(
                str(payload.get("resource_kind") or ""),
                str(payload.get("resource_id") or ""),
                backend_id=payload.get("backend_id"),
            )
        if cmd == "validate-resource-token":
            return svc.validate_resource_token(
                str(payload.get("resource_kind") or ""),
                str(payload.get("resource_id") or ""),
                str(payload.get("token") or ""),
            )
        if cmd == "list-configs":
            return svc.list_engine_configs()
        if cmd == "create-config":
            return svc.create_engine_config(
                name=str(payload.get("name") or "engine_config"),
                config=dict(payload.get("config") or {}),
                overwrite=bool(payload.get("overwrite", False)),
            )
        if cmd == "models-from-config":
            return svc.models_from_config(str(payload.get("config_path") or "default"))
        if cmd == "connect-from-config":
            return svc.connect_from_config(
                config_path=str(payload.get("config_path") or "default"),
                engine_id=payload.get("engine_id"),
                model_path=payload.get("model_path"),
            )
        if cmd == "inspect-capabilities":
            return svc.inspect_engine_capabilities(
                str(payload.get("engine_id") or ""),
                str(payload.get("endpoint") or ""),
            )
        if cmd == "logs-tail":
            return svc.logs_tail(
                str(payload.get("engine_id") or ""),
                lines=int(payload.get("lines") or 200),
                max_bytes=int(payload.get("max_bytes") or 65536),
            )
        if cmd == "logs-follow":
            return svc.logs_follow(
                str(payload.get("engine_id") or ""),
                cursor=int(payload.get("cursor") or 0),
                max_bytes=int(payload.get("max_bytes") or 65536),
                max_lines=int(payload.get("max_lines") or 500),
            )
        if cmd == "get-control-config":
            return svc.get_control_config()
        if cmd == "set-control-config":
            return svc.set_control_config(ssh_key=payload.get("ssh_key"))
        raise ValueError(f"Unknown command '{cmd}'")


def run_daemon_foreground(
    *,
    port: int = DEFAULT_DAEMON_PORT,
    pid_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
) -> None:
    """Start daemon in the foreground (blocks until stopped)."""
    daemon = EngineHostDaemon(
        port=port,
        pid_file=pid_file,
        engines_state_file=engines_state_file,
        control_state_file=control_state_file,
    )
    asyncio.run(daemon.run())


def start_daemon_background(
    *,
    port: int = DEFAULT_DAEMON_PORT,
    pid_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
    wait_ready_seconds: float = 8.0,
) -> Dict[str, Any]:
    """
    Spawn daemon as a detached background process and wait until it is connectable.

    Returns {"pid": N, "port": P} on success.
    Raises RuntimeError if daemon does not become reachable within wait_ready_seconds.
    """
    argv: List[str] = [sys.executable, "-m", "hosting.engine_host_cli", "--daemon", "--port", str(port)]
    if pid_file:
        argv += ["--pid-file", str(pid_file)]
    if engines_state_file:
        argv += ["--engines-state-file", str(engines_state_file)]
    if control_state_file:
        argv += ["--control-state-file", str(control_state_file)]

    # Build environment with src dir on PYTHONPATH so connectors package is found
    import os as _os
    env = dict(_os.environ)
    src_root = str(Path(__file__).resolve().parents[1])
    py_path = str(env.get("PYTHONPATH") or "")
    if src_root not in py_path.split(_os.pathsep):
        env["PYTHONPATH"] = src_root if not py_path else f"{src_root}{_os.pathsep}{py_path}"

    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(argv, **kwargs)  # noqa: S603
    spawned_pid = int(proc.pid)

    # Poll until PID file appears and socket is connectable
    pid_info = DaemonPidFile(pid_file)
    deadline = time.time() + max(1.0, float(wait_ready_seconds))
    while time.time() < deadline:
        time.sleep(0.15)
        if not pid_info.is_alive():
            continue
        actual_port = pid_info.get_port()
        if not actual_port:
            continue
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(("127.0.0.1", actual_port))
            s.close()
            info = pid_info.read() or {}
            return {"pid": int(info.get("pid") or spawned_pid), "port": actual_port}
        except OSError:
            continue

    raise RuntimeError(
        f"Engine host daemon did not become ready within {wait_ready_seconds}s "
        f"(spawned pid={spawned_pid}, port={port})"
    )
