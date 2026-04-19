"""Local IPC engine host daemon runtime."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
import time
from multiprocessing.connection import Client as MPClient
from multiprocessing.connection import Listener as MPListener
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..engine_host_service import EngineHostService
from .constants import DEFAULT_DAEMON_PORT
from .paths import _daemon_local_ipc_endpoint
from .pidfile import DaemonPidFile

logger = logging.getLogger(__name__)


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
        runtime_profile: str = "foreground_terminal_bound",
    ):
        self.port = int(port or DEFAULT_DAEMON_PORT)
        self.pid_file = DaemonPidFile(pid_file)
        self.shutdown_token = secrets.token_urlsafe(24)
        local_transport = _daemon_local_ipc_endpoint(self.pid_file.path)
        self._local_transport = dict(local_transport)
        self.svc = EngineHostService(
            engines_state_file=engines_state_file,
            control_state_file=control_state_file,
        )
        self.svc.assert_runtime_policy_safe()
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._local_listener_thread: Optional[threading.Thread] = None
        self._local_listener_stop = threading.Event()
        self._local_listener_ready = threading.Event()
        self._local_listener_error: Optional[str] = None
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._operations_lock = threading.Lock()
        self._operations_max_entries = 200
        self._operation_tasks: set[asyncio.Task] = set()
        self._operation_tasks_lock = threading.Lock()
        self._endpoint_mode_runtime_override: Optional[str] = None
        self._runtime_profile = str(runtime_profile or "foreground_terminal_bound").strip().lower()
        self._actor_connections: Dict[str, int] = {}
        self._actor_connections_lock = threading.Lock()
        self._last_shutdown_checkpoints: Dict[str, Any] = {}
        self._shutdown_stage_events: List[Dict[str, Any]] = []

    def _serve_local_control_client(self, conn: Any) -> None:
        connection_actor_ids: set[str] = set()
        try:
            while not self._local_listener_stop.is_set():
                try:
                    req_obj = conn.recv()
                except EOFError:
                    break
                if not isinstance(req_obj, dict):
                    try:
                        conn.send(
                            {
                                "seq": -1,
                                "ok": False,
                                "error": "parse_error",
                                "error_code": "parse_error",
                                "error_details": {},
                            }
                        )
                    except Exception:
                        pass
                    continue
                payload_obj = dict(req_obj.get("payload") or {})
                tok = str(
                    payload_obj.get("session_token")
                    or payload_obj.get("auth_token")
                    or ""
                ).strip()
                if tok:
                    actor_id = self.svc.resolve_actor_id_from_session_token(tok)
                    if actor_id and actor_id not in connection_actor_ids:
                        connection_actor_ids.add(actor_id)
                        self._track_actor_connected(actor_id)
                raw = json.dumps(req_obj, ensure_ascii=False)
                loop = self._loop
                if loop is None:
                    response = {
                        "seq": int(req_obj.get("seq") or 0),
                        "ok": False,
                        "error": "daemon_loop_unavailable",
                        "error_code": "daemon_loop_unavailable",
                        "error_details": {},
                    }
                else:
                    fut = asyncio.run_coroutine_threadsafe(
                        self._dispatch(raw, peer_host="127.0.0.1"),
                        loop,
                    )
                    response = fut.result(timeout=60.0)
                conn.send(response)
                if response.get("result") == "shutting_down" and response.get("ok"):
                    break
        except Exception as exc:
            logger.warning("Local IPC client error: %s", exc)
        finally:
            try:
                _ = self._apply_owner_disconnect_policy(connection_actor_ids)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    def _run_local_control_listener(self) -> None:
        family = str(self._local_transport.get("family") or "").strip() or "AF_UNIX"
        address = str(self._local_transport.get("address") or "").strip()
        listener = None
        try:
            if family == "AF_UNIX" and address:
                try:
                    Path(address).unlink(missing_ok=True)
                except Exception:
                    pass
            listener = MPListener(
                address=address,
                family=family,
                authkey=self.shutdown_token.encode("utf-8", errors="ignore"),
            )
            if family == "AF_UNIX" and address:
                try:
                    os.chmod(address, 0o600)
                except Exception:
                    pass
            self._local_listener_ready.set()
            while not self._local_listener_stop.is_set():
                try:
                    conn = listener.accept()
                except Exception:
                    if self._local_listener_stop.is_set():
                        break
                    raise
                t = threading.Thread(
                    target=self._serve_local_control_client,
                    args=(conn,),
                    daemon=True,
                )
                t.start()
        except Exception as exc:
            self._local_listener_error = str(exc)
            self._local_listener_ready.set()
            logger.warning("Local IPC listener failed: %s", exc)
        finally:
            if listener is not None:
                try:
                    listener.close()
                except Exception:
                    pass
            if family == "AF_UNIX" and address:
                try:
                    Path(address).unlink(missing_ok=True)
                except Exception:
                    pass

    def _start_local_control_listener(self) -> None:
        self._local_listener_stop.clear()
        self._local_listener_ready.clear()
        self._local_listener_error = None
        self._local_listener_thread = threading.Thread(
            target=self._run_local_control_listener,
            daemon=True,
            name="engine-host-local-ipc",
        )
        self._local_listener_thread.start()
        if not self._local_listener_ready.wait(timeout=5.0):
            raise RuntimeError("local IPC listener did not become ready")
        if self._local_listener_error:
            raise RuntimeError(self._local_listener_error)

    def _stop_local_control_listener(self) -> None:
        self._local_listener_stop.set()
        family = str(self._local_transport.get("family") or "").strip()
        address = str(self._local_transport.get("address") or "").strip()
        if family and address:
            try:
                conn = MPClient(
                    address=address,
                    family=family,
                    authkey=self.shutdown_token.encode("utf-8", errors="ignore"),
                )
                try:
                    conn.send({"seq": 0, "cmd": "__ping__", "payload": {}})
                except Exception:
                    pass
                conn.close()
            except Exception:
                pass
        thread = self._local_listener_thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._local_listener_thread = None

    def _should_enable_tcp(self) -> bool:
        status = self.svc.auth_status()
        if not status.get("require_auth"):
            return False
        roles = set(status.get("roles") or [])
        return "admin" in roles and "transport" in roles

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.debug("Client connected: %s", peer)
        connection_actor_ids: set[str] = set()
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
                try:
                    req_obj = json.loads(raw)
                    payload_obj = dict((req_obj or {}).get("payload") or {})
                    tok = str(
                        payload_obj.get("session_token")
                        or payload_obj.get("auth_token")
                        or ""
                    ).strip()
                    if tok:
                        actor_id = self.svc.resolve_actor_id_from_session_token(tok)
                        if actor_id and actor_id not in connection_actor_ids:
                            connection_actor_ids.add(actor_id)
                            self._track_actor_connected(actor_id)
                except Exception:
                    pass
                peer_host = ""
                try:
                    if isinstance(peer, tuple) and len(peer) >= 1:
                        peer_host = str(peer[0] or "")
                except Exception:
                    peer_host = ""
                response = await self._dispatch(raw, peer_host=peer_host)
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
                _ = self._apply_owner_disconnect_policy(connection_actor_ids)
            except Exception:
                pass
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug("Client disconnected: %s", peer)

    @staticmethod
    def _operation_event(stage: str, status: str, message: str, **extra: Any) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "stage": str(stage or "unknown"),
            "status": str(status or "info"),
            "message": str(message or ""),
            "timestamp": time.time(),
        }
        if extra:
            event.update({str(k): v for k, v in extra.items()})
        return event

    @staticmethod
    def _operation_public_snapshot(op: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(op or {})
        out.pop("session_token", None)
        return out

    @staticmethod
    def _is_claim_command(cmd: str) -> bool:
        c = str(cmd or "").strip()
        return c in {"claim-engine", "claim-endpoint", "claim-resource"}

    def _effective_endpoint_mode(self) -> Dict[str, str]:
        cfg = self.svc.get_control_config()
        default_mode = str(cfg.get("endpoint_mode_default") or "shared").strip().lower()
        if default_mode not in {"exclusive", "shared"}:
            default_mode = "shared"
        override = str(self._endpoint_mode_runtime_override or "").strip().lower()
        if override not in {"exclusive", "shared"}:
            override = ""
        effective = override or default_mode
        return {
            "default": default_mode,
            "runtime_override": override or None,
            "effective": effective,
        }

    def _inject_runtime_endpoint_mode(self, cmd: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(payload or {})
        if (not self._is_claim_command(cmd)) or ("exclusive" in p):
            return p
        mode = self._effective_endpoint_mode().get("effective") or "shared"
        p["exclusive"] = bool(mode == "exclusive")
        return p

    def _prune_operations_locked(self) -> None:
        if len(self._operations) <= self._operations_max_entries:
            return
        completed = []
        for op_id, op in self._operations.items():
            if bool(op.get("done", False)):
                completed.append((float(op.get("updated_at") or 0.0), op_id))
        completed.sort(key=lambda x: x[0])
        excess = len(self._operations) - self._operations_max_entries
        for _, op_id in completed[:excess]:
            self._operations.pop(op_id, None)

    def _record_shutdown_stage(self, stage: str, status: str, message: str, **extra: Any) -> None:
        event: Dict[str, Any] = {
            "stage": str(stage or "unknown"),
            "status": str(status or "info"),
            "message": str(message or ""),
            "timestamp": time.time(),
        }
        if extra:
            event.update({str(k): v for k, v in extra.items()})
        self._shutdown_stage_events.append(event)

    def _terminal_control_enabled(self) -> bool:
        policy = self.svc.get_lifecycle_policy_effective()
        eff = dict(policy.get("effective") or {})
        return bool(eff.get("terminal_control_enabled", True))

    async def _drain_inflight_operations(self, *, timeout_seconds: float = 5.0) -> Dict[str, Any]:
        with self._operation_tasks_lock:
            pending = [t for t in list(self._operation_tasks) if (t is not None and not t.done())]
        if not pending:
            return {
                "pending_before": 0,
                "pending_after": 0,
                "drained": 0,
                "timed_out": False,
                "timeout_seconds": float(timeout_seconds),
            }
        done, not_done = await asyncio.wait(
            pending,
            timeout=max(0.1, float(timeout_seconds)),
        )
        with self._operation_tasks_lock:
            self._operation_tasks = {t for t in self._operation_tasks if t is not None and not t.done()}
            pending_after = len(self._operation_tasks)
        return {
            "pending_before": len(pending),
            "pending_after": int(pending_after),
            "drained": len(done),
            "timed_out": len(not_done) > 0,
            "timeout_seconds": float(timeout_seconds),
        }

    def _execute_shutdown_checkpoints(self) -> Dict[str, Any]:
        started_at = time.time()
        report: Dict[str, Any] = {
            "status": "ok",
            "started_at": started_at,
            "completed_at": None,
            "attempted": 0,
            "stopped": 0,
            "failed": 0,
            "registrations_before": 0,
            "registrations_after": 0,
            "results": [],
            "error": None,
        }
        try:
            rows = self.svc.discover_running(
                prune_stale=False,
                include_progress=False,
                include_reachability=False,
            )
            registrations = list(rows or []) if isinstance(rows, list) else []
            report["registrations_before"] = len(registrations)
            for row in registrations:
                engine_id = str((row or {}).get("engine_id") or "").strip()
                if not engine_id:
                    continue
                report["attempted"] = int(report.get("attempted") or 0) + 1
                try:
                    out = self.svc.shutdown(engine_id, timeout_seconds=2.0)
                    status = str((out or {}).get("status") or "")
                    ok = status in {"stopped", "already_stopped", "not_found", "invalid_pid"}
                    if ok:
                        report["stopped"] = int(report.get("stopped") or 0) + 1
                    else:
                        report["failed"] = int(report.get("failed") or 0) + 1
                    report["results"].append(
                        {
                            "engine_id": engine_id,
                            "status": status,
                            "ok": ok,
                        }
                    )
                except Exception as exc:
                    report["failed"] = int(report.get("failed") or 0) + 1
                    report["results"].append(
                        {
                            "engine_id": engine_id,
                            "status": "exception",
                            "ok": False,
                            "error": str(exc),
                        }
                    )
            after_rows = self.svc.discover_running(
                prune_stale=False,
                include_progress=False,
                include_reachability=False,
            )
            report["registrations_after"] = len(list(after_rows or [])) if isinstance(after_rows, list) else 0
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = str(exc)
        report["completed_at"] = time.time()
        self._last_shutdown_checkpoints = report
        return dict(report)

    def _track_actor_connected(self, actor_id: str) -> None:
        aid = str(actor_id or "").strip()
        if not aid:
            return
        with self._actor_connections_lock:
            self._actor_connections[aid] = int(self._actor_connections.get(aid) or 0) + 1

    def _track_actor_disconnected(self, actor_id: str) -> int:
        aid = str(actor_id or "").strip()
        if not aid:
            return 0
        with self._actor_connections_lock:
            current = int(self._actor_connections.get(aid) or 0)
            if current <= 1:
                self._actor_connections.pop(aid, None)
                return 0
            next_count = current - 1
            self._actor_connections[aid] = next_count
            return next_count

    def _should_shutdown_on_owner_disconnect(self) -> bool:
        policy = self.svc.get_lifecycle_policy_effective()
        eff = dict(policy.get("effective") or {})
        return bool(eff.get("owner_disconnect_shutdown", False))

    def _apply_owner_disconnect_policy(self, actor_ids: set[str]) -> bool:
        if not actor_ids:
            return False
        for actor_id in sorted({str(x or "").strip() for x in actor_ids if str(x or "").strip()}):
            remaining = self._track_actor_disconnected(actor_id)
            if remaining > 0:
                continue
            if self.svc.is_actor_exclusive_endpoint_owner(actor_id):
                if self._stop_event is not None:
                    self._stop_event.set()
                return True
        return False

    def _create_operation(self, *, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        op_id = secrets.token_urlsafe(12)
        session_token = str(payload.get("session_token") or "").strip()
        op: Dict[str, Any] = {
            "operation_id": op_id,
            "command": str(command or ""),
            "status": "running",
            "stage": "queued",
            "done": False,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "updated_at": now,
            "result": None,
            "error": None,
            "error_code": None,
            "progress_events": [
                self._operation_event("queued", "queued", "Operation queued", command=str(command or ""))
            ],
            "session_token": session_token or None,
        }
        with self._operations_lock:
            self._operations[op_id] = op
            self._prune_operations_locked()
        return self._operation_public_snapshot(op)

    def _get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        with self._operations_lock:
            op = self._operations.get(str(operation_id or ""))
            if not isinstance(op, dict):
                return None
            return dict(op)

    def _replace_operation(self, op: Dict[str, Any]) -> None:
        op_id = str(op.get("operation_id") or "")
        if not op_id:
            return
        with self._operations_lock:
            self._operations[op_id] = dict(op)
            self._prune_operations_locked()

    async def _run_operation(self, operation_id: str, command: str, payload: Dict[str, Any]) -> None:
        op = self._get_operation(operation_id) or {}
        if not op:
            return
        now = time.time()
        op["started_at"] = now
        op["updated_at"] = now
        op["stage"] = "running"
        events = list(op.get("progress_events") or [])
        events.append(self._operation_event("running", "running", "Operation started"))
        op["progress_events"] = events
        self._replace_operation(op)
        try:
            result = await asyncio.to_thread(self._call_service, command, payload)
            now = time.time()
            op = self._get_operation(operation_id) or op
            op["done"] = True
            op["status"] = "completed"
            op["stage"] = "completed"
            op["result"] = result
            op["updated_at"] = now
            op["completed_at"] = now
            events = list(op.get("progress_events") or [])
            if isinstance(result, dict) and isinstance(result.get("progress_events"), list):
                events.extend(list(result.get("progress_events") or []))
            events.append(self._operation_event("completed", "completed", "Operation completed"))
            op["progress_events"] = events
            self._replace_operation(op)
        except Exception as exc:
            now = time.time()
            op = self._get_operation(operation_id) or op
            op["done"] = True
            op["status"] = "failed"
            op["stage"] = "failed"
            op["error"] = str(exc)
            op["error_code"] = "operation_failed"
            op["updated_at"] = now
            op["completed_at"] = now
            events = list(op.get("progress_events") or [])
            events.append(self._operation_event("failed", "failed", str(exc)))
            op["progress_events"] = events
            self._replace_operation(op)

    async def run(self) -> None:
        """Start local IPC control listener and optionally TCP, then write PID file and run until stop event."""
        self._stop_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        enable_tcp = self._should_enable_tcp()
        try:
            self._start_local_control_listener()
            if enable_tcp:
                self._server = await asyncio.start_server(
                    self._handle_client,
                    "127.0.0.1",
                    self.port,
                    limit=2 ** 20,
                )
                try:
                    sockets = list(getattr(self._server, "sockets", []) or [])
                    if sockets:
                        sockname = sockets[0].getsockname()
                        if isinstance(sockname, tuple) and len(sockname) >= 2:
                            self.port = int(sockname[1] or self.port)
                except Exception:
                    pass
            write_kwargs = {
                "pid": os.getpid(),
                "port": self.port,
                "shutdown_token": self.shutdown_token,
                "transport": str(self._local_transport.get("transport") or ""),
                "ipc_family": str(self._local_transport.get("family") or ""),
                "ipc_address": str(self._local_transport.get("address") or ""),
            }
            try:
                self.pid_file.write(**write_kwargs)
            except TypeError:
                self.pid_file.write(
                    pid=int(write_kwargs["pid"]),
                    port=int(write_kwargs["port"]),
                    shutdown_token=str(write_kwargs["shutdown_token"]),
                )
            logger.info(
                "EngineHostDaemon starting on local IPC %s:%s",
                self._local_transport.get("family"),
                self._local_transport.get("address"),
            )
            if enable_tcp:
                logger.info("EngineHostDaemon starting on 127.0.0.1:%d", self.port)
                async with self._server:
                    await self._stop_event.wait()
            else:
                await self._stop_event.wait()
        finally:
            try:
                self._shutdown_stage_events = []
                self._record_shutdown_stage(
                    "shutdown.begin",
                    "running",
                    "Daemon shutdown sequence started",
                )
                self._record_shutdown_stage(
                    "shutdown.operations_drain",
                    "running",
                    "Draining in-flight operations",
                )
                drain_report = await self._drain_inflight_operations(timeout_seconds=5.0)
                self._record_shutdown_stage(
                    "shutdown.operations_drain",
                    "completed",
                    "In-flight operations drain complete",
                    pending_before=int(drain_report.get("pending_before") or 0),
                    pending_after=int(drain_report.get("pending_after") or 0),
                    timed_out=bool(drain_report.get("timed_out", False)),
                )
                self._record_shutdown_stage(
                    "shutdown.managed_workers",
                    "running",
                    "Running managed worker shutdown checkpoints",
                )
                report = await asyncio.to_thread(self._execute_shutdown_checkpoints)
                report["operation_drain"] = dict(drain_report)
                report["shutdown_stages"] = list(self._shutdown_stage_events)
                self._last_shutdown_checkpoints = dict(report)
                self._record_shutdown_stage(
                    "shutdown.managed_workers",
                    "completed",
                    "Managed worker shutdown checkpoints complete",
                    attempted=int(report.get("attempted") or 0),
                    stopped=int(report.get("stopped") or 0),
                    failed=int(report.get("failed") or 0),
                )
                logger.info(
                    "Daemon shutdown checkpoints: attempted=%s stopped=%s failed=%s",
                    report.get("attempted"),
                    report.get("stopped"),
                    report.get("failed"),
                )
            except Exception as exc:
                logger.warning("Shutdown checkpoints failed: %s", exc)
            self._stop_local_control_listener()
            self.pid_file.remove()
            self._loop = None
            logger.info("EngineHostDaemon stopped")

    async def _dispatch(self, raw_line: str, *, peer_host: Optional[str] = None) -> Dict[str, Any]:
        try:
            req = json.loads(raw_line)
        except Exception:
            return {
                "seq": -1,
                "ok": False,
                "error": "parse_error",
                "error_code": "parse_error",
                "error_details": {},
            }
        seq = int(req.get("seq") or 0)
        cmd = str(req.get("cmd") or "").strip()
        payload = dict(req.get("payload") or {})
        host = str(peer_host or "").strip().lower()
        is_localhost = host in {"", "127.0.0.1", "::1", "localhost"}

        if cmd == "__ping__":
            return {"seq": seq, "ok": True, "result": "pong"}

        if cmd == "__shutdown__":
            if not self._terminal_control_enabled():
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "access_denied",
                    "error_code": "terminal_control_disabled",
                    "error_details": {"command": "__shutdown__"},
                }
            token = str(payload.get("shutdown_token") or "")
            if token and token == self.shutdown_token:
                assert self._stop_event is not None
                self._stop_event.set()
                return {"seq": seq, "ok": True, "result": "shutting_down"}
            return {
                "seq": seq,
                "ok": False,
                "error": "invalid_shutdown_token",
                "error_code": "invalid_shutdown_token",
                "error_details": {},
            }

        if cmd == "set-endpoint-mode-override":
            if not self._terminal_control_enabled():
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "access_denied",
                    "error_code": "terminal_control_disabled",
                    "error_details": {"command": "set-endpoint-mode-override"},
                }
            try:
                self.svc.authorize_command(cmd, payload)
                acl = self.svc.enforce_daemon_claim_policy(
                    cmd,
                    payload,
                    peer_host=peer_host,
                    is_localhost=is_localhost,
                )
                if not bool(acl.get("ok", False)):
                    return {
                        "seq": seq,
                        "ok": False,
                        "error": str(acl.get("error") or "access_denied"),
                        "error_code": str(acl.get("error_code") or "access_denied"),
                        "error_details": dict(acl.get("error_details") or {}),
                    }
                payload = dict(acl.get("payload") or payload)
            except PermissionError as exc:
                code = str(exc or "").strip() or "auth_failed"
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "auth_failed",
                    "error_code": code,
                    "error_details": {"reason": code},
                }
            mode = str(payload.get("mode") or "").strip().lower()
            if mode in {"", "default", "clear", "none"}:
                self._endpoint_mode_runtime_override = None
            elif mode in {"exclusive", "shared"}:
                self._endpoint_mode_runtime_override = mode
            else:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "invalid_mode",
                    "error_code": "invalid_mode",
                    "error_details": {"message": "mode must be exclusive|shared|default"},
                }
            return {"seq": seq, "ok": True, "result": self._effective_endpoint_mode()}

        if cmd == "get-endpoint-mode-effective":
            try:
                self.svc.authorize_command(cmd, payload)
                acl = self.svc.enforce_daemon_claim_policy(
                    cmd,
                    payload,
                    peer_host=peer_host,
                    is_localhost=is_localhost,
                )
                if not bool(acl.get("ok", False)):
                    return {
                        "seq": seq,
                        "ok": False,
                        "error": str(acl.get("error") or "access_denied"),
                        "error_code": str(acl.get("error_code") or "access_denied"),
                        "error_details": dict(acl.get("error_details") or {}),
                    }
            except PermissionError as exc:
                code = str(exc or "").strip() or "auth_failed"
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "auth_failed",
                    "error_code": code,
                    "error_details": {"reason": code},
                }
            return {"seq": seq, "ok": True, "result": self._effective_endpoint_mode()}

        if cmd == "op-start":
            target_cmd = str(payload.get("command") or "").strip()
            target_payload = dict(payload.get("payload") or payload.get("command_payload") or {})
            if not target_cmd:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "command_required",
                    "error_code": "command_required",
                    "error_details": {},
                }
            if target_cmd in {"__ping__", "__shutdown__", "op-start", "op-status"}:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "unsupported_operation_command",
                    "error_code": "unsupported_operation_command",
                    "error_details": {"command": target_cmd},
                }
            try:
                target_payload = self._inject_runtime_endpoint_mode(target_cmd, target_payload)
                self.svc.authorize_command(target_cmd, target_payload)
                acl = self.svc.enforce_daemon_claim_policy(
                    target_cmd,
                    target_payload,
                    peer_host=peer_host,
                    is_localhost=is_localhost,
                )
                if not bool(acl.get("ok", False)):
                    return {
                        "seq": seq,
                        "ok": False,
                        "error": str(acl.get("error") or "access_denied"),
                        "error_code": str(acl.get("error_code") or "access_denied"),
                        "error_details": dict(acl.get("error_details") or {}),
                    }
                target_payload = dict(acl.get("payload") or target_payload)
                op_snapshot = self._create_operation(command=target_cmd, payload=target_payload)
                operation_id = str(op_snapshot.get("operation_id") or "")
                task = asyncio.create_task(self._run_operation(operation_id, target_cmd, target_payload))
                with self._operation_tasks_lock:
                    self._operation_tasks.add(task)
                def _on_done(done_task: asyncio.Task) -> None:
                    with self._operation_tasks_lock:
                        self._operation_tasks.discard(done_task)
                task.add_done_callback(_on_done)
                return {"seq": seq, "ok": True, "result": op_snapshot}
            except PermissionError as exc:
                code = str(exc or "").strip() or "auth_failed"
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "auth_failed",
                    "error_code": code,
                    "error_details": {"reason": code},
                }
            except Exception as exc:
                if hasattr(exc, "to_error_payload"):
                    payload = dict(getattr(exc, "to_error_payload")() or {})
                    return {
                        "seq": seq,
                        "ok": False,
                        "error": str(payload.get("error") or "internal_error"),
                        "error_code": str(payload.get("error_code") or "internal_error"),
                        "error_details": dict(payload.get("error_details") or {}),
                    }
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "internal_error",
                    "error_code": "internal_error",
                    "error_details": {"message": str(exc)},
                }

        if cmd == "op-status":
            op_id = str(payload.get("operation_id") or "").strip()
            if not op_id:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "operation_id_required",
                    "error_code": "operation_id_required",
                    "error_details": {},
                }
            op = self._get_operation(op_id)
            if not op:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "operation_not_found",
                    "error_code": "operation_not_found",
                    "error_details": {"operation_id": op_id},
                }
            required_token = str(op.get("session_token") or "").strip()
            provided_token = str(payload.get("session_token") or "").strip()
            if required_token and required_token != provided_token:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "auth_failed",
                    "error_code": "missing_or_invalid_session_token",
                    "error_details": {"operation_id": op_id},
                }
            return {"seq": seq, "ok": True, "result": self._operation_public_snapshot(op)}

        try:
            payload = self._inject_runtime_endpoint_mode(cmd, payload)
            self.svc.authorize_command(cmd, payload)
            acl = self.svc.enforce_daemon_claim_policy(
                cmd,
                payload,
                peer_host=peer_host,
                is_localhost=is_localhost,
            )
            if not bool(acl.get("ok", False)):
                return {
                    "seq": seq,
                    "ok": False,
                    "error": str(acl.get("error") or "access_denied"),
                    "error_code": str(acl.get("error_code") or "access_denied"),
                    "error_details": dict(acl.get("error_details") or {}),
                }
            payload = dict(acl.get("payload") or payload)
            result = await asyncio.to_thread(self._call_service, cmd, payload)
            if isinstance(result, dict) and str(result.get("status") or "").strip().lower() == "denied":
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "access_denied",
                    "error_code": str(result.get("denied_code") or result.get("denied_reason") or "access_denied"),
                    "error_details": dict(result.get("details") or {}),
                    "result": result,
                }
            return {"seq": seq, "ok": True, "result": result}
        except PermissionError as exc:
            code = str(exc or "").strip() or "auth_failed"
            return {
                "seq": seq,
                "ok": False,
                "error": "auth_failed",
                "error_code": code,
                "error_details": {"reason": code},
            }
        except Exception as exc:
            if hasattr(exc, "to_error_payload"):
                payload = dict(getattr(exc, "to_error_payload")() or {})
                return {
                    "seq": seq,
                    "ok": False,
                    "error": str(payload.get("error") or "internal_error"),
                    "error_code": str(payload.get("error_code") or "internal_error"),
                    "error_details": dict(payload.get("error_details") or {}),
                }
            return {
                "seq": seq,
                "ok": False,
                "error": "internal_error",
                "error_code": "internal_error",
                "error_details": {"message": str(exc)},
            }

    def _call_service(self, cmd: str, payload: Dict[str, Any]) -> Any:
        """Synchronous dispatch to EngineHostService (runs in thread pool)."""
        svc = self.svc
        if cmd == "discover-running":
            return svc.discover_running(
                prune_stale=bool(payload.get("prune_stale", True)),
                include_progress=bool(payload.get("include_progress", False)),
                include_reachability=bool(payload.get("include_reachability", True)),
                reachability_timeout_seconds=float(payload.get("reachability_timeout_seconds") or 0.35),
            )
        if cmd == "spawn":
            return svc.spawn(
                engine_id=str(payload.get("engine_id") or ""),
                command=list(payload.get("command") or []),
                cwd=payload.get("cwd"),
                env=dict(payload.get("env") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}),
                executor_kind=payload.get("executor_kind"),
                bundle=dict(payload.get("bundle") or {}),
                environment=dict(payload.get("environment") or {}),
                tool_access=dict(payload.get("tool_access") or {}),
                capabilities=dict(payload.get("capabilities") or {}),
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
                exclusive=payload.get("exclusive"),
                force_override=bool(payload.get("force_override", False)),
                force_override_reason=payload.get("force_override_reason"),
                force_override_emergency=bool(payload.get("force_override_emergency", False)),
                actor_id=payload.get("_claim_actor_id"),
                peer_host=payload.get("_daemon_peer_host"),
            )
        if cmd == "claim-endpoint":
            return svc.claim_endpoint(
                backend_id=payload.get("backend_id"),
                exclusive=payload.get("exclusive"),
                force_override=bool(payload.get("force_override", False)),
                force_override_reason=payload.get("force_override_reason"),
                force_override_emergency=bool(payload.get("force_override_emergency", False)),
                actor_id=payload.get("_claim_actor_id"),
                peer_host=payload.get("_daemon_peer_host"),
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
                exclusive=payload.get("exclusive"),
                force_override=bool(payload.get("force_override", False)),
                force_override_reason=payload.get("force_override_reason"),
                force_override_emergency=bool(payload.get("force_override_emergency", False)),
                actor_id=payload.get("_claim_actor_id"),
                peer_host=payload.get("_daemon_peer_host"),
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
                "",
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
        if cmd == "sandbox-fs-list":
            return svc.sandbox_fs_list(
                engine_id=str(payload.get("engine_id") or ""),
                root_id=str(payload.get("root_id") or ""),
                relative_path=payload.get("relative_path"),
                callback_context=dict(payload.get("callback_context") or {}) if isinstance(payload.get("callback_context"), dict) else None,
            )
        if cmd == "sandbox-fs-read-text":
            return svc.sandbox_fs_read_text(
                engine_id=str(payload.get("engine_id") or ""),
                root_id=str(payload.get("root_id") or ""),
                relative_path=str(payload.get("relative_path") or ""),
                encoding=str(payload.get("encoding") or "utf-8"),
                callback_context=dict(payload.get("callback_context") or {}) if isinstance(payload.get("callback_context"), dict) else None,
            )
        if cmd == "sandbox-fs-write-text":
            return svc.sandbox_fs_write_text(
                engine_id=str(payload.get("engine_id") or ""),
                root_id=str(payload.get("root_id") or ""),
                relative_path=str(payload.get("relative_path") or ""),
                text=str(payload.get("text") or ""),
                encoding=str(payload.get("encoding") or "utf-8"),
                create_parents=bool(payload.get("create_parents", True)),
                callback_context=dict(payload.get("callback_context") or {}) if isinstance(payload.get("callback_context"), dict) else None,
            )
        if cmd == "sandbox-fs-mkdir":
            return svc.sandbox_fs_mkdir(
                engine_id=str(payload.get("engine_id") or ""),
                root_id=str(payload.get("root_id") or ""),
                relative_path=str(payload.get("relative_path") or ""),
                parents=bool(payload.get("parents", True)),
                exist_ok=bool(payload.get("exist_ok", True)),
                callback_context=dict(payload.get("callback_context") or {}) if isinstance(payload.get("callback_context"), dict) else None,
            )
        if cmd == "sandbox-fs-stat":
            return svc.sandbox_fs_stat(
                engine_id=str(payload.get("engine_id") or ""),
                root_id=str(payload.get("root_id") or ""),
                relative_path=payload.get("relative_path"),
                callback_context=dict(payload.get("callback_context") or {}) if isinstance(payload.get("callback_context"), dict) else None,
            )
        if cmd == "sandbox-http-fetch":
            return svc.sandbox_http_fetch(
                engine_id=str(payload.get("engine_id") or ""),
                url=str(payload.get("url") or ""),
                method=str(payload.get("method") or "GET"),
                headers=dict(payload.get("headers") or {}),
                body_b64=str(payload.get("body_b64") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                max_response_bytes=int(payload.get("max_response_bytes") or 1024 * 1024),
                callback_context=dict(payload.get("callback_context") or {}) if isinstance(payload.get("callback_context"), dict) else None,
            )
        if cmd == "toolbox-describe":
            return svc.toolbox_describe(
                engine_id=str(payload.get("engine_id") or ""),
                toolbox_id=str(payload.get("toolbox_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 10.0),
            )
        if cmd == "toolbox-gate":
            return svc.toolbox_gate(
                engine_id=str(payload.get("engine_id") or ""),
                toolbox_id=str(payload.get("toolbox_id") or ""),
                tool_name=str(payload.get("tool_name") or ""),
                tools_view=dict(payload.get("tools_view") or {}) if isinstance(payload.get("tools_view"), dict) else None,
            )
        if cmd == "toolbox-execute":
            return svc.toolbox_execute(
                engine_id=str(payload.get("engine_id") or ""),
                toolbox_id=str(payload.get("toolbox_id") or ""),
                tool_call=dict(payload.get("tool_call") or {}),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                tools_view=dict(payload.get("tools_view") or {}) if isinstance(payload.get("tools_view"), dict) else None,
                callback_binding=dict(payload.get("callback_binding") or {}) if isinstance(payload.get("callback_binding"), dict) else None,
            )
        if cmd == "toolbox-cancel":
            return svc.toolbox_cancel(
                engine_id=str(payload.get("engine_id") or ""),
                toolbox_id=str(payload.get("toolbox_id") or ""),
                tool_name=str(payload.get("tool_name") or ""),
                tool_call_id=str(payload.get("tool_call_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 8.0),
                respawn=bool(payload.get("respawn", True)),
            )
        if cmd == "toolbox-gc":
            return svc.toolbox_gc()
        if cmd == "toolbox-references":
            return svc.toolbox_references()
        if cmd == "toolbox-consistency":
            return svc.toolbox_consistency()
        if cmd == "toolbox-review-snapshot":
            return svc.toolbox_review_snapshot(
                toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()],
            )
        if cmd == "toolbox-repair":
            return svc.toolbox_repair(
                toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()],
                only_inconsistent=bool(payload.get("only_inconsistent", True)),
                details=bool(payload.get("details", False)),
            )
        if cmd == "toolbox-reconcile":
            return svc.toolbox_reconcile(
                toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()],
                only_inconsistent=bool(payload.get("only_inconsistent", True)),
                details=bool(payload.get("details", False)),
            )
        if cmd == "toolbox-register-auto":
            return svc.toolbox_register_auto(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                requests=[dict(item or {}) for item in list(payload.get("requests") or [])],
                python_executable=str(payload.get("python_executable") or "").strip() or None,
                worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
            )
        if cmd == "toolbox-unregister-auto":
            return svc.toolbox_unregister_auto(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()],
                python_executable=str(payload.get("python_executable") or "").strip() or None,
                worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
            )
        if cmd == "toolbox-register-intrinsics":
            return svc.toolbox_register_intrinsics(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                intrinsic_tool_names=[str(item or "").strip() for item in list(payload.get("intrinsic_tool_names") or []) if str(item or "").strip()],
                include_guides=bool(payload.get("include_guides", False)),
                sandbox_profile=dict(payload.get("sandbox_profile") or {}) or None,
                python_executable=str(payload.get("python_executable") or "").strip() or None,
                worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
            )
        if cmd == "toolbox-unregister-intrinsics":
            return svc.toolbox_unregister_intrinsics(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                intrinsic_tool_names=[str(item or "").strip() for item in list(payload.get("intrinsic_tool_names") or []) if str(item or "").strip()],
                include_guides=bool(payload.get("include_guides", False)),
                python_executable=str(payload.get("python_executable") or "").strip() or None,
                worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
            )
        if cmd == "toolbox-register-manual":
            return svc.toolbox_register_manual(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                requests=[dict(item or {}) for item in list(payload.get("requests") or [])],
                python_executable=str(payload.get("python_executable") or "").strip() or None,
                worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
            )
        if cmd == "toolbox-unregister-manual":
            return svc.toolbox_unregister_manual(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()],
                python_executable=str(payload.get("python_executable") or "").strip() or None,
                worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
            )
        if cmd == "toolbox-environment-list":
            return svc.toolbox_environment_description_list()
        if cmd == "toolbox-environment-upsert":
            return svc.toolbox_environment_description_upsert(
                name=str(payload.get("name") or ""),
                base_env_name=str(payload.get("base_env_name") or "").strip() or None,
                extra_packages=[str(item or "").strip() for item in list(payload.get("extra_packages") or []) if str(item or "").strip()],
                allow_online_install=bool(payload.get("allow_online_install", False)),
            )
        if cmd == "toolbox-environment-clone":
            return svc.toolbox_environment_description_clone(
                source_name=str(payload.get("source_name") or ""),
                target_name=str(payload.get("target_name") or ""),
                extra_packages=[str(item or "").strip() for item in list(payload.get("extra_packages") or []) if str(item or "").strip()] if payload.get("extra_packages") is not None else None,
                allow_online_install=payload.get("allow_online_install"),
            )
        if cmd == "toolbox-environment-resolve":
            return svc.toolbox_environment_resolve_requirements(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                environment_name=str(payload.get("environment_name") or "base"),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
            )
        if cmd == "toolbox-environment-apply":
            return svc.toolbox_environment_apply(
                environment_name=str(payload.get("environment_name") or "base"),
                toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()] or None,
            )
        if cmd == "toolbox-environment-realize":
            return svc.toolbox_environment_realize(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                environment_name=str(payload.get("environment_name") or "base"),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
            )
        if cmd == "toolbox-environment-sync":
            return svc.toolbox_environment_sync_description(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                source_environment_name=str(payload.get("source_environment_name") or "base"),
                target_environment_name=str(payload.get("target_environment_name") or "").strip() or None,
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                apply=bool(payload.get("apply", False)),
                realize=bool(payload.get("realize", False)),
            )
        if cmd == "toolbox-environment-prepare-install":
            return svc.toolbox_environment_prepare_install(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                environment_name=str(payload.get("environment_name") or "base"),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
            )
        if cmd == "toolbox-environment-lock-install":
            return svc.toolbox_environment_lock_install(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                environment_name=str(payload.get("environment_name") or "base"),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
            )
        if cmd == "toolbox-environment-resolve-install-lock":
            return svc.toolbox_environment_resolve_install_lock(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                environment_name=str(payload.get("environment_name") or "base"),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                allow_resolution=bool(payload.get("allow_resolution", False)),
            )
        if cmd == "toolbox-environment-verify-install-lock":
            return svc.toolbox_environment_verify_install_lock(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                environment_name=str(payload.get("environment_name") or "base"),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
            )
        if cmd == "toolbox-environment-verify-install-receipt":
            return svc.toolbox_environment_verify_install_receipt(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                environment_name=str(payload.get("environment_name") or "base"),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
            )
        if cmd == "toolbox-environment-execute-install":
            return svc.toolbox_environment_execute_install(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                environment_name=str(payload.get("environment_name") or "base"),
                tool_keys=[str(item or "").strip() for item in list(payload.get("tool_keys") or []) if str(item or "").strip()] or None,
                allow_execution=bool(payload.get("allow_execution", False)),
            )
        if cmd == "proxy-request":
            return svc.proxy_request(
                engine_id=str(payload.get("engine_id") or ""),
                method=str(payload.get("method") or "GET"),
                path=str(payload.get("path") or "/"),
                query=str(payload.get("query") or ""),
                headers=dict(payload.get("headers") or {}),
                body_b64=str(payload.get("body_b64") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                max_response_bytes=int(payload.get("max_response_bytes") or 1024 * 1024),
            )
        if cmd == "proxy-rpc-call":
            return svc.proxy_rpc_call(
                engine_id=str(payload.get("engine_id") or ""),
                method=str(payload.get("method") or ""),
                params=dict(payload.get("params") or {}),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
            )
        if cmd == "proxy-rpc-open":
            return svc.proxy_rpc_open(
                engine_id=str(payload.get("engine_id") or ""),
                method=str(payload.get("method") or ""),
                params=dict(payload.get("params") or {}),
                request_id=str(payload.get("request_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
            )
        if cmd == "proxy-rpc-send":
            return svc.proxy_rpc_send(
                engine_id=str(payload.get("engine_id") or ""),
                stream_id=str(payload.get("stream_id") or ""),
                message=dict(payload.get("message") or {}),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
            )
        if cmd == "proxy-rpc-recv":
            return svc.proxy_rpc_recv(
                engine_id=str(payload.get("engine_id") or ""),
                stream_id=str(payload.get("stream_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 2.0),
                max_items=int(payload.get("max_items") or 64),
            )
        if cmd == "proxy-rpc-close":
            return svc.proxy_rpc_close(
                engine_id=str(payload.get("engine_id") or ""),
                stream_id=str(payload.get("stream_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 10.0),
            )
        if cmd == "proxy-stream-open":
            return svc.proxy_stream_open(
                engine_id=str(payload.get("engine_id") or ""),
                tool=str(payload.get("tool") or "run-inference"),
                arguments=dict(payload.get("arguments") or {}),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
            )
        if cmd == "proxy-stream-send":
            return svc.proxy_stream_send(
                engine_id=str(payload.get("engine_id") or ""),
                stream_id=str(payload.get("stream_id") or ""),
                message=dict(payload.get("message") or {}),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
            )
        if cmd == "proxy-stream-recv":
            return svc.proxy_stream_recv(
                engine_id=str(payload.get("engine_id") or ""),
                stream_id=str(payload.get("stream_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 2.0),
                max_items=int(payload.get("max_items") or 64),
            )
        if cmd == "proxy-stream-close":
            return svc.proxy_stream_close(
                engine_id=str(payload.get("engine_id") or ""),
                stream_id=str(payload.get("stream_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 10.0),
            )
        if cmd == "get-control-config":
            return svc.get_control_config()
        if cmd == "set-control-config":
            return svc.set_control_config(
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
        if cmd == "get-lifecycle-policy-effective":
            return svc.get_lifecycle_policy_effective()
        if cmd == "auth-status":
            return svc.auth_status()
        if cmd == "auth-list-keys":
            return svc.auth_list_keys()
        if cmd == "auth-list-sessions":
            return svc.auth_list_sessions(
                key_id=payload.get("key_id"),
                scope=payload.get("scope"),
                role=payload.get("role"),
                token_preview_contains=payload.get("token_preview_contains"),
                limit=int(payload.get("limit") or 100),
                offset=int(payload.get("offset") or 0),
            )
        if cmd == "auth-list-issued-tokens":
            return svc.auth_list_issued_tokens(
                engine_id=payload.get("engine_id"),
                resource_kind=payload.get("resource_kind"),
                resource_id=payload.get("resource_id"),
                backend_id=payload.get("backend_id"),
                token_preview_contains=payload.get("token_preview_contains"),
                limit=int(payload.get("limit") or 100),
                offset=int(payload.get("offset") or 0),
            )
        if cmd == "auth-audit-list":
            return svc.auth_list_audit_events(
                event_type=payload.get("event_type"),
                actor_key_id=payload.get("actor_key_id"),
                target_key_id=payload.get("target_key_id"),
                result=payload.get("result"),
                limit=int(payload.get("limit") or 100),
                offset=int(payload.get("offset") or 0),
            )
        if cmd == "auth-upsert-key":
            return svc.auth_upsert_key(
                key_id=str(payload.get("key_id") or ""),
                key_secret=str(payload.get("key_secret") or ""),
                role=str(payload.get("role") or ""),
                auth_method=str(payload.get("auth_method") or "shared_secret"),
                public_key=str(payload.get("public_key") or ""),
                allowed_configs=list(payload.get("allowed_configs") or []),
                allowed_engines=list(payload.get("allowed_engines") or []),
                disabled=bool(payload.get("disabled", False)),
            )
        if cmd == "auth-revoke-key":
            return svc.auth_revoke_key(str(payload.get("key_id") or ""))
        if cmd == "auth-issue-session":
            return svc.auth_issue_session(
                key_id=str(payload.get("key_id") or ""),
                key_secret=str(payload.get("key_secret") or ""),
                scope=str(payload.get("scope") or "control"),
                ttl_seconds=int(payload.get("ttl_seconds") or 900),
                config_paths=list(payload.get("config_paths") or []),
                engine_ids=list(payload.get("engine_ids") or []),
                ssh_binding=dict(payload.get("ssh_binding") or {}),
            )
        if cmd == "auth-begin-challenge":
            return svc.auth_begin_challenge(
                key_id=str(payload.get("key_id") or ""),
                scope=str(payload.get("scope") or "control"),
                ttl_seconds=int(payload.get("ttl_seconds") or 120),
                config_paths=list(payload.get("config_paths") or []),
                engine_ids=list(payload.get("engine_ids") or []),
                ssh_binding=dict(payload.get("ssh_binding") or {}),
            )
        if cmd == "auth-complete-challenge":
            return svc.auth_complete_challenge(
                challenge_id=str(payload.get("challenge_id") or ""),
                signature_ssh=str(payload.get("signature_ssh") or ""),
                presented_ssh_binding=dict(payload.get("_ssh_session_binding") or {}),
            )
        if cmd == "auth-revoke-session":
            return svc.auth_revoke_session(str(payload.get("token") or ""))
        if cmd == "host-metrics":
            return svc.get_host_metrics()
        raise ValueError(f"Unknown command '{cmd}'")
