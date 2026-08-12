"""Local IPC engine host daemon runtime."""
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import socket
import struct
import sys
import threading
import time
from multiprocessing.connection import Client as MPClient
from multiprocessing.connection import Listener as MPListener
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..sandbox.host_capabilities import (
    CapabilityAuthorityLease,
    HostCapabilityDescriptor,
    HostCapabilityMethod,
    HostCapabilityProviderRef,
    HostCapabilitySession,
)
from ..service.host_service import EngineHostService
from ..hosting_configuration import load_hosting_configuration
from .constants import DEFAULT_DAEMON_PORT
from .diagnostics import write_daemon_report
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
        mp13_config_file: Optional[Path] = None,
        runtime_profile: str = "foreground_terminal_bound",
    ):
        hosting_configuration = load_hosting_configuration(mp13_config_file)
        self.port = int(port or DEFAULT_DAEMON_PORT)
        self.pid_file = DaemonPidFile(pid_file)
        self.shutdown_token = secrets.token_urlsafe(24)
        local_transport = _daemon_local_ipc_endpoint(self.pid_file.path)
        self._local_transport = dict(local_transport)
        self.svc = EngineHostService(
            engines_state_file=engines_state_file,
            hosting_configuration=hosting_configuration,
        )
        self.svc._toolbox_setup_diagnostic = {  # noqa: SLF001
            "code": "environment_template_missing",
            "summary": "No environment template has been activated.",
        }
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
        self._operations_state_file = (self.svc.hosting_root / "state" / "operations.json").expanduser().resolve()
        self._operations_journal_file = (self.svc.hosting_root / "state" / "operation_audit.jsonl").expanduser().resolve()
        self._operations = self._load_persisted_operations()
        self._operation_tasks: set[asyncio.Task] = set()
        self._operation_tasks_by_id: Dict[str, asyncio.Task] = {}
        self._operation_tasks_lock = threading.Lock()
        self._endpoint_mode_runtime_override: Optional[str] = None
        self._runtime_profile = str(runtime_profile or "foreground_terminal_bound").strip().lower()
        self._started_at: Optional[float] = None
        self._started_monotonic: Optional[float] = None
        self._actor_connections: Dict[str, int] = {}
        self._actor_connections_lock = threading.Lock()
        self._live_connections: Dict[str, Dict[str, Any]] = {}
        self._host_capability_sessions: Dict[str, HostCapabilitySession] = {}
        self._host_capability_sessions_lock = threading.RLock()
        self._last_shutdown_checkpoints: Dict[str, Any] = {}
        self._shutdown_stage_events: List[Dict[str, Any]] = []
        self._shutdown_report: Dict[str, Any] = {
            "reason": "daemon_run_exited",
            "actor": {},
            "details": {},
        }

    def _serve_local_control_client(self, conn: Any) -> None:
        connection_id = secrets.token_urlsafe(9)
        client_pid = self._local_connection_peer_pid(conn)
        process_info = self._process_info_for_pid(client_pid)
        connection_actor_ids: set[str] = set()
        self._register_live_connection(
            connection_id,
            transport="local_ipc",
            peer_host="127.0.0.1",
            pid=client_pid,
            process_info=process_info,
        )
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
                    if actor_id:
                        self._update_live_connection(
                            connection_id,
                            command=str(req_obj.get("cmd") or ""),
                            actor_id=actor_id,
                            session_token=tok,
                        )
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
                        self._dispatch(
                            raw,
                            peer_host="127.0.0.1",
                            peer_pid=client_pid,
                            peer_process_info=process_info,
                            transport="local_ipc",
                        ),
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
            self._unregister_live_connection(connection_id)

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
        # Full control over loopback TCP / SSH port forwarding is not a
        # supported transport yet. Keep the daemon server-side blocked even
        # when remote-capable auth roles exist; see hosting docs for the TBD
        # straight SSH port-forwarding feature.
        return False

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.debug("Client connected: %s", peer)
        connection_id = secrets.token_urlsafe(9)
        peer_host = ""
        try:
            if isinstance(peer, tuple) and len(peer) >= 1:
                peer_host = str(peer[0] or "")
        except Exception:
            peer_host = ""
        self._register_live_connection(
            connection_id,
            transport="tcp",
            peer_host=peer_host,
            pid=None,
            process_info={},
        )
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
                        if actor_id:
                            self._update_live_connection(
                                connection_id,
                                command=str((req_obj or {}).get("cmd") or ""),
                                actor_id=actor_id,
                                session_token=tok,
                            )
                except Exception:
                    pass
                response = await self._dispatch(raw, peer_host=peer_host, transport="tcp")
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
            self._unregister_live_connection(connection_id)

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
    def _operation_payload_hint(command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        cmd = str(command or "").strip()
        p = dict(payload or {})
        allowed = {
            "connect-from-config": [
                "config_path",
                "engine_id",
                "model_path",
                "force_new_worker",
                "launch_policy",
                "target_worker_id",
            ],
            "spawn": [
                "engine_id",
                "cwd",
                "worker_ipc_family",
                "worker_profile_class",
            ],
            "unload-model": [
                "engine_id",
                "shutdown_all",
            ],
            "shutdown": [
                "engine_id",
            ],
            "remove-registration": [
                "engine_id",
            ],
            "prune-stale-registration": [
                "engine_ids",
                "reason",
                "trigger_command",
            ],
        }.get(cmd, [])
        return {key: p.get(key) for key in allowed if key in p}

    def _load_persisted_operations(self) -> Dict[str, Dict[str, Any]]:
        path = self._operations_state_file
        try:
            raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            rows = raw.get("operations") if isinstance(raw, dict) else raw
            out: Dict[str, Dict[str, Any]] = {}
            for row in rows if isinstance(rows, list) else []:
                op = dict(row or {})
                op_id = str(op.get("operation_id") or "").strip()
                if op_id:
                    out[op_id] = op
            return out
        except Exception:
            return {}

    def _persist_operations_locked(self) -> None:
        path = self._operations_state_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                self._operation_public_snapshot(op)
                for op in sorted(
                    self._operations.values(),
                    key=lambda item: float((item or {}).get("updated_at") or (item or {}).get("created_at") or 0.0),
                    reverse=True,
                )
            ][: self._operations_max_entries]
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps({"version": 1, "updated_at": time.time(), "operations": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            logger.debug("Failed to persist host operations", exc_info=True)

    def _append_operation_journal(self, op: Dict[str, Any], *, event: str) -> None:
        try:
            path = self._operations_journal_file
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "event": str(event or "operation_update"),
                "timestamp": time.time(),
                "operation": self._operation_public_snapshot(op),
            }
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to append host operation journal", exc_info=True)

    @staticmethod
    def _operation_target_engine_id(command: str, payload: Dict[str, Any]) -> Optional[str]:
        cmd = str(command or "").strip()
        if cmd not in {"connect-from-config", "spawn", "unload-model", "shutdown", "remove-registration"}:
            return None
        engine_id = str((payload or {}).get("engine_id") or "").strip()
        return engine_id or None

    def _engine_registry_by_id(self) -> Dict[str, Dict[str, Any]]:
        try:
            return {
                str((row or {}).get("engine_id") or "").strip(): dict(row or {})
                for row in list(self.svc._read_engines() or [])
                if isinstance(row, dict) and str((row or {}).get("engine_id") or "").strip()
            }
        except Exception:
            return {}

    def _record_synchronous_operation(
        self,
        *,
        command: str,
        payload: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        op = dict(self._create_operation(command=command, payload=payload))
        now = time.time()
        result_status = str((result or {}).get("status") or "").strip().lower() if isinstance(result, dict) else ""
        failed = bool(error) or result_status in {"failed", "error"}
        failure_message = str(
            error
            or ((result or {}).get("message") if isinstance(result, dict) else "")
            or ((result or {}).get("reason") if isinstance(result, dict) else "")
            or "Operation failed"
        )
        events = list(op.get("progress_events") or [])
        events.append(self._operation_event("running", "running", "Operation started"))
        if failed:
            events.append(self._operation_event("failed", "failed", failure_message))
        else:
            events.append(self._operation_event("completed", "completed", "Operation completed"))
        if isinstance(result, dict):
            result_engine_id = str(result.get("engine_id") or result.get("worker_id") or result.get("model_instance_id") or "").strip()
            if result_engine_id:
                op["target_engine_id"] = result_engine_id
        op.update(
            {
                "status": "failed" if failed else "completed",
                "stage": "failed" if failed else "completed",
                "done": True,
                "started_at": now,
                "completed_at": now,
                "updated_at": now,
                "result": dict(result or {}) if isinstance(result, dict) else result,
                "error": failure_message if failed else None,
                "error_code": str(
                    error_code
                    or ((result or {}).get("reason") if isinstance(result, dict) else "")
                    or "operation_failed"
                )
                if failed
                else None,
                "progress_events": events,
            }
        )
        self._replace_operation(op)
        return self._operation_public_snapshot(op)

    @staticmethod
    def _log_size(path: Path) -> int:
        try:
            return max(0, int(path.stat().st_size))
        except Exception:
            return 0

    def _read_tail_text(self, path: Path, *, max_bytes: int = 262144, start_offset: int = 0) -> str:
        try:
            size = path.stat().st_size
            with path.open("rb") as fp:
                start = max(0, min(int(start_offset or 0), int(size or 0)))
                if size - start > max_bytes:
                    start = max(start, size - max_bytes)
                fp.seek(start)
                return fp.read(max_bytes).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _parse_model_load_progress_from_log(self, log_path: str, *, start_offset: int = 0) -> Dict[str, Any]:
        path = Path(str(log_path or "")).expanduser()
        if not str(log_path or "").strip() or not path.exists():
            return {}
        text = self._read_tail_text(path, start_offset=start_offset)
        if not text:
            return {"log_path": str(path), "exists": True, "start_offset": max(0, int(start_offset or 0))}
        matches = list(
            re.finditer(
                r"(?im)(?:loading|checkpoint|shard|weight|model)[^\r\n%]{0,120}?(\d{1,3})\s*%",
                text,
            )
        )
        if not matches:
            matches = list(re.finditer(r"(?m)(\d{1,3})\s*%\|", text))
        parsed_error = self._parse_worker_log_error(text)
        if not matches:
            out = {"log_path": str(path), "exists": True, "start_offset": max(0, int(start_offset or 0))}
            if parsed_error:
                out.update(parsed_error)
            return out
        percent = max(0, min(100, int(matches[-1].group(1))))
        out = {
            "log_path": str(path),
            "exists": True,
            "start_offset": max(0, int(start_offset or 0)),
            "progress_kind": "model_weights",
            "progress_percent": percent,
            "progress_text": (
                "Model weights loaded; waiting for worker RPC readiness."
                if percent >= 100
                else f"Loading model weights ({percent}%)."
            ),
        }
        if parsed_error:
            out.update(parsed_error)
        return out

    @staticmethod
    def _parse_worker_log_error(text: str) -> Dict[str, Any]:
        lines = [line.strip() for line in str(text or "").replace("\r", "\n").splitlines() if line.strip()]
        if not lines:
            return {}
        markers = (
            "Global Engine Initialization Failed",
            "Initializing engine failed",
            "Traceback (most recent call last)",
            "RepositoryNotFoundError",
            "OSError:",
            "RuntimeError:",
            "EngineInitializationError",
        )
        marker_indexes = [
            idx for idx, line in enumerate(lines)
            if any(marker in line for marker in markers)
        ]
        if not marker_indexes:
            return {}
        preferred = [
            idx for idx in marker_indexes
            if "Traceback (most recent call last)" not in lines[idx]
        ]
        start = (preferred or marker_indexes)[-1]
        excerpt = lines[start:start + 8]
        message = excerpt[0]
        for line in excerpt:
            if "Global Engine Initialization Failed" in line or line.startswith(("OSError:", "RuntimeError:", "EngineInitializationError")):
                message = line
                break
        return {
            "log_error": {
                "message": message,
                "excerpt": excerpt,
            },
            "progress_error": message,
        }

    def _connect_progress_registration(self, op: Dict[str, Any]) -> Dict[str, Any]:
        hint = dict((op or {}).get("payload_hint") or {})
        config_hint = str(hint.get("config_path") or "").strip()
        model_hint = str(hint.get("model_path") or "").strip()
        canonical_config = ""
        canonical_model = ""
        try:
            if config_hint:
                selected = self.svc._resolve_json_config_path(config_hint)
                canonical_config = self.svc._canonical_path_value(str(selected))
                if model_hint:
                    cfg = self.svc._merge_default_and_selected_config(config_hint)
                    canonical_model = self.svc._canonical_path_value(
                        self.svc._resolve_model_path_from_config_value(model_hint, config_path=config_hint, cfg=cfg)
                    )
        except Exception:
            canonical_config = str(Path(config_hint).expanduser().resolve()) if config_hint else ""
            canonical_model = str(Path(model_hint).expanduser().resolve()) if model_hint else ""
        target_engine = str((op or {}).get("target_engine_id") or "").strip()
        best: Dict[str, Any] = {}
        for row in list(self.svc._read_engines() or []):
            reg = dict(row or {})
            engine_ids = {
                str(reg.get("engine_id") or "").strip(),
                str(reg.get("worker_id") or "").strip(),
                str(reg.get("model_instance_id") or "").strip(),
            }
            if target_engine and target_engine in engine_ids:
                return reg
            if canonical_config and str(reg.get("canonical_config_path") or "").strip() == canonical_config:
                best = reg
            if canonical_model and str(reg.get("canonical_model_path") or "").strip() == canonical_model:
                best = reg
            for binding in list(reg.get("config_bindings") or []):
                if canonical_config and str((binding or {}).get("canonical_config_path") or "").strip() == canonical_config:
                    best = reg
            for model in list(reg.get("loaded_models") or []):
                if canonical_model and str((model or {}).get("canonical_model_path") or "").strip() == canonical_model:
                    best = reg
        return best

    def _enrich_operation_progress(self, op: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(op or {})
        if bool(out.get("done", False)) or str(out.get("command") or "").strip() != "connect-from-config":
            return out
        diagnostics = dict(out.get("diagnostics") or {})
        if not str(diagnostics.get("log_path") or "").strip():
            reg = self._connect_progress_registration(out)
            log_path_from_reg = str(reg.get("log_path") or "").strip()
            if log_path_from_reg:
                diagnostics["log_path"] = log_path_from_reg
                diagnostics.setdefault("log_start_offset", 0)
                target = str(reg.get("engine_id") or reg.get("model_instance_id") or reg.get("worker_id") or "").strip()
                if target and not str(out.get("target_engine_id") or "").strip():
                    out["target_engine_id"] = target
        log_path = str(diagnostics.get("log_path") or "").strip()
        has_log_start_offset = "log_start_offset" in diagnostics
        log_start_offset = int(diagnostics.get("log_start_offset") or 0)
        parsed = (
            self._parse_model_load_progress_from_log(log_path, start_offset=log_start_offset)
            if log_path and has_log_start_offset
            else {}
        )
        if parsed:
            diagnostics["worker_log"] = parsed
            if parsed.get("log_error"):
                diagnostics["worker_log_error"] = parsed.get("log_error")
                out["progress_error"] = str(parsed.get("progress_error") or "")
            if log_path:
                diagnostics["log_path"] = log_path
            out["diagnostics"] = diagnostics
        percent = parsed.get("progress_percent")
        if isinstance(percent, int):
            current = int(out.get("progress_percent") or 0)
            if percent < current:
                percent = current
                parsed["progress_percent"] = percent
                parsed["progress_text"] = f"Loading model weights ({percent}%)."
            out["progress_percent"] = percent
            out["progress_text"] = str(parsed.get("progress_text") or "").strip()
            out.pop("progress_estimated", None)
            events = list(out.get("progress_events") or [])
            stage = "connect.load_weights" if percent < 100 else "connect.worker_ready"
            message = str(parsed.get("progress_text") or "").strip()
            existing_sig = None
            for event in reversed(events):
                if str((event or {}).get("stage") or "") in {"connect.load_weights", "connect.worker_ready"} and "progress_percent" in dict(event or {}):
                    existing_sig = (str((event or {}).get("stage") or ""), int((event or {}).get("progress_percent") or -1))
                    break
            next_sig = (stage, percent)
            if existing_sig != next_sig:
                events.append(self._operation_event(stage, "running", message, progress_percent=percent, log_path=log_path or None))
                out["progress_events"] = events
                out["updated_at"] = time.time()
            return out
        current = int(out.get("progress_percent") or 0)
        started_at = float(diagnostics.get("worker_ready_started_at") or out.get("started_at") or out.get("created_at") or time.time())
        elapsed = max(0.0, time.time() - started_at)
        estimated = min(10, max(current, int(elapsed * 2.0)))
        if estimated != current or out.get("progress_percent") is None:
            out["progress_percent"] = estimated
            out["progress_text"] = "Loading model and waiting for worker RPC readiness"
            out["progress_estimated"] = True
            out["updated_at"] = time.time()
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

    def _shutdown_progress_snapshot(self, *, stage: str, status: str, message: str, **extra: Any) -> Dict[str, Any]:
        now = time.time()
        read_pid_file = getattr(self.pid_file, "read", None)
        pid_info = dict(read_pid_file() or {}) if callable(read_pid_file) else {}
        requested_at = float(pid_info.get("shutdown_requested_at") or 0.0)
        progress: Dict[str, Any] = {
            "stage": str(stage or "unknown"),
            "status": str(status or "info"),
            "message": str(message or ""),
            "timestamp": now,
            "pid": os.getpid(),
            "pid_file": str(getattr(self.pid_file, "path", self.pid_file)),
            "runtime_profile": str(self._runtime_profile or ""),
            "local_transport": dict(self._local_transport or {}),
            "shutdown_requested_at": requested_at or None,
            "shutdown_age_seconds": max(0.0, now - requested_at) if requested_at else None,
            "shutdown_reason": str(pid_info.get("shutdown_reason") or self._shutdown_report.get("reason") or ""),
            "shutdown_requested_by": str(
                pid_info.get("shutdown_requested_by")
                or dict(self._shutdown_report.get("actor") or {}).get("requested_by")
                or ""
            ),
            "shutdown_stages": list(self._shutdown_stage_events),
            "last_shutdown_checkpoints": dict(self._last_shutdown_checkpoints or {}),
        }
        if extra:
            progress.update({str(k): v for k, v in extra.items()})
        return progress

    def _publish_shutdown_progress(self, stage: str, status: str, message: str, **extra: Any) -> None:
        self._record_shutdown_stage(stage, status, message, **extra)
        progress = self._shutdown_progress_snapshot(stage=stage, status=status, message=message, **extra)
        try:
            update_progress = getattr(self.pid_file, "update_shutdown_progress", None)
            if callable(update_progress):
                update_progress(progress)
        except Exception:
            logger.debug("Failed to update daemon shutdown progress in pid file", exc_info=True)
        try:
            write_daemon_report(
                event="daemon_shutdown_progress",
                reason=str(stage or "shutdown_progress"),
                actor=dict(self._shutdown_report.get("actor") or {}),
                details={"shutdown_progress": progress},
                path=self.svc.hosting_root / "logs" / "daemon-crash.log",
            )
        except Exception:
            logger.debug("Failed to write daemon shutdown progress report", exc_info=True)

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
            self._operation_tasks_by_id = {
                op_id: t
                for op_id, t in self._operation_tasks_by_id.items()
                if t is not None and not t.done()
            }
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

    def _execute_startup_worker_recovery(self) -> Dict[str, Any]:
        current_pid = os.getpid()
        report: Dict[str, Any] = {
            "status": "ok",
            "current_host_pid": current_pid,
            "registrations_before": 0,
            "foreign_attempted": 0,
            "foreign_stopped": 0,
            "foreign_failed": 0,
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
                reg = dict(row or {})
                owner_pid = int(reg.get("owner_host_pid") or 0)
                engine_id = str(reg.get("engine_id") or "").strip()
                if not engine_id or owner_pid <= 0 or owner_pid == current_pid:
                    continue
                report["foreign_attempted"] = int(report.get("foreign_attempted") or 0) + 1
                try:
                    out = self.svc.shutdown(engine_id, timeout_seconds=3.0)
                    status = str((out or {}).get("status") or "").strip()
                    ok = status in {"stopped", "already_stopped", "not_found", "invalid_pid"}
                    report["foreign_stopped" if ok else "foreign_failed"] = int(
                        report.get("foreign_stopped" if ok else "foreign_failed") or 0
                    ) + 1
                    report["results"].append(
                        {
                            "engine_id": engine_id,
                            "owner_host_pid": owner_pid,
                            "status": status,
                            "ok": ok,
                            "pid": int(reg.get("pid") or 0),
                        }
                    )
                except Exception as exc:
                    report["foreign_failed"] = int(report.get("foreign_failed") or 0) + 1
                    report["results"].append(
                        {
                            "engine_id": engine_id,
                            "owner_host_pid": owner_pid,
                            "status": "exception",
                            "ok": False,
                            "error": str(exc),
                            "pid": int(reg.get("pid") or 0),
                        }
                    )
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = str(exc)
        return report

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

    @staticmethod
    def _local_connection_peer_pid(conn: Any) -> Optional[int]:
        handle = getattr(conn, "_handle", None)
        if handle is None:
            return None
        if sys.platform.startswith("win"):
            try:
                pid = ctypes.c_ulong(0)
                ok = ctypes.windll.kernel32.GetNamedPipeClientProcessId(  # type: ignore[attr-defined]
                    ctypes.c_void_p(int(handle)),
                    ctypes.byref(pid),
                )
                if ok:
                    return int(pid.value)
            except Exception:
                return None
            return None
        try:
            fd = int(handle)
            sock = socket.fromfd(fd, socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                if hasattr(socket, "SO_PEERCRED"):
                    creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
                    pid, _uid, _gid = struct.unpack("3i", creds)
                    return int(pid)
            finally:
                sock.close()
        except Exception:
            return None
        return None

    @staticmethod
    def _classify_consumer_process(command_line: str) -> str:
        cmd = str(command_line or "").lower()
        if "engine_host_cli" in cmd and "--relay-wrapper" in cmd:
            return "ssh_relay_proxy"
        if "ssh" in cmd and ("engine_host_cli" in cmd or "--relay-wrapper" in cmd):
            return "ssh_proxy"
        if "hosting_cli.py" in cmd and "--interactive" in cmd:
            return "interactive_cli"
        if "engine_host_cli_interactive" in cmd:
            return "interactive_cli"
        return "consumer"

    @staticmethod
    def _process_info_for_pid(pid: Optional[int]) -> Dict[str, Any]:
        target = int(pid or 0)
        if target <= 0:
            return {}
        if sys.platform.startswith("win"):
            try:
                from ctypes import wintypes

                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel32.OpenProcess.restype = wintypes.HANDLE
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle.restype = wintypes.BOOL
                kernel32.QueryFullProcessImageNameW.argtypes = [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    wintypes.LPWSTR,
                    ctypes.POINTER(wintypes.DWORD),
                ]
                kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, target)
                if not handle:
                    return {"pid": target, "consumer_kind": "consumer"}
                try:
                    size = wintypes.DWORD(32768)
                    buf = ctypes.create_unicode_buffer(size.value)
                    image_path = None
                    if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                        image_path = str(buf.value or "") or None
                    name = Path(image_path).name if image_path else None
                    # The stdlib has no stable Windows command-line/parent-PID API.
                    # Report PID and image path only instead of launching PowerShell/WMI.
                    command_line = image_path or ""
                    return {
                        "pid": target,
                        "parent_pid": None,
                        "name": name,
                        "image_path": image_path,
                        "command_line": command_line or None,
                        "consumer_kind": EngineHostDaemon._classify_consumer_process(command_line),
                    }
                finally:
                    kernel32.CloseHandle(handle)
            except Exception:
                return {"pid": target, "consumer_kind": "consumer"}
        if sys.platform.startswith("linux"):
            try:
                proc_root = Path("/proc") / str(target)
                command_line = ""
                try:
                    raw_cmd = (proc_root / "cmdline").read_bytes()
                    command_line = raw_cmd.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
                except Exception:
                    command_line = ""
                name = None
                try:
                    name = (proc_root / "comm").read_text(encoding="utf-8", errors="replace").strip() or None
                except Exception:
                    name = None
                parent_pid = None
                try:
                    stat_text = (proc_root / "stat").read_text(encoding="utf-8", errors="replace")
                    stat_end = stat_text.rfind(")")
                    fields = stat_text[stat_end + 2 :].split()
                    parent_pid = int(fields[1]) if len(fields) > 1 else None
                except Exception:
                    parent_pid = None
                return {
                    "pid": target,
                    "parent_pid": parent_pid,
                    "name": name,
                    "command_line": command_line or None,
                    "consumer_kind": EngineHostDaemon._classify_consumer_process(command_line),
                }
            except Exception:
                return {"pid": target, "consumer_kind": "consumer"}
        # macOS does not provide stdlib-only peer process details beyond the PID
        # here; keep process metadata N/A rather than launching ps.
        return {
            "pid": target,
            "parent_pid": None,
            "name": None,
            "command_line": None,
            "consumer_kind": "consumer",
            "process_info_status": "not_available_stdlib",
        }

    def _register_live_connection(
        self,
        connection_id: str,
        *,
        transport: str,
        peer_host: str,
        pid: Optional[int],
        process_info: Dict[str, Any],
    ) -> None:
        cid = str(connection_id or "").strip()
        if not cid:
            return
        now = time.time()
        with self._actor_connections_lock:
            self._live_connections[cid] = {
                "connection_id": cid,
                "transport": str(transport or "unknown"),
                "peer_host": str(peer_host or "") or None,
                "pid": int(pid or 0) or None,
                "process": dict(process_info or {}),
                "consumer_kind": str(dict(process_info or {}).get("consumer_kind") or "") or None,
                "connected_at": now,
                "last_seen_at": now,
                "last_command": None,
                "command_count": 0,
                "actor_ids": [],
                "session_token_previews": [],
            }

    def _update_live_connection(
        self,
        connection_id: str,
        *,
        command: str,
        actor_id: str,
        session_token: str,
    ) -> None:
        cid = str(connection_id or "").strip()
        if not cid:
            return
        aid = str(actor_id or "").strip()
        preview = self.svc._token_preview(str(session_token or "")) if session_token else ""
        with self._actor_connections_lock:
            row = dict(self._live_connections.get(cid) or {})
            if not row:
                return
            row["last_seen_at"] = time.time()
            row["last_command"] = str(command or "") or None
            row["command_count"] = int(row.get("command_count") or 0) + 1
            actor_ids = [str(x) for x in list(row.get("actor_ids") or []) if str(x or "").strip()]
            if aid and aid not in actor_ids:
                actor_ids.append(aid)
            previews = [str(x) for x in list(row.get("session_token_previews") or []) if str(x or "").strip()]
            if preview and preview not in previews:
                previews.append(preview)
            row["actor_ids"] = actor_ids
            row["session_token_previews"] = previews
            self._live_connections[cid] = row

    def _unregister_live_connection(self, connection_id: str) -> None:
        cid = str(connection_id or "").strip()
        if not cid:
            return
        with self._actor_connections_lock:
            self._live_connections.pop(cid, None)

    def _list_live_consumers(self) -> Dict[str, Any]:
        now = time.time()
        with self._actor_connections_lock:
            connections = [dict(row or {}) for row in self._live_connections.values()]
            actor_counts = dict(self._actor_connections)
        for row in connections:
            connected_at = float(row.get("connected_at") or 0.0)
            last_seen_at = float(row.get("last_seen_at") or 0.0)
            row["age_seconds"] = max(0, int(now - connected_at)) if connected_at > 0 else None
            row["idle_seconds"] = max(0, int(now - last_seen_at)) if last_seen_at > 0 else None
        connections.sort(key=lambda x: (str(x.get("transport") or ""), float(x.get("connected_at") or 0.0)))
        actors = [
            {"actor_id": actor_id, "connection_count": int(count or 0)}
            for actor_id, count in sorted(actor_counts.items())
            if int(count or 0) > 0
        ]
        return {
            "timestamp": now,
            "connections_count": len(connections),
            "actors_count": len(actors),
            "connections": connections,
            "actors": actors,
        }

    @staticmethod
    def _host_capability_session_public(session: HostCapabilitySession) -> Dict[str, Any]:
        return session.to_public_dict()

    def _audit_host_capability_session_close(self, session: HostCapabilitySession, *, reason: str) -> None:
        append = getattr(self.svc, "_append_host_capability_audit_event", None)
        if not callable(append):
            return
        try:
            append({
                "event_type": "host_capability_session_close",
                "session_id": session.session_id,
                "provider_id": session.provider_id,
                "result": "closed",
                "reason": str(reason or "closed"),
            })
        except Exception:
            return

    @staticmethod
    def _host_capability_session_methods(
        *,
        provider_id: str,
        owner: str,
        provider_kind: str,
        visibility: str,
        methods: List[Dict[str, Any]],
    ) -> Dict[str, HostCapabilityMethod]:
        if not methods:
            raise ValueError("host_capability_methods_required")
        if len(methods) > 128:
            raise ValueError("host_capability_method_limit_exceeded")
        out: Dict[str, HostCapabilityMethod] = {}
        for raw in methods:
            row = dict(raw or {})
            name = str(row.get("name") or "").strip()
            if not name:
                raise ValueError("host_capability_method_name_required")
            declared_provider_id = str(dict(row.get("provider") or {}).get("provider_id") or "").strip()
            if declared_provider_id and declared_provider_id != provider_id:
                raise ValueError("host_capability_method_provider_id_mismatch")
            provider = HostCapabilityProviderRef(
                provider_id=provider_id,
                kind=provider_kind,
                owner=owner,
                visibility=visibility,
            )
            row["provider"] = provider.to_dict()
            row.setdefault("namespace", name.split(".", 1)[0])
            row.setdefault("group_path", [part.replace("_", " ").title() for part in name.split(".")[:-1]] or ["Host"])
            descriptor = HostCapabilityDescriptor.from_dict(row)
            if descriptor.name in out:
                raise ValueError(f"host_capability_duplicate_method:{descriptor.name}")
            out[descriptor.name] = HostCapabilityMethod(descriptor=descriptor)
        return out

    @staticmethod
    def _normalize_host_capability_binding(payload: Dict[str, Any], *, transport: str) -> Dict[str, Any]:
        binding = dict(payload.get("binding") or {})
        transport_name = str(binding.get("transport") or "").strip().lower()
        if not transport_name:
            transport_name = "ssh_relay" if str(transport or "").strip() == "tcp" else "daemon_callback"
        if transport_name not in {"daemon_callback", "local_ipc", "ssh_relay", "toolbox_harness", "service_broker"}:
            raise ValueError(f"host_capability_invalid_binding_transport:{transport_name}")
        out = dict(binding)
        out["transport"] = transport_name
        return out

    def _register_host_capability_session(
        self,
        payload: Dict[str, Any],
        *,
        transport: str,
        peer_host: str,
        peer_pid: Optional[int],
        peer_process_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        row = dict(payload.get("session") or payload or {})
        actor_id = str(payload.get("_claim_actor_id") or row.get("owner") or "").strip()
        if not actor_id:
            actor_id = self.svc._actor_id_from_payload(self.svc._read_control(), payload)  # noqa: SLF001
        session_id = str(row.get("session_id") or "").strip() or f"cap_{secrets.token_urlsafe(18)}"
        provider_id = str(row.get("provider_id") or dict(row.get("provider") or {}).get("provider_id") or "").strip()
        if not provider_id:
            raise ValueError("host_capability_provider_id_required")
        if provider_id == session_id:
            raise ValueError("host_capability_provider_and_session_id_must_differ")
        provider_kind = str(row.get("provider_kind") or dict(row.get("provider") or {}).get("kind") or "client_session").strip()
        visibility = str(row.get("visibility") or dict(row.get("provider") or {}).get("visibility") or "workflow").strip()
        if provider_kind not in {"client_session", "toolbox_session", "service_broker"}:
            raise ValueError(f"host_capability_invalid_provider_kind:{provider_kind}")
        if visibility not in {"request", "workflow", "instance", "consumer"}:
            raise ValueError(f"host_capability_invalid_visibility:{visibility}")
        scope = dict(row.get("scope") or {})
        allow_override = bool(row.get("allow_override") or row.get("override"))
        methods = self._host_capability_session_methods(
            provider_id=provider_id,
            owner=actor_id,
            provider_kind=provider_kind,
            visibility=visibility,
            methods=[dict(item or {}) for item in list(row.get("methods") or [])],
        )
        if provider_kind == "service_broker":
            raw_binding = dict(row.get("binding") or {})
            raw_binding["transport"] = "service_broker"
            row["binding"] = raw_binding
        binding = self._normalize_host_capability_binding(row, transport=transport)
        binding.setdefault("peer_host", str(peer_host or "") or None)
        binding.setdefault("peer_pid", int(peer_pid or 0) or None)
        binding.setdefault("peer_process", dict(peer_process_info or {}))
        if "close_on_client_disconnect" in row or "expires_at_ms" in row:
            raise ValueError("legacy_host_capability_lifetime_fields_unsupported")
        now_ms = int(time.time() * 1000)
        lease_row = dict(row.get("authority_lease") or {})
        lease_token = secrets.token_urlsafe(32)
        authority_lease = CapabilityAuthorityLease(
            owner_authority_id=actor_id,
            token_digest=hashlib.sha256(lease_token.encode("utf-8")).hexdigest(),
            expires_at_ms=(int(lease_row["expires_at_ms"]) if lease_row.get("expires_at_ms") is not None else None),
            on_transport_loss=str(lease_row.get("on_transport_loss") or "close").strip(),
            on_authority_revoked=str(lease_row.get("on_authority_revoked") or "close").strip(),
            on_request_terminal=str(lease_row.get("on_request_terminal") or "retain").strip(),
        )
        authority_lease.validate(now_ms=now_ms)
        session = HostCapabilitySession(
            session_id=session_id,
            provider_id=provider_id,
            owner=actor_id,
            provider_kind=provider_kind,
            visibility=visibility,
            scope=scope,
            methods=methods,
            binding=binding,
            created_at_ms=now_ms,
            authority_lease=authority_lease,
            allow_override=allow_override,
        )
        with self._host_capability_sessions_lock:
            if session_id in self._host_capability_sessions:
                raise ValueError("host_capability_session_already_exists")
            if any(
                existing.provider_id == provider_id and existing.session_id == session_id
                for existing in self._host_capability_sessions.values()
            ):
                raise ValueError("host_capability_provider_session_already_exists")
            if not allow_override:
                incoming_names = set(methods.keys())
                for existing in self._host_capability_sessions.values():
                    duplicates = sorted(incoming_names.intersection(dict(existing.methods or {}).keys()))
                    if duplicates:
                        raise ValueError(f"host_capability_duplicate_method:{duplicates[0]}")
            self._host_capability_sessions[session_id] = session
        return {
            "status": "ok",
            "session": self._host_capability_session_public(session),
            "authority_lease_token": lease_token,
        }

    def _list_host_capability_sessions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        actor_id = str(payload.get("_claim_actor_id") or "").strip()
        if not actor_id:
            actor_id = self.svc._actor_id_from_payload(self.svc._read_control(), payload)  # noqa: SLF001
        include_all = bool(payload.get("include_all", False))
        now_ms = int(time.time() * 1000)
        with self._host_capability_sessions_lock:
            expired = [
                sid
                for sid, session in self._host_capability_sessions.items()
                if session.authority_lease.expires_at_ms is not None
                and int(session.authority_lease.expires_at_ms or 0) <= now_ms
            ]
            for sid in expired:
                session = self._host_capability_sessions.pop(sid, None)
                if session is not None:
                    self._audit_host_capability_session_close(session, reason="expired")
            sessions = [
                self._host_capability_session_public(session)
                for session in self._host_capability_sessions.values()
                if include_all or str(session.owner or "") == actor_id
            ]
        sessions.sort(key=lambda item: str(item.get("session_id") or ""))
        return {"status": "ok", "sessions": sessions, "count": len(sessions)}

    def _close_host_capability_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("host_capability_session_id_required")
        actor_id = str(payload.get("_claim_actor_id") or "").strip()
        if not actor_id:
            actor_id = self.svc._actor_id_from_payload(self.svc._read_control(), payload)  # noqa: SLF001
        force = bool(payload.get("force", False))
        with self._host_capability_sessions_lock:
            session = self._host_capability_sessions.get(session_id)
            if session is None:
                return {"status": "not_found", "session_id": session_id, "closed": False}
            if str(session.owner or "") != actor_id and not force:
                raise PermissionError("host_capability_session_not_owned")
            self._host_capability_sessions.pop(session_id, None)
        self._audit_host_capability_session_close(session, reason="administrative_force_close" if force else "explicit_close")
        return {"status": "closed", "session_id": session_id, "closed": True}

    @staticmethod
    def _authority_token_matches(session: HostCapabilitySession, token: str) -> bool:
        digest = hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, session.authority_lease.token_digest)

    def _renew_host_capability_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(payload.get("session_id") or "").strip()
        actor_id = str(payload.get("_claim_actor_id") or "").strip()
        token = str(payload.get("authority_lease_token") or "")
        expires_at_ms = payload.get("expires_at_ms")
        if not session_id or expires_at_ms is None or int(expires_at_ms or 0) <= int(time.time() * 1000):
            raise ValueError("capability_authority_renewal_invalid")
        with self._host_capability_sessions_lock:
            session = self._host_capability_sessions.get(session_id)
            if session is None:
                return {"status": "not_found", "session_id": session_id, "renewed": False}
            if session.authority_lease.owner_authority_id != actor_id:
                raise PermissionError("capability_authority_not_owned")
            if not self._authority_token_matches(session, token):
                raise PermissionError("capability_authority_token_invalid")
            session.authority_lease.expires_at_ms = int(expires_at_ms)
            session.authority_lease.renewed_at_ms = int(time.time() * 1000)
            session.authority_lease.validate()
            public = self._host_capability_session_public(session)
        return {"status": "renewed", "session_id": session_id, "renewed": True, "session": public}

    def _revoke_host_capability_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(payload.get("session_id") or "").strip()
        actor_id = str(payload.get("_claim_actor_id") or "").strip()
        force = bool(payload.get("force", False))
        token = str(payload.get("authority_lease_token") or "")
        with self._host_capability_sessions_lock:
            session = self._host_capability_sessions.get(session_id)
            if session is None:
                return {"status": "not_found", "session_id": session_id, "revoked": False}
            if not force:
                if session.authority_lease.owner_authority_id != actor_id:
                    raise PermissionError("capability_authority_not_owned")
                if not self._authority_token_matches(session, token):
                    raise PermissionError("capability_authority_token_invalid")
            self._host_capability_sessions.pop(session_id, None)
        self._audit_host_capability_session_close(session, reason="authority_revoked")
        return {"status": "revoked", "session_id": session_id, "revoked": True}

    def _apply_request_terminal_session_policy(self, result: Any) -> None:
        row = dict(result or {}) if isinstance(result, dict) else {}
        lifecycle = str(row.get("lifecycle") or "")
        if lifecycle not in {
            "terminal_success", "terminal_failure", "terminal_cancellation",
            "interrupted_before_dispatch", "interrupted_after_dispatch_unknown",
        }:
            return
        request_id = str(row.get("request_id") or "").strip()
        if not request_id:
            return
        closed: List[HostCapabilitySession] = []
        with self._host_capability_sessions_lock:
            for session_id, session in list(self._host_capability_sessions.items()):
                if (
                    session.authority_lease.on_request_terminal == "close"
                    and str(dict(session.scope or {}).get("request_id") or "") == request_id
                ):
                    self._host_capability_sessions.pop(session_id, None)
                    closed.append(session)
        for session in closed:
            self._audit_host_capability_session_close(session, reason="request_terminal")

    def _close_host_capability_sessions_for_actor(self, actor_id: str, *, reason: str) -> int:
        aid = str(actor_id or "").strip()
        if not aid:
            return 0
        closed = 0
        closed_sessions: List[HostCapabilitySession] = []
        with self._host_capability_sessions_lock:
            for sid, session in list(self._host_capability_sessions.items()):
                if str(session.owner or "") == aid and session.authority_lease.on_transport_loss == "close":
                    self._host_capability_sessions.pop(sid, None)
                    closed += 1
                    closed_sessions.append(session)
        for session in closed_sessions:
            self._audit_host_capability_session_close(session, reason=reason or "transport_loss")
        return closed

    def _host_capability_sessions_snapshot(self) -> List[HostCapabilitySession]:
        if not hasattr(self, "_host_capability_sessions_lock"):
            return []
        now_ms = int(time.time() * 1000)
        with self._host_capability_sessions_lock:
            for sid, session in list(self._host_capability_sessions.items()):
                if session.authority_lease.expires_at_ms is not None and int(session.authority_lease.expires_at_ms or 0) <= now_ms:
                    self._host_capability_sessions.pop(sid, None)
                    self._audit_host_capability_session_close(session, reason="expired")
            return list(self._host_capability_sessions.values())

    def _host_capability_approval_requester_from_payload(self, payload: Dict[str, Any]) -> Any:
        if payload.get("approval_requester_binding") is None:
            return None
        return self.svc._host_capability_approval_requester_from_binding(payload.get("approval_requester_binding"))  # noqa: SLF001

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
            self._close_host_capability_sessions_for_actor(actor_id, reason="owner_disconnect")
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
            "cancel_requested": False,
            "cancel_requested_at": None,
            "cancel_completed_at": None,
            "cancel_reason": None,
            "cancel_teardown_attempted": False,
            "cancel_teardown_status": None,
            "target_engine_id": self._operation_target_engine_id(command, payload),
            "payload_hint": self._operation_payload_hint(command, payload),
            "diagnostics": {},
            "progress_events": [
                self._operation_event("queued", "queued", "Operation queued", command=str(command or ""))
            ],
            "session_token": session_token or None,
        }
        if str(command or "").strip() == "connect-from-config":
            op["progress_percent"] = 0
            op["progress_text"] = "Operation queued"
            op["progress_events"][0]["progress_percent"] = 0
        with self._operations_lock:
            self._operations[op_id] = op
            self._prune_operations_locked()
            self._persist_operations_locked()
        self._append_operation_journal(op, event="created")
        return self._operation_public_snapshot(op)

    def _get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        op_id = str(operation_id or "").strip()
        with self._operations_lock:
            op = self._operations.get(op_id)
            if not isinstance(op, dict):
                persisted = self._load_persisted_operations().get(op_id)
                return dict(persisted) if isinstance(persisted, dict) else None
            enriched = self._enrich_operation_progress(dict(op))
            if dict(enriched or {}) != dict(op or {}):
                self._operations[op_id] = dict(enriched)
                self._persist_operations_locked()
            return enriched

    def _replace_operation(self, op: Dict[str, Any]) -> None:
        op_id = str(op.get("operation_id") or "")
        if not op_id:
            return
        enriched = self._enrich_operation_progress(dict(op))
        with self._operations_lock:
            self._operations[op_id] = dict(enriched)
            self._prune_operations_locked()
            self._persist_operations_locked()
        self._append_operation_journal(enriched, event="updated")

    def _record_operation_progress_event(self, operation_id: str, event: Dict[str, Any]) -> None:
        op_id = str(operation_id or "").strip()
        if not op_id:
            return
        ev = dict(event or {})
        with self._operations_lock:
            op = dict(self._operations.get(op_id) or {})
            if not op or bool(op.get("done", False)):
                return
            events = list(op.get("progress_events") or [])
            events.append(ev)
            op["progress_events"] = events
            message = str(ev.get("message") or "").strip()
            if message:
                op["progress_text"] = message
            if ev.get("progress_percent") is not None:
                current = int(op.get("progress_percent") or 0)
                incoming = max(0, min(100, int(ev.get("progress_percent") or 0)))
                op["progress_percent"] = max(current, incoming)
            engine_id = str(ev.get("engine_id") or "").strip()
            if engine_id and not str(op.get("target_engine_id") or "").strip():
                op["target_engine_id"] = engine_id
            log_path = str(ev.get("log_path") or "").strip()
            if log_path:
                diagnostics = dict(op.get("diagnostics") or {})
                diagnostics["log_path"] = log_path
                if "log_start_offset" not in diagnostics:
                    diagnostics["log_start_offset"] = self._log_size(Path(log_path).expanduser())
                if str(ev.get("stage") or "") == "connect.worker_ready" and str(ev.get("status") or "") == "running":
                    diagnostics["worker_ready_started_at"] = float(ev.get("timestamp") or time.time())
                op["diagnostics"] = diagnostics
            op["updated_at"] = time.time()
            self._operations[op_id] = op
            self._persist_operations_locked()

    def _finalize_operation_canceled(self, operation_id: str, message: str = "Operation canceled") -> None:
        op = self._get_operation(operation_id) or {}
        if not op or bool(op.get("done", False)):
            return
        now = time.time()
        op["done"] = True
        op["status"] = "canceled"
        op["stage"] = "canceled"
        op["updated_at"] = now
        op["completed_at"] = now
        op["cancel_completed_at"] = now
        events = list(op.get("progress_events") or [])
        events.append(self._operation_event("canceled", "canceled", message))
        op["progress_events"] = events
        self._replace_operation(op)

    def _operation_cancel_teardown(self, op: Dict[str, Any]) -> Dict[str, Any]:
        engine_id = str((op or {}).get("target_engine_id") or "").strip()
        command = str((op or {}).get("command") or "").strip()
        if command not in {"connect-from-config", "spawn"}:
            return {"attempted": False, "status": "not_applicable", "engine_id": engine_id or None}
        if not engine_id:
            return {"attempted": False, "status": "target_engine_id_unknown", "engine_id": None}
        try:
            result = self.svc.shutdown(engine_id, timeout_seconds=2.0)
            status = str((result or {}).get("status") or "").strip() or "unknown"
            ok = status in {"stopped", "already_stopped", "not_found", "invalid_pid"}
            return {
                "attempted": True,
                "status": status,
                "ok": ok,
                "engine_id": engine_id,
                "result": dict(result or {}),
            }
        except Exception as exc:
            return {
                "attempted": True,
                "status": "failed",
                "ok": False,
                "engine_id": engine_id,
                "error": str(exc),
            }

    def _apply_operation_cancel_teardown(self, op: Dict[str, Any], teardown: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(op or {})
        out["cancel_teardown_attempted"] = bool(teardown.get("attempted", False))
        out["cancel_teardown_status"] = str(teardown.get("status") or "").strip() or None
        out["cancel_teardown"] = dict(teardown)
        out["updated_at"] = time.time()
        if bool(teardown.get("attempted", False)) and not bool(teardown.get("ok", True)):
            out["done"] = True
            out["status"] = "cancel_failed"
            out["stage"] = "cancel_failed"
            out["error"] = str(teardown.get("error") or teardown.get("status") or "cancel_failed")
            out["error_code"] = "cancel_failed"
            out["completed_at"] = out["updated_at"]
            events = list(out.get("progress_events") or [])
            events.append(self._operation_event("cancel_failed", "failed", "Operation cancel failed"))
            out["progress_events"] = events
        return out

    async def _request_operation_cancel(self, operation_id: str, *, reason: str = "") -> Dict[str, Any]:
        op_id = str(operation_id or "").strip()
        op = self._get_operation(op_id) or {}
        if not op:
            return {}
        if bool(op.get("done", False)):
            snapshot = self._operation_public_snapshot(op)
            snapshot["cancel_status"] = "already_done"
            return snapshot

        now = time.time()
        events = list(op.get("progress_events") or [])
        events.append(
            self._operation_event(
                "cancel_requested",
                "cancel_requested",
                "Operation cancel requested",
                reason=str(reason or "").strip() or None,
            )
        )
        op["cancel_requested"] = True
        op["cancel_requested_at"] = now
        op["cancel_reason"] = str(reason or "").strip() or None
        op["status"] = "cancel_requested"
        op["stage"] = "cancel_requested"
        op["updated_at"] = now
        op["progress_events"] = events
        self._replace_operation(op)

        with self._operation_tasks_lock:
            task = self._operation_tasks_by_id.get(op_id)
        command = str(op.get("command") or "").strip()
        active_task = bool(task is not None and not task.done())
        wait_for_service_result = bool(active_task and command in {"connect-from-config", "spawn"})
        task_cancel_requested = bool(active_task and not wait_for_service_result)
        if task_cancel_requested:
            task.cancel()

        teardown = await asyncio.to_thread(self._operation_cancel_teardown, op)
        op = self._get_operation(op_id) or op
        if not bool(op.get("done", False)):
            op = self._apply_operation_cancel_teardown(op, teardown)
            self._replace_operation(op)

        if not task_cancel_requested and not wait_for_service_result:
            self._finalize_operation_canceled(op_id)

        snapshot = self._operation_public_snapshot(self._get_operation(op_id) or op)
        snapshot["cancel_status"] = "cancel_requested" if task_cancel_requested else str(snapshot.get("status") or "canceled")
        return snapshot

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
        if str(command or "").strip() == "connect-from-config":
            op["progress_percent"] = 0
            op["progress_text"] = "Resolving engine config"
            events.append(
                self._operation_event(
                    "connect.resolve_config",
                    "running",
                    "Resolving engine config",
                    progress_percent=0,
                )
            )
        op["progress_events"] = events
        self._replace_operation(op)
        try:
            operation_payload = dict(payload or {})
            if str(command or "").strip() == "connect-from-config":
                operation_payload["_progress_callback"] = lambda event: self._record_operation_progress_event(operation_id, dict(event or {}))
            result = await asyncio.to_thread(self._call_service, command, operation_payload)
            now = time.time()
            op = self._get_operation(operation_id) or op
            if bool(op.get("cancel_requested", False)) and not bool(op.get("done", False)):
                if isinstance(result, dict) and not str(op.get("target_engine_id") or "").strip():
                    result_engine_id = str(result.get("engine_id") or "").strip()
                    if result_engine_id:
                        op["target_engine_id"] = result_engine_id
                        self._replace_operation(op)
                teardown = await asyncio.to_thread(self._operation_cancel_teardown, op)
                op = self._get_operation(operation_id) or op
                if not bool(op.get("done", False)):
                    op = self._apply_operation_cancel_teardown(op, teardown)
                    self._replace_operation(op)
                if bool((self._get_operation(operation_id) or {}).get("done", False)):
                    return
                self._finalize_operation_canceled(operation_id)
                return
            service_failed = isinstance(result, dict) and str(result.get("status") or "").strip().lower() in {"failed", "error"}
            if isinstance(result, dict) and not str(op.get("target_engine_id") or "").strip():
                result_engine_id = str(result.get("engine_id") or result.get("worker_id") or result.get("model_instance_id") or "").strip()
                if result_engine_id:
                    op["target_engine_id"] = result_engine_id
            op["done"] = True
            op["status"] = "failed" if service_failed else "completed"
            op["stage"] = "failed" if service_failed else "completed"
            op["result"] = result
            if service_failed and isinstance(result, dict):
                op["error"] = str(result.get("message") or result.get("reason") or "operation_failed")
                op["error_code"] = str(result.get("reason") or "operation_failed")
            if str(command or "").strip() == "connect-from-config" and not service_failed:
                op["progress_percent"] = 100
                op["progress_text"] = "Operation completed"
            op["updated_at"] = now
            op["completed_at"] = now
            events = list(op.get("progress_events") or [])
            if isinstance(result, dict) and isinstance(result.get("progress_events"), list):
                events.extend(list(result.get("progress_events") or []))
            if service_failed:
                events.append(self._operation_event("failed", "failed", str(op.get("error") or "Operation failed")))
            else:
                extra: Dict[str, Any] = {}
                if str(command or "").strip() == "connect-from-config":
                    extra["progress_percent"] = 100
                events.append(self._operation_event("completed", "completed", "Operation completed", **extra))
            op["progress_events"] = events
            self._replace_operation(op)
        except asyncio.CancelledError:
            self._finalize_operation_canceled(operation_id)
            raise
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
        """Start local IPC control listener, then write PID file and run until stop event."""
        self._stop_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        enable_tcp = self._should_enable_tcp()
        started = False
        try:
            read_pid_file = getattr(self.pid_file, "read", None)
            is_pid_alive = getattr(self.pid_file, "is_alive", None)
            is_pid_process_alive = getattr(self.pid_file, "process_alive", None)
            existing = dict(read_pid_file() or {}) if callable(read_pid_file) else {}
            existing_alive = (
                bool(is_pid_process_alive())
                if callable(is_pid_process_alive)
                else bool(is_pid_alive()) if callable(is_pid_alive) else False
            )
            if existing_alive and existing.get("shutdown_token"):
                existing_state = str(existing.get("lifecycle_state") or "running").strip().lower()
                if existing_state in {"shutting_down", "stopping"}:
                    raise RuntimeError(
                        f"Engine host daemon is already shutting down for pid file {self.pid_file.path}"
                    )
                try:
                    from ..engine_host_connection import LocalSocketConnection

                    conn = LocalSocketConnection(
                        port=int(existing.get("port") or self.port),
                        pid_file=self.pid_file.path,
                        timeout=1.0,
                        max_reconnect_attempts=1,
                    )
                    try:
                        if conn.is_alive():
                            raise RuntimeError(
                                f"Engine host daemon is already running for pid file {self.pid_file.path}"
                            )
                        raise RuntimeError(
                            f"Engine host daemon PID is alive but local control is not reachable for pid file {self.pid_file.path}"
                        )
                    finally:
                        conn.close()
                except RuntimeError:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"Engine host daemon PID is alive but local control is not reachable for pid file {self.pid_file.path}: {exc}"
                    ) from exc
            startup_recovery = self._execute_startup_worker_recovery()
            if int(startup_recovery.get("foreign_attempted") or 0):
                logger.warning(
                    "Startup worker recovery stopped registrations from previous daemon owners: attempted=%s stopped=%s failed=%s",
                    startup_recovery.get("foreign_attempted"),
                    startup_recovery.get("foreign_stopped"),
                    startup_recovery.get("foreign_failed"),
                )
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
            self._started_at = time.time()
            self._started_monotonic = time.monotonic()
            write_kwargs["started_at"] = self._started_at
            try:
                self.pid_file.write(**write_kwargs)
            except TypeError:
                self.pid_file.write(
                    pid=int(write_kwargs["pid"]),
                    port=int(write_kwargs["port"]),
                    shutdown_token=str(write_kwargs["shutdown_token"]),
                )
            started = True
            write_daemon_report(
                event="daemon_started",
                reason="daemon process started",
                details={
                    "runtime_profile": str(self._runtime_profile or ""),
                    "pid_file": str(getattr(self.pid_file, "path", self.pid_file)),
                    "port": int(self.port),
                    "local_transport": dict(self._local_transport or {}),
                },
                path=self.svc.hosting_root / "logs" / "daemon-crash.log",
                overwrite=False,
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
            if not started:
                self._loop = None
            else:
                try:
                    self._publish_shutdown_progress(
                        "shutdown.begin",
                        "running",
                        "Daemon shutdown sequence started",
                    )
                    self._publish_shutdown_progress(
                        "shutdown.operations_drain",
                        "running",
                        "Draining in-flight operations",
                    )
                    drain_report = await self._drain_inflight_operations(timeout_seconds=5.0)
                    self._publish_shutdown_progress(
                        "shutdown.operations_drain",
                        "completed",
                        "In-flight operations drain complete",
                        operation_drain=dict(drain_report),
                        pending_before=int(drain_report.get("pending_before") or 0),
                        pending_after=int(drain_report.get("pending_after") or 0),
                        timed_out=bool(drain_report.get("timed_out", False)),
                    )
                    self._publish_shutdown_progress(
                        "shutdown.managed_workers",
                        "running",
                        "Running managed worker shutdown checkpoints",
                        operation_drain=dict(drain_report),
                    )
                    report = await asyncio.to_thread(self._execute_shutdown_checkpoints)
                    report["operation_drain"] = dict(drain_report)
                    report["shutdown_stages"] = list(self._shutdown_stage_events)
                    self._last_shutdown_checkpoints = dict(report)
                    self._publish_shutdown_progress(
                        "shutdown.managed_workers",
                        "completed",
                        "Managed worker shutdown checkpoints complete",
                        shutdown_checkpoints=dict(report),
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
                    self._publish_shutdown_progress(
                        "shutdown.failed",
                        "failed",
                        "Shutdown checkpoints failed",
                        error=str(exc),
                    )
                    logger.warning("Shutdown checkpoints failed: %s", exc)
                self._stop_local_control_listener()
                self.pid_file.remove()
                shutdown_report = dict(self._shutdown_report or {})
                write_daemon_report(
                    event="daemon_stopped",
                    reason=str(shutdown_report.get("reason") or "daemon_run_exited"),
                    actor=dict(shutdown_report.get("actor") or {}),
                    details={
                        **dict(shutdown_report.get("details") or {}),
                        "shutdown_checkpoints": dict(self._last_shutdown_checkpoints or {}),
                    },
                    path=self.svc.hosting_root / "logs" / "daemon-crash.log",
                )
                self._loop = None
                logger.info("EngineHostDaemon stopped")

    def _daemon_runtime_status(self) -> Dict[str, Any]:
        """Return daemon-owned startup timing for control-channel responses."""
        now = time.time()
        started_at = self._started_at
        if started_at is None:
            try:
                persisted = dict(self.pid_file.read() or {})
                raw_started_at = persisted.get("started_at")
                started_at = float(raw_started_at) if raw_started_at is not None else None
            except (TypeError, ValueError):
                started_at = None
        if self._started_monotonic is not None:
            uptime_seconds = max(0.0, time.monotonic() - self._started_monotonic)
        elif started_at is not None:
            uptime_seconds = max(0.0, now - started_at)
        else:
            uptime_seconds = None
        return {
            "pid": os.getpid(),
            "port": int(self.port),
            "started_at": started_at,
            "uptime_seconds": uptime_seconds,
            "lifecycle_state": "running",
        }

    async def _dispatch(
        self,
        raw_line: str,
        *,
        peer_host: Optional[str] = None,
        peer_pid: Optional[int] = None,
        peer_process_info: Optional[Dict[str, Any]] = None,
        transport: str = "",
    ) -> Dict[str, Any]:
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
            return {
                "seq": seq,
                "ok": True,
                "result": "pong",
                **self._daemon_runtime_status(),
            }

        if cmd in {"daemon-status", "__daemon_status__"}:
            return {
                "seq": seq,
                "ok": True,
                "result": self._daemon_runtime_status(),
            }

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
                self._shutdown_report = {
                    "reason": str(payload.get("shutdown_reason") or payload.get("reason") or "client_requested_shutdown"),
                    "actor": {
                        "requested_by": str(payload.get("requested_by") or "unknown"),
                        "transport": str(transport or "unknown"),
                        "peer_host": str(peer_host or "") or None,
                        "peer_pid": int(peer_pid or 0) or None,
                        "peer_process": dict(peer_process_info or {}),
                    },
                    "details": {
                        "command": "__shutdown__",
                        "runtime_profile": str(self._runtime_profile or ""),
                        "pid_file": str(self.pid_file.path),
                        "local_transport": dict(self._local_transport or {}),
                    },
                }
                assert self._stop_event is not None
                self._shutdown_stage_events = []
                mark_shutting_down = getattr(self.pid_file, "mark_shutting_down", None)
                if callable(mark_shutting_down):
                    mark_shutting_down(
                        reason=str(payload.get("shutdown_reason") or payload.get("reason") or "client_requested_shutdown"),
                        requested_by=str(payload.get("requested_by") or "unknown"),
                    )
                self._publish_shutdown_progress(
                    "shutdown.accepted",
                    "running",
                    "Daemon shutdown request accepted",
                )
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
            if payload.get("session_token") and not target_payload.get("session_token"):
                target_payload["session_token"] = payload.get("session_token")
            if payload.get("_ssh_session_binding") and not target_payload.get("_ssh_session_binding"):
                target_payload["_ssh_session_binding"] = payload.get("_ssh_session_binding")
            if not target_cmd:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "command_required",
                    "error_code": "command_required",
                    "error_details": {},
                }
            if target_cmd in {"__ping__", "__shutdown__", "op-start", "op-status", "op-cancel"}:
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
                if target_cmd in {
                    "toolbox-plan-definition",
                    "toolbox-confirm-definition-plan",
                    "toolbox-apply-definition",
                    "environment-remove",
                    "environment-template-construct",
                    "toolbox-gc",
                    "toolbox-repair",
                    "toolbox-reconcile",
                    "toolbox-describe-refresh",
                }:
                    result = await asyncio.to_thread(
                        self._call_service, target_cmd, target_payload
                    )
                    return {"seq": seq, "ok": True, "result": result}
                op_snapshot = self._create_operation(command=target_cmd, payload=target_payload)
                operation_id = str(op_snapshot.get("operation_id") or "")
                task = asyncio.create_task(self._run_operation(operation_id, target_cmd, target_payload))
                with self._operation_tasks_lock:
                    self._operation_tasks.add(task)
                    if operation_id:
                        self._operation_tasks_by_id[operation_id] = task
                def _on_done(done_task: asyncio.Task) -> None:
                    if done_task.cancelled():
                        self._finalize_operation_canceled(operation_id)
                    with self._operation_tasks_lock:
                        self._operation_tasks.discard(done_task)
                        if operation_id and self._operation_tasks_by_id.get(operation_id) is done_task:
                            self._operation_tasks_by_id.pop(operation_id, None)
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
            canonical = self.svc._hosted_operations.get_by_operation_id(op_id)
            if canonical is not None:
                try:
                    self.svc.authorize_command(cmd, payload)
                    acl = self.svc.enforce_daemon_claim_policy(
                        cmd, payload, peer_host=peer_host, is_localhost=is_localhost
                    )
                    if not bool(acl.get("ok", False)):
                        raise PermissionError(str(acl.get("error_code") or "access_denied"))
                    actor = str(dict(acl.get("payload") or {}).get("_claim_actor_id") or "service:local")
                    result = self.svc.hosted_operation_status(
                        ref=dict(canonical["operation"]), owner_actor_id=actor
                    )
                    return {"seq": seq, "ok": True, "result": result}
                except PermissionError as exc:
                    return {
                        "seq": seq,
                        "ok": False,
                        "error": "auth_failed",
                        "error_code": str(exc or "auth_failed"),
                        "error_details": {"operation_id": op_id},
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
            enriched = self._enrich_operation_progress(op)
            if enriched != op:
                self._replace_operation(enriched)
            return {"seq": seq, "ok": True, "result": self._operation_public_snapshot(enriched)}

        if cmd == "op-cancel":
            op_id = str(payload.get("operation_id") or "").strip()
            if not op_id:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "operation_id_required",
                    "error_code": "operation_id_required",
                    "error_details": {},
                }
            canonical = self.svc._hosted_operations.get_by_operation_id(op_id)
            if canonical is not None:
                try:
                    self.svc.authorize_command(cmd, payload)
                    acl = self.svc.enforce_daemon_claim_policy(
                        cmd, payload, peer_host=peer_host, is_localhost=is_localhost
                    )
                    if not bool(acl.get("ok", False)):
                        raise PermissionError(str(acl.get("error_code") or "access_denied"))
                    actor = str(dict(acl.get("payload") or {}).get("_claim_actor_id") or "service:local")
                    result = self.svc.hosted_operation_cancel(
                        ref=dict(canonical["operation"]),
                        reason=str(payload.get("reason") or "client_requested"),
                        owner_actor_id=actor,
                    )
                    return {"seq": seq, "ok": True, "result": result}
                except PermissionError as exc:
                    return {
                        "seq": seq,
                        "ok": False,
                        "error": "auth_failed",
                        "error_code": str(exc or "auth_failed"),
                        "error_details": {"operation_id": op_id},
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
            result = await self._request_operation_cancel(
                op_id,
                reason=str(payload.get("reason") or "").strip(),
            )
            return {"seq": seq, "ok": True, "result": result}

        service_call_started = False
        registry_before: Dict[str, Dict[str, Any]] = {}
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
            if cmd in {
                "toolbox-plan-definition",
                "toolbox-confirm-definition-plan",
                "toolbox-apply-definition",
                "environment-template-construct",
                "toolbox-gc",
                "toolbox-repair",
                "toolbox-reconcile",
            }:
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "operation_wrapper_required",
                    "error_code": "operation_wrapper_required",
                    "error_details": {"command": cmd},
                }
            if cmd == "host-capability-session-register":
                result = self._register_host_capability_session(
                    payload,
                    transport=transport,
                    peer_host=peer_host,
                    peer_pid=peer_pid,
                    peer_process_info=peer_process_info,
                )
                return {"seq": seq, "ok": True, "result": result}
            if cmd == "host-capability-session-list":
                result = self._list_host_capability_sessions(payload)
                return {"seq": seq, "ok": True, "result": result}
            if cmd == "host-capability-session-close":
                result = self._close_host_capability_session(payload)
                return {"seq": seq, "ok": True, "result": result}
            if cmd == "host-capability-session-renew":
                result = self._renew_host_capability_session(payload)
                return {"seq": seq, "ok": True, "result": result}
            if cmd == "host-capability-session-revoke":
                result = self._revoke_host_capability_session(payload)
                return {"seq": seq, "ok": True, "result": result}
            if cmd == "host-capability-audit-list":
                result = await asyncio.to_thread(self._call_service, cmd, payload)
                return {"seq": seq, "ok": True, "result": result}
            if cmd in {"discover-running", "shutdown", "remove-registration"}:
                registry_before = self._engine_registry_by_id()
            service_call_started = True
            result = await asyncio.to_thread(self._call_service, cmd, payload)
            self._apply_request_terminal_session_policy(result)
            if isinstance(result, dict) and str(result.get("status") or "").strip().lower() == "denied":
                return {
                    "seq": seq,
                    "ok": False,
                    "error": "access_denied",
                    "error_code": str(result.get("denied_code") or result.get("denied_reason") or "access_denied"),
                    "error_details": dict(result.get("details") or {}),
                    "result": result,
                }
            if cmd == "unload-model":
                self._record_synchronous_operation(
                    command=cmd,
                    payload=payload,
                    result=dict(result or {}) if isinstance(result, dict) else {"result": result},
                )
            elif cmd == "shutdown" and isinstance(result, dict) and str(result.get("status") or "") in {"stopped", "already_stopped", "invalid_pid"}:
                self._record_synchronous_operation(command=cmd, payload=payload, result=dict(result or {}))
            elif cmd == "remove-registration" and isinstance(result, dict) and bool(result.get("removed", False)):
                self._record_synchronous_operation(command=cmd, payload=payload, result=dict(result or {}))
            elif cmd == "discover-running" and bool(payload.get("prune_stale", True)):
                registry_after = self._engine_registry_by_id()
                removed_ids = sorted(set(registry_before) - set(registry_after))
                if removed_ids:
                    removed = [registry_before[eid] for eid in removed_ids if eid in registry_before]
                    self._record_synchronous_operation(
                        command="prune-stale-registration",
                        payload={
                            "engine_ids": removed_ids,
                            "reason": "discover_running_prune_stale",
                            "trigger_command": "discover-running",
                        },
                        result={
                            "status": "pruned",
                            "reason": "discover_running_prune_stale",
                            "pruned_engine_ids": removed_ids,
                            "pruned_registrations": removed,
                        },
                    )
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
        except ValueError as exc:
            code = str(exc or "").strip() or "invalid_request"
            if code.startswith("host_capability_"):
                return {
                    "seq": seq,
                    "ok": False,
                    "error": code,
                    "error_code": code,
                    "error_details": {"reason": code},
                }
            return {
                "seq": seq,
                "ok": False,
                "error": "invalid_request",
                "error_code": "invalid_request",
                "error_details": {"message": code},
            }
        except Exception as exc:
            if cmd == "unload-model" and service_call_started:
                self._record_synchronous_operation(
                    command=cmd,
                    payload=payload,
                    error=str(exc),
                    error_code="operation_failed",
                )
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
                worker_profile_class=payload.get("worker_profile_class"),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}),
                executor_kind=payload.get("executor_kind"),
                bundle=dict(payload.get("bundle") or {}),
                environment=dict(payload.get("environment") or {}),
                tool_access=dict(payload.get("tool_access") or {}),
                capabilities=dict(payload.get("capabilities") or {}),
            )
        if cmd == "workflow-js-environment-spec":
            return svc.workflow_js_environment_spec(
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
            )
        if cmd == "workflow-js-ensure":
            return svc.ensure_workflow_js(
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
            )
        if cmd == "workflow-js-resources":
            return svc.workflow_js_resources(
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
            )
        if cmd == "workflow-js-execute":
            return svc.execute_workflow_js(
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "workflow-js-action-describe":
            return svc.workflow_js_action_describe(
                request=dict(payload.get("request") or {}),
                include_hidden=bool(payload.get("include_hidden", False)),
                dynamic=bool(payload.get("dynamic", False)),
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                instance_id=str(payload.get("instance_id") or "").strip() or None,
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
            )
        if cmd == "workflow-js-action-execute":
            return svc.execute_workflow_js_action(
                action_name=str(payload.get("action_name") or ""),
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "workflow-js-instance-create":
            return svc.workflow_js_instance_create(
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                instance_id=str(payload.get("instance_id") or "").strip() or None,
                replace=bool(payload.get("replace", False)),
            )
        if cmd == "workflow-js-instance-execute":
            return svc.workflow_js_instance_execute(
                instance_id=str(payload.get("instance_id") or ""),
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "workflow-js-instance-close":
            return svc.workflow_js_instance_close(
                instance_id=str(payload.get("instance_id") or ""),
                reason=str(payload.get("reason") or "client_requested"),
            )
        if cmd == "workflow-js-instance-list":
            return svc.workflow_js_instance_list()
        if cmd == "workflow-js-set-capacity":
            return svc.set_workflow_js_capacity(
                profile=str(payload.get("profile") or "node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                capacity=int(payload.get("capacity") or 1),
            )
        if cmd == "workflow-js-stream-open":
            return svc.workflow_js_stream_open(
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-js-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                node=dict(payload.get("node") or {}),
                javascript=dict(payload.get("javascript") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                capacity=int(payload.get("capacity") or 1),
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
            )
        if cmd == "workflow-js-event-subscribe":
            return svc.workflow_js_event_subscribe(
                stream_id=str(payload.get("stream_id") or ""),
                max_items=int(payload.get("max_items") or 64),
            )
        if cmd == "workflow-js-stream-send":
            return svc.workflow_js_stream_send(
                stream_id=str(payload.get("stream_id") or ""),
                message=dict(payload.get("message") or {}),
            )
        if cmd == "workflow-js-stream-close":
            return svc.workflow_js_stream_close(stream_id=str(payload.get("stream_id") or ""))
        if cmd == "workflow-python-environment-spec":
            return svc.workflow_python_environment_spec(
                profile=str(payload.get("profile") or "helper"),
                environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                python=dict(payload.get("python") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
            )
        if cmd == "workflow-python-prepare-environment":
            return svc.workflow_python_prepare_environment(
                environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                python=dict(payload.get("python") or {}),
                package_id=str(payload.get("package_id") or "").strip() or None,
                workflow_id=str(payload.get("workflow_id") or "").strip() or None,
            )
        if cmd == "workflow-python-lock-environment":
            return svc.workflow_python_lock_environment(environment=dict(payload.get("environment") or {}))
        if cmd == "workflow-python-verify-environment":
            return svc.workflow_python_verify_environment(environment=dict(payload.get("environment") or {}))
        if cmd == "workflow-python-install-environment":
            return svc.workflow_python_install_environment(
                environment=dict(payload.get("environment") or {}),
                allow_execution=bool(payload.get("allow_execution", False)),
            )
        if cmd == "workflow-python-verify-install-receipt":
            return svc.workflow_python_verify_install_receipt(environment=dict(payload.get("environment") or {}))
        if cmd == "sandbox-state-snapshot":
            return svc.sandbox_state_snapshot(
                scope=str(payload.get("scope") or ""),
                workflow_id=str(payload.get("workflow_id") or ""),
                instance_id=str(payload.get("instance_id") or ""),
                request_id=str(payload.get("request_id") or ""),
                prefix=str(payload.get("prefix") or ""),
            )
        if cmd == "sandbox-state-restore":
            return svc.sandbox_state_restore(
                snapshot=dict(payload.get("snapshot") or {}),
                scope=str(payload.get("scope") or ""),
                workflow_id=str(payload.get("workflow_id") or ""),
                instance_id=str(payload.get("instance_id") or ""),
                request_id=str(payload.get("request_id") or ""),
                mode=str(payload.get("mode") or "merge"),
            )
        if cmd == "workflow-artifact-recovery-inspect":
            return svc.workflow_artifact_recovery_inspect(
                request_id=str(payload.get("request_id") or ""),
                names=list(payload.get("names") or []),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
            )
        if cmd == "workflow-artifact-recovery-claim":
            return svc.workflow_artifact_recovery_claim(
                request_id=str(payload.get("request_id") or ""),
                names=list(payload.get("names") or []),
                target_id=str(payload.get("target_id") or ""),
                instance_id=str(payload.get("instance_id") or ""),
                patch_absolute_paths=bool(payload.get("patch_absolute_paths", False)),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
            )
        if cmd == "workflow-artifact-recovery-cleanup":
            return svc.workflow_artifact_recovery_cleanup(
                request_id=str(payload.get("request_id") or ""),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
            )
        if cmd == "workflow-python-ensure":
            return svc.ensure_workflow_python(
                profile=str(payload.get("profile") or "helper"),
                environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                python=dict(payload.get("python") or {}),
                python_executable=payload.get("python_executable"),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                worker_profile_class=str(payload.get("worker_profile_class") or "generic"),
            )
        if cmd == "workflow-python-execute":
            return svc.execute_workflow_python(
                profile=str(payload.get("profile") or "helper"),
                environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "workflow-js-stream-status":
            return svc.workflow_js_stream_status(stream_id=str(payload.get("stream_id") or ""))
        if cmd == "workflow-python-action-describe":
            return svc.workflow_python_action_describe(
                request=dict(payload.get("request") or {}),
                include_hidden=bool(payload.get("include_hidden", False)),
                dynamic=bool(payload.get("dynamic", False)),
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-python-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                instance_id=str(payload.get("instance_id") or "").strip() or None,
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
            )
        if cmd == "workflow-python-action-execute":
            return svc.execute_workflow_python_action(
                action_name=str(payload.get("action_name") or ""),
                profile=str(payload.get("profile") or "helper"),
                environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "workflow-python-instance-create":
            return svc.workflow_python_instance_create(
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-python-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                instance_id=str(payload.get("instance_id") or "").strip() or None,
                replace=bool(payload.get("replace", False)),
            )
        if cmd == "workflow-python-instance-execute":
            return svc.workflow_python_instance_execute(
                instance_id=str(payload.get("instance_id") or ""),
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-python-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                capacity=int(payload.get("capacity") or 1),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "workflow-python-instance-close":
            return svc.workflow_python_instance_close(
                instance_id=str(payload.get("instance_id") or ""),
                reason=str(payload.get("reason") or "client_requested"),
            )
        if cmd == "workflow-python-instance-list":
            return svc.workflow_python_instance_list()
        if cmd == "workflow-python-resources":
            return svc.workflow_python_resources(
                profile=str(payload.get("profile") or "helper"),
                environment_name=str(payload.get("environment_name") or "workflow-python-helper"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                python=dict(payload.get("python") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
            )
        if cmd == "workflow-python-set-capacity":
            return svc.set_workflow_python_capacity(
                profile=str(payload.get("profile") or "helper"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                capacity=int(payload.get("capacity") or 1),
            )
        if cmd == "workflow-python-stream-open":
            return svc.workflow_python_stream_open(
                profile=str(payload.get("profile") or "node"),
                environment_name=str(payload.get("environment_name") or "workflow-python-node"),
                environment_key=str(payload.get("environment_key") or "").strip() or None,
                engine_id=str(payload.get("engine_id") or "").strip() or None,
                request=dict(payload.get("request") or {}),
                python=dict(payload.get("python") or {}),
                sandbox_policy=dict(payload.get("sandbox_policy") or {}) or None,
                capacity=int(payload.get("capacity") or 1),
                host_capability_sessions=self._host_capability_sessions_snapshot(),
                approval_requester=self._host_capability_approval_requester_from_payload(payload),
            )
        if cmd == "workflow-python-event-subscribe":
            return svc.workflow_python_event_subscribe(
                stream_id=str(payload.get("stream_id") or ""),
                max_items=int(payload.get("max_items") or 64),
            )
        if cmd == "workflow-python-stream-send":
            return svc.workflow_python_stream_send(
                stream_id=str(payload.get("stream_id") or ""),
                message=dict(payload.get("message") or {}),
            )
        if cmd == "workflow-python-stream-close":
            return svc.workflow_python_stream_close(stream_id=str(payload.get("stream_id") or ""))
        if cmd == "get-registration":
            return svc.get_registration(str(payload.get("engine_id") or ""))
        if cmd == "shutdown":
            return svc.shutdown(
                str(payload.get("engine_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 8.0),
            )
        if cmd == "ensure-running":
            return svc.ensure_running(str(payload.get("engine_id") or ""))
        if cmd == "unload-model":
            return svc.unload_model(
                str(payload.get("engine_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                shutdown_all=bool(payload.get("shutdown_all", False)),
            )
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
            progress_callback = payload.get("_progress_callback")
            return svc.connect_from_config(
                config_path=str(payload.get("config_path") or "default"),
                engine_id=payload.get("engine_id"),
                model_path=payload.get("model_path"),
                force_new_worker=bool(payload.get("force_new_worker", False)),
                launch_policy=payload.get("launch_policy"),
                target_worker_id=payload.get("target_worker_id"),
                progress_callback=progress_callback if callable(progress_callback) else None,
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
                operator_details=bool(payload.get("operator_details", False)),
                timeout_seconds=float(payload.get("timeout_seconds") or 10.0),
            )
        if cmd == "toolbox-describe-refresh":
            return svc.toolbox_describe_refresh(
                engine_id=str(payload.get("engine_id") or ""),
                toolbox_id=str(payload.get("toolbox_id") or ""),
                request_id=str(payload.get("request_id") or ""),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
                timeout_seconds=float(payload.get("timeout_seconds") or 10.0),
            )
        if cmd == "workflow-python-stream-status":
            return svc.workflow_python_stream_status(stream_id=str(payload.get("stream_id") or ""))
        if cmd == "toolbox-gate":
            return svc.toolbox_gate(
                engine_id=str(payload.get("engine_id") or ""),
                toolbox_id=str(payload.get("toolbox_id") or ""),
                tool_name=str(payload.get("tool_name") or ""),
                tools_view=dict(payload.get("tools_view") or {}) if isinstance(payload.get("tools_view"), dict) else None,
            )
        if cmd == "hosted-operation-status":
            return svc.hosted_operation_status(
                ref=dict(payload.get("ref") or {}),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "hosted-operation-resolve-request":
            return svc.hosted_operation_resolve_request(
                execution_kind=str(payload.get("execution_kind") or ""),
                selector=dict(payload.get("selector") or {}),
                request_id=str(payload.get("request_id") or ""),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "hosted-operation-result":
            return svc.hosted_operation_result(
                ref=dict(payload.get("ref") or {}),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "hosted-operation-cancel":
            return svc.hosted_operation_cancel(
                ref=dict(payload.get("ref") or {}),
                reason=str(payload.get("reason") or "client_requested"),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
                timeout_seconds=float(payload.get("timeout_seconds") or 8.0),
                respawn=bool(payload.get("respawn", True)),
            )
        if cmd == "toolbox-execute":
            return svc.toolbox_execute(
                engine_id=str(payload.get("engine_id") or ""),
                toolbox_id=str(payload.get("toolbox_id") or ""),
                tool_call=dict(payload.get("tool_call") or {}),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                tools_view=dict(payload.get("tools_view") or {}) if isinstance(payload.get("tools_view"), dict) else None,
                callback_binding=dict(payload.get("callback_binding") or {}) if isinstance(payload.get("callback_binding"), dict) else None,
                host_api_approval=dict(payload.get("host_api_approval") or {}) if isinstance(payload.get("host_api_approval"), dict) else None,
                execution_request_id=str(payload.get("execution_request_id") or ""),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "toolbox-gc":
            return svc.toolbox_gc(
                request_id=str(payload.get("request_id") or ""),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "environment-remove":
            return svc.environment_remove(environment_id=str(payload.get("environment_id") or ""))
        if cmd == "toolbox-get-definition":
            actor = str(payload.get("_claim_actor_id") or "service:local")
            return svc.toolbox_get_definition(
                toolbox_id=str(payload.get("toolbox_id") or ""),
                owner_actor_id=actor,
                authority_id=actor,
            )
        if cmd == "toolbox-plan-definition":
            actor = str(payload.get("_claim_actor_id") or "service:local")
            return svc.toolbox_plan_definition(
                definition=dict(payload.get("definition") or {}),
                request_id=str(payload.get("request_id") or ""),
                operator_details=bool(payload.get("operator_details", False)),
                owner_actor_id=actor,
                authority_id=actor,
                ttl_ms=int(payload.get("ttl_ms") or 15 * 60 * 1000),
            )
        if cmd == "toolbox-confirm-definition-plan":
            actor = str(payload.get("_claim_actor_id") or "service:local")
            return svc.toolbox_confirm_definition_plan(
                plan_id=str(payload.get("plan_id") or ""),
                environment_choices=list(payload.get("environment_choices") or []),
                request_id=str(payload.get("request_id") or ""),
                owner_actor_id=actor,
                authority_id=actor,
            )
        if cmd == "toolbox-approve-confirmed-definition-plan":
            actor = str(payload.get("_claim_actor_id") or "service:local")
            return svc.toolbox_approve_confirmed_definition_plan(
                confirmation_ref=str(payload.get("confirmation_ref") or ""),
                approver_actor_id=actor,
                dependency_approver_authorized=True,
            )
        if cmd == "toolbox-apply-definition":
            actor = str(payload.get("_claim_actor_id") or "service:local")
            return svc.toolbox_apply_definition(
                plan_id=str(payload.get("plan_id") or ""),
                confirmation_ref=str(payload.get("confirmation_ref") or ""),
                request_id=str(payload.get("request_id") or ""),
                dependency_approval_ref=payload.get("dependency_approval_ref"),
                owner_actor_id=actor,
                authority_id=actor,
            )
        if cmd == "environment-template-list":
            return svc.environment_template_list(include_revoked=bool(payload.get("include_revoked", False)))
        if cmd == "package-artifact-upload-begin":
            return svc.package_artifact_upload_begin(
                actor_id=str(payload.get("_claim_actor_id") or "service:local"),
                source_id=str(payload.get("source_id") or ""),
                total_size=payload.get("total_size"),
                expected_digest=str(payload.get("expected_digest") or "").strip() or None,
                request_id=str(payload.get("request_id") or ""),
            )
        if cmd == "package-artifact-upload-chunk":
            return svc.package_artifact_upload_chunk(
                actor_id=str(payload.get("_claim_actor_id") or "service:local"),
                upload_id=str(payload.get("upload_id") or ""),
                chunk_index=payload.get("chunk_index"),
                offset=payload.get("offset"),
                chunk_base64url=str(payload.get("chunk_base64url") or ""),
            )
        if cmd == "package-artifact-upload-status":
            return svc.package_artifact_upload_status(
                actor_id=str(payload.get("_claim_actor_id") or "service:local"),
                upload_id=str(payload.get("upload_id") or ""),
            )
        if cmd == "package-artifact-upload-cancel":
            return svc.package_artifact_upload_cancel(
                actor_id=str(payload.get("_claim_actor_id") or "service:local"),
                upload_id=str(payload.get("upload_id") or ""),
                request_id=str(payload.get("request_id") or ""),
            )
        if cmd == "package-artifact-upload-commit":
            return svc.package_artifact_upload_commit(
                actor_id=str(payload.get("_claim_actor_id") or "service:local"),
                upload_id=str(payload.get("upload_id") or ""),
                request_id=str(payload.get("request_id") or ""),
            )
        if cmd == "package-lock-create":
            return svc.package_lock_create(
                lock_id=str(payload.get("lock_id") or ""),
                revision=payload.get("revision"),
                runtime_kind=str(payload.get("runtime_kind") or ""),
                platform=str(payload.get("platform") or ""),
                artifacts=list(payload.get("artifacts") or []),
                dependencies=list(payload.get("dependencies") or []),
            )
        if cmd == "environment-template-describe":
            return svc.environment_template_describe(template_id=str(payload.get("template_id") or ""), revision=payload.get("revision"))
        if cmd == "environment-template-construct":
            return svc.environment_template_construct(template=dict(payload.get("template") or {}))
        if cmd == "environment-template-activate":
            return svc.environment_template_activate(
                template_id=str(payload.get("template_id") or ""),
                revision=payload.get("revision"),
            )
        if cmd == "environment-template-replace":
            return svc.environment_template_replace(template=dict(payload.get("template") or {}))
        if cmd == "environment-template-deprecate":
            return svc.environment_template_deprecate(
                template_id=str(payload.get("template_id") or ""),
                revision=payload.get("revision"),
            )
        if cmd == "environment-template-revoke":
            return svc.environment_template_revoke(
                template_id=str(payload.get("template_id") or ""),
                revision=payload.get("revision"),
            )
        if cmd == "environment-template-prewarm":
            return svc.environment_template_prewarm(request=dict(payload.get("request") or {}))
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
                request_id=str(payload.get("request_id") or ""),
                toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()],
                only_inconsistent=bool(payload.get("only_inconsistent", True)),
                details=bool(payload.get("details", False)),
                apply=bool(payload.get("apply", False)),
                mutation_authorized=True,
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
            )
        if cmd == "toolbox-reconcile":
            return svc.toolbox_reconcile(
                request_id=str(payload.get("request_id") or ""),
                toolbox_ids=[str(item or "").strip() for item in list(payload.get("toolbox_ids") or []) if str(item or "").strip()],
                only_inconsistent=bool(payload.get("only_inconsistent", True)),
                details=bool(payload.get("details", False)),
                owner_actor_id=str(payload.get("_claim_actor_id") or "service:local"),
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
            return svc.auth_status(
                session_token=payload.get("session_token"),
                presented_ssh_binding=dict(payload.get("_ssh_session_binding") or {}),
            )
        if cmd == "hosting-setup-status":
            return svc.hosting_setup_summary()
        if cmd == "model-runtime-status":
            return svc.model_runtime_status()
        if cmd == "hosting-secure-state-status":
            return svc.hosting_secure_state_status()
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
        if cmd == "list-live-consumers":
            return self._list_live_consumers()
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
        if cmd == "host-capability-audit-list":
            return svc.host_capability_audit_list(
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
        if cmd == "auth-validate-session":
            return svc.auth_validate_session(
                token=str(payload.get("token") or payload.get("session_token") or ""),
                scope=str(payload.get("scope") or "control"),
                expected_key_id=payload.get("expected_key_id") or payload.get("key_id"),
                check_ssh_binding=bool(payload.get("check_ssh_binding", True)),
                presented_ssh_binding=dict(payload.get("_ssh_session_binding") or payload.get("ssh_binding") or {}),
            )
        if cmd == "auth-renew-session":
            return svc.auth_renew_session(
                token=str(payload.get("token") or payload.get("session_token") or ""),
                scope=str(payload.get("scope") or "control"),
                ttl_seconds=int(payload.get("ttl_seconds") or 900),
                presented_ssh_binding=dict(payload.get("_ssh_session_binding") or payload.get("ssh_binding") or {}),
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
            return svc.get_host_metrics(session_token=payload.get("session_token"))
        raise ValueError(f"Unknown command '{cmd}'")
