"""Workflow Python node-profile execution runtime.

This module owns node-profile Python execution. It intentionally does not call
the helper worker contract; callers pass normalized node requests and receive
node-shaped execution data plus streamable events.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import posixpath
import queue
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Dict, Optional

from .._process_utils import hidden_subprocess_kwargs, terminate_process_tree
from .child_runtime import ChildRuntimeEventCallback, HostedActiveChildRuntimeRegistry


NodeEventCallback = ChildRuntimeEventCallback
_ACTIVE_NODE_PROCS: set[subprocess.Popen[Any]] = set()
_ACTIVE_NODE_PROCS_LOCK = threading.Lock()


def _remember_proc(proc: subprocess.Popen[Any]) -> None:
    with _ACTIVE_NODE_PROCS_LOCK:
        _ACTIVE_NODE_PROCS.add(proc)


def _forget_proc(proc: subprocess.Popen[Any]) -> None:
    with _ACTIVE_NODE_PROCS_LOCK:
        _ACTIVE_NODE_PROCS.discard(proc)


def _kill_active_node_procs() -> None:
    with _ACTIVE_NODE_PROCS_LOCK:
        procs = list(_ACTIVE_NODE_PROCS)
    for proc in procs:
        try:
            if proc.poll() is None:
                terminate_process_tree(int(proc.pid or 0), timeout_seconds=2.0)
        except Exception:
            pass


atexit.register(_kill_active_node_procs)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _python_executable_from_request(request: Dict[str, Any], *, fallback: Optional[str] = None) -> str:
    py = dict(request.get("python") or {})
    return _clean(py.get("python_executable")) or _clean(fallback) or sys.executable


def _allocate_node_ipc_address(request_id: str) -> tuple[str, str]:
    raw = str(request_id or "workflow-python-node")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_") or "workflow-python-node"
    nonce = secrets.token_hex(6)
    if os.name == "nt":
        return "AF_PIPE", f"\\\\.\\pipe\\mp13-node-{safe[:36]}-{nonce}"
    base = posixpath.abspath(posixpath.expanduser(str(tempfile.gettempdir() or "/tmp")))
    return "AF_UNIX", posixpath.join(base, f"mp13-node-{safe[:24]}-{nonce}.sock")


def _node_startup_timeout_seconds() -> float:
    try:
        return max(1.0, min(float(os.environ.get("MP13_WORKFLOW_PYTHON_NODE_STARTUP_TIMEOUT_SECONDS") or 30.0), 120.0))
    except Exception:
        return 30.0


def _import_allowlist(request: Dict[str, Any]) -> list[str]:
    py = dict(request.get("python") or {})
    out: list[str] = []
    seen: set[str] = set()
    for item in list(py.get("import_allowlist") or []):
        value = _clean(item)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


@dataclass
class WorkflowPythonNodeRuntime:
    python_executable: str
    proc: subprocess.Popen[Any]
    conn: Any = None
    _events: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    _reader: Optional[threading.Thread] = None
    _cancel_requested: bool = False
    _request_id: str = ""
    _busy: bool = False
    _closed: bool = False
    _heartbeat_interval_ms: int = 0

    @classmethod
    def start(cls, *, runtime_key: str, python_executable: str) -> "WorkflowPythonNodeRuntime":
        family, address = _allocate_node_ipc_address(str(runtime_key or "workflow-python-node"))
        auth_token = secrets.token_urlsafe(24)
        listener = Listener(address=address, family=family, authkey=auth_token.encode("utf-8"))
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = src_root if not existing_pythonpath else os.pathsep.join([src_root, existing_pythonpath])
        proc = subprocess.Popen(
            [
                python_executable,
                "-u",
                "-m",
                "hosting.workflow_python_node_worker_ipc",
                "--ipc-family",
                family,
                "--ipc-address",
                address,
                "--auth-token",
                auth_token,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
            **hidden_subprocess_kwargs(),
        )
        _remember_proc(proc)
        accept_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)

        def _accept() -> None:
            try:
                accept_queue.put(listener.accept())
            except Exception as exc:
                accept_queue.put(exc)
            finally:
                try:
                    listener.close()
                except Exception:
                    pass

        accept_thread = threading.Thread(target=_accept, daemon=True, name=f"workflow-python-node-accept-{int(proc.pid or 0)}")
        accept_thread.start()
        try:
            accepted = accept_queue.get(timeout=_node_startup_timeout_seconds())
        except Exception:
            _forget_proc(proc)
            terminate_process_tree(int(proc.pid or 0), timeout_seconds=2.0)
            raise RuntimeError("workflow_python_node_worker_ipc_connect_timeout")
        if isinstance(accepted, BaseException):
            _forget_proc(proc)
            terminate_process_tree(int(proc.pid or 0), timeout_seconds=2.0)
            raise RuntimeError(f"workflow_python_node_worker_ipc_connect_failed:{accepted}") from accepted
        runtime = cls(python_executable=python_executable, proc=proc, conn=accepted)
        runtime._reader = threading.Thread(target=runtime._read_events, daemon=True, name=f"workflow-python-node-{int(proc.pid or 0)}")
        runtime._reader.start()
        return runtime

    @property
    def request_id(self) -> str:
        return self._request_id

    def alive(self) -> bool:
        return not self._closed and self.conn is not None and self.proc.poll() is None

    def send_request(self, request: Dict[str, Any]) -> None:
        if not self.alive():
            raise RuntimeError("workflow_python_node_worker_not_alive")
        self._request_id = _clean(request.get("request_id"))
        self._cancel_requested = False
        self._busy = True
        limits = dict(request.get("limits") or {})
        self._heartbeat_interval_ms = max(0, int(limits.get("heartbeat_interval_ms") or 0))
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                break
        self.conn.send({"kind": "execute", "request": dict(request or {})})

    def respond_host_call(self, *, host_call_id: str, result: Optional[Dict[str, Any]] = None, error: Optional[Dict[str, Any]] = None) -> bool:
        if self.proc.poll() is not None or self.conn is None:
            return False
        row: Dict[str, Any] = {
            "type": "host_response",
            "host_call_id": str(host_call_id or ""),
        }
        if error is not None:
            row.update(
                {
                    "status": "error",
                    "reason": str(dict(error or {}).get("reason") or "host_call_failed"),
                    "message": str(dict(error or {}).get("message") or dict(error or {}).get("reason") or "host_call_failed"),
                    "detail": dict(error or {}),
                }
            )
        else:
            row.update({"status": "ok", "result": dict(result or {})})
        try:
            self.conn.send(row)
            return True
        except Exception:
            return False

    def _read_events(self) -> None:
        if self.conn is None:
            return
        while True:
            try:
                row = self.conn.recv()
                if isinstance(row, dict):
                    self._events.put(row)
            except Exception as exc:
                if self._cancel_requested:
                    self._events.put({"type": "canceled", "reason": "workflow_sandbox_canceled"})
                elif self.proc.poll() is None:
                    self._events.put({"type": "error", "reason": "workflow_sandbox_ipc_error", "detail": {"message": str(exc)}})
                else:
                    self._events.put({"type": "process_exit", "reason": "workflow_sandbox_process_exited"})
                return

    def _close_conn(self) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None
        self._closed = True

    def _kill(self) -> bool:
        if self.proc.poll() is not None:
            _forget_proc(self.proc)
            return False
        self._close_conn()
        try:
            result = terminate_process_tree(int(self.proc.pid or 0), timeout_seconds=2.0)
            return not bool(result.get("alive"))
        except Exception:
            try:
                self.proc.kill()
                return True
            except Exception:
                return False
        finally:
            _forget_proc(self.proc)

    def ensure_stopped(self) -> None:
        if self.proc.poll() is None:
            self._kill()
        else:
            _forget_proc(self.proc)
        self._close_conn()

    def shutdown(self) -> None:
        try:
            if self.alive() and self.conn is not None:
                self.conn.send({"kind": "shutdown"})
        except Exception:
            pass
        self.ensure_stopped()

    def _wait_for_exit(self, *, timeout_seconds: float = 1.0) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.wait(timeout=max(0.05, float(timeout_seconds or 1.0)))
        except Exception:
            self._kill()
            try:
                self.proc.wait(timeout=0.5)
            except Exception:
                pass
        finally:
            if self.proc.poll() is not None:
                _forget_proc(self.proc)

    def cancel(self) -> bool:
        self._cancel_requested = True
        killed = self._kill()
        self._events.put({"type": "canceled", "reason": "workflow_sandbox_canceled"})
        return killed

    def wait(
        self,
        *,
        timeout_ms: int,
        on_event: Optional[NodeEventCallback] = None,
        host_dispatcher: Optional[Any] = None,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + (max(1, int(timeout_ms or 1)) / 1000.0)
        last_stdout = ""
        last_stderr = ""
        last_progress: Optional[Dict[str, Any]] = None
        started_at = time.monotonic()
        heartbeat_interval = float(max(0, int(self._heartbeat_interval_ms or 0))) / 1000.0
        next_heartbeat = started_at + heartbeat_interval if heartbeat_interval > 0 else 0.0

        while True:
            now = time.monotonic()
            if heartbeat_interval > 0 and now >= next_heartbeat:
                if on_event is not None:
                    elapsed_ms = max(0, int((now - started_at) * 1000))
                    remaining_ms = max(0, int((deadline - now) * 1000))
                    on_event(
                        "heartbeat",
                        {
                            "request_id": self.request_id,
                            "status": "running",
                            "elapsed_ms": elapsed_ms,
                            "remaining_ms": remaining_ms,
                        },
                    )
                next_heartbeat = now + heartbeat_interval
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                self._wait_for_exit(timeout_seconds=0.5)
                self._busy = False
                return {
                    "ok": False,
                    "reason": "workflow_sandbox_timeout",
                    "detail": {"timeout_ms": timeout_ms},
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }
            try:
                row = self._events.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                if self.proc.poll() is not None:
                    if self._cancel_requested:
                        return {
                            "ok": False,
                            "reason": "workflow_sandbox_canceled",
                            "detail": {"message": "node runtime canceled"},
                            "stdout": last_stdout,
                            "stderr": last_stderr,
                        }
                    stderr = ""
                    try:
                        if self.proc.stderr is not None:
                            stderr = self.proc.stderr.read() or ""
                    except Exception:
                        stderr = ""
                    return {
                        "ok": False,
                        "reason": "workflow_sandbox_runtime_error",
                        "detail": {"message": "node runtime exited without result"},
                        "stdout": last_stdout,
                        "stderr": last_stderr or stderr,
                    }
                continue
            event_type = _clean(row.get("type"))
            if event_type == "progress":
                payload = dict(row.get("payload") or {})
                last_progress = payload
                if on_event is not None:
                    on_event("progress", payload)
                continue
            if event_type == "host_call":
                payload = {
                    "host_call_id": _clean(row.get("host_call_id")),
                    "method": _clean(row.get("method")),
                    "arguments": dict(row.get("arguments") or {}),
                    "request_id": _clean(row.get("request_id")) or self.request_id,
                }
                if on_event is not None:
                    on_event("host_call", payload)
                if callable(host_dispatcher):
                    try:
                        dispatched = dict(host_dispatcher(payload) or {})
                        self.respond_host_call(host_call_id=payload["host_call_id"], result=dispatched)
                    except Exception as exc:
                        self.respond_host_call(
                            host_call_id=payload["host_call_id"],
                            error={"reason": "host_call_failed", "message": str(exc), "error_type": type(exc).__name__},
                        )
                else:
                    self.respond_host_call(
                        host_call_id=payload["host_call_id"],
                        error={"reason": "host_dispatcher_unavailable", "message": "host dispatcher is not available"},
                    )
                continue
            if event_type == "canceled":
                self._wait_for_exit()
                self._busy = False
                return {
                    "ok": False,
                    "reason": "workflow_sandbox_canceled",
                    "detail": {"message": "node runtime canceled"},
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }
            if event_type == "process_exit":
                self._busy = False
                if self._cancel_requested:
                    return {
                        "ok": False,
                        "reason": "workflow_sandbox_canceled",
                        "detail": {"message": "node runtime canceled"},
                        "stdout": last_stdout,
                        "stderr": last_stderr,
                    }
                return {
                    "ok": False,
                    "reason": _clean(row.get("reason")) or "workflow_sandbox_runtime_error",
                    "detail": {"message": "node runtime exited without result"},
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }
            if event_type == "result":
                last_stdout = str(row.get("stdout") or "")
                last_stderr = str(row.get("stderr") or "")
                self._busy = False
                return {
                    "ok": True,
                    "output": row.get("output"),
                    "state_patch": dict(row.get("state_patch") or {}) or None,
                    "artifacts": list(row.get("artifacts") or []),
                    "progress": dict(row.get("progress") or {}) or last_progress,
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }
            if event_type == "error":
                last_stdout = str(row.get("stdout") or "")
                last_stderr = str(row.get("stderr") or "")
                self._busy = False
                return {
                    "ok": False,
                    "reason": _clean(row.get("reason")) or "workflow_sandbox_runtime_error",
                    "detail": dict(row.get("detail") or {}),
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }


class WorkflowPythonNodeRuntimeRegistry(HostedActiveChildRuntimeRegistry):
    def __init__(self) -> None:
        super().__init__()
        self._warm_lock = threading.Lock()
        self._idle: Dict[str, list[WorkflowPythonNodeRuntime]] = {}

    @staticmethod
    def _runtime_key(request: Dict[str, Any], executable: str) -> str:
        py = dict(request.get("python") or {})
        environment_key = _clean(request.get("environment_key")) or _clean(py.get("environment_key")) or _clean(py.get("environment_name")) or "workflow-python-node"
        return "|".join(
            [
                _clean(executable),
                environment_key,
                ",".join(_import_allowlist(request)),
            ]
        )

    @staticmethod
    def _runtime_key_environment(runtime_key: str) -> str:
        parts = str(runtime_key or "").split("|", 2)
        return parts[1] if len(parts) > 1 else ""

    @staticmethod
    def _warm_reusable(request: Dict[str, Any]) -> bool:
        return _clean(request.get("execution_mode") or "module").lower() != "project"

    def _take_idle(self, runtime_key: str) -> Optional[WorkflowPythonNodeRuntime]:
        with self._warm_lock:
            rows = self._idle.get(runtime_key, [])
            while rows:
                runtime = rows.pop()
                if runtime.alive():
                    return runtime
            self._idle.pop(runtime_key, None)
        return None

    def _release_idle(self, runtime_key: str, runtime: WorkflowPythonNodeRuntime, *, reusable: bool) -> None:
        if not reusable or not runtime.alive() or runtime._cancel_requested:
            runtime.ensure_stopped()
            return
        with self._warm_lock:
            self._idle.setdefault(runtime_key, []).append(runtime)

    def _shutdown_idle(self) -> None:
        with self._warm_lock:
            rows = [runtime for runtimes in self._idle.values() for runtime in runtimes]
            self._idle.clear()
        for runtime in rows:
            runtime.shutdown()

    def trim_idle(self, *, environment_key: str = "", max_idle: int = 0) -> Dict[str, Any]:
        env = _clean(environment_key)
        keep = max(0, int(max_idle or 0))
        stopped: list[Dict[str, Any]] = []
        runtimes_to_stop: list[WorkflowPythonNodeRuntime] = []
        with self._warm_lock:
            matching: list[tuple[str, WorkflowPythonNodeRuntime]] = []
            for key in list(self._idle.keys()):
                runtimes = [runtime for runtime in self._idle.get(key, []) if runtime.alive()]
                if runtimes:
                    self._idle[key] = runtimes
                else:
                    self._idle.pop(key, None)
                if env and self._runtime_key_environment(key) != env:
                    continue
                matching.extend((key, runtime) for runtime in runtimes)
            keep_ids = {id(runtime) for _, runtime in matching[-keep:]} if keep > 0 else set()
            for key in list(self._idle.keys()):
                kept: list[WorkflowPythonNodeRuntime] = []
                for runtime in self._idle.get(key, []):
                    if (not env or self._runtime_key_environment(key) == env) and id(runtime) not in keep_ids:
                        runtimes_to_stop.append(runtime)
                        stopped.append({"runtime_key": key, "pid": int(runtime.proc.pid or 0) or None})
                    else:
                        kept.append(runtime)
                if kept:
                    self._idle[key] = kept
                else:
                    self._idle.pop(key, None)
        for runtime in runtimes_to_stop:
            runtime.shutdown()
        return {"status": "ok", "environment_key": env or None, "max_idle": keep, "stopped": stopped, "stopped_count": len(stopped)}

    def execute(
        self,
        request: Dict[str, Any],
        *,
        python_executable: Optional[str] = None,
        on_event: Optional[NodeEventCallback] = None,
        host_dispatcher: Optional[Any] = None,
    ) -> Dict[str, Any]:
        req = dict(request or {})
        request_id = _clean(req.get("request_id"))
        module_source = str(req.get("module_source") or "")
        expected_sha = _clean(req.get("module_sha256")).lower()
        if not expected_sha or _sha256_text(module_source).lower() != expected_sha:
            return {"ok": False, "reason": "workflow_sandbox_invalid_module_identity", "detail": {}}
        limits = dict(req.get("limits") or {})
        timeout_ms = max(1, min(int(limits.get("timeout_ms") or 5000), 300000))
        output_limit_bytes = max(1, min(int(limits.get("output_limit_bytes") or 65536), 10 * 1024 * 1024))
        child_req = {
            **req,
            "request_id": request_id,
            "export_name": _clean(req.get("export_name")) or _clean(req.get("operation")),
            "import_allowlist": _import_allowlist(req),
            "output_limit_bytes": output_limit_bytes,
        }
        executable = _clean(python_executable) or _python_executable_from_request(req)
        runtime_key = self._runtime_key(child_req, executable)
        reusable = self._warm_reusable(child_req)
        runtime = self._take_idle(runtime_key) if reusable else None
        if runtime is None:
            runtime = WorkflowPythonNodeRuntime.start(runtime_key=runtime_key, python_executable=executable)
        try:
            runtime.send_request(child_req)
        except Exception:
            runtime.ensure_stopped()
            raise
        self.register_active(request_id, runtime)
        try:
            return runtime.wait(timeout_ms=timeout_ms, on_event=on_event, host_dispatcher=host_dispatcher)
        finally:
            self.unregister_active(request_id)
            self._release_idle(runtime_key, runtime, reusable=reusable)

    def resources(self) -> Dict[str, Any]:
        out = super().resources()
        with self._warm_lock:
            idle = [
                {
                    "runtime_key": key,
                    "pid": int(runtime.proc.pid or 0) or None,
                    "alive": runtime.alive(),
                    "python_executable": runtime.python_executable,
                }
                for key, runtimes in self._idle.items()
                for runtime in runtimes
            ]
        out["idle_count"] = len([row for row in idle if bool(row.get("alive"))])
        out["idle_processes"] = idle
        return out

    def shutdown(self) -> None:
        self._shutdown_idle()

    def __del__(self) -> None:
        try:
            self._shutdown_idle()
        except Exception:
            pass


__all__ = ["WorkflowPythonNodeRuntimeRegistry"]
