"""Workflow JavaScript node-profile execution runtime backed by QuickJS."""
from __future__ import annotations

import atexit
import asyncio
import hashlib
import inspect
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
_ACTIVE_JS_PROCS: set[subprocess.Popen[Any]] = set()
_ACTIVE_JS_PROCS_LOCK = threading.Lock()


def _remember_proc(proc: subprocess.Popen[Any]) -> None:
    with _ACTIVE_JS_PROCS_LOCK:
        _ACTIVE_JS_PROCS.add(proc)


def _forget_proc(proc: subprocess.Popen[Any]) -> None:
    with _ACTIVE_JS_PROCS_LOCK:
        _ACTIVE_JS_PROCS.discard(proc)


def _kill_active_js_procs() -> None:
    with _ACTIVE_JS_PROCS_LOCK:
        procs = list(_ACTIVE_JS_PROCS)
    for proc in procs:
        try:
            if proc.poll() is None:
                terminate_process_tree(int(proc.pid or 0), timeout_seconds=2.0)
        except Exception:
            pass


atexit.register(_kill_active_js_procs)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _python_executable_from_request(request: Dict[str, Any], *, fallback: Optional[str] = None) -> str:
    js = dict(request.get("javascript") or {})
    return _clean(js.get("python_executable")) or _clean(fallback) or sys.executable


def _allocate_js_ipc_address(request_id: str) -> tuple[str, str]:
    raw = str(request_id or "workflow-js-node")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_") or "workflow-js-node"
    nonce = secrets.token_hex(6)
    if os.name == "nt":
        return "AF_PIPE", f"\\\\.\\pipe\\mp13-js-node-{safe[:32]}-{nonce}"
    base = posixpath.abspath(posixpath.expanduser(str(tempfile.gettempdir() or "/tmp")))
    return "AF_UNIX", posixpath.join(base, f"mp13-js-node-{safe[:24]}-{nonce}.sock")


def _node_startup_timeout_seconds() -> float:
    try:
        return max(1.0, min(float(os.environ.get("MP13_WORKFLOW_JS_QUICKJS_STARTUP_TIMEOUT_SECONDS") or 30.0), 120.0))
    except Exception:
        return 30.0


@dataclass
class WorkflowJsNodeRuntime:
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
    _send_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def start(cls, *, runtime_key: str, python_executable: str) -> "WorkflowJsNodeRuntime":
        family, address = _allocate_js_ipc_address(str(runtime_key or "workflow-js-node"))
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
                "hosting.workflow_js_node_worker_ipc",
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

        accept_thread = threading.Thread(target=_accept, daemon=True, name=f"workflow-js-node-accept-{int(proc.pid or 0)}")
        accept_thread.start()
        try:
            accepted = accept_queue.get(timeout=_node_startup_timeout_seconds())
        except Exception:
            try:
                listener.close()
            except Exception:
                pass
            _forget_proc(proc)
            terminate_process_tree(int(proc.pid or 0), timeout_seconds=2.0)
            raise RuntimeError("workflow_js_node_worker_ipc_connect_timeout")
        if isinstance(accepted, BaseException):
            try:
                listener.close()
            except Exception:
                pass
            _forget_proc(proc)
            terminate_process_tree(int(proc.pid or 0), timeout_seconds=2.0)
            raise RuntimeError(f"workflow_js_node_worker_ipc_connect_failed:{accepted}") from accepted
        runtime = cls(python_executable=python_executable, proc=proc, conn=accepted)
        runtime._reader = threading.Thread(target=runtime._read_events, daemon=True, name=f"workflow-js-node-{int(proc.pid or 0)}")
        runtime._reader.start()
        return runtime

    @property
    def request_id(self) -> str:
        return self._request_id

    def alive(self) -> bool:
        return not self._closed and self.conn is not None and self.proc.poll() is None

    def send_request(self, request: Dict[str, Any]) -> None:
        if not self.alive():
            raise RuntimeError("workflow_js_node_worker_not_alive")
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
        with self._send_lock:
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
            with self._send_lock:
                self.conn.send(row)
            return True
        except Exception:
            return False

    def _dispatch_host_call(self, payload: Dict[str, Any], host_dispatcher: Optional[Any]) -> None:
        if not callable(host_dispatcher):
            self.respond_host_call(
                host_call_id=str(payload.get("host_call_id") or ""),
                error={"reason": "host_dispatcher_unavailable", "message": "host dispatcher is not available"},
            )
            return
        try:
            dispatched = host_dispatcher(dict(payload or {}))
            if inspect.isawaitable(dispatched):
                dispatched = asyncio.run(dispatched)
            self.respond_host_call(host_call_id=str(payload.get("host_call_id") or ""), result=dict(dispatched or {}))
        except Exception as exc:
            reason = str(getattr(exc, "reason", "") or "host_call_failed")
            detail = dict(getattr(exc, "detail", {}) or {})
            detail.setdefault("error_type", type(exc).__name__)
            self.respond_host_call(
                host_call_id=str(payload.get("host_call_id") or ""),
                error={"reason": reason, "message": str(getattr(exc, "message", "") or str(exc)), **detail},
            )

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
                with self._send_lock:
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
                    on_event(
                        "heartbeat",
                        {
                            "request_id": self.request_id,
                            "status": "running",
                            "elapsed_ms": max(0, int((now - started_at) * 1000)),
                            "remaining_ms": max(0, int((deadline - now) * 1000)),
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
                            "detail": {"message": "JS node runtime canceled"},
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
                        "detail": {"message": "JS node runtime exited without result"},
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
            if event_type == "console":
                payload = dict(row.get("payload") or {})
                if on_event is not None:
                    on_event("console", payload)
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
                threading.Thread(
                    target=self._dispatch_host_call,
                    args=(payload, host_dispatcher),
                    daemon=True,
                    name=f"workflow-js-node-host-call-{payload['host_call_id']}",
                ).start()
                continue
            if event_type == "canceled":
                self._wait_for_exit()
                self._busy = False
                return {
                    "ok": False,
                    "reason": "workflow_sandbox_canceled",
                    "detail": {"message": "JS node runtime canceled"},
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }
            if event_type == "process_exit":
                self._busy = False
                if self._cancel_requested:
                    return {
                        "ok": False,
                        "reason": "workflow_sandbox_canceled",
                        "detail": {"message": "JS node runtime canceled"},
                        "stdout": last_stdout,
                        "stderr": last_stderr,
                    }
                return {
                    "ok": False,
                    "reason": _clean(row.get("reason")) or "workflow_sandbox_runtime_error",
                    "detail": {"message": "JS node runtime exited without result"},
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
                    "runtime": dict(row.get("runtime") or {}),
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
                    "runtime": dict(row.get("runtime") or {}),
                }


class WorkflowJsNodeRuntimeRegistry(HostedActiveChildRuntimeRegistry):
    def __init__(self) -> None:
        super().__init__()
        self._instance_lock = threading.Lock()
        self._instances: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _runtime_key(request: Dict[str, Any], executable: str) -> str:
        js = dict(request.get("javascript") or {})
        environment_key = _clean(request.get("environment_key")) or _clean(js.get("environment_key")) or _clean(js.get("environment_name")) or "workflow-js-node"
        code_revision = _clean(request.get("code_revision")) or _clean(request.get("module_sha256")).lower() or _sha256_text(str(request.get("module_source") or ""))
        package_revision = _clean(request.get("package_source_digest"))
        runtime_hash = _clean(js.get("runtime_hash")) or "quickjs-default"
        return "|".join([_clean(executable), environment_key, runtime_hash, code_revision, package_revision])

    @staticmethod
    def _validate_module_identity(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        module_source = str(request.get("module_source") or "")
        expected_sha = _clean(request.get("module_sha256")).lower()
        if not expected_sha or _sha256_text(module_source).lower() != expected_sha:
            return {"ok": False, "reason": "workflow_sandbox_invalid_module_identity", "detail": {}}
        return None

    @staticmethod
    def _child_request(request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request or {})
        limits = dict(req.get("limits") or {})
        output_limit_bytes = max(1, min(int(limits.get("output_limit_bytes") or 65536), 10 * 1024 * 1024))
        return {
            **req,
            "request_id": _clean(req.get("request_id")),
            "export_name": _clean(req.get("export_name")) or "run",
            "output_limit_bytes": output_limit_bytes,
        }

    @staticmethod
    def _execution_mode(request: Dict[str, Any]) -> str:
        js = dict(request.get("javascript") or {})
        return _clean(request.get("execution_mode") or js.get("execution_mode") or "module").lower()

    @classmethod
    def _instance_reusable(cls, request: Dict[str, Any]) -> bool:
        return cls._execution_mode(request) != "project"

    @staticmethod
    def _project_instance_unsupported_detail() -> Dict[str, Any]:
        return {
            "message": (
                "JS project-mode pinned instances are deferred because workflow_js_instance_* "
                "pins the host worker process, while every request still creates a fresh QuickJS context."
            ),
            "deferred": True,
            "pinned_worker_semantics": "host_worker_process_only",
            "required_runtime_work": [
                "persistent QuickJS context or module graph cache",
                "declared cwd/env/import-cache cleanup policy",
                "snapshot/restore boundary for mutable JS state",
            ],
        }

    def create_instance(
        self,
        request: Dict[str, Any],
        *,
        instance_id: str = "",
        python_executable: Optional[str] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        child_req = self._child_request(request)
        validation_error = self._validate_module_identity(child_req)
        if validation_error is not None:
            return {"status": "error", **validation_error}
        if not self._instance_reusable(child_req):
            return {
                "status": "error",
                "ok": False,
                "reason": "workflow_js_instance_project_mode_unsupported",
                "detail": self._project_instance_unsupported_detail(),
            }
        executable = _clean(python_executable) or _python_executable_from_request(child_req)
        runtime_key = self._runtime_key(child_req, executable)
        iid = _clean(instance_id) or f"jsinst_{secrets.token_urlsafe(18)}"
        existing: Optional[Dict[str, Any]]
        with self._instance_lock:
            existing = self._instances.get(iid)
            if existing is not None and not bool(replace):
                return {"status": "error", "ok": False, "reason": "workflow_js_instance_exists", "instance_id": iid}
        if existing is not None:
            self.close_instance(iid, reason="replace")
        runtime = WorkflowJsNodeRuntime.start(runtime_key=runtime_key, python_executable=executable)
        now = time.time()
        row = {
            "instance_id": iid,
            "runtime_key": runtime_key,
            "environment_key": str(child_req.get("environment_key") or ""),
            "python_executable": executable,
            "pid": int(runtime.proc.pid or 0) or None,
            "created_at": now,
            "last_used_at": now,
            "active_request_id": None,
            "closed": False,
            "runtime": runtime,
        }
        with self._instance_lock:
            self._instances[iid] = row
        return {"status": "ok", "ok": True, **{key: value for key, value in row.items() if key != "runtime"}}

    def close_instance(self, instance_id: str, *, reason: str = "client_requested") -> Dict[str, Any]:
        iid = _clean(instance_id)
        if not iid:
            return {"status": "error", "reason": "instance_id_required", "closed": False}
        with self._instance_lock:
            row = self._instances.pop(iid, None)
        if row is None:
            return {"status": "not_found", "instance_id": iid, "closed": False}
        runtime = row.get("runtime")
        if runtime is not None:
            try:
                runtime.shutdown()
            except Exception:
                try:
                    runtime.ensure_stopped()
                except Exception:
                    pass
        return {
            "status": "ok",
            "instance_id": iid,
            "closed": True,
            "reason": _clean(reason) or "client_requested",
            "runtime_key": row.get("runtime_key"),
            "pid": row.get("pid"),
        }

    def list_instances(self) -> Dict[str, Any]:
        with self._instance_lock:
            rows = [dict(row or {}) for row in self._instances.values()]
        out: list[Dict[str, Any]] = []
        for row in rows:
            runtime = row.pop("runtime", None)
            row["alive"] = bool(runtime is not None and runtime.alive())
            if runtime is not None:
                row["pid"] = int(runtime.proc.pid or 0) or None
            out.append(row)
        out.sort(key=lambda item: str(item.get("instance_id") or ""))
        return {"status": "ok", "instances": out, "count": len(out)}

    def execute(
        self,
        request: Dict[str, Any],
        *,
        python_executable: Optional[str] = None,
        on_event: Optional[NodeEventCallback] = None,
        host_dispatcher: Optional[Any] = None,
        instance_id: str = "",
    ) -> Dict[str, Any]:
        req = dict(request or {})
        request_id = _clean(req.get("request_id"))
        validation_error = self._validate_module_identity(req)
        if validation_error is not None:
            return validation_error
        limits = dict(req.get("limits") or {})
        timeout_ms = max(1, min(int(limits.get("timeout_ms") or 5000), 300000))
        child_req = {**self._child_request(req), "request_id": request_id}
        executable = _clean(python_executable) or _python_executable_from_request(req)
        runtime_key = self._runtime_key(child_req, executable)
        runtime: Optional[WorkflowJsNodeRuntime] = None
        pinned_instance_id = _clean(instance_id)
        if pinned_instance_id:
            with self._instance_lock:
                instance_row = self._instances.get(pinned_instance_id)
                if instance_row is None:
                    return {"ok": False, "reason": "workflow_js_instance_not_found", "detail": {"instance_id": pinned_instance_id}}
                if _clean(instance_row.get("runtime_key")) != runtime_key:
                    return {
                        "ok": False,
                        "reason": "workflow_js_instance_incompatible_request",
                        "detail": {
                            "instance_id": pinned_instance_id,
                            "runtime_key": runtime_key,
                            "expected_runtime_key": instance_row.get("runtime_key"),
                        },
                    }
                if _clean(instance_row.get("active_request_id")):
                    return {"ok": False, "reason": "workflow_js_instance_busy", "detail": {"instance_id": pinned_instance_id}}
                runtime = instance_row.get("runtime")
                if runtime is None or not runtime.alive():
                    return {"ok": False, "reason": "workflow_js_instance_not_alive", "detail": {"instance_id": pinned_instance_id}}
                instance_row["active_request_id"] = request_id
                instance_row["last_used_at"] = time.time()
        else:
            last_start_error: Optional[Exception] = None
            for attempt in range(2):
                try:
                    runtime = WorkflowJsNodeRuntime.start(runtime_key=runtime_key, python_executable=executable)
                    break
                except Exception as exc:
                    last_start_error = exc
                    if attempt == 0 and "connect_timeout" in str(exc):
                        continue
                    break
            if runtime is None:
                return {
                    "ok": False,
                    "reason": "workflow_sandbox_host_unavailable",
                    "detail": {
                        "message": str(last_start_error or "workflow_js_node_worker_start_failed"),
                        "error_type": type(last_start_error).__name__ if last_start_error is not None else "RuntimeError",
                    },
                }
        try:
            runtime.send_request(child_req)
        except Exception:
            if pinned_instance_id:
                with self._instance_lock:
                    row = self._instances.get(pinned_instance_id)
                    if row is not None:
                        row["active_request_id"] = None
            runtime.ensure_stopped()
            raise
        self.register_active(request_id, runtime)
        try:
            return runtime.wait(timeout_ms=timeout_ms, on_event=on_event, host_dispatcher=host_dispatcher)
        finally:
            self.unregister_active(request_id)
            if pinned_instance_id:
                with self._instance_lock:
                    row = self._instances.get(pinned_instance_id)
                    if row is not None:
                        row["active_request_id"] = None
                        row["last_used_at"] = time.time()
            else:
                runtime.ensure_stopped()

    def resources(self) -> Dict[str, Any]:
        out = super().resources()
        out["instances"] = self.list_instances()
        return out

    def shutdown(self) -> None:
        for row in list(self.list_instances().get("instances") or []):
            self.close_instance(str(dict(row or {}).get("instance_id") or ""), reason="registry_shutdown")


__all__ = ["WorkflowJsNodeRuntimeRegistry"]
