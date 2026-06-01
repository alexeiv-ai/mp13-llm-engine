from __future__ import annotations

"""Temporary workflow Python helper compatibility worker.

New host-facing integrations should enter through the `workflow-python-*`
facade commands. This worker remains only as the current helper-profile process
implementation and is marked for removal or reduction to a thin entrypoint once
dependent callers migrate.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Dict, Optional

from ._process_utils import hidden_subprocess_kwargs

PROTOCOL_VERSION = 1
EXECUTION_CONTRACT = "hosting.workflow_helper.worker.v1"
SANDBOX_PROFILE = "workflow_python_helper_v1"
ALLOWED_OPERATIONS = {
    "default",
    "condition",
    "evaluate_condition",
    "routing_hint",
    "route_hint",
    "payload",
    "shape_payload",
}
_CALL_CAPACITY = max(1, int(str(os.environ.get("MP13_WORKFLOW_PYTHON_HELPER_CAPACITY") or "1").strip() or "1"))
_MAX_REQUESTS_PER_PROCESS = max(1, int(str(os.environ.get("MP13_WORKFLOW_PYTHON_HELPER_MAX_REQUESTS_PER_PROCESS") or "256").strip() or "256"))


class _ResizableCapacityGate:
    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity or 1))
        self._active = 0
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._lock:
                if self._active < self._capacity:
                    self._active += 1
                    return True
            if not blocking:
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def set_capacity(self, capacity: int) -> int:
        value = max(1, min(int(capacity or 1), 256))
        with self._lock:
            self._capacity = value
            return self._capacity

    def stats(self) -> Dict[str, int]:
        with self._lock:
            active = int(self._active)
            capacity = int(self._capacity)
        return {
            "capacity": capacity,
            "active_calls": active,
            "available_slots": max(0, capacity - active),
        }


_call_slots = _ResizableCapacityGate(_CALL_CAPACITY)


class _WorkflowPythonHelperRequestCanceled(Exception):
    pass


def _contract_name() -> str:
    return str(os.environ.get("MP13_WORKER_CONTRACT") or EXECUTION_CONTRACT).strip() or EXECUTION_CONTRACT


def _worker_id() -> str:
    return str(os.environ.get("MP13_WORKFLOW_HELPER_WORKER_ID") or os.environ.get("MP13_ENGINE_ID") or "workflow-python-helper").strip() or "workflow-python-helper"


def _python_executable() -> str:
    return str(os.environ.get("MP13_WORKFLOW_PYTHON") or sys.executable).strip() or sys.executable


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _python_version() -> Optional[str]:
    try:
        result = subprocess.run(
            [_python_executable(), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            **hidden_subprocess_kwargs(),
        )
        if int(result.returncode or 0) == 0:
            return (str(result.stdout or "").strip() or str(result.stderr or "").strip() or None)
    except Exception:
        return None
    return None


def _runtime(reason: Optional[str] = None, *, runtime_python: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    gate_stats = _call_slots.stats()
    py = dict(runtime_python or {})
    out = {
        "worker_id": _worker_id(),
        "engine_id": _worker_id(),
        "python_executable": str(py.get("python_executable") or _python_executable()),
        "python_source": str(py.get("python_source") or "worker"),
        "python_version": _python_version(),
        "sandbox_profile": SANDBOX_PROFILE,
        "contract": _contract_name(),
        "capacity": int(gate_stats["capacity"]),
        "active_calls": int(gate_stats["active_calls"]),
        "max_requests_per_process": _MAX_REQUESTS_PER_PROCESS,
    }
    if reason:
        out["reason"] = reason
    return out


def _failure(
    reason: str,
    *,
    detail: Optional[Dict[str, Any]] = None,
    started_at: Optional[float] = None,
    audit: Optional[Dict[str, Any]] = None,
    runtime_python: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000) if started_at is not None else None
    audit_row = {**dict(audit or {}), "elapsed_ms": elapsed_ms, "reason": reason}
    return {"ok": False, "reason": reason, "detail": dict(detail or {}), "runtime": _runtime(reason, runtime_python=runtime_python), "audit": audit_row}


def _success(result: Any, *, started_at: float, audit: Dict[str, Any], runtime_python: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    return {
        "ok": True,
        "result": result,
        "runtime": _runtime(runtime_python=runtime_python),
        "audit": {**dict(audit or {}), "elapsed_ms": elapsed_ms, "reason": None},
    }


def _audit_from_request(req: Dict[str, Any]) -> Dict[str, Any]:
    provenance = dict(req.get("provenance") or {})
    return {
        "package_id": str(req.get("package_id") or "").strip() or None,
        "workflow_id": str(req.get("workflow_id") or "").strip() or None,
        "package_source_digest": str(req.get("package_source_digest") or "").strip() or None,
        "module_sha256": str(req.get("module_sha256") or "").strip() or None,
        "source_path": str(req.get("source_path") or "").strip() or None,
        "request_id": str(req.get("request_id") or "").strip() or None,
        "operation": str(req.get("operation") or "").strip() or None,
        "export_name": str(req.get("export_name") or "").strip() or None,
        "session_id": str(provenance.get("session_id") or "").strip() or None,
        "context_id": str(provenance.get("context_id") or "").strip() or None,
        "cursor_id": str(provenance.get("cursor_id") or "").strip() or None,
        "workflow_root_id": str(provenance.get("workflow_root_id") or "").strip() or None,
        "worker_id": _worker_id(),
        "engine_id": _worker_id(),
    }


def _runtime_python_from_request(req: Dict[str, Any]) -> Dict[str, Any]:
    py = dict(req.get("python") or {})
    requested_name = str(py.get("environment_name") or "workflow-python-helper").strip() or "workflow-python-helper"
    return {
        "python_executable": str(py.get("python_executable") or _python_executable()).strip() or _python_executable(),
        "python_source": str(py.get("python_source") or "worker").strip() or "worker",
        "environment_name": requested_name,
        "import_allowlist": [str(item or "").strip() for item in list(py.get("import_allowlist") or []) if str(item or "").strip()],
        "package_pins": {
            str(k or "").strip(): str(v or "").strip()
            for k, v in dict(py.get("package_pins") or {}).items()
            if str(k or "").strip() and str(v or "").strip()
        },
    }


def _python_child_source() -> str:
    return r'''
import base64
import builtins
import importlib
import json
import sys
import traceback

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "Exception": Exception,
}

def send(row):
    sys.stdout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def detail_from_error(err):
    return {"message": str(err)}

def make_importer(allowlist):
    allowed = {str(item or "").strip().split(".", 1)[0] for item in allowlist if str(item or "").strip()}
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in allowed:
            raise ImportError(f"import not allowed: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)
    return guarded_import

def run_one(req):
    request_id = str(req.get("request_id") or "")
    export_name = str(req.get("export_name") or "")
    source = base64.b64decode(str(req.get("module_source_b64") or "").encode("ascii")).decode("utf-8")
    output_limit_bytes = max(1, int(req.get("output_limit_bytes") or 65536))
    allowlist = list(req.get("import_allowlist") or [])
    try:
        payload = json.loads(str(req.get("payload_json") or "null"))
    except Exception as exc:
        send({"request_id": request_id, "ok": False, "reason": "workflow_sandbox_invalid_result_shape", "detail": detail_from_error(exc)})
        return
    builtins_row = dict(SAFE_BUILTINS)
    if allowlist:
        builtins_row["__import__"] = make_importer(allowlist)
    globals_row = {"__builtins__": builtins_row, "__name__": "workflow_python_helper_module"}
    try:
        exec(compile(source, "<workflow_python_helper>", "exec"), globals_row, globals_row)
        fn = globals_row.get(export_name)
        if not callable(fn):
            send({"request_id": request_id, "ok": False, "reason": "workflow_sandbox_export_not_found", "detail": {"export_name": export_name}})
            return
        value = fn(payload)
        try:
            result_json = json.dumps(None if value is None else value, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            send({"request_id": request_id, "ok": False, "reason": "workflow_sandbox_invalid_json_output", "detail": detail_from_error(exc)})
            return
        if len(result_json.encode("utf-8")) > output_limit_bytes:
            send({"request_id": request_id, "ok": False, "reason": "workflow_sandbox_output_limit_exceeded", "detail": {"output_limit_bytes": output_limit_bytes}})
            return
        send({"request_id": request_id, "ok": True, "result_json": result_json})
    except Exception as exc:
        send({"request_id": request_id, "ok": False, "reason": "workflow_sandbox_runtime_error", "detail": detail_from_error(exc)})

for line in sys.stdin:
    if not line.strip():
        continue
    try:
        run_one(json.loads(line))
    except Exception as exc:
        send({"request_id": "", "ok": False, "reason": "workflow_sandbox_runtime_error", "detail": detail_from_error(exc)})
'''.strip()


class _HotPythonRuntime:
    def __init__(self, python_executable: Optional[str] = None) -> None:
        self._python_executable = str(python_executable or _python_executable()).strip() or _python_executable()
        self._responses: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._request_count = 0
        self._busy = False
        self._active_request_id = ""
        self._canceled_request_ids: set[str] = set()
        self._proc = subprocess.Popen(
            [self._python_executable, "-u", "-c", _python_child_source()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            **hidden_subprocess_kwargs(),
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True, name=f"workflow-python-helper-{int(self._proc.pid or 0)}")
        self._reader.start()

    def _read_stdout(self) -> None:
        stream = self._proc.stdout
        if stream is None:
            return
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
                self._responses.put(dict(row or {}) if isinstance(row, dict) else {"ok": False, "reason": "workflow_sandbox_invalid_json_output"})
            except Exception as exc:
                self._responses.put({"ok": False, "reason": "workflow_sandbox_invalid_json_output", "detail": {"message": str(exc)}})

    def alive(self) -> bool:
        return self._proc.poll() is None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "pid": int(self._proc.pid or 0),
            "alive": self.alive(),
            "python_executable": self._python_executable,
            "busy": bool(self._busy),
            "active_request_id": str(self._active_request_id or "") or None,
            "request_count": int(self._request_count),
            "max_requests": int(_MAX_REQUESTS_PER_PROCESS),
            "reusable": self.reusable(),
        }

    def execute(
        self,
        *,
        request_id: str,
        module_source: str,
        export_name: str,
        payload_json: str,
        import_allowlist: list[str],
        timeout_ms: int,
        output_limit_bytes: int,
    ) -> Dict[str, Any]:
        if not self.alive():
            raise RuntimeError("python_runtime_exited")
        row = {
            "request_id": request_id,
            "module_source_b64": base64.b64encode(module_source.encode("utf-8")).decode("ascii"),
            "export_name": export_name,
            "payload_json": payload_json,
            "import_allowlist": list(import_allowlist or []),
            "output_limit_bytes": int(output_limit_bytes),
        }
        deadline = time.monotonic() + (max(1, int(timeout_ms or 1)) / 1000.0)
        with self._lock:
            self._busy = True
            self._active_request_id = request_id
            try:
                try:
                    assert self._proc.stdin is not None
                    self._proc.stdin.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self._proc.stdin.flush()
                except Exception as exc:
                    raise RuntimeError(f"python_runtime_write_failed:{exc}") from exc
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.close(kill=True)
                        raise TimeoutError("workflow_sandbox_timeout")
                    try:
                        response = self._responses.get(timeout=min(remaining, 0.25))
                    except queue.Empty:
                        if not self.alive():
                            if request_id in self._canceled_request_ids:
                                self._canceled_request_ids.discard(request_id)
                                raise _WorkflowPythonHelperRequestCanceled("workflow_sandbox_canceled")
                            raise RuntimeError("python_runtime_exited")
                        continue
                    if str(response.get("request_id") or "") in {"", request_id}:
                        if request_id in self._canceled_request_ids:
                            self._canceled_request_ids.discard(request_id)
                            raise _WorkflowPythonHelperRequestCanceled("workflow_sandbox_canceled")
                        self._request_count += 1
                        return response
            finally:
                self._busy = False
                self._active_request_id = ""

    def reusable(self) -> bool:
        return self.alive() and self._request_count < _MAX_REQUESTS_PER_PROCESS

    def close(self, *, kill: bool = False) -> None:
        try:
            if self.alive():
                if kill:
                    self._proc.kill()
                else:
                    self._proc.terminate()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=1.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def cancel(self, request_id: str) -> bool:
        rid = str(request_id or "").strip()
        if not rid or str(self._active_request_id or "") != rid:
            return False
        self._canceled_request_ids.add(rid)
        self.close(kill=True)
        return True


class _HotPythonRuntimePool:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity or 1))
        self._lock = threading.Lock()
        self._idle: list[_HotPythonRuntime] = []
        self._all: list[_HotPythonRuntime] = []

    def _prune_locked(self) -> None:
        self._idle = [rt for rt in self._idle if rt.alive()]
        alive = []
        for rt in self._all:
            if rt.alive():
                alive.append(rt)
            else:
                rt.close(kill=True)
        self._all = alive

    def _checkout(self, python_executable: Optional[str] = None) -> _HotPythonRuntime:
        requested_python = str(python_executable or _python_executable()).strip() or _python_executable()
        with self._lock:
            self._prune_locked()
            remaining_idle: list[_HotPythonRuntime] = []
            while self._idle:
                rt = self._idle.pop()
                if rt.alive():
                    if str(getattr(rt, "_python_executable", "") or "") == requested_python:
                        self._idle.extend(remaining_idle)
                        return rt
                    remaining_idle.append(rt)
            self._idle = remaining_idle
            if len(self._all) < self.capacity:
                rt = _HotPythonRuntime(requested_python)
                self._all.append(rt)
                return rt
        raise RuntimeError("workflow_sandbox_capacity_exceeded")

    def _return(self, rt: _HotPythonRuntime) -> None:
        with self._lock:
            if rt.reusable() and rt in self._all:
                self._idle.append(rt)
            else:
                rt.close(kill=not rt.alive())
                self._all = [item for item in self._all if item is not rt]

    def set_capacity(self, capacity: int) -> int:
        value = max(1, min(int(capacity or 1), 256))
        with self._lock:
            self.capacity = value
            idle: list[_HotPythonRuntime] = []
            for rt in self._idle:
                if len(self._all) <= value:
                    idle.append(rt)
                else:
                    rt.close(kill=False)
                    self._all = [item for item in self._all if item is not rt]
            self._idle = idle
            return self.capacity

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            self._prune_locked()
            processes = [rt.snapshot() for rt in self._all]
            capacity = int(self.capacity)
        active = len([row for row in processes if bool(row.get("busy"))])
        alive = len([row for row in processes if bool(row.get("alive"))])
        return {
            "process_count": alive,
            "active_process_count": active,
            "idle_process_count": max(0, alive - active),
            "active_request_ids": [
                str(row.get("active_request_id") or "").strip()
                for row in processes
                if str(row.get("active_request_id") or "").strip()
            ],
            "processes": processes,
        }

    def execute(self, req: Dict[str, Any], *, export_name: str, payload_json: str, import_allowlist: list[str], timeout_ms: int, output_limit_bytes: int) -> Dict[str, Any]:
        runtime_python = _runtime_python_from_request(req)
        rt = self._checkout(str(runtime_python.get("python_executable") or _python_executable()))
        reusable = True
        request_id = str(req.get("request_id") or "").strip() or uuid.uuid4().hex
        try:
            return rt.execute(
                request_id=request_id,
                module_source=str(req.get("module_source") or ""),
                export_name=export_name,
                payload_json=payload_json,
                import_allowlist=import_allowlist,
                timeout_ms=timeout_ms,
                output_limit_bytes=output_limit_bytes,
            )
        except (TimeoutError, _WorkflowPythonHelperRequestCanceled):
            reusable = False
            raise
        except Exception:
            reusable = False
            raise
        finally:
            if reusable:
                self._return(rt)
            else:
                rt.close(kill=True)
                with self._lock:
                    self._all = [item for item in self._all if item is not rt]

    def close_all(self) -> None:
        with self._lock:
            runtimes = list(self._all)
            self._idle = []
            self._all = []
        for rt in runtimes:
            rt.close(kill=True)

    def cancel_request(self, request_id: str) -> Dict[str, Any]:
        rid = str(request_id or "").strip()
        if not rid:
            return {"status": "error", "reason": "request_id_required", "canceled": False}
        with self._lock:
            runtimes = list(self._all)
        for rt in runtimes:
            if rt.cancel(rid):
                with self._lock:
                    self._idle = [item for item in self._idle if item is not rt]
                    self._all = [item for item in self._all if item is not rt]
                return {"status": "ok", "request_id": rid, "canceled": True, "reason": "canceled"}
        return {"status": "ok", "request_id": rid, "canceled": False, "reason": "request_not_found"}


_PYTHON_POOL = _HotPythonRuntimePool(_CALL_CAPACITY)


def _worker_resources() -> Dict[str, Any]:
    gate = _call_slots.stats()
    pool = _PYTHON_POOL.stats()
    return {
        "status": "ok",
        "executor_kind": "workflow_python_helper",
        "worker_id": _worker_id(),
        "engine_id": _worker_id(),
        "python_executable": _python_executable(),
        "python_version": _python_version(),
        "sandbox_profile": SANDBOX_PROFILE,
        "capacity": int(gate["capacity"]),
        "active_calls": int(gate["active_calls"]),
        "available_slots": int(gate["available_slots"]),
        "pool": pool,
    }


def _set_capacity(value: int) -> Dict[str, Any]:
    capacity = _call_slots.set_capacity(value)
    _PYTHON_POOL.set_capacity(capacity)
    return _worker_resources()


def _cancel_request(request_id: str) -> Dict[str, Any]:
    return _PYTHON_POOL.cancel_request(request_id)


def _execute_python(req: Dict[str, Any], *, started_at: float) -> Dict[str, Any]:
    runtime_python = _runtime_python_from_request(req)
    audit = _audit_from_request(req)
    audit["runtime_python_path"] = runtime_python.get("python_executable")
    audit["runtime_python_source"] = runtime_python.get("python_source")
    module_source = str(req.get("module_source") or "")
    expected_sha = str(req.get("module_sha256") or "").strip().lower()
    if not expected_sha or _sha256_text(module_source).lower() != expected_sha:
        return _failure("workflow_sandbox_invalid_module_identity", started_at=started_at, audit=audit, runtime_python=runtime_python)
    operation = str(req.get("operation") or "").strip() or "default"
    if operation not in ALLOWED_OPERATIONS:
        return _failure("workflow_sandbox_operation_not_allowed", detail={"operation": operation}, started_at=started_at, audit=audit, runtime_python=runtime_python)
    export_name = str(req.get("export_name") or "").strip() or ("default" if operation == "default" else operation)
    limits = dict(req.get("limits") or {})
    timeout_ms = max(1, min(int(limits.get("timeout_ms") or 5000), 300000))
    output_limit_bytes = max(1, min(int(limits.get("output_limit_bytes") or 65536), 10 * 1024 * 1024))
    memory_limit_mb = limits.get("memory_limit_mb")
    audit["memory_limit"] = {
        "requested_mb": int(memory_limit_mb) if memory_limit_mb is not None else None,
        "enforcement": "best_effort_unavailable",
    }
    try:
        payload_json = json.dumps(req.get("payload"), ensure_ascii=False)
    except Exception as exc:
        return _failure("workflow_sandbox_invalid_result_shape", detail={"message": str(exc)}, started_at=started_at, audit=audit, runtime_python=runtime_python)
    try:
        response = _PYTHON_POOL.execute(
            dict(req or {}),
            export_name=export_name,
            payload_json=payload_json,
            import_allowlist=list(runtime_python.get("import_allowlist") or []),
            timeout_ms=timeout_ms,
            output_limit_bytes=output_limit_bytes,
        )
    except TimeoutError:
        return _failure("workflow_sandbox_timeout", detail={"timeout_ms": timeout_ms}, started_at=started_at, audit=audit, runtime_python=runtime_python)
    except _WorkflowPythonHelperRequestCanceled:
        return _failure("workflow_sandbox_canceled", detail={"request_id": str(req.get("request_id") or "").strip() or None}, started_at=started_at, audit=audit, runtime_python=runtime_python)
    except FileNotFoundError as exc:
        return _failure("workflow_sandbox_host_unavailable", detail={"message": str(exc)}, started_at=started_at, audit=audit, runtime_python=runtime_python)
    except Exception as exc:
        return _failure("workflow_sandbox_runtime_error", detail={"message": str(exc)}, started_at=started_at, audit=audit, runtime_python=runtime_python)
    if not bool(response.get("ok", False)):
        reason = str(response.get("reason") or "workflow_sandbox_runtime_error")
        detail = dict(response.get("detail") or {})
        if reason == "workflow_sandbox_export_not_found":
            detail.setdefault("export_name", export_name)
        return _failure(reason, detail=detail, started_at=started_at, audit=audit, runtime_python=runtime_python)
    stdout = str(response.get("result_json") or "null").encode("utf-8")
    if len(stdout) > output_limit_bytes:
        return _failure("workflow_sandbox_output_limit_exceeded", detail={"output_limit_bytes": output_limit_bytes}, started_at=started_at, audit=audit, runtime_python=runtime_python)
    try:
        parsed = json.loads(stdout.decode("utf-8") if stdout else "null")
    except Exception as exc:
        return _failure("workflow_sandbox_invalid_json_output", detail={"message": str(exc)}, started_at=started_at, audit=audit, runtime_python=runtime_python)
    return _success(parsed, started_at=started_at, audit=audit, runtime_python=runtime_python)


async def _execute_workflow_python_helper(req: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.monotonic()
    if not _call_slots.acquire(blocking=False):
        return _failure("workflow_sandbox_capacity_exceeded", started_at=started_at, audit=_audit_from_request(dict(req or {})), runtime_python=_runtime_python_from_request(dict(req or {})))
    try:
        return await asyncio.to_thread(_execute_python, dict(req or {}), started_at=started_at)
    finally:
        _call_slots.release()


async def _handle_hello(_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "pid": os.getpid(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "protocol_version": PROTOCOL_VERSION,
        "contract": _contract_name(),
        "execution_contract": EXECUTION_CONTRACT,
        "executor_kind": "workflow_python_helper",
        "sync_rpc": True,
        "async_rpc": False,
        "cancellation": True,
        "workflow_python_helper": {
            "available": bool(Path(_python_executable()).exists()),
            "python_executable": _python_executable(),
            "python_version": _python_version(),
            "sandbox_profile": SANDBOX_PROFILE,
            "capacity": int(_call_slots.stats()["capacity"]),
            "active_calls": int(_call_slots.stats()["active_calls"]),
            "max_requests_per_process": _MAX_REQUESTS_PER_PROCESS,
            "cancel_request": True,
        },
    }


async def _handle_rpc_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method") or "").strip()
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if method in {"rpc.describe", "describe", "capabilities"}:
        return await _handle_hello(payload)
    if method in {"worker.resources", "workflow_python_helper.resources"}:
        return {"status": "ok", "result": _worker_resources()}
    if method in {"workflow_python_helper.set_capacity", "worker.set_capacity"}:
        return {"status": "ok", "result": _set_capacity(int(dict(params or {}).get("capacity") or 1))}
    if method in {"workflow_python_helper.cancel_request", "worker.cancel_request"}:
        return {"status": "ok", "result": _cancel_request(str(dict(params or {}).get("request_id") or ""))}
    if method != "execute_workflow_python_helper":
        return {"status": "error", "message": "unsupported_method"}
    result = await _execute_workflow_python_helper(dict(params or {}))
    return {"status": "ok", "result": result}


def _handle_conn(conn: Any, stop_event: threading.Event) -> None:
    try:
        req = conn.recv()
        if not isinstance(req, dict):
            conn.send({"status": "error", "message": "invalid_request"})
            return
        kind = str(req.get("kind") or "").strip().lower()
        if kind == "shutdown":
            conn.send({"status": "ok"})
            stop_event.set()
            return
        if kind == "hello":
            conn.send(asyncio.run(_handle_hello(req)))
            return
        if kind == "rpc_call":
            conn.send(asyncio.run(_handle_rpc_call(req)))
            return
        conn.send({"status": "error", "message": "unsupported_kind"})
    except Exception as exc:
        try:
            conn.send({"status": "error", "message": f"worker_exception:{exc}"})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _serve_loop(*, family: str, address: str, authkey: bytes) -> int:
    listener = None
    unix_path = Path(address) if family == "AF_UNIX" else None
    stop_event = threading.Event()
    workers: list[threading.Thread] = []
    accepted: "queue.Queue[Any]" = queue.Queue()
    accept_errors: "queue.Queue[BaseException]" = queue.Queue()

    def _accept_loop() -> None:
        assert listener is not None
        while not stop_event.is_set():
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if stop_event.is_set():
                    break
                accept_errors.put(exc)
                break
            except Exception as exc:
                if stop_event.is_set():
                    break
                accept_errors.put(exc)
                break
            accepted.put(conn)

    if unix_path is not None:
        try:
            if unix_path.exists():
                unix_path.unlink()
        except Exception:
            pass
    accept_thread: Optional[threading.Thread] = None
    try:
        listener = Listener(address=address, family=family, authkey=authkey)
        try:
            raw_sock = getattr(getattr(listener, "_listener", None), "_socket", None)
            if raw_sock is not None:
                raw_sock.settimeout(0.5)
        except Exception:
            pass
        accept_thread = threading.Thread(target=_accept_loop, daemon=True)
        accept_thread.start()
        while not stop_event.is_set():
            try:
                if not accept_errors.empty():
                    raise accept_errors.get()
                conn = accepted.get(timeout=0.2)
            except queue.Empty:
                continue
            t = threading.Thread(target=_handle_conn, args=(conn, stop_event), daemon=True)
            t.start()
            workers.append(t)
        try:
            listener.close()
        except Exception:
            pass
    finally:
        _PYTHON_POOL.close_all()
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
        if accept_thread is not None:
            accept_thread.join(timeout=1.0)
        for t in workers[-256:]:
            t.join(timeout=0.5)
        if unix_path is not None:
            try:
                if unix_path.exists():
                    unix_path.unlink()
            except Exception:
                pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipc-family", required=True, choices=["AF_UNIX", "AF_PIPE"])
    ap.add_argument("--ipc-address", required=True)
    args = ap.parse_args()

    auth_token = str(os.environ.get("MP13_ENGINE_HOST_TOKEN") or "").strip()
    if not auth_token:
        print("Missing MP13_ENGINE_HOST_TOKEN", flush=True)
        return 2
    os.environ.setdefault("MP13_WORKER_CONTRACT", EXECUTION_CONTRACT)
    os.environ.setdefault("MP13_WORKFLOW_HELPER_WORKER_ID", str(os.environ.get("MP13_ENGINE_ID") or "workflow-python-helper"))
    return _serve_loop(
        family=str(args.ipc_family),
        address=str(args.ipc_address),
        authkey=auth_token.encode("utf-8", errors="ignore"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
