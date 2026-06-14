"""Workflow Python node-profile execution runtime.

This module owns node-profile Python execution. It intentionally does not call
the helper worker contract; callers pass normalized node requests and receive
node-shaped execution data plus streamable events.
"""
from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .._process_utils import hidden_subprocess_kwargs


NodeEventCallback = Callable[[str, Dict[str, Any]], None]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _python_executable_from_request(request: Dict[str, Any], *, fallback: Optional[str] = None) -> str:
    py = dict(request.get("python") or {})
    return _clean(py.get("python_executable")) or _clean(fallback) or sys.executable


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


def _child_source() -> str:
    return r'''
import builtins
import contextlib
import io
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
    "print": print,
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
    sys.__stdout__.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.__stdout__.flush()

def detail_from_error(err):
    tb = traceback.format_exception(type(err), err, err.__traceback__, limit=6)
    return {
        "message": str(err),
        "error_type": type(err).__name__,
        "traceback_summary": "".join(tb)[-4096:],
    }

def make_importer(allowlist):
    allowed = {str(item or "").strip().split(".", 1)[0] for item in allowlist if str(item or "").strip()}
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in allowed:
            raise ImportError(f"import not allowed: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)
    return guarded_import

def normalize_result(value):
    if isinstance(value, dict):
        return {
            "output": value.get("output") if "output" in value else value,
            "state_patch": value.get("state_patch") if isinstance(value.get("state_patch"), dict) else None,
            "artifacts": value.get("artifacts") if isinstance(value.get("artifacts"), list) else [],
            "progress": value.get("progress") if isinstance(value.get("progress"), dict) else None,
        }
    return {"output": value, "state_patch": None, "artifacts": [], "progress": None}

def main():
    try:
        req = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        send({"type": "error", "reason": "workflow_python_node_invalid_request", "detail": detail_from_error(exc)})
        return 0
    request_id = str(req.get("request_id") or "")
    source = str(req.get("module_source") or "")
    export_name = str(req.get("export_name") or req.get("operation") or "")
    allowlist = list(req.get("import_allowlist") or [])
    payload = req.get("payload")
    output_limit_bytes = max(1, int(req.get("output_limit_bytes") or 65536))
    builtins_row = dict(SAFE_BUILTINS)
    if allowlist:
        builtins_row["__import__"] = make_importer(allowlist)

    def progress(payload):
        row = payload if isinstance(payload, dict) else {"value": payload}
        send({"type": "progress", "request_id": request_id, "payload": row})

    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    globals_row = {
        "__builtins__": builtins_row,
        "__name__": "workflow_python_node_module",
        "progress": progress,
        "emit_progress": progress,
    }
    try:
        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
            exec(compile(source, "<workflow_python_node>", "exec"), globals_row, globals_row)
            fn = globals_row.get(export_name)
            if not callable(fn):
                send({
                    "type": "error",
                    "request_id": request_id,
                    "reason": "workflow_sandbox_export_not_found",
                    "detail": {"export_name": export_name},
                    "stdout": stdout_io.getvalue(),
                    "stderr": stderr_io.getvalue(),
                })
                return 0
            value = fn(payload)
        normalized = normalize_result(value)
        result_json = json.dumps(normalized.get("output"), ensure_ascii=False, separators=(",", ":"))
        if len(result_json.encode("utf-8")) > output_limit_bytes:
            send({
                "type": "error",
                "request_id": request_id,
                "reason": "workflow_sandbox_output_limit_exceeded",
                "detail": {"output_limit_bytes": output_limit_bytes},
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
            })
            return 0
        send({
            "type": "result",
            "request_id": request_id,
            "output": normalized.get("output"),
            "state_patch": normalized.get("state_patch"),
            "artifacts": normalized.get("artifacts") or [],
            "progress": normalized.get("progress"),
            "stdout": stdout_io.getvalue(),
            "stderr": stderr_io.getvalue(),
        })
    except Exception as exc:
        send({
            "type": "error",
            "request_id": request_id,
            "reason": "workflow_sandbox_runtime_error",
            "detail": detail_from_error(exc),
            "stdout": stdout_io.getvalue(),
            "stderr": stderr_io.getvalue(),
        })
    return 0

raise SystemExit(main())
'''.strip()


@dataclass
class WorkflowPythonNodeRuntime:
    request_id: str
    python_executable: str
    proc: subprocess.Popen[Any]
    _events: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    _reader: Optional[threading.Thread] = None

    @classmethod
    def start(cls, *, request: Dict[str, Any], python_executable: str) -> "WorkflowPythonNodeRuntime":
        proc = subprocess.Popen(
            [python_executable, "-u", "-c", _child_source()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            **hidden_subprocess_kwargs(),
        )
        runtime = cls(request_id=_clean(request.get("request_id")), python_executable=python_executable, proc=proc)
        runtime._reader = threading.Thread(target=runtime._read_stdout, daemon=True, name=f"workflow-python-node-{int(proc.pid or 0)}")
        runtime._reader.start()
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        proc.stdin.close()
        return runtime

    def _read_stdout(self) -> None:
        stream = self.proc.stdout
        if stream is None:
            return
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    self._events.put(row)
            except Exception as exc:
                self._events.put({"type": "error", "reason": "workflow_sandbox_invalid_json_output", "detail": {"message": str(exc)}})

    def cancel(self) -> bool:
        if self.proc.poll() is not None:
            return False
        try:
            self.proc.kill()
            return True
        except Exception:
            return False

    def wait(self, *, timeout_ms: int, on_event: Optional[NodeEventCallback] = None) -> Dict[str, Any]:
        deadline = time.monotonic() + (max(1, int(timeout_ms or 1)) / 1000.0)
        last_stdout = ""
        last_stderr = ""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.cancel()
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
                if on_event is not None:
                    on_event("progress", payload)
                continue
            if event_type == "result":
                last_stdout = str(row.get("stdout") or "")
                last_stderr = str(row.get("stderr") or "")
                return {
                    "ok": True,
                    "output": row.get("output"),
                    "state_patch": dict(row.get("state_patch") or {}) or None,
                    "artifacts": list(row.get("artifacts") or []),
                    "progress": dict(row.get("progress") or {}) or None,
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }
            if event_type == "error":
                last_stdout = str(row.get("stdout") or "")
                last_stderr = str(row.get("stderr") or "")
                return {
                    "ok": False,
                    "reason": _clean(row.get("reason")) or "workflow_sandbox_runtime_error",
                    "detail": dict(row.get("detail") or {}),
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }


class WorkflowPythonNodeRuntimeRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Dict[str, WorkflowPythonNodeRuntime] = {}

    def execute(
        self,
        request: Dict[str, Any],
        *,
        python_executable: Optional[str] = None,
        on_event: Optional[NodeEventCallback] = None,
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
        runtime = WorkflowPythonNodeRuntime.start(request=child_req, python_executable=executable)
        with self._lock:
            if request_id:
                self._active[request_id] = runtime
        try:
            return runtime.wait(timeout_ms=timeout_ms, on_event=on_event)
        finally:
            with self._lock:
                if request_id:
                    self._active.pop(request_id, None)

    def cancel(self, request_id: str) -> Dict[str, Any]:
        rid = _clean(request_id)
        if not rid:
            return {"status": "error", "reason": "request_id_required", "canceled": False}
        with self._lock:
            runtime = self._active.get(rid)
        if runtime is None:
            return {"status": "ok", "request_id": rid, "canceled": False, "reason": "request_not_found"}
        return {"status": "ok", "request_id": rid, "canceled": runtime.cancel(), "reason": "canceled"}


__all__ = ["WorkflowPythonNodeRuntimeRegistry"]
