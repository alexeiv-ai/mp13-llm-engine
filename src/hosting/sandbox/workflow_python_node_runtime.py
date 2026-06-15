"""Workflow Python node-profile execution runtime.

This module owns node-profile Python execution. It intentionally does not call
the helper worker contract; callers pass normalized node requests and receive
node-shaped execution data plus streamable events.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
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
import importlib
import importlib.machinery
import io
import json
import os
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

def recv():
    line = sys.__stdin__.readline()
    if not line:
        raise RuntimeError("host_channel_closed")
    try:
        row = json.loads(line)
    except Exception as exc:
        raise RuntimeError(f"host_channel_invalid_json:{exc}") from exc
    if not isinstance(row, dict):
        raise RuntimeError("host_channel_invalid_message")
    return row

def detail_from_error(err):
    tb = traceback.format_exception(type(err), err, err.__traceback__, limit=6)
    return {
        "message": str(err),
        "error_type": type(err).__name__,
        "traceback_summary": "".join(tb)[-4096:],
    }

def _under_root(path, root):
    try:
        target = os.path.abspath(str(path or ""))
        base = os.path.abspath(str(root or ""))
        return target == base or target.startswith(base + os.sep)
    except Exception:
        return False

def _project_module_allowed(root, name):
    root = os.path.abspath(str(root or ""))
    if not root or not os.path.isdir(root):
        return False
    module_root = str(name or "").split(".", 1)[0]
    try:
        spec = importlib.machinery.PathFinder.find_spec(module_root, [root])
    except Exception:
        return False
    if spec is None:
        return False
    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "frozen"} and _under_root(origin, root):
        return True
    for item in list(getattr(spec, "submodule_search_locations", None) or []):
        if _under_root(item, root):
            return True
    return False

def make_importer(allowlist, project_roots=None):
    allowed = {str(item or "").strip().split(".", 1)[0] for item in allowlist if str(item or "").strip()}
    roots = [os.path.abspath(str(item or "")) for item in list(project_roots or []) if str(item or "")]
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in allowed and not any(_project_module_allowed(project_root, root) for project_root in roots):
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

def make_artifact_open(inputs, outputs):
    readable = {os.path.abspath(str(path)) for path in dict(inputs or {}).values() if str(path or "")}
    writable = {os.path.abspath(str(path)) for path in dict(outputs or {}).values() if str(path or "")}
    def under_any(target, roots):
        for root in roots:
            if target == root:
                return True
            if os.path.isdir(root) and target.startswith(root + os.sep):
                return True
        return False

    def guarded_open(path, mode="r", *args, **kwargs):
        target = os.path.abspath(str(path or ""))
        write_mode = any(flag in str(mode or "") for flag in ("w", "a", "x", "+"))
        if write_mode:
            if not under_any(target, writable):
                raise PermissionError(f"artifact output path not allowed: {path}")
        elif not under_any(target, readable) and not under_any(target, writable):
            raise PermissionError(f"artifact input path not allowed: {path}")
        return builtins.open(target, mode, *args, **kwargs)

    return guarded_open

class HostApi:
    def __init__(self, request_id):
        self.request_id = str(request_id or "")
        self._seq = 0

    def call(self, method, arguments=None):
        meth = str(method or "").strip()
        if not meth:
            raise RuntimeError("host_method_required")
        self._seq += 1
        call_id = f"{self.request_id}:{self._seq}"
        send({
            "type": "host_call",
            "request_id": self.request_id,
            "host_call_id": call_id,
            "method": meth,
            "arguments": dict(arguments or {}) if isinstance(arguments, dict) else {},
        })
        response = recv()
        if str(response.get("type") or "") != "host_response" or str(response.get("host_call_id") or "") != call_id:
            raise RuntimeError("host_response_mismatch")
        if str(response.get("status") or "").strip().lower() == "error":
            detail = response.get("detail") if isinstance(response.get("detail"), dict) else {}
            message = str(response.get("message") or detail.get("message") or response.get("reason") or "host_call_failed")
            raise RuntimeError(message)
        return response.get("result")

    def describe(self):
        return self.call("host.describe", {})

    def fs_read_text(self, root_id, relative_path="", encoding="utf-8"):
        return self.call("fs.read_text", {"root_id": root_id, "relative_path": relative_path, "encoding": encoding})

    def fs_write_text(self, root_id, relative_path="", text="", encoding="utf-8", create_parents=True):
        return self.call("fs.write_text", {
            "root_id": root_id,
            "relative_path": relative_path,
            "text": text,
            "encoding": encoding,
            "create_parents": bool(create_parents),
        })

    def fs_list(self, root_id, relative_path=""):
        return self.call("fs.list", {"root_id": root_id, "relative_path": relative_path})

    def fs_stat(self, root_id, relative_path=""):
        return self.call("fs.stat", {"root_id": root_id, "relative_path": relative_path})

    def fs_mkdir(self, root_id, relative_path="", parents=True, exist_ok=True):
        return self.call("fs.mkdir", {"root_id": root_id, "relative_path": relative_path, "parents": bool(parents), "exist_ok": bool(exist_ok)})

def main():
    try:
        req = json.loads(sys.__stdin__.readline() or "{}")
    except Exception as exc:
        send({"type": "error", "reason": "workflow_python_node_invalid_request", "detail": detail_from_error(exc)})
        return 0
    request_id = str(req.get("request_id") or "")
    source = str(req.get("module_source") or "")
    export_name = str(req.get("export_name") or req.get("operation") or "")
    execution_mode = str(req.get("execution_mode") or "module").strip().lower() or "module"
    project = req.get("project") if isinstance(req.get("project"), dict) else {}
    allowlist = list(req.get("import_allowlist") or [])
    payload = req.get("payload")
    artifact_context = req.get("artifact_context") if isinstance(req.get("artifact_context"), dict) else {}
    artifact_inputs = artifact_context.get("inputs") if isinstance(artifact_context.get("inputs"), dict) else {}
    artifact_outputs = artifact_context.get("outputs") if isinstance(artifact_context.get("outputs"), dict) else {}
    project_roots = []
    project_root = ""
    if execution_mode == "project":
        root_input = str(project.get("root_input") or project.get("input") or "project")
        project_root = os.path.abspath(str(artifact_inputs.get(root_input) or ""))
        if project_root:
            project_roots.append(project_root)
    output_limit_bytes = max(1, int(req.get("output_limit_bytes") or 65536))
    builtins_row = dict(SAFE_BUILTINS)
    builtins_row["__import__"] = make_importer(allowlist, project_roots)
    if artifact_inputs or artifact_outputs:
        builtins_row["open"] = make_artifact_open(artifact_inputs, artifact_outputs)

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
        "host": HostApi(request_id),
        "artifact_inputs": dict(artifact_inputs or {}),
        "artifact_outputs": dict(artifact_outputs or {}),
        "payload": payload,
    }
    try:
        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
            if execution_mode == "snippet":
                exec(compile(source, "<workflow_python_snippet>", "exec"), globals_row, globals_row)
                value = globals_row.get("result")
            elif execution_mode == "project":
                if not project_root or not os.path.isdir(project_root):
                    send({
                        "type": "error",
                        "request_id": request_id,
                        "reason": "workflow_sandbox_project_root_unavailable",
                        "detail": {"root_input": str(project.get("root_input") or project.get("input") or "project")},
                        "stdout": stdout_io.getvalue(),
                        "stderr": stderr_io.getvalue(),
                    })
                    return 0
                workdir = str(project.get("working_directory") or project.get("cwd") or "").strip().replace("\\", "/").strip("/")
                cwd = os.path.abspath(os.path.join(project_root, workdir)) if workdir else project_root
                if not _under_root(cwd, project_root) or not os.path.isdir(cwd):
                    send({
                        "type": "error",
                        "request_id": request_id,
                        "reason": "workflow_sandbox_project_cwd_invalid",
                        "detail": {"working_directory": workdir},
                        "stdout": stdout_io.getvalue(),
                        "stderr": stderr_io.getvalue(),
                    })
                    return 0
                env = project.get("env") if isinstance(project.get("env"), dict) else {}
                for key, val in env.items():
                    if str(key or "").strip():
                        os.environ[str(key)] = str(val)
                sys.path.insert(0, project_root)
                os.chdir(cwd)
                module_name = str(project.get("entrypoint") or project.get("module") or "").strip()
                callable_name = str(project.get("callable") or project.get("function") or export_name or "run").strip()
                module = importlib.import_module(module_name)
                fn = getattr(module, callable_name, None)
                if not callable(fn):
                    send({
                        "type": "error",
                        "request_id": request_id,
                        "reason": "workflow_sandbox_export_not_found",
                        "detail": {"export_name": callable_name, "module": module_name},
                        "stdout": stdout_io.getvalue(),
                        "stderr": stderr_io.getvalue(),
                    })
                    return 0
                value = fn(payload)
            else:
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
    _cancel_requested: bool = False

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
        _remember_proc(proc)
        runtime = cls(request_id=_clean(request.get("request_id")), python_executable=python_executable, proc=proc)
        runtime._reader = threading.Thread(target=runtime._read_stdout, daemon=True, name=f"workflow-python-node-{int(proc.pid or 0)}")
        runtime._reader.start()
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        return runtime

    def respond_host_call(self, *, host_call_id: str, result: Optional[Dict[str, Any]] = None, error: Optional[Dict[str, Any]] = None) -> bool:
        if self.proc.poll() is not None or self.proc.stdin is None:
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
            self.proc.stdin.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
            return True
        except Exception:
            return False

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

    def _kill(self) -> bool:
        if self.proc.poll() is not None:
            _forget_proc(self.proc)
            return False
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

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                self._wait_for_exit(timeout_seconds=0.5)
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
                return {
                    "ok": False,
                    "reason": "workflow_sandbox_canceled",
                    "detail": {"message": "node runtime canceled"},
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }
            if event_type == "result":
                last_stdout = str(row.get("stdout") or "")
                last_stderr = str(row.get("stderr") or "")
                self._wait_for_exit()
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
                self._wait_for_exit()
                return {
                    "ok": False,
                    "reason": _clean(row.get("reason")) or "workflow_sandbox_runtime_error",
                    "detail": dict(row.get("detail") or {}),
                    "stdout": last_stdout,
                    "stderr": last_stderr,
                }


class WorkflowPythonNodeRuntimeRegistry(HostedActiveChildRuntimeRegistry):
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
        runtime = WorkflowPythonNodeRuntime.start(request=child_req, python_executable=executable)
        self.register_active(request_id, runtime)
        try:
            return runtime.wait(timeout_ms=timeout_ms, on_event=on_event, host_dispatcher=host_dispatcher)
        finally:
            runtime.ensure_stopped()
            self.unregister_active(request_id)


__all__ = ["WorkflowPythonNodeRuntimeRegistry"]
