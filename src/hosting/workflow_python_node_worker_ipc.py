from __future__ import annotations

import argparse
import builtins
import contextlib
import importlib
import importlib.machinery
import io
import json
import os
import sys
import traceback
from multiprocessing.connection import Client
from typing import Any, Dict, Optional


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


def _detail_from_error(err: BaseException) -> Dict[str, Any]:
    tb = traceback.format_exception(type(err), err, err.__traceback__, limit=6)
    return {
        "message": str(err),
        "error_type": type(err).__name__,
        "traceback_summary": "".join(tb)[-4096:],
    }


def _under_root(path: Any, root: Any) -> bool:
    try:
        target = os.path.abspath(str(path or ""))
        base = os.path.abspath(str(root or ""))
        return target == base or target.startswith(base + os.sep)
    except Exception:
        return False


def _project_module_allowed(root: str, name: str) -> bool:
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


def _make_importer(allowlist: list[str], project_roots: Optional[list[str]] = None):
    allowed = {str(item or "").strip().split(".", 1)[0] for item in allowlist if str(item or "").strip()}
    roots = [os.path.abspath(str(item or "")) for item in list(project_roots or []) if str(item or "")]

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in allowed and not any(_project_module_allowed(project_root, root) for project_root in roots):
            raise ImportError(f"import not allowed: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    return guarded_import


def _normalize_result(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {
            "output": value.get("output") if "output" in value else value,
            "state_patch": value.get("state_patch") if isinstance(value.get("state_patch"), dict) else None,
            "artifacts": value.get("artifacts") if isinstance(value.get("artifacts"), list) else [],
            "progress": value.get("progress") if isinstance(value.get("progress"), dict) else None,
        }
    return {"output": value, "state_patch": None, "artifacts": [], "progress": None}


def _make_artifact_open(inputs: Dict[str, str], outputs: Dict[str, str]):
    readable = {os.path.abspath(str(path)) for path in dict(inputs or {}).values() if str(path or "")}
    writable = {os.path.abspath(str(path)) for path in dict(outputs or {}).values() if str(path or "")}

    def under_any(target: str, roots: set[str]) -> bool:
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
    def __init__(self, *, conn: Any, request_id: str) -> None:
        self.conn = conn
        self.request_id = str(request_id or "")
        self._seq = 0

    def call(self, method: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meth = str(method or "").strip()
        if not meth:
            raise RuntimeError("host_method_required")
        self._seq += 1
        call_id = f"{self.request_id}:{self._seq}"
        self.conn.send(
            {
                "type": "host_call",
                "request_id": self.request_id,
                "host_call_id": call_id,
                "method": meth,
                "arguments": dict(arguments or {}) if isinstance(arguments, dict) else {},
            }
        )
        response = dict(self.conn.recv() or {})
        if str(response.get("type") or "") != "host_response" or str(response.get("host_call_id") or "") != call_id:
            raise RuntimeError("host_response_mismatch")
        if str(response.get("status") or "").strip().lower() == "error":
            detail = response.get("detail") if isinstance(response.get("detail"), dict) else {}
            message = str(response.get("message") or detail.get("message") or response.get("reason") or "host_call_failed")
            raise RuntimeError(message)
        return dict(response.get("result") or {})

    def describe(self) -> Dict[str, Any]:
        return self.call("host.describe", {})

    def fs_read_text(self, root_id: str, relative_path: str = "", encoding: str = "utf-8") -> Dict[str, Any]:
        return self.call("fs.read_text", {"root_id": root_id, "relative_path": relative_path, "encoding": encoding})

    def fs_write_text(self, root_id: str, relative_path: str = "", text: str = "", encoding: str = "utf-8", create_parents: bool = True) -> Dict[str, Any]:
        return self.call(
            "fs.write_text",
            {"root_id": root_id, "relative_path": relative_path, "text": text, "encoding": encoding, "create_parents": bool(create_parents)},
        )

    def fs_list(self, root_id: str, relative_path: str = "") -> Dict[str, Any]:
        return self.call("fs.list", {"root_id": root_id, "relative_path": relative_path})

    def fs_stat(self, root_id: str, relative_path: str = "") -> Dict[str, Any]:
        return self.call("fs.stat", {"root_id": root_id, "relative_path": relative_path})

    def fs_mkdir(self, root_id: str, relative_path: str = "", parents: bool = True, exist_ok: bool = True) -> Dict[str, Any]:
        return self.call("fs.mkdir", {"root_id": root_id, "relative_path": relative_path, "parents": bool(parents), "exist_ok": bool(exist_ok)})


def _send(conn: Any, row: Dict[str, Any]) -> None:
    conn.send(dict(row or {}))


def _execute(conn: Any, req: Dict[str, Any]) -> int:
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
    project_roots: list[str] = []
    project_root = ""
    if execution_mode == "project":
        root_input = str(project.get("root_input") or project.get("input") or "project")
        project_root = os.path.abspath(str(artifact_inputs.get(root_input) or ""))
        if project_root:
            project_roots.append(project_root)
    output_limit_bytes = max(1, int(req.get("output_limit_bytes") or 65536))
    builtins_row = dict(SAFE_BUILTINS)
    builtins_row["__import__"] = _make_importer(allowlist, project_roots)
    if artifact_inputs or artifact_outputs:
        builtins_row["open"] = _make_artifact_open(artifact_inputs, artifact_outputs)

    def progress(progress_payload: Any) -> None:
        row = progress_payload if isinstance(progress_payload, dict) else {"value": progress_payload}
        _send(conn, {"type": "progress", "request_id": request_id, "payload": row})

    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    globals_row = {
        "__builtins__": builtins_row,
        "__name__": "workflow_python_node_module",
        "progress": progress,
        "emit_progress": progress,
        "host": HostApi(conn=conn, request_id=request_id),
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
                    _send(
                        conn,
                        {
                            "type": "error",
                            "request_id": request_id,
                            "reason": "workflow_sandbox_project_root_unavailable",
                            "detail": {"root_input": str(project.get("root_input") or project.get("input") or "project")},
                            "stdout": stdout_io.getvalue(),
                            "stderr": stderr_io.getvalue(),
                        },
                    )
                    return 0
                workdir = str(project.get("working_directory") or project.get("cwd") or "").strip().replace("\\", "/").strip("/")
                cwd = os.path.abspath(os.path.join(project_root, workdir)) if workdir else project_root
                if not _under_root(cwd, project_root) or not os.path.isdir(cwd):
                    _send(
                        conn,
                        {
                            "type": "error",
                            "request_id": request_id,
                            "reason": "workflow_sandbox_project_cwd_invalid",
                            "detail": {"working_directory": workdir},
                            "stdout": stdout_io.getvalue(),
                            "stderr": stderr_io.getvalue(),
                        },
                    )
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
                    _send(
                        conn,
                        {
                            "type": "error",
                            "request_id": request_id,
                            "reason": "workflow_sandbox_export_not_found",
                            "detail": {"export_name": callable_name, "module": module_name},
                            "stdout": stdout_io.getvalue(),
                            "stderr": stderr_io.getvalue(),
                        },
                    )
                    return 0
                value = fn(payload)
            else:
                exec(compile(source, "<workflow_python_node>", "exec"), globals_row, globals_row)
                fn = globals_row.get(export_name)
                if not callable(fn):
                    _send(
                        conn,
                        {
                            "type": "error",
                            "request_id": request_id,
                            "reason": "workflow_sandbox_export_not_found",
                            "detail": {"export_name": export_name},
                            "stdout": stdout_io.getvalue(),
                            "stderr": stderr_io.getvalue(),
                        },
                    )
                    return 0
                value = fn(payload)
        normalized = _normalize_result(value)
        result_json = json.dumps(normalized.get("output"), ensure_ascii=False, separators=(",", ":"))
        if len(result_json.encode("utf-8")) > output_limit_bytes:
            _send(
                conn,
                {
                    "type": "error",
                    "request_id": request_id,
                    "reason": "workflow_sandbox_output_limit_exceeded",
                    "detail": {"output_limit_bytes": output_limit_bytes},
                    "stdout": stdout_io.getvalue(),
                    "stderr": stderr_io.getvalue(),
                },
            )
            return 0
        _send(
            conn,
            {
                "type": "result",
                "request_id": request_id,
                "output": normalized.get("output"),
                "state_patch": normalized.get("state_patch"),
                "artifacts": normalized.get("artifacts") or [],
                "progress": normalized.get("progress"),
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
            },
        )
    except Exception as exc:
        _send(
            conn,
            {
                "type": "error",
                "request_id": request_id,
                "reason": "workflow_sandbox_runtime_error",
                "detail": _detail_from_error(exc),
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
            },
        )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-family", required=True, choices=["AF_UNIX", "AF_PIPE"])
    parser.add_argument("--ipc-address", required=True)
    parser.add_argument("--auth-token", default="")
    args = parser.parse_args(argv)
    conn = Client(args.ipc_address, family=args.ipc_family, authkey=str(args.auth_token or "").encode("utf-8"))
    try:
        req = dict(conn.recv() or {})
        return _execute(conn, req)
    except Exception as exc:
        try:
            _send(conn, {"type": "error", "reason": "workflow_sandbox_invalid_request", "detail": _detail_from_error(exc)})
        except Exception:
            pass
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
