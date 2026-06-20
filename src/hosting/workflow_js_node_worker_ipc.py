from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
import time
import traceback
from multiprocessing.connection import Client
from typing import Any, Dict, Optional


class HostApiError(RuntimeError):
    def __init__(self, *, reason: str, message: str, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.reason = str(reason or "host_call_failed")
        self.message = str(message or self.reason)
        self.detail = dict(detail or {})


def _detail_from_error(err: BaseException) -> Dict[str, Any]:
    tb = traceback.format_exception(type(err), err, err.__traceback__, limit=6)
    return {
        "message": str(err),
        "error_type": type(err).__name__,
        "traceback_summary": "".join(tb)[-4096:],
    }


def _send(conn: Any, row: Dict[str, Any]) -> None:
    conn.send(dict(row or {}))


class HostApiBridge:
    def __init__(self, *, conn: Any, request_id: str) -> None:
        self.conn = conn
        self.request_id = str(request_id or "")
        self.last_error: Optional[HostApiError] = None
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending_responses: Dict[str, Dict[str, Any]] = {}

    def _next_call_id(self) -> str:
        with self._seq_lock:
            self._seq += 1
            return f"{self.request_id}:{self._seq}"

    @staticmethod
    def _result_from_response(response: Dict[str, Any]) -> str:
        if str(response.get("status") or "").strip().lower() == "error":
            detail = response.get("detail") if isinstance(response.get("detail"), dict) else {}
            message = str(response.get("message") or detail.get("message") or response.get("reason") or "host_call_failed")
            raise HostApiError(
                reason=str(response.get("reason") or detail.get("reason") or "host_call_failed"),
                message=message,
                detail={**dict(detail or {}), "host_call_id": str(response.get("host_call_id") or "")},
            )
        return json.dumps(dict(response.get("result") or {}), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _arguments_from_json(arguments_json: str = "{}") -> Dict[str, Any]:
        try:
            arguments = json.loads(str(arguments_json or "{}"))
        except Exception as exc:
            raise RuntimeError(f"host_arguments_invalid_json:{exc}") from exc
        return dict(arguments or {}) if isinstance(arguments, dict) else {}

    def send_call(self, method: str, arguments_json: str = "{}") -> str:
        meth = str(method or "").strip()
        if not meth:
            raise RuntimeError("host_method_required")
        arguments = self._arguments_from_json(arguments_json)
        call_id = self._next_call_id()
        with self._send_lock:
            self.conn.send(
                {
                    "type": "host_call",
                    "request_id": self.request_id,
                    "host_call_id": call_id,
                    "method": meth,
                    "arguments": arguments,
                }
            )
        return call_id

    def _read_host_response(self, *, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if timeout is not None and not self.conn.poll(max(0.0, float(timeout))):
            return None
        row = dict(self.conn.recv() or {})
        if str(row.get("type") or "") != "host_response":
            raise RuntimeError("host_response_mismatch")
        response_id = str(row.get("host_call_id") or "")
        if not response_id:
            raise RuntimeError("host_response_mismatch")
        return row

    def _pop_pending_response(self, call_id: str) -> Optional[Dict[str, Any]]:
        return self._pending_responses.pop(str(call_id or ""), None)

    def poll_response(self, *, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        if self._pending_responses:
            first_key = next(iter(self._pending_responses))
            return self._pending_responses.pop(first_key)
        return self._read_host_response(timeout=timeout)

    def call_json(self, method: str, arguments_json: str = "{}") -> str:
        call_id = self.send_call(method, arguments_json)
        pending = self._pop_pending_response(call_id)
        if pending is not None:
            try:
                return self._result_from_response(pending)
            except HostApiError as exc:
                self.last_error = exc
                raise
        while True:
            row = self._read_host_response()
            if row is None:
                continue
            response_id = str(row.get("host_call_id") or "")
            if response_id == call_id:
                try:
                    return self._result_from_response(row)
                except HostApiError as exc:
                    self.last_error = exc
                    raise
            self._pending_responses[response_id] = row


def _pump_quickjs_until_settled(
    *,
    ctx: Any,
    host: HostApiBridge,
    timeout_ms: int,
) -> str:
    deadline = time.monotonic() + (max(1, int(timeout_ms or 1)) / 1000.0)
    max_jobs_per_turn = 1000

    while True:
        ran_job = False
        for _ in range(max_jobs_per_turn):
            if not ctx.execute_pending_job():
                break
            ran_job = True

        if bool(ctx.get("__workflow_async_settled")):
            result_json = ctx.get("__workflow_result_json")
            if not isinstance(result_json, str):
                raise RuntimeError("workflow_sandbox_invalid_json_output")
            return result_json

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return json.dumps(
                {
                    "__workflow_error": {
                        "reason": "workflow_sandbox_timeout",
                        "detail": {"timeout_ms": timeout_ms},
                    }
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

        response = host.poll_response(timeout=0.0 if ran_job else min(remaining, 0.01))
        if response is not None:
            ctx.set("__workflow_host_response_json", json.dumps(response, ensure_ascii=False, separators=(",", ":")))
            ctx.eval("__workflowHandleHostResponse(globalThis.__workflow_host_response_json)")


def _normalize_console_args(args_json: str) -> str:
    try:
        args = json.loads(str(args_json or "[]"))
    except Exception:
        args = [str(args_json or "")]
    if not isinstance(args, list):
        args = [args]
    return " ".join(str(item) for item in args)


def _runtime_metadata(*, memory_limit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        import quickjs  # type: ignore

        return {
            "quickjs_binding": "quickjs",
            "quickjs_binding_version": str(getattr(quickjs, "__version__", "") or "unknown"),
            "quickjs_available": True,
            "memory_limit": dict(memory_limit or {"requested_mb": None, "enforced": False, "reason": "not_requested"}),
        }
    except Exception as exc:
        return {
            "quickjs_binding": "quickjs",
            "quickjs_binding_version": None,
            "quickjs_available": False,
            "quickjs_error": str(exc),
            "memory_limit": dict(memory_limit or {"requested_mb": None, "enforced": False, "reason": "runtime_unavailable"}),
        }


def _execute_quickjs(conn: Any, req: Dict[str, Any]) -> int:
    request_id = str(req.get("request_id") or "")
    source = str(req.get("module_source") or "")
    export_name = str(req.get("export_name") or "run").strip() or "run"
    execution_mode = str(req.get("execution_mode") or "script").strip().lower() or "script"
    payload = req.get("payload")
    output_limit_bytes = max(1, int(req.get("output_limit_bytes") or 65536))
    limits = dict(req.get("limits") or {})
    timeout_ms = max(1, int(limits.get("timeout_ms") or 5000))
    memory_limit_mb = limits.get("memory_limit_mb")

    try:
        import quickjs  # type: ignore
    except Exception as exc:
        _send(
            conn,
            {
                "type": "error",
                "request_id": request_id,
                "reason": "workflow_sandbox_host_unavailable",
                "detail": {"message": str(exc), "runtime": _runtime_metadata()},
                "stdout": "",
                "stderr": "",
            },
        )
        return 0

    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    host = HostApiBridge(conn=conn, request_id=request_id)
    ctx = quickjs.Context()
    memory_limit_report: Dict[str, Any] = {
        "requested_mb": int(memory_limit_mb) if memory_limit_mb is not None else None,
        "enforced": False,
        "reason": "not_requested" if memory_limit_mb is None else "not_enforced",
    }
    # The Python quickjs binding cannot call Python callbacks while a context
    # time limit is active. Host APIs, console, and progress all use callbacks,
    # so wall-clock limits are enforced by the parent process for this slice.
    if memory_limit_mb is not None:
        try:
            ctx.set_memory_limit(max(1, int(memory_limit_mb)) * 1024 * 1024)
            memory_limit_report["enforced"] = True
            memory_limit_report["reason"] = None
        except Exception as exc:
            memory_limit_report["reason"] = f"unavailable:{type(exc).__name__}"

    def _host_call(method: str, args_json: str = "{}") -> str:
        return host.call_json(method, args_json)

    def _host_call_async(method: str, args_json: str = "{}") -> str:
        return host.send_call(method, args_json)

    def _progress(payload_json: str = "{}") -> None:
        try:
            payload_row = json.loads(str(payload_json or "{}"))
        except Exception:
            payload_row = {"value": str(payload_json or "")}
        if not isinstance(payload_row, dict):
            payload_row = {"value": payload_row}
        _send(conn, {"type": "progress", "request_id": request_id, "payload": payload_row})

    def _console(level: str, args_json: str = "[]") -> None:
        message = _normalize_console_args(args_json)
        line = f"{str(level or 'log')}: {message}"
        stdout_io.write(line + "\n")
        _send(conn, {"type": "console", "request_id": request_id, "payload": {"level": str(level or "log"), "message": message}})

    def _base64_encode(value: str = "") -> str:
        return base64.b64encode(str(value or "").encode("utf-8")).decode("ascii")

    def _base64_decode(value: str = "") -> str:
        return base64.b64decode(str(value or "").encode("ascii"), validate=True).decode("utf-8")

    def _sha256(value: str = "") -> str:
        return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()

    ctx.add_callable("__host_call", _host_call)
    ctx.add_callable("__host_call_async", _host_call_async)
    ctx.add_callable("__progress", _progress)
    ctx.add_callable("__console", _console)
    ctx.add_callable("__base64_encode", _base64_encode)
    ctx.add_callable("__base64_decode", _base64_decode)
    ctx.add_callable("__sha256", _sha256)
    ctx.set("__payload_json", json.dumps(payload, ensure_ascii=False))
    ctx.set("__export_name", export_name)
    prelude = r"""
globalThis.exports = {};
globalThis.payload = JSON.parse(globalThis.__payload_json || "null");
globalThis.progress = function (value) { __progress(JSON.stringify(value === undefined ? null : value)); };
globalThis.emitProgress = globalThis.progress;
globalThis.console = {
  log: function (...args) { __console("log", JSON.stringify(args)); },
  info: function (...args) { __console("info", JSON.stringify(args)); },
  warn: function (...args) { __console("warn", JSON.stringify(args)); },
  error: function (...args) { __console("error", JSON.stringify(args)); }
};
globalThis.__workflowPendingHostCalls = {};
globalThis.__workflow_async_settled = false;
globalThis.__workflow_result_json = "";
globalThis.__workflowHostError = function (row) {
  const detail = row && row.detail && typeof row.detail === "object" ? row.detail : {};
  const message = String((row && (row.message || detail.message || row.reason)) || "host_call_failed");
  const err = new Error(message);
  err.reason = String((row && (row.reason || detail.reason)) || "host_call_failed");
  err.detail = detail;
  err.host_call_id = String((row && row.host_call_id) || "");
  return err;
};
globalThis.__workflowHandleHostResponse = function (rowJson) {
  const row = JSON.parse(String(rowJson || "{}"));
  const callId = String(row.host_call_id || "");
  const pending = globalThis.__workflowPendingHostCalls[callId];
  if (!pending) {
    return;
  }
  delete globalThis.__workflowPendingHostCalls[callId];
  if (String(row.status || "").toLowerCase() === "error") {
    pending.reject(globalThis.__workflowHostError(row));
    return;
  }
  pending.resolve(row.result || {});
};
globalThis.api = {
  call: function (method, args) { return JSON.parse(__host_call(String(method || ""), JSON.stringify(args || {}))); },
  callAsync: function (method, args) {
    const callId = String(__host_call_async(String(method || ""), JSON.stringify(args || {})));
    return new Promise(function (resolve, reject) {
      globalThis.__workflowPendingHostCalls[callId] = {resolve: resolve, reject: reject};
    });
  },
  describe: function () { return this.call("host.describe", {}); },
  describeAsync: function () { return this.callAsync("host.describe", {}); },
  progress: globalThis.progress,
  fs: {
    readText: function (rootId, relativePath, encoding) {
      return api.call("fs.read_text", {root_id: String(rootId || ""), relative_path: String(relativePath || ""), encoding: String(encoding || "utf-8")}).text;
    },
    writeText: function (rootId, relativePath, text, encoding) {
      return api.call("fs.write_text", {root_id: String(rootId || ""), relative_path: String(relativePath || ""), text: String(text || ""), encoding: String(encoding || "utf-8"), create_parents: true});
    },
    readTextAsync: function (rootId, relativePath, encoding) {
      return api.callAsync("fs.read_text", {root_id: String(rootId || ""), relative_path: String(relativePath || ""), encoding: String(encoding || "utf-8")}).then(function (result) { return result.text; });
    },
    writeTextAsync: function (rootId, relativePath, text, encoding) {
      return api.callAsync("fs.write_text", {root_id: String(rootId || ""), relative_path: String(relativePath || ""), text: String(text || ""), encoding: String(encoding || "utf-8"), create_parents: true});
    },
    list: function (rootId, relativePath) { return api.call("fs.list", {root_id: String(rootId || ""), relative_path: String(relativePath || "")}); },
    listAsync: function (rootId, relativePath) { return api.callAsync("fs.list", {root_id: String(rootId || ""), relative_path: String(relativePath || "")}); },
    stat: function (rootId, relativePath) { return api.call("fs.stat", {root_id: String(rootId || ""), relative_path: String(relativePath || "")}); },
    statAsync: function (rootId, relativePath) { return api.callAsync("fs.stat", {root_id: String(rootId || ""), relative_path: String(relativePath || "")}); },
    mkdir: function (rootId, relativePath, options) {
      options = options || {};
      return api.call("fs.mkdir", {root_id: String(rootId || ""), relative_path: String(relativePath || ""), parents: options.parents !== false, exist_ok: options.exist_ok !== false});
    },
    mkdirAsync: function (rootId, relativePath, options) {
      options = options || {};
      return api.callAsync("fs.mkdir", {root_id: String(rootId || ""), relative_path: String(relativePath || ""), parents: options.parents !== false, exist_ok: options.exist_ok !== false});
    }
  },
  http: {
    fetch: function (url, options) {
      options = options || {};
      return api.call("http.fetch", {url: String(url || ""), method: String(options.method || "GET"), headers: options.headers || {}, body_b64: String(options.body_b64 || ""), timeout_seconds: Number(options.timeout_seconds || 30), max_response_bytes: Number(options.max_response_bytes || 1048576)});
    },
    fetchAsync: function (url, options) {
      options = options || {};
      return api.callAsync("http.fetch", {url: String(url || ""), method: String(options.method || "GET"), headers: options.headers || {}, body_b64: String(options.body_b64 || ""), timeout_seconds: Number(options.timeout_seconds || 30), max_response_bytes: Number(options.max_response_bytes || 1048576)});
    }
  },
  codec: {
    base64Encode: function (text) { return __base64_encode(String(text || "")); },
    base64Decode: function (text) { return __base64_decode(String(text || "")); }
  },
  crypto: {
    sha256: function (text) { return __sha256(String(text || "")); }
  }
};
""".strip()
    runner = r"""
(function () {
  function errorDetail(err) {
    const detail = err && err.detail && typeof err.detail === "object" ? err.detail : {};
    const message = String((err && err.message) || err || "workflow_sandbox_runtime_error");
    const out = {message: message};
    if (err && err.reason) out.reason = String(err.reason);
    if (err && err.host_call_id) out.host_call_id = String(err.host_call_id);
    if (err && err.stack) out.stack = String(err.stack);
    for (const key in detail) out[key] = detail[key];
    return out;
  }
  function normalize(value) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return {
        output: Object.prototype.hasOwnProperty.call(value, "output") ? value.output : value,
        state_patch: value.state_patch && typeof value.state_patch === "object" && !Array.isArray(value.state_patch) ? value.state_patch : null,
        artifacts: Array.isArray(value.artifacts) ? value.artifacts : [],
        progress: value.progress && typeof value.progress === "object" && !Array.isArray(value.progress) ? value.progress : null
      };
    }
    return {output: value === undefined ? null : value, state_patch: null, artifacts: [], progress: null};
  }
  function settleOk(value) {
    try {
      globalThis.__workflow_result_json = JSON.stringify(normalize(value === undefined ? null : value));
    } catch (err) {
      globalThis.__workflow_result_json = JSON.stringify({__workflow_error: {reason: "workflow_sandbox_invalid_output", detail: errorDetail(err)}});
    }
    globalThis.__workflow_async_settled = true;
  }
  function settleErr(err) {
    const reason = String((err && err.reason) || "workflow_sandbox_runtime_error");
    globalThis.__workflow_result_json = JSON.stringify({__workflow_error: {reason: reason, detail: errorDetail(err)}});
    globalThis.__workflow_async_settled = true;
  }
  let value;
  if ("%EXECUTION_MODE%" === "snippet") {
    value = globalThis.result;
  } else {
    const fn = exports[globalThis.__export_name || "run"];
    if (typeof fn !== "function") {
      return JSON.stringify({__workflow_error: {reason: "workflow_sandbox_export_not_found", detail: {export_name: globalThis.__export_name || "run"}}});
    }
    value = fn(globalThis.payload, globalThis.api);
  }
  try {
    if (value && typeof value.then === "function") {
      globalThis.__workflow_async_settled = false;
      globalThis.__workflow_result_json = "";
      Promise.resolve(value).then(settleOk, settleErr);
      return "__workflow_pending__";
    }
    return JSON.stringify(normalize(value === undefined ? null : value));
  } catch (err) {
    return JSON.stringify({__workflow_error: {reason: "workflow_sandbox_invalid_output", detail: errorDetail(err)}});
  }
})()
""".replace("%EXECUTION_MODE%", execution_mode)

    try:
        ctx.eval(prelude)
        ctx.eval(source)
        result_json = ctx.eval(runner)
        if result_json == "__workflow_pending__":
            result_json = _pump_quickjs_until_settled(ctx=ctx, host=host, timeout_ms=timeout_ms)
        if not isinstance(result_json, str):
            raise RuntimeError("workflow_sandbox_invalid_json_output")
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
        normalized = json.loads(result_json)
        if isinstance(normalized, dict) and isinstance(normalized.get("__workflow_error"), dict):
            error = dict(normalized.get("__workflow_error") or {})
            _send(
                conn,
                {
                    "type": "error",
                    "request_id": request_id,
                    "reason": str(error.get("reason") or "workflow_sandbox_runtime_error"),
                    "detail": dict(error.get("detail") or {}),
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
                "output": normalized.get("output") if isinstance(normalized, dict) else normalized,
                "state_patch": dict(normalized.get("state_patch") or {}) if isinstance(normalized, dict) else None,
                "artifacts": list(normalized.get("artifacts") or []) if isinstance(normalized, dict) else [],
                "progress": dict(normalized.get("progress") or {}) if isinstance(normalized, dict) and isinstance(normalized.get("progress"), dict) else None,
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
                "runtime": _runtime_metadata(memory_limit=memory_limit_report),
            },
        )
    except Exception as exc:
        host_error = host.last_error
        reason = host_error.reason if host_error is not None else "workflow_sandbox_runtime_error"
        detail = (
            {
                "message": host_error.message,
                "reason": host_error.reason,
                **dict(host_error.detail or {}),
            }
            if host_error is not None
            else _detail_from_error(exc)
        )
        _send(
            conn,
            {
                "type": "error",
                "request_id": request_id,
                "reason": reason,
                "detail": detail,
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
                "runtime": _runtime_metadata(memory_limit=memory_limit_report),
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
        while True:
            msg = conn.recv()
            if not isinstance(msg, dict):
                _send(conn, {"type": "error", "reason": "workflow_sandbox_invalid_request", "detail": {"message": "request must be an object"}})
                continue
            kind = str(msg.get("kind") or "").strip().lower()
            if kind == "shutdown":
                _send(conn, {"type": "shutdown_ack"})
                return 0
            req = dict(msg.get("request") or {}) if kind == "execute" else dict(msg or {})
            _execute_quickjs(conn, req)
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
