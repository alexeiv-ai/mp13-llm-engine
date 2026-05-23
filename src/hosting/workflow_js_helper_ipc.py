from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Dict, Optional

from ._process_utils import hidden_subprocess_kwargs

PROTOCOL_VERSION = 1
EXECUTION_CONTRACT = "hosting.workflow_helper.worker.v1"
SANDBOX_PROFILE = "workflow_js_helper_v1"
ALLOWED_OPERATIONS = {
    "default",
    "condition",
    "evaluate_condition",
    "routing_hint",
    "route_hint",
    "payload",
    "shape_payload",
}
_CALL_CAPACITY = max(1, int(str(os.environ.get("MP13_WORKFLOW_JS_HELPER_CAPACITY") or "1").strip() or "1"))
_MAX_REQUESTS_PER_NODE = max(1, int(str(os.environ.get("MP13_WORKFLOW_JS_HELPER_MAX_REQUESTS_PER_NODE") or "256").strip() or "256"))


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


def _contract_name() -> str:
    return str(os.environ.get("MP13_WORKER_CONTRACT") or EXECUTION_CONTRACT).strip() or EXECUTION_CONTRACT


def _worker_id() -> str:
    return str(os.environ.get("MP13_WORKFLOW_HELPER_WORKER_ID") or os.environ.get("MP13_ENGINE_ID") or "workflow-js-helper").strip() or "workflow-js-helper"


def _node_executable() -> str:
    return str(os.environ.get("MP13_WORKFLOW_JS_NODE") or shutil.which("node") or "node").strip() or "node"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _node_version() -> Optional[str]:
    try:
        result = subprocess.run(  # noqa: S603
            [_node_executable(), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            **hidden_subprocess_kwargs(),
        )
        if int(result.returncode or 0) == 0:
            return str(result.stdout or "").strip() or None
    except Exception:
        return None
    return None


def _runtime(reason: Optional[str] = None) -> Dict[str, Any]:
    gate_stats = _call_slots.stats()
    out = {
        "worker_id": _worker_id(),
        "engine_id": _worker_id(),
        "node_executable": _node_executable(),
        "node_version": _node_version(),
        "sandbox_profile": SANDBOX_PROFILE,
        "contract": _contract_name(),
        "capacity": int(gate_stats["capacity"]),
        "active_calls": int(gate_stats["active_calls"]),
        "max_requests_per_node": _MAX_REQUESTS_PER_NODE,
    }
    if reason:
        out["reason"] = reason
    return out


def _failure(reason: str, *, detail: Optional[Dict[str, Any]] = None, started_at: Optional[float] = None) -> Dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000) if started_at is not None else None
    audit = {"elapsed_ms": elapsed_ms, "reason": reason}
    return {"ok": False, "reason": reason, "detail": dict(detail or {}), "runtime": _runtime(reason), "audit": audit}


def _success(result: Any, *, started_at: float, audit: Dict[str, Any]) -> Dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    return {
        "ok": True,
        "result": result,
        "runtime": _runtime(),
        "audit": {**dict(audit or {}), "elapsed_ms": elapsed_ms, "reason": None},
    }


def _audit_from_request(req: Dict[str, Any]) -> Dict[str, Any]:
    provenance = dict(req.get("provenance") or {})
    return {
        "package_id": str(req.get("package_id") or "").strip() or None,
        "workflow_id": str(req.get("workflow_id") or "").strip() or None,
        "package_source_digest": str(req.get("package_source_digest") or "").strip() or None,
        "module_sha256": str(req.get("module_sha256") or "").strip() or None,
        "operation": str(req.get("operation") or "").strip() or None,
        "export_name": str(req.get("export_name") or "").strip() or None,
        "session_id": str(provenance.get("session_id") or "").strip() or None,
        "context_id": str(provenance.get("context_id") or "").strip() or None,
        "cursor_id": str(provenance.get("cursor_id") or "").strip() or None,
        "workflow_root_id": str(provenance.get("workflow_root_id") or "").strip() or None,
        "worker_id": _worker_id(),
        "engine_id": _worker_id(),
    }


def _node_worker_source() -> str:
    return """
import { createInterface } from 'node:readline';

const originalStdoutWrite = process.stdout.write.bind(process.stdout);
const originalStderrWrite = process.stderr.write.bind(process.stderr);
const encoder = new TextEncoder();

function send(row) {
  originalStdoutWrite(JSON.stringify(row) + '\\n');
}

function detailFromError(err) {
  return { message: String((err && err.message) || err) };
}

async function runOne(req) {
  const requestId = String(req.request_id || '');
  const exportName = String(req.export_name || '');
  const sourceB64 = String(req.module_source_b64 || '');
  const payloadJson = String(req.payload_json || 'null');
  const outputLimitBytes = Math.max(1, Number(req.output_limit_bytes || 65536));
  let payload;
  try {
    payload = JSON.parse(payloadJson);
  } catch (err) {
    send({ request_id: requestId, ok: false, reason: 'workflow_sandbox_invalid_result_shape', detail: detailFromError(err) });
    return;
  }
  const previousStdoutWrite = process.stdout.write;
  process.stdout.write = (...args) => {
    try { originalStderrWrite(...args); } catch (_) {}
    return true;
  };
  try {
    const moduleUrl = `data:text/javascript;base64,${sourceB64}#${encodeURIComponent(requestId)}`;
    const mod = await import(moduleUrl);
    const fn = mod[exportName];
    if (typeof fn !== 'function') {
      send({ request_id: requestId, ok: false, reason: 'workflow_sandbox_export_not_found', detail: { export_name: exportName } });
      return;
    }
    const value = await fn(payload);
    let resultJson;
    try {
      resultJson = JSON.stringify(value === undefined ? null : value);
    } catch (err) {
      send({ request_id: requestId, ok: false, reason: 'workflow_sandbox_invalid_json_output', detail: detailFromError(err) });
      return;
    }
    if (encoder.encode(resultJson || '').byteLength > outputLimitBytes) {
      send({ request_id: requestId, ok: false, reason: 'workflow_sandbox_output_limit_exceeded', detail: { output_limit_bytes: outputLimitBytes } });
      return;
    }
    send({ request_id: requestId, ok: true, result_json: resultJson });
  } catch (err) {
    send({ request_id: requestId, ok: false, reason: 'workflow_sandbox_runtime_error', detail: detailFromError(err) });
  } finally {
    process.stdout.write = previousStdoutWrite;
  }
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!String(line || '').trim()) {
    continue;
  }
  try {
    await runOne(JSON.parse(line));
  } catch (err) {
    send({ request_id: '', ok: false, reason: 'workflow_sandbox_runtime_error', detail: detailFromError(err) });
  }
}
""".strip()


class _HotNodeRuntime:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mp13-workflow-js-node-")
        self._tmp_path = Path(self._tmp.name)
        self._worker_path = self._tmp_path / "worker.mjs"
        self._worker_path.write_text(_node_worker_source(), encoding="utf-8")
        self._responses: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._request_count = 0
        self._busy = False
        self._proc = subprocess.Popen(  # noqa: S603
            [_node_executable(), str(self._worker_path)],
            cwd=str(self._tmp_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            **hidden_subprocess_kwargs(),
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True, name=f"workflow-js-node-{int(self._proc.pid or 0)}")
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
            "busy": bool(self._busy),
            "request_count": int(self._request_count),
            "max_requests": int(_MAX_REQUESTS_PER_NODE),
            "reusable": self.reusable(),
        }

    def execute(
        self,
        *,
        request_id: str,
        module_source: str,
        export_name: str,
        payload_json: str,
        timeout_ms: int,
        output_limit_bytes: int,
    ) -> Dict[str, Any]:
        if not self.alive():
            raise RuntimeError("node_runtime_exited")
        source_b64 = base64.b64encode(module_source.encode("utf-8")).decode("ascii")
        row = {
            "request_id": request_id,
            "module_source_b64": source_b64,
            "export_name": export_name,
            "payload_json": payload_json,
            "output_limit_bytes": int(output_limit_bytes),
        }
        deadline = time.monotonic() + (max(1, int(timeout_ms or 1)) / 1000.0)
        with self._lock:
            self._busy = True
            try:
                try:
                    assert self._proc.stdin is not None
                    self._proc.stdin.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self._proc.stdin.flush()
                except Exception as exc:
                    raise RuntimeError(f"node_runtime_write_failed:{exc}") from exc
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.close(kill=True)
                        raise TimeoutError("workflow_sandbox_timeout")
                    try:
                        response = self._responses.get(timeout=min(remaining, 0.25))
                    except queue.Empty:
                        if not self.alive():
                            raise RuntimeError("node_runtime_exited")
                        continue
                    if str(response.get("request_id") or "") in {"", request_id}:
                        self._request_count += 1
                        return response
            finally:
                self._busy = False

    def reusable(self) -> bool:
        return self.alive() and self._request_count < _MAX_REQUESTS_PER_NODE

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
        try:
            self._tmp.cleanup()
        except Exception:
            pass


class _HotNodeRuntimePool:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity or 1))
        self._lock = threading.Lock()
        self._idle: list[_HotNodeRuntime] = []
        self._all: list[_HotNodeRuntime] = []

    def _prune_locked(self) -> None:
        self._idle = [rt for rt in self._idle if rt.alive()]
        alive = []
        for rt in self._all:
            if rt.alive():
                alive.append(rt)
            else:
                rt.close(kill=True)
        self._all = alive

    def _checkout(self) -> _HotNodeRuntime:
        with self._lock:
            self._prune_locked()
            while self._idle:
                rt = self._idle.pop()
                if rt.alive():
                    return rt
            if len(self._all) < self.capacity:
                rt = _HotNodeRuntime()
                self._all.append(rt)
                return rt
        raise RuntimeError("workflow_sandbox_capacity_exceeded")

    def _return(self, rt: _HotNodeRuntime) -> None:
        with self._lock:
            if rt.reusable() and rt in self._all:
                self._idle.append(rt)
            else:
                try:
                    rt.close(kill=not rt.alive())
                except Exception:
                    pass
                self._all = [item for item in self._all if item is not rt]

    def set_capacity(self, capacity: int) -> int:
        value = max(1, min(int(capacity or 1), 256))
        with self._lock:
            self.capacity = value
            idle: list[_HotNodeRuntime] = []
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
            nodes = [rt.snapshot() for rt in self._all]
            capacity = int(self.capacity)
        active = len([row for row in nodes if bool(row.get("busy"))])
        alive = len([row for row in nodes if bool(row.get("alive"))])
        idle = max(0, alive - active)
        return {
            "status": "ok",
            "capacity": capacity,
            "node_process_count": alive,
            "active_node_process_count": active,
            "idle_node_process_count": idle,
            "node_processes": nodes,
            "max_requests_per_node": int(_MAX_REQUESTS_PER_NODE),
        }

    def execute(self, req: Dict[str, Any], *, export_name: str, payload_json: str, timeout_ms: int, output_limit_bytes: int) -> Dict[str, Any]:
        rt = self._checkout()
        reusable = True
        try:
            return rt.execute(
                request_id=uuid.uuid4().hex,
                module_source=str(req.get("module_source") or ""),
                export_name=export_name,
                payload_json=payload_json,
                timeout_ms=timeout_ms,
                output_limit_bytes=output_limit_bytes,
            )
        except TimeoutError:
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


_NODE_POOL = _HotNodeRuntimePool(_CALL_CAPACITY)


def _worker_resources() -> Dict[str, Any]:
    pool = _NODE_POOL.stats()
    gate = _call_slots.stats()
    return {
        "status": "ok",
        "executor_kind": "workflow_js_helper",
        "worker_id": _worker_id(),
        "engine_id": _worker_id(),
        "node_executable": _node_executable(),
        "node_version": _node_version(),
        "sandbox_profile": SANDBOX_PROFILE,
        "capacity": int(gate["capacity"]),
        "active_calls": int(gate["active_calls"]),
        "available_slots": int(gate["available_slots"]),
        "node_pool": pool,
    }


def _set_capacity(value: int) -> Dict[str, Any]:
    capacity = _call_slots.set_capacity(value)
    _NODE_POOL.set_capacity(capacity)
    return _worker_resources()


def _execute_node(req: Dict[str, Any], *, started_at: float) -> Dict[str, Any]:
    module_source = str(req.get("module_source") or "")
    expected_sha = str(req.get("module_sha256") or "").strip().lower()
    if not expected_sha or _sha256_text(module_source).lower() != expected_sha:
        return _failure("workflow_sandbox_invalid_module_identity", started_at=started_at)
    operation = str(req.get("operation") or "").strip() or "default"
    if operation not in ALLOWED_OPERATIONS:
        return _failure("workflow_sandbox_operation_not_allowed", detail={"operation": operation}, started_at=started_at)
    export_name = str(req.get("export_name") or "").strip() or ("default" if operation == "default" else operation)
    payload = req.get("payload")
    limits = dict(req.get("limits") or {})
    timeout_ms = max(1, min(int(limits.get("timeout_ms") or 5000), 300000))
    output_limit_bytes = max(1, min(int(limits.get("output_limit_bytes") or 65536), 10 * 1024 * 1024))
    memory_limit_mb = limits.get("memory_limit_mb")
    audit = _audit_from_request(req)
    audit["memory_limit"] = {
        "requested_mb": int(memory_limit_mb) if memory_limit_mb is not None else None,
        "enforcement": "best_effort_unavailable",
    }
    try:
        payload_json = json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return _failure("workflow_sandbox_invalid_result_shape", detail={"message": str(exc)}, started_at=started_at)
    try:
        response = _NODE_POOL.execute(
            dict(req or {}),
            export_name=export_name,
            payload_json=payload_json,
            timeout_ms=timeout_ms,
            output_limit_bytes=output_limit_bytes,
        )
    except TimeoutError:
        return _failure("workflow_sandbox_timeout", detail={"timeout_ms": timeout_ms}, started_at=started_at)
    except FileNotFoundError as exc:
        return _failure("workflow_sandbox_host_unavailable", detail={"message": str(exc)}, started_at=started_at)
    except Exception as exc:
        return _failure("workflow_sandbox_runtime_error", detail={"message": str(exc)}, started_at=started_at)
    if not bool(response.get("ok", False)):
        reason = str(response.get("reason") or "workflow_sandbox_runtime_error")
        detail = dict(response.get("detail") or {})
        if reason == "workflow_sandbox_export_not_found":
            detail.setdefault("export_name", export_name)
        return _failure(reason, detail=detail, started_at=started_at)
    stdout = str(response.get("result_json") or "null").encode("utf-8")
    if len(stdout) > output_limit_bytes:
        return _failure("workflow_sandbox_output_limit_exceeded", detail={"output_limit_bytes": output_limit_bytes}, started_at=started_at)
    try:
        parsed = json.loads(stdout.decode("utf-8") if stdout else "null")
    except Exception as exc:
        return _failure("workflow_sandbox_invalid_json_output", detail={"message": str(exc)}, started_at=started_at)
    return _success(parsed, started_at=started_at, audit=audit)


async def _execute_workflow_js_helper(req: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.monotonic()
    if not _call_slots.acquire(blocking=False):
        return _failure("workflow_sandbox_capacity_exceeded", started_at=started_at)
    try:
        return await asyncio.to_thread(_execute_node, dict(req or {}), started_at=started_at)
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
        "executor_kind": "workflow_js_helper",
        "sync_rpc": True,
        "async_rpc": False,
        "cancellation": False,
        "workflow_js_helper": {
            "available": bool(shutil.which(_node_executable()) or Path(_node_executable()).exists()),
            "node_executable": _node_executable(),
            "node_version": _node_version(),
            "sandbox_profile": SANDBOX_PROFILE,
            "capacity": int(_call_slots.stats()["capacity"]),
            "active_calls": int(_call_slots.stats()["active_calls"]),
            "max_requests_per_node": _MAX_REQUESTS_PER_NODE,
        },
    }


async def _handle_rpc_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method") or "").strip()
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if method in {"rpc.describe", "describe", "capabilities"}:
        return await _handle_hello(payload)
    if method in {"worker.resources", "workflow_js_helper.resources"}:
        return {"status": "ok", "result": _worker_resources()}
    if method in {"workflow_js_helper.set_capacity", "worker.set_capacity"}:
        value = int(dict(params or {}).get("capacity") or 1)
        return {"status": "ok", "result": _set_capacity(value)}
    if method != "execute_workflow_js_helper":
        return {"status": "error", "message": "unsupported_method"}
    result = await _execute_workflow_js_helper(dict(params or {}))
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
        _NODE_POOL.close_all()
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
    os.environ.setdefault("MP13_WORKFLOW_HELPER_WORKER_ID", str(os.environ.get("MP13_ENGINE_ID") or "workflow-js-helper"))
    return _serve_loop(
        family=str(args.ipc_family),
        address=str(args.ipc_address),
        authkey=auth_token.encode("utf-8", errors="ignore"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
