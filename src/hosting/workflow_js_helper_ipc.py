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
_call_slots = threading.BoundedSemaphore(_CALL_CAPACITY)


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
    out = {
        "worker_id": _worker_id(),
        "engine_id": _worker_id(),
        "node_executable": _node_executable(),
        "node_version": _node_version(),
        "sandbox_profile": SANDBOX_PROFILE,
        "contract": _contract_name(),
        "capacity": _CALL_CAPACITY,
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


def _runner_source() -> str:
    return """
const [modulePath, exportName, payloadB64] = process.argv.slice(2);
const payload = JSON.parse(Buffer.from(payloadB64, 'base64').toString('utf8'));
const mod = await import(modulePath);
if (!Object.prototype.hasOwnProperty.call(mod, exportName)) {
  console.error(JSON.stringify({ reason: 'workflow_sandbox_export_not_found' }));
  process.exit(21);
}
const fn = mod[exportName];
if (typeof fn !== 'function') {
  console.error(JSON.stringify({ reason: 'workflow_sandbox_export_not_found' }));
  process.exit(21);
}
const value = await fn(payload);
try {
  process.stdout.write(JSON.stringify(value === undefined ? null : value));
} catch (err) {
  console.error(JSON.stringify({ reason: 'workflow_sandbox_invalid_json_output', message: String(err && err.message || err) }));
  process.exit(22);
}
""".strip()


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
    payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("ascii")
    with tempfile.TemporaryDirectory(prefix="mp13-workflow-js-helper-") as tmp:
        tmp_path = Path(tmp)
        module_path = tmp_path / "helper.mjs"
        runner_path = tmp_path / "runner.mjs"
        module_path.write_text(module_source, encoding="utf-8")
        runner_path.write_text(_runner_source(), encoding="utf-8")
        command = [_node_executable(), str(runner_path), module_path.as_uri(), export_name, payload_b64]
        try:
            result = subprocess.run(  # noqa: S603
                command,
                cwd=str(tmp_path),
                capture_output=True,
                text=False,
                timeout=timeout_ms / 1000.0,
                check=False,
                **hidden_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired:
            return _failure("workflow_sandbox_timeout", detail={"timeout_ms": timeout_ms}, started_at=started_at)
        except FileNotFoundError as exc:
            return _failure("workflow_sandbox_host_unavailable", detail={"message": str(exc)}, started_at=started_at)
        except Exception as exc:
            return _failure("workflow_sandbox_runtime_error", detail={"message": str(exc)}, started_at=started_at)
    stdout = bytes(result.stdout or b"")
    stderr = bytes(result.stderr or b"")
    if len(stdout) > output_limit_bytes:
        return _failure("workflow_sandbox_output_limit_exceeded", detail={"output_limit_bytes": output_limit_bytes}, started_at=started_at)
    if int(result.returncode or 0) == 21:
        return _failure("workflow_sandbox_export_not_found", detail={"export_name": export_name}, started_at=started_at)
    if int(result.returncode or 0) == 22:
        return _failure("workflow_sandbox_invalid_json_output", detail={"stderr": stderr.decode("utf-8", errors="replace")}, started_at=started_at)
    if int(result.returncode or 0) != 0:
        return _failure(
            "workflow_sandbox_runtime_error",
            detail={"returncode": int(result.returncode or 0), "stderr": stderr.decode("utf-8", errors="replace")},
            started_at=started_at,
        )
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
            "capacity": _CALL_CAPACITY,
        },
    }


async def _handle_rpc_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method") or "").strip()
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if method in {"rpc.describe", "describe", "capabilities"}:
        return await _handle_hello(payload)
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
