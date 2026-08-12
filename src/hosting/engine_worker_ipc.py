"""
Local worker transport over cross-platform IPC.

Worker contract supports:
- hello       : capability handshake and limits
- rpc_call    : synchronous RPC call (method + params)
- stream_open : open async RPC stream (requires request_id)
- stream_recv : receive stream events
- stream_send : control stream (cancel)
- stream_close: close stream
- http_request: compatibility shim for /health, /capabilities, /inference
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import queue
import secrets
import socket
import sys
import threading
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Dict, Optional

from ._process_utils import configure_parent_death_signal

PROTOCOL_VERSION = 1
_loaded_models_lock = threading.Lock()
_loaded_models: Dict[str, Dict[str, Any]] = {}
_config_bindings: Dict[str, Dict[str, Any]] = {}


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except Exception:
        return default
    return max(lo, min(v, hi))


def _limits() -> Dict[str, int]:
    return {
        "max_concurrent_streams": _env_int("MP13_WORKER_MAX_CONCURRENT_STREAMS", 128, lo=1, hi=4096),
        "stream_queue_max_items": _env_int("MP13_WORKER_STREAM_QUEUE_MAX_ITEMS", 2048, lo=16, hi=16384),
        "max_stream_recv_items": _env_int("MP13_WORKER_MAX_STREAM_RECV_ITEMS", 256, lo=1, hi=4096),
    }


def _contract_name() -> str:
    return str(os.environ.get("MP13_WORKER_CONTRACT") or "mp13.worker.rpc.v1").strip() or "mp13.worker.rpc.v1"


def _json_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = json.dumps(dict(payload or {}), ensure_ascii=False).encode("utf-8")
    return {
        "status": "ok",
        "status_code": int(status_code),
        "headers": {"Content-Type": "application/json", "Content-Length": str(len(raw))},
        "body_b64": base64.b64encode(raw).decode("ascii"),
    }


def _load_json_file(path: Optional[str]) -> Dict[str, Any]:
    p = Path(str(path or "").strip())
    if not p or not str(p):
        return {}
    try:
        if p.exists():
            out = json.loads(p.read_text(encoding="utf-8"))
            return dict(out or {}) if isinstance(out, dict) else {}
    except Exception:
        return {}
    return {}


async def _init_engine() -> Dict[str, Any]:
    from mp13_engine.mp13_engine_api import handle_call_tool

    engine_id = str(os.environ.get("MP13_ENGINE_ID") or "engine").strip() or "engine"
    model_path = str(os.environ.get("MP13_MODEL_PATH") or "").strip()
    config_path = str(os.environ.get("MP13_ENGINE_CONFIG_PATH") or "").strip()
    cfg = _load_json_file(config_path)
    base_model = model_path or str(cfg.get("base_model_name_or_path") or "").strip()
    if not base_model:
        return {"ok": False, "message": "Missing MP13_MODEL_PATH and config base_model_name_or_path"}
    init_args = {"instance_id": engine_id, "base_model_name_or_path": base_model}
    resp = await handle_call_tool("initialize-engine", init_args)
    if str(getattr(resp, "status", "")) != "success":
        return {"ok": False, "message": str(getattr(resp, "message", "initialize-engine failed"))}
    with _loaded_models_lock:
        _loaded_models[str(engine_id)] = {
            "model_instance_id": str(engine_id),
            "engine_id": str(engine_id),
            "model_path": str(base_model),
            "config_path": str(config_path),
            "loaded_at": time.time(),
        }
        if config_path:
            _config_bindings[str(engine_id)] = {
                "config_binding_id": str(engine_id),
                "engine_id": str(engine_id),
                "model_instance_id": str(engine_id),
                "config_path": str(config_path),
            }
    return {"ok": True}


def _targeted_arguments(engine_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(arguments or {})
    eid = str(engine_id or "").strip()
    if eid and "instance_id" not in out:
        out["instance_id"] = eid
    return out


async def _run_tool(tool: str, arguments: Dict[str, Any], *, engine_id: str = "") -> Dict[str, Any]:
    from mp13_engine.mp13_engine_api import handle_call_tool, inference_stream_to_dict_stream

    resp = await handle_call_tool(str(tool), _targeted_arguments(engine_id, dict(arguments or {})))
    if str(getattr(resp, "status", "")) != "success":
        msg = str(getattr(resp, "message", "tool failed"))
        return {"ok": False, "status_code": 400, "payload": {"status": "error", "message": msg}}
    out: Dict[str, Any] = {
        "status": "success",
        "message": str(getattr(resp, "message", "")),
        "data": getattr(resp, "data", None),
        "details": getattr(resp, "details", None),
    }
    stream = getattr(resp, "stream", None)
    if stream is not None:
        chunks = []
        async for item in inference_stream_to_dict_stream(stream):
            chunks.append(item)
        out["stream"] = chunks
    return {"ok": True, "status_code": 200, "payload": out}


async def _rpc_call(method: str, params: Dict[str, Any], *, engine_id: str = "") -> Dict[str, Any]:
    m = str(method or "").strip()
    p = dict(params or {})
    if m in {"rpc.describe", "describe", "capabilities"}:
        lim = _limits()
        return {
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "contract": _contract_name(),
            "sync_rpc": True,
            "async_rpc": True,
            "cancellation": True,
            "model_management": True,
            "limits": lim,
        }
    if m in {"worker.resources", "worker.resource-status"}:
        return _worker_resource_status()
    if m.startswith("model."):
        return await _model_rpc_call(m, p)
    out = await _run_tool(m, p, engine_id=engine_id)
    if not bool(out.get("ok")):
        return {"status": "error", "message": str((out.get("payload") or {}).get("message") or "rpc_call_failed")}
    return {"status": "ok", "result": dict(out.get("payload") or {})}


async def _model_rpc_call(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from mp13_engine.mp13_engine_api import handle_call_tool

    m = str(method or "").strip()
    p = dict(params or {})
    if m in {"model.list", "model.describe"}:
        resp = await handle_call_tool("list-engines", {})
        payload = {
            "status": "success",
            "data": getattr(resp, "data", None),
            "message": str(getattr(resp, "message", "")),
            "loaded_models": list(_loaded_models.values()),
            "config_bindings": list(_config_bindings.values()),
        }
        return {"status": "ok", "result": payload}
    if m == "model.load":
        model_instance_id = str(p.get("model_instance_id") or p.get("engine_id") or p.get("instance_id") or "").strip()
        model_path = str(p.get("model_path") or p.get("base_model_name_or_path") or "").strip()
        config_path = str(p.get("config_path") or "").strip()
        if not model_instance_id:
            return {"status": "error", "message": "model_instance_id_required"}
        if not model_path:
            return {"status": "error", "message": "model_path_required"}
        with _loaded_models_lock:
            existing = dict(_loaded_models.get(model_instance_id) or {})
        if existing:
            return {"status": "ok", "result": {"status": "already_loaded", "model": existing}}
        resp = await handle_call_tool(
            "initialize-engine",
            {"instance_id": model_instance_id, "base_model_name_or_path": model_path},
        )
        if str(getattr(resp, "status", "")) != "success":
            return {"status": "error", "message": str(getattr(resp, "message", "model_load_failed"))}
        model = {
            "model_instance_id": model_instance_id,
            "engine_id": model_instance_id,
            "model_path": model_path,
            "config_path": config_path,
        }
        with _loaded_models_lock:
            _loaded_models[model_instance_id] = model
        return {"status": "ok", "result": {"status": "loaded", "model": model, "data": getattr(resp, "data", None)}}
    if m == "model.unload":
        if p.get("shutdown_all") is True:
            resp = await handle_call_tool("shutdown-engine", {"shutdown_all": True})
            if str(getattr(resp, "status", "")) != "success":
                return {"status": "error", "message": str(getattr(resp, "message", "model_unload_failed"))}
            with _loaded_models_lock:
                _loaded_models.clear()
                _config_bindings.clear()
            return {"status": "ok", "result": {"status": "unloaded_all"}}
        model_instance_id = str(p.get("model_instance_id") or p.get("engine_id") or p.get("instance_id") or "").strip()
        if not model_instance_id:
            return {"status": "error", "message": "model_instance_id_required"}
        resp = await handle_call_tool("shutdown-engine", {"instance_id": model_instance_id})
        if str(getattr(resp, "status", "")) != "success":
            return {"status": "error", "message": str(getattr(resp, "message", "model_unload_failed"))}
        with _loaded_models_lock:
            _loaded_models.pop(model_instance_id, None)
            for key, binding in list(_config_bindings.items()):
                if str((binding or {}).get("model_instance_id") or "") == model_instance_id:
                    _config_bindings.pop(key, None)
        return {"status": "ok", "result": {"status": "unloaded", "model_instance_id": model_instance_id}}
    return {"status": "error", "message": "unsupported_model_method"}


def _worker_resource_status() -> Dict[str, Any]:
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return {
            "status": "ok",
            "result": {
                "status": "pending",
                "message": "torch_module_not_loaded",
                "data": {
                    "pid": os.getpid(),
                    "gpu_vram_pending": True,
                    "gpu_vram_source": "worker_torch_module_pending",
                },
            },
        }
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None:
        return {
            "status": "ok",
            "result": {
                "status": "pending",
                "message": "torch_cuda_unavailable",
                "data": {
                    "pid": os.getpid(),
                    "gpu_vram_pending": True,
                    "gpu_vram_source": "worker_torch_cuda_pending",
                },
            },
        }
    try:
        available = bool(cuda.is_available())
    except Exception as exc:
        return {
            "status": "ok",
            "result": {
                "status": "pending",
                "message": f"torch_cuda_status_error:{exc}",
                "data": {
                    "pid": os.getpid(),
                    "gpu_vram_pending": True,
                    "gpu_vram_source": "worker_torch_cuda_pending",
                },
            },
        }
    if not available:
        return {
            "status": "ok",
            "result": {
                "status": "ok",
                "message": "cuda_not_available",
                "data": {
                    "pid": os.getpid(),
                    "gpu_info": [],
                    "current_gpu_mem_allocated_mb": 0.0,
                    "current_gpu_mem_reserved_mb": 0.0,
                    "gpu_vram_source": "worker_torch_cuda",
                },
            },
        }
    try:
        count = int(cuda.device_count())
    except Exception:
        count = 0
    devices = []
    allocated_mb = 0.0
    reserved_mb = 0.0
    for idx in range(max(0, count)):
        try:
            alloc = float(cuda.memory_allocated(idx)) / (1024.0 * 1024.0)
        except Exception:
            alloc = 0.0
        try:
            reserved = float(cuda.memory_reserved(idx)) / (1024.0 * 1024.0)
        except Exception:
            reserved = 0.0
        allocated_mb += alloc
        reserved_mb += reserved
        devices.append(
            {
                "device_id": idx,
                "memory_allocated_mb": round(alloc, 1),
                "memory_reserved_mb": round(reserved, 1),
            }
        )
    return {
        "status": "ok",
        "result": {
            "status": "success",
            "message": "Worker resources retrieved.",
            "data": {
                "pid": os.getpid(),
                "gpu_info": devices,
                "current_gpu_mem_allocated_mb": round(allocated_mb, 1),
                "current_gpu_mem_reserved_mb": round(reserved_mb, 1),
                "gpu_vram_source": "worker_torch_cuda",
            },
        },
    }


class _StreamSession:
    def __init__(
        self,
        *,
        stream_id: str,
        engine_id: str,
        method: str,
        params: Dict[str, Any],
        request_id: str,
        queue_max_items: int,
    ) -> None:
        self.stream_id = str(stream_id)
        self.engine_id = str(engine_id)
        self.method = str(method)
        self.params = dict(params or {})
        self.request_id = str(request_id)
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(16, int(queue_max_items)))
        self.stop_event = threading.Event()
        self.done = False
        self.closed = False
        self.error: Optional[str] = None
        self.final_response: Optional[Dict[str, Any]] = None
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"ipc-stream-{self.stream_id[:8]}")

    def start(self) -> None:
        self._thread.start()

    def _emit(self, event: Dict[str, Any]) -> bool:
        row = dict(event or {})
        row.setdefault("stream_id", self.stream_id)
        row.setdefault("request_id", self.request_id)
        try:
            self.events.put_nowait(row)
            return True
        except queue.Full:
            self.error = "stream_queue_full"
            return False

    def _emit_final(self) -> None:
        final_event: Dict[str, Any] = {
            "event": "final",
            "ok": self.error is None and not self.stop_event.is_set(),
        }
        if self.final_response is not None:
            final_event["response"] = dict(self.final_response)
            final_event["final_response"] = dict(self.final_response)
        row = dict(final_event)
        row.setdefault("stream_id", self.stream_id)
        row.setdefault("request_id", self.request_id)
        while True:
            try:
                self.events.put_nowait(row)
                return
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    return

    def _record_final_response(self, item: Any) -> None:
        if not isinstance(item, dict):
            return
        response_text = item.get("response_text")
        chunk_text = item.get("chunk_text")
        is_final_chunk = bool(item.get("is_final_chunk"))
        chunk_type = item.get("chunkType")
        if not is_final_chunk and response_text is None:
            return
        final_response = dict(item)
        if response_text is not None:
            final_response["response_text"] = str(response_text)
        elif chunk_text is not None:
            final_response["response_text"] = str(chunk_text)
        if chunk_type is not None:
            final_response["chunkType"] = str(chunk_type)
        self.final_response = final_response

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        from mp13_engine.mp13_engine_api import handle_call_tool, inference_stream_to_dict_stream

        try:
            resp = await handle_call_tool(self.method, _targeted_arguments(self.engine_id, dict(self.params or {})))
            if str(getattr(resp, "status", "")) != "success":
                self.error = str(getattr(resp, "message", "rpc_failed"))
                self._emit({"event": "error", "message": self.error})
                return
            self._emit(
                {
                    "event": "accepted",
                    "message": str(getattr(resp, "message", "")),
                    "data": getattr(resp, "data", None),
                    "details": getattr(resp, "details", None),
                }
            )
            stream = getattr(resp, "stream", None)
            if stream is None:
                self._emit({"event": "result", "data": getattr(resp, "data", None), "details": getattr(resp, "details", None)})
                return
            seq = 0
            async for item in inference_stream_to_dict_stream(stream):
                if self.stop_event.is_set():
                    break
                self._record_final_response(item)
                if not self._emit({"event": "chunk", "seq": seq, "chunk": item}):
                    self.stop_event.set()
                    break
                seq += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._emit({"event": "error", "message": self.error})
        finally:
            self._emit_final()
            self.done = True


_stream_lock = threading.Lock()
_stream_sessions: Dict[str, _StreamSession] = {}


def _stream_get(stream_id: str) -> Optional[_StreamSession]:
    sid = str(stream_id or "").strip()
    if not sid:
        return None
    with _stream_lock:
        return _stream_sessions.get(sid)


def _stream_pop(stream_id: str) -> Optional[_StreamSession]:
    sid = str(stream_id or "").strip()
    if not sid:
        return None
    with _stream_lock:
        return _stream_sessions.pop(sid, None)


def _stream_create(*, engine_id: str, method: str, params: Dict[str, Any], request_id: str) -> _StreamSession:
    lim = _limits()
    with _stream_lock:
        if len(_stream_sessions) >= int(lim["max_concurrent_streams"]):
            raise RuntimeError("max_concurrent_streams_reached")
        sid = secrets.token_hex(12)
        sess = _StreamSession(
            stream_id=sid,
            engine_id=engine_id,
            method=method,
            params=params,
            request_id=request_id,
            queue_max_items=int(lim["stream_queue_max_items"]),
        )
        _stream_sessions[sid] = sess
    sess.start()
    return sess


async def _handle_hello(_payload: Dict[str, Any]) -> Dict[str, Any]:
    lim = _limits()
    return {
        "status": "ok",
        "pid": os.getpid(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "protocol_version": PROTOCOL_VERSION,
        "contract": _contract_name(),
        "sync_rpc": True,
        "async_rpc": True,
        "cancellation": True,
        "model_management": True,
        "limits": lim,
    }


async def _handle_rpc_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method") or "").strip()
    engine_id = str(payload.get("engine_id") or "").strip()
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if not method:
        return {"status": "error", "message": "method_required"}
    return await _rpc_call(method, dict(params or {}), engine_id=engine_id)


async def _handle_stream_open(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine_id = str(payload.get("engine_id") or "").strip() or str(os.environ.get("MP13_ENGINE_ID") or "engine").strip()
    method = str(payload.get("method") or payload.get("tool") or "run-inference").strip() or "run-inference"
    params = payload.get("params") if isinstance(payload.get("params"), dict) else payload.get("arguments")
    if not isinstance(params, dict):
        params = {}
    request_id = str(payload.get("request_id") or (params.get("request_id") if isinstance(params, dict) else "") or "").strip()
    if not request_id:
        return {"status": "error", "message": "request_id_required"}
    try:
        sess = _stream_create(engine_id=engine_id, method=method, params=dict(params or {}), request_id=request_id)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    return {"status": "ok", "stream_id": sess.stream_id, "engine_id": sess.engine_id, "request_id": request_id}


async def _handle_stream_recv(payload: Dict[str, Any]) -> Dict[str, Any]:
    stream_id = str(payload.get("stream_id") or "").strip()
    timeout_seconds = max(0.0, float(payload.get("timeout_seconds") or 2.0))
    max_items = max(1, min(int(payload.get("max_items") or 64), int(_limits()["max_stream_recv_items"])))
    sess = _stream_get(stream_id)
    if sess is None:
        return {"status": "error", "message": "stream_not_found", "stream_id": stream_id}
    items: list[Dict[str, Any]] = []
    first_wait = timeout_seconds
    while len(items) < max_items:
        try:
            if first_wait > 0:
                row = sess.events.get(timeout=first_wait)
                first_wait = 0.0
            else:
                row = sess.events.get_nowait()
            items.append(dict(row or {}))
        except queue.Empty:
            break
    done = bool(sess.done and sess.events.empty())
    if done:
        _stream_pop(stream_id)
    out: Dict[str, Any] = {
        "status": "ok",
        "stream_id": stream_id,
        "engine_id": sess.engine_id,
        "request_id": sess.request_id,
        "events": items,
        "done": done,
    }
    if done and sess.final_response is not None:
        out["response"] = dict(sess.final_response)
        out["final_response"] = dict(sess.final_response)
    return out


async def _handle_stream_send(payload: Dict[str, Any]) -> Dict[str, Any]:
    stream_id = str(payload.get("stream_id") or "").strip()
    raw_message = payload.get("message")
    message: Dict[str, Any] = dict(raw_message) if isinstance(raw_message, dict) else {}
    sess = _stream_get(stream_id)
    if sess is None:
        return {"status": "error", "message": "stream_not_found", "stream_id": stream_id}
    action = str(message.get("action") or "").strip().lower()
    if action == "cancel":
        req_id = str(message.get("request_id") or "").strip()
        if not req_id:
            return {"status": "error", "message": "request_id_required"}
        if req_id != sess.request_id:
            return {"status": "error", "message": "request_id_mismatch", "request_id": req_id}
        _ = await _run_tool("cancel-request", {"request_id": req_id})
        sess.stop_event.set()
        return {"status": "ok", "stream_id": stream_id, "accepted": True, "action": "cancel", "request_id": req_id}
    return {"status": "ok", "stream_id": stream_id, "accepted": False, "message": "unsupported_action"}


async def _handle_stream_close(payload: Dict[str, Any]) -> Dict[str, Any]:
    stream_id = str(payload.get("stream_id") or "").strip()
    sess = _stream_get(stream_id)
    if sess is None:
        return {"status": "ok", "stream_id": stream_id, "closed": False, "status_message": "not_found"}
    sess.closed = True
    sess.stop_event.set()
    if sess.done and sess.events.empty():
        _stream_pop(stream_id)
    return {"status": "ok", "stream_id": stream_id, "closed": True, "request_id": sess.request_id}


async def _handle_http_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method") or "GET").strip().upper()
    path = str(payload.get("path") or "/").strip() or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    body_b64 = str(payload.get("body_b64") or "")

    if path == "/health" and method == "GET":
        return _json_response(200, {"ok": True, "transport": "ipc", "contract": _contract_name(), "protocol_version": PROTOCOL_VERSION})

    if path == "/capabilities" and method == "GET":
        return _json_response(200, {
            "health": True,
            "capabilities": True,
            "inference": True,
            "rpc": True,
            "async_rpc": True,
            "cancellation": True,
            "model_management": True,
            "ws": False,
            "contract": _contract_name(),
            "protocol_version": PROTOCOL_VERSION,
            "limits": _limits(),
        })

    if path == "/inference":
        if method != "POST":
            return _json_response(405, {"status": "error", "message": "method_not_allowed"})
        try:
            raw = base64.b64decode(body_b64) if body_b64 else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}
            req = dict(data or {}) if isinstance(data, dict) else {}
        except Exception as exc:
            return _json_response(400, {"status": "error", "message": f"invalid_json:{exc}"})
        tool = str(req.get("tool") or "run-inference").strip() or "run-inference"
        arguments = req.get("arguments") if isinstance(req.get("arguments"), dict) else req
        target_engine_id = str(payload.get("engine_id") or "").strip()
        out = await _run_tool(tool, dict(arguments or {}), engine_id=target_engine_id)
        return _json_response(int(out.get("status_code") or 200), dict(out.get("payload") or {}))

    return _json_response(404, {"status": "error", "message": "not_found"})


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
        if kind == "http_request":
            conn.send(asyncio.run(_handle_http_request(req)))
            return
        if kind == "stream_open":
            conn.send(asyncio.run(_handle_stream_open(req)))
            return
        if kind == "stream_recv":
            conn.send(asyncio.run(_handle_stream_recv(req)))
            return
        if kind == "stream_send":
            conn.send(asyncio.run(_handle_stream_send(req)))
            return
        if kind == "stream_close":
            conn.send(asyncio.run(_handle_stream_close(req)))
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
            except Exception as exc:  # pragma: no cover - defensive edge path
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
    configure_parent_death_signal()

    try:
        init = asyncio.run(_init_engine())
        if not bool(init.get("ok")):
            print(str(init.get("message") or "engine init failed"), flush=True)
            return 3
    except Exception:
        print(traceback.format_exc(), flush=True)
        return 4

    return _serve_loop(
        family=str(args.ipc_family),
        address=str(args.ipc_address),
        authkey=auth_token.encode("utf-8", errors="ignore"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
