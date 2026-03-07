"""
Local engine worker transport over cross-platform IPC.

This worker is host-managed and intended for local-only hosting mode.
It serves a small HTTP-like request contract over multiprocessing IPC:
  - /health
  - /capabilities
  - /inference (POST JSON)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import queue
import secrets
import threading
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Dict, Optional


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
    return {"ok": True}


async def _run_tool(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    from mp13_engine.mp13_engine_api import handle_call_tool, inference_stream_to_dict_stream

    resp = await handle_call_tool(str(tool), dict(arguments or {}))
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


class _StreamSession:
    def __init__(self, *, stream_id: str, engine_id: str, tool: str, arguments: Dict[str, Any]) -> None:
        self.stream_id = str(stream_id)
        self.engine_id = str(engine_id)
        self.tool = str(tool)
        self.arguments = dict(arguments or {})
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2048)
        self.stop_event = threading.Event()
        self.done = False
        self.closed = False
        self.error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"ipc-stream-{self.stream_id[:8]}")

    def start(self) -> None:
        self._thread.start()

    def _emit(self, event: Dict[str, Any]) -> None:
        row = dict(event or {})
        row.setdefault("stream_id", self.stream_id)
        try:
            self.events.put_nowait(row)
        except queue.Full:
            self.error = "stream_queue_full"
            self.done = True

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        from mp13_engine.mp13_engine_api import handle_call_tool, inference_stream_to_dict_stream

        try:
            resp = await handle_call_tool(self.tool, dict(self.arguments or {}))
            if str(getattr(resp, "status", "")) != "success":
                self._emit({"event": "error", "message": str(getattr(resp, "message", "tool failed"))})
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
                self._emit({"event": "chunk", "seq": seq, "chunk": item})
                seq += 1
        except Exception as exc:
            self._emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.done = True
            self._emit({"event": "final", "ok": self.error is None and not self.stop_event.is_set()})


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


def _stream_create(*, engine_id: str, tool: str, arguments: Dict[str, Any]) -> _StreamSession:
    sid = secrets.token_hex(12)
    sess = _StreamSession(stream_id=sid, engine_id=engine_id, tool=tool, arguments=arguments)
    with _stream_lock:
        _stream_sessions[sid] = sess
    sess.start()
    return sess


async def _handle_stream_open(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine_id = str(payload.get("engine_id") or "").strip() or str(os.environ.get("MP13_ENGINE_ID") or "engine").strip()
    tool = str(payload.get("tool") or "run-inference").strip() or "run-inference"
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    sess = _stream_create(engine_id=engine_id, tool=tool, arguments=dict(arguments or {}))
    return {"status": "ok", "stream_id": sess.stream_id, "engine_id": sess.engine_id}


async def _handle_stream_recv(payload: Dict[str, Any]) -> Dict[str, Any]:
    stream_id = str(payload.get("stream_id") or "").strip()
    timeout_seconds = max(0.0, float(payload.get("timeout_seconds") or 2.0))
    max_items = max(1, min(int(payload.get("max_items") or 64), 1024))
    sess = _stream_get(stream_id)
    if sess is None:
        return {"status": "error", "message": "stream_not_found", "stream_id": stream_id}
    items = []
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
    return {"status": "ok", "stream_id": stream_id, "engine_id": sess.engine_id, "events": items, "done": done}


async def _handle_stream_send(payload: Dict[str, Any]) -> Dict[str, Any]:
    stream_id = str(payload.get("stream_id") or "").strip()
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    sess = _stream_get(stream_id)
    if sess is None:
        return {"status": "error", "message": "stream_not_found", "stream_id": stream_id}
    action = str(message.get("action") or "").strip().lower()
    if action == "cancel":
        req_id = str(message.get("request_id") or sess.arguments.get("request_id") or "").strip()
        if req_id:
            _ = await _run_tool("cancel-request", {"request_id": req_id})
        sess.stop_event.set()
        return {"status": "ok", "stream_id": stream_id, "accepted": True, "action": "cancel"}
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
    return {"status": "ok", "stream_id": stream_id, "closed": True}


async def _handle_http_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method") or "GET").strip().upper()
    path = str(payload.get("path") or "/").strip() or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    body_b64 = str(payload.get("body_b64") or "")

    if path == "/health" and method == "GET":
        return _json_response(200, {"ok": True, "transport": "ipc"})

    if path == "/capabilities" and method == "GET":
        return _json_response(200, {"health": True, "capabilities": True, "inference": True, "ws": False})

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
        out = await _run_tool(tool, dict(arguments or {}))
        return _json_response(int(out.get("status_code") or 200), dict(out.get("payload") or {}))

    return _json_response(404, {"status": "error", "message": "not_found"})


def _serve_loop(*, family: str, address: str, authkey: bytes) -> int:
    listener = None
    unix_path = Path(address) if family == "AF_UNIX" else None
    if unix_path is not None:
        try:
            if unix_path.exists():
                unix_path.unlink()
        except Exception:
            pass
    try:
        listener = Listener(address=address, family=family, authkey=authkey)
        while True:
            conn = listener.accept()
            try:
                req = conn.recv()
                if not isinstance(req, dict):
                    conn.send({"status": "error", "message": "invalid_request"})
                    continue
                kind = str(req.get("kind") or "").strip().lower()
                if kind == "shutdown":
                    conn.send({"status": "ok"})
                    break
                if kind == "http_request":
                    out = asyncio.run(_handle_http_request(req))
                    conn.send(out)
                    continue
                if kind == "stream_open":
                    conn.send(asyncio.run(_handle_stream_open(req)))
                    continue
                if kind == "stream_recv":
                    conn.send(asyncio.run(_handle_stream_recv(req)))
                    continue
                if kind == "stream_send":
                    conn.send(asyncio.run(_handle_stream_send(req)))
                    continue
                if kind == "stream_close":
                    conn.send(asyncio.run(_handle_stream_close(req)))
                    continue
                conn.send({"status": "error", "message": "unsupported_kind"})
            except Exception as exc:
                conn.send({"status": "error", "message": f"worker_exception:{exc}"})
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
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
