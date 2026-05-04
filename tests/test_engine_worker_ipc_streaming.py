from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import hosting.engine_worker_ipc as worker_ipc


def _make_session() -> worker_ipc._StreamSession:
    return worker_ipc._StreamSession(
        stream_id="stream-1",
        engine_id="engine-1",
        method="run-inference",
        params={},
        request_id="req-1",
        queue_max_items=1,
    )


def test_stream_final_event_survives_full_queue_and_carries_response() -> None:
    sess = _make_session()
    final_chunk = {
        "chunkType": "streaming_chunk",
        "prompt_index": 0,
        "chunk_text": "",
        "response_text": "granite response",
        "is_final_chunk": True,
    }

    for seq in range(sess.events.maxsize):
        assert sess._emit({"event": "chunk", "seq": seq, "chunk": {"chunk_text": "stale"}}) is True
    sess._record_final_response(final_chunk)
    sess._emit_final()
    sess.done = True

    with worker_ipc._stream_lock:
        worker_ipc._stream_sessions[sess.stream_id] = sess
    try:
        out = asyncio.run(
            worker_ipc._handle_stream_recv(
                {
                    "stream_id": sess.stream_id,
                    "timeout_seconds": 0,
                    "max_items": sess.events.maxsize,
                }
            )
        )
    finally:
        worker_ipc._stream_pop(sess.stream_id)

    assert out["status"] == "ok"
    assert out["done"] is True
    assert out["response"]["response_text"] == "granite response"
    assert out["final_response"]["response_text"] == "granite response"
    assert len(out["events"]) == sess.events.maxsize
    assert out["events"][-1]["event"] == "final"
    assert out["events"][-1]["response"]["response_text"] == "granite response"


def test_stream_open_recv_contract_returns_terminal_response(monkeypatch) -> None:
    async def fake_handle_call_tool(method: str, params: dict) -> SimpleNamespace:
        async def stream():
            yield {"chunkType": "streaming_chunk", "prompt_index": 0, "chunk_text": "hello", "is_final_chunk": False}
            yield {
                "chunkType": "streaming_chunk",
                "prompt_index": 0,
                "chunk_text": "",
                "response_text": "hello granite",
                "is_final_chunk": True,
            }

        assert method == "run-inference"
        assert params["stream"] is True
        return SimpleNamespace(status="success", message="Inference stream started.", data=None, details=None, stream=stream())

    import mp13_engine.mp13_engine_api as engine_api

    monkeypatch.setattr(engine_api, "handle_call_tool", fake_handle_call_tool)

    opened = asyncio.run(
        worker_ipc._handle_stream_open(
            {
                "engine_id": "engine-1",
                "method": "run-inference",
                "params": {"stream": True},
                "request_id": "req-contract",
            }
        )
    )
    assert opened["status"] == "ok"

    seen_events: list[dict] = []
    terminal = None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        out = asyncio.run(
            worker_ipc._handle_stream_recv(
                {
                    "stream_id": opened["stream_id"],
                    "timeout_seconds": 0.1,
                    "max_items": 16,
                }
            )
        )
        assert out["status"] == "ok"
        seen_events.extend(out["events"])
        if out["done"]:
            terminal = out
            break

    assert terminal is not None
    assert terminal["events"]
    assert terminal["response"]["response_text"] == "hello granite"
    assert terminal["final_response"]["response_text"] == "hello granite"
    assert any(event.get("event") == "accepted" for event in seen_events)
    assert any(event.get("event") == "chunk" and event.get("chunk", {}).get("chunk_text") == "hello" for event in seen_events)
    assert seen_events[-1]["event"] == "final"
    assert seen_events[-1]["response"]["response_text"] == "hello granite"

    closed = asyncio.run(worker_ipc._handle_stream_close({"stream_id": opened["stream_id"]}))
    assert closed["status"] == "ok"
