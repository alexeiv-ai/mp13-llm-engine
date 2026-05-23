from __future__ import annotations

import asyncio
import hashlib

import hosting.workflow_js_helper_ipc as worker


def _request(source: str, **overrides):
    payload = {
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg-demo",
        "workflow_id": "config/demo",
        "package_source_digest": "sha256:digest",
        "export_name": "condition",
        "operation": "condition",
        "payload": {"value": 3},
        "provenance": {
            "session_id": "session-1",
            "context_id": "context-1",
            "cursor_id": "cursor-1",
            "workflow_root_id": "root-1",
        },
        "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
    }
    payload.update(overrides)
    return payload


class _FakeNodePool:
    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = dict(response or {"ok": True, "result_json": '{"ok":true}'})
        self.exc = exc
        self.calls = []

    def execute(self, req, **kwargs):
        self.calls.append({"req": dict(req or {}), **dict(kwargs or {})})
        if self.exc is not None:
            raise self.exc
        return dict(self.response)


def test_workflow_js_helper_rejects_invalid_module_identity() -> None:
    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_js_helper",
                "params": _request("export function condition(input) { return true; }", module_sha256="bad"),
            }
        )
    )

    assert out["status"] == "ok"
    result = out["result"]
    assert result["ok"] is False
    assert result["reason"] == "workflow_sandbox_invalid_module_identity"


def test_workflow_js_helper_rejects_disallowed_operation() -> None:
    source = "export function condition(input) { return true; }"
    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_js_helper",
                "params": _request(source, operation="general_eval"),
            }
        )
    )

    result = out["result"]
    assert result["ok"] is False
    assert result["reason"] == "workflow_sandbox_operation_not_allowed"


def test_workflow_js_helper_executes_named_export(monkeypatch) -> None:
    source = "export function condition(input) { return {ok: input.value === 3}; }"

    fake_pool = _FakeNodePool({"ok": True, "result_json": '{"ok":true}'})
    monkeypatch.setattr(worker, "_node_version", lambda: "v20.0.0")
    monkeypatch.setattr(worker, "_NODE_POOL", fake_pool)

    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_js_helper",
                "params": _request(source),
            }
        )
    )

    result = out["result"]
    assert result["ok"] is True
    assert result["result"] == {"ok": True}
    assert result["runtime"]["node_version"] == "v20.0.0"
    assert result["audit"]["package_id"] == "pkg-demo"
    assert result["audit"]["workflow_id"] == "config/demo"
    assert result["audit"]["session_id"] == "session-1"
    assert fake_pool.calls[0]["export_name"] == "condition"


def test_workflow_js_helper_maps_missing_export(monkeypatch) -> None:
    source = "export function condition(input) { return true; }"

    monkeypatch.setattr(worker, "_node_version", lambda: "v20.0.0")
    monkeypatch.setattr(
        worker,
        "_NODE_POOL",
        _FakeNodePool({"ok": False, "reason": "workflow_sandbox_export_not_found", "detail": {}}),
    )

    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_js_helper",
                "params": _request(source, export_name="missing"),
            }
        )
    )

    result = out["result"]
    assert result["ok"] is False
    assert result["reason"] == "workflow_sandbox_export_not_found"


def test_workflow_js_helper_maps_timeout(monkeypatch) -> None:
    source = "export function condition(input) { return true; }"

    monkeypatch.setattr(worker, "_node_version", lambda: "v20.0.0")
    monkeypatch.setattr(worker, "_NODE_POOL", _FakeNodePool(exc=TimeoutError("timeout")))

    out = asyncio.run(worker._handle_rpc_call({"method": "execute_workflow_js_helper", "params": _request(source)}))

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_timeout"


def test_workflow_js_helper_maps_output_limit(monkeypatch) -> None:
    source = "export function condition(input) { return 'too much'; }"

    monkeypatch.setattr(worker, "_node_version", lambda: "v20.0.0")
    monkeypatch.setattr(worker, "_NODE_POOL", _FakeNodePool({"ok": True, "result_json": '{"data":"abcdef"}'}))

    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_js_helper",
                "params": _request(source, limits={"timeout_ms": 5000, "output_limit_bytes": 4}),
            }
        )
    )

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_output_limit_exceeded"


def test_workflow_js_helper_maps_invalid_json_output(monkeypatch) -> None:
    source = "export function condition(input) { return true; }"

    monkeypatch.setattr(worker, "_node_version", lambda: "v20.0.0")
    monkeypatch.setattr(worker, "_NODE_POOL", _FakeNodePool({"ok": True, "result_json": "not-json"}))

    out = asyncio.run(worker._handle_rpc_call({"method": "execute_workflow_js_helper", "params": _request(source)}))

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_invalid_json_output"


def test_workflow_js_helper_maps_runtime_error(monkeypatch) -> None:
    source = "export function condition(input) { throw new Error('boom'); }"

    monkeypatch.setattr(worker, "_node_version", lambda: "v20.0.0")
    monkeypatch.setattr(
        worker,
        "_NODE_POOL",
        _FakeNodePool({"ok": False, "reason": "workflow_sandbox_runtime_error", "detail": {"message": "boom"}}),
    )

    out = asyncio.run(worker._handle_rpc_call({"method": "execute_workflow_js_helper", "params": _request(source)}))

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_runtime_error"


def test_workflow_js_helper_maps_host_unavailable(monkeypatch) -> None:
    source = "export function condition(input) { return true; }"

    monkeypatch.setattr(worker, "_node_version", lambda: None)
    monkeypatch.setattr(worker, "_NODE_POOL", _FakeNodePool(exc=FileNotFoundError("node missing")))

    out = asyncio.run(worker._handle_rpc_call({"method": "execute_workflow_js_helper", "params": _request(source)}))

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_host_unavailable"


def test_workflow_js_helper_maps_invalid_result_shape_for_non_json_payload() -> None:
    source = "export function condition(input) { return true; }"
    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_js_helper",
                "params": _request(source, payload=object()),
            }
        )
    )

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_invalid_result_shape"


def test_workflow_js_helper_reports_capacity_exceeded() -> None:
    acquired = worker._call_slots.acquire(blocking=False)
    assert acquired is True
    try:
        source = "export function condition(input) { return true; }"
        out = asyncio.run(
            worker._handle_rpc_call(
                {
                    "method": "execute_workflow_js_helper",
                    "params": _request(source),
                }
            )
        )
    finally:
        worker._call_slots.release()

    result = out["result"]
    assert result["ok"] is False
    assert result["reason"] == "workflow_sandbox_capacity_exceeded"


def test_hot_node_runtime_pool_reuses_then_recycles(monkeypatch) -> None:
    created = []

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False
            created.append(self)

        def alive(self) -> bool:
            return not self.closed

        def reusable(self) -> bool:
            return self.alive() and self.calls < 2

        def execute(self, **_kwargs):
            self.calls += 1
            return {"ok": True, "result_json": "{}"}

        def close(self, *, kill: bool = False) -> None:
            self.closed = True

    monkeypatch.setattr(worker, "_HotNodeRuntime", FakeRuntime)
    pool = worker._HotNodeRuntimePool(capacity=1)
    req = _request("export function condition(input) { return {}; }")

    assert pool.execute(req, export_name="condition", payload_json="{}", timeout_ms=1000, output_limit_bytes=1000)["ok"] is True
    assert pool.execute(req, export_name="condition", payload_json="{}", timeout_ms=1000, output_limit_bytes=1000)["ok"] is True
    assert len(created) == 1
    assert created[0].closed is True

    assert pool.execute(req, export_name="condition", payload_json="{}", timeout_ms=1000, output_limit_bytes=1000)["ok"] is True
    assert len(created) == 2
