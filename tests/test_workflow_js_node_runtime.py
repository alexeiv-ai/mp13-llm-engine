from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Dict

import pytest

from hosting.sandbox.workflow_js_node_runtime import WorkflowJsNodeRuntimeRegistry


def _request(source: str, **overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "request_id": "req-js-node",
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg-demo",
        "workflow_id": "config/demo",
        "package_source_digest": "sha256:digest",
        "payload": {"value": 3},
        "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
    }
    payload.update(overrides)
    return payload


def test_workflow_js_node_rejects_invalid_module_identity() -> None:
    source = "exports.run = function(input) { return input; };"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source, module_sha256="bad"))

    assert out["ok"] is False
    assert out["reason"] == "workflow_sandbox_invalid_module_identity"


def test_workflow_js_node_executes_exports_run() -> None:
    source = """
exports.run = function(input, api) {
  console.log("value", input.value);
  api.progress({step: "checking"});
  return {output: {accepted: input.value === 3}, state_patch: {seen: true}};
};
"""
    events = []

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), on_event=lambda event_type, payload: events.append((event_type, dict(payload or {}))))

    assert out["ok"] is True
    assert out["output"] == {"accepted": True}
    assert out["state_patch"] == {"seen": True}
    assert out["progress"] == {"step": "checking"}
    assert out["runtime"]["quickjs_available"] is True
    assert out["runtime"]["memory_limit"]["requested_mb"] == 128
    assert out["runtime"]["memory_limit"]["enforced"] is True
    assert ("progress", {"step": "checking"}) in events
    assert any(event_type == "console" and payload["message"] == "value 3" for event_type, payload in events)


def test_workflow_js_node_maps_missing_export() -> None:
    source = "exports.other = function(input) { return input; };"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source))

    assert out["ok"] is False
    assert out["reason"] == "workflow_sandbox_export_not_found"
    assert out["detail"]["export_name"] == "run"


def test_workflow_js_node_maps_output_limit() -> None:
    source = "exports.run = function() { return {output: 'abcdef'}; };"
    out = WorkflowJsNodeRuntimeRegistry().execute(
        _request(source, limits={"timeout_ms": 5000, "output_limit_bytes": 4}),
    )

    assert out["ok"] is False
    assert out["reason"] == "workflow_sandbox_output_limit_exceeded"


def test_workflow_js_node_uses_host_dispatcher_for_api_calls() -> None:
    source = """
exports.run = function(input, api) {
  const desc = api.describe();
  const text = api.fs.readText("seed", "");
  api.fs.writeText("report", "", text.toUpperCase());
  return {output: {methods: desc.methods, seed: text}};
};
"""
    calls = []

    def dispatcher(call: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(dict(call))
        method = call["method"]
        if method == "host.describe":
            return {"methods": ["host.describe", "fs.read_text", "fs.write_text"]}
        if method == "fs.read_text":
            assert call["arguments"]["root_id"] == "seed"
            return {"text": "demo"}
        if method == "fs.write_text":
            assert call["arguments"]["root_id"] == "report"
            assert call["arguments"]["text"] == "DEMO"
            return {"ok": True}
        raise AssertionError(method)

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is True
    assert out["output"] == {"methods": ["host.describe", "fs.read_text", "fs.write_text"], "seed": "demo"}
    assert [call["method"] for call in calls] == ["host.describe", "fs.read_text", "fs.write_text"]


def test_workflow_js_node_rejects_promise_return_for_sync_v1() -> None:
    source = "exports.run = function() { return Promise.resolve({output: true}); };"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source))

    assert out["ok"] is False
    assert out["reason"] == "workflow_sandbox_async_unsupported"


def test_workflow_js_node_returns_structured_runtime_error() -> None:
    source = "exports.run = function() { throw new Error('boom'); };"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source))

    assert out["ok"] is False
    assert out["reason"] == "workflow_sandbox_runtime_error"
    assert out["detail"]["message"]
    assert "boom" in out["detail"]["traceback_summary"]
    assert out["runtime"]["quickjs_available"] is True


def test_workflow_js_node_times_out_busy_loop() -> None:
    source = "exports.run = function() { while (true) {} };"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source, limits={"timeout_ms": 100, "output_limit_bytes": 65536}))

    assert out["ok"] is False
    assert out["reason"] == "workflow_sandbox_timeout"
    assert out["detail"]["timeout_ms"] == 100


def test_workflow_js_node_can_cancel_active_request() -> None:
    source = "exports.run = function() { while (true) {} };"
    registry = WorkflowJsNodeRuntimeRegistry()
    result: Dict[str, Any] = {}

    def run() -> None:
        result.update(registry.execute(_request(source, request_id="req-js-cancel", limits={"timeout_ms": 5000})))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if int(registry.resources().get("active_count") or 0) > 0:
            break
        time.sleep(0.05)

    canceled = registry.cancel("req-js-cancel")
    thread.join(timeout=5.0)

    assert canceled["canceled"] is True
    assert result["ok"] is False
    assert result["reason"] == "workflow_sandbox_canceled"


def test_workflow_js_node_maps_invalid_output() -> None:
    source = "exports.run = function() { return {output: 1n}; };"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source))

    assert out["ok"] is False
    assert out["reason"] == "workflow_sandbox_invalid_output"
    assert out["detail"]["message"]


def test_workflow_js_node_preserves_host_api_failure_detail() -> None:
    source = "exports.run = function(input, api) { return api.call('missing.method', {}); };"

    def dispatcher(_call: Dict[str, Any]) -> Dict[str, Any]:
        raise PermissionError("policy denied")

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is False
    assert out["reason"] == "host_call_failed"
    assert out["detail"]["message"] == "policy denied"
    assert out["detail"]["error_type"] == "PermissionError"


def test_legacy_js_helper_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("hosting.workflow_js_helper_ipc")
