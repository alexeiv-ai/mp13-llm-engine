from __future__ import annotations

import asyncio
import base64
import hashlib
import threading
import time
from typing import Any, Dict

from hosting.sandbox.host_capabilities import HostCapabilityTimeout
from hosting.sandbox.workflow_js_node_runtime import WorkflowJsNodeRuntime, WorkflowJsNodeRuntimeRegistry


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


def test_workflow_js_node_exposes_sandbox_describe() -> None:
    source = """
exports.run = function(input, api) {
  const described = sandbox.describe();
  return {output: {contract: described.contract, methods: described.methods}};
};
"""
    calls = []

    def dispatcher(call: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(dict(call))
        if call["method"] == "sandbox.describe":
            return {"contract": "hosting.sandbox.discovery.v1", "methods": ["host.describe", "sandbox.describe"]}
        raise AssertionError(call["method"])

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is True
    assert out["output"] == {"contract": "hosting.sandbox.discovery.v1", "methods": ["host.describe", "sandbox.describe"]}
    assert [call["method"] for call in calls] == ["sandbox.describe"]


def test_workflow_js_node_resolves_promise_return() -> None:
    source = "exports.run = function() { return Promise.resolve({output: true}); };"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source))

    assert out["ok"] is True
    assert out["output"] is True


def test_workflow_js_node_maps_promise_rejection() -> None:
    source = "exports.run = function() { return Promise.reject(new Error('async boom')); };"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source))

    assert out["ok"] is False
    assert out["reason"] == "workflow_sandbox_runtime_error"
    assert out["detail"]["message"] == "async boom"


def test_workflow_js_node_resolves_snippet_promise_result() -> None:
    source = "result = Promise.resolve({output: {snippet: payload.value}});"
    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source, execution_mode="snippet"))

    assert out["ok"] is True
    assert out["output"] == {"snippet": 3}


def test_workflow_js_node_supports_async_host_calls() -> None:
    source = """
exports.run = async function(input, api) {
  const text = await api.fs.readTextAsync("seed", "");
  await api.fs.writeTextAsync("report", "", text.toUpperCase());
  return {output: {seed: text}};
};
"""
    calls = []

    def dispatcher(call: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(dict(call))
        if call["method"] == "fs.read_text":
            return {"text": "demo"}
        if call["method"] == "fs.write_text":
            assert call["arguments"]["text"] == "DEMO"
            return {"ok": True}
        raise AssertionError(call["method"])

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is True
    assert out["output"] == {"seed": "demo"}
    assert [call["method"] for call in calls] == ["fs.read_text", "fs.write_text"]


def test_workflow_js_node_correlates_out_of_order_async_host_responses() -> None:
    source = """
exports.run = async function(input, api) {
  const slow = api.callAsync("demo.slow", {});
  const fast = api.callAsync("demo.fast", {});
  const values = await Promise.all([slow, fast]);
  return {output: values.map((value) => value.name)};
};
"""

    def dispatcher(call: Dict[str, Any]) -> Dict[str, Any]:
        if call["method"] == "demo.slow":
            time.sleep(0.1)
            return {"name": "slow"}
        if call["method"] == "demo.fast":
            return {"name": "fast"}
        raise AssertionError(call["method"])

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is True
    assert out["output"] == ["slow", "fast"]


def test_workflow_js_node_buffers_async_response_seen_by_sync_call() -> None:
    source = """
exports.run = async function(input, api) {
  const fast = api.callAsync("demo.fast", {});
  const slow = api.call("demo.slow", {});
  const fastValue = await fast;
  return {output: {fast: fastValue.name, slow: slow.name}};
};
"""

    def dispatcher(call: Dict[str, Any]) -> Dict[str, Any]:
        if call["method"] == "demo.fast":
            return {"name": "fast"}
        if call["method"] == "demo.slow":
            time.sleep(0.1)
            return {"name": "slow"}
        raise AssertionError(call["method"])

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is True
    assert out["output"] == {"fast": "fast", "slow": "slow"}


def test_workflow_js_node_supports_awaitable_host_dispatcher() -> None:
    source = """
exports.run = async function(input, api) {
  const value = await api.callAsync("demo.async", {value: input.value});
  return {output: value};
};
"""

    async def dispatcher(call: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"accepted": call["arguments"]["value"] == 3}

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is True
    assert out["output"] == {"accepted": True}


def test_workflow_js_node_supports_fetch_json_async_wrapper() -> None:
    source = """
exports.run = async function(input, api) {
  const value = await api.http.fetchJsonAsync("https://example.test/data.json", {method: "GET"});
  return {output: value};
};
"""

    def dispatcher(call: Dict[str, Any]) -> Dict[str, Any]:
        assert call["method"] == "http.fetch"
        assert call["arguments"]["url"] == "https://example.test/data.json"
        body = base64.b64encode(b'{"accepted":true,"count":2}').decode("ascii")
        return {"status_code": 200, "headers": {"content-type": "application/json"}, "body_b64": body}

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is True
    assert out["output"] == {"accepted": True, "count": 2}


def test_workflow_js_node_maps_async_host_api_failure_detail() -> None:
    source = "exports.run = async function(input, api) { return await api.callAsync('missing.method', {}); };"

    def dispatcher(_call: Dict[str, Any]) -> Dict[str, Any]:
        raise PermissionError("policy denied")

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is False
    assert out["reason"] == "host_call_failed"
    assert out["detail"]["message"] == "policy denied"
    assert out["detail"]["error_type"] == "PermissionError"


def test_workflow_js_node_rejects_unknown_async_host_response(monkeypatch) -> None:
    source = """
exports.run = async function(input, api) {
  const value = await api.callAsync("demo.value", {});
  return {output: value};
};
"""
    original = WorkflowJsNodeRuntime._dispatch_host_call

    def dispatch_with_unknown_response(self: WorkflowJsNodeRuntime, payload: Dict[str, Any], host_dispatcher: Any) -> None:
        self.respond_host_call(host_call_id="unknown-host-call", result={"value": "wrong"})
        original(self, payload, host_dispatcher)

    def dispatcher(_call: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "right"}

    monkeypatch.setattr(WorkflowJsNodeRuntime, "_dispatch_host_call", dispatch_with_unknown_response)

    out = WorkflowJsNodeRuntimeRegistry().execute(_request(source), host_dispatcher=dispatcher)

    assert out["ok"] is False
    assert out["reason"] == "host_response_unknown_host_call_id"
    assert out["detail"]["host_call_id"] == "unknown-host-call"


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


def test_workflow_js_node_preserves_structured_host_api_error_reason() -> None:
    runtime = object.__new__(WorkflowJsNodeRuntime)
    captured: Dict[str, Any] = {}

    def respond_host_call(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    runtime.respond_host_call = respond_host_call  # type: ignore[method-assign]

    def dispatcher(_call: Dict[str, Any]) -> Dict[str, Any]:
        raise HostCapabilityTimeout(detail={"provider_call_id": "call-1"})

    runtime._dispatch_host_call({"host_call_id": "host-call-1"}, dispatcher)

    assert captured["host_call_id"] == "host-call-1"
    assert captured["error"]["reason"] == "host_call_timeout"
    assert captured["error"]["provider_call_id"] == "call-1"
