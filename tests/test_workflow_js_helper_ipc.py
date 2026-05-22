from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

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

    def fake_run(command, **kwargs):
        assert command[0].lower().endswith(("node", "node.exe"))
        assert command[2].endswith("helper.mjs")
        assert command[3] == "condition"
        return SimpleNamespace(returncode=0, stdout=b'{"ok":true}', stderr=b"")

    monkeypatch.setattr(worker, "_node_version", lambda: "v20.0.0")
    monkeypatch.setattr(worker.subprocess, "run", fake_run)

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


def test_workflow_js_helper_maps_missing_export(monkeypatch) -> None:
    source = "export function condition(input) { return true; }"

    def fake_run(_command, **_kwargs):
        return SimpleNamespace(returncode=21, stdout=b"", stderr=b'{"reason":"workflow_sandbox_export_not_found"}')

    monkeypatch.setattr(worker, "_node_version", lambda: "v20.0.0")
    monkeypatch.setattr(worker.subprocess, "run", fake_run)

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
