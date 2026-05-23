from __future__ import annotations

import asyncio
import hashlib

import hosting.workflow_python_helper_ipc as worker


def _request(source: str, **overrides):
    payload = {
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg-demo",
        "workflow_id": "config/demo",
        "package_source_digest": "sha256:digest",
        "source_path": "helpers/condition.py",
        "export_name": "condition",
        "operation": "condition",
        "payload": {"value": 3},
        "python": {
            "import_allowlist": [],
            "package_pins": {},
            "environment_name": "workflow-python-helper",
        },
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


class _FakePythonPool:
    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = dict(response or {"ok": True, "result_json": '{"ok":true}'})
        self.exc = exc
        self.calls = []
        self.cancel_calls = []

    def execute(self, req, **kwargs):
        self.calls.append({"req": dict(req or {}), **dict(kwargs or {})})
        if self.exc is not None:
            raise self.exc
        return dict(self.response)

    def cancel_request(self, request_id):
        self.cancel_calls.append(str(request_id or ""))
        return {"status": "ok", "request_id": str(request_id or ""), "canceled": True, "reason": "canceled"}


def test_workflow_python_helper_rejects_invalid_module_identity() -> None:
    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request("def condition(input):\n    return True\n", module_sha256="bad"),
            }
        )
    )

    assert out["status"] == "ok"
    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_invalid_module_identity"


def test_workflow_python_helper_rejects_disallowed_operation() -> None:
    source = "def condition(input):\n    return True\n"
    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request(source, operation="general_eval"),
            }
        )
    )

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_operation_not_allowed"


def test_workflow_python_helper_executes_named_export(monkeypatch) -> None:
    source = "def condition(input):\n    return {'ok': input['value'] == 3}\n"
    fake_pool = _FakePythonPool({"ok": True, "result_json": '{"ok":true}'})
    monkeypatch.setattr(worker, "_PYTHON_POOL", fake_pool)

    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request(source, request_id="req-123"),
            }
        )
    )

    result = out["result"]
    assert result["ok"] is True
    assert result["result"] == {"ok": True}
    assert result["audit"]["package_id"] == "pkg-demo"
    assert result["audit"]["source_path"] == "helpers/condition.py"
    assert result["audit"]["request_id"] == "req-123"
    assert result["audit"]["memory_limit"]["requested_mb"] == 128
    assert result["audit"]["memory_limit"]["enforcement"] == "best_effort_unavailable"
    assert fake_pool.calls[0]["req"]["request_id"] == "req-123"
    assert fake_pool.calls[0]["export_name"] == "condition"


def test_workflow_python_helper_maps_missing_export(monkeypatch) -> None:
    source = "def condition(input):\n    return True\n"
    monkeypatch.setattr(
        worker,
        "_PYTHON_POOL",
        _FakePythonPool({"ok": False, "reason": "workflow_sandbox_export_not_found", "detail": {}}),
    )

    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request(source, export_name="missing"),
            }
        )
    )

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_export_not_found"


def test_workflow_python_helper_maps_timeout(monkeypatch) -> None:
    source = "def condition(input):\n    return True\n"
    monkeypatch.setattr(worker, "_PYTHON_POOL", _FakePythonPool(exc=TimeoutError("timeout")))

    out = asyncio.run(worker._handle_rpc_call({"method": "execute_workflow_python_helper", "params": _request(source)}))

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_timeout"


def test_workflow_python_helper_maps_canceled_request(monkeypatch) -> None:
    source = "def condition(input):\n    return True\n"
    monkeypatch.setattr(worker, "_PYTHON_POOL", _FakePythonPool(exc=worker._WorkflowPythonHelperRequestCanceled("canceled")))

    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request(source, request_id="req-cancel"),
            }
        )
    )

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_canceled"
    assert out["result"]["detail"]["request_id"] == "req-cancel"


def test_workflow_python_helper_maps_output_limit(monkeypatch) -> None:
    source = "def condition(input):\n    return 'too much'\n"
    monkeypatch.setattr(worker, "_PYTHON_POOL", _FakePythonPool({"ok": True, "result_json": '{"data":"abcdef"}'}))

    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request(source, limits={"timeout_ms": 5000, "output_limit_bytes": 4}),
            }
        )
    )

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_output_limit_exceeded"


def test_workflow_python_helper_maps_invalid_json_output(monkeypatch) -> None:
    source = "def condition(input):\n    return True\n"
    monkeypatch.setattr(worker, "_PYTHON_POOL", _FakePythonPool({"ok": True, "result_json": "not-json"}))

    out = asyncio.run(worker._handle_rpc_call({"method": "execute_workflow_python_helper", "params": _request(source)}))

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_invalid_json_output"


def test_workflow_python_helper_reports_capacity_exceeded() -> None:
    acquired = worker._call_slots.acquire(blocking=False)
    assert acquired is True
    try:
        source = "def condition(input):\n    return True\n"
        out = asyncio.run(
            worker._handle_rpc_call(
                {
                    "method": "execute_workflow_python_helper",
                    "params": _request(source),
                }
            )
        )
    finally:
        worker._call_slots.release()

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_capacity_exceeded"


def test_hot_python_runtime_pool_reuses_then_recycles(monkeypatch) -> None:
    created = []

    class FakeRuntime:
        def __init__(self, *_args) -> None:
            self.calls = 0
            self.closed = False
            self._python_executable = worker._python_executable()
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

    monkeypatch.setattr(worker, "_HotPythonRuntime", FakeRuntime)
    pool = worker._HotPythonRuntimePool(capacity=1)
    req = _request("def condition(input):\n    return {}\n")

    assert pool.execute(req, export_name="condition", payload_json="{}", import_allowlist=[], timeout_ms=1000, output_limit_bytes=1000)["ok"] is True
    assert pool.execute(req, export_name="condition", payload_json="{}", import_allowlist=[], timeout_ms=1000, output_limit_bytes=1000)["ok"] is True
    assert len(created) == 1
    assert created[0].closed is True

    assert pool.execute(req, export_name="condition", payload_json="{}", import_allowlist=[], timeout_ms=1000, output_limit_bytes=1000)["ok"] is True
    assert len(created) == 2


def test_workflow_python_helper_resources_and_cancel(monkeypatch) -> None:
    fake_pool = _FakePythonPool()
    monkeypatch.setattr(worker, "_PYTHON_POOL", fake_pool)

    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "workflow_python_helper.cancel_request",
                "params": {"request_id": "req-456"},
            }
        )
    )

    assert out["status"] == "ok"
    assert out["result"]["canceled"] is True
    assert fake_pool.cancel_calls == ["req-456"]


def test_workflow_python_helper_real_child_round_trip() -> None:
    source = "def condition(input):\n    return {'accepted': input['value'] == 7}\n"
    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request(source, payload={"value": 7}, request_id="req-real"),
            }
        )
    )

    assert out["status"] == "ok"
    assert out["result"]["ok"] is True
    assert out["result"]["result"] == {"accepted": True}


def test_workflow_python_helper_child_denies_filesystem_builtin() -> None:
    source = "def condition(input):\n    return open('x.txt', 'w')\n"
    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request(source, request_id="req-fs"),
            }
        )
    )

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_runtime_error"


def test_workflow_python_helper_child_denies_unallowlisted_import() -> None:
    source = "import subprocess\n\ndef condition(input):\n    return {'ok': True}\n"
    out = asyncio.run(
        worker._handle_rpc_call(
            {
                "method": "execute_workflow_python_helper",
                "params": _request(source, request_id="req-import"),
            }
        )
    )

    assert out["result"]["ok"] is False
    assert out["result"]["reason"] == "workflow_sandbox_runtime_error"
