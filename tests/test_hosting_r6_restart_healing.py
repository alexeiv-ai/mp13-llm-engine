from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hosting._process_utils import configure_parent_death_signal
from hosting.operation_contract import HostedExecutionKind, HostedOperationSelector, hosted_execution_fingerprint
from hosting.service.host_service import EngineHostService
from tests.hosting_v3_fixtures import hosting_configuration


def _service(tmp_path: Path) -> EngineHostService:
    return EngineHostService(
        hosting_configuration=hosting_configuration(tmp_path),
    )


def test_registration_normalizes_manifest_and_rejects_implicit_replacement(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.register_spawned(
        engine_id="candidate-one",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        routing_state="candidate",
        bundle={"toolbox_id": "demo", "manifest_hash": "a" * 64},
    )
    assert service.get_registration("candidate-one")["bundle"]["manifest_hash"] == "sha256:" + "a" * 64
    with pytest.raises(RuntimeError, match="runtime_registration_conflict"):
        service.register_spawned(
            engine_id="candidate-one",
            pid=5678,
            command=["python", "replacement.py"],
            executor_kind="toolbox_executor",
            routing_state="candidate",
            bundle={"toolbox_id": "demo", "manifest_hash": "b" * 64},
        )


def test_duplicate_toolbox_execution_returns_status_without_waiting(tmp_path: Path) -> None:
    service = _service(tmp_path)
    request_id = f"duplicate-r6-{time.time_ns()}"
    service.register_spawned(
        engine_id="executor-one",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        bundle={"toolbox_id": "demo", "sandbox_profile_id": "default"},
        environment={"environment_key": "demo-env"},
        tool_access={"allowed_tool_names": ["write"]},
    )
    submitted = threading.Event()
    release = threading.Event()

    class BlockingBase:
        def resources(self, environment_key):
            return {"environment_key": environment_key, "active": 0}

        def submit_request(self, **kwargs):
            submitted.set()
            release.wait(2)
            return {"status": "ok", "request": {"request_id": kwargs["request_id"], "status": "queued"}}

        def claim_dispatch(self, **kwargs):
            return {"status": "error", "request": {"request_id": kwargs["request_id"], "status": "canceled"}}

    service._toolbox_runtime_base = lambda: BlockingBase()  # type: ignore[method-assign]
    result: list[dict] = []
    worker = threading.Thread(
        target=lambda: result.append(
            service.toolbox_execute(
                engine_id="executor-one",
                execution_request_id=request_id,
                tool_call={"id": "call-1", "name": "write", "arguments": {}},
            )
        ),
        daemon=True,
    )
    worker.start()
    assert submitted.wait(2)
    started = time.monotonic()
    duplicate = service.toolbox_execute(
        engine_id="executor-one",
        execution_request_id=request_id,
        tool_call={"id": "call-1", "name": "write", "arguments": {}},
    )
    assert time.monotonic() - started < 1.0
    assert duplicate["lifecycle"] in {"queued", "running"}
    release.set()
    worker.join(2)
    assert not worker.is_alive()


def test_toolbox_cancellation_acknowledges_before_teardown(tmp_path: Path) -> None:
    service = _service(tmp_path)
    request_id = f"cancel-r6-{time.time_ns()}"
    prepared = service._hosted_operations.prepare(
        owner_actor_id="service:local",
        execution_kind=HostedExecutionKind.TOOLBOX,
        selector=HostedOperationSelector(kind="engine_id", id="executor-one"),
        namespace="engine:executor-one",
        request_id=request_id,
        fingerprint=hosted_execution_fingerprint({"r6": "cancel"}),
        metadata={"engine_id": "executor-one", "tool_name": "write"},
    )
    operation = dict(prepared["status"]["operation"])
    service._hosted_operations.mark_dispatch_claimed(operation_id=operation["operation_id"])
    finished = threading.Event()

    def slow_cancel(**kwargs):
        time.sleep(0.4)
        finished.set()
        return service._hosted_operations.finish(
            operation_id=operation["operation_id"],
            lifecycle="terminal_cancellation",
            envelope={"status": "canceled"},
        )

    service._cancel_toolbox_operation = slow_cancel  # type: ignore[method-assign]
    started = time.monotonic()
    acknowledged = service.hosted_operation_cancel(ref=operation)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert acknowledged["progress"]["phase"] == "cancellation"
    assert not finished.is_set()
    assert finished.wait(2)


def test_describe_refresh_is_durable_and_live(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service._toolbox_describe_live = lambda **kwargs: {"status": "ok", "cache": "live"}  # type: ignore[method-assign]
    started = service.toolbox_describe_refresh(engine_id="executor-one", request_id="refresh-r6")
    terminal = service._hosted_operations.wait_for_terminal(
        operation_id=started["operation"]["operation_id"], timeout_seconds=5
    )
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["description"]["cache"] == "live"


def test_parent_death_hook_reports_native_or_fallback_state() -> None:
    report = configure_parent_death_signal()
    assert report["status"] in {"job_object", "prctl", "unsupported"}
    assert isinstance(report["configured"], bool)
