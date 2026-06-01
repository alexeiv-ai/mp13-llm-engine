from __future__ import annotations

from hosting.sandbox.process_base import HostedProcessSandboxBase
from hosting.sandbox.runtime_base import HostedPoolKey, HostedStreamEvent, HostedWorkerSlot


class WorkflowPythonProcessBase(HostedProcessSandboxBase):
    sandbox_kind = "workflow_python"


def _factory(pool_key: HostedPoolKey, capacity: int) -> HostedWorkerSlot:
    return HostedProcessSandboxBase.worker_slot(
        engine_id=f"{pool_key.normalized()['sandbox_kind']}-{pool_key.normalized()['environment_key']}",
        environment_key=pool_key.normalized()["environment_key"],
        capacity=capacity,
        pid=1234,
        status="running",
    )


def test_process_base_tracks_pool_capacity_and_request_lifetime() -> None:
    base = WorkflowPythonProcessBase()

    submitted = base.submit_request(
        environment_key="env-a",
        request_id="req-1",
        profile="helper",
        desired_capacity=2,
        factory=_factory,
        operation_id="condition",
        input_bytes=32,
    )
    finished = base.finish_request(environment_key="env-a", request_id="req-1", output_bytes=64)
    resources = base.resources("env-a")

    assert submitted["status"] == "ok"
    assert submitted["worker"]["capacity"] == 2
    assert finished["request"]["status"] == "ok"
    assert finished["request"]["input_bytes"] == 32
    assert finished["request"]["output_bytes"] == 64
    assert resources["pool_id"] == "workflow_python/env-a"
    assert resources["metrics"]["desired_capacity"] == 2


def test_process_base_reports_status_progress_and_cancel() -> None:
    base = WorkflowPythonProcessBase()

    base.submit_request(
        environment_key="env-a",
        request_id="req-1",
        profile="node",
        factory=_factory,
    )
    base.record_stream_event(
        environment_key="env-a",
        request_id="req-1",
        event=HostedStreamEvent(
            type="progress",
            request_id="req-1",
            payload={"progress_percent": 25},
        ),
    )
    status = base.request_status(environment_key="env-a", request_id="req-1")
    canceled = base.cancel_request(environment_key="env-a", request_id="req-1")

    assert status["status"] == "ok"
    assert status["request"]["latest_progress"]["payload"]["progress_percent"] == 25
    assert canceled["request"]["status"] == "canceled"
    assert base.resources("env-a")["metrics"]["cancellation_count"] == 1


def test_process_base_stream_lifecycle_records_events_and_close() -> None:
    base = WorkflowPythonProcessBase()

    opened = base.stream_open(
        environment_key="env-a",
        request_id="req-stream",
        profile="node",
        factory=_factory,
    )
    base.stream_emit(
        stream_id=str(opened["stream_id"]),
        event_type="progress",
        payload={"progress_percent": 40},
    )
    received = base.stream_recv(stream_id=str(opened["stream_id"]), max_items=4)
    closed = base.stream_close(stream_id=str(opened["stream_id"]))
    status = base.request_status(environment_key="env-a", request_id="req-stream")

    assert opened["status"] == "ok"
    assert [row["type"] for row in received["events"]] == ["started", "progress"]
    assert closed["closed"] is True
    assert status["request"]["status"] == "ok"
    assert status["request"]["latest_progress"]["payload"]["progress_percent"] == 40


def test_process_base_stream_cancel_uses_pool_cancellation() -> None:
    base = WorkflowPythonProcessBase()

    opened = base.stream_open(
        environment_key="env-a",
        request_id="req-stream",
        profile="node",
        factory=_factory,
    )
    canceled = base.stream_send(
        stream_id=str(opened["stream_id"]),
        message={"action": "cancel", "reason": "user"},
    )
    received = base.stream_recv(stream_id=str(opened["stream_id"]), max_items=4)

    assert canceled["accepted"] is True
    assert canceled["workflow_pool_cancel"]["request"]["status"] == "canceled"
    assert received["canceled"] is True
    assert received["events"][-1]["type"] == "canceled"


def test_process_base_missing_pool_results_are_structured() -> None:
    base = WorkflowPythonProcessBase()

    assert base.resources("missing") == {
        "status": "not_found",
        "sandbox_kind": "workflow_python",
        "environment_key": "missing",
    }
    assert base.cancel_request(environment_key="missing", request_id="req-1") == {
        "status": "not_found",
        "environment_key": "missing",
        "request_id": "req-1",
    }
