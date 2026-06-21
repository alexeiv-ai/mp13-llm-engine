from __future__ import annotations

from hosting.sandbox.process_base import HostedProcessSandboxBase
from hosting.sandbox.child_runtime import HostedActiveChildRuntimeRegistry
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
    assert [row["kind"] for row in received["batch"]["frames"]] == ["started", "progress"]
    assert closed["closed"] is True
    assert status["request"]["status"] == "ok"
    assert status["request"]["latest_progress"]["payload"]["progress_percent"] == 40


def test_process_base_stream_reports_progress_replacement_when_retention_is_exceeded() -> None:
    base = WorkflowPythonProcessBase()

    opened = base.stream_open(
        environment_key="env-a",
        request_id="req-stream-retention",
        profile="node",
        factory=_factory,
        max_events=2,
    )
    for value in range(4):
        base.stream_emit(
            stream_id=str(opened["stream_id"]),
            event_type="progress",
            payload={"value": value},
        )
    received = base.stream_recv(stream_id=str(opened["stream_id"]), max_items=10)
    status = base.request_status(environment_key="env-a", request_id="req-stream-retention")

    assert received["max_events"] == 2
    assert received["dropped_event_count"] == 3
    assert [row["type"] for row in received["events"]] == ["started", "progress"]
    assert received["events"][-1]["payload"].get("value") == 3
    assert received["batch"]["loss"] == {"output": 0, "event": 3, "audit": 0}
    assert received["batch"]["frames"][-1]["value"] == 3
    assert status["request"]["stream_event_count"] == 5


def test_process_base_stream_replaces_latest_heartbeat_and_metric() -> None:
    base = WorkflowPythonProcessBase()

    opened = base.stream_open(
        environment_key="env-a",
        request_id="req-stream-replacement",
        profile="node",
        factory=_factory,
        max_events=8,
    )
    base.stream_emit(stream_id=str(opened["stream_id"]), event_type="heartbeat", payload={"status": "first"})
    base.stream_emit(stream_id=str(opened["stream_id"]), event_type="heartbeat", payload={"status": "second"})
    base.stream_emit(stream_id=str(opened["stream_id"]), event_type="metric", payload={"name": "rows", "current": 1})
    base.stream_emit(stream_id=str(opened["stream_id"]), event_type="metric", payload={"name": "rows", "current": 2})
    received = base.stream_recv(stream_id=str(opened["stream_id"]), max_items=8)

    assert [row["type"] for row in received["events"]] == ["started", "heartbeat", "metric"]
    assert received["events"][1]["payload"]["status"] == "second"
    assert received["events"][2]["payload"]["current"] == 2
    assert received["batch"]["loss"] == {"output": 0, "event": 2, "audit": 0}


def test_process_base_stream_keeps_first_bounded_non_stdout_events() -> None:
    for event_type, payload in [
        ("stderr", {"text": "first"}),
        ("log", {"level": "info", "message": "first"}),
        ("artifact", {"name": "report", "ref": "artifact://first"}),
    ]:
        base = WorkflowPythonProcessBase()
        opened = base.stream_open(
            environment_key="env-a",
            request_id=f"req-stream-{event_type}",
            profile="node",
            factory=_factory,
            max_events=2,
        )
        base.stream_emit(stream_id=str(opened["stream_id"]), event_type=event_type, payload=payload)
        base.stream_emit(stream_id=str(opened["stream_id"]), event_type=event_type, payload={**payload, "message": "dropped", "text": "dropped"})
        received = base.stream_recv(stream_id=str(opened["stream_id"]), max_items=8)

        assert [row["type"] for row in received["events"]] == ["started", event_type]
        assert received["events"][1]["payload"] == payload
        assert received["dropped_event_count"] == 1


def test_process_base_stream_keeps_first_output_and_prioritizes_control() -> None:
    base = WorkflowPythonProcessBase()

    opened = base.stream_open(
        environment_key="env-a",
        request_id="req-stream-lanes",
        profile="node",
        factory=_factory,
        max_events=2,
    )
    base.stream_emit(stream_id=str(opened["stream_id"]), event_type="stdout", payload={"text": "first"})
    base.stream_emit(stream_id=str(opened["stream_id"]), event_type="stdout", payload={"text": "dropped"})
    canceled = base.stream_send(
        stream_id=str(opened["stream_id"]),
        message={"action": "cancel", "reason": "user"},
    )
    received = base.stream_recv(stream_id=str(opened["stream_id"]), max_items=1)

    assert canceled["accepted"] is True
    assert [row["type"] for row in received["events"]] == ["canceled"]
    assert received["batch"]["loss"] == {"output": 2, "event": 0, "audit": 0}
    assert received["batch"]["frames"][0]["kind"] == "canceled"
    assert received["retained_event_count"] == 1


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


def test_active_child_runtime_registry_tracks_resources_and_cancel() -> None:
    class Proc:
        pid = 4321

        def poll(self):
            return None

    class Runtime:
        proc = Proc()
        python_executable = "python-test"
        _cancel_requested = False

        def cancel(self):
            self._cancel_requested = True
            return True

    registry = HostedActiveChildRuntimeRegistry()
    runtime = Runtime()

    registry.register_active("req-1", runtime)
    resources = registry.resources()
    canceled = registry.cancel("req-1")
    registry.unregister_active("req-1")

    assert resources["active_count"] == 1
    assert resources["processes"][0]["pid"] == 4321
    assert resources["processes"][0]["python_executable"] == "python-test"
    assert canceled["canceled"] is True
    assert runtime._cancel_requested is True
    assert registry.resources()["active_count"] == 0
