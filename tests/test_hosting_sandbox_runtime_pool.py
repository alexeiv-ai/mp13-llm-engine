from __future__ import annotations

import threading
import time

import pytest

from hosting.sandbox.runtime_base import (
    HOSTED_STREAM_EVENT_TYPES,
    HostedPoolKey,
    HostedRequestLifecycle,
    HostedStreamEvent,
    HostedWorkerSlot,
    hosted_stream_cancel_message,
)
from hosting.sandbox.runtime_pool import HostedProcessPoolRegistry


def _factory(pool_key: HostedPoolKey, capacity: int) -> HostedWorkerSlot:
    env_key = pool_key.normalized()["environment_key"]
    return HostedWorkerSlot(
        engine_id=f"{pool_key.normalized()['sandbox_kind']}-{env_key}-1",
        environment_key=env_key,
        capacity=capacity,
        pid=1234,
        status="running",
    )


def _request(request_id: str, *, submitted_at: float = 10.0) -> HostedRequestLifecycle:
    return HostedRequestLifecycle(
        request_id=request_id,
        environment_key="env-a",
        sandbox_kind="workflow_python",
        profile="helper",
        submitted_at=submitted_at,
    )


def test_registry_creates_pool_and_schedules_first_worker() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="workflow_python", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=2)

    out = pool.submit_request(_request("req-1"), factory=_factory, start_timestamp=10.5)

    assert out["status"] == "ok"
    assert out["worker"]["engine_id"] == "workflow_python-env-a-1"
    assert out["request"]["queue_wait_ms"] == 500
    resources = pool.resources()
    assert resources["environment_key"] == "env-a"
    assert resources["metrics"]["desired_capacity"] == 2
    assert resources["metrics"]["active_calls"] == 1
    assert resources["metrics"]["available_slots"] == 1


def test_pool_reports_capacity_exceeded_and_records_saturation() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="workflow_python", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=1)

    assert pool.submit_request(_request("req-1"), factory=_factory, start_timestamp=10.0)["status"] == "ok"
    out = pool.submit_request(_request("req-2"), factory=_factory, start_timestamp=10.1)

    assert out["status"] == "error"
    assert out["reason"] == "capacity_exceeded"
    resources = pool.resources()["metrics"]
    assert resources["saturation_count"] == 1
    assert resources["recent_requests"][0]["request_id"] == "req-2"
    assert resources["recent_requests"][0]["reason"] == "capacity_exceeded"


def test_finish_and_cancel_update_worker_and_metrics() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="workflow_python", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=2)

    pool.submit_request(_request("req-1"), factory=_factory, start_timestamp=10.0)
    pool.submit_request(_request("req-2"), factory=_factory, start_timestamp=10.1)
    finished = pool.finish_request("req-1", status="timeout", reason="timeout", timestamp=11.0)
    canceled = pool.cancel_request("req-2", timestamp=11.2)

    assert finished["request"]["status"] == "timeout"
    assert canceled["request"]["status"] == "canceled"
    metrics = pool.resources()["metrics"]
    assert metrics["active_calls"] == 0
    assert metrics["timeout_count"] == 1
    assert metrics["cancellation_count"] == 1
    assert [row["request_id"] for row in metrics["recent_requests"]] == ["req-1", "req-2"]


def test_error_metrics_are_grouped_by_reason() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="workflow_python", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=1)

    pool.submit_request(_request("req-1"), factory=_factory, start_timestamp=10.0)
    pool.finish_request("req-1", status="error", reason="boom", timestamp=10.5)

    metrics = pool.resources()["metrics"]
    assert metrics["error_count"] == 1
    assert metrics["errors_by_reason"] == {"boom": 1}


def test_pool_records_progress_snapshot_and_request_status() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="workflow_python", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=1)

    pool.submit_request(_request("req-1"), factory=_factory, start_timestamp=10.0)
    event = HostedStreamEvent(
        type="progress",
        request_id="req-1",
        sequence=3,
        timestamp=10.4,
        payload={"progress_percent": 50, "progress_text": "Half done"},
    )
    recorded = pool.record_stream_event("req-1", event)
    status = registry.request_status(key, "req-1")

    assert recorded["status"] == "ok"
    assert status["status"] == "ok"
    assert status["source"] == "active"
    assert status["request"]["stream_event_count"] == 1
    assert status["request"]["latest_progress"]["payload"]["progress_percent"] == 50


def test_common_stream_event_contract_and_cancel_message() -> None:
    assert "progress" in HOSTED_STREAM_EVENT_TYPES
    assert "artifact" in HOSTED_STREAM_EVENT_TYPES
    assert "done" in HOSTED_STREAM_EVENT_TYPES

    event = HostedStreamEvent(type="log", request_id="req-1", payload={"message": "debug"}).to_dict()
    cancel = hosted_stream_cancel_message("req-1", reason="user")

    assert event["type"] == "log"
    assert event["payload"] == {"message": "debug"}
    with pytest.raises(ValueError, match="unsupported_stream_event_kind"):
        HostedStreamEvent(type="custom-debug", request_id="req-1", payload={"message": "debug"}).to_dict()
    assert cancel == {"action": "cancel", "request_id": "req-1", "reason": "user"}


def test_request_status_checks_recent_finished_requests() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="workflow_python", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=1)

    pool.submit_request(_request("req-1"), factory=_factory, start_timestamp=10.0)
    pool.record_stream_event(
        "req-1",
        {
            "type": "progress",
            "request_id": "req-1",
            "sequence": 1,
            "timestamp": 10.2,
            "payload": {"progress_text": "Finishing"},
        },
    )
    pool.finish_request("req-1", status="ok", timestamp=10.5)
    status = registry.request_status(key, "req-1")

    assert status["status"] == "ok"
    assert status["source"] == "active"
    assert status["request"]["status"] == "ok"
    assert status["request"]["latest_progress"]["payload"]["progress_text"] == "Finishing"


def test_registry_resources_roll_up_pools() -> None:
    registry = HostedProcessPoolRegistry()
    registry.get_or_create(HostedPoolKey(sandbox_kind="workflow_python", environment_key="env-a"), desired_capacity=1)
    registry.get_or_create(HostedPoolKey(sandbox_kind="workflow_js", environment_key="env-b"), desired_capacity=2)

    out = registry.resources()

    assert out["status"] == "ok"
    assert out["pool_count"] == 2
    assert sorted(out["pools"].keys()) == ["workflow_js/env-b", "workflow_python/env-a"]


def test_bounded_queue_waits_then_admits_atomically() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="toolbox_executor", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=1, queue_policy="bounded", queue_depth=1, queue_timeout_seconds=1.0)
    assert pool.submit_request(_request("req-1"), factory=_factory)["status"] == "ok"
    result: dict[str, object] = {}

    def _submit() -> None:
        result.update(
            pool.submit_request(
                _request("req-2"),
                factory=_factory,
                queue_policy="bounded",
                queue_depth=1,
                queue_timeout_seconds=1.0,
            )
        )

    thread = threading.Thread(target=_submit)
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and pool.resources()["metrics"]["queued_calls"] != 1:
        time.sleep(0.01)
    assert pool.resources()["metrics"]["queued_calls"] == 1
    pool.finish_request("req-1", status="ok")
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert result["status"] == "ok"
    assert result["request"]["admission"] == "admitted"
    assert result["request"]["queue_wait_ms"] is not None


def test_bounded_queue_reports_full_and_timeout() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="toolbox_executor", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=1, queue_policy="bounded", queue_depth=1, queue_timeout_seconds=0.03)
    assert pool.submit_request(_request("req-1"), factory=_factory)["status"] == "ok"
    queued_result: dict[str, object] = {}
    thread = threading.Thread(
        target=lambda: queued_result.update(
            pool.submit_request(_request("req-2"), factory=_factory, queue_policy="bounded", queue_depth=1, queue_timeout_seconds=0.03)
        )
    )
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and pool.resources()["metrics"]["queued_calls"] != 1:
        time.sleep(0.01)
    full = pool.submit_request(_request("req-3"), factory=_factory, queue_policy="bounded", queue_depth=1, queue_timeout_seconds=0.0)
    assert full["reason"] == "queue_full"
    thread.join(timeout=1.0)
    assert queued_result["reason"] == "queue_timeout"


def test_concurrency_policy_serial_and_keyed_gates_are_independent() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="toolbox_executor", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=3, queue_policy="fail_fast")
    assert pool.submit_request(_request("serial-1"), factory=_factory, concurrency={"mode": "serial", "group": "mutate"})["status"] == "ok"
    serial = pool.submit_request(_request("serial-2"), factory=_factory, concurrency={"mode": "serial", "group": "mutate"})
    assert serial["reason"] == "capacity_exceeded"
    pool.finish_request("serial-1")

    assert pool.submit_request(
        _request("key-a"), factory=_factory, concurrency={"mode": "keyed", "group": "file", "resource_key": "a"}
    )["status"] == "ok"
    assert pool.submit_request(
        _request("key-b"), factory=_factory, concurrency={"mode": "keyed", "group": "file", "resource_key": "b"}
    )["status"] == "ok"
    same_key = pool.submit_request(
        _request("key-a-2"), factory=_factory, concurrency={"mode": "keyed", "group": "file", "resource_key": "a"}
    )
    assert same_key["reason"] == "capacity_exceeded"


def test_canceling_queued_request_wakes_submitter_without_touching_sibling() -> None:
    registry = HostedProcessPoolRegistry()
    key = HostedPoolKey(sandbox_kind="toolbox_executor", environment_key="env-a")
    pool = registry.get_or_create(key, desired_capacity=1, queue_policy="bounded", queue_depth=1, queue_timeout_seconds=1.0)
    assert pool.submit_request(_request("req-1"), factory=_factory)["status"] == "ok"
    result: dict[str, object] = {}

    def _submit() -> None:
        result.update(pool.submit_request(_request("req-2"), factory=_factory, queue_policy="bounded", queue_depth=1, queue_timeout_seconds=1.0))

    thread = threading.Thread(target=_submit)
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and pool.resources()["metrics"]["queued_calls"] != 1:
        time.sleep(0.01)
    canceled = pool.cancel_request("req-2")
    thread.join(timeout=1.0)
    assert canceled["request"]["status"] == "canceled"
    assert result["reason"] == "canceled"
    assert pool.resources()["metrics"]["active_calls"] == 1
