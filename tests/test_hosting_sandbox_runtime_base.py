from __future__ import annotations

import pytest

from hosting.sandbox.runtime_base import (
    HostedEnvironmentKeySpec,
    HostedPoolKey,
    HostedPoolMetrics,
    HostedRequestLifecycle,
    HostedRuntimeIdentity,
    HostedStreamBatch,
    HostedStreamContext,
    HostedStreamEvent,
    HostedStreamFrame,
    HostedStreamLoss,
    HostedWorkerSlot,
    HOSTED_IPC_MESSAGE_FAMILIES,
    HOSTED_STREAM_CONTRACT_VERSION,
    HOSTED_STREAM_KIND_REGISTRY,
    hosted_cancellation_result,
    hosted_log_summary,
    hosted_registration_environment_metadata,
    hosted_resource_response,
    hosted_stream_kind_lane,
    hosted_stream_kind_spec,
    hosted_stream_validate_kind,
    sandbox_policy_hash,
)


def _runtime() -> HostedRuntimeIdentity:
    return HostedRuntimeIdentity(
        runtime_kind="workflow_python",
        profile="helper",
        runtime_hash="python-3.12-demo",
        runtime_version="3.12.0",
    )


def _policy(enabled: bool = True):
    return {
        "sandbox": {
            "enabled": enabled,
            "profile": "workflow_python_helper_v1",
            "network": {"mode": "disabled"},
            "brokered_io": {"filesystem": False, "http": False, "subprocess": False},
        }
    }


def test_environment_key_is_stable_for_normalized_dependency_order() -> None:
    left = HostedEnvironmentKeySpec(
        environment_name="workflow-python-helper",
        runtime=_runtime(),
        sandbox_policy=_policy(),
        required_imports=["json", "math", "json"],
        package_pins={"b": "2.0", "a": "1.0"},
    )
    right = HostedEnvironmentKeySpec(
        environment_name="workflow-python-helper",
        runtime=_runtime(),
        sandbox_policy=_policy(),
        required_imports=["json", "math"],
        package_pins={"a": "1.0", "b": "2.0"},
    )

    assert left.full_key() == right.full_key()
    assert left.short_key() == right.short_key()
    assert left.to_dict()["required_imports"] == ["json", "math"]
    assert left.to_dict()["package_pins"] == {"a": "1.0", "b": "2.0"}


def test_environment_key_changes_for_policy_or_runtime_changes() -> None:
    base = HostedEnvironmentKeySpec(
        environment_name="workflow-python-helper",
        runtime=_runtime(),
        sandbox_policy=_policy(enabled=True),
        required_imports=["json"],
    )
    changed_policy = HostedEnvironmentKeySpec(
        environment_name="workflow-python-helper",
        runtime=_runtime(),
        sandbox_policy=_policy(enabled=False),
        required_imports=["json"],
    )
    changed_runtime = HostedEnvironmentKeySpec(
        environment_name="workflow-python-helper",
        runtime=HostedRuntimeIdentity(runtime_kind="workflow_python", profile="node", runtime_hash="python-3.12-demo"),
        sandbox_policy=_policy(enabled=True),
        required_imports=["json"],
    )

    assert base.full_key() != changed_policy.full_key()
    assert base.full_key() != changed_runtime.full_key()


def test_sandbox_policy_hash_normalizes_missing_defaults() -> None:
    assert sandbox_policy_hash({}) == sandbox_policy_hash({"sandbox": {}})
    assert sandbox_policy_hash(_policy(enabled=True)) != sandbox_policy_hash(_policy(enabled=False))


def test_pool_key_and_worker_slot_shapes() -> None:
    pool_key = HostedPoolKey(sandbox_kind="workflow_python", environment_key="abc123")
    worker = HostedWorkerSlot(
        engine_id="workflow-python-abc123-1",
        environment_key="abc123",
        capacity=3,
        active_request_ids=["req-1", "req-1", "req-2"],
        pid=1234,
        status="running",
    )

    assert pool_key.pool_id() == "workflow_python/abc123"
    assert worker.available_slots() == 1
    assert worker.to_dict()["active_request_ids"] == ["req-1", "req-2"]
    assert worker.to_dict()["active_calls"] == 2


def test_request_lifecycle_calculates_queue_execution_and_lifetime() -> None:
    request = HostedRequestLifecycle(
        request_id="req-1",
        environment_key="abc123",
        sandbox_kind="workflow_python",
        profile="helper",
        submitted_at=10.0,
    )

    request.mark_started(timestamp=10.25, engine_id="worker-1")
    request.mark_finished("ok", timestamp=11.0)

    row = request.to_dict()
    assert row["engine_id"] == "worker-1"
    assert row["status"] == "ok"
    assert row["queue_wait_ms"] == 250
    assert row["execution_ms"] == 750
    assert row["lifetime_ms"] == 1000


def test_stream_event_and_pool_metrics_shapes() -> None:
    event = HostedStreamEvent(
        type="progress",
        request_id="req-1",
        sequence=2,
        timestamp=12.0,
        payload={"message": "running"},
    )
    request = HostedRequestLifecycle(
        request_id="req-1",
        environment_key="abc123",
        sandbox_kind="workflow_python",
        profile="node",
        submitted_at=10.0,
    )
    worker = HostedWorkerSlot(engine_id="worker-1", environment_key="abc123", capacity=2, active_request_ids=["req-1"])
    metrics = HostedPoolMetrics(
        desired_capacity=2,
        workers=[worker],
        recent_requests=[request],
        saturation_count=1,
        timeout_count=2,
        cancellation_count=3,
        error_count=4,
        errors_by_reason={"boom": 4},
    )

    assert event.to_dict() == {
        "type": "progress",
        "request_id": "req-1",
        "sequence": 2,
        "timestamp": 12.0,
        "payload": {"message": "running"},
    }
    row = metrics.to_dict()
    assert row["desired_capacity"] == 2
    assert row["worker_count"] == 1
    assert row["active_calls"] == 1
    assert row["available_slots"] == 1
    assert row["errors_by_reason"] == {"boom": 4}


def test_stream_kind_registry_declares_lane_and_queue_policy() -> None:
    assert hosted_stream_kind_lane("stdout") == "output"
    assert hosted_stream_kind_spec("progress").replacement_fields == ("key",)
    assert hosted_stream_kind_spec("host_call").queue_decision == "non_droppable"
    assert hosted_stream_kind_spec("done").final is True
    assert HOSTED_STREAM_KIND_REGISTRY["approval"].decision_bearing is True

    with pytest.raises(ValueError, match="unsupported_stream_event_kind"):
        hosted_stream_validate_kind("unknown-kind")


def test_stream_frame_and_batch_shape_are_compact_and_expandable() -> None:
    frame = HostedStreamFrame(
        kind="stdout",
        dt_ms=8,
        text="Installing package\n",
        boundary=True,
        expected_bytes=1024,
        offset=0,
        length=19,
        extra={"producer": "pip"},
    )
    batch = HostedStreamBatch(
        context=HostedStreamContext(stream_id="stream-1", request_id="req-1", instance_id="worker-1"),
        frames=[
            HostedStreamFrame(kind="progress", pct=40, message="installing"),
            frame,
            HostedStreamFrame(kind="done", dt_ms=11, status="ok"),
        ],
        sequence=100,
        timestamp_ms=1781913600000,
        loss=HostedStreamLoss(output=1),
        more=True,
    )

    row = batch.to_dict()
    assert row == {
        "version": HOSTED_STREAM_CONTRACT_VERSION,
        "context": {"stream_id": "stream-1", "request_id": "req-1", "instance_id": "worker-1"},
        "base": {"sequence": 100, "timestamp_ms": 1781913600000},
        "loss": {"output": 1, "event": 0, "audit": 0},
        "frames": [
            {"dt_ms": 0, "kind": "progress", "message": "installing", "pct": 40.0},
            {
                "dt_ms": 8,
                "kind": "stdout",
                "expected_bytes": 1024,
                "offset": 0,
                "length": 19,
                "text": "Installing package\n",
                "boundary": True,
                "producer": "pip",
            },
            {"dt_ms": 11, "kind": "done", "status": "ok"},
        ],
        "more": True,
    }
    expanded = batch.expanded_frames()
    assert expanded[0]["sequence"] == 100
    assert expanded[1]["timestamp_ms"] == 1781913600008
    assert expanded[2]["request_id"] == "req-1"


def test_stream_batch_parser_validates_version_and_event_kind() -> None:
    with pytest.raises(ValueError, match="unsupported_stream_version"):
        HostedStreamBatch.from_dict({"version": 99, "frames": []})

    with pytest.raises(ValueError, match="unsupported_stream_event_kind"):
        HostedStreamBatch(
            context=HostedStreamContext(request_id="req-1"),
            frames=[HostedStreamFrame(kind="unknown")],
        ).to_dict()

    parsed = HostedStreamBatch.from_dict(
        {
            "version": HOSTED_STREAM_CONTRACT_VERSION,
            "context": {"request_id": "req-1"},
            "base": {"sequence": 7, "timestamp_ms": 10},
            "loss": {"event": 2},
            "frames": [{"kind": "metric", "name": "rows", "current": 3, "unit": "count"}],
        }
    )
    assert parsed.to_dict()["frames"] == [{"dt_ms": 0, "kind": "metric", "current": 3.0, "name": "rows", "unit": "count"}]
    assert parsed.loss.detected() is True


def test_shared_registration_resource_and_cancel_shapes() -> None:
    env = hosted_registration_environment_metadata(
        environment={
            "environment_key": "env-a",
            "environment_name": "workflow-python-helper",
            "required_imports": ["json", "json"],
            "package_pins": {"demo": "1.0.0"},
            "install_status": "verified",
        },
        runtime_kind="workflow_python",
        profile="helper",
    )
    resources = hosted_resource_response(
        sandbox_kind="workflow_python",
        profile="helper",
        environment_key="env-a",
        engine_id="wf-py",
        pool={"pool_id": "workflow_python/env-a"},
        resources={"status": "ok", "capacity": 2},
    )
    cancel = hosted_cancellation_result(
        request_id="req-1",
        environment_key="env-a",
        canceled=True,
        worker_result={"status": "ok"},
        pool_result={"status": "ok"},
    )

    assert env["workflow_runtime_kind"] == "workflow_python"
    assert env["required_imports"] == ["json"]
    assert resources["workflow_pool"]["pool_id"] == "workflow_python/env-a"
    assert resources["capacity"] == 2
    assert cancel["canceled"] is True
    assert cancel["workflow_pool_cancel"] == {"status": "ok"}


def test_base_ipc_message_family_names_are_stable() -> None:
    assert HOSTED_IPC_MESSAGE_FAMILIES == [
        "hello",
        "rpc_call",
        "stream_open",
        "stream_recv",
        "stream_send",
        "stream_close",
        "shutdown",
    ]


def test_hosted_log_summary_truncates_each_stream() -> None:
    out = hosted_log_summary(stdout="abcdef", stderr="xyz", max_bytes=3)

    assert out["stdout"] == "abc"
    assert out["stderr"] == "xyz"
    assert out["summary"] == "abc"
    assert out["stdout_truncated"] is True
    assert out["stderr_truncated"] is False
    assert out["summary_truncated"] is True
