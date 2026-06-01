from __future__ import annotations

from hosting.sandbox.runtime_base import (
    HostedEnvironmentKeySpec,
    HostedPoolKey,
    HostedPoolMetrics,
    HostedRequestLifecycle,
    HostedRuntimeIdentity,
    HostedStreamEvent,
    HostedWorkerSlot,
    hosted_cancellation_result,
    hosted_registration_environment_metadata,
    hosted_resource_response,
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
