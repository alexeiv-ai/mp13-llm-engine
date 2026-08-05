from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from hosting.daemon import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel
from hosting.operation_contract import HostedOperationLifecycle, hosted_execution_fingerprint
from hosting.service.host_service import EngineHostService


def _service(tmp_path: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )


def _register_executor(service: EngineHostService) -> None:
    service.register_spawned(
        engine_id="executor-a",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        bundle={"toolbox_id": "demo", "sandbox_profile_id": "default"},
        environment={"environment_key": "demo-env"},
        tool_access={"allowed_tool_names": ["write"]},
    )


def test_service_duplicate_replays_canonical_terminal_status_without_second_dispatch(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _register_executor(service)
    dispatches: list[dict] = []
    service._ipc_call = lambda **kwargs: dispatches.append(kwargs) or {  # type: ignore[method-assign]
        "status": "ok",
        "tool_call": {"id": "call-1", "result": "done"},
    }
    call = {"id": "call-1", "name": "write", "arguments": {"value": 1}}

    first = service.toolbox_execute(engine_id="executor-a", execution_request_id="request-1", tool_call=call)
    duplicate = service.toolbox_execute(engine_id="executor-a", execution_request_id="request-1", tool_call=call)
    conflict = service.toolbox_execute(
        engine_id="executor-a",
        execution_request_id="request-1",
        tool_call={"id": "call-2", "name": "write", "arguments": {"value": 2}},
    )

    assert len(dispatches) == 1
    assert first["contract"] == duplicate["contract"] == "hosting.operation_status"
    assert first["lifecycle"] == duplicate["lifecycle"] == "terminal_success"
    assert duplicate["operation"] == first["operation"]
    assert duplicate["result"]["tool_call"]["result"] == "done"
    assert conflict["api_status"] == "error"
    assert conflict["lifecycle"] == "idempotency_conflict"

    recreated = _service(tmp_path)
    status = recreated.hosted_operation_status(ref=first["operation"])
    assert status["lifecycle"] == "terminal_success"
    assert status["result"]["tool_call"]["result"] == "done"


def test_service_ref_authorization_hides_operation_from_different_actor(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _register_executor(service)
    service._ipc_call = lambda **_kwargs: {"status": "ok", "result": 7}  # type: ignore[method-assign]
    terminal = service.toolbox_execute(
        engine_id="executor-a",
        execution_request_id="request-1",
        tool_call={"id": "call-1", "name": "write", "arguments": {}},
        owner_actor_id="actor:a",
    )

    hidden_status = service.hosted_operation_status(ref=terminal["operation"], owner_actor_id="actor:b")
    hidden_cancel = service.hosted_operation_cancel(ref=terminal["operation"], owner_actor_id="actor:b")
    assert hidden_status["lifecycle"] == hidden_cancel["lifecycle"] == "unknown_outside_retention"
    assert hidden_status["reason"] == hidden_cancel["reason"] == "operation_not_found"


def test_service_queued_cancel_race_never_invokes_tool_and_cleans_pool_request(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _register_executor(service)
    submitted = threading.Event()
    release = threading.Event()
    canceled: list[str] = []
    ipc_calls: list[dict] = []

    class _BlockingBase:
        def submit_request(self, **kwargs):
            submitted.set()
            release.wait(2)
            return {"status": "ok", "request": {"request_id": kwargs["request_id"], "status": "queued"}}

        def claim_dispatch(self, **kwargs):
            return {"status": "ok", "request": {"request_id": kwargs["request_id"], "status": "running"}}

        def cancel_request(self, **kwargs):
            canceled.append(kwargs["request_id"])
            return {"status": "ok", "request": {"request_id": kwargs["request_id"], "status": "canceled"}}

    service._toolbox_runtime_base = lambda: _BlockingBase()  # type: ignore[method-assign]
    service._ipc_call = lambda **kwargs: ipc_calls.append(kwargs) or {"status": "ok"}  # type: ignore[method-assign]
    results: list[dict] = []
    worker = threading.Thread(
        target=lambda: results.append(
            service.toolbox_execute(
                engine_id="executor-a",
                execution_request_id="queued-cancel",
                tool_call={"id": "call-1", "name": "write", "arguments": {}},
            )
        )
    )
    worker.start()
    assert submitted.wait(2)
    record = service._hosted_operations.get_by_request(
        owner_actor_id="service:local",
        namespace="engine:executor-a",
        request_id="queued-cancel",
    )
    ref = record["operation"]
    cancel = service.hosted_operation_cancel(ref=ref, reason="workspace_unload", respawn=False)
    release.set()
    worker.join(2)

    assert cancel["lifecycle"] == "terminal_cancellation"
    assert results[0]["lifecycle"] == "terminal_cancellation"
    assert canceled == ["queued-cancel"]
    assert ipc_calls == []


@pytest.mark.parametrize("execution_kind", ["workflow_python", "workflow_js"])
def test_workflow_service_recreation_replays_and_recovers_without_worker_startup(
    tmp_path: Path, execution_kind: str
) -> None:
    def prepare(service: EngineHostService, request_id: str) -> dict:
        return service._hosted_operations.prepare(
            owner_actor_id="service:local",
            execution_kind=execution_kind,
            selector={"kind": "engine_id", "id": f"{execution_kind}-runtime"},
            namespace=f"{execution_kind}:runtime",
            request_id=request_id,
            fingerprint=hosted_execution_fingerprint({"execution_kind": execution_kind, "request_id": request_id}),
            metadata={"engine_id": f"{execution_kind}-runtime", "environment_key": f"{execution_kind}-environment"},
        )

    def recreate(root: Path, service: EngineHostService) -> EngineHostService:
        repository_key = str(service._hosted_operations.path.resolve())
        EngineHostService._operation_repositories.pop(repository_key, None)
        return _service(root)

    terminal_root = tmp_path / "terminal"
    terminal_service = _service(terminal_root)
    terminal = prepare(terminal_service, "terminal")
    terminal_id = terminal["status"]["operation"]["operation_id"]
    terminal_service._hosted_operations.mark_dispatch_claimed(operation_id=terminal_id)
    terminal_service._hosted_operations.finish(
        operation_id=terminal_id,
        lifecycle="terminal_success",
        envelope={"status": "ok", "answer": execution_kind},
    )
    terminal_recreated = recreate(terminal_root, terminal_service)
    replay = terminal_recreated.hosted_operation_status(ref=terminal["status"]["operation"])
    assert replay["lifecycle"] == "terminal_success"
    assert replay["result"]["answer"] == execution_kind

    before_root = tmp_path / "before"
    before_service = _service(before_root)
    before = prepare(before_service, "before")
    before_recreated = recreate(before_root, before_service)
    interrupted_before = before_recreated.hosted_operation_status(ref=before["status"]["operation"])
    assert interrupted_before["lifecycle"] == "interrupted_before_dispatch"
    canceled = before_recreated.hosted_operation_cancel(ref=before["status"]["operation"])
    assert canceled["lifecycle"] == "terminal_cancellation"

    after_root = tmp_path / "after"
    after_service = _service(after_root)
    after = prepare(after_service, "after")
    after_service._hosted_operations.mark_dispatch_claimed(
        operation_id=after["status"]["operation"]["operation_id"]
    )
    after_recreated = recreate(after_root, after_service)
    interrupted_after = after_recreated.hosted_operation_status(ref=after["status"]["operation"])
    assert interrupted_after["lifecycle"] == "interrupted_after_dispatch_unknown"

    assert terminal_recreated.discover_running(prune_stale=False, include_reachability=False) == []
    assert before_recreated.discover_running(prune_stale=False, include_reachability=False) == []
    assert after_recreated.discover_running(prune_stale=False, include_reachability=False) == []


def test_daemon_restart_smoke_reads_terminal_operation_without_starting_worker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("hosting.daemon.security._tighten_windows_acl", lambda *_args, **_kwargs: None)
    pid_file = tmp_path / "daemon.pid"
    control_file = tmp_path / "access_control.json"
    engine_file = tmp_path / "managed_engines.json"

    first_daemon = EngineHostDaemon(
        pid_file=pid_file,
        engines_state_file=engine_file,
        control_state_file=control_file,
    )
    first_daemon._execute_startup_worker_recovery = lambda: {"status": "ok"}  # type: ignore[method-assign]
    actor_id = first_daemon.svc._actor_id_from_payload(first_daemon.svc._read_control(), {})
    prepared = first_daemon.svc._hosted_operations.prepare(
        owner_actor_id=actor_id,
        execution_kind="toolbox",
        selector={"kind": "engine_id", "id": "receipt-smoke"},
        namespace="engine:receipt-smoke",
        request_id="restart-1",
        fingerprint=hosted_execution_fingerprint({"tool_name": "smoke", "arguments": {}, "policy": {}}),
        metadata={"engine_id": "receipt-smoke", "tool_name": "smoke"},
    )
    operation_id = prepared["status"]["operation"]["operation_id"]
    first_daemon.svc._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
    first_daemon.svc._hosted_operations.finish(
        operation_id=operation_id,
        lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
        envelope={"status": "ok", "request_id": "restart-1", "result": "persisted"},
    )
    ref = prepared["status"]["operation"]

    def _run(daemon: EngineHostDaemon) -> threading.Thread:
        thread = threading.Thread(target=lambda: asyncio.run(daemon.run()), daemon=True)
        thread.start()
        channel = EngineHostControlChannel(
            {"engine_host_daemon_pid_file": str(pid_file), "engine_host_daemon_auto_bootstrap": False}
        )
        for _ in range(200):
            if pid_file.exists():
                try:
                    if channel.discover_running() is not None:
                        return thread
                except Exception:
                    pass
            time.sleep(0.01)
        raise AssertionError("daemon did not become ready")

    channel = EngineHostControlChannel(
        {"engine_host_daemon_pid_file": str(pid_file), "engine_host_daemon_auto_bootstrap": False}
    )
    first_thread = _run(first_daemon)
    assert channel.hosted_operation_status(ref=ref)["lifecycle"] == "terminal_success"
    channel.stop_daemon(reason="restart_smoke", requested_by="test")
    first_thread.join(5)
    assert not first_thread.is_alive()

    repository_key = str((tmp_path / "state" / "hosted_operations.json").resolve())
    EngineHostService._operation_repositories.pop(repository_key, None)
    second_daemon = EngineHostDaemon(
        pid_file=pid_file,
        engines_state_file=engine_file,
        control_state_file=control_file,
    )
    second_daemon._execute_startup_worker_recovery = lambda: {"status": "ok"}  # type: ignore[method-assign]
    second_thread = _run(second_daemon)
    try:
        status = channel.hosted_operation_status(ref=ref)
        assert status["lifecycle"] == "terminal_success"
        assert status["result"]["result"] == "persisted"
        assert second_daemon.svc.discover_running(prune_stale=False, include_reachability=False) == []
    finally:
        channel.stop_daemon(reason="restart_smoke_complete", requested_by="test")
        second_thread.join(5)
