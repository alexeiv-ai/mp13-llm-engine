from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from hosting.service.execution_receipts import ToolboxExecutionReceiptLedger, execution_fingerprint
from hosting.service.host_service import EngineHostService
from hosting.daemon import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel


def _ledger(path: Path, **kwargs) -> ToolboxExecutionReceiptLedger:
    return ToolboxExecutionReceiptLedger(path, **kwargs)


def test_same_fingerprint_gets_one_dispatch_permission_in_every_lifecycle(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "receipts.json")
    fingerprint = execution_fingerprint({"tool_name": "write", "arguments": {"x": 1}, "policy": {}})

    first = ledger.prepare(namespace="toolbox:demo", request_id="request-1", fingerprint=fingerprint)
    queued_duplicate = ledger.prepare(namespace="toolbox:demo", request_id="request-1", fingerprint=fingerprint)
    ledger.mark_dispatch_claimed(namespace="toolbox:demo", request_id="request-1")
    running_duplicate = ledger.prepare(namespace="toolbox:demo", request_id="request-1", fingerprint=fingerprint)
    ledger.finish(
        namespace="toolbox:demo",
        request_id="request-1",
        state="terminal_success",
        envelope={"status": "ok", "request_id": "request-1", "result": 7},
    )
    terminal_duplicate = ledger.prepare(namespace="toolbox:demo", request_id="request-1", fingerprint=fingerprint)

    assert first["action"] == "dispatch"
    assert queued_duplicate["action"] == "attach"
    assert running_duplicate["action"] == "attach"
    assert terminal_duplicate["action"] == "replay"
    assert terminal_duplicate["receipt"]["terminal_envelope"]["result"] == 7


def test_fingerprint_conflict_never_gets_dispatch_permission(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "receipts.json")
    first = execution_fingerprint({"tool_name": "write", "arguments": {"x": 1}, "policy": {}})
    changed = execution_fingerprint({"tool_name": "write", "arguments": {"x": 2}, "policy": {}})
    assert ledger.prepare(namespace="engine:a", request_id="same", fingerprint=first)["action"] == "dispatch"
    assert ledger.prepare(namespace="engine:a", request_id="same", fingerprint=changed)["action"] == "conflict"


def test_restart_recovers_pre_dispatch_once_and_fails_closed_after_dispatch(tmp_path: Path) -> None:
    path = tmp_path / "receipts.json"
    fingerprint = execution_fingerprint({"tool_name": "write", "arguments": {}, "policy": {}})
    _ledger(path).prepare(namespace="engine:a", request_id="before", fingerprint=fingerprint)

    recovered = _ledger(path)
    assert recovered.status(namespace="engine:a", request_id="before")["state"] == "interrupted_before_dispatch"
    assert recovered.prepare(namespace="engine:a", request_id="before", fingerprint=fingerprint)["action"] == "dispatch"
    assert recovered.prepare(namespace="engine:a", request_id="before", fingerprint=fingerprint)["action"] == "attach"
    recovered.mark_dispatch_claimed(namespace="engine:a", request_id="before")

    failed_closed = _ledger(path)
    assert failed_closed.status(namespace="engine:a", request_id="before")["state"] == "interrupted_after_dispatch_unknown"
    assert failed_closed.prepare(namespace="engine:a", request_id="before", fingerprint=fingerprint)["action"] == "attach"


def test_queued_cancellation_and_terminal_results_survive_recreation(tmp_path: Path) -> None:
    path = tmp_path / "receipts.json"
    fingerprint = execution_fingerprint({"tool_name": "write", "arguments": {}, "policy": {}})
    ledger = _ledger(path)
    ledger.prepare(namespace="toolbox:demo", request_id="cancel-me", fingerprint=fingerprint)
    ledger.cancel_before_dispatch(
        namespace="toolbox:demo",
        request_id="cancel-me",
        envelope={"status": "ok", "outcome": "canceled", "request_id": "cancel-me"},
    )
    for request_id, state, status_value in (
        ("succeeded", "terminal_success", "ok"),
        ("failed", "terminal_failure", "error"),
    ):
        ledger.prepare(namespace="toolbox:demo", request_id=request_id, fingerprint=fingerprint)
        ledger.mark_dispatch_claimed(namespace="toolbox:demo", request_id=request_id)
        ledger.finish(
            namespace="toolbox:demo",
            request_id=request_id,
            state=state,
            envelope={"status": status_value, "request_id": request_id},
        )

    recreated = _ledger(path)
    status = recreated.status(namespace="toolbox:demo", request_id="cancel-me")
    assert status["state"] == "terminal_cancellation"
    assert recreated.prepare(namespace="toolbox:demo", request_id="cancel-me", fingerprint=fingerprint)["action"] == "replay"
    assert recreated.status(namespace="toolbox:demo", request_id="succeeded")["state"] == "terminal_success"
    assert recreated.status(namespace="toolbox:demo", request_id="failed")["state"] == "terminal_failure"


def test_compaction_retains_bounded_tombstones_then_reports_unknown(tmp_path: Path) -> None:
    now = [100.0]
    path = tmp_path / "receipts.json"
    ledger = _ledger(
        path,
        receipt_retention_seconds=10,
        tombstone_retention_seconds=20,
        max_receipts=2,
        max_tombstones=2,
        clock=lambda: now[0],
    )
    for index in range(3):
        request_id = f"request-{index}"
        fingerprint = execution_fingerprint({"index": index})
        ledger.prepare(namespace="engine:a", request_id=request_id, fingerprint=fingerprint)
        ledger.finish(
            namespace="engine:a",
            request_id=request_id,
            state="terminal_success",
            envelope={"status": "ok", "request_id": request_id},
        )
        now[0] += 1
    assert ledger.status(namespace="engine:a", request_id="request-0")["state"] == "forgotten"
    now[0] = 125.0
    ledger.compact()
    assert ledger.status(namespace="engine:a", request_id="request-0")["state"] == "unknown_outside_retention"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert len(persisted["tombstones"]) <= 2


def test_persisted_envelopes_redact_credentials_and_bound_large_results(tmp_path: Path) -> None:
    path = tmp_path / "receipts.json"
    ledger = _ledger(path, max_result_bytes=512)
    fingerprint = execution_fingerprint({"arguments": {"password": "digest-only"}})
    ledger.prepare(
        namespace="engine:a",
        request_id="secret-result",
        fingerprint=fingerprint,
        metadata={"access_token": "must-not-persist", "tool_name": "read"},
    )
    ledger.finish(
        namespace="engine:a",
        request_id="secret-result",
        state="terminal_success",
        envelope={"status": "ok", "authorization": "Bearer secret", "result": "x" * 10_000},
    )

    raw = path.read_text(encoding="utf-8")
    assert "must-not-persist" not in raw
    assert "Bearer secret" not in raw
    status = ledger.status(namespace="engine:a", request_id="secret-result")
    reference = status["terminal_envelope"]["result_reference"]
    assert reference["kind"] == "omitted_oversize_terminal_envelope"
    assert reference["size_bytes"] > 512


def test_waiting_duplicate_observes_one_terminal_dispatch(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "receipts.json")
    fingerprint = execution_fingerprint({"tool_name": "slow"})
    dispatch_count = 0
    started = threading.Event()
    release = threading.Event()
    results: list[dict] = []

    def execute() -> None:
        nonlocal dispatch_count
        prepared = ledger.prepare(namespace="engine:a", request_id="slow-1", fingerprint=fingerprint)
        if prepared["action"] == "dispatch":
            dispatch_count += 1
            ledger.mark_dispatch_claimed(namespace="engine:a", request_id="slow-1")
            started.set()
            release.wait(2)
            ledger.finish(
                namespace="engine:a",
                request_id="slow-1",
                state="terminal_success",
                envelope={"status": "ok", "request_id": "slow-1"},
            )
        else:
            results.append(ledger.wait_for_terminal(namespace="engine:a", request_id="slow-1", timeout_seconds=2))

    first = threading.Thread(target=execute)
    second = threading.Thread(target=execute)
    first.start()
    assert started.wait(2)
    second.start()
    release.set()
    first.join(2)
    second.join(2)
    assert dispatch_count == 1
    assert results[0]["state"] == "terminal_success"


def test_service_duplicate_replays_terminal_envelope_without_second_ipc_dispatch(tmp_path: Path) -> None:
    service = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    service.register_spawned(
        engine_id="executor-a",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        bundle={"toolbox_id": "demo", "sandbox_profile_id": "default"},
        environment={"environment_key": "demo-env"},
        tool_access={"allowed_tool_names": ["write"]},
    )
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
    assert first["status"] == duplicate["status"] == "ok"
    assert duplicate["idempotent_replay"] is True
    assert conflict["outcome"] == "idempotency_conflict"

    recreated = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    status = recreated.toolbox_request_status(engine_id="executor-a", request_id="request-1")
    assert status["lifecycle_state"] == "terminal_success"
    assert status["receipt"]["terminal_envelope"]["tool_call"]["result"] == "done"


def test_service_queued_cancel_race_never_invokes_tool_and_cleans_pool_request(tmp_path: Path) -> None:
    service = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    service.register_spawned(
        engine_id="executor-a",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        bundle={"toolbox_id": "demo", "sandbox_profile_id": "default"},
        environment={"environment_key": "demo-env"},
        tool_access={"allowed_tool_names": ["write"]},
    )
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
    cancel = service.toolbox_cancel(engine_id="executor-a", request_id="queued-cancel", respawn=False)
    release.set()
    worker.join(2)

    assert cancel["outcome"] == "canceled"
    assert results[0]["outcome"] == "canceled"
    assert canceled == ["queued-cancel"]
    assert ipc_calls == []


def test_daemon_restart_smoke_reads_terminal_receipt_without_starting_worker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("hosting.daemon.security._tighten_windows_acl", lambda *_args, **_kwargs: None)
    pid_file = tmp_path / "daemon.pid"
    control_file = tmp_path / "access_control.json"
    engine_file = tmp_path / "managed_engines.json"
    fingerprint = execution_fingerprint({"tool_name": "smoke", "arguments": {}, "policy": {}})

    first_daemon = EngineHostDaemon(
        pid_file=pid_file,
        engines_state_file=engine_file,
        control_state_file=control_file,
    )
    first_daemon._execute_startup_worker_recovery = lambda: {"status": "ok"}  # type: ignore[method-assign]
    first_daemon.svc._toolbox_execution_receipts.prepare(
        namespace="engine:receipt-smoke",
        request_id="restart-1",
        fingerprint=fingerprint,
    )
    first_daemon.svc._toolbox_execution_receipts.mark_dispatch_claimed(
        namespace="engine:receipt-smoke",
        request_id="restart-1",
    )
    first_daemon.svc._toolbox_execution_receipts.finish(
        namespace="engine:receipt-smoke",
        request_id="restart-1",
        state="terminal_success",
        envelope={"status": "ok", "request_id": "restart-1", "result": "persisted"},
    )

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
    assert channel.toolbox_request_status(engine_id="receipt-smoke", request_id="restart-1")["lifecycle_state"] == "terminal_success"
    channel.stop_daemon(reason="restart_smoke", requested_by="test")
    first_thread.join(5)
    assert not first_thread.is_alive()

    ledger_key = str((tmp_path / "state" / "toolbox_execution_receipts.json").resolve())
    EngineHostService._receipt_ledgers.pop(ledger_key, None)
    second_daemon = EngineHostDaemon(
        pid_file=pid_file,
        engines_state_file=engine_file,
        control_state_file=control_file,
    )
    second_daemon._execute_startup_worker_recovery = lambda: {"status": "ok"}  # type: ignore[method-assign]
    second_thread = _run(second_daemon)
    try:
        status = channel.toolbox_request_status(engine_id="receipt-smoke", request_id="restart-1")
        assert status["lifecycle_state"] == "terminal_success"
        assert status["receipt"]["terminal_envelope"]["result"] == "persisted"
        assert second_daemon.svc.discover_running(prune_stale=False, include_reachability=False) == []
    finally:
        channel.stop_daemon(reason="restart_smoke_complete", requested_by="test")
        second_thread.join(5)
