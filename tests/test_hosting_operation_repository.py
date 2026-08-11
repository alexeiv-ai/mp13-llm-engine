from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from hosting.operation_contract import (
    HostedOperationLifecycle,
    HostedOperationProgress,
    hosted_execution_fingerprint,
)
from hosting.service.operation_repository import (
    AtomicJsonHostedOperationRepository,
    LegacyOperationRepositoryError,
    OPERATION_REPOSITORY_CONTRACT,
)


def _repository(path: Path, **kwargs) -> AtomicJsonHostedOperationRepository:
    return AtomicJsonHostedOperationRepository(path, **kwargs)


def _prepare(repository: AtomicJsonHostedOperationRepository, request_id: str = "request-1", **kwargs):
    return repository.prepare(
        owner_actor_id=kwargs.pop("owner_actor_id", "actor:a"),
        execution_kind=kwargs.pop("execution_kind", "toolbox"),
        selector=kwargs.pop("selector", {"kind": "toolbox_id", "id": "demo"}),
        namespace=kwargs.pop("namespace", "toolbox:demo"),
        request_id=request_id,
        fingerprint=kwargs.pop("fingerprint", hosted_execution_fingerprint({"request_id": request_id})),
        metadata=kwargs.pop("metadata", {"tool_name": "write"}),
        **kwargs,
    )


def _prepare_from_process(path: str, barrier: Any, results: Any) -> None:
    try:
        repository = _repository(Path(path))
        barrier.wait(timeout=30)
        results.put(("ok", _prepare(repository, request_id="shared")["action"]))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def test_prepare_mints_stable_ref_and_enforces_one_dispatch(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json")
    first = _prepare(repository)
    duplicate = _prepare(repository)
    ref = first["status"]["operation"]

    assert first["action"] == "dispatch"
    assert duplicate["action"] == "attach"
    assert duplicate["status"]["operation"] == ref
    assert ref["operation_id"].startswith("op_")
    assert repository.get_by_request(
        owner_actor_id="actor:a", namespace="toolbox:demo", request_id="request-1"
    )["operation"] == ref
    assert repository.get_by_operation_id(ref["operation_id"])["operation"] == ref


def test_concurrent_prepare_grants_exactly_one_dispatch(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json")
    barrier = threading.Barrier(12)
    actions: list[str] = []

    def prepare() -> None:
        barrier.wait()
        actions.append(_prepare(repository)["action"])

    threads = [threading.Thread(target=prepare) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert actions.count("dispatch") == 1
    assert actions.count("attach") == 11


def test_concurrent_multi_process_prepare_preserves_single_idempotent_receipt(tmp_path: Path) -> None:
    path = tmp_path / "operations.json"
    context = multiprocessing.get_context("spawn")
    worker_count = 6
    barrier = context.Barrier(worker_count)
    results = context.Queue()
    processes = [
        context.Process(target=_prepare_from_process, args=(str(path), barrier, results))
        for _ in range(worker_count)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
        observed = [results.get(timeout=5) for _ in range(worker_count)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
        results.close()
        results.join_thread()

    assert all(kind == "ok" for kind, _ in observed), observed
    actions = [value for _, value in observed]
    assert actions.count("dispatch") == 1
    assert actions.count("attach") == worker_count - 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["receipts"]) == 1


def test_interrupted_atomic_replace_preserves_last_valid_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    repository = _repository(path)
    first = _prepare(repository, request_id="preserved")
    before = path.read_bytes()

    def fail_replace(_source, _target) -> None:
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr("hosting.service.operation_repository.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated interrupted replace"):
        _prepare(repository, request_id="not-persisted")

    assert path.read_bytes() == before
    temporary_files = list(path.parent.glob(f".{path.name}.{os.getpid()}.*.tmp"))
    assert temporary_files == []
    monkeypatch.undo()
    recreated = _repository(path)
    assert recreated.get_by_operation_id(first["status"]["operation"]["operation_id"]) is not None
    assert recreated.get_by_request(
        owner_actor_id="actor:a", namespace="toolbox:demo", request_id="not-persisted"
    ) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific replace retry behavior")
def test_windows_replace_retries_are_bounded_and_eventually_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path / "operations.json")
    real_replace = os.replace
    attempts = {"count": 0}

    def flaky_replace(source, target) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("simulated sharing violation")
        real_replace(source, target)

    monkeypatch.setattr("hosting.service.operation_repository.sys.platform", "win32")
    monkeypatch.setattr("hosting.service.operation_repository.os.replace", flaky_replace)
    monkeypatch.setattr("hosting.service.operation_repository.time.sleep", lambda _seconds: None)

    prepared = _prepare(repository)

    assert prepared["action"] == "dispatch"
    assert attempts["count"] == 3


def test_conflict_never_receives_dispatch_and_preserves_existing_ref(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json")
    first = _prepare(repository)
    conflict = _prepare(repository, fingerprint=hosted_execution_fingerprint({"changed": True}))

    assert conflict["action"] == "conflict"
    assert conflict["status"]["api_status"] == "error"
    assert conflict["status"]["lifecycle"] == "idempotency_conflict"
    assert conflict["status"]["operation"] == first["status"]["operation"]


def test_owner_is_part_of_request_identity_and_ref_authorization(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json")
    actor_a = _prepare(repository, owner_actor_id="actor:a")
    actor_b = _prepare(repository, owner_actor_id="actor:b")

    assert actor_a["status"]["operation"]["operation_id"] != actor_b["status"]["operation"]["operation_id"]
    hidden = repository.status(ref=actor_a["status"]["operation"], owner_actor_id="actor:b")
    assert hidden["api_status"] == "error"
    assert hidden["lifecycle"] == "unknown_outside_retention"
    assert hidden["reason"] == "operation_not_found"


def test_altered_ref_is_not_resolved_or_used_for_routing(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json")
    prepared = _prepare(repository)
    altered = dict(prepared["status"]["operation"])
    altered["selector"] = {"kind": "toolbox_id", "id": "other"}

    assert repository.resolve(ref=altered, owner_actor_id="actor:a") is None
    assert repository.status(ref=altered, owner_actor_id="actor:a")["lifecycle"] == "unknown_outside_retention"


def test_dispatch_terminal_replay_and_redaction_survive_recreation(tmp_path: Path) -> None:
    path = tmp_path / "operations.json"
    repository = _repository(path)
    prepared = _prepare(repository, metadata={"access_token": "metadata-secret"})
    operation_id = prepared["status"]["operation"]["operation_id"]
    running = repository.mark_dispatch_claimed(operation_id=operation_id)
    terminal = repository.finish(
        operation_id=operation_id,
        lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
        envelope={"status": "ok", "authorization": "Bearer secret", "answer": 7},
    )

    assert running["lifecycle"] == "running"
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["authorization"] == "[REDACTED]"
    raw = path.read_text(encoding="utf-8")
    assert "metadata-secret" not in raw
    assert "Bearer secret" not in raw

    recreated = _repository(path)
    replay = _prepare(recreated)
    assert replay["action"] == "replay"
    assert replay["status"]["result"]["answer"] == 7


def test_large_terminal_result_is_an_explicit_digest_only_omission(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json", max_inline_result_bytes=512)
    prepared = _prepare(repository)
    operation_id = prepared["status"]["operation"]["operation_id"]
    repository.mark_dispatch_claimed(operation_id=operation_id)
    terminal = repository.finish(
        operation_id=operation_id,
        lifecycle="terminal_success",
        envelope={"status": "ok", "result": "x" * 10_000},
    )

    assert terminal["result"] is None
    assert terminal["result_ref"] is None
    assert terminal["result_omission"]["reason"] == "retention_not_permitted"
    assert terminal["result_omission"]["digest"].startswith("sha256:")
    assert terminal["result_omission"]["size_bytes"] > 512


def test_restart_recovers_before_dispatch_once_and_fails_closed_after_dispatch(tmp_path: Path) -> None:
    before_path = tmp_path / "before.json"
    first = _repository(before_path)
    before = _prepare(first)
    recovered = _repository(before_path)
    status = recovered.status(ref=before["status"]["operation"], owner_actor_id="actor:a")
    assert status["lifecycle"] == "interrupted_before_dispatch"
    assert _prepare(recovered)["action"] == "dispatch"
    assert _prepare(recovered)["action"] == "attach"

    after_path = tmp_path / "after.json"
    second = _repository(after_path)
    after = _prepare(second)
    second.mark_dispatch_claimed(operation_id=after["status"]["operation"]["operation_id"])
    failed_closed = _repository(after_path)
    status = failed_closed.status(ref=after["status"]["operation"], owner_actor_id="actor:a")
    assert status["lifecycle"] == "interrupted_after_dispatch_unknown"
    assert _prepare(failed_closed)["action"] == "attach"


@pytest.mark.parametrize(
    "execution_kind",
    ["toolbox", "toolbox_definition_apply", "workflow_python", "workflow_js"],
)
def test_execution_families_share_every_lifecycle_shape(tmp_path: Path, execution_kind: str) -> None:
    selector = {
        "kind": "toolbox_id" if execution_kind in {"toolbox", "toolbox_definition_apply"} else "engine_id",
        "id": f"{execution_kind}-target",
    }
    namespace = f"{execution_kind}:target"
    expected_fields = {
        "contract", "api_status", "operation", "lifecycle", "request_id",
        "created_at_ms", "updated_at_ms", "dispatch_claimed_at_ms", "terminal_at_ms",
        "reason", "result", "result_ref", "result_omission", "progress",
    }

    def prepare(repository, request_id: str, fingerprint_payload=None):
        return _prepare(
            repository,
            request_id=request_id,
            execution_kind=execution_kind,
            selector=selector,
            namespace=namespace,
            fingerprint=hosted_execution_fingerprint(fingerprint_payload or {"request_id": request_id}),
        )

    repository = _repository(tmp_path / f"{execution_kind}.json")
    observed: list[dict] = []
    queued = prepare(repository, "queued")
    observed.append(queued["status"])
    observed.append(repository.mark_dispatch_claimed(operation_id=queued["status"]["operation"]["operation_id"]))

    for lifecycle in ("terminal_success", "terminal_failure", "terminal_cancellation"):
        item = prepare(repository, lifecycle)
        repository.mark_dispatch_claimed(operation_id=item["status"]["operation"]["operation_id"])
        observed.append(
            repository.finish(
                operation_id=item["status"]["operation"]["operation_id"],
                lifecycle=lifecycle,
                envelope={"status": "ok" if lifecycle == "terminal_success" else "error"},
                reason="test_terminal" if lifecycle != "terminal_success" else "",
            )
        )

    conflict = prepare(repository, "terminal_success", {"changed": True})
    observed.append(conflict["status"])

    before_path = tmp_path / f"{execution_kind}-before.json"
    before_repository = _repository(before_path)
    before = prepare(before_repository, "before")
    observed.append(
        _repository(before_path).status(ref=before["status"]["operation"], owner_actor_id="actor:a")
    )

    after_path = tmp_path / f"{execution_kind}-after.json"
    after_repository = _repository(after_path)
    after = prepare(after_repository, "after")
    after_repository.mark_dispatch_claimed(operation_id=after["status"]["operation"]["operation_id"])
    observed.append(
        _repository(after_path).status(ref=after["status"]["operation"], owner_actor_id="actor:a")
    )

    now = [100.0]
    retained = _repository(
        tmp_path / f"{execution_kind}-retention.json",
        receipt_retention_seconds=1,
        tombstone_retention_seconds=1,
        clock=lambda: now[0],
    )
    expiring = prepare(retained, "expiring")
    retained.finish(
        operation_id=expiring["status"]["operation"]["operation_id"],
        lifecycle="terminal_success",
        envelope={"status": "ok"},
    )
    now[0] = 102.0
    retained.prune()
    observed.append(retained.status(ref=expiring["status"]["operation"], owner_actor_id="actor:a"))
    now[0] = 104.0
    retained.prune()
    observed.append(retained.status(ref=expiring["status"]["operation"], owner_actor_id="actor:a"))

    assert {row["lifecycle"] for row in observed} == {
        "queued", "running", "terminal_success", "terminal_failure", "terminal_cancellation",
        "interrupted_before_dispatch", "interrupted_after_dispatch_unknown", "forgotten",
        "unknown_outside_retention", "idempotency_conflict",
    }
    assert all(set(row) == expected_fields for row in observed)


def test_progress_is_persisted_recovered_and_cancellation_boundary_is_monotonic(tmp_path: Path) -> None:
    now = [100.0]
    path = tmp_path / "operations.json"
    repository = _repository(path, clock=lambda: now[0])
    prepared = _prepare(repository, execution_kind="toolbox_definition_apply")
    operation_id = prepared["status"]["operation"]["operation_id"]
    repository.mark_dispatch_claimed(operation_id=operation_id)
    warmup = repository.update_progress(
        operation_id=operation_id,
        progress=HostedOperationProgress(
            phase="warmup",
            code="candidate_warmup",
            completed_units=1,
            total_units=2,
            updated_at_ms=100_000,
            summary="Warming candidate workers.",
            cancellable=True,
        ),
    )
    assert warmup["progress"]["phase"] == "warmup"
    now[0] = 101.0
    published = repository.update_progress(
        operation_id=operation_id,
        progress={
            "phase": "publication",
            "code": "routes_published",
            "completed_units": 2,
            "total_units": 2,
            "updated_at_ms": 101_000,
            "summary": "The active routes were published.",
            "cancellable": False,
        },
    )
    assert published["progress"]["cancellable"] is False
    with pytest.raises(ValueError, match="operation_progress_cancellation_boundary_regression"):
        repository.update_progress(
            operation_id=operation_id,
            progress={**published["progress"], "phase": "cleanup", "cancellable": True},
        )

    recreated = _repository(path, clock=lambda: now[0])
    status = recreated.status(ref=prepared["status"]["operation"], owner_actor_id="actor:a")
    assert status["lifecycle"] == "interrupted_after_dispatch_unknown"
    assert status["progress"] == published["progress"]
    with pytest.raises(ValueError, match="operation_progress_terminal_update_denied"):
        recreated.update_progress(operation_id=operation_id, progress=published["progress"])


def test_progress_rejects_a_future_checkpoint_timestamp(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json", clock=lambda: 100.0)
    prepared = _prepare(repository)
    with pytest.raises(ValueError, match="operation_progress_future_timestamp"):
        repository.update_progress(
            operation_id=prepared["status"]["operation"]["operation_id"],
            progress={
                "phase": "work",
                "code": "work_started",
                "completed_units": None,
                "total_units": None,
                "updated_at_ms": 100_001,
                "summary": "Work started.",
                "cancellable": True,
            },
        )


def test_cancel_before_dispatch_is_atomic_and_replayed(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json")
    prepared = _prepare(repository)
    operation_id = prepared["status"]["operation"]["operation_id"]
    canceled = repository.cancel_before_dispatch(operation_id=operation_id, reason="workspace_unload")

    assert canceled["lifecycle"] == "terminal_cancellation"
    assert canceled["reason"] == "workspace_unload"
    assert repository.cancel_before_dispatch(operation_id=operation_id, reason="again") is None
    assert _prepare(repository)["action"] == "replay"


def test_waiting_duplicate_observes_one_terminal_transition(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "operations.json")
    prepared = _prepare(repository)
    operation_id = prepared["status"]["operation"]["operation_id"]
    repository.mark_dispatch_claimed(operation_id=operation_id)
    results: list[dict] = []

    waiter = threading.Thread(
        target=lambda: results.append(repository.wait_for_terminal(operation_id=operation_id, timeout_seconds=2))
    )
    waiter.start()
    repository.finish(operation_id=operation_id, lifecycle="terminal_success", envelope={"answer": 7})
    waiter.join(2)

    assert not waiter.is_alive()
    assert results[0]["lifecycle"] == "terminal_success"


def test_pruning_retains_bounded_tombstones_then_reports_unknown(tmp_path: Path) -> None:
    now = [100.0]
    repository = _repository(
        tmp_path / "operations.json",
        receipt_retention_seconds=10,
        tombstone_retention_seconds=20,
        max_receipts=2,
        max_tombstones=2,
        clock=lambda: now[0],
    )
    refs = []
    for index in range(3):
        prepared = _prepare(repository, request_id=f"request-{index}")
        refs.append(prepared["status"]["operation"])
        repository.finish(
            operation_id=prepared["status"]["operation"]["operation_id"],
            lifecycle="terminal_success",
            envelope={"index": index},
        )
        now[0] += 1
    forgotten = repository.status(ref=refs[0], owner_actor_id="actor:a")
    assert forgotten["lifecycle"] == "forgotten"
    now[0] = 125.0
    repository.prune()
    assert repository.status(ref=refs[0], owner_actor_id="actor:a")["lifecycle"] == "unknown_outside_retention"


def test_legacy_and_corrupt_checkpoints_fail_closed(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"version": 1, "receipts": {}, "tombstones": {}}), encoding="utf-8")
    with pytest.raises(LegacyOperationRepositoryError, match="hosting-receipt-ledger-cutover"):
        _repository(legacy)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="repository is unreadable"):
        _repository(corrupt)


def test_cutover_requires_acknowledgement_and_archives_legacy_file(tmp_path: Path) -> None:
    legacy = tmp_path / "toolbox_execution_receipts.json"
    legacy.write_text(json.dumps({"version": 1, "receipts": {}, "tombstones": {}}), encoding="utf-8")
    with pytest.raises(PermissionError, match="cutover_acknowledgement_required"):
        AtomicJsonHostedOperationRepository.archive_legacy_checkpoint(
            legacy, acknowledge_replay_window_clear=False
        )
    archived = AtomicJsonHostedOperationRepository.archive_legacy_checkpoint(
        legacy, acknowledge_replay_window_clear=True, clock=lambda: 123.0
    )
    assert archived.name.endswith(".legacy-123000.archive")
    assert archived.exists()
    assert not legacy.exists()


def test_persisted_indexes_are_rebuilt_and_duplicate_identity_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "operations.json"
    repository = _repository(path)
    first = _prepare(repository, request_id="a")
    second = _prepare(repository, request_id="b")
    payload = json.loads(path.read_text(encoding="utf-8"))
    second_id = second["status"]["operation"]["operation_id"]
    payload["receipts"][second_id]["operation"]["request_id"] = "a"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate request identity"):
        _repository(path)
    assert payload["contract"] == OPERATION_REPOSITORY_CONTRACT
