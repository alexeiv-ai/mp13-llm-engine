from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from hosting.operation_contract import HostedOperationLifecycle, hosted_execution_fingerprint
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
