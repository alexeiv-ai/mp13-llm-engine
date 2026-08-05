from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from hosting.operation_contract import HostedExecutionKind, HostedOperationSelector
from hosting.service.operation_repository import AtomicJsonHostedOperationRepository
from hosting.service.result_artifacts import ResultArtifactError, TerminalResultArtifactStore


def _large_result() -> dict:
    return {"status": "ok", "payload": "x" * 2048, "access_token": "do-not-store"}


def _fingerprint(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _terminal_repository(tmp_path: Path, *, retain: bool = True, clock=None):
    store = TerminalResultArtifactStore(
        tmp_path / "results",
        max_bytes=4096,
        ttl_seconds=10,
        clock=clock or (lambda: 100.0),
    )
    repository = AtomicJsonHostedOperationRepository(
        tmp_path / "operations.json",
        max_inline_result_bytes=256,
        result_artifact_store=store,
        clock=clock or (lambda: 100.0),
    )
    prepared = repository.prepare(
        owner_actor_id="actor:a",
        execution_kind=HostedExecutionKind.WORKFLOW_PYTHON,
        selector=HostedOperationSelector(kind="engine_id", id="wf-py"),
        namespace="workflow_python:wf-py",
        request_id="request-1",
        fingerprint=_fingerprint(b"request-1"),
        metadata={"retain_terminal_result": retain},
    )
    ref = prepared["status"]["operation"]
    repository.mark_dispatch_claimed(operation_id=ref["operation_id"])
    status = repository.finish(
        operation_id=ref["operation_id"],
        lifecycle="terminal_success",
        envelope=_large_result(),
    )
    return repository, store, ref, status


def test_retained_large_result_is_redacted_retrievable_and_multi_read(tmp_path: Path) -> None:
    repository, _store, ref, status = _terminal_repository(tmp_path)
    assert status["result"] is None
    assert status["result_omission"] is None
    assert status["result_ref"]["contract"] == "hosting.result_ref"
    first = repository.read_result(ref=ref, owner_actor_id="actor:a")
    second = repository.read_result(ref=ref, owner_actor_id="actor:a")
    assert first == second
    assert first["content"]["access_token"] == "[REDACTED]"
    assert first["result_ref"]["size_bytes"] > 256


def test_denied_and_oversized_retention_emit_explicit_omissions(tmp_path: Path) -> None:
    _repository, _store, _ref, denied = _terminal_repository(tmp_path / "denied", retain=False)
    assert denied["result_ref"] is None
    assert denied["result_omission"]["reason"] == "retention_not_permitted"

    store = TerminalResultArtifactStore(tmp_path / "small" / "results", max_bytes=300)
    repository = AtomicJsonHostedOperationRepository(
        tmp_path / "small" / "operations.json",
        max_inline_result_bytes=256,
        result_artifact_store=store,
    )
    prepared = repository.prepare(
        owner_actor_id="actor:a",
        execution_kind="workflow_js",
        selector={"kind": "engine_id", "id": "wf-js"},
        namespace="workflow_js:wf-js",
        request_id="request-2",
        fingerprint=_fingerprint(b"request-2"),
        metadata={"retain_terminal_result": True},
    )
    status = repository.finish(
        operation_id=prepared["status"]["operation"]["operation_id"],
        lifecycle="terminal_success",
        envelope=_large_result(),
    )
    assert status["result_omission"]["reason"] == "result_artifact_too_large"


def test_result_read_denies_other_actor_and_detects_tampering(tmp_path: Path) -> None:
    repository, store, ref, status = _terminal_repository(tmp_path)
    with pytest.raises(ResultArtifactError, match="unauthorized"):
        repository.read_result(ref=ref, owner_actor_id="actor:b")
    data_path, _ = store._paths(status["result_ref"]["artifact_id"])
    data_path.write_bytes(b"{}")
    with pytest.raises(ResultArtifactError, match="size_mismatch|digest_mismatch"):
        repository.read_result(ref=ref, owner_actor_id="actor:a")


def test_result_read_detects_metadata_mismatch_and_missing_file(tmp_path: Path) -> None:
    repository, store, ref, status = _terminal_repository(tmp_path)
    data_path, meta_path = store._paths(status["result_ref"]["artifact_id"])
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["digest"] = "sha256:" + "0" * 64
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ResultArtifactError, match="metadata_mismatch"):
        repository.read_result(ref=ref, owner_actor_id="actor:a")
    meta_path.unlink()
    data_path.unlink()
    with pytest.raises(ResultArtifactError, match="missing"):
        repository.read_result(ref=ref, owner_actor_id="actor:a")


def test_expiry_and_receipt_pruning_delete_result_artifact(tmp_path: Path) -> None:
    now = [100.0]
    repository, store, ref, status = _terminal_repository(tmp_path, clock=lambda: now[0])
    data_path, meta_path = store._paths(status["result_ref"]["artifact_id"])
    now[0] = 111.0
    with pytest.raises(ResultArtifactError, match="expired"):
        repository.read_result(ref=ref, owner_actor_id="actor:a")
    assert not data_path.exists() and not meta_path.exists()

    now[0] = 200.0
    repository, store, _ref, status = _terminal_repository(tmp_path / "prune", clock=lambda: now[0])
    repository.receipt_retention_ms = 0
    artifact_id = status["result_ref"]["artifact_id"]
    repository.prune()
    data_path, meta_path = store._paths(artifact_id)
    assert not data_path.exists() and not meta_path.exists()
