from __future__ import annotations

import time
import threading
from pathlib import Path

import pytest

from hosting.operation_contract import hosted_execution_fingerprint
from hosting.service.host_service import EngineHostService


def _service(tmp_path: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )


def _prepare(service: EngineHostService, *, request_id: str = "apply-1") -> dict:
    return service._hosted_operations.prepare(
        owner_actor_id="actor:a",
        execution_kind="toolbox_definition_apply",
        selector={"kind": "toolbox_id", "id": "demo"},
        namespace="toolbox-definition:demo",
        request_id=request_id,
        fingerprint=hosted_execution_fingerprint({"plan_id": "plan-1", "request_id": request_id}),
        metadata={"toolbox_id": "demo", "plan_id": "plan-1"},
    )


def _progress(phase: str, *, cancellable: bool) -> dict:
    return {
        "phase": phase,
        "code": f"apply_{phase}",
        "completed_units": 0,
        "total_units": 1,
        "updated_at_ms": int(time.time() * 1000),
        "summary": f"Apply phase: {phase}.",
        "cancellable": cancellable,
    }


def test_definition_apply_request_recovery_uses_exact_toolbox_namespace(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prepared = _prepare(service, request_id="lost-response")

    recovered = service.hosted_operation_resolve_request(
        execution_kind="toolbox_definition_apply",
        selector={"kind": "toolbox_id", "id": "demo"},
        request_id="lost-response",
        owner_actor_id="actor:a",
    )

    assert recovered == prepared["status"]
    with pytest.raises(ValueError, match="toolbox_definition_apply_selector_must_be_toolbox_id"):
        service.hosted_operation_resolve_request(
            execution_kind="toolbox_definition_apply",
            selector={"kind": "engine_id", "id": "executor-1"},
            request_id="lost-response",
            owner_actor_id="actor:a",
        )


def test_definition_apply_cancellation_cleans_candidates_and_is_durable(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prepared = _prepare(service)
    operation_id = prepared["status"]["operation"]["operation_id"]
    service._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
    service._hosted_operations.update_progress(
        operation_id=operation_id,
        progress=_progress("warmup", cancellable=True),
    )
    cleanups: list[str] = []
    service._cleanup_toolbox_definition_apply_candidates = (  # type: ignore[attr-defined]
        lambda *, record: cleanups.append(record["operation"]["operation_id"])
        or {"status": "complete", "candidate_count": 2}
    )

    canceled = service.hosted_operation_cancel(
        ref=prepared["status"]["operation"],
        reason="client_requested",
        owner_actor_id="actor:a",
    )

    assert canceled["lifecycle"] == "terminal_cancellation"
    assert canceled["reason"] == "client_requested"
    assert canceled["result"]["code"] == "apply_canceled_before_publication"
    assert canceled["result"]["diagnostics"]["candidate_cleanup"] == {
        "status": "complete",
        "candidate_count": 2,
    }
    assert cleanups == [operation_id]
    assert service.hosted_operation_status(
        ref=prepared["status"]["operation"], owner_actor_id="actor:a"
    ) == canceled


def test_definition_apply_cancellation_is_denied_once_publication_is_persisted(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prepared = _prepare(service)
    operation_id = prepared["status"]["operation"]["operation_id"]
    service._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
    service._hosted_operations.update_progress(
        operation_id=operation_id,
        progress=_progress("publication", cancellable=False),
    )
    cleanups: list[str] = []
    service._cleanup_toolbox_definition_apply_candidates = (  # type: ignore[attr-defined]
        lambda *, record: cleanups.append(record["operation"]["operation_id"]) or {}
    )

    denied = service.hosted_operation_cancel(
        ref=prepared["status"]["operation"],
        owner_actor_id="actor:a",
    )

    assert denied["api_status"] == "error"
    assert denied["lifecycle"] == "running"
    assert denied["reason"] == "apply_publication_committed"
    assert denied["progress"]["cancellable"] is False
    assert cleanups == []


def test_definition_apply_cancel_and_publication_checkpoint_are_atomic(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prepared = _prepare(service)
    operation_id = prepared["status"]["operation"]["operation_id"]
    service._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
    service._hosted_operations.update_progress(
        operation_id=operation_id,
        progress=_progress("warmup", cancellable=True),
    )
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()

    def cleanup(*, record: dict) -> dict:
        assert record["operation"]["operation_id"] == operation_id
        cleanup_started.set()
        assert release_cleanup.wait(2)
        return {"status": "complete", "candidate_count": 1}

    service._cleanup_toolbox_definition_apply_candidates = cleanup  # type: ignore[attr-defined]
    cancellations: list[dict] = []
    publication_errors: list[str] = []
    cancel_thread = threading.Thread(
        target=lambda: cancellations.append(
            service.hosted_operation_cancel(
                ref=prepared["status"]["operation"], owner_actor_id="actor:a"
            )
        )
    )
    cancel_thread.start()
    assert cleanup_started.wait(2)

    def publish() -> None:
        try:
            service._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=_progress("publication", cancellable=False),
            )
        except ValueError as exc:
            publication_errors.append(str(exc))

    publish_thread = threading.Thread(target=publish)
    publish_thread.start()
    release_cleanup.set()
    cancel_thread.join(2)
    publish_thread.join(2)

    assert cancellations[0]["lifecycle"] == "terminal_cancellation"
    assert publication_errors == ["operation_progress_terminal_update_denied"]
