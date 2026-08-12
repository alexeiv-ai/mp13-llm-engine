from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from hosting.service.toolbox_approvals import (
    AtomicJsonToolboxDependencyApprovalRepository,
    ToolboxDependencyApprovalError,
)
from hosting.service.toolbox_runtime import ToolboxRuntimeMixin


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _mint(repository: AtomicJsonToolboxDependencyApprovalRepository) -> dict:
    return repository.mint(
        owner_actor_id="actor:one",
        authority_id="workspace:one",
        approver_actor_id="approver:one",
        toolbox_id="demo",
        plan_id=_digest("1"),
        confirmation_ref_digest=_digest("2"),
        effective_definition_revision=_digest("3"),
        exact_resolution_digest=_digest("4"),
        plan_pins_digest=_digest("5"),
        now_ms=1_000,
        expires_at_ms=10_000,
    )


def _consume(repository, approval_ref: str, *, request_id: str, resolution=_digest("4")):
    return repository.validate_and_consume(
        approval_ref=approval_ref,
        owner_actor_id="actor:one",
        authority_id="workspace:one",
        toolbox_id="demo",
        plan_id=_digest("1"),
        confirmation_ref_digest=_digest("2"),
        effective_definition_revision=_digest("3"),
        exact_resolution_digest=resolution,
        plan_pins_digest=_digest("5"),
        request_id=request_id,
        now_ms=2_000,
    )


def test_stale_approval_does_not_consume_and_restart_retry_is_single_request_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "approvals.json"
    first = AtomicJsonToolboxDependencyApprovalRepository(path)
    minted = _mint(first)
    approval_ref = minted["approval_ref"]

    with pytest.raises(ToolboxDependencyApprovalError, match="dependency_approval_invalid"):
        _consume(first, approval_ref, request_id="apply-1", resolution=_digest("9"))
    state = json.loads(path.read_text(encoding="utf-8"))
    record = next(iter(state["approvals"].values()))
    assert record["consumed_request_id"] is None

    restarted = AtomicJsonToolboxDependencyApprovalRepository(path)
    consumed = _consume(restarted, approval_ref, request_id="apply-1")
    assert consumed["consumed_request_id"] == "apply-1"
    assert _consume(
        AtomicJsonToolboxDependencyApprovalRepository(path),
        approval_ref,
        request_id="apply-1",
    ) == consumed
    with pytest.raises(ToolboxDependencyApprovalError, match="dependency_approval_invalid"):
        _consume(restarted, approval_ref, request_id="apply-2")


def test_expired_or_wrong_actor_approval_never_mutates_consumption(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    repository = AtomicJsonToolboxDependencyApprovalRepository(path)
    approval_ref = _mint(repository)["approval_ref"]
    kwargs = {
        "approval_ref": approval_ref,
        "owner_actor_id": "actor:other",
        "authority_id": "workspace:one",
        "toolbox_id": "demo",
        "plan_id": _digest("1"),
        "confirmation_ref_digest": _digest("2"),
        "effective_definition_revision": _digest("3"),
        "exact_resolution_digest": _digest("4"),
        "plan_pins_digest": _digest("5"),
        "request_id": "apply-1",
        "now_ms": 20_000,
    }
    with pytest.raises(ToolboxDependencyApprovalError, match="dependency_approval_invalid"):
        repository.validate_and_consume(**kwargs)
    record = next(iter(json.loads(path.read_text(encoding="utf-8"))["approvals"].values()))
    assert record["consumed_request_id"] is None


def test_source_bytes_mutated_after_plan_fail_before_confirmation(tmp_path: Path) -> None:
    artifact = tmp_path / "planned.whl"
    original = b"approved-wheel-bytes"
    artifact.write_bytes(original)
    digest = "sha256:" + hashlib.sha256(original).hexdigest()
    ToolboxRuntimeMixin._verify_toolbox_planned_artifact(
        path=artifact, expected_digest=digest, expected_size=len(original)
    )

    artifact.write_bytes(b"mutated-wheel-bytes!")  # same length, different identity
    assert artifact.stat().st_size == len(original)
    with pytest.raises(ValueError, match="toolbox_confirmation_artifact_changed"):
        ToolboxRuntimeMixin._verify_toolbox_planned_artifact(
            path=artifact, expected_digest=digest, expected_size=len(original)
        )
