from __future__ import annotations

import json
from pathlib import Path

import pytest

from hosting.service.toolbox_candidates import AtomicJsonToolboxDefinitionCandidateRepository
from hosting.toolbox.bundle_models import ToolboxPlanPins


DIGESTS = ["sha256:" + character * 64 for character in "abcdef"]


def _pins() -> ToolboxPlanPins:
    return ToolboxPlanPins(
        active_definition_revision=None,
        target="cp312-win_amd64",
        configuration_revision=DIGESTS[0],
        catalog_revision=DIGESTS[1],
        host_config_revision=DIGESTS[2],
        dependency_policy_revision=DIGESTS[3],
        source_set_revision=DIGESTS[4],
    )


def _create(repository: AtomicJsonToolboxDefinitionCandidateRepository, **overrides):
    values = {
        "plan_id": DIGESTS[0],
        "confirmation_ref": "confirmation_one",
        "toolbox_id": "demo",
        "definition_revision": DIGESTS[5],
        "changed_tool_keys": ["alpha"],
        "pins": _pins(),
        "owner_actor_id": "actor:a",
        "authority_id": "authority:a",
        "request_id": "prepare-1",
        "requested_lifetime_ms": None,
        "retained_payload": {"candidate_engine_ids": ["engine-1"]},
        "now_ms": 1_000,
    }
    values.update(overrides)
    return repository.create(**values)


def test_candidate_record_is_bounded_actor_scoped_and_restart_durable(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    repository = AtomicJsonToolboxDefinitionCandidateRepository(
        path, retention_ms=600_000, limit_per_actor=2
    )
    candidate_ref, record = _create(repository)
    assert set(record.public_projection(candidate_ref)) == {
        "contract", "candidate_ref", "toolbox_id", "definition_revision",
        "changed_tool_keys", "created_at_ms", "expires_at_ms", "state", "user_projection",
    }
    assert "engine-1" not in json.dumps(record.public_projection(candidate_ref))
    restarted = AtomicJsonToolboxDefinitionCandidateRepository(
        path, retention_ms=600_000, limit_per_actor=2
    )
    assert restarted.get(
        candidate_ref, owner_actor_id="actor:a", authority_id="authority:a", now_ms=2_000
    ) == record
    with pytest.raises(PermissionError, match="candidate_not_found"):
        restarted.get(
            candidate_ref, owner_actor_id="actor:b", authority_id="authority:a", now_ms=2_000
        )


def test_prepare_retry_and_actor_quota_are_atomic(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionCandidateRepository(
        tmp_path / "candidates.json", retention_ms=600_000, limit_per_actor=1
    )
    first_ref, first = _create(repository)
    retry_ref, retry = _create(repository, now_ms=5_000)
    assert (retry_ref, retry) == (first_ref, first)
    with pytest.raises(ValueError, match="toolbox_candidate_idempotency_conflict"):
        _create(repository, retained_payload={"candidate_engine_ids": ["different"]})
    with pytest.raises(ValueError, match="toolbox_candidate_limit_exceeded"):
        _create(repository, request_id="prepare-2")
    second_ref, _ = _create(repository, request_id="prepare-2", now_ms=700_000)
    assert second_ref != first_ref


def test_renewal_is_repeatable_idempotent_and_policy_bounded(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionCandidateRepository(
        tmp_path / "candidates.json", retention_ms=900_000, limit_per_actor=2
    )
    candidate_ref, _ = _create(repository, requested_lifetime_ms=300_000)
    renewed = repository.renew(
        candidate_ref,
        owner_actor_id="actor:a",
        authority_id="authority:a",
        request_id="renew-1",
        requested_lifetime_ms=600_000,
        now_ms=10_000,
    )
    assert renewed.expires_at_ms == 610_000
    assert repository.renew(
        candidate_ref,
        owner_actor_id="actor:a",
        authority_id="authority:a",
        request_id="renew-1",
        requested_lifetime_ms=600_000,
        now_ms=20_000,
    ).expires_at_ms == 610_000
    repeated = repository.renew(
        candidate_ref,
        owner_actor_id="actor:a",
        authority_id="authority:a",
        request_id="renew-2",
        requested_lifetime_ms=300_000,
        now_ms=400_000,
    )
    assert repeated.expires_at_ms == 700_000
    with pytest.raises(ValueError, match="candidate_renewal_denied"):
        repository.renew(
            candidate_ref,
            owner_actor_id="actor:a",
            authority_id="authority:a",
            request_id="renew-3",
            requested_lifetime_ms=1_000_000,
            now_ms=40_000,
        )


def test_execution_lease_defers_expiry_until_terminal_release(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionCandidateRepository(
        tmp_path / "candidates.json", retention_ms=300_000, limit_per_actor=1
    )
    candidate_ref, _ = _create(repository)
    leased = repository.acquire_execution_lease(
        candidate_ref,
        owner_actor_id="actor:a",
        authority_id="authority:a",
        execution_request_id="execute-1",
        now_ms=2_000,
    )
    assert leased.state == "ready" and leased.execution_leases
    still_ready = repository.get(
        candidate_ref, owner_actor_id="actor:a", authority_id="authority:a", now_ms=400_000
    )
    assert still_ready.state == "ready"
    with pytest.raises(ValueError, match="candidate_execution_denied"):
        repository.acquire_execution_lease(
            candidate_ref,
            owner_actor_id="actor:a",
            authority_id="authority:a",
            execution_request_id="execute-2",
            now_ms=400_000,
        )
    expired = repository.release_execution_lease(
        candidate_ref, execution_request_id="execute-1", now_ms=400_000
    )
    assert expired.state == "expired" and not expired.execution_leases


def test_publish_and_discard_are_terminal_idempotent_transitions(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionCandidateRepository(
        tmp_path / "candidates.json", retention_ms=300_000, limit_per_actor=2
    )
    candidate_ref, _ = _create(repository)
    published = repository.transition(
        candidate_ref,
        owner_actor_id="actor:a",
        authority_id="authority:a",
        state_name="published",
        now_ms=2_000,
    )
    assert published.state == "published"
    assert repository.transition(
        candidate_ref,
        owner_actor_id="actor:a",
        authority_id="authority:a",
        state_name="published",
        now_ms=3_000,
    ) == published
    retry_ref, retry = _create(repository, now_ms=4_000)
    assert retry_ref == candidate_ref and retry.state == "published"
    with pytest.raises(ValueError, match="candidate_stale"):
        repository.transition(
            candidate_ref,
            owner_actor_id="actor:a",
            authority_id="authority:a",
            state_name="discarded",
            now_ms=4_000,
        )
