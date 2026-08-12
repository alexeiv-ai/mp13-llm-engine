"""Durable actor-bound records for warmed toolbox definition candidates."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..toolbox.bundle_models import ToolboxPlanPins
from ..toolbox.identity import identity_digest, require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


CANDIDATE_STATE_CONTRACT = "hosting.toolbox.definition_candidate_state.v1"
CANDIDATE_RECORD_CONTRACT = "hosting.toolbox.definition_candidate_record.v1"
CANDIDATE_PUBLIC_CONTRACT = "hosting.toolbox.definition_candidate.v1"
MIN_CANDIDATE_LIFETIME_MS = 300_000
MAX_CANDIDATE_LIFETIME_MS = 14_400_000
MAX_CANDIDATE_RECORDS = 256
MAX_CANDIDATE_STATE_BYTES = 16 * 1024 * 1024
MAX_CANDIDATE_RENEWALS = 128
MAX_CANDIDATE_LEASES = 128


def _ref_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _request_digest(value: str, *, label: str) -> str:
    request_id = str(value or "").strip()
    if not request_id or len(request_id) > 512:
        raise ValueError(f"{label}_invalid")
    return _ref_digest(request_id)


def _lifetime(value: int, *, maximum: int, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < MIN_CANDIDATE_LIFETIME_MS
        or value > MAX_CANDIDATE_LIFETIME_MS
        or value > maximum
    ):
        raise ValueError(label)
    return value


@dataclass(frozen=True)
class ToolboxDefinitionCandidateRecord:
    candidate_ref_digest: str
    prepare_request_digest: str
    plan_id: str
    confirmation_ref_digest: str
    toolbox_id: str
    definition_revision: str
    changed_tool_keys: tuple[str, ...]
    pins: ToolboxPlanPins
    owner_actor_id: str
    authority_id: str
    created_at_ms: int
    expires_at_ms: int
    state: str
    retained_payload: Mapping[str, Any]
    renewal_requests: Mapping[str, Mapping[str, int]]
    execution_leases: Mapping[str, int]
    contract: str = CANDIDATE_RECORD_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != CANDIDATE_RECORD_CONTRACT:
            raise ValueError("toolbox_candidate_contract_invalid")
        for label, value in (
            ("ref", self.candidate_ref_digest),
            ("prepare_request", self.prepare_request_digest),
            ("plan", self.plan_id),
            ("confirmation", self.confirmation_ref_digest),
            ("definition", self.definition_revision),
        ):
            require_digest(value, label=f"toolbox_candidate_{label}")
        if not isinstance(self.pins, ToolboxPlanPins):
            raise ValueError("toolbox_candidate_pins_invalid")
        if not all(str(value or "").strip() for value in (self.toolbox_id, self.owner_actor_id, self.authority_id)):
            raise ValueError("toolbox_candidate_owner_invalid")
        keys = tuple(sorted(str(item or "").strip() for item in self.changed_tool_keys))
        if not keys or any(not item for item in keys) or len(set(keys)) != len(keys):
            raise ValueError("toolbox_candidate_changed_tools_invalid")
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.created_at_ms < 0
            or self.expires_at_ms <= self.created_at_ms
        ):
            raise ValueError("toolbox_candidate_lifetime_invalid")
        if self.state not in {"ready", "published", "discarded", "expired"}:
            raise ValueError("toolbox_candidate_state_invalid")
        payload = copy.deepcopy(dict(self.retained_payload or {}))
        renewals = copy.deepcopy(dict(self.renewal_requests or {}))
        leases = dict(self.execution_leases or {})
        if len(renewals) > MAX_CANDIDATE_RENEWALS or len(leases) > MAX_CANDIDATE_LEASES:
            raise ValueError("toolbox_candidate_history_capacity")
        for key, raw in renewals.items():
            require_digest(key, label="toolbox_candidate_renewal_request")
            row = dict(raw or {})
            if set(row) != {"requested_lifetime_ms", "expires_at_ms"} or any(
                isinstance(row[name], bool) or not isinstance(row[name], int)
                for name in row
            ):
                raise ValueError("toolbox_candidate_renewal_invalid")
        for key, acquired_at_ms in leases.items():
            require_digest(key, label="toolbox_candidate_execution_request")
            if isinstance(acquired_at_ms, bool) or not isinstance(acquired_at_ms, int) or acquired_at_ms < 0:
                raise ValueError("toolbox_candidate_lease_invalid")
        object.__setattr__(self, "toolbox_id", str(self.toolbox_id).strip())
        object.__setattr__(self, "owner_actor_id", str(self.owner_actor_id).strip())
        object.__setattr__(self, "authority_id", str(self.authority_id).strip())
        object.__setattr__(self, "changed_tool_keys", keys)
        object.__setattr__(self, "retained_payload", payload)
        object.__setattr__(self, "renewal_requests", renewals)
        object.__setattr__(self, "execution_leases", leases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "candidate_ref_digest": self.candidate_ref_digest,
            "prepare_request_digest": self.prepare_request_digest,
            "plan_id": self.plan_id,
            "confirmation_ref_digest": self.confirmation_ref_digest,
            "toolbox_id": self.toolbox_id,
            "definition_revision": self.definition_revision,
            "changed_tool_keys": list(self.changed_tool_keys),
            "pins": self.pins.to_dict(),
            "owner_actor_id": self.owner_actor_id,
            "authority_id": self.authority_id,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "state": self.state,
            "retained_payload": copy.deepcopy(dict(self.retained_payload)),
            "renewal_requests": copy.deepcopy(dict(self.renewal_requests)),
            "execution_leases": dict(self.execution_leases),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxDefinitionCandidateRecord":
        row = dict(payload or {})
        fields = {
            "contract", "candidate_ref_digest", "prepare_request_digest", "plan_id",
            "confirmation_ref_digest", "toolbox_id", "definition_revision",
            "changed_tool_keys", "pins", "owner_actor_id", "authority_id",
            "created_at_ms", "expires_at_ms", "state", "retained_payload",
            "renewal_requests", "execution_leases",
        }
        if set(row) != fields:
            raise ValueError("toolbox_candidate_fields_invalid")
        return cls(
            **{
                **row,
                "changed_tool_keys": tuple(row["changed_tool_keys"]),
                "pins": ToolboxPlanPins.from_dict(row["pins"]),
            }
        )

    def public_projection(self, candidate_ref: str) -> dict[str, Any]:
        if _ref_digest(candidate_ref) != self.candidate_ref_digest:
            raise ValueError("toolbox_candidate_ref_mismatch")
        state_code = {
            "ready": "candidate_ready",
            "published": "candidate_published",
            "discarded": "candidate_discarded",
            "expired": "candidate_expired",
        }[self.state]
        return {
            "contract": CANDIDATE_PUBLIC_CONTRACT,
            "candidate_ref": candidate_ref,
            "toolbox_id": self.toolbox_id,
            "definition_revision": self.definition_revision,
            "changed_tool_keys": list(self.changed_tool_keys),
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "state": self.state,
            "user_projection": {"code": state_code, "summary": state_code.replace("_", " ").capitalize() + "."},
        }


class AtomicJsonToolboxDefinitionCandidateRepository:
    def __init__(self, path: Path, *, retention_ms: int, limit_per_actor: int):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.retention_ms = _lifetime(
            retention_ms, maximum=MAX_CANDIDATE_LIFETIME_MS, label="toolbox_candidate_retention_invalid"
        )
        if isinstance(limit_per_actor, bool) or not isinstance(limit_per_actor, int) or not 1 <= limit_per_actor <= 16:
            raise ValueError("toolbox_candidate_limit_invalid")
        self.limit_per_actor = limit_per_actor

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": CANDIDATE_STATE_CONTRACT, "candidates": {}}

    @classmethod
    def _validate(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {"contract", "candidates"} or row.get("contract") != CANDIDATE_STATE_CONTRACT:
            raise ValueError("toolbox_candidate_state_contract_invalid")
        if not isinstance(row["candidates"], dict) or len(row["candidates"]) > MAX_CANDIDATE_RECORDS:
            raise ValueError("toolbox_candidate_state_capacity_invalid")
        candidates: dict[str, Any] = {}
        for key, raw in row["candidates"].items():
            record = ToolboxDefinitionCandidateRecord.from_dict(raw)
            if key != record.candidate_ref_digest:
                raise ValueError("toolbox_candidate_state_key_invalid")
            candidates[key] = record.to_dict()
        result = {"contract": CANDIDATE_STATE_CONTRACT, "candidates": candidates}
        if len(json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")) > MAX_CANDIDATE_STATE_BYTES:
            raise ValueError("toolbox_candidate_state_too_large")
        return result

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("toolbox_candidate_state_corrupt") from exc
        return self._validate(payload)

    def _write(self, payload: Mapping[str, Any]) -> None:
        state = self._validate(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(raw)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_bounded_retries(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _candidate_ref(*, plan_id: str, owner_actor_id: str, authority_id: str, request_digest: str) -> str:
        digest = identity_digest(
            "hosting.toolbox.definition_candidate_ref.v1",
            {
                "plan_id": plan_id,
                "owner_actor_id": owner_actor_id,
                "authority_id": authority_id,
                "prepare_request_digest": request_digest,
            },
        )
        return "candidate_" + digest.removeprefix("sha256:")

    @staticmethod
    def _owned(record: ToolboxDefinitionCandidateRecord, *, owner_actor_id: str, authority_id: str) -> None:
        if record.owner_actor_id != str(owner_actor_id or "").strip() or record.authority_id != str(authority_id or "").strip():
            raise PermissionError("candidate_not_found")

    @staticmethod
    def _expire(record: ToolboxDefinitionCandidateRecord, *, now_ms: int) -> ToolboxDefinitionCandidateRecord:
        if record.state == "ready" and record.expires_at_ms <= now_ms and not record.execution_leases:
            return ToolboxDefinitionCandidateRecord.from_dict({**record.to_dict(), "state": "expired"})
        return record

    def create(
        self,
        *,
        plan_id: str,
        confirmation_ref: str,
        toolbox_id: str,
        definition_revision: str,
        changed_tool_keys: Sequence[str],
        pins: ToolboxPlanPins,
        owner_actor_id: str,
        authority_id: str,
        request_id: str,
        requested_lifetime_ms: int | None,
        retained_payload: Mapping[str, Any],
        now_ms: int,
    ) -> tuple[str, ToolboxDefinitionCandidateRecord]:
        lifetime = self.retention_ms if requested_lifetime_ms is None else _lifetime(
            requested_lifetime_ms,
            maximum=self.retention_ms,
            label="candidate_renewal_denied",
        )
        owner = str(owner_actor_id or "").strip()
        authority = str(authority_id or "").strip()
        request_digest = _request_digest(request_id, label="toolbox_candidate_prepare_request")
        candidate_ref = self._candidate_ref(
            plan_id=require_digest(plan_id, label="toolbox_candidate_plan"),
            owner_actor_id=owner,
            authority_id=authority,
            request_digest=request_digest,
        )
        record = ToolboxDefinitionCandidateRecord(
            candidate_ref_digest=_ref_digest(candidate_ref),
            prepare_request_digest=request_digest,
            plan_id=plan_id,
            confirmation_ref_digest=_ref_digest(confirmation_ref),
            toolbox_id=toolbox_id,
            definition_revision=definition_revision,
            changed_tool_keys=tuple(changed_tool_keys),
            pins=pins,
            owner_actor_id=owner,
            authority_id=authority,
            created_at_ms=now_ms,
            expires_at_ms=now_ms + lifetime,
            state="ready",
            retained_payload=retained_payload,
            renewal_requests={},
            execution_leases={},
        )
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            existing = state["candidates"].get(record.candidate_ref_digest)
            if existing is not None:
                prior = ToolboxDefinitionCandidateRecord.from_dict(existing)
                immutable_fields = {
                    "candidate_ref_digest", "prepare_request_digest", "plan_id",
                    "confirmation_ref_digest", "toolbox_id", "definition_revision",
                    "changed_tool_keys", "pins", "owner_actor_id", "authority_id",
                    "retained_payload",
                }
                prior_payload = prior.to_dict()
                new_payload = record.to_dict()
                if any(prior_payload[field] != new_payload[field] for field in immutable_fields):
                    raise ValueError("toolbox_candidate_idempotency_conflict")
                return candidate_ref, prior
            live_for_actor = 0
            for key, raw in list(state["candidates"].items()):
                current = self._expire(ToolboxDefinitionCandidateRecord.from_dict(raw), now_ms=now_ms)
                state["candidates"][key] = current.to_dict()
                if current.owner_actor_id == owner and current.state == "ready":
                    live_for_actor += 1
            if live_for_actor >= self.limit_per_actor:
                raise ValueError("toolbox_candidate_limit_exceeded")
            if len(state["candidates"]) >= MAX_CANDIDATE_RECORDS:
                raise ValueError("toolbox_candidate_capacity")
            state["candidates"][record.candidate_ref_digest] = record.to_dict()
            self._write(state)
        return candidate_ref, record

    def get(
        self, candidate_ref: str, *, owner_actor_id: str, authority_id: str, now_ms: int
    ) -> ToolboxDefinitionCandidateRecord:
        key = _ref_digest(candidate_ref)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            raw = state["candidates"].get(key)
            if raw is None:
                raise PermissionError("candidate_not_found")
            record = ToolboxDefinitionCandidateRecord.from_dict(raw)
            self._owned(record, owner_actor_id=owner_actor_id, authority_id=authority_id)
            current = self._expire(record, now_ms=now_ms)
            if current != record:
                state["candidates"][key] = current.to_dict()
                self._write(state)
            return current

    def renew(
        self,
        candidate_ref: str,
        *,
        owner_actor_id: str,
        authority_id: str,
        request_id: str,
        requested_lifetime_ms: int,
        now_ms: int,
    ) -> ToolboxDefinitionCandidateRecord:
        lifetime = _lifetime(requested_lifetime_ms, maximum=self.retention_ms, label="candidate_renewal_denied")
        request_digest = _request_digest(request_id, label="toolbox_candidate_renew_request")
        key = _ref_digest(candidate_ref)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            raw = state["candidates"].get(key)
            if raw is None:
                raise PermissionError("candidate_not_found")
            record = ToolboxDefinitionCandidateRecord.from_dict(raw)
            self._owned(record, owner_actor_id=owner_actor_id, authority_id=authority_id)
            record = self._expire(record, now_ms=now_ms)
            prior = dict(record.renewal_requests).get(request_digest)
            if prior is not None:
                if prior["requested_lifetime_ms"] != lifetime:
                    raise ValueError("toolbox_candidate_idempotency_conflict")
                return record
            if record.state != "ready" or record.expires_at_ms <= now_ms:
                if record != ToolboxDefinitionCandidateRecord.from_dict(raw):
                    state["candidates"][key] = record.to_dict()
                    self._write(state)
                raise ValueError("candidate_renewal_denied")
            if len(record.renewal_requests) >= MAX_CANDIDATE_RENEWALS:
                raise ValueError("toolbox_candidate_history_capacity")
            expires_at_ms = max(record.expires_at_ms, now_ms + lifetime)
            renewals = {**dict(record.renewal_requests), request_digest: {
                "requested_lifetime_ms": lifetime, "expires_at_ms": expires_at_ms,
            }}
            renewed = ToolboxDefinitionCandidateRecord.from_dict({
                **record.to_dict(), "expires_at_ms": expires_at_ms, "renewal_requests": renewals,
            })
            state["candidates"][key] = renewed.to_dict()
            self._write(state)
            return renewed

    def acquire_execution_lease(
        self,
        candidate_ref: str,
        *,
        owner_actor_id: str,
        authority_id: str,
        execution_request_id: str,
        now_ms: int,
    ) -> ToolboxDefinitionCandidateRecord:
        request_digest = _request_digest(execution_request_id, label="toolbox_candidate_execution_request")
        key = _ref_digest(candidate_ref)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            raw = state["candidates"].get(key)
            if raw is None:
                raise PermissionError("candidate_not_found")
            record = ToolboxDefinitionCandidateRecord.from_dict(raw)
            self._owned(record, owner_actor_id=owner_actor_id, authority_id=authority_id)
            if request_digest in record.execution_leases:
                return record
            if record.state != "ready" or record.expires_at_ms <= now_ms:
                expired = self._expire(record, now_ms=now_ms)
                if expired != record:
                    state["candidates"][key] = expired.to_dict()
                    self._write(state)
                raise ValueError("candidate_execution_denied")
            if len(record.execution_leases) >= MAX_CANDIDATE_LEASES:
                raise ValueError("toolbox_candidate_history_capacity")
            leased = ToolboxDefinitionCandidateRecord.from_dict({
                **record.to_dict(),
                "execution_leases": {**dict(record.execution_leases), request_digest: now_ms},
            })
            state["candidates"][key] = leased.to_dict()
            self._write(state)
            return leased

    def release_execution_lease(
        self, candidate_ref: str, *, execution_request_id: str, now_ms: int
    ) -> ToolboxDefinitionCandidateRecord:
        request_digest = _request_digest(execution_request_id, label="toolbox_candidate_execution_request")
        key = _ref_digest(candidate_ref)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            raw = state["candidates"].get(key)
            if raw is None:
                raise PermissionError("candidate_not_found")
            record = ToolboxDefinitionCandidateRecord.from_dict(raw)
            leases = dict(record.execution_leases)
            leases.pop(request_digest, None)
            released = ToolboxDefinitionCandidateRecord.from_dict({**record.to_dict(), "execution_leases": leases})
            released = self._expire(released, now_ms=now_ms)
            if released != record:
                state["candidates"][key] = released.to_dict()
                self._write(state)
            return released

    def transition(
        self,
        candidate_ref: str,
        *,
        owner_actor_id: str,
        authority_id: str,
        state_name: str,
        now_ms: int,
    ) -> ToolboxDefinitionCandidateRecord:
        if state_name not in {"published", "discarded"}:
            raise ValueError("toolbox_candidate_transition_invalid")
        key = _ref_digest(candidate_ref)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            raw = state["candidates"].get(key)
            if raw is None:
                raise PermissionError("candidate_not_found")
            record = ToolboxDefinitionCandidateRecord.from_dict(raw)
            self._owned(record, owner_actor_id=owner_actor_id, authority_id=authority_id)
            record = self._expire(record, now_ms=now_ms)
            if record.state == state_name:
                return record
            if record.state != "ready" or record.expires_at_ms <= now_ms:
                if record.to_dict() != raw:
                    state["candidates"][key] = record.to_dict()
                    self._write(state)
                raise ValueError("candidate_expired" if record.state == "expired" else "candidate_stale")
            transitioned = ToolboxDefinitionCandidateRecord.from_dict({**record.to_dict(), "state": state_name})
            state["candidates"][key] = transitioned.to_dict()
            self._write(state)
            return transitioned


__all__ = [
    "AtomicJsonToolboxDefinitionCandidateRepository",
    "ToolboxDefinitionCandidateRecord",
]
