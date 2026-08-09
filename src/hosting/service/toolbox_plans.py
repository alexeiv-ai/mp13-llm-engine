"""Strict process-safe repository for bounded expiring toolbox definition plans."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..toolbox.bundle_models import ResolvedToolboxProfileSpec, ToolboxDefinitionSpec
from ..toolbox.definition_planner import (
    ActiveToolboxProfileSnapshot,
    ToolboxDefinitionPlanDraft,
    classify_toolbox_profiles,
)
from ..toolbox.identity import identity_digest, require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


TOOLBOX_PLAN_STATE_CONTRACT = "hosting.toolbox.definition_plan_state.v1"
TOOLBOX_PLAN_CONTRACT = "hosting.toolbox.definition_plan.v1"
TOOLBOX_PLAN_ID_DOMAIN = "hosting.toolbox.definition_plan_id.v1"
MAX_TOOLBOX_PLANS = 256
MAX_TOOLBOX_PLAN_BYTES = 4 * 1024 * 1024
MAX_TOOLBOX_PLAN_TTL_MS = 15 * 60 * 1000


@dataclass(frozen=True)
class PersistedToolboxDefinitionPlan:
    plan_id: str
    toolbox_id: str
    definition_revision: str
    expected_revision: str | None
    catalog_revision: str
    package_policy_revision: str
    created_at_ms: int
    expires_at_ms: int
    plan: Mapping[str, Any]
    profile_changes: tuple[Mapping[str, Any], ...]
    contract: str = TOOLBOX_PLAN_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != TOOLBOX_PLAN_CONTRACT:
            raise ValueError("toolbox_plan_contract_invalid")
        require_digest(self.plan_id, label="toolbox_plan_id")
        require_digest(self.definition_revision, label="toolbox_plan_definition_revision")
        if self.expected_revision is not None:
            require_digest(self.expected_revision, label="toolbox_plan_expected_revision")
        require_digest(self.catalog_revision, label="toolbox_plan_catalog_revision")
        require_digest(self.package_policy_revision, label="toolbox_plan_package_policy_revision")
        if not str(self.toolbox_id or "").strip():
            raise ValueError("toolbox_plan_toolbox_id_required")
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.created_at_ms < 0
            or self.expires_at_ms <= self.created_at_ms
            or self.expires_at_ms - self.created_at_ms > MAX_TOOLBOX_PLAN_TTL_MS
        ):
            raise ValueError("toolbox_plan_lifetime_invalid")
        plan = dict(self.plan or {})
        fields = {"definition", "definition_revision", "profiles", "bundles", "custom_environment_count"}
        if set(plan) != fields:
            raise ValueError("toolbox_plan_payload_fields_invalid")
        definition = ToolboxDefinitionSpec.from_dict(plan["definition"])
        if (
            definition.toolbox_id != self.toolbox_id
            or definition.revision != self.definition_revision
            or plan["definition_revision"] != self.definition_revision
            or definition.expected_revision != self.expected_revision
        ):
            raise ValueError("toolbox_plan_definition_mismatch")
        if not isinstance(plan["profiles"], list) or not isinstance(plan["bundles"], list):
            raise ValueError("toolbox_plan_profiles_invalid")
        profiles = [ResolvedToolboxProfileSpec.from_dict(item) for item in plan["profiles"]]
        if len(profiles) != len(plan["bundles"]):
            raise ValueError("toolbox_plan_bundle_count_mismatch")
        if isinstance(plan["custom_environment_count"], bool) or plan["custom_environment_count"] != sum(
            item.custom_resolved_lock_digest is not None for item in profiles
        ):
            raise ValueError("toolbox_plan_custom_count_invalid")
        changes = tuple(dict(item) for item in self.profile_changes)
        allowed = {"reused", "added", "replaced", "removed"}
        for item in changes:
            if set(item) != {"classification", "active_profile_id", "proposed_profile_id", "changed_fields"}:
                raise ValueError("toolbox_plan_profile_change_fields_invalid")
            if item["classification"] not in allowed or not isinstance(item["changed_fields"], list):
                raise ValueError("toolbox_plan_profile_change_invalid")
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_TOOLBOX_PLAN_BYTES:
            raise ValueError("toolbox_plan_too_large")
        object.__setattr__(self, "toolbox_id", str(self.toolbox_id).strip())
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "profile_changes", changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "plan_id": self.plan_id,
            "toolbox_id": self.toolbox_id,
            "definition_revision": self.definition_revision,
            "expected_revision": self.expected_revision,
            "catalog_revision": self.catalog_revision,
            "package_policy_revision": self.package_policy_revision,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "plan": dict(self.plan),
            "profile_changes": [dict(item) for item in self.profile_changes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersistedToolboxDefinitionPlan":
        row = dict(payload or {})
        fields = {
            "contract", "plan_id", "toolbox_id", "definition_revision",
            "expected_revision", "catalog_revision", "package_policy_revision",
            "created_at_ms", "expires_at_ms", "plan", "profile_changes",
        }
        if set(row) != fields:
            raise ValueError("toolbox_plan_fields_invalid")
        return cls(**{**row, "profile_changes": tuple(row["profile_changes"])})


class AtomicJsonToolboxDefinitionPlanRepository:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": TOOLBOX_PLAN_STATE_CONTRACT, "plans": {}}

    @classmethod
    def _validate_state(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {"contract", "plans"} or row.get("contract") != TOOLBOX_PLAN_STATE_CONTRACT:
            raise ValueError("toolbox_plan_state_contract_invalid")
        if not isinstance(row["plans"], dict) or len(row["plans"]) > MAX_TOOLBOX_PLANS:
            raise ValueError("toolbox_plan_state_capacity_invalid")
        plans: dict[str, dict[str, Any]] = {}
        for key, value in row["plans"].items():
            plan = PersistedToolboxDefinitionPlan.from_dict(value)
            if key != plan.plan_id:
                raise ValueError("toolbox_plan_state_key_invalid")
            plans[key] = plan.to_dict()
        return {"contract": TOOLBOX_PLAN_STATE_CONTRACT, "plans": plans}

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("toolbox_plan_state_corrupt") from exc
        if not isinstance(payload, dict):
            raise ValueError("toolbox_plan_state_corrupt")
        return self._validate_state(payload)

    def _write_unlocked(self, state: Mapping[str, Any]) -> None:
        value = self._validate_state(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(raw)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_bounded_retries(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _prune(state: dict[str, Any], *, now_ms: int) -> bool:
        expired = [key for key, row in state["plans"].items() if row["expires_at_ms"] <= now_ms]
        for key in expired:
            state["plans"].pop(key)
        return bool(expired)

    def create(
        self,
        draft: ToolboxDefinitionPlanDraft,
        *,
        active_profiles: Sequence[ActiveToolboxProfileSnapshot | Mapping[str, Any]],
        catalog_revision: str,
        package_policy_revision: str,
        now_ms: int,
        ttl_ms: int,
    ) -> PersistedToolboxDefinitionPlan:
        if not isinstance(draft, ToolboxDefinitionPlanDraft):
            raise ValueError("toolbox_plan_draft_required")
        if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms < 1 or ttl_ms > MAX_TOOLBOX_PLAN_TTL_MS:
            raise ValueError("toolbox_plan_ttl_invalid")
        catalog = require_digest(catalog_revision, label="toolbox_plan_catalog_revision")
        policy = require_digest(package_policy_revision, label="toolbox_plan_package_policy_revision")
        plan_payload = draft.to_dict()
        identity_payload = {
            "toolbox_id": draft.definition.toolbox_id,
            "definition_revision": draft.definition.revision,
            "expected_revision": draft.definition.expected_revision,
            "catalog_revision": catalog,
            "package_policy_revision": policy,
            "profiles": plan_payload["profiles"],
            "bundles": [
                {
                    "bundle_id": item["bundle_id"],
                    "manifest_hash": item["manifest_hash"],
                    "dependency_lock_hash": item["dependency_lock_hash"],
                }
                for item in plan_payload["bundles"]
            ],
        }
        record = PersistedToolboxDefinitionPlan(
            plan_id=identity_digest(TOOLBOX_PLAN_ID_DOMAIN, identity_payload),
            toolbox_id=draft.definition.toolbox_id,
            definition_revision=draft.definition.revision,
            expected_revision=draft.definition.expected_revision,
            catalog_revision=catalog,
            package_policy_revision=policy,
            created_at_ms=now_ms,
            expires_at_ms=now_ms + ttl_ms,
            plan=plan_payload,
            profile_changes=classify_toolbox_profiles(draft, active_profiles),
        )
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            changed = self._prune(state, now_ms=now_ms)
            existing = state["plans"].get(record.plan_id)
            if existing is not None:
                existing_record = PersistedToolboxDefinitionPlan.from_dict(existing)
                if (
                    existing_record.plan != record.plan
                    or existing_record.catalog_revision != record.catalog_revision
                    or existing_record.package_policy_revision != record.package_policy_revision
                    or existing_record.profile_changes != record.profile_changes
                ):
                    raise ValueError("toolbox_plan_id_conflict")
                if changed:
                    self._write_unlocked(state)
                return existing_record
            if len(state["plans"]) >= MAX_TOOLBOX_PLANS:
                raise ValueError("toolbox_plan_capacity")
            state["plans"][record.plan_id] = record.to_dict()
            self._write_unlocked(state)
        return record

    def get(self, plan_id: str, *, now_ms: int) -> PersistedToolboxDefinitionPlan:
        key = require_digest(plan_id, label="toolbox_plan_id")
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            existing = state["plans"].get(key)
            if existing is not None and existing["expires_at_ms"] <= now_ms:
                state["plans"].pop(key)
                self._write_unlocked(state)
                raise ValueError("toolbox_definition_plan_expired")
            changed = self._prune(state, now_ms=now_ms)
            if changed:
                self._write_unlocked(state)
            if existing is None:
                raise ValueError("toolbox_definition_plan_not_found")
            return PersistedToolboxDefinitionPlan.from_dict(existing)

    def list(self, *, now_ms: int) -> tuple[PersistedToolboxDefinitionPlan, ...]:
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            if self._prune(state, now_ms=now_ms):
                self._write_unlocked(state)
            plans = [PersistedToolboxDefinitionPlan.from_dict(item) for item in state["plans"].values()]
        return tuple(sorted(plans, key=lambda item: (item.created_at_ms, item.plan_id)))


__all__ = [
    "AtomicJsonToolboxDefinitionPlanRepository",
    "MAX_TOOLBOX_PLAN_BYTES",
    "MAX_TOOLBOX_PLAN_TTL_MS",
    "MAX_TOOLBOX_PLANS",
    "PersistedToolboxDefinitionPlan",
]
