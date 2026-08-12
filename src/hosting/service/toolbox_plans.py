"""Strict process-safe repository for bounded expiring toolbox definition plans."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..environments.contracts import EnvironmentRequest
from ..packages.contracts import PackageLock
from ..toolbox.bundle_models import (
    ResolvedToolboxProfileSpec,
    ToolboxDefinitionSpec,
    ToolboxEnvironmentMutationSpec,
    ToolboxPlanPins,
)
from ..toolbox.definition_planner import (
    ActiveToolboxProfileSnapshot,
    ToolboxDefinitionPlanDraft,
    classify_toolbox_profiles,
)
from ..toolbox.identity import identity_digest, require_digest
from ..toolbox.tool_changes import NormalizedToolboxToolChange
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


TOOLBOX_PLAN_STATE_CONTRACT = "hosting.toolbox.definition_plan_state.v1"
TOOLBOX_PLAN_CONTRACT = "hosting.toolbox.definition_plan.v1"
TOOLBOX_PLAN_ID_DOMAIN = "hosting.toolbox.definition_plan_id.v1"
TOOLBOX_COMPLETE_PLAN_STATE_CONTRACT = "hosting.toolbox.definition_plan_state.v2"
TOOLBOX_COMPLETE_PLAN_CONTRACT = "hosting.toolbox.definition_plan.v2"
TOOLBOX_COMPLETE_PLAN_ID_DOMAIN = "hosting.toolbox.definition_plan_id.v2"
MAX_TOOLBOX_PLANS = 256
MAX_TOOLBOX_PLAN_BYTES = 4 * 1024 * 1024
MAX_TOOLBOX_PLAN_TTL_MS = 15 * 60 * 1000


@dataclass(frozen=True)
class ToolboxPlannedEnvironmentRecord:
    environment_id: str
    alternative_id: str
    package_lock: PackageLock
    environment_request: EnvironmentRequest
    contract: str = "hosting.toolbox.planned_environment.v1"

    def __post_init__(self) -> None:
        if self.contract != "hosting.toolbox.planned_environment.v1":
            raise ValueError("toolbox_planned_environment_contract_invalid")
        object.__setattr__(
            self, "environment_id", require_digest(
                self.environment_id, label="toolbox_planned_environment_id"
            )
        )
        object.__setattr__(
            self, "alternative_id", require_digest(
                self.alternative_id, label="toolbox_planned_alternative_id"
            )
        )
        if not isinstance(self.package_lock, PackageLock):
            raise ValueError("toolbox_planned_package_lock_invalid")
        if not isinstance(self.environment_request, EnvironmentRequest):
            raise ValueError("toolbox_planned_environment_request_invalid")
        if self.environment_request.package_lock_digest != self.package_lock.lock_digest:
            raise ValueError("toolbox_planned_environment_lock_mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "environment_id": self.environment_id,
            "alternative_id": self.alternative_id,
            "package_lock": self.package_lock.to_dict(),
            "environment_request": self.environment_request.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxPlannedEnvironmentRecord":
        row = dict(payload or {})
        if set(row) != {
            "contract", "environment_id", "alternative_id", "package_lock",
            "environment_request",
        }:
            raise ValueError("toolbox_planned_environment_fields_invalid")
        return cls(
            contract=row["contract"],
            environment_id=row["environment_id"],
            alternative_id=row["alternative_id"],
            package_lock=PackageLock.from_dict(row["package_lock"]),
            environment_request=EnvironmentRequest.from_dict(row["environment_request"]),
        )


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
    owner_actor_id: str = "service:local"
    authority_id: str = "authority:local"
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
        owner = str(self.owner_actor_id or "").strip()
        authority = str(self.authority_id or "").strip()
        if not owner or not authority:
            raise ValueError("toolbox_plan_owner_required")
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
        object.__setattr__(self, "owner_actor_id", owner)
        object.__setattr__(self, "authority_id", authority)
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
            "owner_actor_id": self.owner_actor_id,
            "authority_id": self.authority_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersistedToolboxDefinitionPlan":
        row = dict(payload or {})
        fields = {
            "contract", "plan_id", "toolbox_id", "definition_revision",
            "expected_revision", "catalog_revision", "package_policy_revision",
            "created_at_ms", "expires_at_ms", "plan", "profile_changes",
            "owner_actor_id", "authority_id",
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
        owner_actor_id: str = "service:local",
        authority_id: str = "authority:local",
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
            "owner_actor_id": str(owner_actor_id or "").strip(),
            "authority_id": str(authority_id or "").strip(),
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
            owner_actor_id=str(owner_actor_id or "").strip(),
            authority_id=str(authority_id or "").strip(),
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
                    or existing_record.owner_actor_id != record.owner_actor_id
                    or existing_record.authority_id != record.authority_id
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

    def invalidate_all(self) -> int:
        """Remove unused immutable plans after a host configuration transition."""
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            count = len(state["plans"])
            if count:
                state["plans"] = {}
                self._write_unlocked(state)
        return count


@dataclass(frozen=True)
class PersistedCompleteToolboxDefinitionPlan:
    plan_id: str
    active_definition: ToolboxDefinitionSpec
    proposed_definition: ToolboxDefinitionSpec
    pins: ToolboxPlanPins
    environment_mutations: tuple[ToolboxEnvironmentMutationSpec, ...]
    planned_environments: tuple[ToolboxPlannedEnvironmentRecord, ...]
    proposal_kind: str
    changes: tuple[NormalizedToolboxToolChange, ...]
    parent_plan_id: str | None
    reduction: Mapping[str, Any] | None
    draft_plan: Mapping[str, Any]
    profile_changes: tuple[Mapping[str, Any], ...]
    created_at_ms: int
    expires_at_ms: int
    owner_actor_id: str
    authority_id: str
    contract: str = TOOLBOX_COMPLETE_PLAN_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != TOOLBOX_COMPLETE_PLAN_CONTRACT:
            raise ValueError("toolbox_complete_plan_contract_invalid")
        object.__setattr__(
            self, "plan_id", require_digest(self.plan_id, label="toolbox_complete_plan_id")
        )
        if not isinstance(self.active_definition, ToolboxDefinitionSpec) or not isinstance(
            self.proposed_definition, ToolboxDefinitionSpec
        ):
            raise ValueError("toolbox_complete_plan_definition_invalid")
        if self.active_definition.toolbox_id != self.proposed_definition.toolbox_id:
            raise ValueError("toolbox_complete_plan_toolbox_mismatch")
        active_empty = (
            not self.active_definition.auto_requests
            and not self.active_definition.manual_requests
            and not self.active_definition.intrinsics.names
        )
        active_revision = None if active_empty else self.active_definition.revision
        if (
            self.proposed_definition.expected_revision != active_revision
            or self.pins.active_definition_revision != active_revision
        ):
            raise ValueError("toolbox_complete_plan_active_revision_mismatch")
        if not isinstance(self.pins, ToolboxPlanPins):
            raise ValueError("toolbox_complete_plan_pins_invalid")
        mutations = tuple(sorted(self.environment_mutations, key=lambda item: item.environment_id))
        if not mutations or len({item.environment_id for item in mutations}) != len(mutations):
            raise ValueError("toolbox_complete_plan_environment_mutations_invalid")
        offered_tools = [
            tool.tool_key for environment in mutations for tool in environment.tool_mutations
        ]
        expected_tools = {
            *(item.stable_key for item in self.active_definition.auto_requests),
            *(item.stable_key for item in self.active_definition.manual_requests),
            *(f"intrinsic:{item}" for item in self.active_definition.intrinsics.names),
            *(item.stable_key for item in self.proposed_definition.auto_requests),
            *(item.stable_key for item in self.proposed_definition.manual_requests),
            *(f"intrinsic:{item}" for item in self.proposed_definition.intrinsics.names),
        }
        if len(set(offered_tools)) != len(offered_tools) or set(offered_tools) != expected_tools:
            raise ValueError("toolbox_complete_plan_offered_tools_invalid")
        planned = tuple(
            sorted(
                self.planned_environments,
                key=lambda item: (item.environment_id, item.alternative_id),
            )
        )
        expected_planned = {
            (environment.environment_id, alternative.alternative_id)
            for environment in mutations
            if any(tool.change != "removed" for tool in environment.tool_mutations)
            for alternative in environment.alternatives
        }
        if (
            any(not isinstance(item, ToolboxPlannedEnvironmentRecord) for item in planned)
            or len({(item.environment_id, item.alternative_id) for item in planned}) != len(planned)
            or {(item.environment_id, item.alternative_id) for item in planned} != expected_planned
        ):
            raise ValueError("toolbox_complete_plan_planned_environments_invalid")
        offers = {item.environment_id: item for item in mutations}
        for item in planned:
            offer = offers[item.environment_id]
            alternative = next(
                value for value in offer.alternatives
                if value.alternative_id == item.alternative_id
            )
            request = item.environment_request
            if (
                request.consumer_kind != "toolbox"
                or request.consumer_id != self.proposed_definition.toolbox_id
                or request.template_id != offer.base_template_id
                or request.configuration_revision != self.pins.configuration_revision
            ):
                raise ValueError("toolbox_complete_plan_environment_request_mismatch")
            expected_dependencies = {
                (artifact.distribution, artifact.version, artifact.artifact_digest)
                for artifact in alternative.artifacts
            }
            actual_dependencies = {
                (value["name"], value["version"], value["artifact_id"])
                for value in item.package_lock.dependencies
            }
            expected_artifacts = {
                (artifact.artifact_digest, artifact.source_id)
                for artifact in alternative.artifacts
            }
            actual_artifacts = {
                (value["artifact_id"], value["source_id"])
                for value in item.package_lock.artifacts
            }
            if actual_dependencies != expected_dependencies or actual_artifacts != expected_artifacts:
                raise ValueError("toolbox_complete_plan_package_lock_mismatch")
        if self.proposal_kind not in {"complete_definition", "tool_changes"}:
            raise ValueError("toolbox_complete_plan_proposal_kind_invalid")
        changes = tuple(sorted(self.changes, key=lambda item: item.change_id))
        if (
            any(not isinstance(item, NormalizedToolboxToolChange) for item in changes)
            or len({item.change_id for item in changes}) != len(changes)
            or (self.proposal_kind == "tool_changes" and not changes)
            or (
                self.proposal_kind == "complete_definition"
                and any(not item.change_id.startswith("host:sha256:") for item in changes)
            )
        ):
            raise ValueError("toolbox_complete_plan_changes_invalid")
        actual_changed_keys = {
            tool.tool_key
            for environment in mutations
            for tool in environment.tool_mutations
            if tool.change != "unchanged"
        }
        planned_changed_keys = {
            key
            for item in changes
            for key in (item.prior_tool_key, item.tool_key)
            if key is not None
        }
        if actual_changed_keys != planned_changed_keys:
            raise ValueError("toolbox_complete_plan_change_coverage_invalid")
        parent_plan_id = (
            None if self.parent_plan_id is None
            else require_digest(self.parent_plan_id, label="toolbox_parent_plan_id")
        )
        if parent_plan_id is None:
            if self.reduction is not None:
                raise ValueError("toolbox_complete_plan_root_reduction_invalid")
            reduction = None
        else:
            reduction = dict(self.reduction or {})
            if set(reduction) != {
                "excluded_changes", "preserved_active_tool_keys", "cascade_exclusions"
            }:
                raise ValueError("toolbox_complete_plan_reduction_invalid")
        draft = dict(self.draft_plan or {})
        fields = {
            "definition", "definition_revision", "profiles", "bundles",
            "custom_environment_count",
        }
        if set(draft) != fields:
            raise ValueError("toolbox_complete_plan_draft_fields_invalid")
        if (
            draft["definition"] != self.proposed_definition.to_dict()
            or draft["definition_revision"] != self.proposed_definition.revision
        ):
            raise ValueError("toolbox_complete_plan_draft_definition_mismatch")
        parsed_draft = ToolboxDefinitionPlanDraft.from_persisted_dict(draft)
        profiles = parsed_draft.profiles
        if draft["custom_environment_count"] != sum(
            item.custom_resolved_lock_digest is not None for item in profiles
        ):
            raise ValueError("toolbox_complete_plan_draft_profiles_invalid")
        profile_changes = tuple(dict(item) for item in self.profile_changes)
        for item in profile_changes:
            if set(item) != {
                "classification", "active_profile_id", "proposed_profile_id", "changed_fields"
            } or item["classification"] not in {"reused", "added", "replaced", "removed"}:
                raise ValueError("toolbox_complete_plan_profile_change_invalid")
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.created_at_ms < 0
            or not self.created_at_ms < self.expires_at_ms <= self.created_at_ms + MAX_TOOLBOX_PLAN_TTL_MS
        ):
            raise ValueError("toolbox_complete_plan_lifetime_invalid")
        owner = str(self.owner_actor_id or "").strip()
        authority = str(self.authority_id or "").strip()
        if not owner or not authority:
            raise ValueError("toolbox_complete_plan_owner_invalid")
        object.__setattr__(self, "environment_mutations", mutations)
        object.__setattr__(self, "planned_environments", planned)
        object.__setattr__(self, "changes", changes)
        object.__setattr__(self, "parent_plan_id", parent_plan_id)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "draft_plan", draft)
        object.__setattr__(self, "profile_changes", profile_changes)
        object.__setattr__(self, "owner_actor_id", owner)
        object.__setattr__(self, "authority_id", authority)
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > MAX_TOOLBOX_PLAN_BYTES:
            raise ValueError("toolbox_complete_plan_too_large")

    @property
    def toolbox_id(self) -> str:
        return self.proposed_definition.toolbox_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "plan_id": self.plan_id,
            "active_definition": self.active_definition.to_dict(),
            "proposed_definition": self.proposed_definition.to_dict(),
            "pins": self.pins.to_dict(),
            "environment_mutations": [item.to_dict() for item in self.environment_mutations],
            "planned_environments": [item.to_dict() for item in self.planned_environments],
            "proposal_kind": self.proposal_kind,
            "changes": [item.to_dict() for item in self.changes],
            "parent_plan_id": self.parent_plan_id,
            "reduction": None if self.reduction is None else dict(self.reduction),
            "draft_plan": dict(self.draft_plan),
            "profile_changes": [dict(item) for item in self.profile_changes],
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "owner_actor_id": self.owner_actor_id,
            "authority_id": self.authority_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersistedCompleteToolboxDefinitionPlan":
        row = dict(payload or {})
        fields = {
            "contract", "plan_id", "active_definition", "proposed_definition", "pins",
            "environment_mutations", "planned_environments", "proposal_kind", "changes",
            "parent_plan_id", "reduction", "draft_plan", "profile_changes", "created_at_ms",
            "expires_at_ms", "owner_actor_id", "authority_id",
        }
        if set(row) != fields:
            raise ValueError("toolbox_complete_plan_fields_invalid")
        return cls(
            **{
                **row,
                "active_definition": ToolboxDefinitionSpec.from_dict(row["active_definition"]),
                "proposed_definition": ToolboxDefinitionSpec.from_dict(row["proposed_definition"]),
                "pins": ToolboxPlanPins.from_dict(row["pins"]),
                "environment_mutations": tuple(
                    ToolboxEnvironmentMutationSpec.from_dict(item)
                    for item in row["environment_mutations"]
                ),
                "planned_environments": tuple(
                    ToolboxPlannedEnvironmentRecord.from_dict(item)
                    for item in row["planned_environments"]
                ),
                "changes": tuple(
                    NormalizedToolboxToolChange.from_dict(item)
                    for item in row["changes"]
                ),
                "profile_changes": tuple(row["profile_changes"]),
            }
        )


class AtomicJsonCompleteToolboxDefinitionPlanRepository:
    """Process-safe immutable complete plan store used by hosted plan/confirm/apply."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": TOOLBOX_COMPLETE_PLAN_STATE_CONTRACT, "plans": {}}

    @staticmethod
    def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if (
            set(row) != {"contract", "plans"}
            or row.get("contract") != TOOLBOX_COMPLETE_PLAN_STATE_CONTRACT
            or not isinstance(row["plans"], dict)
            or len(row["plans"]) > MAX_TOOLBOX_PLANS
        ):
            raise ValueError("toolbox_complete_plan_state_invalid")
        plans = {}
        for plan_id, raw in row["plans"].items():
            plan = PersistedCompleteToolboxDefinitionPlan.from_dict(raw)
            if plan_id != plan.plan_id:
                raise ValueError("toolbox_complete_plan_state_key_invalid")
            plans[plan_id] = plan.to_dict()
        return {"contract": TOOLBOX_COMPLETE_PLAN_STATE_CONTRACT, "plans": plans}

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("toolbox_complete_plan_state_corrupt") from exc
        return self._validate(value)

    def _write(self, state: Mapping[str, Any]) -> None:
        value = self._validate(state)
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
        active_definition: ToolboxDefinitionSpec,
        pins: ToolboxPlanPins,
        environment_mutations: Sequence[ToolboxEnvironmentMutationSpec],
        planned_environments: Sequence[ToolboxPlannedEnvironmentRecord],
        proposal_kind: str,
        changes: Sequence[NormalizedToolboxToolChange],
        parent_plan_id: str | None,
        reduction: Mapping[str, Any] | None,
        active_profiles: Sequence[ActiveToolboxProfileSnapshot | Mapping[str, Any]],
        now_ms: int,
        ttl_ms: int,
        owner_actor_id: str,
        authority_id: str,
    ) -> PersistedCompleteToolboxDefinitionPlan:
        if not isinstance(draft, ToolboxDefinitionPlanDraft):
            raise ValueError("toolbox_complete_plan_draft_required")
        if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or not 1 <= ttl_ms <= MAX_TOOLBOX_PLAN_TTL_MS:
            raise ValueError("toolbox_complete_plan_ttl_invalid")
        identity_payload = {
            "active_definition": active_definition.to_dict(),
            "proposed_definition": draft.definition.to_dict(),
            "pins": pins.to_dict(),
            "environment_mutations": [item.to_dict() for item in environment_mutations],
            "planned_environments": [item.to_dict() for item in planned_environments],
            "proposal_kind": proposal_kind,
            "changes": [item.to_dict() for item in changes],
            "parent_plan_id": parent_plan_id,
            "reduction": None if reduction is None else dict(reduction),
            "draft_plan": draft.to_persisted_dict(),
            "profile_changes": list(classify_toolbox_profiles(draft, active_profiles)),
            "owner_actor_id": str(owner_actor_id or "").strip(),
            "authority_id": str(authority_id or "").strip(),
        }
        record = PersistedCompleteToolboxDefinitionPlan(
            plan_id=identity_digest(TOOLBOX_COMPLETE_PLAN_ID_DOMAIN, identity_payload),
            active_definition=active_definition,
            proposed_definition=draft.definition,
            pins=pins,
            environment_mutations=tuple(environment_mutations),
            planned_environments=tuple(planned_environments),
            proposal_kind=proposal_kind,
            changes=tuple(changes),
            parent_plan_id=parent_plan_id,
            reduction=reduction,
            draft_plan=draft.to_persisted_dict(),
            profile_changes=tuple(identity_payload["profile_changes"]),
            created_at_ms=now_ms,
            expires_at_ms=now_ms + ttl_ms,
            owner_actor_id=owner_actor_id,
            authority_id=authority_id,
        )
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            changed = self._prune(state, now_ms=now_ms)
            existing = state["plans"].get(record.plan_id)
            if existing is not None:
                recovered = PersistedCompleteToolboxDefinitionPlan.from_dict(existing)
                recovered_identity = recovered.to_dict()
                record_identity = record.to_dict()
                for field_name in ("created_at_ms", "expires_at_ms"):
                    recovered_identity.pop(field_name)
                    record_identity.pop(field_name)
                if recovered_identity != record_identity:
                    raise ValueError("toolbox_complete_plan_id_conflict")
                if changed:
                    self._write(state)
                return recovered
            if len(state["plans"]) >= MAX_TOOLBOX_PLANS:
                raise ValueError("toolbox_complete_plan_capacity")
            state["plans"][record.plan_id] = record.to_dict()
            self._write(state)
        return record

    def get(self, plan_id: str, *, now_ms: int) -> PersistedCompleteToolboxDefinitionPlan:
        key = require_digest(plan_id, label="toolbox_complete_plan_id")
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            existing = state["plans"].get(key)
            if existing is not None and existing["expires_at_ms"] <= now_ms:
                state["plans"].pop(key)
                self._write(state)
                raise ValueError("toolbox_definition_plan_expired")
            if self._prune(state, now_ms=now_ms):
                self._write(state)
            if existing is None:
                raise ValueError("toolbox_definition_plan_not_found")
            return PersistedCompleteToolboxDefinitionPlan.from_dict(existing)

    def list(self, *, now_ms: int) -> tuple[PersistedCompleteToolboxDefinitionPlan, ...]:
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            if self._prune(state, now_ms=now_ms):
                self._write(state)
            plans = tuple(PersistedCompleteToolboxDefinitionPlan.from_dict(item) for item in state["plans"].values())
        return tuple(sorted(plans, key=lambda item: (item.created_at_ms, item.plan_id)))

    def invalidate_all(self) -> int:
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            count = len(state["plans"])
            if count:
                state["plans"] = {}
                self._write(state)
        return count


__all__ = [
    "AtomicJsonCompleteToolboxDefinitionPlanRepository",
    "AtomicJsonToolboxDefinitionPlanRepository",
    "MAX_TOOLBOX_PLAN_BYTES",
    "MAX_TOOLBOX_PLAN_TTL_MS",
    "MAX_TOOLBOX_PLANS",
    "PersistedCompleteToolboxDefinitionPlan",
    "PersistedToolboxDefinitionPlan",
]
