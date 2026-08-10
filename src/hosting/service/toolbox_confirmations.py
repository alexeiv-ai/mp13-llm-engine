"""Immutable actor-bound receipts for confirmed toolbox definition plans."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..toolbox.bundle_models import ToolboxDefinitionSpec
from ..toolbox.definition_planner import ToolboxConfirmationReduction
from ..toolbox.identity import identity_digest, require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


CONFIRMATION_STATE_CONTRACT = "hosting.toolbox.confirmation_state.v1"
CONFIRMATION_RECEIPT_CONTRACT = "hosting.toolbox.confirmation_receipt.v1"


def _ref_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolboxConfirmationReceipt:
    confirmation_ref_digest: str
    plan_id: str
    toolbox_id: str
    owner_actor_id: str
    authority_id: str
    choices_digest: str
    reduction: Mapping[str, Any]
    confirmed_draft: Mapping[str, Any]
    created_at_ms: int
    expires_at_ms: int
    contract: str = CONFIRMATION_RECEIPT_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != CONFIRMATION_RECEIPT_CONTRACT:
            raise ValueError("toolbox_confirmation_receipt_contract_invalid")
        for label, value in (
            ("ref", self.confirmation_ref_digest),
            ("plan", self.plan_id),
            ("choices", self.choices_digest),
        ):
            require_digest(value, label=f"toolbox_confirmation_{label}_digest")
        if not all(
            str(value or "").strip()
            for value in (self.toolbox_id, self.owner_actor_id, self.authority_id)
        ):
            raise ValueError("toolbox_confirmation_receipt_owner_invalid")
        row = dict(self.reduction or {})
        fields = {
            "effective_definition", "effective_definition_revision",
            "selected_alternatives", "accepted_tool_keys", "skipped_tools",
            "preserved_active_tool_keys", "removed_tool_keys", "package_mutations",
            "dependency_approval_required",
        }
        if set(row) != fields:
            raise ValueError("toolbox_confirmation_reduction_fields_invalid")
        definition = ToolboxDefinitionSpec.from_dict(row["effective_definition"])
        if definition.toolbox_id != self.toolbox_id or definition.revision != row["effective_definition_revision"]:
            raise ValueError("toolbox_confirmation_effective_definition_invalid")
        if not isinstance(row["dependency_approval_required"], bool):
            raise ValueError("toolbox_confirmation_approval_flag_invalid")
        if not self.created_at_ms < self.expires_at_ms:
            raise ValueError("toolbox_confirmation_lifetime_invalid")
        object.__setattr__(self, "reduction", copy.deepcopy(row))
        draft = copy.deepcopy(dict(self.confirmed_draft or {}))
        if set(draft) != {
            "definition", "definition_revision", "profiles", "bundles",
            "custom_environment_count",
        } or draft["definition"] != definition.to_dict() or draft["definition_revision"] != definition.revision:
            raise ValueError("toolbox_confirmation_draft_invalid")
        object.__setattr__(self, "confirmed_draft", draft)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "confirmation_ref_digest": self.confirmation_ref_digest,
            "plan_id": self.plan_id,
            "toolbox_id": self.toolbox_id,
            "owner_actor_id": self.owner_actor_id,
            "authority_id": self.authority_id,
            "choices_digest": self.choices_digest,
            "reduction": copy.deepcopy(dict(self.reduction)),
            "confirmed_draft": copy.deepcopy(dict(self.confirmed_draft)),
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxConfirmationReceipt":
        row = dict(payload or {})
        if set(row) != {
            "contract", "confirmation_ref_digest", "plan_id", "toolbox_id",
            "owner_actor_id", "authority_id", "choices_digest", "reduction",
            "confirmed_draft",
            "created_at_ms", "expires_at_ms",
        }:
            raise ValueError("toolbox_confirmation_receipt_fields_invalid")
        return cls(**row)


class AtomicJsonToolboxConfirmationRepository:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": CONFIRMATION_STATE_CONTRACT, "receipts": {}}

    @staticmethod
    def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {"contract", "receipts"} or row.get("contract") != CONFIRMATION_STATE_CONTRACT or not isinstance(row["receipts"], dict):
            raise ValueError("toolbox_confirmation_state_invalid")
        receipts = {}
        for key, raw in row["receipts"].items():
            receipt = ToolboxConfirmationReceipt.from_dict(raw)
            if key != receipt.confirmation_ref_digest:
                raise ValueError("toolbox_confirmation_state_key_invalid")
            receipts[key] = receipt.to_dict()
        return {"contract": CONFIRMATION_STATE_CONTRACT, "receipts": receipts}

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            return self._validate(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("toolbox_confirmation_state_corrupt") from exc

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

    def create(
        self,
        *,
        plan_id: str,
        toolbox_id: str,
        owner_actor_id: str,
        authority_id: str,
        choices: Sequence[Mapping[str, Any]],
        reduction: ToolboxConfirmationReduction,
        confirmed_draft: Mapping[str, Any],
        now_ms: int,
        expires_at_ms: int,
    ) -> tuple[str, ToolboxConfirmationReceipt]:
        choices_payload = [dict(item) for item in choices]
        choices_digest = identity_digest("hosting.toolbox.confirmation.choices.v1", choices_payload)
        identity = identity_digest(
            "hosting.toolbox.confirmation.ref.v1",
            {
                "plan_id": plan_id,
                "owner_actor_id": owner_actor_id,
                "authority_id": authority_id,
                "choices_digest": choices_digest,
                "reduction": reduction.to_dict(),
            },
        )
        confirmation_ref = "confirmation_" + identity.removeprefix("sha256:")
        receipt = ToolboxConfirmationReceipt(
            confirmation_ref_digest=_ref_digest(confirmation_ref),
            plan_id=plan_id,
            toolbox_id=toolbox_id,
            owner_actor_id=str(owner_actor_id or "").strip(),
            authority_id=str(authority_id or "").strip(),
            choices_digest=choices_digest,
            reduction=reduction.to_dict(),
            confirmed_draft=dict(confirmed_draft),
            created_at_ms=int(now_ms),
            expires_at_ms=int(expires_at_ms),
        )
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            existing = state["receipts"].get(receipt.confirmation_ref_digest)
            if existing is not None:
                prior = ToolboxConfirmationReceipt.from_dict(existing)
                comparable_prior = prior.to_dict()
                comparable_new = receipt.to_dict()
                comparable_new["created_at_ms"] = comparable_prior["created_at_ms"]
                if comparable_prior != comparable_new:
                    raise ValueError("toolbox_confirmation_receipt_conflict")
                return confirmation_ref, prior
            state["receipts"][receipt.confirmation_ref_digest] = receipt.to_dict()
            self._write(state)
        return confirmation_ref, receipt

    def get(
        self,
        confirmation_ref: str,
        *,
        owner_actor_id: str,
        authority_id: str,
        now_ms: int,
    ) -> ToolboxConfirmationReceipt:
        key = _ref_digest(str(confirmation_ref or ""))
        with _exclusive_process_file_lock(self.lock_path):
            receipt_raw = self._read()["receipts"].get(key)
        if receipt_raw is None:
            raise PermissionError("toolbox_confirmation_not_found")
        receipt = ToolboxConfirmationReceipt.from_dict(receipt_raw)
        if receipt.owner_actor_id != str(owner_actor_id or "").strip() or receipt.authority_id != str(authority_id or "").strip():
            raise PermissionError("toolbox_confirmation_not_found")
        if receipt.expires_at_ms <= int(now_ms):
            raise ValueError("toolbox_confirmation_expired")
        return receipt

    def get_for_approval(self, confirmation_ref: str, *, now_ms: int) -> ToolboxConfirmationReceipt:
        """Resolve a receipt for the separately authorized approver surface."""
        key = _ref_digest(str(confirmation_ref or ""))
        with _exclusive_process_file_lock(self.lock_path):
            receipt_raw = self._read()["receipts"].get(key)
        if receipt_raw is None:
            raise PermissionError("toolbox_confirmation_not_found")
        receipt = ToolboxConfirmationReceipt.from_dict(receipt_raw)
        if receipt.expires_at_ms <= int(now_ms):
            raise ValueError("toolbox_confirmation_expired")
        return receipt


__all__ = ["AtomicJsonToolboxConfirmationRepository", "ToolboxConfirmationReceipt"]
