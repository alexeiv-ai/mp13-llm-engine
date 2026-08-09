"""Actor-bound parent-minted approvals for exact custom dependency plans."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..toolbox.identity import require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


APPROVAL_STATE_CONTRACT = "hosting.toolbox.dependency_approval_state.v1"
APPROVAL_CONTRACT = "hosting.toolbox.dependency_approval"


class ToolboxDependencyApprovalError(PermissionError):
    pass


class AtomicJsonToolboxDependencyApprovalRepository:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": APPROVAL_STATE_CONTRACT, "approvals": {}}

    @staticmethod
    def _ref_digest(approval_ref: str) -> str:
        return f"sha256:{hashlib.sha256(str(approval_ref).encode('utf-8')).hexdigest()}"

    @classmethod
    def _validate(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {"contract", "approvals"} or row.get("contract") != APPROVAL_STATE_CONTRACT:
            raise ValueError("toolbox_approval_state_invalid")
        if not isinstance(row.get("approvals"), dict):
            raise ValueError("toolbox_approval_state_invalid")
        approvals: dict[str, dict[str, Any]] = {}
        fields = {
            "approval_ref_digest", "owner_actor_id", "authority_id", "toolbox_id", "plan_id",
            "definition_revision", "custom_delta_digest", "catalog_revision", "package_policy_revision",
            "decision", "minted_at_ms", "expires_at_ms", "consumed_request_id", "revoked_at_ms",
        }
        for key, value in row["approvals"].items():
            item = dict(value or {})
            if set(item) != fields or key != item.get("approval_ref_digest"):
                raise ValueError("toolbox_approval_record_invalid")
            for digest_field in (
                "approval_ref_digest", "plan_id", "definition_revision", "custom_delta_digest",
                "catalog_revision", "package_policy_revision",
            ):
                require_digest(item[digest_field], label=f"approval_{digest_field}")
            if item["decision"] != "approved":
                raise ValueError("toolbox_approval_decision_invalid")
            if any(not str(item[field] or "").strip() for field in ("owner_actor_id", "authority_id", "toolbox_id")):
                raise ValueError("toolbox_approval_owner_invalid")
            if (
                isinstance(item["minted_at_ms"], bool)
                or not isinstance(item["minted_at_ms"], int)
                or isinstance(item["expires_at_ms"], bool)
                or not isinstance(item["expires_at_ms"], int)
                or item["expires_at_ms"] <= item["minted_at_ms"]
            ):
                raise ValueError("toolbox_approval_lifetime_invalid")
            approvals[key] = item
        return {"contract": APPROVAL_STATE_CONTRACT, "approvals": approvals}

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("toolbox_approval_state_corrupt") from exc
        if not isinstance(payload, dict):
            raise ValueError("toolbox_approval_state_corrupt")
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

    def mint(
        self,
        *,
        owner_actor_id: str,
        authority_id: str,
        toolbox_id: str,
        plan_id: str,
        definition_revision: str,
        custom_delta_digest: str,
        catalog_revision: str,
        package_policy_revision: str,
        now_ms: int,
        expires_at_ms: int,
    ) -> dict[str, Any]:
        approval_ref = f"approval_{secrets.token_urlsafe(32)}"
        key = self._ref_digest(approval_ref)
        record = {
            "approval_ref_digest": key,
            "owner_actor_id": str(owner_actor_id or "").strip(),
            "authority_id": str(authority_id or "").strip(),
            "toolbox_id": str(toolbox_id or "").strip(),
            "plan_id": require_digest(plan_id, label="approval_plan_id"),
            "definition_revision": require_digest(definition_revision, label="approval_definition_revision"),
            "custom_delta_digest": require_digest(custom_delta_digest, label="approval_custom_delta_digest"),
            "catalog_revision": require_digest(catalog_revision, label="approval_catalog_revision"),
            "package_policy_revision": require_digest(package_policy_revision, label="approval_policy_revision"),
            "decision": "approved",
            "minted_at_ms": int(now_ms),
            "expires_at_ms": int(expires_at_ms),
            "consumed_request_id": None,
            "revoked_at_ms": None,
        }
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            state["approvals"][key] = record
            self._write(state)
        return {
            "contract": APPROVAL_CONTRACT,
            "approval_ref": approval_ref,
            "plan_id": plan_id,
            "expires_at_ms": expires_at_ms,
            "user_projection": {
                "state": "ready",
                "code": "custom_dependency_approved",
                "summary": "The planned additional packages were approved.",
            },
        }

    def validate_and_consume(
        self,
        *,
        approval_ref: str,
        owner_actor_id: str,
        authority_id: str,
        toolbox_id: str,
        plan_id: str,
        definition_revision: str,
        custom_delta_digest: str,
        catalog_revision: str,
        package_policy_revision: str,
        request_id: str,
        now_ms: int,
    ) -> dict[str, Any]:
        key = self._ref_digest(str(approval_ref or ""))
        expected = {
            "owner_actor_id": str(owner_actor_id or "").strip(),
            "authority_id": str(authority_id or "").strip(),
            "toolbox_id": str(toolbox_id or "").strip(),
            "plan_id": plan_id,
            "definition_revision": definition_revision,
            "custom_delta_digest": custom_delta_digest,
            "catalog_revision": catalog_revision,
            "package_policy_revision": package_policy_revision,
        }
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            record = dict(state["approvals"].get(key) or {})
            valid = bool(record) and all(record.get(field) == value for field, value in expected.items())
            valid = valid and record.get("revoked_at_ms") is None and int(record.get("expires_at_ms") or 0) > int(now_ms)
            consumed = str(record.get("consumed_request_id") or "").strip()
            valid = valid and (not consumed or consumed == str(request_id or "").strip())
            if not valid:
                raise ToolboxDependencyApprovalError("dependency_approval_invalid")
            record["consumed_request_id"] = str(request_id or "").strip()
            state["approvals"][key] = record
            self._write(state)
            return copy.deepcopy(record)


__all__ = [
    "AtomicJsonToolboxDependencyApprovalRepository",
    "ToolboxDependencyApprovalError",
]
