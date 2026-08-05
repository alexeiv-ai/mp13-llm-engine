"""Generic hosted-operation status, cancellation, and ledger cutover service API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from ..operation_contract import HostedExecutionKind, HostedOperationRef, HostedOperationSelector
from .operation_repository import AtomicJsonHostedOperationRepository


class HostedOperationsMixin:
    @staticmethod
    def _operation_owner(owner_actor_id: str) -> str:
        return str(owner_actor_id or "service:local").strip() or "service:local"

    def hosted_operation_status(
        self,
        *,
        ref: HostedOperationRef | Mapping[str, Any],
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        operation = ref if isinstance(ref, HostedOperationRef) else HostedOperationRef.from_dict(ref)
        return self._hosted_operations.status(
            ref=operation,
            owner_actor_id=self._operation_owner(owner_actor_id),
        )

    def hosted_operation_resolve_request(
        self,
        *,
        execution_kind: HostedExecutionKind | str,
        selector: HostedOperationSelector | Mapping[str, Any],
        request_id: str,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        """Recover a canonical ref when the original execute response was lost."""

        kind = execution_kind if isinstance(execution_kind, HostedExecutionKind) else HostedExecutionKind(str(execution_kind))
        target = selector if isinstance(selector, HostedOperationSelector) else HostedOperationSelector.from_dict(selector)
        if kind == HostedExecutionKind.TOOLBOX:
            namespace = f"toolbox:{target.id}" if target.kind == "toolbox_id" else f"engine:{target.id}"
        else:
            if target.kind != "engine_id":
                raise ValueError("workflow_operation_selector_must_be_engine_id")
            namespace = f"{kind.value}:{target.id}"
        owner = self._operation_owner(owner_actor_id)
        record = self._hosted_operations.get_by_request(
            owner_actor_id=owner,
            namespace=namespace,
            request_id=str(request_id or "").strip(),
        )
        if record is None or str(dict(record.get("operation") or {}).get("execution_kind") or "") != kind.value:
            return {"status": "not_found", "reason": "operation_not_found"}
        return self._hosted_operations.status(ref=dict(record["operation"]), owner_actor_id=owner)

    def hosted_operation_result(
        self,
        *,
        ref: HostedOperationRef | Mapping[str, Any],
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        operation = ref if isinstance(ref, HostedOperationRef) else HostedOperationRef.from_dict(ref)
        return self._hosted_operations.read_result(
            ref=operation,
            owner_actor_id=self._operation_owner(owner_actor_id),
        )

    def hosted_operation_cancel(
        self,
        *,
        ref: HostedOperationRef | Mapping[str, Any],
        reason: str = "client_requested",
        owner_actor_id: str = "service:local",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
    ) -> Dict[str, Any]:
        operation = ref if isinstance(ref, HostedOperationRef) else HostedOperationRef.from_dict(ref)
        owner = self._operation_owner(owner_actor_id)
        record = self._hosted_operations.resolve(ref=operation, owner_actor_id=owner)
        if record is None:
            return self._hosted_operations.status(ref=operation, owner_actor_id=owner)
        if operation.execution_kind == HostedExecutionKind.TOOLBOX:
            return self._cancel_toolbox_operation(
                record=record,
                reason=str(reason or "client_requested"),
                timeout_seconds=float(timeout_seconds or 8.0),
                respawn=bool(respawn),
            )
        return self._cancel_workflow_operation(record=record, reason=str(reason or "client_requested"))

    def hosting_receipt_ledger_cutover(
        self,
        *,
        acknowledge_replay_window_clear: bool,
    ) -> Dict[str, Any]:
        state_root = (Path(self.hosting_root).expanduser().resolve() / "state").resolve()
        legacy_path = (state_root / "toolbox_execution_receipts.json").resolve()
        new_path = (state_root / "hosted_operations.json").resolve()
        if new_path.exists():
            raise FileExistsError("new hosted-operation repository already exists")
        archived = AtomicJsonHostedOperationRepository.archive_legacy_checkpoint(
            legacy_path,
            acknowledge_replay_window_clear=bool(acknowledge_replay_window_clear),
        )
        return {
            "status": "ok",
            "legacy_path": str(legacy_path),
            "archived_path": str(archived),
            "new_repository_path": str(new_path),
        }


__all__ = ["HostedOperationsMixin"]
