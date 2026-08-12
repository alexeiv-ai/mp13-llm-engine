"""Generic hosted-operation status, cancellation, and ledger cutover service API."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from ..operation_contract import (
    TOOLBOX_DEFINITION_APPLY_COMMITTED_PHASES,
    HostedExecutionKind,
    HostedOperationRef,
    HostedOperationSelector,
    HostedOperationProgress,
)
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
        elif kind == HostedExecutionKind.ENVIRONMENT_TEMPLATE_PREWARM:
            if target.kind != "template_id":
                raise ValueError("template_prewarm_selector_must_be_template_id")
            namespace = f"environment_template_prewarm:{target.id}"
        elif kind == HostedExecutionKind.TOOLBOX_DEFINITION_APPLY:
            if target.kind != "toolbox_id":
                raise ValueError("toolbox_definition_apply_selector_must_be_toolbox_id")
            namespace = f"toolbox-definition:{target.id}"
        elif kind in {
            HostedExecutionKind.TOOLBOX_DEFINITION_PLAN,
            HostedExecutionKind.TOOLBOX_DEFINITION_PLAN_REVISION,
        }:
            if target.kind != "toolbox_id":
                raise ValueError("toolbox_definition_plan_selector_must_be_toolbox_id")
            namespace = f"toolbox-definition-plan:{target.id}"
        elif kind == HostedExecutionKind.TOOLBOX_DEFINITION_CONFIRMATION:
            if target.kind != "toolbox_id":
                raise ValueError("toolbox_definition_confirmation_selector_must_be_toolbox_id")
            namespace = f"toolbox-definition-confirmation:{target.id}"
        elif kind == HostedExecutionKind.TOOLBOX_ARTIFACT_IMPORT:
            if target.kind != "upload_id":
                raise ValueError("toolbox_artifact_import_selector_must_be_upload_id")
            namespace = f"toolbox_artifact_import:{target.id}"
        elif kind == HostedExecutionKind.ENVIRONMENT_REMOVE:
            if target.kind != "environment_id":
                raise ValueError("environment_remove_selector_must_be_environment_id")
            namespace = f"environment_remove:{target.id}"
        elif kind == HostedExecutionKind.ENVIRONMENT_TEMPLATE_CONSTRUCT:
            if target.kind != "template_id":
                raise ValueError("environment_template_construct_selector_must_be_template_id")
            namespace = f"environment_template_construct:{target.id}"
        elif kind == HostedExecutionKind.HOSTING_GC:
            if target.kind != "host_scope" or target.id != "hosting":
                raise ValueError("hosting_gc_selector_must_be_host_scope")
            namespace = "hosting_gc:hosting"
        elif kind == HostedExecutionKind.TOOLBOX_MAINTENANCE:
            if target.kind != "host_scope" or target.id != "toolbox-host":
                raise ValueError("toolbox_maintenance_selector_must_be_host_scope")
            namespace = "toolbox_maintenance:toolbox-host"
        elif kind == HostedExecutionKind.TOOLBOX_DESCRIBE_REFRESH:
            if target.kind not in {"toolbox_id", "engine_id"}:
                raise ValueError("toolbox_describe_refresh_selector_invalid")
            namespace = f"toolbox_describe_refresh:{target.kind}:{target.id}"
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
            lifecycle = str(record.get("lifecycle") or "")
            if lifecycle not in {
                "queued",
                "interrupted_before_dispatch",
                "terminal_success",
                "terminal_failure",
                "terminal_cancellation",
                "forgotten",
            }:
                lock = getattr(self, "_toolbox_cancel_lock", None)
                if lock is None:
                    lock = threading.RLock()
                    self._toolbox_cancel_lock = lock
                active = getattr(self, "_toolbox_cancel_operations", None)
                if active is None:
                    active = set()
                    self._toolbox_cancel_operations = active
                with lock:
                    if operation.operation_id in active:
                        return self._hosted_operations.status(ref=operation, owner_actor_id=owner)
                    active.add(operation.operation_id)
                try:
                    self._hosted_operations.update_progress(
                        operation_id=operation.operation_id,
                        progress=HostedOperationProgress(
                            phase="cancellation",
                            code="toolbox_cancellation_requested",
                            completed_units=0,
                            total_units=1,
                            updated_at_ms=int(time.time() * 1000),
                            summary="Cancellation was acknowledged; worker teardown is continuing.",
                            cancellable=False,
                        ),
                    )
                except Exception:
                    pass

                def _cancel_in_background() -> None:
                    try:
                        self._cancel_toolbox_operation(
                            record=record,
                            reason=str(reason or "client_requested"),
                            timeout_seconds=float(timeout_seconds or 8.0),
                            respawn=bool(respawn),
                        )
                    finally:
                        with lock:
                            active.discard(operation.operation_id)

                threading.Thread(
                    target=_cancel_in_background,
                    name=f"toolbox-cancel-{operation.operation_id[:12]}",
                    daemon=True,
                ).start()
                return self._hosted_operations.status(ref=operation, owner_actor_id=owner)
            return self._cancel_toolbox_operation(
                record=record,
                reason=str(reason or "client_requested"),
                timeout_seconds=float(timeout_seconds or 8.0),
                respawn=bool(respawn),
            )
        if operation.execution_kind in {
            HostedExecutionKind.ENVIRONMENT_TEMPLATE_PREWARM,
            HostedExecutionKind.TOOLBOX_DEFINITION_PLAN,
            HostedExecutionKind.TOOLBOX_DEFINITION_PLAN_REVISION,
            HostedExecutionKind.TOOLBOX_DEFINITION_CONFIRMATION,
        }:
            canceled = self._hosted_operations.cancel_before_dispatch(
                operation_id=operation.operation_id,
                reason=str(reason or "client_requested"),
            )
            if canceled is not None:
                return canceled
            return self._hosted_operations.status(ref=operation, owner_actor_id=owner)
        if operation.execution_kind == HostedExecutionKind.TOOLBOX_DEFINITION_APPLY:
            cleanup = getattr(self, "_cleanup_toolbox_definition_apply_candidates", None)

            def cancellation_envelope() -> Dict[str, Any]:
                cleanup_diagnostics: Mapping[str, Any] = {
                    "status": "not_required",
                    "candidate_count": 0,
                }
                if callable(cleanup):
                    cleanup_diagnostics = dict(cleanup(record=record) or {})
                return {
                    "contract": "hosting.toolbox.definition_apply_result",
                    "status": "canceled",
                    "code": "apply_canceled_before_publication",
                    "diagnostics": {"candidate_cleanup": dict(cleanup_diagnostics)},
                }

            return self._hosted_operations.cancel_before_progress_commit(
                operation_id=operation.operation_id,
                committed_phases=tuple(sorted(TOOLBOX_DEFINITION_APPLY_COMMITTED_PHASES)),
                reason=str(reason or "client_requested"),
                envelope_factory=cancellation_envelope,
            )
        if operation.execution_kind == HostedExecutionKind.TOOLBOX_MAINTENANCE:
            return self._hosted_operations.cancel_before_progress_commit(
                operation_id=operation.operation_id,
                committed_phases=("recovery", "repair", "gc", "cleanup"),
                reason=str(reason or "client_requested"),
                envelope_factory=lambda: {
                    "contract": "hosting.toolbox.maintenance_result.v1",
                    "status": "canceled",
                    "code": "toolbox_maintenance_canceled_before_mutation",
                },
                committed_reason="toolbox_maintenance_mutation_started",
            )
        if operation.execution_kind == HostedExecutionKind.TOOLBOX_DESCRIBE_REFRESH:
            return self._hosted_operations.cancel_before_progress_commit(
                operation_id=operation.operation_id,
                committed_phases=("refresh", "cleanup"),
                reason=str(reason or "client_requested"),
                envelope_factory=lambda: {
                    "contract": "hosting.toolbox.describe_refresh_result.v1",
                    "status": "canceled",
                    "code": "toolbox_describe_refresh_canceled",
                },
                committed_reason="toolbox_describe_refresh_started",
            )
        if operation.execution_kind in {
            HostedExecutionKind.TOOLBOX_ARTIFACT_IMPORT,
            HostedExecutionKind.ENVIRONMENT_REMOVE,
            HostedExecutionKind.ENVIRONMENT_TEMPLATE_CONSTRUCT,
        }:
            return self._hosted_operations.status(ref=operation, owner_actor_id=owner)
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
