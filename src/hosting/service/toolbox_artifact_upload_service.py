from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from ..operation_contract import (
    HostedExecutionKind,
    HostedOperationLifecycle,
    HostedOperationProgress,
    HostedOperationSelector,
    hosted_execution_fingerprint,
)
from ..toolbox.host_project_config import ToolboxHostProjectConfiguration
from .toolbox_artifact_uploads import (
    AtomicToolboxArtifactUploadRepository,
    ToolboxArtifactUploadError,
)


class ToolboxArtifactUploadMixin:
    @property
    def _toolbox_artifact_upload_repository(self) -> AtomicToolboxArtifactUploadRepository:
        configuration = getattr(self, "_toolbox_host_project_config", None)
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        repository = getattr(self, "_toolbox_artifact_upload_repository_instance", None)
        if repository is None:
            repository = AtomicToolboxArtifactUploadRepository(
                self.hosting_root / "toolbox_artifact_uploads",
                configuration=configuration,
            )
            self._toolbox_artifact_upload_repository_instance = repository
        return repository

    def toolbox_artifact_upload_begin(
        self,
        *,
        source_id: str,
        total_size: int,
        archive_sha256: str,
        request_id: str,
        owner_actor_id: str,
    ) -> dict[str, Any]:
        return self._toolbox_artifact_upload_repository.begin(
            owner_actor_id=owner_actor_id,
            request_id=request_id,
            source_id=source_id,
            total_size=total_size,
            archive_sha256=archive_sha256,
        )

    def toolbox_artifact_upload_chunk(
        self,
        *,
        upload_id: str,
        chunk_index: int,
        offset: int,
        chunk_base64url: str,
        owner_actor_id: str,
    ) -> dict[str, Any]:
        return self._toolbox_artifact_upload_repository.append_chunk(
            owner_actor_id=owner_actor_id,
            upload_id=upload_id,
            chunk_index=chunk_index,
            offset=offset,
            chunk_base64url=chunk_base64url,
        )

    def toolbox_artifact_upload_status(
        self, *, upload_id: str, owner_actor_id: str
    ) -> dict[str, Any]:
        return self._toolbox_artifact_upload_repository.status(
            owner_actor_id=owner_actor_id, upload_id=upload_id
        )

    def toolbox_artifact_upload_cancel(
        self, *, upload_id: str, owner_actor_id: str
    ) -> dict[str, Any]:
        return self._toolbox_artifact_upload_repository.cancel(
            owner_actor_id=owner_actor_id, upload_id=upload_id
        )

    def toolbox_artifact_upload_commit(
        self, *, upload_id: str, request_id: str, owner_actor_id: str
    ) -> dict[str, Any]:
        upload = self._toolbox_artifact_upload_repository.reserve_commit(
            owner_actor_id=owner_actor_id,
            upload_id=upload_id,
            request_id=request_id,
        )
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.TOOLBOX_ARTIFACT_IMPORT.value,
                "upload_id": upload["upload_id"],
                "source_id": upload["source_id"],
                "config_revision": upload["config_revision"],
                "source_set_revision": upload["source_set_revision"],
                "target": upload["target"],
                "archive_sha256": upload["archive_sha256"],
                "total_size": upload["total_size"],
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id=owner_actor_id,
            execution_kind=HostedExecutionKind.TOOLBOX_ARTIFACT_IMPORT,
            selector=HostedOperationSelector(kind="upload_id", id=upload_id),
            namespace=f"toolbox_artifact_import:{upload_id}",
            request_id=request_id,
            fingerprint=fingerprint,
            metadata={
                "source_id": upload["source_id"],
                "config_revision": upload["config_revision"],
                "source_set_revision": upload["source_set_revision"],
                "target": upload["target"],
            },
        )
        status = prepared.get("status")
        if status is None:
            raise RuntimeError("hosted_operation_capacity")
        operation_id = str(status["operation"]["operation_id"])
        self._toolbox_artifact_upload_repository.bind_commit_operation(
            owner_actor_id=owner_actor_id,
            upload_id=upload_id,
            request_id=request_id,
            operation_id=operation_id,
        )
        if prepared["action"] != "dispatch":
            current = dict(status)
            if current.get("lifecycle") == (
                HostedOperationLifecycle.INTERRUPTED_AFTER_DISPATCH_UNKNOWN.value
            ):
                return self._reconcile_interrupted_artifact_import(
                    owner_actor_id=owner_actor_id,
                    upload_id=upload_id,
                    operation_id=operation_id,
                )
            return current
        thread = threading.Thread(
            target=self._run_toolbox_artifact_import,
            kwargs={
                "owner_actor_id": owner_actor_id,
                "upload_id": upload_id,
                "operation_id": operation_id,
            },
            name=f"toolbox-artifact-import-{operation_id[-8:]}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            failure = {
                "status": "error",
                "code": "artifact_upload_dispatch_failed",
                "summary": "The artifact import worker could not be started.",
            }
            self._toolbox_artifact_upload_repository.finish_commit(
                owner_actor_id=owner_actor_id,
                upload_id=upload_id,
                operation_id=operation_id,
                success=False,
                result=failure,
            )
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope=failure,
                reason=failure["code"],
            )
            raise
        return dict(status)

    def _artifact_import_progress(
        self,
        *,
        operation_id: str,
        phase: str,
        code: str,
        completed_units: int | None,
        total_units: int | None,
        summary: str,
    ) -> None:
        self._hosted_operations.update_progress(
            operation_id=operation_id,
            progress=HostedOperationProgress(
                phase=phase,
                code=code,
                completed_units=completed_units,
                total_units=total_units,
                updated_at_ms=int(time.time() * 1000),
                summary=summary,
                cancellable=False,
            ),
        )

    def _run_toolbox_artifact_import(
        self, *, owner_actor_id: str, upload_id: str, operation_id: str
    ) -> None:
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        try:
            path, upload = self._toolbox_artifact_upload_repository.staged_path_for_commit(
                owner_actor_id=owner_actor_id,
                upload_id=upload_id,
                operation_id=operation_id,
            )
            self._artifact_import_progress(
                operation_id=operation_id,
                phase="validation",
                code="artifact_upload_digest_verifying",
                completed_units=0,
                total_units=upload["total_size"],
                summary="The complete staged archive digest is being verified.",
            )
            hasher = hashlib.sha256()
            completed = 0
            with Path(path).open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
                    completed += len(chunk)
                    self._artifact_import_progress(
                        operation_id=operation_id,
                        phase="validation",
                        code="artifact_upload_digest_verifying",
                        completed_units=completed,
                        total_units=upload["total_size"],
                        summary="The complete staged archive digest is being verified.",
                    )
            if completed != upload["total_size"] or (
                f"sha256:{hasher.hexdigest()}" != upload["archive_sha256"]
            ):
                raise ToolboxArtifactUploadError("artifact_upload_chunk_invalid")
            self._artifact_import_progress(
                operation_id=operation_id,
                phase="artifact_verification",
                code="artifact_bundle_verifying",
                completed_units=0,
                total_units=1,
                summary="The signed bundle and complete wheel closure are being verified.",
            )
            imported = self._toolbox_artifact_store.import_signed_bundle(
                path,
                configuration=self._toolbox_host_project_config,
                trust_public_keys=self._toolbox_trust_public_keys,
                expected_source_id=upload["source_id"],
            )
            self._artifact_import_progress(
                operation_id=operation_id,
                phase="publication",
                code="artifact_bundle_published",
                completed_units=1,
                total_units=1,
                summary="The verified bundle is atomically visible in the artifact store.",
            )
            result = {
                "status": "ok",
                "code": "artifact_upload_committed",
                "upload_id": upload_id,
                "bundle_id": imported["bundle_id"],
                "manifest_digest": imported["manifest_digest"],
                "artifact_digests": imported["artifact_digests"],
            }
            self._toolbox_artifact_upload_repository.finish_commit(
                owner_actor_id=owner_actor_id,
                upload_id=upload_id,
                operation_id=operation_id,
                success=True,
                result=result,
            )
            self._artifact_import_progress(
                operation_id=operation_id,
                phase="cleanup",
                code="artifact_upload_stage_removed",
                completed_units=1,
                total_units=1,
                summary="The committed untrusted stage file was removed.",
            )
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope=result,
            )
        except Exception as exc:
            try:
                upload = self._toolbox_artifact_upload_repository.status(
                    owner_actor_id=owner_actor_id, upload_id=upload_id
                )
            except Exception:
                upload = {}
            if upload.get("state") == "committed" and isinstance(upload.get("result"), dict):
                self._hosted_operations.finish(
                    operation_id=operation_id,
                    lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                    envelope=upload["result"],
                )
                return
            candidate = str(getattr(exc, "code", "") or str(exc)).strip()
            code = candidate if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", candidate) else "artifact_upload_commit_failed"
            failure = {
                "status": "error",
                "code": code,
                "summary": "The staged artifact bundle failed verification before publication.",
            }
            try:
                self._toolbox_artifact_upload_repository.finish_commit(
                    owner_actor_id=owner_actor_id,
                    upload_id=upload_id,
                    operation_id=operation_id,
                    success=False,
                    result=failure,
                )
            finally:
                self._hosted_operations.finish(
                    operation_id=operation_id,
                    lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                    envelope=failure,
                    reason=code,
                )

    def _reconcile_interrupted_artifact_import(
        self, *, owner_actor_id: str, upload_id: str, operation_id: str
    ) -> dict[str, Any]:
        upload = self._toolbox_artifact_upload_repository.status(
            owner_actor_id=owner_actor_id, upload_id=upload_id
        )
        if upload["state"] == "committed" and isinstance(upload.get("result"), dict):
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope=upload["result"],
            )
        failure = {
            "status": "error",
            "code": "artifact_upload_interrupted_after_dispatch",
            "summary": "Artifact import was interrupted without a durable committed result.",
        }
        self._toolbox_artifact_upload_repository.finish_commit(
            owner_actor_id=owner_actor_id,
            upload_id=upload_id,
            operation_id=operation_id,
            success=False,
            result=failure,
        )
        return self._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
            envelope=failure,
            reason=failure["code"],
        )


__all__ = ["ToolboxArtifactUploadMixin"]
