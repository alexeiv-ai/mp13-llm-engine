"""Atomic JSON repository for durable hosted-operation truth."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Mapping, Optional, Protocol, Sequence

from ..operation_contract import (
    HOSTED_OPERATION_REF_CONTRACT,
    MAX_INLINE_RESULT_BYTES,
    HostedExecutionKind,
    HostedOperationLifecycle,
    HostedOperationProgress,
    HostedOperationRef,
    HostedOperationSelector,
    HostedOperationStatus,
    HostedResultOmission,
    HostedResultRef,
    TERMINAL_OPERATION_LIFECYCLES,
    canonical_json_bytes,
    canonical_sha256_digest,
)
from .result_artifacts import ResultArtifactError, TerminalResultArtifactStore


OPERATION_REPOSITORY_CONTRACT = "hosting.operation_repository"
MAX_METADATA_BYTES = 8 * 1024
MAX_OWNER_ACTOR_ID_BYTES = 256
PROCESS_LOCK_TIMEOUT_SECONDS = 30.0
WINDOWS_REPLACE_RETRY_ATTEMPTS = 8
WINDOWS_REPLACE_RETRY_DELAY_SECONDS = 0.05
_SECRET_KEY = re.compile(
    r"(?:authorization|credential|password|passwd|secret|session[_-]?token|access[_-]?token|refresh[_-]?token|api[_-]?key|private[_-]?key)$",
    re.IGNORECASE,
)


class LegacyOperationRepositoryError(RuntimeError):
    """Raised when a legacy or otherwise unsupported checkpoint is present."""


class HostedOperationRepository(Protocol):
    def prepare(
        self,
        *,
        owner_actor_id: str,
        execution_kind: HostedExecutionKind | str,
        selector: HostedOperationSelector | Mapping[str, Any],
        namespace: str,
        request_id: str,
        fingerprint: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    def mark_dispatch_claimed(self, *, operation_id: str) -> Dict[str, Any]: ...

    def update_progress(
        self,
        *,
        operation_id: str,
        progress: HostedOperationProgress | Mapping[str, Any],
    ) -> Dict[str, Any]: ...

    def finish(
        self,
        *,
        operation_id: str,
        lifecycle: HostedOperationLifecycle | str,
        envelope: Mapping[str, Any],
        reason: str = "",
    ) -> Dict[str, Any]: ...

    def cancel_before_dispatch(self, *, operation_id: str, reason: str) -> Optional[Dict[str, Any]]: ...

    def merge_metadata(self, *, operation_id: str, metadata: Mapping[str, Any]) -> Dict[str, Any]: ...

    def cancel_before_progress_commit(
        self,
        *,
        operation_id: str,
        committed_phases: Sequence[str],
        reason: str,
        envelope_factory: Callable[[], Mapping[str, Any]],
        committed_reason: str = "apply_publication_committed",
    ) -> Dict[str, Any]: ...

    def requeue_interrupted_after_dispatch(self, *, operation_id: str) -> Optional[Dict[str, Any]]: ...

    def status(self, *, ref: HostedOperationRef | Mapping[str, Any], owner_actor_id: str) -> Dict[str, Any]: ...

    def get_by_operation_id(self, operation_id: str) -> Optional[Dict[str, Any]]: ...

    def active_records(self, *, execution_kind: HostedExecutionKind | str | None = None) -> list[Dict[str, Any]]: ...

    def get_by_request(self, *, owner_actor_id: str, namespace: str, request_id: str) -> Optional[Dict[str, Any]]: ...

    def wait_for_terminal(self, *, operation_id: str, timeout_seconds: float) -> Dict[str, Any]: ...

    def prune(self) -> None: ...


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _bounded_identity(value: Any, *, label: str, max_bytes: int = 256) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label}_required")
    if any(ord(char) < 32 for char in text):
        raise ValueError(f"{label}_contains_control_characters")
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError(f"{label}_too_large")
    return text


@contextlib.contextmanager
def _exclusive_process_file_lock(
    path: Path,
    *,
    timeout_seconds: float = PROCESS_LOCK_TIMEOUT_SECONDS,
) -> Generator[None, None, None]:
    """Acquire an inter-process lock using a stable sidecar lock file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        locked = False
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            deadline = time.monotonic() + max(0.1, float(timeout_seconds))
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out acquiring hosted operation repository lock: {path}")
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        try:
            yield
        finally:
            if locked:
                if sys.platform == "win32":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _replace_with_bounded_retries(source: Path, target: Path) -> None:
    attempts = WINDOWS_REPLACE_RETRY_ATTEMPTS if sys.platform == "win32" else 1
    last_error: Optional[OSError] = None
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise
            time.sleep(WINDOWS_REPLACE_RETRY_DELAY_SECONDS * (attempt + 1))
    if last_error is not None:
        raise last_error


class AtomicJsonHostedOperationRepository:
    """Process-locked, bounded, atomic hosted-operation repository."""

    def __init__(
        self,
        path: Path,
        *,
        receipt_retention_seconds: float = 7 * 24 * 3600,
        tombstone_retention_seconds: float = 14 * 24 * 3600,
        max_receipts: int = 10_000,
        max_tombstones: int = 20_000,
        max_inline_result_bytes: int = MAX_INLINE_RESULT_BYTES,
        result_artifact_store: Optional[TerminalResultArtifactStore] = None,
        clock: Any = time.time,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.receipt_retention_ms = max(0, int(float(receipt_retention_seconds) * 1000))
        self.tombstone_retention_ms = max(0, int(float(tombstone_retention_seconds) * 1000))
        self.max_receipts = max(1, int(max_receipts))
        self.max_tombstones = max(1, int(max_tombstones))
        self.max_inline_result_bytes = max(256, min(int(max_inline_result_bytes), MAX_INLINE_RESULT_BYTES))
        self.result_artifact_store = result_artifact_store
        self._clock = clock
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._condition = threading.Condition(threading.RLock())
        self._data = self._new_data()
        self._request_index: Dict[tuple[str, str, str], str] = {}
        self._operation_index: Dict[str, tuple[str, bool]] = {}
        with self._condition:
            with _exclusive_process_file_lock(self._lock_path):
                self._data = self._load()
                self._rebuild_indexes_locked()
                changed = self._recover_interrupted_locked()
                changed = self._prune_locked() or changed
                if changed:
                    self._persist_locked()

    def _now_ms(self) -> int:
        return max(0, int(float(self._clock()) * 1000))

    @staticmethod
    def _request_key(owner_actor_id: str, namespace: str, request_id: str) -> tuple[str, str, str]:
        return owner_actor_id, namespace, request_id

    @staticmethod
    def _new_data() -> Dict[str, Any]:
        return {"contract": OPERATION_REPOSITORY_CONTRACT, "receipts": {}, "tombstones": {}}

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._new_data()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"hosted operation repository is unreadable: {self.path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"hosted operation repository is invalid: {self.path}")
        if payload.get("contract") != OPERATION_REPOSITORY_CONTRACT:
            if "version" in payload or payload.get("receipts") is not None:
                raise LegacyOperationRepositoryError(
                    "legacy hosted-operation receipt schema is unsupported; "
                    "run hosting-receipt-ledger-cutover after confirming the replay window is clear"
                )
            raise RuntimeError(f"hosted operation repository contract is invalid: {self.path}")
        if not isinstance(payload.get("receipts"), dict) or not isinstance(payload.get("tombstones"), dict):
            raise RuntimeError(f"hosted operation repository collections are invalid: {self.path}")
        return {
            "contract": OPERATION_REPOSITORY_CONTRACT,
            "receipts": dict(payload["receipts"]),
            "tombstones": dict(payload["tombstones"]),
        }

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            encoded = canonical_json_bytes(self._data)
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_bounded_retries(temporary, self.path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _reload_locked(self) -> None:
        self._data = self._load()
        self._rebuild_indexes_locked()

    @contextlib.contextmanager
    def _state_lock(self) -> Generator[None, None, None]:
        with self._condition:
            with _exclusive_process_file_lock(self._lock_path):
                self._reload_locked()
                yield

    def _validate_row(self, operation_id: str, row: Mapping[str, Any], *, tombstone: bool) -> HostedOperationRef:
        record = dict(row or {})
        operation = HostedOperationRef.from_dict(dict(record.get("operation") or {}))
        if operation.operation_id != operation_id:
            raise RuntimeError("hosted operation repository operation index mismatch")
        owner = _bounded_identity(record.get("owner_actor_id"), label="owner_actor_id", max_bytes=MAX_OWNER_ACTOR_ID_BYTES)
        del owner
        lifecycle = HostedOperationLifecycle(str(record.get("lifecycle") or ""))
        if tombstone and lifecycle != HostedOperationLifecycle.FORGOTTEN:
            raise RuntimeError("hosted operation tombstone lifecycle invalid")
        for name in ("created_at_ms", "updated_at_ms"):
            if int(record.get(name) or 0) < 0:
                raise RuntimeError(f"hosted operation {name} invalid")
        raw_progress = record.get("progress")
        if raw_progress is not None:
            if not isinstance(raw_progress, Mapping):
                raise RuntimeError("hosted operation progress invalid")
            progress = HostedOperationProgress.from_dict(raw_progress)
            if progress.updated_at_ms < int(record.get("created_at_ms") or 0):
                raise RuntimeError("hosted operation progress predates receipt")
            if progress.updated_at_ms > int(record.get("updated_at_ms") or 0):
                raise RuntimeError("hosted operation progress exceeds receipt update time")
        self._status_from_row(record)
        return operation

    def _rebuild_indexes_locked(self) -> None:
        request_index: Dict[tuple[str, str, str], str] = {}
        operation_index: Dict[str, tuple[str, bool]] = {}
        for collection_name, tombstone in (("receipts", False), ("tombstones", True)):
            for operation_id, raw in self._data[collection_name].items():
                operation = self._validate_row(str(operation_id), dict(raw or {}), tombstone=tombstone)
                row = dict(raw or {})
                key = self._request_key(str(row["owner_actor_id"]), operation.receipt_namespace, operation.request_id)
                if operation.operation_id in operation_index:
                    raise RuntimeError("hosted operation repository duplicate operation_id")
                if key in request_index:
                    raise RuntimeError("hosted operation repository duplicate request identity")
                operation_index[operation.operation_id] = (collection_name, tombstone)
                request_index[key] = operation.operation_id
        self._request_index = request_index
        self._operation_index = operation_index

    def _recover_interrupted_locked(self) -> bool:
        changed = False
        now_ms = self._now_ms()
        for row in self._data["receipts"].values():
            lifecycle = HostedOperationLifecycle(str(row.get("lifecycle") or ""))
            if lifecycle == HostedOperationLifecycle.QUEUED:
                row["lifecycle"] = HostedOperationLifecycle.INTERRUPTED_BEFORE_DISPATCH.value
                row["updated_at_ms"] = now_ms
                changed = True
            elif lifecycle == HostedOperationLifecycle.RUNNING:
                row["lifecycle"] = HostedOperationLifecycle.INTERRUPTED_AFTER_DISPATCH_UNKNOWN.value
                row["updated_at_ms"] = now_ms
                changed = True
        return changed

    @staticmethod
    def _public_record(row: Mapping[str, Any]) -> Dict[str, Any]:
        public = copy.deepcopy(dict(row or {}))
        public.pop("owner_actor_id", None)
        public.pop("metadata", None)
        return public

    @staticmethod
    def _status_from_row(
        row: Mapping[str, Any],
        *,
        lifecycle: Optional[HostedOperationLifecycle] = None,
        api_status: str = "ok",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = dict(row or {})
        terminal = dict(record.get("terminal") or {})
        result_ref = terminal.get("result_ref")
        result_omission = terminal.get("result_omission")
        model = HostedOperationStatus(
            operation=HostedOperationRef.from_dict(dict(record.get("operation") or {})),
            lifecycle=lifecycle or HostedOperationLifecycle(str(record.get("lifecycle") or "")),
            request_id=str(dict(record.get("operation") or {}).get("request_id") or ""),
            created_at_ms=int(record.get("created_at_ms") or 0),
            updated_at_ms=int(record.get("updated_at_ms") or 0),
            api_status=api_status,
            dispatch_claimed_at_ms=(
                int(record["dispatch_claimed_at_ms"])
                if record.get("dispatch_claimed_at_ms") is not None
                else None
            ),
            terminal_at_ms=int(record["terminal_at_ms"]) if record.get("terminal_at_ms") is not None else None,
            reason=reason if reason is not None else record.get("reason"),
            result=copy.deepcopy(terminal.get("result")),
            result_ref=HostedResultRef.from_dict(result_ref) if isinstance(result_ref, dict) else None,
            result_omission=HostedResultOmission.from_dict(result_omission) if isinstance(result_omission, dict) else None,
            progress=(
                HostedOperationProgress.from_dict(record["progress"])
                if isinstance(record.get("progress"), Mapping)
                else None
            ),
        )
        return model.to_dict()

    def _unknown_status(self, ref: HostedOperationRef) -> Dict[str, Any]:
        return HostedOperationStatus(
            operation=ref,
            lifecycle=HostedOperationLifecycle.UNKNOWN_OUTSIDE_RETENTION,
            request_id=ref.request_id,
            created_at_ms=0,
            updated_at_ms=0,
            api_status="error",
            reason="operation_not_found",
        ).to_dict()

    def _forget_locked(self, operation_id: str, row: Dict[str, Any], now_ms: int) -> None:
        result_ref = dict(dict(row.get("terminal") or {}).get("result_ref") or {})
        if self.result_artifact_store is not None and result_ref.get("artifact_id"):
            try:
                self.result_artifact_store.delete(str(result_ref["artifact_id"]))
            except Exception:
                pass
        self._data["receipts"].pop(operation_id, None)
        forgotten = copy.deepcopy(row)
        forgotten["lifecycle"] = HostedOperationLifecycle.FORGOTTEN.value
        forgotten["updated_at_ms"] = now_ms
        forgotten["forgotten_at_ms"] = now_ms
        forgotten["expires_at_ms"] = now_ms + self.tombstone_retention_ms
        forgotten.pop("terminal", None)
        self._data["tombstones"][operation_id] = forgotten

    def _prune_locked(self) -> bool:
        now_ms = self._now_ms()
        changed = False
        receipts: Dict[str, Dict[str, Any]] = self._data["receipts"]
        eligible = sorted(
            (
                (int(row.get("updated_at_ms") or row.get("created_at_ms") or 0), operation_id, row)
                for operation_id, row in receipts.items()
                if HostedOperationLifecycle(str(row.get("lifecycle") or ""))
                not in {HostedOperationLifecycle.QUEUED, HostedOperationLifecycle.RUNNING}
            ),
            key=lambda item: (item[0], item[1]),
        )
        for updated_at_ms, operation_id, row in eligible:
            if now_ms - updated_at_ms >= self.receipt_retention_ms:
                self._forget_locked(operation_id, row, now_ms)
                changed = True
        overflow = max(0, len(receipts) - self.max_receipts)
        if overflow:
            remaining = sorted(
                (
                    (int(row.get("updated_at_ms") or row.get("created_at_ms") or 0), operation_id, row)
                    for operation_id, row in receipts.items()
                    if HostedOperationLifecycle(str(row.get("lifecycle") or ""))
                    not in {HostedOperationLifecycle.QUEUED, HostedOperationLifecycle.RUNNING}
                ),
                key=lambda item: (item[0], item[1]),
            )
            for _, operation_id, row in remaining[:overflow]:
                self._forget_locked(operation_id, row, now_ms)
                changed = True
        tombstones: Dict[str, Dict[str, Any]] = self._data["tombstones"]
        for operation_id, row in list(tombstones.items()):
            if now_ms >= int(row.get("expires_at_ms") or 0):
                tombstones.pop(operation_id, None)
                changed = True
        if len(tombstones) > self.max_tombstones:
            ordered = sorted(
                tombstones.items(), key=lambda item: (int(item[1].get("forgotten_at_ms") or 0), item[0])
            )
            for operation_id, _ in ordered[: len(tombstones) - self.max_tombstones]:
                tombstones.pop(operation_id, None)
                changed = True
        if changed:
            self._rebuild_indexes_locked()
        return changed

    def prepare(
        self,
        *,
        owner_actor_id: str,
        execution_kind: HostedExecutionKind | str,
        selector: HostedOperationSelector | Mapping[str, Any],
        namespace: str,
        request_id: str,
        fingerprint: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        owner = _bounded_identity(owner_actor_id, label="owner_actor_id", max_bytes=MAX_OWNER_ACTOR_ID_BYTES)
        kind = execution_kind if isinstance(execution_kind, HostedExecutionKind) else HostedExecutionKind(str(execution_kind))
        operation_selector = (
            selector if isinstance(selector, HostedOperationSelector) else HostedOperationSelector.from_dict(selector)
        )
        ns = _bounded_identity(namespace, label="receipt_namespace")
        rid = _bounded_identity(request_id, label="request_id")
        digest = canonical_sha256_digest(fingerprint, label="operation_fingerprint")
        safe_metadata = _redact(copy.deepcopy(dict(metadata or {})))
        if len(canonical_json_bytes(safe_metadata)) > MAX_METADATA_BYTES:
            raise ValueError("operation_metadata_too_large")
        key = self._request_key(owner, ns, rid)
        now_ms = self._now_ms()
        with self._state_lock():
            changed = self._prune_locked()
            existing_id = self._request_index.get(key)
            if existing_id:
                collection_name, tombstone = self._operation_index[existing_id]
                row = self._data[collection_name][existing_id]
                operation = HostedOperationRef.from_dict(dict(row["operation"]))
                if operation.fingerprint != digest:
                    if changed:
                        self._persist_locked()
                    return {
                        "action": "conflict",
                        "status": self._status_from_row(
                            row,
                            lifecycle=HostedOperationLifecycle.IDEMPOTENCY_CONFLICT,
                            api_status="error",
                            reason="idempotency_conflict",
                        ),
                    }
                if tombstone:
                    if changed:
                        self._persist_locked()
                    return {"action": "forgotten", "status": self._status_from_row(row)}
                lifecycle = HostedOperationLifecycle(str(row["lifecycle"]))
                if lifecycle in TERMINAL_OPERATION_LIFECYCLES:
                    if changed:
                        self._persist_locked()
                    return {"action": "replay", "status": self._status_from_row(row)}
                if lifecycle == HostedOperationLifecycle.INTERRUPTED_BEFORE_DISPATCH:
                    row["lifecycle"] = HostedOperationLifecycle.QUEUED.value
                    row["updated_at_ms"] = now_ms
                    row["resume_count"] = min(1, int(row.get("resume_count") or 0) + 1)
                    self._persist_locked()
                    return {"action": "dispatch", "status": self._status_from_row(row)}
                if changed:
                    self._persist_locked()
                return {"action": "attach", "status": self._status_from_row(row)}
            if len(self._data["receipts"]) >= self.max_receipts:
                eligible = sorted(
                    (
                        (int(row.get("updated_at_ms") or row.get("created_at_ms") or 0), operation_id, row)
                        for operation_id, row in self._data["receipts"].items()
                        if HostedOperationLifecycle(str(row.get("lifecycle") or ""))
                        not in {HostedOperationLifecycle.QUEUED, HostedOperationLifecycle.RUNNING}
                    ),
                    key=lambda item: (item[0], item[1]),
                )
                if eligible:
                    _, oldest_id, oldest = eligible[0]
                    self._forget_locked(oldest_id, oldest, now_ms)
                    self._rebuild_indexes_locked()
                    self._prune_locked()
                    changed = True
                if len(self._data["receipts"]) >= self.max_receipts:
                    if changed:
                        self._persist_locked()
                    return {"action": "capacity", "status": None}
            operation = HostedOperationRef(
                operation_id=f"op_{secrets.token_urlsafe(24)}",
                request_id=rid,
                execution_kind=kind,
                selector=operation_selector,
                fingerprint=digest,
                receipt_namespace=ns,
            )
            row = {
                "operation": operation.to_dict(),
                "owner_actor_id": owner,
                "lifecycle": HostedOperationLifecycle.QUEUED.value,
                "created_at_ms": now_ms,
                "updated_at_ms": now_ms,
                "dispatch_claimed_at_ms": None,
                "terminal_at_ms": None,
                "reason": None,
                "resume_count": 0,
                "metadata": safe_metadata,
                "terminal": {},
            }
            self._data["receipts"][operation.operation_id] = row
            self._rebuild_indexes_locked()
            self._persist_locked()
            return {"action": "dispatch", "status": self._status_from_row(row)}

    def mark_dispatch_claimed(self, *, operation_id: str) -> Dict[str, Any]:
        oid = _bounded_identity(operation_id, label="operation_id")
        now_ms = self._now_ms()
        with self._state_lock():
            location = self._operation_index.get(oid)
            if location is None or location[1]:
                raise KeyError(oid)
            row = self._data[location[0]][oid]
            if HostedOperationLifecycle(str(row["lifecycle"])) == HostedOperationLifecycle.QUEUED:
                row["lifecycle"] = HostedOperationLifecycle.RUNNING.value
                row["dispatch_claimed_at_ms"] = now_ms
                row["updated_at_ms"] = now_ms
                self._persist_locked()
                self._condition.notify_all()
            return self._status_from_row(row)

    def merge_metadata(self, *, operation_id: str, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        oid = _bounded_identity(operation_id, label="operation_id")
        updates = _redact(copy.deepcopy(dict(metadata or {})))
        with self._state_lock():
            location = self._operation_index.get(oid)
            if location is None or location[1]:
                raise KeyError(oid)
            row = self._data[location[0]][oid]
            lifecycle = HostedOperationLifecycle(str(row["lifecycle"]))
            if lifecycle in TERMINAL_OPERATION_LIFECYCLES:
                return self._status_from_row(row)
            merged = {**dict(row.get("metadata") or {}), **updates}
            if len(canonical_json_bytes(merged)) > MAX_METADATA_BYTES:
                raise ValueError("operation_metadata_too_large")
            row["metadata"] = merged
            row["updated_at_ms"] = self._now_ms()
            self._persist_locked()
            return self._status_from_row(row)

    def update_progress(
        self,
        *,
        operation_id: str,
        progress: HostedOperationProgress | Mapping[str, Any],
    ) -> Dict[str, Any]:
        oid = _bounded_identity(operation_id, label="operation_id")
        model = (
            progress
            if isinstance(progress, HostedOperationProgress)
            else HostedOperationProgress.from_dict(progress)
        )
        now_ms = self._now_ms()
        if model.updated_at_ms > now_ms:
            raise ValueError("operation_progress_future_timestamp")
        with self._state_lock():
            location = self._operation_index.get(oid)
            if location is None or location[1]:
                raise KeyError(oid)
            row = self._data[location[0]][oid]
            lifecycle = HostedOperationLifecycle(str(row["lifecycle"]))
            if lifecycle not in {HostedOperationLifecycle.QUEUED, HostedOperationLifecycle.RUNNING}:
                raise ValueError("operation_progress_terminal_update_denied")
            previous = row.get("progress")
            if isinstance(previous, Mapping):
                previous_model = HostedOperationProgress.from_dict(previous)
                if model.updated_at_ms < previous_model.updated_at_ms:
                    raise ValueError("operation_progress_timestamp_regression")
                if not previous_model.cancellable and model.cancellable:
                    raise ValueError("operation_progress_cancellation_boundary_regression")
            row["progress"] = model.to_dict()
            row["updated_at_ms"] = max(
                int(row.get("updated_at_ms") or 0),
                int(model.updated_at_ms),
                now_ms,
            )
            status = self._status_from_row(row)
            self._persist_locked()
            self._condition.notify_all()
            return status

    def _bounded_terminal(self, envelope: Mapping[str, Any], *, row: Mapping[str, Any]) -> Dict[str, Any]:
        redacted = _redact(copy.deepcopy(dict(envelope or {})))
        encoded = canonical_json_bytes(redacted)
        digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        if len(encoded) <= self.max_inline_result_bytes:
            return {"result": redacted, "digest": digest, "size_bytes": len(encoded)}
        retain = bool(dict(row.get("metadata") or {}).get("retain_terminal_result"))
        if retain and self.result_artifact_store is not None:
            try:
                result_ref = self.result_artifact_store.put(
                    owner_actor_id=str(row.get("owner_actor_id") or ""),
                    operation_id=str(dict(row.get("operation") or {}).get("operation_id") or ""),
                    content=encoded,
                )
                return {"result_ref": result_ref.to_dict(), "digest": digest, "size_bytes": len(encoded)}
            except ResultArtifactError as exc:
                omission_reason = str(exc)
            except Exception:
                omission_reason = "result_artifact_store_failed"
        else:
            omission_reason = "retention_not_permitted"
        omission = HostedResultOmission(
            digest=digest,
            size_bytes=len(encoded),
            reason=omission_reason,
        )
        return {"result_omission": omission.to_dict(), "digest": digest, "size_bytes": len(encoded)}

    def finish(
        self,
        *,
        operation_id: str,
        lifecycle: HostedOperationLifecycle | str,
        envelope: Mapping[str, Any],
        reason: str = "",
    ) -> Dict[str, Any]:
        oid = _bounded_identity(operation_id, label="operation_id")
        terminal_lifecycle = (
            lifecycle if isinstance(lifecycle, HostedOperationLifecycle) else HostedOperationLifecycle(str(lifecycle))
        )
        if terminal_lifecycle not in TERMINAL_OPERATION_LIFECYCLES:
            raise ValueError("operation_terminal_lifecycle_invalid")
        terminal_reason = str(reason or "").strip() or None
        now_ms = self._now_ms()
        with self._state_lock():
            location = self._operation_index.get(oid)
            if location is None or location[1]:
                raise KeyError(oid)
            row = self._data[location[0]][oid]
            if HostedOperationLifecycle(str(row["lifecycle"])) in TERMINAL_OPERATION_LIFECYCLES:
                return self._status_from_row(row)
            row["lifecycle"] = terminal_lifecycle.value
            row["terminal_at_ms"] = now_ms
            row["updated_at_ms"] = now_ms
            row["reason"] = terminal_reason
            row["terminal"] = self._bounded_terminal(envelope, row=row)
            self._prune_locked()
            self._persist_locked()
            self._condition.notify_all()
            return self._status_from_row(row)

    def cancel_before_dispatch(self, *, operation_id: str, reason: str) -> Optional[Dict[str, Any]]:
        oid = _bounded_identity(operation_id, label="operation_id")
        with self._state_lock():
            location = self._operation_index.get(oid)
            if location is None or location[1]:
                return None
            row = self._data[location[0]][oid]
            lifecycle = HostedOperationLifecycle(str(row["lifecycle"]))
            if lifecycle not in {
                HostedOperationLifecycle.QUEUED,
                HostedOperationLifecycle.INTERRUPTED_BEFORE_DISPATCH,
            }:
                return None
        return self.finish(
            operation_id=oid,
            lifecycle=HostedOperationLifecycle.TERMINAL_CANCELLATION,
            envelope={"status": "canceled", "reason": str(reason or "canceled_before_dispatch")},
            reason=str(reason or "canceled_before_dispatch"),
        )

    def requeue_interrupted_after_dispatch(
        self, *, operation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Claim one idempotent restart retry for an interrupted durable worker."""

        oid = _bounded_identity(operation_id, label="operation_id")
        now_ms = self._now_ms()
        with self._state_lock():
            location = self._operation_index.get(oid)
            if location is None or location[1]:
                return None
            row = self._data[location[0]][oid]
            if HostedOperationLifecycle(str(row["lifecycle"])) != (
                HostedOperationLifecycle.INTERRUPTED_AFTER_DISPATCH_UNKNOWN
            ):
                return None
            row["lifecycle"] = HostedOperationLifecycle.QUEUED.value
            row["dispatch_claimed_at_ms"] = None
            row["progress"] = None
            row["updated_at_ms"] = now_ms
            row["resume_count"] = min(1, int(row.get("resume_count") or 0) + 1)
            self._persist_locked()
            self._condition.notify_all()
            return self._status_from_row(row)

    def cancel_before_progress_commit(
        self,
        *,
        operation_id: str,
        committed_phases: Sequence[str],
        reason: str,
        envelope_factory: Callable[[], Mapping[str, Any]],
        committed_reason: str = "apply_publication_committed",
    ) -> Dict[str, Any]:
        """Atomically cancel an operation only while its persisted progress is pre-commit.

        The cleanup callback runs while the same repository/process lock that
        guards progress is held. A concurrent publication checkpoint therefore
        wins wholly before cancellation, or observes the terminal cancellation
        and is rejected; the boundary cannot be split by a race.
        """

        oid = _bounded_identity(operation_id, label="operation_id")
        phases = frozenset(str(item or "").strip() for item in committed_phases)
        terminal_reason = str(reason or "client_requested").strip() or "client_requested"
        now_ms = self._now_ms()
        with self._state_lock():
            location = self._operation_index.get(oid)
            if location is None or location[1]:
                raise KeyError(oid)
            row = self._data[location[0]][oid]
            lifecycle = HostedOperationLifecycle(str(row["lifecycle"]))
            if lifecycle in TERMINAL_OPERATION_LIFECYCLES:
                return self._status_from_row(row)
            progress = row.get("progress")
            progress_model = HostedOperationProgress.from_dict(progress) if isinstance(progress, Mapping) else None
            if progress_model is not None and (
                progress_model.phase in phases or not progress_model.cancellable
            ):
                return self._status_from_row(
                    row,
                    api_status="error",
                    reason=str(committed_reason or "operation_commit_started"),
                )
            envelope = dict(envelope_factory() or {})
            row["lifecycle"] = HostedOperationLifecycle.TERMINAL_CANCELLATION.value
            row["terminal_at_ms"] = now_ms
            row["updated_at_ms"] = now_ms
            row["reason"] = terminal_reason
            row["terminal"] = self._bounded_terminal(envelope, row=row)
            self._prune_locked()
            self._persist_locked()
            self._condition.notify_all()
            return self._status_from_row(row)

    def get_by_operation_id(self, operation_id: str) -> Optional[Dict[str, Any]]:
        oid = str(operation_id or "").strip()
        with self._state_lock():
            location = self._operation_index.get(oid)
            if location is None:
                return None
            return copy.deepcopy(self._data[location[0]][oid])

    def active_records(self, *, execution_kind: HostedExecutionKind | str | None = None) -> list[Dict[str, Any]]:
        kind = (
            execution_kind.value
            if isinstance(execution_kind, HostedExecutionKind)
            else str(execution_kind or "").strip()
        )
        with self._state_lock():
            rows = []
            for row in self._data["receipts"].values():
                lifecycle = HostedOperationLifecycle(str(row.get("lifecycle") or ""))
                if lifecycle in TERMINAL_OPERATION_LIFECYCLES:
                    continue
                if kind and str(dict(row.get("operation") or {}).get("execution_kind") or "") != kind:
                    continue
                rows.append(copy.deepcopy(row))
            return sorted(rows, key=lambda item: str(dict(item.get("operation") or {}).get("operation_id") or ""))

    def get_by_request(self, *, owner_actor_id: str, namespace: str, request_id: str) -> Optional[Dict[str, Any]]:
        key = self._request_key(str(owner_actor_id or "").strip(), str(namespace or "").strip(), str(request_id or "").strip())
        with self._state_lock():
            operation_id = self._request_index.get(key)
            if not operation_id:
                return None
            location = self._operation_index.get(operation_id)
            if location is None:
                return None
            return copy.deepcopy(self._data[location[0]][operation_id])

    def _resolve_locked(
        self,
        *,
        operation: HostedOperationRef,
        owner_actor_id: str,
    ) -> Optional[Dict[str, Any]]:
        location = self._operation_index.get(operation.operation_id)
        if location is None:
            return None
        row = self._data[location[0]][operation.operation_id]
        stored = HostedOperationRef.from_dict(dict(row["operation"]))
        if str(row.get("owner_actor_id") or "") != owner_actor_id or stored != operation:
            return None
        return copy.deepcopy(row)

    def resolve(
        self,
        *,
        ref: HostedOperationRef | Mapping[str, Any],
        owner_actor_id: str,
    ) -> Optional[Dict[str, Any]]:
        operation = ref if isinstance(ref, HostedOperationRef) else HostedOperationRef.from_dict(ref)
        owner = _bounded_identity(owner_actor_id, label="owner_actor_id", max_bytes=MAX_OWNER_ACTOR_ID_BYTES)
        with self._state_lock():
            return self._resolve_locked(operation=operation, owner_actor_id=owner)

    def status(self, *, ref: HostedOperationRef | Mapping[str, Any], owner_actor_id: str) -> Dict[str, Any]:
        operation = ref if isinstance(ref, HostedOperationRef) else HostedOperationRef.from_dict(ref)
        owner = _bounded_identity(owner_actor_id, label="owner_actor_id", max_bytes=MAX_OWNER_ACTOR_ID_BYTES)
        with self._state_lock():
            changed = self._prune_locked()
            row = self._resolve_locked(operation=operation, owner_actor_id=owner)
            if changed:
                self._persist_locked()
            return self._status_from_row(row) if row is not None else self._unknown_status(operation)

    def wait_for_terminal(self, *, operation_id: str, timeout_seconds: float) -> Dict[str, Any]:
        oid = _bounded_identity(operation_id, label="operation_id")
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        with self._condition:
            while True:
                with _exclusive_process_file_lock(self._lock_path):
                    self._reload_locked()
                    location = self._operation_index.get(oid)
                    if location is None:
                        raise KeyError(oid)
                    row = self._data[location[0]][oid]
                    lifecycle = HostedOperationLifecycle(str(row["lifecycle"]))
                    if lifecycle not in {HostedOperationLifecycle.QUEUED, HostedOperationLifecycle.RUNNING}:
                        return self._status_from_row(row)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._status_from_row(row)
                self._condition.wait(timeout=min(remaining, 0.25))

    def prune(self) -> None:
        with self._state_lock():
            if self._prune_locked():
                self._persist_locked()
            if self.result_artifact_store is not None:
                live_ids = {
                    str(result_ref["artifact_id"])
                    for row in self._data["receipts"].values()
                    for result_ref in [dict(dict(row.get("terminal") or {}).get("result_ref") or {})]
                    if result_ref.get("artifact_id")
                }
                self.result_artifact_store.prune(live_artifact_ids=live_ids)

    def read_result(
        self,
        *,
        ref: HostedOperationRef | Mapping[str, Any],
        owner_actor_id: str,
    ) -> Dict[str, Any]:
        if self.result_artifact_store is None:
            raise ResultArtifactError("result_artifact_store_unavailable")
        operation = ref if isinstance(ref, HostedOperationRef) else HostedOperationRef.from_dict(ref)
        owner = _bounded_identity(owner_actor_id, label="owner_actor_id", max_bytes=MAX_OWNER_ACTOR_ID_BYTES)
        with self._state_lock():
            row = self._resolve_locked(operation=operation, owner_actor_id=owner)
            if row is None:
                raise ResultArtifactError("result_artifact_unauthorized")
            result_ref_row = dict(dict(row.get("terminal") or {}).get("result_ref") or {})
            if not result_ref_row:
                raise ResultArtifactError("operation_has_no_result_artifact")
            result_ref = HostedResultRef.from_dict(result_ref_row)
        payload = self.result_artifact_store.read(
            ref=result_ref,
            owner_actor_id=owner_actor_id,
            operation_id=operation.operation_id,
        )
        try:
            content = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ResultArtifactError("result_artifact_content_invalid") from exc
        return {
            "contract": "hosting.result_content",
            "result_ref": result_ref.to_dict(),
            "content": content,
        }

    @classmethod
    def archive_legacy_checkpoint(
        cls,
        path: Path,
        *,
        acknowledge_replay_window_clear: bool,
        clock: Any = time.time,
    ) -> Path:
        source = Path(path).expanduser().resolve()
        if not acknowledge_replay_window_clear:
            raise PermissionError("receipt_ledger_cutover_acknowledgement_required")
        lock_path = source.with_name(f"{source.name}.lock")
        with _exclusive_process_file_lock(lock_path):
            if not source.exists() or not source.is_file():
                raise FileNotFoundError(source)
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"legacy receipt ledger is unreadable: {source}") from exc
            if not isinstance(payload, dict) or "version" not in payload:
                raise ValueError("receipt_ledger_is_not_legacy_schema")
            timestamp = int(float(clock()) * 1000)
            target = source.with_name(f"{source.name}.legacy-{timestamp}.archive")
            if target.exists():
                raise FileExistsError(target)
            _replace_with_bounded_retries(source, target)
            return target


__all__ = [
    "AtomicJsonHostedOperationRepository",
    "HostedOperationRepository",
    "LegacyOperationRepositoryError",
    "MAX_METADATA_BYTES",
    "MAX_OWNER_ACTOR_ID_BYTES",
    "OPERATION_REPOSITORY_CONTRACT",
]
