"""Authoritative hosted-operation client contracts.

The models in this module are intentionally strict: they reject unknown fields,
non-canonical digests, unbounded text, and contradictory terminal payloads at
the client/daemon boundary. Transport status is represented by ``api_status``;
``lifecycle`` always describes the durable hosted operation itself.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, Mapping, Optional


HOSTED_OPERATION_REF_CONTRACT = "hosting.operation_ref"
HOSTED_OPERATION_STATUS_CONTRACT = "hosting.operation_status"
HOSTED_RESULT_REF_CONTRACT = "hosting.result_ref"
HOSTED_RESULT_OMISSION_CONTRACT = "hosting.result_omission"

MAX_OPERATION_ID_BYTES = 256
MAX_REQUEST_ID_BYTES = 256
MAX_RECEIPT_NAMESPACE_BYTES = 256
MAX_SELECTOR_ID_BYTES = 256
MAX_REASON_BYTES = 512
MAX_MEDIA_TYPE_BYTES = 128
MAX_INLINE_RESULT_BYTES = 64 * 1024
MAX_REFERENCE_FIELDS = 8

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9._~-]+$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_FIELDS = {
    "contract",
    "operation_id",
    "request_id",
    "execution_kind",
    "selector",
    "fingerprint",
    "receipt_namespace",
}
_SELECTOR_FIELDS = {"kind", "id"}
_RESULT_REF_FIELDS = {"contract", "artifact_id", "digest", "size_bytes", "media_type", "expires_at_ms"}
_OMISSION_FIELDS = {"contract", "digest", "size_bytes", "reason"}
_STATUS_FIELDS = {
    "contract",
    "api_status",
    "operation",
    "lifecycle",
    "request_id",
    "created_at_ms",
    "updated_at_ms",
    "dispatch_claimed_at_ms",
    "terminal_at_ms",
    "reason",
    "result",
    "result_ref",
    "result_omission",
}


class HostedExecutionKind(StrEnum):
    TOOLBOX = "toolbox"
    WORKFLOW_PYTHON = "workflow_python"
    WORKFLOW_JS = "workflow_js"


class HostedOperationLifecycle(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_FAILURE = "terminal_failure"
    TERMINAL_CANCELLATION = "terminal_cancellation"
    INTERRUPTED_BEFORE_DISPATCH = "interrupted_before_dispatch"
    INTERRUPTED_AFTER_DISPATCH_UNKNOWN = "interrupted_after_dispatch_unknown"
    FORGOTTEN = "forgotten"
    UNKNOWN_OUTSIDE_RETENTION = "unknown_outside_retention"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"


TERMINAL_OPERATION_LIFECYCLES = frozenset(
    {
        HostedOperationLifecycle.TERMINAL_SUCCESS,
        HostedOperationLifecycle.TERMINAL_FAILURE,
        HostedOperationLifecycle.TERMINAL_CANCELLATION,
    }
)


def _strict_fields(row: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(str(key) for key in set(row.keys()) - allowed)
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")
    if len(row) > max(len(allowed), MAX_REFERENCE_FIELDS):
        raise ValueError(f"{label}_too_many_fields")


def _bounded_text(value: Any, *, label: str, max_bytes: int, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{label}_required")
    if any(ord(char) < 32 for char in text):
        raise ValueError(f"{label}_contains_control_characters")
    if len(text.encode("utf-8", errors="strict")) > max_bytes:
        raise ValueError(f"{label}_too_large")
    return text


def _opaque_id(value: Any, *, label: str, max_bytes: int) -> str:
    text = _bounded_text(value, label=label, max_bytes=max_bytes)
    if not _OPAQUE_ID_RE.fullmatch(text):
        raise ValueError(f"{label}_invalid")
    return text


def canonical_sha256_digest(value: Any, *, label: str = "digest") -> str:
    digest = _bounded_text(value, label=label, max_bytes=71)
    if not _DIGEST_RE.fullmatch(digest):
        raise ValueError(f"{label}_must_be_canonical_sha256")
    return digest


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value_must_be_json_serializable") from exc


def hosted_execution_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return the canonical dispatch fingerprint used by every execution family."""
    return f"sha256:{hashlib.sha256(canonical_json_bytes(dict(payload or {}))).hexdigest()}"


@dataclass(frozen=True)
class HostedOperationSelector:
    kind: str
    id: str

    def __post_init__(self) -> None:
        if self.kind not in {"toolbox_id", "engine_id"}:
            raise ValueError("operation_selector_kind_invalid")
        _bounded_text(self.id, label="operation_selector_id", max_bytes=MAX_SELECTOR_ID_BYTES)

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "id": self.id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HostedOperationSelector":
        row = dict(payload or {})
        _strict_fields(row, _SELECTOR_FIELDS, label="operation_selector")
        return cls(
            kind=_bounded_text(row.get("kind"), label="operation_selector_kind", max_bytes=32),
            id=_bounded_text(row.get("id"), label="operation_selector_id", max_bytes=MAX_SELECTOR_ID_BYTES),
        )


@dataclass(frozen=True)
class HostedOperationRef:
    operation_id: str
    request_id: str
    execution_kind: HostedExecutionKind
    selector: HostedOperationSelector
    fingerprint: str
    receipt_namespace: str
    contract: str = HOSTED_OPERATION_REF_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != HOSTED_OPERATION_REF_CONTRACT:
            raise ValueError("operation_ref_contract_invalid")
        _opaque_id(self.operation_id, label="operation_id", max_bytes=MAX_OPERATION_ID_BYTES)
        _bounded_text(self.request_id, label="operation_request_id", max_bytes=MAX_REQUEST_ID_BYTES)
        if not isinstance(self.execution_kind, HostedExecutionKind):
            raise ValueError("operation_execution_kind_invalid")
        if not isinstance(self.selector, HostedOperationSelector):
            raise ValueError("operation_selector_invalid")
        canonical_sha256_digest(self.fingerprint, label="operation_fingerprint")
        _bounded_text(
            self.receipt_namespace,
            label="operation_receipt_namespace",
            max_bytes=MAX_RECEIPT_NAMESPACE_BYTES,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "operation_id": self.operation_id,
            "request_id": self.request_id,
            "execution_kind": self.execution_kind.value,
            "selector": self.selector.to_dict(),
            "fingerprint": self.fingerprint,
            "receipt_namespace": self.receipt_namespace,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HostedOperationRef":
        row = dict(payload or {})
        _strict_fields(row, _REF_FIELDS, label="operation_ref")
        try:
            execution_kind = HostedExecutionKind(str(row.get("execution_kind") or ""))
        except ValueError as exc:
            raise ValueError("operation_execution_kind_invalid") from exc
        return cls(
            contract=str(row.get("contract") or ""),
            operation_id=_opaque_id(row.get("operation_id"), label="operation_id", max_bytes=MAX_OPERATION_ID_BYTES),
            request_id=_bounded_text(
                row.get("request_id"), label="operation_request_id", max_bytes=MAX_REQUEST_ID_BYTES
            ),
            execution_kind=execution_kind,
            selector=HostedOperationSelector.from_dict(dict(row.get("selector") or {})),
            fingerprint=canonical_sha256_digest(row.get("fingerprint"), label="operation_fingerprint"),
            receipt_namespace=_bounded_text(
                row.get("receipt_namespace"),
                label="operation_receipt_namespace",
                max_bytes=MAX_RECEIPT_NAMESPACE_BYTES,
            ),
        )


@dataclass(frozen=True)
class HostedResultRef:
    artifact_id: str
    digest: str
    size_bytes: int
    media_type: str = "application/json"
    expires_at_ms: Optional[int] = None
    contract: str = HOSTED_RESULT_REF_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != HOSTED_RESULT_REF_CONTRACT:
            raise ValueError("result_ref_contract_invalid")
        _opaque_id(self.artifact_id, label="result_artifact_id", max_bytes=MAX_OPERATION_ID_BYTES)
        canonical_sha256_digest(self.digest, label="result_digest")
        if int(self.size_bytes) < 0:
            raise ValueError("result_size_bytes_invalid")
        _bounded_text(self.media_type, label="result_media_type", max_bytes=MAX_MEDIA_TYPE_BYTES)
        if self.expires_at_ms is not None and int(self.expires_at_ms) < 0:
            raise ValueError("result_expires_at_ms_invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "size_bytes": int(self.size_bytes),
            "media_type": self.media_type,
            "expires_at_ms": self.expires_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HostedResultRef":
        row = dict(payload or {})
        _strict_fields(row, _RESULT_REF_FIELDS, label="result_ref")
        return cls(
            contract=str(row.get("contract") or ""),
            artifact_id=str(row.get("artifact_id") or ""),
            digest=str(row.get("digest") or ""),
            size_bytes=int(row.get("size_bytes") or 0),
            media_type=str(row.get("media_type") or ""),
            expires_at_ms=int(row["expires_at_ms"]) if row.get("expires_at_ms") is not None else None,
        )


@dataclass(frozen=True)
class HostedResultOmission:
    digest: str
    size_bytes: int
    reason: str
    contract: str = HOSTED_RESULT_OMISSION_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != HOSTED_RESULT_OMISSION_CONTRACT:
            raise ValueError("result_omission_contract_invalid")
        canonical_sha256_digest(self.digest, label="result_digest")
        if int(self.size_bytes) < 0:
            raise ValueError("result_size_bytes_invalid")
        _bounded_text(self.reason, label="result_omission_reason", max_bytes=MAX_REASON_BYTES)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "digest": self.digest,
            "size_bytes": int(self.size_bytes),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HostedResultOmission":
        row = dict(payload or {})
        _strict_fields(row, _OMISSION_FIELDS, label="result_omission")
        return cls(
            contract=str(row.get("contract") or ""),
            digest=str(row.get("digest") or ""),
            size_bytes=int(row.get("size_bytes") or 0),
            reason=str(row.get("reason") or ""),
        )


@dataclass(frozen=True)
class HostedOperationStatus:
    operation: HostedOperationRef
    lifecycle: HostedOperationLifecycle
    request_id: str
    created_at_ms: int
    updated_at_ms: int
    api_status: str = "ok"
    dispatch_claimed_at_ms: Optional[int] = None
    terminal_at_ms: Optional[int] = None
    reason: Optional[str] = None
    result: Any = None
    result_ref: Optional[HostedResultRef] = None
    result_omission: Optional[HostedResultOmission] = None
    contract: str = HOSTED_OPERATION_STATUS_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != HOSTED_OPERATION_STATUS_CONTRACT:
            raise ValueError("operation_status_contract_invalid")
        if self.api_status not in {"ok", "error"}:
            raise ValueError("operation_api_status_invalid")
        if not isinstance(self.operation, HostedOperationRef):
            raise ValueError("operation_status_ref_invalid")
        if not isinstance(self.lifecycle, HostedOperationLifecycle):
            raise ValueError("operation_lifecycle_invalid")
        request_id = _bounded_text(
            self.request_id, label="operation_request_id", max_bytes=MAX_REQUEST_ID_BYTES
        )
        if request_id != self.operation.request_id:
            raise ValueError("operation_status_request_id_mismatch")
        for name, value in (
            ("created_at_ms", self.created_at_ms),
            ("updated_at_ms", self.updated_at_ms),
            ("dispatch_claimed_at_ms", self.dispatch_claimed_at_ms),
            ("terminal_at_ms", self.terminal_at_ms),
        ):
            if value is not None and int(value) < 0:
                raise ValueError(f"operation_{name}_invalid")
        if int(self.updated_at_ms) < int(self.created_at_ms):
            raise ValueError("operation_updated_at_before_created_at")
        if self.reason is not None:
            _bounded_text(self.reason, label="operation_reason", max_bytes=MAX_REASON_BYTES, required=False)
        terminal_values = sum(value is not None for value in (self.result, self.result_ref, self.result_omission))
        if terminal_values > 1:
            raise ValueError("operation_terminal_payload_conflict")
        if self.result is not None and len(canonical_json_bytes(self.result)) > MAX_INLINE_RESULT_BYTES:
            raise ValueError("operation_inline_result_too_large")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "api_status": self.api_status,
            "operation": self.operation.to_dict(),
            "lifecycle": self.lifecycle.value,
            "request_id": self.request_id,
            "created_at_ms": int(self.created_at_ms),
            "updated_at_ms": int(self.updated_at_ms),
            "dispatch_claimed_at_ms": self.dispatch_claimed_at_ms,
            "terminal_at_ms": self.terminal_at_ms,
            "reason": self.reason,
            "result": copy.deepcopy(self.result),
            "result_ref": self.result_ref.to_dict() if self.result_ref is not None else None,
            "result_omission": self.result_omission.to_dict() if self.result_omission is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HostedOperationStatus":
        row = dict(payload or {})
        _strict_fields(row, _STATUS_FIELDS, label="operation_status")
        try:
            lifecycle = HostedOperationLifecycle(str(row.get("lifecycle") or ""))
        except ValueError as exc:
            raise ValueError("operation_lifecycle_invalid") from exc
        result_ref = row.get("result_ref")
        result_omission = row.get("result_omission")
        return cls(
            contract=str(row.get("contract") or ""),
            api_status=str(row.get("api_status") or ""),
            operation=HostedOperationRef.from_dict(dict(row.get("operation") or {})),
            lifecycle=lifecycle,
            request_id=str(row.get("request_id") or ""),
            created_at_ms=int(row.get("created_at_ms") or 0),
            updated_at_ms=int(row.get("updated_at_ms") or 0),
            dispatch_claimed_at_ms=(
                int(row["dispatch_claimed_at_ms"]) if row.get("dispatch_claimed_at_ms") is not None else None
            ),
            terminal_at_ms=int(row["terminal_at_ms"]) if row.get("terminal_at_ms") is not None else None,
            reason=str(row["reason"]) if row.get("reason") is not None else None,
            result=copy.deepcopy(row.get("result")),
            result_ref=HostedResultRef.from_dict(result_ref) if isinstance(result_ref, Mapping) else None,
            result_omission=(
                HostedResultOmission.from_dict(result_omission) if isinstance(result_omission, Mapping) else None
            ),
        )


__all__ = [
    "HOSTED_OPERATION_REF_CONTRACT",
    "HOSTED_OPERATION_STATUS_CONTRACT",
    "HOSTED_RESULT_OMISSION_CONTRACT",
    "HOSTED_RESULT_REF_CONTRACT",
    "MAX_INLINE_RESULT_BYTES",
    "MAX_MEDIA_TYPE_BYTES",
    "MAX_OPERATION_ID_BYTES",
    "MAX_REASON_BYTES",
    "MAX_RECEIPT_NAMESPACE_BYTES",
    "MAX_REQUEST_ID_BYTES",
    "MAX_SELECTOR_ID_BYTES",
    "TERMINAL_OPERATION_LIFECYCLES",
    "HostedExecutionKind",
    "HostedOperationLifecycle",
    "HostedOperationRef",
    "HostedOperationSelector",
    "HostedOperationStatus",
    "HostedResultOmission",
    "HostedResultRef",
    "canonical_json_bytes",
    "canonical_sha256_digest",
    "hosted_execution_fingerprint",
]
