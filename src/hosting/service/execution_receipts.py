"""Durable, bounded idempotency receipts for hosted toolbox execution."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


RECEIPT_VERSION = 1
TERMINAL_STATES = {"terminal_success", "terminal_failure", "terminal_cancellation"}
ACTIVE_STATES = {"queued", "running"}
INTERRUPTED_STATES = {"interrupted_before_dispatch", "interrupted_after_dispatch_unknown"}
_SECRET_KEY = re.compile(
    r"(?:authorization|credential|password|passwd|secret|session[_-]?token|access[_-]?token|refresh[_-]?token|api[_-]?key|private[_-]?key)$",
    re.IGNORECASE,
)
_MAX_ID_CHARS = 256


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8", errors="replace"
    )


def execution_fingerprint(payload: Dict[str, Any]) -> str:
    """Return the canonical SHA-256 fingerprint; no source payload is persisted."""
    return hashlib.sha256(_canonical_json(dict(payload or {}))).hexdigest()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


class ToolboxExecutionReceiptLedger:
    """Atomic JSON checkpoint whose size is bounded by receipt/tombstone retention."""

    def __init__(
        self,
        path: Path,
        *,
        receipt_retention_seconds: float = 7 * 24 * 3600,
        tombstone_retention_seconds: float = 14 * 24 * 3600,
        max_receipts: int = 10_000,
        max_tombstones: int = 20_000,
        max_result_bytes: int = 64 * 1024,
        clock: Any = time.time,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.receipt_retention_seconds = max(0.0, float(receipt_retention_seconds))
        self.tombstone_retention_seconds = max(0.0, float(tombstone_retention_seconds))
        self.max_receipts = max(1, int(max_receipts))
        self.max_tombstones = max(1, int(max_tombstones))
        self.max_result_bytes = max(256, int(max_result_bytes))
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._data = self._load()
        with self._condition:
            changed = self._recover_interrupted_locked()
            changed = self._compact_locked() or changed
            if changed:
                self._persist_locked()

    @staticmethod
    def _key(namespace: str, request_id: str) -> str:
        raw = f"{namespace}\0{request_id}".encode("utf-8", errors="replace")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _validate_identity(namespace: str, request_id: str) -> tuple[str, str]:
        ns = str(namespace or "").strip()
        rid = str(request_id or "").strip()
        if not ns:
            raise ValueError("receipt namespace is required")
        if not rid:
            raise ValueError("execution_request_id is required")
        if len(ns) > _MAX_ID_CHARS or len(rid) > _MAX_ID_CHARS:
            raise ValueError("receipt namespace and execution_request_id are limited to 256 characters")
        return ns, rid

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"version": RECEIPT_VERSION, "receipts": {}, "tombstones": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"toolbox receipt ledger is unreadable: {self.path}") from exc
        if not isinstance(payload, dict) or int(payload.get("version") or 0) != RECEIPT_VERSION:
            raise RuntimeError(f"unsupported toolbox receipt ledger: {self.path}")
        receipts = payload.get("receipts")
        tombstones = payload.get("tombstones")
        if not isinstance(receipts, dict) or not isinstance(tombstones, dict):
            raise RuntimeError(f"invalid toolbox receipt ledger: {self.path}")
        return {"version": RECEIPT_VERSION, "receipts": receipts, "tombstones": tombstones}

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            encoded = json.dumps(self._data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            with temp.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except Exception:
                pass

    def _recover_interrupted_locked(self) -> bool:
        changed = False
        now = float(self._clock())
        for receipt in self._data["receipts"].values():
            state = str(receipt.get("state") or "")
            if state == "queued":
                receipt["state"] = "interrupted_before_dispatch"
                receipt["updated_at"] = now
                changed = True
            elif state == "running":
                receipt["state"] = "interrupted_after_dispatch_unknown"
                receipt["updated_at"] = now
                changed = True
        return changed

    def _forget_locked(self, key: str, receipt: Dict[str, Any], now: float) -> None:
        self._data["receipts"].pop(key, None)
        self._data["tombstones"][key] = {
            "version": RECEIPT_VERSION,
            "namespace": str(receipt.get("namespace") or "")[:_MAX_ID_CHARS],
            "request_id": str(receipt.get("request_id") or "")[:_MAX_ID_CHARS],
            "fingerprint": str(receipt.get("fingerprint") or ""),
            "state": "forgotten",
            "forgotten_at": now,
            "expires_at": now + self.tombstone_retention_seconds,
        }

    def _compact_locked(self) -> bool:
        now = float(self._clock())
        changed = False
        receipts: Dict[str, Dict[str, Any]] = self._data["receipts"]
        eligible = sorted(
            (
                (float(row.get("updated_at") or row.get("created_at") or 0.0), key, row)
                for key, row in receipts.items()
                if str(row.get("state") or "") in TERMINAL_STATES | INTERRUPTED_STATES
            ),
            key=lambda item: (item[0], item[1]),
        )
        for updated_at, key, row in eligible:
            if now - updated_at >= self.receipt_retention_seconds:
                self._forget_locked(key, row, now)
                changed = True
        overflow = max(0, len(receipts) - self.max_receipts)
        if overflow:
            remaining = sorted(
                (
                    (float(row.get("updated_at") or row.get("created_at") or 0.0), key, row)
                    for key, row in receipts.items()
                    if str(row.get("state") or "") in TERMINAL_STATES | INTERRUPTED_STATES
                ),
                key=lambda item: (item[0], item[1]),
            )
            for _, key, row in remaining[:overflow]:
                self._forget_locked(key, row, now)
                changed = True
        tombstones: Dict[str, Dict[str, Any]] = self._data["tombstones"]
        for key, row in list(tombstones.items()):
            if now >= float(row.get("expires_at") or 0.0):
                tombstones.pop(key, None)
                changed = True
        if len(tombstones) > self.max_tombstones:
            ordered = sorted(
                tombstones.items(),
                key=lambda item: (float(item[1].get("forgotten_at") or 0.0), item[0]),
            )
            for key, _ in ordered[: len(tombstones) - self.max_tombstones]:
                tombstones.pop(key, None)
                changed = True
        return changed

    @staticmethod
    def _public(receipt: Dict[str, Any], *, include_envelope: bool = True) -> Dict[str, Any]:
        row = copy.deepcopy(dict(receipt or {}))
        if not include_envelope:
            row.pop("terminal_envelope", None)
        return row

    def prepare(
        self,
        *,
        namespace: str,
        request_id: str,
        fingerprint: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ns, rid = self._validate_identity(namespace, request_id)
        digest = str(fingerprint or "").strip()
        if len(digest) != 64:
            raise ValueError("a SHA-256 execution fingerprint is required")
        key = self._key(ns, rid)
        now = float(self._clock())
        with self._condition:
            changed = self._compact_locked()
            receipt = self._data["receipts"].get(key)
            tombstone = self._data["tombstones"].get(key)
            if receipt is not None:
                if str(receipt.get("fingerprint") or "") != digest:
                    if changed:
                        self._persist_locked()
                    return {"action": "conflict", "receipt": self._public(receipt, include_envelope=False)}
                state = str(receipt.get("state") or "")
                if state in TERMINAL_STATES:
                    if changed:
                        self._persist_locked()
                    return {"action": "replay", "receipt": self._public(receipt)}
                if state == "interrupted_before_dispatch":
                    receipt["state"] = "queued"
                    receipt["updated_at"] = now
                    receipt["resume_count"] = min(1, int(receipt.get("resume_count") or 0) + 1)
                    self._persist_locked()
                    return {"action": "dispatch", "receipt": self._public(receipt)}
                if changed:
                    self._persist_locked()
                return {"action": "attach", "receipt": self._public(receipt, include_envelope=False)}
            if tombstone is not None:
                if str(tombstone.get("fingerprint") or "") and str(tombstone.get("fingerprint")) != digest:
                    if changed:
                        self._persist_locked()
                    return {"action": "conflict", "receipt": self._public(tombstone, include_envelope=False)}
                if changed:
                    self._persist_locked()
                return {"action": "forgotten", "receipt": self._public(tombstone, include_envelope=False)}
            if len(self._data["receipts"]) >= self.max_receipts:
                eligible = sorted(
                    (
                        (float(row.get("updated_at") or row.get("created_at") or 0.0), item_key, row)
                        for item_key, row in self._data["receipts"].items()
                        if str(row.get("state") or "") in TERMINAL_STATES | INTERRUPTED_STATES
                    ),
                    key=lambda item: (item[0], item[1]),
                )
                if eligible:
                    _, oldest_key, oldest = eligible[0]
                    self._forget_locked(oldest_key, oldest, now)
                    self._compact_locked()
                if len(self._data["receipts"]) >= self.max_receipts:
                    if changed:
                        self._persist_locked()
                    return {"action": "capacity", "receipt": {"state": "receipt_capacity_exceeded"}}
            receipt = {
                "version": RECEIPT_VERSION,
                "namespace": ns,
                "request_id": rid,
                "fingerprint": digest,
                "state": "queued",
                "created_at": now,
                "updated_at": now,
                "resume_count": 0,
                "metadata": _redact(dict(metadata or {})),
            }
            self._data["receipts"][key] = receipt
            self._persist_locked()
            return {"action": "dispatch", "receipt": self._public(receipt)}

    def mark_dispatch_claimed(self, *, namespace: str, request_id: str) -> Dict[str, Any]:
        ns, rid = self._validate_identity(namespace, request_id)
        key = self._key(ns, rid)
        now = float(self._clock())
        with self._condition:
            receipt = self._data["receipts"].get(key)
            if receipt is None:
                raise KeyError(rid)
            state = str(receipt.get("state") or "")
            if state != "queued":
                return self._public(receipt)
            receipt["state"] = "running"
            receipt["dispatch_claimed_at"] = now
            receipt["updated_at"] = now
            self._persist_locked()
            self._condition.notify_all()
            return self._public(receipt)

    def _bounded_envelope(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        redacted = _redact(copy.deepcopy(dict(envelope or {})))
        encoded = _canonical_json(redacted)
        if len(encoded) <= self.max_result_bytes:
            return redacted
        digest = hashlib.sha256(encoded).hexdigest()
        safe_keys = {
            "status", "outcome", "reason", "error", "error_type", "engine_id", "toolbox_id",
            "tool_name", "tool_call_id", "request_id", "environment_key", "worker_id", "retry_count",
            "admission", "concurrency",
        }
        bounded = {key: redacted.get(key) for key in safe_keys if key in redacted}
        bounded["result_reference"] = {
            "kind": "omitted_oversize_terminal_envelope",
            "sha256": digest,
            "size_bytes": len(encoded),
        }
        return bounded

    def finish(
        self,
        *,
        namespace: str,
        request_id: str,
        state: str,
        envelope: Dict[str, Any],
    ) -> Dict[str, Any]:
        if state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal receipt state: {state}")
        ns, rid = self._validate_identity(namespace, request_id)
        key = self._key(ns, rid)
        now = float(self._clock())
        with self._condition:
            receipt = self._data["receipts"].get(key)
            if receipt is None:
                raise KeyError(rid)
            if str(receipt.get("state") or "") in TERMINAL_STATES:
                return self._public(receipt)
            receipt["state"] = state
            receipt["terminal_at"] = now
            receipt["updated_at"] = now
            receipt["terminal_envelope"] = self._bounded_envelope(envelope)
            self._compact_locked()
            self._persist_locked()
            self._condition.notify_all()
            return self._public(receipt)

    def status(self, *, namespace: str, request_id: str) -> Dict[str, Any]:
        ns, rid = self._validate_identity(namespace, request_id)
        key = self._key(ns, rid)
        with self._condition:
            changed = self._compact_locked()
            receipt = self._data["receipts"].get(key)
            if receipt is not None:
                out = self._public(receipt)
            else:
                tombstone = self._data["tombstones"].get(key)
                out = self._public(tombstone, include_envelope=False) if tombstone is not None else {
                    "version": RECEIPT_VERSION,
                    "namespace": ns,
                    "request_id": rid,
                    "state": "unknown_outside_retention",
                }
            if changed:
                self._persist_locked()
            return out

    def wait_for_terminal(self, *, namespace: str, request_id: str, timeout_seconds: float) -> Dict[str, Any]:
        ns, rid = self._validate_identity(namespace, request_id)
        key = self._key(ns, rid)
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        with self._condition:
            while True:
                receipt = self._data["receipts"].get(key)
                if receipt is None or str(receipt.get("state") or "") not in ACTIVE_STATES:
                    return self.status(namespace=ns, request_id=rid)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._public(receipt, include_envelope=False)
                self._condition.wait(timeout=remaining)

    def cancel_before_dispatch(
        self,
        *,
        namespace: str,
        request_id: str,
        envelope: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        ns, rid = self._validate_identity(namespace, request_id)
        key = self._key(ns, rid)
        with self._condition:
            receipt = self._data["receipts"].get(key)
            if receipt is None or str(receipt.get("state") or "") not in {"queued", "interrupted_before_dispatch"}:
                return None
        return self.finish(
            namespace=ns,
            request_id=rid,
            state="terminal_cancellation",
            envelope=envelope,
        )

    def compact(self) -> None:
        with self._condition:
            if self._compact_locked():
                self._persist_locked()
