"""Shared hosted sandbox runtime models.

This module is intentionally behavior-light. It defines deterministic identity,
pool, request-lifecycle, stream-event, and metrics shapes that concrete hosted
sandboxes can share without inheriting toolbox, workflow, or model semantics.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .policy import WorkerSandboxPolicy


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def stable_hash(payload: Any) -> str:
    return sha256_text(stable_json(payload))


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _unique_strings(items: Any) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        value = _clean(item)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _string_map(payload: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in dict(payload or {}).items():
        k = _clean(key)
        v = _clean(value)
        if k and v:
            out[k] = v
    return dict(sorted(out.items()))


def normalize_sandbox_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return WorkerSandboxPolicy.from_mapping(dict(policy or {})).to_dict()


def sandbox_policy_hash(policy: Optional[Dict[str, Any]]) -> str:
    return stable_hash(normalize_sandbox_policy(policy))


@dataclass(frozen=True)
class HostedRuntimeIdentity:
    runtime_kind: str
    profile: str
    runtime_hash: str
    runtime_version: Optional[str] = None
    capability_profile: Optional[str] = None

    def normalized(self) -> Dict[str, Any]:
        return {
            "runtime_kind": _clean(self.runtime_kind) or "generic",
            "profile": _clean(self.profile) or "default",
            "runtime_hash": _clean(self.runtime_hash) or "unknown",
            "runtime_version": _clean(self.runtime_version) or None,
            "capability_profile": _clean(self.capability_profile) or None,
        }


@dataclass(frozen=True)
class HostedEnvironmentKeySpec:
    environment_name: str
    runtime: HostedRuntimeIdentity
    sandbox_policy: Dict[str, Any] = field(default_factory=dict)
    required_imports: List[str] = field(default_factory=list)
    package_pins: Dict[str, str] = field(default_factory=dict)
    dependency_lock_hash: Optional[str] = None

    def normalized(self) -> Dict[str, Any]:
        return {
            "environment_name": _clean(self.environment_name) or "default",
            "runtime": self.runtime.normalized(),
            "required_imports": _unique_strings(self.required_imports),
            "package_pins": _string_map(self.package_pins),
            "dependency_lock_hash": _clean(self.dependency_lock_hash) or None,
            "sandbox_policy_hash": sandbox_policy_hash(self.sandbox_policy),
        }

    def full_key(self) -> str:
        return stable_hash(self.normalized())

    def short_key(self, length: int = 16) -> str:
        return self.full_key()[: max(8, min(int(length or 16), 64))]

    def to_dict(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            **normalized,
            "environment_key": self.short_key(),
            "environment_key_full": self.full_key(),
        }


@dataclass(frozen=True)
class HostedPoolKey:
    sandbox_kind: str
    environment_key: str

    def normalized(self) -> Dict[str, str]:
        return {
            "sandbox_kind": _clean(self.sandbox_kind) or "generic",
            "environment_key": _clean(self.environment_key),
        }

    def pool_id(self) -> str:
        row = self.normalized()
        return f"{row['sandbox_kind']}/{row['environment_key']}"


@dataclass
class HostedWorkerSlot:
    engine_id: str
    environment_key: str
    capacity: int = 1
    active_request_ids: List[str] = field(default_factory=list)
    pid: Optional[int] = None
    status: str = "unknown"
    metrics: Dict[str, Any] = field(default_factory=dict)

    def available_slots(self) -> int:
        return max(0, int(self.capacity or 0) - len(_unique_strings(self.active_request_ids)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": _clean(self.engine_id),
            "environment_key": _clean(self.environment_key),
            "pid": int(self.pid) if self.pid is not None else None,
            "status": _clean(self.status) or "unknown",
            "capacity": max(0, int(self.capacity or 0)),
            "active_request_ids": _unique_strings(self.active_request_ids),
            "active_calls": len(_unique_strings(self.active_request_ids)),
            "available_slots": self.available_slots(),
            "metrics": dict(self.metrics or {}),
        }


@dataclass
class HostedRequestLifecycle:
    request_id: str
    environment_key: str
    sandbox_kind: str
    profile: str
    operation_id: Optional[str] = None
    engine_id: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "submitted"
    reason: Optional[str] = None
    input_bytes: Optional[int] = None
    output_bytes: Optional[int] = None
    latest_progress: Optional[Dict[str, Any]] = None
    stream_event_count: int = 0

    def mark_started(self, *, timestamp: Optional[float] = None, engine_id: Optional[str] = None) -> None:
        self.started_at = float(timestamp if timestamp is not None else time.time())
        if engine_id is not None:
            self.engine_id = _clean(engine_id) or None
        self.status = "running"

    def mark_finished(self, status: str, *, reason: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        self.finished_at = float(timestamp if timestamp is not None else time.time())
        self.status = _clean(status) or "finished"
        self.reason = _clean(reason) or None

    def record_stream_event(self, event: Dict[str, Any]) -> None:
        row = dict(event or {})
        self.stream_event_count += 1
        event_type = _clean(row.get("type"))
        if event_type == "progress":
            payload = dict(row.get("payload") or {})
            self.latest_progress = {
                "type": event_type,
                "timestamp": float(row.get("timestamp") or time.time()),
                "sequence": max(0, int(row.get("sequence") or 0)),
                "payload": payload,
            }

    def queue_wait_ms(self) -> Optional[int]:
        if self.started_at is None:
            return None
        return max(0, int((self.started_at - self.submitted_at) * 1000))

    def execution_ms(self) -> Optional[int]:
        if self.started_at is None or self.finished_at is None:
            return None
        return max(0, int((self.finished_at - self.started_at) * 1000))

    def lifetime_ms(self) -> Optional[int]:
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0, int((end - self.submitted_at) * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": _clean(self.request_id),
            "operation_id": _clean(self.operation_id) or None,
            "environment_key": _clean(self.environment_key),
            "sandbox_kind": _clean(self.sandbox_kind),
            "profile": _clean(self.profile),
            "engine_id": _clean(self.engine_id) or None,
            "submitted_at": float(self.submitted_at),
            "started_at": float(self.started_at) if self.started_at is not None else None,
            "finished_at": float(self.finished_at) if self.finished_at is not None else None,
            "status": _clean(self.status) or "submitted",
            "reason": _clean(self.reason) or None,
            "queue_wait_ms": self.queue_wait_ms(),
            "execution_ms": self.execution_ms(),
            "lifetime_ms": self.lifetime_ms(),
            "input_bytes": int(self.input_bytes) if self.input_bytes is not None else None,
            "output_bytes": int(self.output_bytes) if self.output_bytes is not None else None,
            "latest_progress": dict(self.latest_progress or {}) or None,
            "stream_event_count": max(0, int(self.stream_event_count or 0)),
        }


@dataclass(frozen=True)
class HostedStreamEvent:
    type: str
    request_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": _clean(self.type) or "event",
            "request_id": _clean(self.request_id),
            "sequence": max(0, int(self.sequence or 0)),
            "timestamp": float(self.timestamp),
            "payload": dict(self.payload or {}),
        }


@dataclass
class HostedPoolMetrics:
    desired_capacity: int = 1
    workers: List[HostedWorkerSlot] = field(default_factory=list)
    recent_requests: List[HostedRequestLifecycle] = field(default_factory=list)
    saturation_count: int = 0
    timeout_count: int = 0
    cancellation_count: int = 0
    error_count: int = 0
    errors_by_reason: Dict[str, int] = field(default_factory=dict)

    def active_calls(self) -> int:
        return sum(len(_unique_strings(worker.active_request_ids)) for worker in self.workers)

    def available_slots(self) -> int:
        return sum(worker.available_slots() for worker in self.workers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "desired_capacity": max(0, int(self.desired_capacity or 0)),
            "worker_count": len(self.workers),
            "active_calls": self.active_calls(),
            "available_slots": self.available_slots(),
            "saturation_count": max(0, int(self.saturation_count or 0)),
            "timeout_count": max(0, int(self.timeout_count or 0)),
            "cancellation_count": max(0, int(self.cancellation_count or 0)),
            "error_count": max(0, int(self.error_count or 0)),
            "errors_by_reason": {str(k): int(v) for k, v in dict(self.errors_by_reason or {}).items()},
            "workers": [worker.to_dict() for worker in self.workers],
            "recent_requests": [request.to_dict() for request in self.recent_requests],
        }


__all__ = [
    "HostedEnvironmentKeySpec",
    "HostedPoolKey",
    "HostedPoolMetrics",
    "HostedRequestLifecycle",
    "HostedRuntimeIdentity",
    "HostedStreamEvent",
    "HostedWorkerSlot",
    "normalize_sandbox_policy",
    "sandbox_policy_hash",
    "stable_hash",
    "stable_json",
]
