"""Shared hosted sandbox runtime models.

This module is intentionally behavior-light. It defines deterministic identity,
pool, request-lifecycle, stream-event, and metrics shapes that concrete hosted
sandboxes can share without inheriting toolbox, workflow, or model semantics.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

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


HOSTED_STREAM_CONTRACT_VERSION = 1

HOSTED_STREAM_LANES = ["control", "event", "output", "audit"]

HOSTED_STREAM_QUEUE_DECISIONS = [
    "non_droppable",
    "latest",
    "latest_by_key",
    "keep_first",
    "keep_first_by_window",
]


@dataclass(frozen=True)
class HostedStreamKindSpec:
    kind: str
    lane: str
    queue_decision: str
    replacement_fields: Tuple[str, ...] = ()
    terminal: bool = False
    final: bool = False
    decision_bearing: bool = False

    def __post_init__(self) -> None:
        lane = _clean(self.lane)
        if lane not in HOSTED_STREAM_LANES:
            raise ValueError(f"unsupported_stream_lane:{lane}")
        decision = _clean(self.queue_decision)
        if decision not in HOSTED_STREAM_QUEUE_DECISIONS:
            raise ValueError(f"unsupported_stream_queue_decision:{decision}")
        if not _clean(self.kind):
            raise ValueError("missing_stream_event_kind")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": _clean(self.kind),
            "lane": _clean(self.lane),
            "queue_decision": _clean(self.queue_decision),
            "replacement_fields": list(self.replacement_fields),
            "terminal": bool(self.terminal),
            "final": bool(self.final),
            "decision_bearing": bool(self.decision_bearing),
        }


HOSTED_STREAM_KIND_REGISTRY: Dict[str, HostedStreamKindSpec] = {
    "started": HostedStreamKindSpec(kind="started", lane="event", queue_decision="keep_first"),
    "heartbeat": HostedStreamKindSpec(kind="heartbeat", lane="event", queue_decision="latest"),
    "progress": HostedStreamKindSpec(kind="progress", lane="event", queue_decision="latest_by_key", replacement_fields=("key",)),
    "stdout": HostedStreamKindSpec(kind="stdout", lane="output", queue_decision="keep_first"),
    "stderr": HostedStreamKindSpec(kind="stderr", lane="output", queue_decision="keep_first"),
    "log": HostedStreamKindSpec(kind="log", lane="output", queue_decision="keep_first_by_window", replacement_fields=("source", "level")),
    "metric": HostedStreamKindSpec(kind="metric", lane="event", queue_decision="latest_by_key", replacement_fields=("name",)),
    "artifact": HostedStreamKindSpec(kind="artifact", lane="event", queue_decision="keep_first"),
    "host_call": HostedStreamKindSpec(kind="host_call", lane="control", queue_decision="non_droppable"),
    "host_response": HostedStreamKindSpec(kind="host_response", lane="control", queue_decision="non_droppable"),
    "approval": HostedStreamKindSpec(kind="approval", lane="audit", queue_decision="keep_first", decision_bearing=True),
    "result": HostedStreamKindSpec(kind="result", lane="control", queue_decision="non_droppable", terminal=True),
    "error": HostedStreamKindSpec(kind="error", lane="control", queue_decision="non_droppable", terminal=True),
    "canceled": HostedStreamKindSpec(kind="canceled", lane="control", queue_decision="non_droppable", terminal=True),
    "done": HostedStreamKindSpec(kind="done", lane="control", queue_decision="non_droppable", terminal=True, final=True),
    "state_notice": HostedStreamKindSpec(kind="state_notice", lane="audit", queue_decision="latest_by_key", replacement_fields=("scope", "partition")),
    "action_notice": HostedStreamKindSpec(kind="action_notice", lane="event", queue_decision="latest_by_key", replacement_fields=("action_id",)),
}


HOSTED_STREAM_EVENT_TYPES = [
    "started",
    "heartbeat",
    "progress",
    "stdout",
    "stderr",
    "log",
    "artifact",
    "metric",
    "result",
    "error",
    "canceled",
    "done",
    "host_call",
    "host_response",
    "approval",
    "state_notice",
    "action_notice",
]

HOSTED_IPC_MESSAGE_FAMILIES = [
    "hello",
    "rpc_call",
    "stream_open",
    "stream_recv",
    "stream_send",
    "stream_close",
    "shutdown",
]


def hosted_stream_kind_spec(kind: str) -> HostedStreamKindSpec:
    event_kind = _clean(kind)
    try:
        return HOSTED_STREAM_KIND_REGISTRY[event_kind]
    except KeyError as exc:
        raise ValueError(f"unsupported_stream_event_kind:{event_kind}") from exc


def hosted_stream_validate_kind(kind: str) -> str:
    return hosted_stream_kind_spec(kind).kind


def hosted_stream_kind_lane(kind: str) -> str:
    return hosted_stream_kind_spec(kind).lane


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
    admission: str = "submitted"
    concurrency_mode: str = "parallel"
    concurrency_group: Optional[str] = None
    resource_key: Optional[str] = None
    cancellation_requested: bool = False
    dispatch_started: bool = False

    def mark_started(self, *, timestamp: Optional[float] = None, engine_id: Optional[str] = None) -> None:
        self.started_at = float(timestamp if timestamp is not None else time.time())
        if engine_id is not None:
            self.engine_id = _clean(engine_id) or None
        self.status = "running"
        self.admission = "admitted"

    def mark_finished(self, status: str, *, reason: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        self.finished_at = float(timestamp if timestamp is not None else time.time())
        self.status = _clean(status) or "finished"
        self.reason = _clean(reason) or None

    def record_stream_event(self, event: Dict[str, Any]) -> None:
        row = dict(event or {})
        self.stream_event_count += 1
        event_type = _clean(row.get("type") or row.get("kind"))
        if event_type == "progress":
            payload = dict(row.get("payload") or {})
            if not payload:
                payload = {
                    key: value
                    for key, value in row.items()
                    if key not in {"type", "kind", "request_id", "stream_id", "instance_id", "sequence", "timestamp", "timestamp_ms", "dt_ms"}
                }
            self.latest_progress = {
                "type": event_type,
                "timestamp": float(
                    row.get("timestamp")
                    or (
                        float(str(row.get("timestamp_ms"))) / 1000.0
                        if row.get("timestamp_ms") is not None
                        else time.time()
                    )
                ),
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
            "admission": _clean(self.admission) or "submitted",
            "concurrency_mode": _clean(self.concurrency_mode) or "parallel",
            "concurrency_group": _clean(self.concurrency_group) or None,
            "resource_key": _clean(self.resource_key) or None,
            "cancellation_requested": bool(self.cancellation_requested),
            "dispatch_started": bool(self.dispatch_started),
        }


def _optional_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class HostedStreamContext:
    stream_id: Optional[str] = None
    request_id: Optional[str] = None
    instance_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _optional_dict(
            {
                "stream_id": _clean(self.stream_id) or None,
                "request_id": _clean(self.request_id) or None,
                "instance_id": _clean(self.instance_id) or None,
            }
        )

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "HostedStreamContext":
        row = dict(payload or {})
        return cls(
            stream_id=_clean(row.get("stream_id")) or None,
            request_id=_clean(row.get("request_id")) or None,
            instance_id=_clean(row.get("instance_id")) or None,
        )


@dataclass(frozen=True)
class HostedStreamLoss:
    output: int = 0
    event: int = 0
    audit: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "output": max(0, int(self.output or 0)),
            "event": max(0, int(self.event or 0)),
            "audit": max(0, int(self.audit or 0)),
        }

    def detected(self) -> bool:
        row = self.to_dict()
        return any(value > 0 for value in row.values())

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "HostedStreamLoss":
        row = dict(payload or {})
        return cls(output=int(row.get("output") or 0), event=int(row.get("event") or 0), audit=int(row.get("audit") or 0))


@dataclass(frozen=True)
class HostedStreamFrame:
    kind: str
    dt_ms: int = 0
    key: Optional[str] = None
    message: Optional[str] = None
    status: Optional[str] = None
    correlation_id: Optional[str] = None
    scope: Optional[str] = None
    operation: Optional[str] = None
    ref: Optional[str] = None
    expected_bytes: Optional[int] = None
    offset: Optional[int] = None
    length: Optional[int] = None
    text: Optional[str] = None
    data_b64: Optional[str] = None
    encoding: Optional[str] = None
    boundary: Optional[bool] = None
    final: Optional[bool] = None
    truncated: Optional[bool] = None
    dropped_before: Optional[bool] = None
    ack_id: Optional[str] = None
    stream_id: Optional[str] = None
    sequence: Optional[int] = None
    timestamp_ms: Optional[int] = None
    origin: Optional[str] = None
    source: Optional[str] = None
    visibility: Optional[str] = None
    redacted: Optional[bool] = None
    level: Optional[str] = None
    pct: Optional[float] = None
    current: Optional[float] = None
    total: Optional[float] = None
    reason: Optional[str] = None
    error_type: Optional[str] = None
    traceback_summary: Optional[str] = None
    call_id: Optional[str] = None
    method: Optional[str] = None
    provider_id: Optional[str] = None
    capability_id: Optional[str] = None
    arguments_ref: Optional[str] = None
    result_ref: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    partition: Optional[str] = None
    version: Optional[str] = None
    approval_id: Optional[str] = None
    action_id: Optional[str] = None
    card_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = _optional_dict(
            {
                "dt_ms": max(0, int(self.dt_ms or 0)),
                "kind": hosted_stream_validate_kind(self.kind),
                "key": _clean(self.key) or None,
                "message": self.message,
                "status": _clean(self.status) or None,
                "correlation_id": _clean(self.correlation_id) or None,
                "scope": _clean(self.scope) or None,
                "operation": _clean(self.operation) or None,
                "ref": _clean(self.ref) or None,
                "expected_bytes": int(self.expected_bytes) if self.expected_bytes is not None else None,
                "offset": max(0, int(self.offset)) if self.offset is not None else None,
                "length": max(0, int(self.length)) if self.length is not None else None,
                "text": self.text,
                "data_b64": self.data_b64,
                "encoding": _clean(self.encoding) or None,
                "boundary": bool(self.boundary) if self.boundary is not None else None,
                "final": bool(self.final) if self.final is not None else None,
                "truncated": bool(self.truncated) if self.truncated is not None else None,
                "dropped_before": bool(self.dropped_before) if self.dropped_before is not None else None,
                "ack_id": _clean(self.ack_id) or None,
                "stream_id": _clean(self.stream_id) or None,
                "sequence": max(0, int(self.sequence)) if self.sequence is not None else None,
                "timestamp_ms": max(0, int(self.timestamp_ms)) if self.timestamp_ms is not None else None,
                "origin": _clean(self.origin) or None,
                "source": _clean(self.source) or None,
                "visibility": _clean(self.visibility) or None,
                "redacted": bool(self.redacted) if self.redacted is not None else None,
                "level": _clean(self.level) or None,
                "pct": float(self.pct) if self.pct is not None else None,
                "current": float(self.current) if self.current is not None else None,
                "total": float(self.total) if self.total is not None else None,
                "reason": _clean(self.reason) or None,
                "error_type": _clean(self.error_type) or None,
                "traceback_summary": self.traceback_summary,
                "call_id": _clean(self.call_id) or None,
                "method": _clean(self.method) or None,
                "provider_id": _clean(self.provider_id) or None,
                "capability_id": _clean(self.capability_id) or None,
                "arguments_ref": _clean(self.arguments_ref) or None,
                "result_ref": _clean(self.result_ref) or None,
                "error": dict(self.error) if self.error is not None else None,
                "name": _clean(self.name) or None,
                "partition": _clean(self.partition) or None,
                "version": _clean(self.version) or None,
                "approval_id": _clean(self.approval_id) or None,
                "action_id": _clean(self.action_id) or None,
                "card_id": _clean(self.card_id) or None,
            }
        )
        for key, value in dict(self.extra or {}).items():
            if value is not None and key not in row:
                row[str(key)] = value
        return row

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HostedStreamFrame":
        row = dict(payload or {})
        known = {item.name for item in fields(cls)}
        extra = dict(row.pop("extra", {}) or {})
        kwargs = {key: row.pop(key) for key in list(row.keys()) if key in known and key != "extra"}
        extra.update(row)
        return cls(extra=extra, **kwargs)


@dataclass(frozen=True)
class HostedStreamBatch:
    context: HostedStreamContext = field(default_factory=HostedStreamContext)
    frames: List[HostedStreamFrame] = field(default_factory=list)
    sequence: int = 0
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    loss: HostedStreamLoss = field(default_factory=HostedStreamLoss)
    more: bool = False
    version: int = HOSTED_STREAM_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        if int(self.version or 0) != HOSTED_STREAM_CONTRACT_VERSION:
            raise ValueError(f"unsupported_stream_version:{self.version}")
        return {
            "version": HOSTED_STREAM_CONTRACT_VERSION,
            "context": self.context.to_dict(),
            "base": {
                "sequence": max(0, int(self.sequence or 0)),
                "timestamp_ms": max(0, int(self.timestamp_ms or 0)),
            },
            "loss": self.loss.to_dict(),
            "frames": [frame.to_dict() for frame in list(self.frames or [])],
            "more": bool(self.more),
        }

    def expanded_frames(self) -> List[Dict[str, Any]]:
        batch = self.to_dict()
        context = dict(batch.get("context") or {})
        base = dict(batch.get("base") or {})
        sequence_base = max(0, int(base.get("sequence") or 0))
        timestamp_base = max(0, int(base.get("timestamp_ms") or 0))
        out: List[Dict[str, Any]] = []
        for index, frame in enumerate(list(batch.get("frames") or [])):
            row = {**context, **dict(frame or {})}
            row.setdefault("sequence", sequence_base + index)
            row.setdefault("timestamp_ms", timestamp_base + max(0, int(row.get("dt_ms") or 0)))
            out.append(row)
        return out

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HostedStreamBatch":
        row = dict(payload or {})
        version = int(row.get("version") or 0)
        if version != HOSTED_STREAM_CONTRACT_VERSION:
            raise ValueError(f"unsupported_stream_version:{version}")
        base = dict(row.get("base") or {})
        frames = [HostedStreamFrame.from_dict(dict(frame or {})) for frame in list(row.get("frames") or [])]
        return cls(
            context=HostedStreamContext.from_dict(row.get("context")),
            frames=frames,
            sequence=int(base.get("sequence") or 0),
            timestamp_ms=int(base.get("timestamp_ms") or 0),
            loss=HostedStreamLoss.from_dict(row.get("loss")),
            more=bool(row.get("more")),
            version=version,
        )


class HostedStreamLossError(RuntimeError):
    def __init__(self, loss: Dict[str, int], *, batch: Optional[Dict[str, Any]] = None) -> None:
        self.loss = dict(loss or {})
        self.batch = dict(batch or {}) if batch is not None else None
        super().__init__(f"stream_loss:{stable_json(self.loss)}")


def hosted_stream_normalize_batch(
    batch: HostedStreamBatch | Dict[str, Any],
    *,
    on_loss: str = "mark",
) -> List[Dict[str, Any]]:
    mode = _clean(on_loss) or "mark"
    if mode not in {"mark", "raise"}:
        raise ValueError(f"unsupported_stream_loss_policy:{mode}")
    parsed = batch if isinstance(batch, HostedStreamBatch) else HostedStreamBatch.from_dict(dict(batch or {}))
    row = parsed.to_dict()
    loss = parsed.loss.to_dict()
    loss_detected = parsed.loss.detected()
    if loss_detected and mode == "raise":
        raise HostedStreamLossError(loss, batch=row)

    events: List[Dict[str, Any]] = []
    context = dict(row.get("context") or {})
    if loss_detected:
        events.append(
            {
                "kind": "stream_loss",
                **context,
                "loss": loss,
                "loss_detected": True,
            }
        )
    for frame in parsed.expanded_frames():
        events.append({**frame, "loss_detected": False})
    return events


@dataclass(frozen=True)
class HostedStreamEvent:
    type: str
    request_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_frame(self) -> HostedStreamFrame:
        event_type = hosted_stream_validate_kind(self.type)
        row = dict(self.payload or {})
        row.pop("kind", None)
        row.pop("type", None)
        return HostedStreamFrame.from_dict({"kind": event_type, **row})

    def to_batch(
        self,
        *,
        stream_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        loss: Optional[HostedStreamLoss] = None,
        more: bool = False,
    ) -> HostedStreamBatch:
        return HostedStreamBatch(
            context=HostedStreamContext(
                stream_id=_clean(stream_id) or None,
                request_id=_clean(self.request_id) or None,
                instance_id=_clean(instance_id) or None,
            ),
            frames=[self.to_frame()],
            sequence=max(0, int(self.sequence or 0)),
            timestamp_ms=max(0, int(float(self.timestamp) * 1000)),
            loss=loss or HostedStreamLoss(),
            more=more,
        )

    def to_dict(self) -> Dict[str, Any]:
        event_type = hosted_stream_validate_kind(self.type)
        return {
            "type": event_type,
            "request_id": _clean(self.request_id),
            "sequence": max(0, int(self.sequence or 0)),
            "timestamp": float(self.timestamp),
            "payload": dict(self.payload or {}),
        }


def hosted_stream_cancel_message(request_id: str, *, reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "action": "cancel",
        "request_id": _clean(request_id),
        "reason": _clean(reason) or None,
    }


def hosted_registration_environment_metadata(
    *,
    environment: Dict[str, Any],
    runtime_kind: str,
    profile: str,
) -> Dict[str, Any]:
    env = dict(environment or {})
    return {
        **env,
        "environment_key": _clean(env.get("environment_key")) or None,
        "environment_name": _clean(env.get("environment_name")) or None,
        "workflow_runtime_kind": _clean(runtime_kind) or _clean(env.get("workflow_runtime_kind")) or None,
        "workflow_profile": _clean(profile) or _clean(env.get("workflow_profile")) or None,
        "sandbox_policy_hash": _clean(env.get("sandbox_policy_hash")) or None,
        "required_imports": _unique_strings(env.get("required_imports") or []),
        "package_pins": _string_map(env.get("package_pins") or {}),
        "dependency_lock_hash": _clean(env.get("dependency_lock_hash")) or None,
        "install_status": env.get("install_status"),
    }


def hosted_resource_response(
    *,
    sandbox_kind: str,
    profile: str,
    environment_key: str,
    engine_id: Optional[str] = None,
    pool: Optional[Dict[str, Any]] = None,
    resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        **dict(resources or {}),
        "status": str(dict(resources or {}).get("status") or "ok"),
        "sandbox_kind": _clean(sandbox_kind) or None,
        "profile": _clean(profile) or None,
        "engine_id": _clean(engine_id) or None,
        "environment_key": _clean(environment_key) or None,
        "workflow_pool": dict(pool or {}) or None,
    }


def hosted_cancellation_result(
    *,
    request_id: str,
    environment_key: str,
    canceled: bool,
    reason: str = "canceled",
    worker_result: Optional[Dict[str, Any]] = None,
    pool_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "status": "ok" if canceled else "not_found",
        "request_id": _clean(request_id),
        "environment_key": _clean(environment_key) or None,
        "canceled": bool(canceled),
        "reason": _clean(reason) or None,
        "worker_cancel": dict(worker_result or {}) or None,
        "workflow_pool_cancel": dict(pool_result or {}) or None,
    }


def hosted_log_summary(
    *,
    stdout: str = "",
    stderr: str = "",
    max_bytes: int = 4096,
) -> Dict[str, Any]:
    limit = max(0, int(max_bytes or 0))

    def _truncate(value: str) -> tuple[str, bool]:
        text = str(value or "")
        raw = text.encode("utf-8", errors="replace")
        if limit <= 0 or len(raw) <= limit:
            return text, False
        clipped = raw[:limit].decode("utf-8", errors="replace")
        return clipped, True

    out_stdout, stdout_truncated = _truncate(stdout)
    out_stderr, stderr_truncated = _truncate(stderr)
    parts = []
    if out_stdout:
        parts.append(out_stdout)
    if out_stderr:
        parts.append(out_stderr)
    summary, summary_truncated = _truncate("\n".join(parts))
    return {
        "stdout": out_stdout,
        "stderr": out_stderr,
        "summary": summary,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "summary_truncated": summary_truncated,
        "output_limit_bytes": limit,
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
    queue_policy: str = "fail_fast"
    queue_depth: int = 0
    queue_timeout_seconds: float = 0.0
    queued_calls: int = 0

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
            "queued_calls": max(0, int(self.queued_calls or 0)),
            "queue_policy": _clean(self.queue_policy) or "fail_fast",
            "queue_depth": max(0, int(self.queue_depth or 0)),
            "queue_timeout_seconds": max(0.0, float(self.queue_timeout_seconds or 0.0)),
            "logical_call_capacity": max(0, int(self.desired_capacity or 0)),
            "worker_process_count": len(self.workers),
            "execution_model": "threaded_worker",
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
    "HostedStreamBatch",
    "HostedStreamContext",
    "HostedStreamEvent",
    "HostedStreamFrame",
    "HostedStreamKindSpec",
    "HostedStreamLoss",
    "HostedStreamLossError",
    "HostedWorkerSlot",
    "HOSTED_IPC_MESSAGE_FAMILIES",
    "HOSTED_STREAM_CONTRACT_VERSION",
    "HOSTED_STREAM_EVENT_TYPES",
    "HOSTED_STREAM_KIND_REGISTRY",
    "HOSTED_STREAM_LANES",
    "HOSTED_STREAM_QUEUE_DECISIONS",
    "hosted_cancellation_result",
    "hosted_log_summary",
    "hosted_registration_environment_metadata",
    "hosted_resource_response",
    "hosted_stream_cancel_message",
    "hosted_stream_kind_lane",
    "hosted_stream_kind_spec",
    "hosted_stream_normalize_batch",
    "hosted_stream_validate_kind",
    "normalize_sandbox_policy",
    "sandbox_policy_hash",
    "stable_hash",
    "stable_json",
]
