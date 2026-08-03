"""Internal hosted process sandbox base primitives.

This is not a public sandbox kind. It is a small composition layer over the
runtime pool models so concrete runtimes can share pool, capacity, request
status, progress, and cancellation plumbing without inheriting workflow,
toolbox, or model semantics.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .runtime_base import (
    HOSTED_STREAM_LANES,
    HostedPoolKey,
    HostedRequestLifecycle,
    HostedStreamBatch,
    HostedStreamContext,
    HostedStreamEvent,
    HostedStreamFrame,
    HostedStreamLoss,
    HostedWorkerSlot,
    hosted_stream_cancel_message,
    hosted_stream_kind_lane,
    hosted_stream_kind_spec,
    hosted_stream_normalize_batch,
)
from .runtime_pool import HostedProcessPool, HostedProcessPoolRegistry, WorkerFactory


_PROCESS_STREAM_LANE_ORDER = ["control", "audit", "event", "output"]
_PROCESS_STREAM_DROPPABLE_LANE_ORDER = ["output", "event", "audit"]


@dataclass
class HostedProcessStreamSession:
    stream_id: str
    environment_key: str
    request_id: str
    profile: str
    max_events: int = 256
    events_by_lane: Dict[str, Deque[Dict[str, object]]] = field(
        default_factory=lambda: {lane: deque() for lane in HOSTED_STREAM_LANES}
    )
    closed: bool = False
    canceled: bool = False
    sequence: int = 0
    dropped_frame_count: int = 0
    pending_loss: Dict[str, int] = field(default_factory=lambda: {"output": 0, "event": 0, "audit": 0})
    output_offsets: Dict[str, int] = field(default_factory=dict)
    non_ack_output_limit_bytes: int = 4 * 1024 * 1024
    non_ack_output_bytes: int = 0
    accepted_output_stream: bool = False
    output_credit_bytes: int = 0
    output_inflight_bytes: int = 0
    output_acked_bytes: int = 0
    output_max_chunk_size: Optional[int] = None
    output_closed_by_client: bool = False
    output_close_reason: Optional[str] = None

    def _queued_count(self) -> int:
        return sum(len(queue) for queue in self.events_by_lane.values())

    def _loss_lane(self, lane: str) -> Optional[str]:
        if lane in {"output", "event", "audit"}:
            return lane
        return None

    def _record_loss(self, lane: str, count: int = 1) -> None:
        loss_lane = self._loss_lane(lane)
        if loss_lane is not None:
            self.pending_loss[loss_lane] = max(0, int(self.pending_loss.get(loss_lane, 0))) + max(0, int(count or 0))
        self.dropped_frame_count += max(0, int(count or 0))

    def _replacement_value(self, row: Dict[str, object]) -> str:
        spec = hosted_stream_kind_spec(str(row.get("kind") or row.get("type") or ""))
        payload = self._payload_from_frame(row)
        if spec.queue_decision == "latest":
            return str(row.get("kind") or row.get("type") or "")
        for field_name in spec.replacement_fields:
            value = str(payload.get(field_name) or row.get(field_name) or "").strip()
            if value:
                return f"{field_name}:{value}"
        return str(row.get("kind") or row.get("type") or "")

    def _replace_existing_latest(self, lane: str, row: Dict[str, object]) -> bool:
        spec = hosted_stream_kind_spec(str(row.get("kind") or row.get("type") or ""))
        if spec.queue_decision not in {"latest", "latest_by_key"}:
            return False
        queue = self.events_by_lane.setdefault(lane, deque())
        replacement_value = self._replacement_value(row)
        for index, existing in enumerate(list(queue)):
            if str(existing.get("kind") or existing.get("type") or "") == str(row.get("kind") or row.get("type") or "") and self._replacement_value(existing) == replacement_value:
                queue[index] = row
                self._record_loss(lane)
                return True
        return False

    def _drop_one_for_capacity(self) -> bool:
        for lane in _PROCESS_STREAM_DROPPABLE_LANE_ORDER:
            queue = self.events_by_lane.setdefault(lane, deque())
            if queue:
                dropped = queue.popleft()
                if lane == "output" and not self._requires_ack(self._payload_from_frame(dropped)):
                    self.non_ack_output_bytes = max(0, int(self.non_ack_output_bytes or 0) - self._output_length(dropped))
                self._record_loss(lane)
                return True
        return False

    def _prepare_output_payload(self, event_type: str, payload: Dict[str, object]) -> Dict[str, object]:
        row = dict(payload or {})
        text_value = row.get("text")
        if text_value is None and event_type == "log":
            text_value = row.get("message")
        if isinstance(text_value, str):
            raw_len = len(text_value.encode("utf-8", errors="replace"))
            offset = max(0, int(self.output_offsets.get(event_type, 0)))
            row.setdefault("encoding", "utf-8")
            row.setdefault("offset", offset)
            row.setdefault("length", raw_len)
            row.setdefault("boundary", event_type == "log" or text_value.endswith("\n"))
            self.output_offsets[event_type] = offset + raw_len
        return row

    def _requires_ack(self, payload: Dict[str, object]) -> bool:
        return bool(payload.get("requires_ack") or payload.get("ack_id"))

    def _output_length(self, payload: Dict[str, object]) -> int:
        row = self._payload_from_frame(payload) if "kind" in payload else dict(payload or {})
        return max(0, int(row.get("length") or 0))

    def _consume_output_credit(self, payload: Dict[str, object]) -> None:
        if not self._requires_ack(payload):
            return
        if self.output_closed_by_client:
            raise ValueError("stream_abandoned")
        if not self.accepted_output_stream:
            raise ValueError("stream_not_accepted")
        length = max(0, int(payload.get("length") or 0))
        if length > max(0, int(self.output_credit_bytes or 0)):
            raise ValueError("stream_credit_exhausted")
        self.output_credit_bytes -= length
        self.output_inflight_bytes += length

    def accept_output_stream(self, *, credit_bytes: int, max_chunk_size: Optional[int] = None) -> Dict[str, object]:
        credit = max(0, int(credit_bytes or 0))
        self.accepted_output_stream = True
        self.output_closed_by_client = False
        self.output_close_reason = None
        self.output_credit_bytes += credit
        if max_chunk_size is not None:
            self.output_max_chunk_size = max(1, int(max_chunk_size or 1))
        return {
            "accepted": True,
            "stream_id": self.stream_id,
            "credit_bytes": max(0, int(self.output_credit_bytes or 0)),
            "max_chunk_size": self.output_max_chunk_size,
        }

    def ack_output_stream(self, *, consumed_bytes: int = 0, additional_credit_bytes: int = 0, ack_id: Optional[str] = None) -> Dict[str, object]:
        consumed = max(0, int(consumed_bytes or 0))
        additional = max(0, int(additional_credit_bytes or 0))
        self.output_inflight_bytes = max(0, int(self.output_inflight_bytes or 0) - consumed)
        self.output_acked_bytes += consumed
        self.output_credit_bytes += additional
        return {
            "accepted": True,
            "stream_id": self.stream_id,
            "ack_id": str(ack_id or "").strip() or None,
            "acked_bytes": max(0, int(self.output_acked_bytes or 0)),
            "inflight_bytes": max(0, int(self.output_inflight_bytes or 0)),
            "credit_bytes": max(0, int(self.output_credit_bytes or 0)),
        }

    def abandon_output_stream(self, *, reason: Optional[str] = None) -> Dict[str, object]:
        self.output_closed_by_client = True
        self.output_close_reason = str(reason or "").strip() or "stream_closed_by_client"
        return {
            "accepted": True,
            "stream_id": self.stream_id,
            "closed": True,
            "reason": self.output_close_reason,
        }

    def _payload_from_frame(self, row: Dict[str, object]) -> Dict[str, object]:
        return {
            key: value
            for key, value in dict(row or {}).items()
            if key
            not in {
                "kind",
                "type",
                "dt_ms",
                "request_id",
                "stream_id",
                "instance_id",
                "sequence",
                "timestamp",
                "timestamp_ms",
                "origin",
                "visibility",
                "loss_detected",
            }
        }

    def _event_row(self, event_type: str, payload: Dict[str, object]) -> Dict[str, object]:
        row = dict(payload or {})
        row.pop("kind", None)
        row.pop("type", None)
        frame = HostedStreamFrame.from_dict({"kind": event_type, **row})
        batch = HostedStreamBatch(
            context=HostedStreamContext(stream_id=self.stream_id, request_id=self.request_id),
            frames=[frame],
            sequence=self.sequence,
            timestamp_ms=int(time.time() * 1000),
        )
        return batch.expanded_frames()[0]

    def append(self, event_type: str, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        self.sequence += 1
        lane = hosted_stream_kind_lane(event_type)
        spec = hosted_stream_kind_spec(event_type)
        payload_row = dict(payload or {})
        if lane == "output":
            payload_row = self._prepare_output_payload(event_type, payload_row)
            self._consume_output_credit(payload_row)
            if not self._requires_ack(payload_row):
                limit = max(0, int(self.non_ack_output_limit_bytes or 0))
                length = self._output_length(payload_row)
                if limit > 0 and max(0, int(self.non_ack_output_bytes or 0)) + length > limit:
                    self._record_loss(lane)
                    return self._event_row(event_type, payload_row)
        row = self._event_row(event_type, payload_row)
        if self._replace_existing_latest(lane, row):
            return row
        capacity = max(1, int(self.max_events or 1))
        if self._queued_count() >= capacity and lane != "control" and spec.queue_decision in {"keep_first", "keep_first_by_window"}:
            self._record_loss(lane)
            return row
        self.events_by_lane.setdefault(lane, deque()).append(row)
        if lane == "output" and not self._requires_ack(payload_row):
            self.non_ack_output_bytes += self._output_length(payload_row)
        while self._queued_count() > capacity:
            if not self._drop_one_for_capacity():
                break
        return row

    def recv(self, max_items: int) -> List[Dict[str, object]]:
        limit = max(1, int(max_items or 1))
        queued: List[tuple[str, Dict[str, object]]] = []
        for lane in _PROCESS_STREAM_LANE_ORDER:
            queued.extend((lane, dict(row)) for row in self.events_by_lane.setdefault(lane, deque()))
        if len(queued) <= limit:
            selected = sorted(queued, key=lambda item: int(item[1].get("sequence") or 0))
        else:
            control = sorted((item for item in queued if item[0] == "control"), key=lambda item: int(item[1].get("sequence") or 0))
            selected = control[:limit]
            selected_sequences = {int(row.get("sequence") or 0) for _, row in selected}
            if len(selected) < limit:
                remainder = sorted(
                    (item for item in queued if int(item[1].get("sequence") or 0) not in selected_sequences),
                    key=lambda item: int(item[1].get("sequence") or 0),
                )
                selected.extend(remainder[: limit - len(selected)])
            selected = sorted(selected, key=lambda item: int(item[1].get("sequence") or 0))

        selected_sequences = {int(row.get("sequence") or 0) for _, row in selected}
        for lane in _PROCESS_STREAM_LANE_ORDER:
            queue = self.events_by_lane.setdefault(lane, deque())
            retained = deque(row for row in queue if int(row.get("sequence") or 0) not in selected_sequences)
            self.events_by_lane[lane] = retained
        return [dict(row) for _, row in selected]

    def retained_count(self) -> int:
        return self._queued_count()

    def take_loss(self) -> HostedStreamLoss:
        loss = HostedStreamLoss(
            output=max(0, int(self.pending_loss.get("output", 0))),
            event=max(0, int(self.pending_loss.get("event", 0))),
            audit=max(0, int(self.pending_loss.get("audit", 0))),
        )
        self.pending_loss = {"output": 0, "event": 0, "audit": 0}
        return loss

    def batch_from_events(self, events: List[Dict[str, object]], *, loss: HostedStreamLoss, more: bool) -> Dict[str, object]:
        frames: List[HostedStreamFrame] = []
        sequence = 0
        timestamp_ms = int(time.time() * 1000)
        for index, event in enumerate(events):
            payload = dict(event.get("payload") or self._payload_from_frame(event))
            if payload.get("host_call_id") is not None and payload.get("call_id") is None:
                payload["call_id"] = payload.get("host_call_id")
            payload.pop("kind", None)
            payload.pop("type", None)
            frame = HostedStreamFrame.from_dict(
                {
                    "kind": str(event.get("kind") or event.get("type") or ""),
                    **payload,
                    "sequence": max(0, int(event.get("sequence") or 0)),
                    "timestamp_ms": max(
                        0,
                        int(event.get("timestamp_ms") or int(float(event.get("timestamp") or 0.0) * 1000)),
                    ),
                }
            )
            if index == 0:
                sequence = max(0, int(event.get("sequence") or 0))
                timestamp_ms = max(
                    0,
                    int(event.get("timestamp_ms") or int(float(event.get("timestamp") or time.time()) * 1000)),
                )
            frames.append(frame)
        return HostedStreamBatch(
            context=HostedStreamContext(stream_id=self.stream_id, request_id=self.request_id),
            frames=frames,
            sequence=sequence,
            timestamp_ms=timestamp_ms,
            loss=loss,
            more=more,
        ).to_dict()


class HostedProcessSandboxBase:
    """Language-neutral internal base for concrete hosted process runtimes."""

    sandbox_kind: str = "generic"

    def __init__(self, *, pool_registry: Optional[HostedProcessPoolRegistry] = None) -> None:
        self.pool_registry = pool_registry or HostedProcessPoolRegistry()
        self._streams: Dict[str, HostedProcessStreamSession] = {}

    def pool_key(self, environment_key: str) -> HostedPoolKey:
        return HostedPoolKey(sandbox_kind=self.sandbox_kind, environment_key=str(environment_key or "").strip())

    def get_or_create_pool(self, environment_key: str, *, desired_capacity: int = 1) -> HostedProcessPool:
        return self.pool_registry.get_or_create(self.pool_key(environment_key), desired_capacity=desired_capacity)

    def resources(self, environment_key: str) -> Dict[str, object]:
        pool = self.pool_registry.get(self.pool_key(environment_key))
        if pool is None:
            return {
                "status": "not_found",
                "sandbox_kind": self.sandbox_kind,
                "environment_key": str(environment_key or "").strip(),
            }
        return pool.resources()

    def set_capacity(self, environment_key: str, *, capacity: int) -> Dict[str, object]:
        pool = self.get_or_create_pool(environment_key, desired_capacity=capacity)
        return {
            "status": "ok",
            "sandbox_kind": self.sandbox_kind,
            "environment_key": str(environment_key or "").strip(),
            "capacity": pool.set_capacity(capacity),
            "workflow_pool": pool.resources(),
        }

    def submit_request(
        self,
        *,
        environment_key: str,
        request_id: str,
        profile: str,
        factory: Optional[WorkerFactory] = None,
        desired_capacity: int = 1,
        operation_id: Optional[str] = None,
        input_bytes: Optional[int] = None,
        queue_policy: str = "fail_fast",
        queue_depth: int = 0,
        queue_timeout_seconds: float = 0.0,
        concurrency: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        pool = self.pool_registry.get_or_create(
            self.pool_key(environment_key),
            desired_capacity=desired_capacity,
            queue_policy=queue_policy,
            queue_depth=queue_depth,
            queue_timeout_seconds=queue_timeout_seconds,
        )
        lifecycle = HostedRequestLifecycle(
            request_id=str(request_id or "").strip(),
            environment_key=str(environment_key or "").strip(),
            sandbox_kind=self.sandbox_kind,
            profile=str(profile or "").strip() or "default",
            operation_id=str(operation_id or "").strip() or None,
            input_bytes=input_bytes,
            submitted_at=time.time(),
        )
        return pool.submit_request(
            lifecycle,
            factory=factory,
            queue_policy=queue_policy,
            queue_depth=queue_depth,
            queue_timeout_seconds=queue_timeout_seconds,
            concurrency=concurrency,
        )

    def finish_request(
        self,
        *,
        environment_key: str,
        request_id: str,
        status: str = "ok",
        reason: Optional[str] = None,
        output_bytes: Optional[int] = None,
    ) -> Dict[str, object]:
        pool = self.pool_registry.get(self.pool_key(environment_key))
        if pool is None:
            return {"status": "not_found", "environment_key": str(environment_key or "").strip(), "request_id": str(request_id or "").strip()}
        return pool.finish_request(request_id, status=status, reason=reason, output_bytes=output_bytes)

    def cancel_request(self, *, environment_key: str, request_id: str) -> Dict[str, object]:
        pool = self.pool_registry.get(self.pool_key(environment_key))
        if pool is None:
            return {"status": "not_found", "environment_key": str(environment_key or "").strip(), "request_id": str(request_id or "").strip()}
        return pool.cancel_request(request_id)

    def request_status(self, *, environment_key: str, request_id: str) -> Dict[str, object]:
        return self.pool_registry.request_status(self.pool_key(environment_key), request_id)

    def record_stream_event(self, *, environment_key: str, request_id: str, event: HostedStreamEvent | Dict[str, object]) -> Dict[str, object]:
        pool = self.pool_registry.get(self.pool_key(environment_key))
        if pool is None:
            return {"status": "not_found", "environment_key": str(environment_key or "").strip(), "request_id": str(request_id or "").strip()}
        return pool.record_stream_event(request_id, event)

    def stream_open(
        self,
        *,
        environment_key: str,
        request_id: str,
        profile: str,
        factory: Optional[WorkerFactory] = None,
        desired_capacity: int = 1,
        max_events: int = 256,
    ) -> Dict[str, object]:
        rid = str(request_id or "").strip() or f"{self.sandbox_kind}-stream-{int(time.time() * 1000)}"
        scheduled = self.submit_request(
            environment_key=environment_key,
            request_id=rid,
            profile=profile,
            factory=factory,
            desired_capacity=desired_capacity,
        )
        if str(scheduled.get("status") or "") != "ok":
            return scheduled
        stream_id = f"{self.sandbox_kind}-{uuid.uuid4().hex}"
        session = HostedProcessStreamSession(
            stream_id=stream_id,
            environment_key=str(environment_key or "").strip(),
            request_id=rid,
            profile=str(profile or "").strip() or "default",
            max_events=max_events,
        )
        started = session.append("started", {"environment_key": session.environment_key, "profile": session.profile})
        self._streams[stream_id] = session
        self.record_stream_event(environment_key=session.environment_key, request_id=rid, event=started)
        return {"status": "ok", "stream_id": stream_id, "request_id": rid, "environment_key": session.environment_key}

    def stream_emit(self, *, stream_id: str, event_type: str, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        sid = str(stream_id or "").strip()
        session = self._streams.get(sid)
        if session is None:
            return {"status": "not_found", "stream_id": sid}
        try:
            event = session.append(event_type, payload)
        except ValueError as exc:
            return {"status": "error", "stream_id": sid, "reason": str(exc), "event_type": str(event_type or "").strip()}
        self.record_stream_event(environment_key=session.environment_key, request_id=session.request_id, event=event)
        return {"status": "ok", "stream_id": sid, "event": event}

    def event_subscribe(self, *, stream_id: str, max_items: int = 64) -> Dict[str, object]:
        sid = str(stream_id or "").strip()
        session = self._streams.get(sid)
        if session is None:
            return {"status": "not_found", "stream_id": sid}
        events = session.recv(max_items)
        loss = session.take_loss()
        batch = session.batch_from_events(events, loss=loss, more=session.retained_count() > 0)
        return {
            "status": "ok",
            "stream_id": sid,
            "request_id": session.request_id,
            "batch": batch,
            "normalized_events": hosted_stream_normalize_batch(batch, on_loss="mark"),
            "closed": session.closed,
            "canceled": session.canceled,
        }

    def stream_send(self, *, stream_id: str, message: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        sid = str(stream_id or "").strip()
        session = self._streams.get(sid)
        if session is None:
            return {"status": "not_found", "stream_id": sid}
        msg = dict(message or {})
        action = str(msg.get("action") or "").strip()
        if action == "stream_accept":
            credit = int(msg.get("initial_credit_bytes") or msg.get("credit_bytes") or 1048576)
            max_chunk_size = msg.get("max_chunk_size")
            return {"status": "ok", "stream_id": sid, "action": action, **session.accept_output_stream(credit_bytes=credit, max_chunk_size=int(max_chunk_size) if max_chunk_size is not None else None)}
        if action == "stream_ack":
            return {
                "status": "ok",
                "stream_id": sid,
                "action": action,
                **session.ack_output_stream(
                    consumed_bytes=int(msg.get("consumed_bytes") or msg.get("length") or 0),
                    additional_credit_bytes=int(msg.get("additional_credit_bytes") or msg.get("credit_bytes") or 0),
                    ack_id=str(msg.get("ack_id") or ""),
                ),
            }
        if action == "stream_close":
            return {"status": "ok", "stream_id": sid, "action": action, **session.abandon_output_stream(reason=str(msg.get("reason") or ""))}
        if action != "cancel":
            return {"status": "ok", "stream_id": sid, "accepted": False, "message": "unsupported_action"}
        session.canceled = True
        cancel = hosted_stream_cancel_message(session.request_id, reason=str(msg.get("reason") or "client_cancel"))
        event = session.append("canceled", cancel)
        self.record_stream_event(environment_key=session.environment_key, request_id=session.request_id, event=event)
        pool_cancel = self.cancel_request(environment_key=session.environment_key, request_id=session.request_id)
        return {"status": "ok", "stream_id": sid, "accepted": True, "message": cancel, "workflow_pool_cancel": pool_cancel}

    def stream_close(self, *, stream_id: str) -> Dict[str, object]:
        sid = str(stream_id or "").strip()
        session = self._streams.pop(sid, None)
        if session is None:
            return {"status": "ok", "stream_id": sid, "closed": False, "status_message": "not_found"}
        was_closed = bool(session.closed)
        session.closed = True
        if not session.canceled and not was_closed:
            event = session.append("done", {"closed": True})
            self.record_stream_event(environment_key=session.environment_key, request_id=session.request_id, event=event)
            self.finish_request(environment_key=session.environment_key, request_id=session.request_id, status="ok")
        return {"status": "ok", "stream_id": sid, "closed": True, "request_id": session.request_id}

    @staticmethod
    def worker_slot(
        *,
        engine_id: str,
        environment_key: str,
        capacity: int = 1,
        pid: Optional[int] = None,
        status: str = "unknown",
    ) -> HostedWorkerSlot:
        return HostedWorkerSlot(
            engine_id=str(engine_id or "").strip(),
            environment_key=str(environment_key or "").strip(),
            capacity=max(1, int(capacity or 1)),
            pid=pid,
            status=str(status or "").strip() or "unknown",
        )


__all__ = ["HostedProcessSandboxBase", "HostedProcessStreamSession"]
