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
from typing import Deque, Dict, Optional

from .runtime_base import HostedPoolKey, HostedRequestLifecycle, HostedStreamEvent, HostedWorkerSlot, hosted_stream_cancel_message
from .runtime_pool import HostedProcessPool, HostedProcessPoolRegistry, WorkerFactory


@dataclass
class HostedProcessStreamSession:
    stream_id: str
    environment_key: str
    request_id: str
    profile: str
    max_events: int = 256
    events: Deque[Dict[str, object]] = field(default_factory=deque)
    closed: bool = False
    canceled: bool = False
    sequence: int = 0

    def append(self, event_type: str, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        self.sequence += 1
        row = HostedStreamEvent(
            type=event_type,
            request_id=self.request_id,
            sequence=self.sequence,
            payload=dict(payload or {}),
        ).to_dict()
        self.events.append(row)
        while len(self.events) > max(1, int(self.max_events or 1)):
            self.events.popleft()
        return row

    def recv(self, max_items: int) -> list[Dict[str, object]]:
        out: list[Dict[str, object]] = []
        limit = max(1, int(max_items or 1))
        while self.events and len(out) < limit:
            out.append(dict(self.events.popleft()))
        return out


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
    ) -> Dict[str, object]:
        pool = self.get_or_create_pool(environment_key, desired_capacity=desired_capacity)
        lifecycle = HostedRequestLifecycle(
            request_id=str(request_id or "").strip(),
            environment_key=str(environment_key or "").strip(),
            sandbox_kind=self.sandbox_kind,
            profile=str(profile or "").strip() or "default",
            operation_id=str(operation_id or "").strip() or None,
            input_bytes=input_bytes,
            submitted_at=time.time(),
        )
        return pool.submit_request(lifecycle, factory=factory)

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
        event = session.append(event_type, payload)
        self.record_stream_event(environment_key=session.environment_key, request_id=session.request_id, event=event)
        return {"status": "ok", "stream_id": sid, "event": event}

    def stream_recv(self, *, stream_id: str, max_items: int = 64) -> Dict[str, object]:
        sid = str(stream_id or "").strip()
        session = self._streams.get(sid)
        if session is None:
            return {"status": "not_found", "stream_id": sid}
        events = session.recv(max_items)
        return {
            "status": "ok",
            "stream_id": sid,
            "request_id": session.request_id,
            "events": events,
            "closed": session.closed,
            "canceled": session.canceled,
        }

    def stream_send(self, *, stream_id: str, message: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        sid = str(stream_id or "").strip()
        session = self._streams.get(sid)
        if session is None:
            return {"status": "not_found", "stream_id": sid}
        msg = dict(message or {})
        if str(msg.get("action") or "").strip() != "cancel":
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
        session.closed = True
        if not session.canceled:
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
