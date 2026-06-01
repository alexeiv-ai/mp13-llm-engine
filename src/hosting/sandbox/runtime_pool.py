"""Internal hosted process pool primitives.

The pool registry is intentionally independent from concrete worker spawn code.
Callers provide a small worker factory when a pool needs its first worker; later
phases can wire that factory to EngineHostService spawn/ensure behavior.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from .runtime_base import (
    HostedPoolKey,
    HostedPoolMetrics,
    HostedRequestLifecycle,
    HostedStreamEvent,
    HostedWorkerSlot,
)


WorkerFactory = Callable[[HostedPoolKey, int], HostedWorkerSlot]


@dataclass
class HostedProcessPool:
    pool_key: HostedPoolKey
    desired_capacity: int = 1
    recent_limit: int = 100
    workers: List[HostedWorkerSlot] = field(default_factory=list)
    requests: Dict[str, HostedRequestLifecycle] = field(default_factory=dict)
    recent_requests: Deque[HostedRequestLifecycle] = field(default_factory=deque)
    saturation_count: int = 0
    timeout_count: int = 0
    cancellation_count: int = 0
    error_count: int = 0
    errors_by_reason: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.desired_capacity = max(1, int(self.desired_capacity or 1))
        self.recent_limit = max(1, int(self.recent_limit or 100))

    def set_capacity(self, capacity: int) -> int:
        self.desired_capacity = max(1, int(capacity or 1))
        if len(self.workers) == 1:
            self.workers[0].capacity = self.desired_capacity
        return self.desired_capacity

    def ensure_worker(self, factory: WorkerFactory) -> HostedWorkerSlot:
        if self.workers:
            if len(self.workers) == 1:
                self.workers[0].capacity = self.desired_capacity
            return self.workers[0]
        worker = factory(self.pool_key, self.desired_capacity)
        worker.environment_key = self.pool_key.normalized()["environment_key"]
        worker.capacity = max(1, int(worker.capacity or self.desired_capacity))
        self.workers.append(worker)
        return worker

    def select_worker(self, factory: Optional[WorkerFactory] = None) -> Optional[HostedWorkerSlot]:
        if not self.workers and factory is not None:
            return self.ensure_worker(factory)
        for worker in self.workers:
            if worker.available_slots() > 0:
                return worker
        self.saturation_count += 1
        return None

    def submit_request(
        self,
        request: HostedRequestLifecycle,
        *,
        factory: Optional[WorkerFactory] = None,
        start_timestamp: Optional[float] = None,
    ) -> Dict[str, object]:
        self.requests[request.request_id] = request
        worker = self.select_worker(factory)
        if worker is None:
            request.mark_finished("error", reason="capacity_exceeded", timestamp=start_timestamp)
            self._remember_request(request)
            return {"status": "error", "reason": "capacity_exceeded", "request": request.to_dict()}
        if request.request_id not in worker.active_request_ids:
            worker.active_request_ids.append(request.request_id)
        request.mark_started(timestamp=start_timestamp, engine_id=worker.engine_id)
        return {"status": "ok", "worker": worker.to_dict(), "request": request.to_dict()}

    def finish_request(
        self,
        request_id: str,
        *,
        status: str = "ok",
        reason: Optional[str] = None,
        timestamp: Optional[float] = None,
        output_bytes: Optional[int] = None,
    ) -> Dict[str, object]:
        rid = str(request_id or "").strip()
        request = self.requests.get(rid)
        if request is None:
            return {"status": "not_found", "request_id": rid}
        if output_bytes is not None:
            request.output_bytes = int(output_bytes)
        request.mark_finished(status, reason=reason, timestamp=timestamp)
        for worker in self.workers:
            worker.active_request_ids = [item for item in worker.active_request_ids if str(item or "").strip() != rid]
        if status == "timeout":
            self.timeout_count += 1
        elif status == "canceled":
            self.cancellation_count += 1
        elif status not in {"ok", "success", "done"}:
            self.error_count += 1
            key = str(reason or status or "error").strip() or "error"
            self.errors_by_reason[key] = int(self.errors_by_reason.get(key, 0)) + 1
        self._remember_request(request)
        return {"status": "ok", "request": request.to_dict()}

    def cancel_request(self, request_id: str, *, timestamp: Optional[float] = None) -> Dict[str, object]:
        return self.finish_request(request_id, status="canceled", reason="canceled", timestamp=timestamp)

    def record_stream_event(self, request_id: str, event: HostedStreamEvent | Dict[str, object]) -> Dict[str, object]:
        rid = str(request_id or "").strip()
        request = self.requests.get(rid)
        if request is None:
            return {"status": "not_found", "request_id": rid}
        row = event.to_dict() if isinstance(event, HostedStreamEvent) else dict(event or {})
        request.record_stream_event(row)
        return {"status": "ok", "request": request.to_dict(), "event": row}

    def request_status(self, request_id: str) -> Dict[str, object]:
        rid = str(request_id or "").strip()
        request = self.requests.get(rid)
        if request is not None:
            return {"status": "ok", "request": request.to_dict(), "source": "active"}
        for recent in reversed(self.recent_requests):
            if str(recent.request_id or "").strip() == rid:
                return {"status": "ok", "request": recent.to_dict(), "source": "recent"}
        return {"status": "not_found", "request_id": rid}

    def _remember_request(self, request: HostedRequestLifecycle) -> None:
        self.recent_requests.append(request)
        while len(self.recent_requests) > self.recent_limit:
            self.recent_requests.popleft()

    def resources(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "pool_id": self.pool_key.pool_id(),
            "sandbox_kind": self.pool_key.normalized()["sandbox_kind"],
            "environment_key": self.pool_key.normalized()["environment_key"],
            "metrics": HostedPoolMetrics(
                desired_capacity=self.desired_capacity,
                workers=list(self.workers),
                recent_requests=list(self.recent_requests),
                saturation_count=self.saturation_count,
                timeout_count=self.timeout_count,
                cancellation_count=self.cancellation_count,
                error_count=self.error_count,
                errors_by_reason=dict(self.errors_by_reason),
            ).to_dict(),
        }


class HostedProcessPoolRegistry:
    def __init__(self, *, recent_limit: int = 100) -> None:
        self.recent_limit = max(1, int(recent_limit or 100))
        self._pools: Dict[str, HostedProcessPool] = {}

    def get_or_create(self, pool_key: HostedPoolKey, *, desired_capacity: int = 1) -> HostedProcessPool:
        pool_id = pool_key.pool_id()
        existing = self._pools.get(pool_id)
        if existing is not None:
            existing.set_capacity(desired_capacity)
            return existing
        pool = HostedProcessPool(
            pool_key=pool_key,
            desired_capacity=max(1, int(desired_capacity or 1)),
            recent_limit=self.recent_limit,
        )
        self._pools[pool_id] = pool
        return pool

    def get(self, pool_key: HostedPoolKey) -> Optional[HostedProcessPool]:
        return self._pools.get(pool_key.pool_id())

    def request_status(self, pool_key: HostedPoolKey, request_id: str) -> Dict[str, object]:
        pool = self.get(pool_key)
        if pool is None:
            return {
                "status": "not_found",
                "pool_id": pool_key.pool_id(),
                "request_id": str(request_id or "").strip(),
            }
        return {**pool.request_status(request_id), "pool_id": pool_key.pool_id()}

    def resources(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "pool_count": len(self._pools),
            "pools": {pool_id: pool.resources() for pool_id, pool in sorted(self._pools.items())},
        }


__all__ = [
    "HostedProcessPool",
    "HostedProcessPoolRegistry",
    "WorkerFactory",
]
