"""Internal hosted process pool primitives.

The pool registry is intentionally independent from concrete worker spawn code.
Callers provide a small worker factory when a pool needs its first worker; later
phases can wire that factory to EngineHostService spawn/ensure behavior.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional

from .runtime_base import (
    HostedPoolKey,
    HostedPoolMetrics,
    HostedRequestLifecycle,
    HostedStreamEvent,
    HostedWorkerSlot,
)


WorkerFactory = Callable[[HostedPoolKey, int], HostedWorkerSlot]


@dataclass
class _QueuedSubmission:
    request_id: str
    factory: Optional[WorkerFactory]
    policy: Dict[str, Any]
    start_timestamp: Optional[float] = None


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
    queue_policy: str = "fail_fast"
    queue_depth: int = 0
    queue_timeout_seconds: float = 0.0
    _condition: threading.Condition = field(init=False, repr=False)
    _queued: Deque[_QueuedSubmission] = field(default_factory=deque, init=False, repr=False)
    _active_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.desired_capacity = max(1, int(self.desired_capacity or 1))
        self.recent_limit = max(1, int(self.recent_limit or 100))
        self.queue_policy = self._normalize_queue_policy(self.queue_policy)
        self.queue_depth = max(0, int(self.queue_depth or 0))
        self.queue_timeout_seconds = max(0.0, float(self.queue_timeout_seconds or 0.0))
        self._condition = threading.Condition(threading.RLock())

    @staticmethod
    def _normalize_queue_policy(value: Any) -> str:
        policy = str(value or "fail_fast").strip().lower()
        return policy if policy in {"fail_fast", "bounded"} else "fail_fast"

    @staticmethod
    def _normalize_concurrency_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        row = dict(policy or {})
        mode = str(row.get("mode") or "parallel").strip().lower()
        if mode not in {"parallel", "serial", "keyed", "exclusive"}:
            mode = "parallel"
        group = str(row.get("group") or "").strip()
        resource_key = str(row.get("resource_key") or "").strip()
        try:
            max_concurrency = int(row.get("max_concurrency") or 0)
        except Exception:
            max_concurrency = 0
        if mode == "serial":
            max_concurrency = 1
        elif max_concurrency < 0:
            max_concurrency = 0
        return {
            "mode": mode,
            "group": group,
            "resource_key": resource_key,
            "max_concurrency": max_concurrency,
            "decision": str(row.get("decision") or "parallel"),
        }

    @staticmethod
    def _policy_slot(policy: Dict[str, Any]) -> str:
        mode = str(policy.get("mode") or "parallel")
        group = str(policy.get("group") or "default")
        if mode == "keyed":
            return f"keyed:{group}:{str(policy.get('resource_key') or '__missing__')}"
        if mode == "serial":
            return f"serial:{group}"
        if mode == "exclusive":
            return f"exclusive:{group}"
        return f"parallel:{group}" if group != "default" else ""

    def _policy_allows_locked(self, policy: Dict[str, Any]) -> bool:
        mode = str(policy.get("mode") or "parallel")
        candidate_slot = self._policy_slot(policy)
        if mode == "exclusive" and self._active_policies:
            return False
        for active in self._active_policies.values():
            active_mode = str(active.get("mode") or "parallel")
            if active_mode == "exclusive":
                return False
            active_slot = self._policy_slot(active)
            if (
                candidate_slot
                and candidate_slot == active_slot
                and (mode in {"serial", "keyed"} or active_mode in {"serial", "keyed"})
            ):
                return False
            max_concurrency = int(policy.get("max_concurrency") or 0)
            group = str(policy.get("group") or "")
            if max_concurrency > 0 and group and group == str(active.get("group") or ""):
                active_same_group = sum(
                    1 for item in self._active_policies.values()
                    if str(item.get("group") or "") == group
                )
                if active_same_group >= max_concurrency:
                    return False
        return True

    def _prune_queue_locked(self) -> None:
        self._queued = deque(
            item
            for item in self._queued
            if item.request_id in self.requests
            and self.requests[item.request_id].finished_at is None
        )

    def _is_next_queued_locked(self, request_id: str) -> bool:
        self._prune_queue_locked()
        return not self._queued or self._queued[0].request_id == request_id

    def _select_worker_locked(self, factory: Optional[WorkerFactory] = None) -> Optional[HostedWorkerSlot]:
        if not self.workers and factory is not None:
            self.ensure_worker(factory)
        for worker in self.workers:
            if worker.available_slots() > 0:
                return worker
        self.saturation_count += 1
        return None

    def _admit_locked(
        self,
        request: HostedRequestLifecycle,
        *,
        factory: Optional[WorkerFactory],
        policy: Dict[str, Any],
        start_timestamp: Optional[float],
    ) -> Optional[HostedWorkerSlot]:
        worker = self._select_worker_locked(factory)
        if worker is None or not self._policy_allows_locked(policy):
            return None
        if request.request_id not in worker.active_request_ids:
            worker.active_request_ids.append(request.request_id)
        self._active_policies[request.request_id] = dict(policy)
        request.concurrency_mode = str(policy.get("mode") or "parallel")
        request.concurrency_group = str(policy.get("group") or "").strip() or None
        request.resource_key = str(policy.get("resource_key") or "").strip() or None
        request.mark_started(timestamp=start_timestamp, engine_id=worker.engine_id)
        return worker

    def set_capacity(self, capacity: int) -> int:
        with self._condition:
            self.desired_capacity = max(1, int(capacity or 1))
            if len(self.workers) == 1:
                self.workers[0].capacity = self.desired_capacity
            self._condition.notify_all()
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
        with self._condition:
            return self._select_worker_locked(factory)

    def submit_request(
        self,
        request: HostedRequestLifecycle,
        *,
        factory: Optional[WorkerFactory] = None,
        start_timestamp: Optional[float] = None,
        queue_policy: Optional[str] = None,
        queue_depth: Optional[int] = None,
        queue_timeout_seconds: Optional[float] = None,
        concurrency: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        rid = str(request.request_id or "").strip()
        if not rid:
            request.mark_finished("error", reason="request_id_required", timestamp=start_timestamp)
            return {"status": "error", "reason": "request_id_required", "request": request.to_dict()}
        policy = self._normalize_concurrency_policy(concurrency)
        selected_queue_policy = self._normalize_queue_policy(queue_policy or self.queue_policy)
        selected_queue_depth = max(0, int(self.queue_depth if queue_depth is None else queue_depth or 0))
        selected_timeout = max(
            0.0,
            float(self.queue_timeout_seconds if queue_timeout_seconds is None else queue_timeout_seconds or 0.0),
        )
        with self._condition:
            existing = self.requests.get(rid)
            if existing is not None and existing.finished_at is None:
                return {"status": "error", "reason": "duplicate_request_id", "request": existing.to_dict()}
            self.requests[rid] = request
            request.admission = "submitted"
            queued = False
            deadline = time.monotonic() + selected_timeout if selected_timeout > 0 else None
            while True:
                if request.finished_at is not None:
                    return {"status": "error", "reason": request.reason or "canceled", "request": request.to_dict()}
                if not queued and not self._is_next_queued_locked(rid):
                    if selected_queue_policy != "bounded":
                        request.mark_finished("error", reason="capacity_exceeded", timestamp=start_timestamp)
                        self._remember_request(request)
                        return {"status": "error", "reason": "capacity_exceeded", "request": request.to_dict()}
                is_next = self._is_next_queued_locked(rid)
                worker = (
                    self._admit_locked(
                        request,
                        factory=factory,
                        policy=policy,
                        start_timestamp=start_timestamp if not queued else None,
                    )
                    if is_next
                    else None
                )
                if worker is not None:
                    if queued:
                        self._queued = deque(item for item in self._queued if item.request_id != rid)
                    return {"status": "ok", "worker": worker.to_dict(), "request": request.to_dict()}
                if selected_queue_policy != "bounded" or selected_queue_depth <= 0:
                    request.admission = "rejected"
                    request.mark_finished("error", reason="capacity_exceeded", timestamp=start_timestamp)
                    self._remember_request(request)
                    return {"status": "error", "reason": "capacity_exceeded", "request": request.to_dict()}
                if not queued:
                    self._prune_queue_locked()
                    if len(self._queued) >= selected_queue_depth:
                        request.admission = "rejected"
                        request.mark_finished("error", reason="queue_full", timestamp=start_timestamp)
                        self._remember_request(request)
                        return {"status": "error", "reason": "queue_full", "request": request.to_dict()}
                    request.status = "queued"
                    request.admission = "queued"
                    self._queued.append(
                        _QueuedSubmission(
                            request_id=rid,
                            factory=factory,
                            policy=policy,
                            start_timestamp=start_timestamp,
                        )
                    )
                    queued = True
                    if deadline is None:
                        deadline = time.monotonic() + selected_timeout if selected_timeout > 0 else None
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    self._queued = deque(item for item in self._queued if item.request_id != rid)
                    request.admission = "timed_out"
                    request.mark_finished("error", reason="queue_timeout", timestamp=time.time())
                    self.timeout_count += 1
                    self._remember_request(request)
                    self._condition.notify_all()
                    return {"status": "error", "reason": "queue_timeout", "request": request.to_dict()}
                self._condition.wait(timeout=remaining)

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
        with self._condition:
            request = self.requests.get(rid)
            if request is None:
                return {"status": "not_found", "request_id": rid}
            if request.finished_at is not None:
                return {"status": "ok", "request": request.to_dict(), "already_finished": True}
            if output_bytes is not None:
                request.output_bytes = int(output_bytes)
            request.mark_finished(status, reason=reason, timestamp=timestamp)
            for worker in self.workers:
                worker.active_request_ids = [item for item in worker.active_request_ids if str(item or "").strip() != rid]
            self._active_policies.pop(rid, None)
            if status == "timeout":
                self.timeout_count += 1
            elif status == "canceled":
                self.cancellation_count += 1
            elif status not in {"ok", "success", "done"}:
                self.error_count += 1
                key = str(reason or status or "error").strip() or "error"
                self.errors_by_reason[key] = int(self.errors_by_reason.get(key, 0)) + 1
            self._remember_request(request)
            self._condition.notify_all()
            return {"status": "ok", "request": request.to_dict()}

    def claim_dispatch(self, request_id: str) -> Dict[str, object]:
        """Atomically claim an admitted request before dispatching it to a worker."""
        rid = str(request_id or "").strip()
        with self._condition:
            request = self.requests.get(rid)
            if request is None:
                return {"status": "not_found", "request_id": rid}
            if request.finished_at is not None or request.cancellation_requested:
                return {"status": "canceled", "request": request.to_dict()}
            request.dispatch_started = True
            request.admission = "dispatching"
            return {"status": "ok", "request": request.to_dict()}

    def cancel_request(self, request_id: str, *, timestamp: Optional[float] = None) -> Dict[str, object]:
        rid = str(request_id or "").strip()
        with self._condition:
            request = self.requests.get(rid)
            if request is None:
                return {"status": "not_found", "request_id": rid}
            if request.finished_at is not None:
                return {"status": "ok", "request": request.to_dict(), "already_finished": True}
            request.cancellation_requested = True
            request.admission = "canceled"
            self._queued = deque(item for item in self._queued if item.request_id != rid)
        return self.finish_request(rid, status="canceled", reason="canceled", timestamp=timestamp)

    def record_stream_event(self, request_id: str, event: HostedStreamEvent | Dict[str, object]) -> Dict[str, object]:
        rid = str(request_id or "").strip()
        with self._condition:
            request = self.requests.get(rid)
            if request is None:
                return {"status": "not_found", "request_id": rid}
            row = event.to_batch().expanded_frames()[0] if isinstance(event, HostedStreamEvent) else dict(event or {})
            request.record_stream_event(row)
            return {"status": "ok", "request": request.to_dict(), "event": row}

    def request_status(self, request_id: str) -> Dict[str, object]:
        rid = str(request_id or "").strip()
        with self._condition:
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
        with self._condition:
            self._prune_queue_locked()
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
                    queue_policy=self.queue_policy,
                    queue_depth=self.queue_depth,
                    queue_timeout_seconds=self.queue_timeout_seconds,
                    queued_calls=len(self._queued),
                ).to_dict(),
            }


class HostedProcessPoolRegistry:
    def __init__(self, *, recent_limit: int = 100) -> None:
        self.recent_limit = max(1, int(recent_limit or 100))
        self._pools: Dict[str, HostedProcessPool] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        pool_key: HostedPoolKey,
        *,
        desired_capacity: int = 1,
        queue_policy: str = "fail_fast",
        queue_depth: int = 0,
        queue_timeout_seconds: float = 0.0,
    ) -> HostedProcessPool:
        pool_id = pool_key.pool_id()
        with self._lock:
            existing = self._pools.get(pool_id)
            if existing is not None:
                existing.set_capacity(desired_capacity)
                existing.queue_policy = existing._normalize_queue_policy(queue_policy or existing.queue_policy)
                existing.queue_depth = max(0, int(queue_depth or existing.queue_depth or 0))
                existing.queue_timeout_seconds = max(0.0, float(queue_timeout_seconds or existing.queue_timeout_seconds or 0.0))
                return existing
            pool = HostedProcessPool(
                pool_key=pool_key,
                desired_capacity=max(1, int(desired_capacity or 1)),
                recent_limit=self.recent_limit,
                queue_policy=queue_policy,
                queue_depth=queue_depth,
                queue_timeout_seconds=queue_timeout_seconds,
            )
            self._pools[pool_id] = pool
            return pool

    def get(self, pool_key: HostedPoolKey) -> Optional[HostedProcessPool]:
        with self._lock:
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
        with self._lock:
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
