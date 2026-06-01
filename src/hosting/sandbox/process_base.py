"""Internal hosted process sandbox base primitives.

This is not a public sandbox kind. It is a small composition layer over the
runtime pool models so concrete runtimes can share pool, capacity, request
status, progress, and cancellation plumbing without inheriting workflow,
toolbox, or model semantics.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from .runtime_base import HostedPoolKey, HostedRequestLifecycle, HostedStreamEvent, HostedWorkerSlot
from .runtime_pool import HostedProcessPool, HostedProcessPoolRegistry, WorkerFactory


class HostedProcessSandboxBase:
    """Language-neutral internal base for concrete hosted process runtimes."""

    sandbox_kind: str = "generic"

    def __init__(self, *, pool_registry: Optional[HostedProcessPoolRegistry] = None) -> None:
        self.pool_registry = pool_registry or HostedProcessPoolRegistry()

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


__all__ = ["HostedProcessSandboxBase"]
