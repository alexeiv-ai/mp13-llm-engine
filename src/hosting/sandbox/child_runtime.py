"""Shared hosted child runtime interface.

This protocol is intentionally smaller than the host-side pool base. Concrete
runtimes own their child process protocol, while host services can depend on
the same execute/cancel/resources shape.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Protocol


ChildRuntimeEventCallback = Callable[[str, Dict[str, Any]], None]


def wait_for_child_ipc_connection(
    *,
    accept_queue: "queue.Queue[Any]",
    process: Any,
    timeout_seconds: float,
    timeout_error: str,
    exited_error: str,
    poll_interval_seconds: float = 0.1,
) -> Any:
    """Wait for listener acceptance while also observing child-process exit."""
    deadline = time.monotonic() + max(0.01, float(timeout_seconds))
    interval = max(0.01, min(float(poll_interval_seconds), 0.5))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(timeout_error)
        try:
            return accept_queue.get(timeout=min(interval, remaining))
        except queue.Empty:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(f"{exited_error}:{int(return_code)}")


class HostedChildRuntime(Protocol):
    def execute(
        self,
        request: Dict[str, Any],
        *,
        python_executable: Optional[str] = None,
        on_event: Optional[ChildRuntimeEventCallback] = None,
    ) -> Dict[str, Any]:
        ...

    def cancel(self, request_id: str) -> Dict[str, Any]:
        ...

    def resources(self) -> Dict[str, Any]:
        ...


class HostedActiveChildRuntimeRegistry:
    """Shared active child process registry for direct host-managed runtimes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Dict[str, Any] = {}
        self._pending_cancel: set[str] = set()

    def register_active(self, request_id: str, runtime: Any) -> None:
        rid = str(request_id or "").strip()
        if not rid:
            return
        cancel_now = False
        with self._lock:
            self._active[rid] = runtime
            cancel_now = rid in self._pending_cancel
            self._pending_cancel.discard(rid)
        if cancel_now:
            cancel = getattr(runtime, "cancel", None)
            if callable(cancel):
                cancel()

    def unregister_active(self, request_id: str) -> None:
        rid = str(request_id or "").strip()
        if not rid:
            return
        with self._lock:
            self._active.pop(rid, None)

    def active_runtime(self, request_id: str) -> Any:
        rid = str(request_id or "").strip()
        with self._lock:
            return self._active.get(rid)

    def cancel(self, request_id: str) -> Dict[str, Any]:
        rid = str(request_id or "").strip()
        if not rid:
            return {"status": "error", "reason": "request_id_required", "canceled": False}
        runtime = self.active_runtime(rid)
        if runtime is None:
            with self._lock:
                self._pending_cancel.add(rid)
            return {"status": "ok", "request_id": rid, "canceled": True, "reason": "cancel_pending"}
        cancel = getattr(runtime, "cancel", None)
        if not callable(cancel):
            return {"status": "ok", "request_id": rid, "canceled": False, "reason": "cancel_not_supported"}
        return {"status": "ok", "request_id": rid, "canceled": bool(cancel()), "reason": "canceled"}

    def resources(self) -> Dict[str, Any]:
        with self._lock:
            active = list(self._active.items())
        processes = []
        for request_id, runtime in active:
            proc = getattr(runtime, "proc", None)
            pid = int(getattr(proc, "pid", 0) or 0) if proc is not None else 0
            alive = None
            if proc is not None and callable(getattr(proc, "poll", None)):
                alive = proc.poll() is None
            processes.append(
                {
                    "request_id": str(request_id or "").strip(),
                    "pid": pid or None,
                    "alive": alive,
                    "python_executable": str(getattr(runtime, "python_executable", "") or "") or None,
                    "canceled": bool(getattr(runtime, "_cancel_requested", False)),
                }
            )
        return {
            "status": "ok",
            "active_count": len(processes),
            "processes": processes,
        }


__all__ = [
    "ChildRuntimeEventCallback",
    "HostedActiveChildRuntimeRegistry",
    "HostedChildRuntime",
    "wait_for_child_ipc_connection",
]
