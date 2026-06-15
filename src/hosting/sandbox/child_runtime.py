"""Shared hosted child runtime interface.

This protocol is intentionally smaller than the host-side pool base. Concrete
runtimes own their child process protocol, while host services can depend on
the same execute/cancel/resources shape.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Protocol


ChildRuntimeEventCallback = Callable[[str, Dict[str, Any]], None]


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


__all__ = ["ChildRuntimeEventCallback", "HostedActiveChildRuntimeRegistry", "HostedChildRuntime"]
