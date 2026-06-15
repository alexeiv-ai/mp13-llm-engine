"""Shared hosted child runtime interface.

This protocol is intentionally smaller than the host-side pool base. Concrete
runtimes own their child process protocol, while host services can depend on
the same execute/cancel/resources shape.
"""
from __future__ import annotations

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


__all__ = ["ChildRuntimeEventCallback", "HostedChildRuntime"]
