from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from hosting import HostedToolBoxRef
from hosting.toolbox_harness import (
    ToolboxExecutionHarness,
    ToolboxHarnessConfig,
    is_canceled_tool_error as _is_canceled_tool_error,
    should_resubmit_canceled_tool_call as _should_resubmit_canceled_tool_call,
)


def create_hosted_toolbox_ref(
    *,
    host: Any,
    toolbox_id: str,
    python_executable: Optional[str] = None,
    worker_profile_class: str = "generic",
) -> HostedToolBoxRef:
    """
    Public helper for wrappers/automation that want the hosted toolbox-ref API
    without importing the full chat runtime.
    """
    return HostedToolBoxRef(
        toolbox_id=str(toolbox_id or "").strip(),
        host=host,
        python_executable=python_executable,
        worker_profile_class=worker_profile_class,
    )


def register_hosted_tool_callable(
    func: Callable[..., Any],
    *,
    host: Any,
    toolbox_id: str,
    environment_name: str = "base",
    required_imports: Optional[Sequence[str]] = None,
    sandbox_policy: Optional[Dict[str, Any]] = None,
    python_executable: Optional[str] = None,
    worker_profile_class: str = "generic",
    activate: bool = True,
    non_restartable: bool = False,
    guide_content: Optional[Dict[str, List[str]]] = None,
    guide_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for registering a Python callable into hosted toolbox
    sandbox management without forcing callers to instantiate the ref directly.
    """
    hosted_ref = create_hosted_toolbox_ref(
        host=host,
        toolbox_id=toolbox_id,
        python_executable=python_executable,
        worker_profile_class=worker_profile_class,
    )
    return hosted_ref.register_python_callable(
        func,
        environment_name=environment_name,
        required_imports=required_imports,
        sandbox_policy=sandbox_policy,
        activate=activate,
        non_restartable=non_restartable,
        guide_content=guide_content,
        guide_description=guide_description,
    )


def create_hosted_toolbox_executor(
    *,
    control_channel: Any,
    toolbox_id: str = "",
    engine_ids: Optional[Sequence[str]] = None,
    sandbox_selection: str = "round_robin",
) -> ToolboxExecutionHarness:
    """
    Construct a hosted toolbox execution harness suitable for app/runtime use.
    """
    return ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(
            mode="sandbox",
            sandbox_toolbox_id=str(toolbox_id or "").strip() or None,
            sandbox_engine_ids=[str(item or "").strip() for item in list(engine_ids or []) if str(item or "").strip()],
            sandbox_selection=str(sandbox_selection or "round_robin").strip() or "round_robin",
        ),
        control_channel=control_channel,
    )


def is_hosted_tool_call_canceled(tool_call: Any) -> bool:
    """
    Return True when a hosted tool-call result represents coarse sandbox recycle
    cancellation rather than a normal tool failure.
    """
    return _is_canceled_tool_error(tool_call)


def should_resubmit_hosted_tool_call(
    tool_call: Any,
    *,
    non_restartable: bool = False,
) -> bool:
    """
    Helper for wrappers that want a default retry decision after coarse hosted
    sandbox recycling. Non-restartable tools stay opted out.
    """
    return _should_resubmit_canceled_tool_call(
        tool_call,
        non_restartable=non_restartable,
    )


class HostedToolExecutionRouter:
    """
    Small app-facing router that preserves a native toolbox execution path while
    allowing actual tool execution to be redirected through hosted sandbox IPC.
    """

    def __init__(self) -> None:
        self._hosted_executor: Optional[ToolboxExecutionHarness] = None
        self._hosted_advertised_tool_names: Optional[List[str]] = None
        self._hosted_toolbox_description: Optional[Dict[str, Any]] = None

    def configure_hosted_execution(
        self,
        *,
        control_channel: Any,
        toolbox_id: str = "",
        engine_ids: Optional[Sequence[str]] = None,
        sandbox_selection: str = "round_robin",
        advertised_tool_names: Optional[Sequence[str]] = None,
    ) -> ToolboxExecutionHarness:
        self._hosted_executor = create_hosted_toolbox_executor(
            control_channel=control_channel,
            toolbox_id=toolbox_id,
            engine_ids=engine_ids,
            sandbox_selection=sandbox_selection,
        )
        explicit_names = [
            str(item or "").strip()
            for item in list(advertised_tool_names or [])
            if str(item or "").strip()
        ]
        if explicit_names:
            self._hosted_advertised_tool_names = explicit_names
            self._hosted_toolbox_description = {
                "status": "ok",
                "toolbox_id": str(toolbox_id or "").strip(),
                "all_registered_tool_names": list(explicit_names),
                "advertised_tool_names": list(explicit_names),
                "hidden_allowed_tool_names": [],
                "source": "explicit",
            }
        else:
            described_names: List[str] = []
            try:
                if str(toolbox_id or "").strip() and hasattr(control_channel, "toolbox_describe"):
                    payload = dict(control_channel.toolbox_describe(toolbox_id=str(toolbox_id or "").strip()) or {})
                    self._hosted_toolbox_description = dict(payload or {})
                    described_names = [
                        str(item or "").strip()
                        for item in list(
                            payload.get("advertised_tool_names") or []
                        )
                        if str(item or "").strip()
                    ]
            except Exception:
                described_names = []
                self._hosted_toolbox_description = None
            self._hosted_advertised_tool_names = described_names or None
        return self._hosted_executor

    def clear_hosted_execution(self) -> None:
        self._hosted_executor = None
        self._hosted_advertised_tool_names = None
        self._hosted_toolbox_description = None

    def active_executor(self, native_toolbox: Any = None) -> Optional[ToolboxExecutionHarness]:
        if self._hosted_executor is not None:
            self._hosted_executor.native_toolbox = native_toolbox
            return self._hosted_executor
        if native_toolbox is None:
            return None
        return ToolboxExecutionHarness(native_toolbox=native_toolbox)

    def hosted_advertised_tool_names(self) -> Optional[List[str]]:
        if self._hosted_advertised_tool_names is None:
            return None
        return list(self._hosted_advertised_tool_names)

    def hosted_toolbox_summary(self) -> Optional[Dict[str, Any]]:
        if self._hosted_executor is None:
            return None
        summary = dict(self._hosted_toolbox_description or {})
        summary.setdefault("mode", "sandbox")
        summary.setdefault("all_registered_tool_names", [])
        summary.setdefault("advertised_tool_names", list(self._hosted_advertised_tool_names or []))
        summary.setdefault(
            "hidden_allowed_tool_names",
            [
                str(item or "").strip()
                for item in list(summary.get("hidden_allowed_tool_names") or [])
                if str(item or "").strip()
            ],
        )
        summary["all_registered_tool_names"] = [
            str(item or "").strip()
            for item in list(summary.get("all_registered_tool_names") or [])
            if str(item or "").strip()
        ]
        summary.pop("tool_names", None)
        return summary
