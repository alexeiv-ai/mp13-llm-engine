from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from hosting.engine_host_channel import EngineHostControlChannel
from hosting import HostedToolBoxRef
from hosting.toolbox_harness import (
    HostedToolBoxRef,
    ToolboxExecutionHarness,
    ToolboxHarnessConfig,
    is_canceled_tool_error as _is_canceled_tool_error,
    should_resubmit_canceled_tool_call as _should_resubmit_canceled_tool_call,
)


@dataclass
class HostedToolboxAttachment:
    control_channel: EngineHostControlChannel
    toolbox_ref: HostedToolBoxRef
    executor: ToolboxExecutionHarness
    summary: Dict[str, Any]


def create_hosted_control_channel(
    *,
    engines_state_file: Any,
    mp13_config_file: Any,
    timeout_seconds: float = 15.0,
    auto_bootstrap: bool = True,
) -> EngineHostControlChannel:
    """
    Build a local hosted-control channel backed by existing host state files.
    This is the app-facing entry point for attaching to an already provisioned
    hosted toolbox without going through the demo setup path.
    """
    return EngineHostControlChannel(
        {
            "engine_host_state_file": str(engines_state_file),
            "engine_host_mp13_config_file": str(mp13_config_file),
            "engine_host_timeout_seconds": float(timeout_seconds or 15.0),
            "engine_host_daemon_auto_bootstrap": bool(auto_bootstrap),
        }
    )


def attach_existing_hosted_toolbox(
    *,
    toolbox_id: str,
    engines_state_file: Any,
    mp13_config_file: Any,
    timeout_seconds: float = 15.0,
    auto_bootstrap: bool = True,
) -> HostedToolboxAttachment:
    """
    Public app/helper entry point for attaching to an existing hosted toolbox
    deployment. This is intended for thin wrappers and automation that want the
    hosted control channel, hosted ref, execution harness, and current summary
    without reimplementing the mp13chat CLI wiring.
    """
    tid = str(toolbox_id or "").strip()
    if not tid:
        raise ValueError("toolbox_id is required")
    control_channel = create_hosted_control_channel(
        engines_state_file=engines_state_file,
        mp13_config_file=mp13_config_file,
        timeout_seconds=timeout_seconds,
        auto_bootstrap=auto_bootstrap,
    )
    toolbox_ref = create_hosted_toolbox_ref(
        host=control_channel,
        toolbox_id=tid,
    )
    router = HostedToolExecutionRouter()
    executor = router.configure_hosted_execution(
        control_channel=control_channel,
        toolbox_id=tid,
    )
    summary = dict(router.hosted_toolbox_summary() or {})
    return HostedToolboxAttachment(
        control_channel=control_channel,
        toolbox_ref=toolbox_ref,
        executor=executor,
        summary=summary,
    )


def create_hosted_toolbox_ref(
    *,
    host: Any,
    toolbox_id: str,
) -> HostedToolBoxRef:
    """
    Public helper for wrappers/automation that want the hosted toolbox-ref API
    without importing the full chat runtime.
    """
    return HostedToolBoxRef(
        toolbox_id=str(toolbox_id or "").strip(),
        host=host,
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

    Public approval-callback note:
    - gated tools can trigger the hosted callback processor with callback name
      `tool_requires_confirmation`
    - the callback payload kind is `tool_approval_request`
    - supported decisions are `deny`, `allow_once`, and `add_to_scope`
    - `allow_once` affects only the current call
    - `add_to_scope` persists only when the caller also supplies a durable
      scope target such as a `ToolBoxRef`
    - this helper only constructs the executor; it does not auto-supply that
      scope target on its own
    - compact approval example:

      def callback_processor(*, callback_name, payload, context):
          if callback_name != "tool_requires_confirmation":
              return {"decision": "deny"}
          return {
              "decision": "add_to_scope",
              "scope_constraints": {
                  "search_files": {
                      "argument_policy": {
                          "implied_args": {"root_path": "docs/api"},
                          "locked_args": ["root_path"],
                      }
                  }
              },
          }

    Constraint-aware tool note:
    - kwargs-capable tools can accept `tool_constraints_view`
    - that helper exposes `resolve_argument(...)`,
      `resolve_filesystem_root(...)`, and `resolve_url(...)`
    - example:

      def search_files(name_mask: str, root_path: str = "", **kwargs):
          scoped = kwargs["tool_constraints_view"]
          effective_root = scoped.resolve_filesystem_root(root_path or None)
          ...
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

    Wrapper consistency rule:
    wrappers that expose hosted approval should route execution through a path
    that also forwards a durable scope target when they want `add_to_scope` to
    persist beyond one call.
    `execute_tool_round_on_cursor(...)` already does this automatically;
    direct `HostedToolBoxRef.execute(...)` calls can now either:
    - pass `scope_ref=...`
    - or pass `callback_context` with `toolbox_ref` or `cursor`.
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
