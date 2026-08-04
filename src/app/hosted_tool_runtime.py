from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from app.context_cursor import ChatCursor
from hosting.toolbox_harness import is_canceled_tool_error, should_resubmit_canceled_tool_call
from mp13_engine.mp13_config import InferenceResponse, ParserProfile, ToolCall, ToolCallBlock
from mp13_engine.mp13_toolbox import ToolsView
from mp13_engine.tool_round import coordinate_tool_round, normalize_server_tool_events


def _build_hosted_callback_context(*, cursor: ChatCursor, callback_context: Any) -> Any:
    toolbox_ref = getattr(getattr(cursor, "context", None), "toolbox_ref", None)
    if isinstance(callback_context, dict):
        merged = dict(callback_context)
    elif callback_context is None:
        merged = {}
    else:
        merged = {"user_context": callback_context}
    merged.setdefault("cursor", cursor)
    if toolbox_ref is not None:
        merged.setdefault("toolbox_ref", toolbox_ref)
    return merged


def _tool_blocks_have_results(blocks: Sequence[ToolCallBlock]) -> bool:
    for block in list(blocks or []):
        for tool_call in list(getattr(block, "calls", []) or []):
            if getattr(tool_call, "result", None) is not None or getattr(tool_call, "error", None):
                return True
    return False


def _tool_blocks_have_abort(blocks: Sequence[ToolCallBlock]) -> bool:
    for block in list(blocks or []):
        if ToolCall.Abort in list(getattr(block, "action_block", []) or []):
            return True
        for tool_call in list(getattr(block, "calls", []) or []):
            if ToolCall.Abort in list(getattr(tool_call, "action", []) or []):
                return True
    return False


def _tool_call_has_error(blocks: Sequence[ToolCallBlock]) -> bool:
    for block in list(blocks or []):
        if getattr(block, "error_block", None):
            return True
        for tool_call in list(getattr(block, "calls", []) or []):
            if getattr(tool_call, "error", None):
                return True
    return False


def summarize_canceled_tool_calls(
    blocks: Sequence[ToolCallBlock],
    *,
    non_restartable_tool_names: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    non_restartable = {
        str(item or "").strip()
        for item in list(non_restartable_tool_names or [])
        if str(item or "").strip()
    }
    canceled: List[str] = []
    resubmittable: List[str] = []
    seen_canceled: set[str] = set()
    seen_resubmittable: set[str] = set()
    for block in list(blocks or []):
        for tool_call in list(getattr(block, "calls", []) or []):
            tool_name = str(getattr(tool_call, "name", "") or "").strip()
            if not tool_name or not is_canceled_tool_error(tool_call):
                continue
            if tool_name not in seen_canceled:
                seen_canceled.add(tool_name)
                canceled.append(tool_name)
            if should_resubmit_canceled_tool_call(
                tool_call,
                non_restartable=tool_name in non_restartable,
            ) and tool_name not in seen_resubmittable:
                seen_resubmittable.add(tool_name)
                resubmittable.append(tool_name)
    return {
        "canceled_tool_names": canceled,
        "resubmittable_tool_names": resubmittable,
    }


@dataclass
class ToolRoundResult:
    had_tool_blocks: bool
    executed: bool
    scheduled_auto_iteration: bool
    aborted: bool
    had_server_tool_events: bool = False
    server_events_recorded: bool = False
    tool_result_cursor_id: Optional[str] = None
    canceled_tool_names: List[str] = field(default_factory=list)
    resubmittable_tool_names: List[str] = field(default_factory=list)


async def execute_tool_round_on_cursor(
    *,
    cursor: ChatCursor,
    final_response_items: List[InferenceResponse],
    responses_in_progress: Dict[int, str],
    parser_profile: ParserProfile,
    tool_executor: Any,
    action_handler: Callable[..., Any],
    tools_view: Optional[ToolsView] = None,
    pt_session: Optional[Any] = None,
    is_manual_continue: bool = False,
    tool_retries_max: Optional[int] = None,
    tool_retries_left: Optional[int] = None,
    auto_tool_retry_limit: int = 5,
    auto_anchor_prefix: str = "auto_tool",
    serial_execution: bool = False,
    max_concurrency: Optional[int] = None,
    non_restartable_tool_names: Optional[Sequence[str]] = None,
    callback_processor: Optional[Callable[..., Any]] = None,
    callback_context: Any = None,
    host_api_approval: Optional[Dict[str, Any]] = None,
    control_tool_handlers: Optional[Mapping[str, Callable[..., Any]]] = None,
    server_tool_events: Optional[Sequence[Mapping[str, Any]]] = None,
) -> ToolRoundResult:
    """
    Execute one hosted/native tool round for a chat cursor.

    Public hosted approval contract:
    - when hosted execution encounters a gated tool, the supplied
      `callback_processor` can receive `tool_requires_confirmation`
    - the payload kind is `tool_approval_request`
    - valid decisions are `deny`, `allow_once`, and `add_to_scope`
    - `add_to_scope` may also return `scope_constraints` for future calls

    Constraint-aware tool pattern:
    - hosted/native execution can inject `tool_constraints_view` into
      kwargs-capable tools
    - tools that accept optional contextual narrowing should prefer:
      `tool_constraints_view.resolve_argument(...)`
      `tool_constraints_view.resolve_filesystem_root(...)`
      `tool_constraints_view.resolve_url(...)`
      instead of re-parsing raw constraint payloads directly
    - compact example:

      def search_files(name_mask: str, root_path: str = "", **kwargs):
          scoped = kwargs["tool_constraints_view"]
          effective_root = scoped.resolve_filesystem_root(root_path or None)
          ...

    Wrapper behavior:
    - this helper automatically forwards the active `cursor` and context
      `toolbox_ref` inside `callback_context`
    - that gives hosted approval flows a stable scope target so
      `add_to_scope` can persist for later calls in the same chat context
    - `host_api_approval` is forwarded to hosted sandbox execution for
      per-IO approvals raised by `context.fs`, `context.http`, or
      `context.host.call(...)` inside hosted tools
    - `control_tool_handlers` intercept engine-owned control tools after provider
      parsing but before local execution; intercepted calls still use this
      round's normal call/result persistence and auto-continue path
    """
    normalized_server_events = normalize_server_tool_events(server_tool_events)
    all_tool_blocks: List[ToolCallBlock] = []
    for item in list(final_response_items or []):
        if item.tool_blocks:
            all_tool_blocks.extend(list(item.tool_blocks or []))
    if not all_tool_blocks and not normalized_server_events:
        return ToolRoundResult(
            had_tool_blocks=False,
            executed=False,
            scheduled_auto_iteration=False,
            aborted=False,
            canceled_tool_names=[],
            resubmittable_tool_names=[],
        )

    if not all_tool_blocks:
        cursor.add_assistant(
            content=responses_in_progress.get(0, "") or "",
            server_tool_events=normalized_server_events,
            archived=False,
            do_continue=is_manual_continue,
        )
        return ToolRoundResult(
            had_tool_blocks=False,
            executed=False,
            scheduled_auto_iteration=False,
            aborted=False,
            had_server_tool_events=True,
            server_events_recorded=True,
            canceled_tool_names=[],
            resubmittable_tool_names=[],
        )

    if tool_executor is None:
        cursor.add_assistant(
            content=responses_in_progress.get(0, "") or "",
            tool_blocks=all_tool_blocks,
            server_tool_events=normalized_server_events,
            archived=False,
            do_continue=is_manual_continue,
        )
        return ToolRoundResult(
            had_tool_blocks=True,
            executed=False,
            scheduled_auto_iteration=False,
            aborted=False,
            had_server_tool_events=bool(normalized_server_events),
            server_events_recorded=bool(normalized_server_events),
            canceled_tool_names=[],
            resubmittable_tool_names=[],
        )

    canceled_summary: Dict[str, List[str]] = {
        "canceled_tool_names": [],
        "resubmittable_tool_names": [],
    }

    def _record_calls(_blocks: Sequence[ToolCallBlock]) -> Any:
        return cursor.add_assistant(
            content=responses_in_progress.get(0, "") or "",
            tool_blocks=all_tool_blocks,
            server_tool_events=normalized_server_events,
            archived=False,
            do_continue=is_manual_continue,
        )

    async def _execute_calls(_blocks: Sequence[ToolCallBlock]) -> Sequence[ToolCallBlock]:
        nonlocal canceled_summary
        normalized_control_handlers = {
            str(name or "").strip(): handler
            for name, handler in dict(control_tool_handlers or {}).items()
            if str(name or "").strip() and callable(handler)
        }

        async def _action_with_control_tools(*, execute_stage: str, **action_kwargs: Any) -> Any:
            if execute_stage == "calls_parsed" and normalized_control_handlers:
                for response_item in list(final_response_items or []):
                    for block in list(getattr(response_item, "tool_blocks", None) or []):
                        for tool_call in list(getattr(block, "calls", None) or []):
                            handler = normalized_control_handlers.get(str(getattr(tool_call, "name", "") or "").strip())
                            if handler is None or ToolCall.Ignore in list(getattr(tool_call, "action", None) or []):
                                continue
                            try:
                                handled = handler(
                                    tool_call=tool_call,
                                    cursor=cursor,
                                    tools_view=tools_view,
                                    callback_processor=callback_processor,
                                    callback_context=_build_hosted_callback_context(
                                        cursor=cursor,
                                        callback_context=callback_context,
                                    ),
                                )
                                if inspect.isawaitable(handled):
                                    handled = await handled
                                tool_call.result = handled
                            except Exception as exc:
                                tool_call.error = str(exc).strip() or "control_tool_failed"
                            if ToolCall.Ignore not in tool_call.action:
                                tool_call.action.append(ToolCall.Ignore)
            handled_action = action_handler(execute_stage=execute_stage, **action_kwargs)
            return await handled_action if inspect.isawaitable(handled_action) else handled_action

        execute_kwargs = {
            "parser_profile": parser_profile,
            "final_response_items": final_response_items,
            "action_handler": _action_with_control_tools,
            "serial_execution": bool(serial_execution),
            "tools_view": tools_view,
            "pt_session": pt_session,
            "context": cursor.current_turn,
            "tool_retries_max": tool_retries_max,
            "tool_retries_left": tool_retries_left,
            "callback_processor": callback_processor,
            "callback_context": _build_hosted_callback_context(cursor=cursor, callback_context=callback_context),
        }
        if isinstance(host_api_approval, dict):
            execute_kwargs["host_api_approval"] = dict(host_api_approval or {})
        if max_concurrency is not None:
            execute_kwargs["max_concurrency"] = int(max_concurrency)
        await tool_executor.execute_request_tools(**execute_kwargs)
        canceled_summary = summarize_canceled_tool_calls(
            all_tool_blocks,
            non_restartable_tool_names=non_restartable_tool_names,
        )
        return all_tool_blocks

    def _record_results(
        _blocks: Sequence[ToolCallBlock],
        _executed: Sequence[ToolCallBlock],
    ) -> Any:
        anchor_name = f"{str(auto_anchor_prefix or 'auto_tool')}:{cursor.context_id or getattr(cursor.head, 'gen_id', '') or 'cursor'}"
        scope = getattr(cursor, "scope", None)
        if scope:
            tool_anchor = scope.find_active_anchor("auto_tool", cursor)
            if not tool_anchor:
                tool_anchor = scope.start_try_out_anchor(
                    anchor_name,
                    cursor.head,
                    kind="auto_tool",
                    retry_limit=int(auto_tool_retry_limit or 5),
                    origin_cursor=cursor,
                )
        else:
            tool_anchor = cursor.context.start_try_out_anchor(
                anchor_name,
                cursor.head,
                kind="auto_tool",
                retry_limit=int(auto_tool_retry_limit or 5),
                origin_cursor=cursor,
            )
        if _tool_call_has_error(all_tool_blocks) and tool_anchor.retries_remaining > 0:
            cursor.context.decrement_try_out_anchor_retry(
                tool_anchor.anchor_name,
                scope=tool_anchor.owner_scope,
            )
        _, tryout_cursor = cursor.add_try_out(
            anchor=tool_anchor,
            anchor_turn=cursor.current_turn or tool_anchor.anchor_turn or cursor.head,
            keep_in_main=True,
            convert_existing=True,
        )
        try:
            tryout_cursor.set_main_thread(True)
        except Exception:
            if tryout_cursor.head:
                tryout_cursor.head.main_thread = True
        tool_results_cursor = tryout_cursor.add_tool_results(all_tool_blocks)
        tool_results_cursor.set_auto(True)
        try:
            tool_results_cursor.set_main_thread(True)
        except Exception:
            if tool_results_cursor.head:
                tool_results_cursor.head.main_thread = True
        return tool_results_cursor

    def _schedule_continue(tool_results_cursor: Any) -> None:
        scope = getattr(cursor, "scope", None)
        if scope:
            scope.set_active_cursor(tool_results_cursor)
            scope.request_auto_iteration()
        else:
            cursor.context.set_active_cursor(tool_results_cursor)
            cursor.context.request_auto_iteration()

    coordinated = await coordinate_tool_round(
        all_tool_blocks,
        record_calls=_record_calls,
        execute_calls=_execute_calls,
        has_results=lambda blocks: bool(blocks)
        and not _tool_blocks_have_abort(blocks)
        and _tool_blocks_have_results(blocks),
        record_results=_record_results,
        schedule_continue=_schedule_continue,
    )
    aborted = _tool_blocks_have_abort(all_tool_blocks)
    tool_results_cursor = coordinated.result_ref
    return ToolRoundResult(
        had_tool_blocks=True,
        executed=True,
        scheduled_auto_iteration=coordinated.scheduled_continue,
        aborted=aborted,
        had_server_tool_events=bool(normalized_server_events),
        server_events_recorded=bool(normalized_server_events),
        tool_result_cursor_id=(tool_results_cursor.context_id if tool_results_cursor is not None else None),
        canceled_tool_names=list(canceled_summary.get("canceled_tool_names") or []),
        resubmittable_tool_names=list(canceled_summary.get("resubmittable_tool_names") or []),
    )
