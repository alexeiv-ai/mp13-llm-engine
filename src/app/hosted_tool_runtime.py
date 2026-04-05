from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.context_cursor import ChatCursor
from hosting.toolbox_harness import is_canceled_tool_error, should_resubmit_canceled_tool_call
from mp13_engine.mp13_config import InferenceResponse, ParserProfile, ToolCall, ToolCallBlock
from mp13_engine.mp13_toolbox import ToolsView


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
    non_restartable_tool_names: Optional[Sequence[str]] = None,
) -> ToolRoundResult:
    all_tool_blocks: List[ToolCallBlock] = []
    for item in list(final_response_items or []):
        if item.tool_blocks:
            all_tool_blocks.extend(list(item.tool_blocks or []))
    if not all_tool_blocks:
        return ToolRoundResult(
            had_tool_blocks=False,
            executed=False,
            scheduled_auto_iteration=False,
            aborted=False,
            canceled_tool_names=[],
            resubmittable_tool_names=[],
        )

    final_text_response = responses_in_progress.get(0, "")
    cursor.add_assistant(
        content=final_text_response or "",
        tool_blocks=all_tool_blocks,
        archived=False,
        do_continue=is_manual_continue,
    )

    if tool_executor is None:
        return ToolRoundResult(
            had_tool_blocks=True,
            executed=False,
            scheduled_auto_iteration=False,
            aborted=False,
            canceled_tool_names=[],
            resubmittable_tool_names=[],
        )

    await tool_executor.execute_request_tools(
        parser_profile=parser_profile,
        final_response_items=final_response_items,
        action_handler=action_handler,
        serial_execution=bool(serial_execution),
        tools_view=tools_view,
        pt_session=pt_session,
        context=cursor.current_turn,
        tool_retries_max=tool_retries_max,
        tool_retries_left=tool_retries_left,
    )
    canceled_summary = summarize_canceled_tool_calls(
        all_tool_blocks,
        non_restartable_tool_names=non_restartable_tool_names,
    )

    aborted = _tool_blocks_have_abort(all_tool_blocks)
    if aborted:
        return ToolRoundResult(
            had_tool_blocks=True,
            executed=True,
            scheduled_auto_iteration=False,
            aborted=True,
            canceled_tool_names=list(canceled_summary.get("canceled_tool_names") or []),
            resubmittable_tool_names=list(canceled_summary.get("resubmittable_tool_names") or []),
        )

    if not _tool_blocks_have_results(all_tool_blocks):
        return ToolRoundResult(
            had_tool_blocks=True,
            executed=True,
            scheduled_auto_iteration=False,
            aborted=False,
            canceled_tool_names=list(canceled_summary.get("canceled_tool_names") or []),
            resubmittable_tool_names=list(canceled_summary.get("resubmittable_tool_names") or []),
        )

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
        tool_anchor.retries_remaining -= 1

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

    if scope:
        scope.set_active_cursor(tool_results_cursor)
        scope.request_auto_iteration()
    else:
        cursor.context.set_active_cursor(tool_results_cursor)
        cursor.context.request_auto_iteration()

    return ToolRoundResult(
        had_tool_blocks=True,
        executed=True,
        scheduled_auto_iteration=True,
        aborted=False,
        tool_result_cursor_id=tool_results_cursor.context_id,
        canceled_tool_names=list(canceled_summary.get("canceled_tool_names") or []),
        resubmittable_tool_names=list(canceled_summary.get("resubmittable_tool_names") or []),
    )
