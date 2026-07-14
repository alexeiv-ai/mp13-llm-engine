"""Provider-neutral orchestration for one model-requested tool round."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@dataclass(frozen=True)
class CoordinatedToolRound:
    had_calls: bool
    recorded: bool
    executed: bool
    results_recorded: bool
    scheduled_continue: bool
    call_ref: Any = None
    result_ref: Any = None
    execution_result: Any = None


async def coordinate_tool_round(
    calls: Sequence[Any],
    *,
    record_calls: Callable[[Sequence[Any]], Any],
    execute_calls: Callable[[Sequence[Any]], Any],
    has_results: Callable[[Any], bool],
    record_results: Callable[[Sequence[Any], Any], Any],
    schedule_continue: Callable[[Any], Any],
) -> CoordinatedToolRound:
    """Record, execute, persist, and schedule exactly once in a fixed order."""

    normalized_calls = list(calls or [])
    if not normalized_calls:
        return CoordinatedToolRound(
            had_calls=False,
            recorded=False,
            executed=False,
            results_recorded=False,
            scheduled_continue=False,
        )
    call_ref = await _await(record_calls(normalized_calls))
    execution_result = await _await(execute_calls(normalized_calls))
    if not has_results(execution_result):
        return CoordinatedToolRound(
            had_calls=True,
            recorded=True,
            executed=True,
            results_recorded=False,
            scheduled_continue=False,
            call_ref=call_ref,
            execution_result=execution_result,
        )
    result_ref = await _await(record_results(normalized_calls, execution_result))
    await _await(schedule_continue(result_ref))
    return CoordinatedToolRound(
        had_calls=True,
        recorded=True,
        executed=True,
        results_recorded=True,
        scheduled_continue=True,
        call_ref=call_ref,
        result_ref=result_ref,
        execution_result=execution_result,
    )


__all__ = ["CoordinatedToolRound", "coordinate_tool_round"]
