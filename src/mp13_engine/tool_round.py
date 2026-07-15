"""Provider-neutral orchestration for one model-requested tool round."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence


@dataclass(frozen=True)
class NormalizedServerToolEvent:
    """Provider-neutral, persistence-safe server-tool event metadata."""

    provider_id: str
    tool_id: str
    item_type: str
    status: str = ""
    schema_version: str = "server_tool.event.v1"
    kind: str = "server_tool_call"

    def __post_init__(self) -> None:
        if self.schema_version != "server_tool.event.v1":
            raise ValueError("unsupported server-tool event schema")
        if self.kind != "server_tool_call":
            raise ValueError("companion engine accepts only server-tool call events")
        for field_name in ("provider_id", "tool_id", "item_type"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"server-tool event {field_name} is required")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NormalizedServerToolEvent":
        row = dict(value or {})
        allowed = {
            "schema_version",
            "kind",
            "provider_id",
            "tool_id",
            "item_type",
            "status",
        }
        extras = sorted(set(row) - allowed)
        if extras:
            raise ValueError(
                "server-tool events contain unsupported fields: " + ", ".join(extras)
            )
        return cls(
            schema_version=str(row.get("schema_version") or "server_tool.event.v1"),
            kind=str(row.get("kind") or "server_tool_call"),
            provider_id=str(row.get("provider_id") or "").strip(),
            tool_id=str(row.get("tool_id") or "").strip(),
            item_type=str(row.get("item_type") or "").strip(),
            status=str(row.get("status") or "").strip(),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "provider_id": self.provider_id,
            "tool_id": self.tool_id,
            "item_type": self.item_type,
            "status": self.status,
        }


def normalize_server_tool_events(
    values: Sequence[Mapping[str, Any] | NormalizedServerToolEvent] | None,
) -> list[Dict[str, str]]:
    return [
        (value if isinstance(value, NormalizedServerToolEvent) else NormalizedServerToolEvent.from_mapping(value)).to_dict()
        for value in list(values or [])
    ]


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


__all__ = [
    "CoordinatedToolRound",
    "NormalizedServerToolEvent",
    "coordinate_tool_round",
    "normalize_server_tool_events",
]
