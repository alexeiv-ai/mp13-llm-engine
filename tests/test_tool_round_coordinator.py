from __future__ import annotations

import pytest

from mp13_engine.tool_round import coordinate_tool_round


@pytest.mark.asyncio
async def test_coordinator_runs_effectful_phases_once_in_order():
    events = []

    async def execute(calls):
        events.append(("execute", list(calls)))
        return [{"result": "ok"}]

    outcome = await coordinate_tool_round(
        [{"id": "call-1"}],
        record_calls=lambda calls: events.append(("record_calls", list(calls))) or "turn-a",
        execute_calls=execute,
        has_results=bool,
        record_results=lambda calls, results: events.append(("record_results", list(results))) or "turn-r",
        schedule_continue=lambda result_ref: events.append(("schedule", result_ref)),
    )
    assert [name for name, _ in events] == [
        "record_calls",
        "execute",
        "record_results",
        "schedule",
    ]
    assert outcome.call_ref == "turn-a"
    assert outcome.result_ref == "turn-r"
    assert outcome.scheduled_continue is True


@pytest.mark.asyncio
async def test_coordinator_does_not_record_results_or_schedule_without_results():
    events = []
    outcome = await coordinate_tool_round(
        ["call"],
        record_calls=lambda calls: events.append("record"),
        execute_calls=lambda calls: events.append("execute") or [],
        has_results=bool,
        record_results=lambda calls, results: events.append("results"),
        schedule_continue=lambda result_ref: events.append("schedule"),
    )
    assert events == ["record", "execute"]
    assert outcome.executed is True
    assert outcome.scheduled_continue is False
