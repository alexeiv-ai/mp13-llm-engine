from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "misc" / "hosting_test_lanes.py"
SPEC = importlib.util.spec_from_file_location("hosting_test_lanes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LANES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LANES
SPEC.loader.exec_module(LANES)


def test_lane_commands_are_explicit_and_process_lane_is_serial() -> None:
    command = LANES.build_pytest_command(lane="process", durations=17)

    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert command[-2:] == ["-m", "process"]
    assert "--durations=17" in command
    assert "-n" not in command


def test_lane_summary_reports_median_budget_and_failures() -> None:
    passing = LANES.summarize_lane(
        lane="process",
        durations_seconds=[260.0, 220.0, 240.0],
        return_codes=[0, 0, 0],
    )
    failing = LANES.summarize_lane(
        lane="fast",
        durations_seconds=[1.0, 2.0],
        return_codes=[0, 1],
    )

    assert passing["median_duration_seconds"] == 240.0
    assert passing["within_budget"] is True
    assert passing["all_passed"] is True
    assert failing["all_passed"] is False
