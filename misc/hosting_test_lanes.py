"""Repeatable hosting pytest lane runner with median-duration evidence."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LANE_SELECTORS = {
    "fast": ("-m", "fast"),
    "process": ("-m", "process"),
    "native": ("-m", "native"),
    "full": (),
}
LANE_BUDGET_SECONDS = {
    "fast": 420.0,
    "process": 360.0,
    "native": 120.0,
    "full": 720.0,
}


def build_pytest_command(
    *,
    lane: str,
    durations: int,
    collect_only: bool = False,
    extra_args: Sequence[str] = (),
) -> list[str]:
    if lane not in LANE_SELECTORS:
        raise ValueError(f"unknown hosting test lane: {lane}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--durations={max(0, int(durations))}",
        *LANE_SELECTORS[lane],
    ]
    if collect_only:
        command.append("--collect-only")
    command.extend(str(value) for value in extra_args)
    return command


def summarize_lane(*, lane: str, durations_seconds: Sequence[float], return_codes: Sequence[int]) -> dict:
    samples = [round(float(value), 3) for value in durations_seconds]
    budget = float(LANE_BUDGET_SECONDS[lane])
    median = round(float(statistics.median(samples)), 3) if samples else None
    return {
        "contract": "hosting.test-lane.evidence.v1",
        "lane": lane,
        "runs": len(samples),
        "durations_seconds": samples,
        "median_duration_seconds": median,
        "budget_seconds": budget,
        "within_budget": bool(median is not None and median <= budget),
        "return_codes": [int(value) for value in return_codes],
        "all_passed": bool(return_codes) and all(int(value) == 0 for value in return_codes),
        "python_executable": sys.executable,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=tuple(LANE_SELECTORS), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--durations", type=int, default=25)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--enforce-budget", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    repeat = int(args.repeat)
    if repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    extra_args = list(args.pytest_args or [])
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    command = build_pytest_command(
        lane=str(args.lane),
        durations=int(args.durations),
        collect_only=bool(args.collect_only),
        extra_args=extra_args,
    )
    elapsed: list[float] = []
    return_codes: list[int] = []
    for index in range(repeat):
        print(
            f"hosting lane {args.lane}: run {index + 1}/{repeat}: "
            + subprocess.list2cmdline(command),
            flush=True,
        )
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)  # noqa: S603
        elapsed.append(time.perf_counter() - started)
        return_codes.append(int(completed.returncode))
        if int(completed.returncode) != 0:
            break
    summary = summarize_lane(
        lane=str(args.lane),
        durations_seconds=elapsed,
        return_codes=return_codes,
    )
    rendered = json.dumps(summary, sort_keys=True)
    print(f"HOSTING_LANE_RESULT={rendered}", flush=True)
    if args.json_output is not None:
        output = Path(args.json_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not summary["all_passed"]:
        return 1
    if bool(args.enforce_budget) and not summary["within_budget"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
