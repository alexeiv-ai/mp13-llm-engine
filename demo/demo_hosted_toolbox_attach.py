from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app import attach_existing_hosted_toolbox


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach to an existing hosted toolbox deployment and optionally execute one tool call.",
    )
    parser.add_argument(
        "--toolbox-id",
        required=True,
        help="Logical hosted toolbox id to attach to.",
    )
    parser.add_argument(
        "--engines-state-file",
        required=True,
        help="Path to the hosted engines-state file.",
    )
    parser.add_argument(
        "--control-state-file",
        required=True,
        help="Path to the hosted control-state file.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="Control-channel timeout. Default: 15.0",
    )
    parser.add_argument(
        "--no-auto-bootstrap",
        action="store_true",
        help="Disable automatic daemon bootstrap when attaching.",
    )
    parser.add_argument(
        "--tool-name",
        type=str,
        default=None,
        help="Optional hosted tool to execute after attach.",
    )
    parser.add_argument(
        "--tool-arguments",
        type=str,
        default="{}",
        help="JSON object of tool arguments for --tool-name. Default: {}",
    )
    return parser.parse_args()


def _load_tool_arguments(raw: str) -> Dict[str, Any]:
    try:
        payload = json.loads(str(raw or "{}"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --tool-arguments JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--tool-arguments must decode to a JSON object.")
    return dict(payload)


def main() -> int:
    args = _parse_args()
    attached = attach_existing_hosted_toolbox(
        toolbox_id=str(args.toolbox_id or "").strip(),
        engines_state_file=Path(str(args.engines_state_file)).expanduser().resolve(),
        control_state_file=Path(str(args.control_state_file)).expanduser().resolve(),
        timeout_seconds=float(args.timeout_seconds or 15.0),
        auto_bootstrap=not bool(args.no_auto_bootstrap),
        python_executable=sys.executable,
    )

    summary = dict(attached.summary or {})
    print("Hosted toolbox attached.")
    print(f"  toolbox_id: {args.toolbox_id}")
    print(f"  engines_state_file: {Path(args.engines_state_file).expanduser().resolve()}")
    print(f"  control_state_file: {Path(args.control_state_file).expanduser().resolve()}")
    print(f"  advertised_tool_names: {summary.get('advertised_tool_names') or []}")
    print(f"  hidden_allowed_tool_names: {summary.get('hidden_allowed_tool_names') or []}")

    tool_name = str(args.tool_name or "").strip()
    if not tool_name:
        return 0

    tool_arguments = _load_tool_arguments(args.tool_arguments)
    payload = attached.control_channel.toolbox_execute(
        toolbox_id=str(args.toolbox_id or "").strip(),
        tool_call={
            "name": tool_name,
            "arguments": tool_arguments,
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
