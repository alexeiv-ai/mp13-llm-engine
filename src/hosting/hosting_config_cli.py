"""Host-local CLI for the unified hosting configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .hosting_setup_api import (
    apply_local_hosting_setup,
    get_local_hosting_setup_status,
    inspect_local_hosting_setup,
    plan_local_hosting_setup,
    reset_local_hosting_setup,
)


class UserCancelled(RuntimeError):
    def __init__(self, message: str = "cancelled by user", *, via_keyboard: bool = False) -> None:
        super().__init__(message)
        self.via_keyboard = bool(via_keyboard)


def _set_color_scheme(_scheme: str) -> None:
    return None


def _c(_kind: str, text: Any) -> str:
    return str(text)


def _print_title(text: str) -> None:
    print(f"\n{text}")


def _print_block(title: str, **_kwargs: Any) -> None:
    print(f"\n{title}\n" + "." * 78)


def _kv_rows(rows: list[tuple[str, Any]], *, indent: str = "  ", min_width: int = 24) -> None:
    width = max([min_width, *[len(str(label)) for label, _ in rows]])
    for label, value in rows:
        print(f"{indent}{str(label).ljust(width)} : {value}")


def _prompt_menu(
    question: str,
    options: Dict[str, Any],
    default: str,
    *,
    allow_back: bool = False,
    allow_changes: bool = True,
    enter_hint: str = "default/keep",
) -> str:
    del allow_changes, enter_hint
    while True:
        _print_block(question.strip(": \n") or "Menu")
        for key, item in options.items():
            label = str(item[0] if isinstance(item, tuple) else item)
            print(f"  [{key}] {label}")
        try:
            raw = input(f"Select [{default}]: ").strip().lower()
        except KeyboardInterrupt as exc:
            raise UserCancelled(via_keyboard=True) from exc
        if raw in {"q", "quit", "exit"}:
            raise UserCancelled()
        if raw == "b" and allow_back:
            return "back"
        if not raw:
            return default
        if raw in options:
            return raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan, apply, inspect, or reset hosting.configuration.v3")
    parser.add_argument("operation", choices=("plan", "apply", "inspect", "status", "reset"), nargs="?", default="inspect")
    parser.add_argument("--mp13-config-file", type=Path, default=None)
    parser.add_argument("--hosting-root", default="")
    parser.add_argument("--packages-root", default="")
    parser.add_argument("--environments-root", default="")
    parser.add_argument("--hosting-config-json", default="")
    parser.add_argument("--hosting-config-file", type=Path, default=None)
    parser.add_argument("--expected-config-revision", default="")
    parser.add_argument("--expected-hosting-revision", default="")
    parser.add_argument("--allow-nonempty-destinations", action="store_true")
    parser.add_argument("--allow-cross-volume", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    return parser


def _configuration(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.hosting_config_json and args.hosting_config_file:
        raise ValueError("hosting_configuration_inputs_conflict")
    raw = str(args.hosting_config_json or "")
    if args.hosting_config_file:
        raw = args.hosting_config_file.read_text(encoding="utf-8")
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("hosting_configuration_type_invalid")
    return dict(value)


def _request(args: argparse.Namespace) -> Dict[str, Any]:
    roots = {
        key: value
        for key, value in {
            "hosting_root_dir": str(args.hosting_root or "").strip(),
            "packages_root_dir": str(args.packages_root or "").strip(),
            "environments_root_dir": str(args.environments_root or "").strip(),
        }.items()
        if value
    }
    return {
        "contract": "hosting.setup.v1",
        "operation": args.operation,
        "mp13_config_file": args.mp13_config_file,
        "roots": roots or None,
        "hosting_configuration": _configuration(args),
        "expected_config_revision": str(args.expected_config_revision or ""),
        "expected_hosting_revision": str(args.expected_hosting_revision or ""),
        "allow_nonempty_destinations": bool(args.allow_nonempty_destinations),
        "allow_cross_volume": bool(args.allow_cross_volume),
        "confirm": bool(args.confirm),
        "confirm_reset": bool(args.confirm),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        request = _request(args)
        action = {
            "plan": plan_local_hosting_setup,
            "apply": apply_local_hosting_setup,
            "inspect": inspect_local_hosting_setup,
            "status": get_local_hosting_setup_status,
            "reset": reset_local_hosting_setup,
        }[args.operation]
        result = action(request)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        code = str(getattr(exc, "code", "") or str(exc) or "hosting_setup_failed")
        print(json.dumps({"status": "error", "code": code}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
