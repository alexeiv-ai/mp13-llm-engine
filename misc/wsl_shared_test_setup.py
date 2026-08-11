from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_ENTRIES = (
    "src",
    "tests",
    "misc",
    "pyproject.toml",
    "README.md",
)

OPTIONAL_ENTRIES = (
    "mp13chat.py",
    "mp13config.py",
    "configs",
    "demo",
)


def _project_root(raw_root: str | None) -> Path:
    if raw_root:
        return Path(raw_root).expanduser().absolute()
    return Path.cwd().absolute()


def _run(argv: list[str], *, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )


def _is_wsl() -> bool:
    try:
        text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return "microsoft" in text or "wsl" in text


def cmd_check(root: Path) -> int:
    issues: list[str] = []
    print(f"root: {root}")
    print(f"platform: {sys.platform}")
    print(f"wsl: {'yes' if _is_wsl() else 'no'}")

    for name in REQUIRED_ENTRIES:
        path = root / name
        exists = path.exists()
        kind = "symlink" if path.is_symlink() else "local"
        print(f"{name}: {'ok' if exists else 'missing'} ({kind if exists else 'n/a'})")
        if not exists:
            issues.append(f"missing required entry: {name}")

    for name in OPTIONAL_ENTRIES:
        path = root / name
        if path.exists():
            kind = "symlink" if path.is_symlink() else "local"
            print(f"{name}: present ({kind})")

    poetry = _run(["poetry", "env", "info", "-p"], root=root)
    if poetry.returncode != 0:
        issues.append("poetry env info -p failed")
        print("poetry_env: error")
        if poetry.stderr.strip():
            print(f"detail: {poetry.stderr.strip()}")
    else:
        env_path = poetry.stdout.strip()
        print(f"poetry_env: {env_path}")
        if not env_path:
            issues.append("poetry env path is empty")

    imports = _run(
        [
            "poetry",
            "run",
            "python",
            "-c",
            (
                "import sys; sys.path.insert(0, 'src'); "
                "import pydantic, pytest; "
                "from hosting.service.host_service import EngineHostService; "
                "print('imports-ok')"
            ),
        ],
        root=root,
    )
    if imports.returncode != 0:
        issues.append("poetry run import check failed")
        print("imports: error")
        err = (imports.stderr or imports.stdout).strip()
        if err:
            print(f"detail: {err}")
    else:
        print(imports.stdout.strip() or "imports: ok")

    if issues:
        print("status: error")
        for item in issues:
            print(f"issue: {item}")
        return 1

    print("status: ok")
    return 0


def cmd_commands(root: Path) -> int:
    print(f"cd {root}")
    print("# Use a WSL/Linux Poetry environment; do not reuse the Windows .venv or daemon state.")
    print("export POETRY_VIRTUALENVS_IN_PROJECT=true")
    print("# Keep pytest sockets and daemon state short and private to this WSL lane.")
    print("export PYTEST_DEBUG_TEMPROOT=/home/alx/r8p")
    print("export TMPDIR=/home/alx/r8t")
    print("mkdir -p \"$PYTEST_DEBUG_TEMPROOT\" \"$TMPDIR\"")
    print("poetry install --no-interaction")
    print(
        "poetry run python misc/hosting_test_lanes.py "
        "--lane fast --repeat 3 --durations 25 --json-output .tmp/wsl-fast-baseline.json"
    )
    print(
        "poetry run python misc/hosting_test_lanes.py "
        "--lane process --repeat 3 --durations 20 --json-output .tmp/wsl-process-baseline.json"
    )
    print(
        "poetry run python misc/hosting_test_lanes.py "
        "--lane native --collect-only --json-output .tmp/wsl-native-collection.json"
    )
    print("PYTHONPATH=src poetry run pytest tests/test_hosting_daemon_pidfile.py -q")
    print(
        "PYTHONPATH=src poetry run pytest "
        "tests/test_hosting_toolbox_sandbox.py -q "
        "-k 'startup_spec or spec_path or spec_hosting or toolbox_executor_ipc_end_to_end'"
    )
    print("PYTHONPATH=src poetry run pytest tests/test_engine_host_channel.py -q")
    print(
        "PYTHONPATH=src poetry run pytest "
        "tests/test_hosting_toolbox_sandbox.py "
        "tests/test_engine_host_channel.py "
        "tests/test_hosting_daemon_pidfile.py -q"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Helper script to validate and run tests in a WSL (Windows Subsystem for Linux) 'shadow' setup.\n\n"
            "Purpose:\n"
            "  When working on a project from Windows but running tests in WSL, this script helps\n"
            "  ensure that your WSL environment has the correct symlinks or mounted files\n"
            "  (src, tests, configs, etc.) and that the Poetry environment is correctly configured.\n\n"
            "Usage:\n"
            "  python wsl_shared_test_setup.py check     # Validates the directory structure and imports\n"
            "  python wsl_shared_test_setup.py commands  # Prints recommended pytest commands to run"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--root", type=str, help="Path to the shadow project root in WSL. Defaults to the current working directory.")
    sub = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    sub.add_parser("check", help="Validate the current WSL shadow setup (checks files, poetry env, and imports).")
    sub.add_parser("commands", help="Print the recommended WSL test commands to run manually.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = _project_root(args.root)
    if args.command == "check":
        return cmd_check(root)
    if args.command == "commands":
        return cmd_commands(root)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
