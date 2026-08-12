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


def _filesystem_type(path: Path) -> str:
    completed = subprocess.run(
        ["stat", "-f", "-c", "%T", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip().lower() if completed.returncode == 0 else "unknown"


def cmd_check(root: Path, *, venv: Path | None = None) -> int:
    issues: list[str] = []
    print(f"root: {root}")
    print(f"platform: {sys.platform}")
    print(f"wsl: {'yes' if _is_wsl() else 'no'}")
    print(f"source_filesystem: {_filesystem_type(root)}")

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

    if venv is not None:
        env_path = str(venv.expanduser().absolute())
        python = Path(env_path) / "bin" / "python"
        print(f"poetry_env: {env_path} (explicit)")
        if not python.is_file():
            issues.append(f"explicit WSL environment has no Python executable: {python}")
        env_filesystem = _filesystem_type(Path(env_path))
        print(f"poetry_env_filesystem: {env_filesystem}")
        if _is_wsl() and env_filesystem in {"9p", "v9fs"}:
            issues.append("WSL Poetry environment must use native Linux storage, not DrvFS/9P")
    else:
        poetry = _run(["poetry", "env", "info", "-p"], root=root)
        if poetry.returncode != 0:
            issues.append("poetry env info -p failed")
            print("poetry_env: error")
            if poetry.stderr.strip():
                print(f"detail: {poetry.stderr.strip()}")
            env_path = ""
            python = Path("python")
        else:
            env_path = poetry.stdout.strip()
            python = Path(env_path) / "bin" / "python"
            print(f"poetry_env: {env_path}")
            if not env_path:
                issues.append("poetry env path is empty")
            else:
                env_filesystem = _filesystem_type(Path(env_path))
                print(f"poetry_env_filesystem: {env_filesystem}")
                if _is_wsl() and env_filesystem in {"9p", "v9fs"}:
                    issues.append("WSL Poetry environment must use native Linux storage, not DrvFS/9P")

    imports = _run(
        [
            str(python),
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


def cmd_commands(root: Path, *, venv: Path | None = None) -> int:
    print(f"cd {root}")
    print("# Sources may remain Windows-mounted; all generated/runtime state stays on native WSL storage.")
    print(f"export WSL_VENV={venv or Path('/home/alx/mp13-wsl-venv')}")
    print("export POETRY_CACHE_DIR=/home/alx/.cache/pypoetry/mp13-cache")
    print("export PYTHONPYCACHEPREFIX=/home/alx/.cache/mp13-pycache")
    print("export PYTEST_DEBUG_TEMPROOT=/home/alx/.cache/mp13/pytest")
    print("export TMPDIR=/home/alx/.cache/mp13/tmp")
    print(
        "mkdir -p \"$WSL_VENV\" \"$POETRY_CACHE_DIR\" "
        "\"$PYTHONPYCACHEPREFIX\" \"$PYTEST_DEBUG_TEMPROOT\" \"$TMPDIR\""
    )
    print("# For a fresh environment: python3 -m venv \"$WSL_VENV\"")
    print("# Poetry downloads Linux packages directly into native storage; do not copy the Windows environment.")
    print("# Then install without selecting the mounted checkout's Windows .venv:")
    print("# VIRTUAL_ENV=\"$WSL_VENV\" PATH=\"$WSL_VENV/bin:$PATH\" POETRY_VIRTUALENVS_CREATE=false poetry install --no-interaction")
    print(
        "\"$WSL_VENV/bin/python\" misc/hosting_test_lanes.py "
        "--lane process --repeat 1 --durations 20 --json-output /home/alx/.cache/mp13/wsl-process.json"
    )
    print(
        "\"$WSL_VENV/bin/python\" misc/hosting_test_lanes.py "
        "--lane native --collect-only --json-output /home/alx/.cache/mp13/wsl-native-collection.json"
    )
    print("PYTHONPATH=src \"$WSL_VENV/bin/python\" -m pytest tests/test_hosting_daemon_pidfile.py -q")
    print(
        "PYTHONPATH=src \"$WSL_VENV/bin/python\" -m pytest "
        "tests/test_hosting_toolbox_sandbox.py -q "
        "-k 'startup_spec or spec_path or spec_hosting or toolbox_executor_ipc_end_to_end'"
    )
    print("PYTHONPATH=src \"$WSL_VENV/bin/python\" -m pytest tests/test_engine_host_channel.py -q")
    print(
        "PYTHONPATH=src \"$WSL_VENV/bin/python\" -m pytest "
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
    parser.add_argument("--venv", type=Path, help="Explicit native-WSL Poetry environment; avoids a mounted Windows .venv.")
    sub = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    sub.add_parser("check", help="Validate the current WSL shadow setup (checks files, poetry env, and imports).")
    sub.add_parser("commands", help="Print the recommended WSL test commands to run manually.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = _project_root(args.root)
    if args.command == "check":
        return cmd_check(root, venv=args.venv)
    if args.command == "commands":
        return cmd_commands(root, venv=args.venv)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
