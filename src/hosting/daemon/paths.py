"""Daemon path and local IPC endpoint helpers."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict


def _default_state_dir() -> Path:
    # Keep hosting bootstrap lightweight: avoid importing mp13_engine package
    # during module import to prevent unrelated heavy dependency side-effects.
    return (Path.home() / ".mp13-llm" / "hosting" / "state").expanduser().resolve()


def _default_pid_file() -> Path:
    return _default_state_dir() / "daemon.pid"


def _default_http_pid_file() -> Path:
    return _default_state_dir() / "daemon_http.pid"


def _daemon_local_ipc_endpoint(pid_path: Path) -> Dict[str, str]:
    resolved = pid_path.expanduser().resolve()
    suffix = hashlib.sha256(str(resolved).encode("utf-8", errors="ignore")).hexdigest()[:16]
    if os.name == "nt":
        return {
            "transport": "local_ipc",
            "family": "AF_PIPE",
            "address": f"\\\\.\\pipe\\mp13-host-daemon-{suffix}",
        }
    return {
        "transport": "local_ipc",
        "family": "AF_UNIX",
        "address": str((resolved.parent / f"{resolved.stem}-{suffix}.sock").resolve()),
    }
