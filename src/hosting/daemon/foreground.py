"""Foreground daemon entrypoints."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_DAEMON_PORT, DEFAULT_HTTP_INGRESS_PORT
from .http_ingress import EngineHostHttpIngressDaemon
from .lifecycle import _apply_foreground_terminal_disconnect_policy
from .local_ipc import EngineHostDaemon


def run_daemon_foreground(
    *,
    port: int = DEFAULT_DAEMON_PORT,
    pid_file: Optional[Path] = None,
    mp13_config_file: Optional[Path] = None,
    runtime_profile: str = "foreground_terminal_bound",
) -> None:
    """Start daemon in the foreground (blocks until stopped)."""
    daemon = EngineHostDaemon(
        port=port,
        pid_file=pid_file,
        mp13_config_file=mp13_config_file,
        runtime_profile=runtime_profile,
    )
    _apply_foreground_terminal_disconnect_policy(daemon)
    asyncio.run(daemon.run())

def run_http_ingress_foreground(
    *,
    port: int = DEFAULT_HTTP_INGRESS_PORT,
    pid_file: Optional[Path] = None,
    mp13_config_file: Optional[Path] = None,
) -> None:
    """Start HTTP ingress daemon in the foreground (blocks until stopped)."""
    daemon = EngineHostHttpIngressDaemon(
        port=port,
        pid_file=pid_file,
        mp13_config_file=mp13_config_file,
    )
    daemon.run()
