"""
mp13 engine hosting package.

Public API — stdlib only at module level, no heavy imports (torch/transformers etc.).

Key classes and functions available for import:
    EngineHostService         — file-backed process lifecycle and control-plane state
    EngineHostDaemon          — asyncio TCP daemon server (127.0.0.1:default 19876)
    DaemonPidFile             — read/write/probe the daemon PID file
    DEFAULT_DAEMON_PORT       — default TCP port (19876)
    run_daemon_foreground()   — start daemon blocking in foreground
    start_daemon_background() — spawn daemon as detached background process
    LocalSocketConnection     — persistent TCP socket to local daemon
    SSHRelayConnection        — persistent SSH subprocess running --relay
    BaseConnection            — abstract base for connection strategies
    ConnectionError           — raised on unrecoverable connection failure
    EngineHostControlChannel  — command channel with daemon connection + subprocess fallback
    EngineProcessSupervisor   — in-process persisted tracker for managed worker processes
"""
from __future__ import annotations

from .engine_host_service import EngineHostService
from .engine_host_daemon import (
    EngineHostDaemon,
    DaemonPidFile,
    DEFAULT_DAEMON_PORT,
    run_daemon_foreground,
    start_daemon_background,
)
from .engine_host_connection import (
    BaseConnection,
    LocalSocketConnection,
    SSHRelayConnection,
    ConnectionError,
)
from .engine_host_channel import EngineHostControlChannel
from .engine_process_supervisor import EngineProcessSupervisor

__all__ = [
    "EngineHostService",
    "EngineHostDaemon",
    "DaemonPidFile",
    "DEFAULT_DAEMON_PORT",
    "run_daemon_foreground",
    "start_daemon_background",
    "BaseConnection",
    "LocalSocketConnection",
    "SSHRelayConnection",
    "ConnectionError",
    "EngineHostControlChannel",
    "EngineProcessSupervisor",
]
