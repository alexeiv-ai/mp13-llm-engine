"""Engine host daemon implementation package."""
from __future__ import annotations

from .background import start_daemon_background, start_http_ingress_background
from .constants import DEFAULT_DAEMON_PORT, DEFAULT_HTTP_INGRESS_PORT
from .foreground import run_daemon_foreground, run_http_ingress_foreground
from .http_ingress import EngineHostHttpIngressDaemon
from .lifecycle import _apply_foreground_terminal_disconnect_policy
from .local_ipc import EngineHostDaemon
from .paths import (
    _daemon_local_ipc_endpoint,
    _default_http_pid_file,
    _default_pid_file,
    _default_state_dir,
)
from .pidfile import DaemonPidFile
from .security import (
    _atomic_write_secure_json,
    _current_windows_account_name,
    _secure_path,
    _secure_state_parent_dir,
    _tighten_windows_acl,
)

__all__ = [
    "DEFAULT_DAEMON_PORT",
    "DEFAULT_HTTP_INGRESS_PORT",
    "DaemonPidFile",
    "EngineHostDaemon",
    "EngineHostHttpIngressDaemon",
    "run_daemon_foreground",
    "run_http_ingress_foreground",
    "start_daemon_background",
    "start_http_ingress_background",
    "_default_state_dir",
    "_default_pid_file",
    "_default_http_pid_file",
    "_daemon_local_ipc_endpoint",
    "_current_windows_account_name",
    "_tighten_windows_acl",
    "_secure_state_parent_dir",
    "_secure_path",
    "_atomic_write_secure_json",
    "_apply_foreground_terminal_disconnect_policy",
]
