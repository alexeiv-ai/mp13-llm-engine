"""Foreground daemon lifecycle policy helpers."""
from __future__ import annotations

import signal
from typing import Dict

from .local_ipc import EngineHostDaemon


def _apply_foreground_terminal_disconnect_policy(daemon: EngineHostDaemon) -> str:
    """
    Apply terminal-disconnect handling for foreground runtime profile.

    In foreground mode, keep-daemon-running policy ignores SIGHUP where available.
    """
    mode = str(daemon._runtime_profile or "").strip().lower()  # noqa: SLF001
    if mode != "foreground_terminal_bound":
        return "not_foreground"
    policy = daemon.svc.get_lifecycle_policy_effective()
    policy_cfg = dict(policy.get("policy") or {})
    action = str(policy_cfg.get("on_terminal_disconnect") or "stop_daemon").strip().lower()
    if action != "keep_daemon_running":
        return "stop_daemon"
    sighup = getattr(signal, "SIGHUP", None)
    if sighup is None:
        return "keep_daemon_running_no_sighup"
    signal.signal(sighup, signal.SIG_IGN)
    return "keep_daemon_running_ignore_sighup"
