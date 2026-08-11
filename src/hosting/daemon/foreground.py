"""Foreground daemon entrypoints."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping, Optional

from .constants import DEFAULT_DAEMON_PORT, DEFAULT_HTTP_INGRESS_PORT
from .http_ingress import EngineHostHttpIngressDaemon
from .lifecycle import _apply_foreground_terminal_disconnect_policy
from .local_ipc import EngineHostDaemon
from .toolbox_launch_config import resolve_toolbox_launch_configuration


def run_daemon_foreground(
    *,
    port: int = DEFAULT_DAEMON_PORT,
    pid_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
    runtime_profile: str = "foreground_terminal_bound",
    toolbox_config_file: Optional[Path] = None,
    toolbox_host_project_configuration: Optional[Mapping[str, Any]] = None,
    toolbox_artifact_sources: Optional[Mapping[str, Path]] = None,
    toolbox_trust_public_keys: Optional[Mapping[str, str]] = None,
    toolbox_source_credentials: Optional[Mapping[str, str]] = None,
    toolbox_dependency_policy: Optional[Mapping[str, Any]] = None,
) -> None:
    """Start daemon in the foreground (blocks until stopped)."""
    toolbox_kwargs = resolve_toolbox_launch_configuration(
        toolbox_config_file=toolbox_config_file,
        toolbox_host_project_configuration=toolbox_host_project_configuration,
        toolbox_artifact_sources=toolbox_artifact_sources,
        toolbox_trust_public_keys=toolbox_trust_public_keys,
        toolbox_source_credentials=toolbox_source_credentials,
        toolbox_dependency_policy=toolbox_dependency_policy,
    )
    daemon = EngineHostDaemon(
        port=port,
        pid_file=pid_file,
        engines_state_file=engines_state_file,
        control_state_file=control_state_file,
        runtime_profile=runtime_profile,
        **toolbox_kwargs,
    )
    _apply_foreground_terminal_disconnect_policy(daemon)
    asyncio.run(daemon.run())

def run_http_ingress_foreground(
    *,
    port: int = DEFAULT_HTTP_INGRESS_PORT,
    pid_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
) -> None:
    """Start HTTP ingress daemon in the foreground (blocks until stopped)."""
    daemon = EngineHostHttpIngressDaemon(
        port=port,
        pid_file=pid_file,
        engines_state_file=engines_state_file,
        control_state_file=control_state_file,
    )
    daemon.run()
