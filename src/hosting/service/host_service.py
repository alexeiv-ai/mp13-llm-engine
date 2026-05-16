"""
Standalone engine host service logic.

This module is intentionally backend-agnostic: it only manages engine-host
process lifecycle and generic control-plane state (claims/tokens/resources).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .auth import AuthMixin
from .claims import ClaimsMixin
from .constants import (
    DEFAULT_CONTROL_STATE_FILE,
    DEFAULT_ENGINES_STATE_FILE,
    VALID_AUTH_ROLES,
)
from .configs import ConfigMixin
from .control import ControlMixin
from .core import CoreMixin
from .engines import EnginesMixin
from .errors import ToolboxRolloutError
from .logs import LogsMixin
from .metrics import MetricsMixin
from .policy import PolicyMixin
from .proxy import ProxyMixin
from .sandbox_api import SandboxApiMixin
from .state import StateMixin
from .toolbox_env import ToolboxEnvironmentMixin
from .toolbox_runtime import ToolboxRuntimeMixin


class EngineHostService(CoreMixin, MetricsMixin, StateMixin, ConfigMixin, ControlMixin, AuthMixin, ClaimsMixin, PolicyMixin, EnginesMixin, ProxyMixin, SandboxApiMixin, LogsMixin, ToolboxEnvironmentMixin, ToolboxRuntimeMixin):
    """Engine host service for terminal-command control."""
    _metrics_lock = threading.Lock()
    _runtime_metrics: Optional[Dict[str, Any]] = None
    _toolbox_lock_guard = threading.Lock()
    _toolbox_locks: Dict[str, threading.RLock] = {}

    def __init__(
        self,
        *,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
    ):
        self.engines_state_file = (engines_state_file or DEFAULT_ENGINES_STATE_FILE).expanduser().resolve()
        raw_control = (control_state_file or DEFAULT_CONTROL_STATE_FILE).expanduser().resolve()
        if raw_control.suffix:
            self.hosting_root = raw_control.parent.resolve()
            self.control_state_file = self.hosting_root / "access_control.json"
        else:
            self.hosting_root = raw_control.resolve()
            self.control_state_file = self.hosting_root / "access_control.json"
        self._runtime_engines_lock = threading.RLock()
        self._runtime_engines: list[Dict[str, Any]] = []
        self._ensure_metrics_initialized()




