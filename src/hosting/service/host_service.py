"""
Standalone engine host service logic.

This module is intentionally backend-agnostic: it only manages engine-host
process lifecycle and generic control-plane state (claims/tokens/resources).
"""
from __future__ import annotations

import os
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
from .execution_receipts import ToolboxExecutionReceiptLedger
from .logs import LogsMixin
from .metrics import MetricsMixin
from .policy import PolicyMixin
from .proxy import ProxyMixin
from .sandbox_api import SandboxApiMixin
from .state import StateMixin
from .toolbox_env import ToolboxEnvironmentMixin
from .toolbox_runtime import ToolboxRuntimeMixin
from .workflow_helpers import WorkflowHelperMixin


class EngineHostService(CoreMixin, MetricsMixin, StateMixin, ConfigMixin, ControlMixin, AuthMixin, ClaimsMixin, PolicyMixin, EnginesMixin, ProxyMixin, SandboxApiMixin, LogsMixin, ToolboxEnvironmentMixin, ToolboxRuntimeMixin, WorkflowHelperMixin):
    """Engine host service for terminal-command control."""
    _metrics_lock = threading.Lock()
    _runtime_metrics: Optional[Dict[str, Any]] = None
    _toolbox_lock_guard = threading.Lock()
    _toolbox_locks: Dict[str, threading.RLock] = {}
    _receipt_ledger_guard = threading.Lock()
    _receipt_ledgers: Dict[str, ToolboxExecutionReceiptLedger] = {}

    def __init__(
        self,
        *,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
        toolbox_receipt_retention_seconds: Optional[float] = None,
        toolbox_receipt_tombstone_seconds: Optional[float] = None,
        toolbox_receipt_max_count: Optional[int] = None,
        toolbox_receipt_max_tombstones: Optional[int] = None,
        toolbox_receipt_max_result_bytes: Optional[int] = None,
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

        def _float_setting(value: Optional[float], env_name: str, default: float) -> float:
            if value is not None:
                return float(value)
            try:
                return float(os.environ.get(env_name, default))
            except (TypeError, ValueError):
                return default

        def _int_setting(value: Optional[int], env_name: str, default: int) -> int:
            if value is not None:
                return int(value)
            try:
                return int(os.environ.get(env_name, default))
            except (TypeError, ValueError):
                return default

        self._toolbox_execution_receipt_options = {
            "receipt_retention_seconds": _float_setting(
                toolbox_receipt_retention_seconds,
                "MP13_TOOLBOX_RECEIPT_RETENTION_SECONDS",
                7 * 24 * 3600,
            ),
            "tombstone_retention_seconds": _float_setting(
                toolbox_receipt_tombstone_seconds,
                "MP13_TOOLBOX_RECEIPT_TOMBSTONE_SECONDS",
                14 * 24 * 3600,
            ),
            "max_receipts": _int_setting(toolbox_receipt_max_count, "MP13_TOOLBOX_RECEIPT_MAX_COUNT", 10_000),
            "max_tombstones": _int_setting(
                toolbox_receipt_max_tombstones,
                "MP13_TOOLBOX_RECEIPT_MAX_TOMBSTONES",
                20_000,
            ),
            "max_result_bytes": _int_setting(
                toolbox_receipt_max_result_bytes,
                "MP13_TOOLBOX_RECEIPT_MAX_RESULT_BYTES",
                64 * 1024,
            ),
        }
        self._ensure_metrics_initialized()

    @property
    def _toolbox_execution_receipts(self) -> ToolboxExecutionReceiptLedger:
        path = (self.hosting_root / "state" / "toolbox_execution_receipts.json").resolve()
        key = str(path)
        with self._receipt_ledger_guard:
            ledger = self._receipt_ledgers.get(key)
            if ledger is None:
                ledger = ToolboxExecutionReceiptLedger(path, **self._toolbox_execution_receipt_options)
                self._receipt_ledgers[key] = ledger
            return ledger

    def close(self) -> None:
        node_registry = getattr(self, "_workflow_python_node_runtime_registry_instance", None)
        if node_registry is not None:
            try:
                node_registry.shutdown()
            except Exception:
                pass
            try:
                setattr(self, "_workflow_python_node_runtime_registry_instance", None)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
