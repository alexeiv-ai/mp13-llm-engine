"""Daemon PID file management."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .._process_utils import pid_alive
from .paths import _default_pid_file
from .security import _atomic_write_secure_json


class DaemonPidFile:
    """Read/write the daemon PID file used for discovery by CLI and channel."""

    def __init__(self, path: Optional[Path] = None):
        self.path = (Path(path) if path else _default_pid_file()).expanduser().resolve()

    def write(
        self,
        *,
        pid: int,
        port: int,
        shutdown_token: str,
        transport: Optional[str] = None,
        ipc_family: Optional[str] = None,
        ipc_address: Optional[str] = None,
        lifecycle_state: str = "running",
    ) -> None:
        payload = {
            "version": 1,
            "pid": int(pid),
            "port": int(port),
            "started_at": time.time(),
            "shutdown_token": str(shutdown_token),
            "transport": str(transport or "").strip() or None,
            "ipc_family": str(ipc_family or "").strip() or None,
            "ipc_address": str(ipc_address or "").strip() or None,
            "lifecycle_state": str(lifecycle_state or "running").strip() or "running",
        }
        _atomic_write_secure_json(self.path, payload)

    def read(self) -> Optional[Dict[str, Any]]:
        try:
            if not self.path.exists():
                return None
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return dict(raw) if isinstance(raw, dict) else None
        except SystemError:
            # Defensive guard for Windows stale-exception edge cases surfaced
            # through pathlib/os.stat while probing daemon readiness.
            return None
        except Exception:
            return None

    def remove(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass

    def mark_shutting_down(self, *, reason: Optional[str] = None, requested_by: Optional[str] = None) -> None:
        info = self.read() or {}
        if not info:
            return
        payload = dict(info)
        payload["version"] = int(payload.get("version") or 1)
        payload["lifecycle_state"] = "shutting_down"
        payload["shutdown_requested_at"] = time.time()
        if reason:
            payload["shutdown_reason"] = str(reason)
        if requested_by:
            payload["shutdown_requested_by"] = str(requested_by)
        _atomic_write_secure_json(self.path, payload)

    def update_shutdown_progress(self, progress: Dict[str, Any]) -> None:
        info = self.read() or {}
        if not info:
            return
        payload = dict(info)
        payload["version"] = int(payload.get("version") or 1)
        payload["lifecycle_state"] = "shutting_down"
        payload["shutdown_progress"] = dict(progress or {})
        payload["shutdown_progress_updated_at"] = time.time()
        _atomic_write_secure_json(self.path, payload)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        return pid_alive(pid)

    def is_alive(self) -> bool:
        info = self.read()
        if not info:
            return False
        state = str(info.get("lifecycle_state") or "running").strip().lower()
        if state in {"shutting_down", "stopping", "stopped"}:
            return False
        return self._pid_alive(int(info.get("pid") or 0))

    def process_alive(self) -> bool:
        info = self.read()
        if not info:
            return False
        return self._pid_alive(int(info.get("pid") or 0))

    def lifecycle_state(self) -> str:
        info = self.read() or {}
        return str(info.get("lifecycle_state") or "running").strip() or "running"

    def get_port(self) -> Optional[int]:
        info = self.read()
        if not info:
            return None
        port = int(info.get("port") or 0)
        return port if port > 0 else None

    def get_shutdown_token(self) -> Optional[str]:
        info = self.read()
        if not info:
            return None
        return str(info.get("shutdown_token") or "").strip() or None

    def get_local_transport(self) -> Dict[str, Optional[str]]:
        info = self.read() or {}
        return {
            "transport": str(info.get("transport") or "").strip() or None,
            "ipc_family": str(info.get("ipc_family") or "").strip() or None,
            "ipc_address": str(info.get("ipc_address") or "").strip() or None,
            "shutdown_token": str(info.get("shutdown_token") or "").strip() or None,
        }
