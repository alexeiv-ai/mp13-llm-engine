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

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        return pid_alive(pid)

    def is_alive(self) -> bool:
        info = self.read()
        if not info:
            return False
        return self._pid_alive(int(info.get("pid") or 0))

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
