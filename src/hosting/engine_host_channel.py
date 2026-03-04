"""
Backend-side adapter to interact with engine host.

Primary path: persistent connection to a running EngineHostDaemon.
  - Local mode (no SSH): LocalSocketConnection via TCP to 127.0.0.1:<port>
  - SSH mode:            SSHRelayConnection via SSH subprocess running --relay

Fallback: original per-command subprocess (engine_host_cli) when no daemon is
reachable and auto-bootstrap is disabled or fails.

The entire existing public API is preserved; no callers need changes.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EngineHostControlChannel:
    """Command-channel wrapper — persistent daemon connection with subprocess fallback."""

    def __init__(self, control_settings: Optional[Dict[str, Any]] = None):
        self.control_settings: Dict[str, Any] = dict(control_settings or {})
        self._base_cmd: List[str] = []
        self._engines_state_file = self.control_settings.get("engine_host_state_file")
        self._control_state_file = self.control_settings.get("engine_host_control_state_file")
        self._timeout = float(self.control_settings.get("engine_host_timeout_seconds") or 15.0)
        # Connection management
        self._connection: Optional[Any] = None  # BaseConnection instance
        self._connection_lock = threading.Lock()
        self._auto_bootstrap_daemon: bool = bool(
            self.control_settings.get("engine_host_daemon_auto_bootstrap", True)
        )
        self._daemon_port_override: int = int(
            self.control_settings.get("engine_host_daemon_port") or 0
        )
        self._refresh_base_cmd()

    @staticmethod
    def _is_localhost_target(target: str) -> bool:
        t = str(target or "").strip().lower()
        if not t:
            return True
        if t.startswith("ssh://"):
            t = t[6:]
        if "@" in t:
            t = t.split("@", 1)[1]
        t = t.split(":", 1)[0]
        return t in {"localhost", "127.0.0.1", "::1"}

    def _refresh_base_cmd(self) -> None:
        raw_cmd = self.control_settings.get("engine_host_cmd")
        if isinstance(raw_cmd, str) and str(raw_cmd).strip():
            self._base_cmd = shlex.split(str(raw_cmd))
            return
        if isinstance(raw_cmd, list) and raw_cmd:
            self._base_cmd = [str(x) for x in raw_cmd]
            return
        target = str(self.control_settings.get("engine_host_ssh_target") or "").strip()
        if not target:
            ctrl = str(self.control_settings.get("control_endpoint") or "").strip()
            if ctrl and (ctrl.startswith("ssh://") or ("@" in ctrl and not self._is_localhost_target(ctrl))):
                target = ctrl
        if target:
            if target.startswith("ssh://"):
                target = target[6:]
            remote_cmd = str(self.control_settings.get("engine_host_remote_cmd") or "python -m hosting.engine_host_cli")
            ssh_key = str(self.control_settings.get("control_ssh_key") or "").strip()
            prefix: List[str] = ["ssh"]
            if ssh_key:
                prefix += ["-i", ssh_key]
            self._base_cmd = prefix + [target] + shlex.split(remote_cmd)
            return
        self._base_cmd = [sys.executable, "-m", "hosting.engine_host_cli"]

    def _ensure_ssh_key_policy(self) -> None:
        info = self.get_target()
        if str(info.get("mode") or "") != "ssh":
            return
        key = str(self.control_settings.get("control_ssh_key") or "").strip()
        if key:
            return
        raise RuntimeError("SSH key is required for non-local engine host target")

    def get_target(self) -> Dict[str, Any]:
        cmd = list(self._base_cmd)
        is_ssh = bool(cmd and str(cmd[0]).lower().endswith("ssh"))
        target = str(self.control_settings.get("engine_host_ssh_target") or "")
        if not target and is_ssh and len(cmd) > 1:
            target = str(cmd[1] or "")
        return {
            "mode": "ssh" if is_ssh else "local",
            "target": target or None,
            "remote_cmd": str(self.control_settings.get("engine_host_remote_cmd") or "python -m hosting.engine_host_cli"),
            "base_cmd": cmd,
        }

    def set_target(
        self,
        *,
        mode: str,
        target: Optional[str] = None,
        ssh_key: Optional[str] = None,
        remote_cmd: Optional[str] = None,
        engine_host_cmd: Optional[List[str]] = None,
        engine_host_state_file: Optional[str] = None,
        engine_host_control_state_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        m = str(mode or "local").strip().lower()
        if m not in {"local", "ssh"}:
            raise ValueError("mode must be 'local' or 'ssh'")
        if m == "local":
            self.control_settings["engine_host_ssh_target"] = None
        else:
            tgt = str(target or "").strip()
            if not tgt:
                raise ValueError("target is required for ssh mode")
            self.control_settings["engine_host_ssh_target"] = tgt
            existing_key = str(self.control_settings.get("control_ssh_key") or "").strip()
            incoming_key = str(ssh_key or "").strip()
            if not (incoming_key or existing_key):
                raise ValueError("ssh_key is required for ssh mode")
        if ssh_key is not None:
            self.control_settings["control_ssh_key"] = str(ssh_key).strip() or None
        if remote_cmd is not None:
            self.control_settings["engine_host_remote_cmd"] = str(remote_cmd).strip() or None
        if engine_host_cmd is not None:
            self.control_settings["engine_host_cmd"] = [str(x) for x in list(engine_host_cmd or [])] or None
        if engine_host_state_file is not None:
            self.control_settings["engine_host_state_file"] = str(engine_host_state_file).strip() or None
        if engine_host_control_state_file is not None:
            self.control_settings["engine_host_control_state_file"] = str(engine_host_control_state_file).strip() or None
        self._engines_state_file = self.control_settings.get("engine_host_state_file")
        self._control_state_file = self.control_settings.get("engine_host_control_state_file")
        self._refresh_base_cmd()
        # Reset persistent connection: target has changed
        with self._connection_lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None
        if ssh_key is not None:
            try:
                _ = self._invoke("set-control-config", {"ssh_key": str(ssh_key).strip() or None})
            except Exception:
                pass
        return self.get_target()

    def _cmd_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parents[1])
        py_path = str(env.get("PYTHONPATH") or "")
        if src_root not in py_path.split(os.pathsep):
            env["PYTHONPATH"] = src_root if not py_path else f"{src_root}{os.pathsep}{py_path}"
        return env

    # ------------------------------------------------------------------
    # Persistent connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> Optional[Any]:
        """
        Return an active persistent connection or None (fall back to subprocess).

        SSH mode  → SSHRelayConnection (lazy-created, re-created if dead)
        Local mode → LocalSocketConnection to running daemon (auto-bootstrap if enabled)
        """
        from .engine_host_connection import LocalSocketConnection, SSHRelayConnection

        with self._connection_lock:
            target = self.get_target()
            mode = str(target.get("mode") or "local")

            if mode == "ssh":
                if self._connection is None or not self._connection.is_alive():
                    ssh_target = str(target.get("target") or "")
                    if not ssh_target:
                        return None
                    # Remote command defaults to --relay variant for SSH mode
                    raw_remote = str(
                        self.control_settings.get("engine_host_remote_cmd") or ""
                    ).strip()
                    if not raw_remote or "--relay" not in raw_remote:
                        raw_remote = "python -m hosting.engine_host_cli --relay"
                    self._connection = SSHRelayConnection(
                        ssh_target=ssh_target,
                        ssh_key=str(self.control_settings.get("control_ssh_key") or "").strip() or None,
                        remote_cmd=raw_remote,
                        timeout=self._timeout,
                    )
                return self._connection

            if mode == "local":
                from .engine_host_daemon import DaemonPidFile, start_daemon_background, DEFAULT_DAEMON_PORT

                pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
                pid_info = DaemonPidFile(pid_file_path)

                # If we already have a connection, check it first
                if self._connection is not None and isinstance(self._connection, LocalSocketConnection):
                    if self._connection.is_alive():
                        return self._connection
                    self._connection = None

                port = self._daemon_port_override or pid_info.get_port()
                if port and pid_info.is_alive():
                    conn = LocalSocketConnection(port=port, timeout=self._timeout)
                    self._connection = conn
                    return conn

                # Daemon not running: auto-bootstrap if enabled
                if self._auto_bootstrap_daemon:
                    try:
                        wait = float(
                            self.control_settings.get("engine_host_daemon_wait_ready_seconds") or 8.0
                        )
                        result = start_daemon_background(
                            port=self._daemon_port_override or DEFAULT_DAEMON_PORT,
                            wait_ready_seconds=wait,
                        )
                        new_port = int(result.get("port") or DEFAULT_DAEMON_PORT)
                        conn = LocalSocketConnection(port=new_port, timeout=self._timeout)
                        self._connection = conn
                        return conn
                    except Exception as exc:
                        logger.warning("Auto-bootstrap of local daemon failed: %s", exc)

                return None

            return None

    def _invoke_subprocess(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """Original per-command subprocess path (fallback)."""
        self._ensure_ssh_key_policy()
        argv = list(self._base_cmd) + ["--payload-stdin"]
        if self._engines_state_file:
            argv += ["--engines-state-file", str(self._engines_state_file)]
        if self._control_state_file:
            argv += ["--control-state-file", str(self._control_state_file)]
        argv += [str(command)]
        proc = subprocess.run(  # noqa: S603
            argv,
            input=json.dumps(dict(payload or {}), ensure_ascii=False),
            text=True,
            capture_output=True,
            env=self._cmd_env(),
            timeout=self._timeout,
            check=False,
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            raw = (proc.stderr or "").strip()
        if not raw:
            raise RuntimeError(f"engine host command '{command}' returned no output")
        try:
            out = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"engine host returned invalid json: {raw}") from exc
        if not isinstance(out, dict) or not out.get("ok"):
            msg = str((out or {}).get("error") or f"engine host command '{command}' failed")
            raise RuntimeError(msg)
        return out.get("result")

    def _invoke(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """
        Send a command and return the result.

        Tries persistent connection first; falls back to per-command subprocess.
        """
        conn = self._get_connection()
        if conn is not None:
            try:
                return conn.invoke(command, payload)
            except Exception as exc:
                logger.warning(
                    "Persistent connection failed for '%s': %s. Falling back to subprocess.",
                    command,
                    exc,
                )
                with self._connection_lock:
                    self._connection = None
        return self._invoke_subprocess(command, payload)

    # ------------------------------------------------------------------
    # Daemon lifecycle management (new public API)
    # ------------------------------------------------------------------

    def get_daemon_status(self) -> Dict[str, Any]:
        """Return current local daemon PID file info and liveness."""
        from .engine_host_daemon import DaemonPidFile
        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        info = pid_info.read() or {}
        return {
            "pid_file": str(pid_info.path),
            "pid": info.get("pid"),
            "port": info.get("port"),
            "started_at": info.get("started_at"),
            "alive": pid_info.is_alive(),
        }

    def bootstrap_daemon(self, *, wait_ready_seconds: float = 8.0) -> Dict[str, Any]:
        """Start local daemon if not already running. Returns daemon status dict."""
        from .engine_host_daemon import DaemonPidFile, start_daemon_background, DEFAULT_DAEMON_PORT
        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        if pid_info.is_alive():
            return {"already_running": True, **self.get_daemon_status()}
        result = start_daemon_background(
            port=self._daemon_port_override or DEFAULT_DAEMON_PORT,
            wait_ready_seconds=wait_ready_seconds,
        )
        with self._connection_lock:
            self._connection = None  # Force reconnect on next invoke
        return {"already_running": False, **result, **self.get_daemon_status()}

    def stop_daemon(self) -> Dict[str, Any]:
        """Send graceful shutdown signal to local daemon."""
        from .engine_host_daemon import DaemonPidFile
        from .engine_host_connection import LocalSocketConnection
        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        info = pid_info.read()
        if not info:
            return {"status": "not_running"}
        port = int(info.get("port") or 0)
        token = str(info.get("shutdown_token") or "")
        if not port or not token:
            return {"status": "invalid_pid_file"}
        try:
            conn = LocalSocketConnection(port=port, timeout=5.0, max_reconnect_attempts=1)
            conn.invoke("__shutdown__", {"shutdown_token": token})
            conn.close()
            with self._connection_lock:
                self._connection = None
            return {"status": "shutdown_sent"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def close_connection(self) -> None:
        """Close and discard the current persistent connection."""
        with self._connection_lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None

    def discover_running(self) -> List[Dict[str, Any]]:
        res = self._invoke("discover-running", {})
        return list(res or []) if isinstance(res, list) else []

    def get_registration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        res = self._invoke("get-registration", {"engine_id": str(engine_id)})
        return dict(res or {}) if isinstance(res, dict) else None

    def spawn_process(self, *, engine_id: str, command: List[str], cwd: Optional[str] = None, endpoint: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        res = self._invoke(
            "spawn",
            {
                "engine_id": str(engine_id),
                "command": [str(x) for x in list(command or [])],
                "cwd": str(cwd) if cwd else None,
                "endpoint": str(endpoint) if endpoint else None,
                "env": dict(env or {}),
            },
        )
        return dict(res or {})

    def shutdown_managed(self, engine_id: str, *, timeout_seconds: float = 8.0) -> Dict[str, Any]:
        res = self._invoke("shutdown", {"engine_id": str(engine_id), "timeout_seconds": float(timeout_seconds)})
        return dict(res or {})

    def ensure_running(self, engine_id: str) -> Dict[str, Any]:
        res = self._invoke("ensure-running", {"engine_id": str(engine_id)})
        return dict(res or {})

    def remove_registration(self, engine_id: str) -> Dict[str, Any]:
        res = self._invoke("remove-registration", {"engine_id": str(engine_id)})
        return dict(res or {})

    def claim_engine(self, engine_id: str, *, backend_id: Optional[str], exclusive: bool = False) -> Dict[str, Any]:
        res = self._invoke("claim-engine", {"engine_id": str(engine_id), "backend_id": backend_id, "exclusive": bool(exclusive)})
        return dict(res or {})

    def claim_endpoint(self, *, backend_id: Optional[str], exclusive: bool = False) -> Dict[str, Any]:
        res = self._invoke("claim-endpoint", {"backend_id": backend_id, "exclusive": bool(exclusive)})
        return dict(res or {})

    def get_claim_status(self, engine_id: str) -> Dict[str, Any]:
        res = self._invoke("claim-status", {"engine_id": str(engine_id)})
        return dict(res or {})

    def issue_token(self, engine_id: str, *, backend_id: Optional[str]) -> Dict[str, Any]:
        res = self._invoke("issue-token", {"engine_id": str(engine_id), "backend_id": backend_id})
        return dict(res or {})

    def validate_token(self, engine_id: str, token: str) -> bool:
        res = self._invoke("validate-token", {"engine_id": str(engine_id), "token": str(token or "")})
        return bool(res)

    def claim_resource(self, resource_kind: str, resource_id: str, *, backend_id: Optional[str], exclusive: bool = False) -> Dict[str, Any]:
        res = self._invoke(
            "claim-resource",
            {"resource_kind": str(resource_kind), "resource_id": str(resource_id), "backend_id": backend_id, "exclusive": bool(exclusive)},
        )
        return dict(res or {})

    def get_resource_claim_status(self, resource_kind: str, resource_id: str) -> Dict[str, Any]:
        res = self._invoke("resource-claim-status", {"resource_kind": str(resource_kind), "resource_id": str(resource_id)})
        return dict(res or {})

    def issue_resource_token(self, resource_kind: str, resource_id: str, *, backend_id: Optional[str]) -> Dict[str, Any]:
        res = self._invoke("issue-resource-token", {"resource_kind": str(resource_kind), "resource_id": str(resource_id), "backend_id": backend_id})
        return dict(res or {})

    def validate_resource_token(self, resource_kind: str, resource_id: str, token: str) -> bool:
        res = self._invoke("validate-resource-token", {"resource_kind": str(resource_kind), "resource_id": str(resource_id), "token": str(token or "")})
        return bool(res)

    def list_engine_configs(self) -> List[Dict[str, Any]]:
        res = self._invoke("list-configs", {})
        return list(res or []) if isinstance(res, list) else []

    def create_engine_config(self, *, name: str, config: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        res = self._invoke("create-config", {"name": str(name), "config": dict(config or {}), "overwrite": bool(overwrite)})
        return dict(res or {})

    def models_from_config(self, config_path: str) -> List[Dict[str, Any]]:
        res = self._invoke("models-from-config", {"config_path": str(config_path or "default")})
        return list(res or []) if isinstance(res, list) else []

    def connect_from_config(self, *, config_path: str, engine_id: Optional[str] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
        res = self._invoke(
            "connect-from-config",
            {
                "config_path": str(config_path or "default"),
                "engine_id": str(engine_id).strip() if engine_id else None,
                "model_path": str(model_path).strip() if model_path else None,
            },
        )
        return dict(res or {})

    def inspect_engine_capabilities(self, *, engine_id: str, endpoint: str) -> Dict[str, Any]:
        res = self._invoke(
            "inspect-capabilities",
            {
                "engine_id": str(engine_id or "").strip(),
                "endpoint": str(endpoint or "").strip(),
            },
        )
        return dict(res or {})

    def logs_tail(self, *, engine_id: str, lines: int = 200, max_bytes: int = 65536) -> Dict[str, Any]:
        res = self._invoke(
            "logs-tail",
            {
                "engine_id": str(engine_id or "").strip(),
                "lines": int(lines or 200),
                "max_bytes": int(max_bytes or 65536),
            },
        )
        return dict(res or {})

    def logs_follow(self, *, engine_id: str, cursor: int = 0, max_bytes: int = 65536, max_lines: int = 500) -> Dict[str, Any]:
        res = self._invoke(
            "logs-follow",
            {
                "engine_id": str(engine_id or "").strip(),
                "cursor": int(cursor or 0),
                "max_bytes": int(max_bytes or 65536),
                "max_lines": int(max_lines or 500),
            },
        )
        return dict(res or {})

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            p = int(pid or 0)
            if p <= 0:
                return False
            os.kill(p, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False
