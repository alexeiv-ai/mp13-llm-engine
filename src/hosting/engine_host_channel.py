"""
Backend-side adapter to interact with engine host.

Primary path: persistent connection to a running EngineHostDaemon.
  - Local mode (no SSH): LocalSocketConnection via local IPC discovered from PID file
  - SSH mode:            SSHRelayConnection via SSH subprocess running --relay

Fallback: original per-command subprocess (engine_host_cli) when no daemon is
reachable and auto-bootstrap is disabled or fails.

The entire existing public API is preserved; no callers need changes.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Keywords that indicate an expired or invalid session token in daemon error strings.
_SESSION_AUTH_ERROR_KEYWORDS = (
    "auth_failed",
    "session_expired",
    "invalid_session",
    "missing_or_invalid_session_token",
    "session_token_required",
    "session_not_found",
)


def _is_session_auth_error(msg: str) -> bool:
    ml = msg.lower()
    return any(k in ml for k in _SESSION_AUTH_ERROR_KEYWORDS)


class EngineHostControlChannel:
    """Command-channel wrapper — persistent daemon connection with subprocess fallback."""

    def __init__(self, control_settings: Optional[Dict[str, Any]] = None):
        self.control_settings: Dict[str, Any] = dict(control_settings or {})
        self._base_cmd: List[str] = []
        self._engines_state_file = self.control_settings.get("engine_host_state_file")
        self._control_state_file = self.control_settings.get("engine_host_control_state_file")
        self._timeout = float(self.control_settings.get("engine_host_timeout_seconds") or 15.0)
        self._session_token: Optional[str] = str(
            self.control_settings.get("engine_host_session_token") or ""
        ).strip() or None
        self._key_id: Optional[str] = str(
            self.control_settings.get("engine_host_key_id") or ""
        ).strip() or None
        self._key_secret: Optional[str] = str(
            self.control_settings.get("engine_host_key_secret") or ""
        ).strip() or None
        self._session_scope: str = str(
            self.control_settings.get("engine_host_session_scope") or "control"
        ).strip().lower() or "control"
        self._session_ttl_seconds: int = int(
            self.control_settings.get("engine_host_session_ttl_seconds") or 900
        )
        self._bind_session_to_ssh: bool = bool(
            self.control_settings.get("engine_host_bind_session_to_ssh", True)
        )
        # Connection management
        self._connection: Optional[Any] = None  # BaseConnection instance
        self._connection_lock = threading.Lock()
        self._auto_bootstrap_daemon: bool = bool(
            self.control_settings.get("engine_host_daemon_auto_bootstrap", True)
        )
        self._daemon_port_override: int = int(
            self.control_settings.get("engine_host_daemon_port") or 0
        )
        self._daemon_log_file: Optional[str] = str(
            self.control_settings.get("engine_host_daemon_log_file") or ""
        ).strip() or None
        self._last_daemon_status_fingerprint: Optional[Dict[str, Any]] = None
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

    def _local_control_state_path(self) -> Optional[Path]:
        raw = str(self._control_state_file or "").strip()
        if not raw:
            return None
        return Path(raw).expanduser().resolve()

    def _read_local_control_snapshot(self) -> Optional[Dict[str, Any]]:
        if str(self.get_target().get("mode") or "local") != "local":
            return None
        try:
            from .engine_host_service import EngineHostService

            svc = EngineHostService(control_state_file=self._local_control_state_path())
            return dict(svc.get_control_config() or {})
        except Exception as exc:
            logger.debug("Failed to read local hosting control snapshot: %s", exc)
            return None

    @staticmethod
    def _compose_unconfigured_hosting_warning(snapshot: Optional[Dict[str, Any]]) -> Optional[str]:
        cfg = dict(snapshot or {})
        if int(cfg.get("keys_count") or 0) != 0:
            return None
        if bool(cfg.get("require_auth", True)):
            return None
        if str(cfg.get("endpoint_mode_default") or "").strip().lower() != "exclusive":
            return None
        return (
            "Hosting access is not configured yet. The local daemon is running in temporary "
            "local-only no-auth exclusive mode. Configure hosting_access as soon as possible."
        )

    @staticmethod
    def _daemon_status_fingerprint(status: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pid_file_present": bool(status.get("pid") or status.get("port") or status.get("started_at")),
            "pid": status.get("pid"),
            "port": status.get("port"),
            "started_at": status.get("started_at"),
            "pid_alive": bool(status.get("pid_alive", False)),
            "reachable": bool(status.get("reachable", False)),
        }

    def _daemon_status_event(self, status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = self._daemon_status_fingerprint(status)
        previous = self._last_daemon_status_fingerprint
        self._last_daemon_status_fingerprint = dict(current)
        if previous is None or previous == current:
            return None
        reason = "daemon_status_changed"
        if bool(previous.get("pid_file_present")) and not bool(current.get("pid_file_present")):
            reason = "pid_file_removed"
        elif not bool(previous.get("pid_file_present")) and bool(current.get("pid_file_present")):
            reason = "pid_file_created"
        elif (
            previous.get("pid") != current.get("pid")
            or previous.get("port") != current.get("port")
            or previous.get("started_at") != current.get("started_at")
        ):
            reason = "pid_file_updated"
        elif previous.get("reachable") != current.get("reachable"):
            reason = "reachability_changed"
        return {
            "event": "daemon_status_changed",
            "reason": reason,
            "previous": dict(previous),
            "current": dict(current),
        }

    def _finalize_daemon_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = self._read_local_control_snapshot()
        warning = self._compose_unconfigured_hosting_warning(snapshot)
        status["control_config"] = snapshot
        status["require_auth"] = (
            bool((status.get("auth_status") or {}).get("require_auth"))
            if isinstance(status.get("auth_status"), dict) and "require_auth" in dict(status.get("auth_status") or {})
            else (bool(snapshot.get("require_auth")) if isinstance(snapshot, dict) else None)
        )
        status["keys_count"] = (
            int((status.get("auth_status") or {}).get("keys_count") or 0)
            if isinstance(status.get("auth_status"), dict) and "keys_count" in dict(status.get("auth_status") or {})
            else (int(snapshot.get("keys_count") or 0) if isinstance(snapshot, dict) else None)
        )
        status["endpoint_mode_default"] = (
            str(snapshot.get("endpoint_mode_default") or "").strip().lower() if isinstance(snapshot, dict) else None
        ) or None
        status["warnings"] = [warning] if warning else []
        status["status_event"] = self._daemon_status_event(status)
        return status

    def _prepare_local_unconfigured_bootstrap(self) -> Optional[Dict[str, Any]]:
        if str(self.get_target().get("mode") or "local") != "local":
            return None
        snapshot = self._read_local_control_snapshot()
        if not isinstance(snapshot, dict):
            return None
        if int(snapshot.get("keys_count") or 0) != 0:
            return snapshot
        try:
            from .engine_host_service import EngineHostService

            svc = EngineHostService(control_state_file=self._local_control_state_path())
            updated = svc.set_control_config(
                require_auth=False,
                access_profile={"connectivity_mode": "local_only"},
                endpoint_mode_default="exclusive",
            )
            logger.warning(
                "Local hosting was unconfigured (keys_count=0); forcing temporary local-only "
                "no-auth exclusive bootstrap. Configure hosting_access as soon as possible."
            )
            return dict(updated or {})
        except Exception as exc:
            logger.warning("Failed to prepare local unconfigured hosting bootstrap defaults: %s", exc)
            return snapshot

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
                        known_hosts_line=str(self.control_settings.get("ssh_known_hosts_line") or "").strip() or None,
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
                    conn = LocalSocketConnection(
                        port=port,
                        pid_file=Path(pid_file_path) if pid_file_path else None,
                        timeout=self._timeout,
                    )
                    if conn.is_alive():
                        self._connection = conn
                        return conn

                # Daemon not running: auto-bootstrap if enabled
                if self._auto_bootstrap_daemon:
                    try:
                        _ = self._prepare_local_unconfigured_bootstrap()
                        wait = float(
                            self.control_settings.get("engine_host_daemon_wait_ready_seconds") or 8.0
                        )
                        result = start_daemon_background(
                            port=self._daemon_port_override or DEFAULT_DAEMON_PORT,
                            pid_file=Path(pid_file_path) if pid_file_path else None,
                            log_file=Path(self._daemon_log_file) if self._daemon_log_file else None,
                            wait_ready_seconds=wait,
                        )
                        new_port = int(result.get("port") or DEFAULT_DAEMON_PORT)
                        conn = LocalSocketConnection(
                            port=new_port,
                            pid_file=Path(pid_file_path) if pid_file_path else None,
                            timeout=self._timeout,
                        )
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

    def _invoke(
        self,
        command: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        allow_auto_session: bool = True,
        _retry_on_auth_error: bool = True,
    ) -> Any:
        """
        Send a command and return the result.

        Tries persistent connection first; falls back to per-command subprocess.
        On auth/session errors, clears the session token and retries once so that
        an auto-issued fresh session can succeed without caller intervention.
        """
        if (
            allow_auto_session
            and not self._session_token
            and command != "auth-issue-session"
            and self._key_id
            and self._key_secret
        ):
            try:
                target_mode = str(self.get_target().get("mode") or "").strip().lower()
                if target_mode == "ssh":
                    # Shared-secret auto-bootstrap is local-only by policy; SSH-targeted
                    # channels must use a pre-issued token or explicit challenge flow.
                    raise RuntimeError("auto_shared_secret_bootstrap_not_supported_for_ssh_target")
                ssh_binding = self._current_ssh_session_binding()
                issued = self._invoke(
                    "auth-issue-session",
                    {
                        "key_id": self._key_id,
                        "key_secret": self._key_secret,
                        "scope": self._session_scope,
                        "ttl_seconds": self._session_ttl_seconds,
                        "ssh_binding": ssh_binding if ssh_binding else None,
                    },
                    allow_auto_session=False,
                )
                token = str((issued or {}).get("token") or "").strip()
                if token:
                    self.set_session_token(token)
            except Exception as exc:
                logger.debug("Auto session issuance failed: %s", exc)

        effective_payload = dict(payload or {})
        ssh_binding = self._current_ssh_session_binding()
        if command == "auth-issue-session":
            if ssh_binding and not effective_payload.get("ssh_binding"):
                effective_payload["ssh_binding"] = ssh_binding
        elif ssh_binding:
            effective_payload.setdefault("_ssh_session_binding", ssh_binding)
        if self._session_token and command not in {"auth-issue-session"}:
            effective_payload.setdefault("session_token", self._session_token)

        conn = self._get_connection()
        if conn is not None:
            try:
                return conn.invoke(command, effective_payload)
            except Exception as exc:
                logger.warning(
                    "Persistent connection failed for '%s': %s. Falling back to subprocess.",
                    command,
                    exc,
                )
                with self._connection_lock:
                    self._connection = None
        try:
            return self._invoke_subprocess(command, effective_payload)
        except RuntimeError as exc:
            # Retry once on session/auth errors: clear stale token so auto-issue fires again.
            _no_retry_cmds = {"auth-issue-session", "auth-status", "auth-begin-challenge"}
            if (
                _retry_on_auth_error
                and self._session_token
                and command not in _no_retry_cmds
                and _is_session_auth_error(str(exc))
            ):
                logger.info(
                    "Auth error on '%s' (likely expired session); clearing token and retrying: %s",
                    command,
                    exc,
                )
                self._session_token = None
                self.control_settings["engine_host_session_token"] = None
                return self._invoke(command, payload, allow_auto_session=True, _retry_on_auth_error=False)
            raise

    def set_session_token(self, token: Optional[str]) -> None:
        self._session_token = str(token or "").strip() or None
        self.control_settings["engine_host_session_token"] = self._session_token

    def get_session_token(self) -> Optional[str]:
        return self._session_token

    def _current_ssh_session_binding(self) -> Optional[Dict[str, str]]:
        if not self._bind_session_to_ssh:
            return None
        target = self.get_target()
        if str(target.get("mode") or "") != "ssh":
            return None
        raw_target = str(target.get("target") or "").strip()
        if raw_target.startswith("ssh://"):
            raw_target = raw_target[6:]
        if not raw_target:
            return None
        fp = str(self.control_settings.get("control_ssh_fingerprint") or "").strip()
        binding: Dict[str, str] = {"target": raw_target}
        if fp:
            binding["key_fingerprint"] = fp
        return binding

    # ------------------------------------------------------------------
    # Daemon lifecycle management (new public API)
    # ------------------------------------------------------------------

    def get_daemon_status(self) -> Dict[str, Any]:
        """Return local daemon PID status plus auth-status snapshot when reachable."""
        from .engine_host_daemon import DaemonPidFile
        from .engine_host_connection import LocalSocketConnection
        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        info = pid_info.read() or {}
        pid_alive = bool(pid_info.is_alive())
        status: Dict[str, Any] = {
            "pid_file": str(pid_info.path),
            "pid": info.get("pid"),
            "port": info.get("port"),
            "started_at": info.get("started_at"),
            "pid_alive": pid_alive,
            "reachable": False,
            "reachability_error": None,
            "alive": False,
            "auth_status": None,
            "auth_status_error": None,
        }
        if not pid_alive:
            return self._finalize_daemon_status(status)
        try:
            port = int(info.get("port") or 0)
            if port <= 0:
                status["reachability_error"] = "missing_daemon_port"
                return self._finalize_daemon_status(status)
            conn = LocalSocketConnection(
                port=port,
                pid_file=Path(pid_file_path) if pid_file_path else None,
                timeout=min(self._timeout, 5.0),
                max_reconnect_attempts=1,
            )
            pong = conn.invoke("__ping__", {})
            status["reachable"] = str(pong or "") == "pong"
            status["alive"] = bool(status["reachable"])
            if not status["reachable"]:
                status["reachability_error"] = "daemon_ping_failed"
                conn.close()
                return self._finalize_daemon_status(status)
            payload: Dict[str, Any] = {}
            if self._session_token:
                payload["session_token"] = self._session_token
            auth = conn.invoke("auth-status", payload)
            conn.close()
            status["auth_status"] = dict(auth or {}) if isinstance(auth, dict) else None
        except Exception as exc:
            status["alive"] = False
            status["reachable"] = False
            status["reachability_error"] = str(exc)
            status["auth_status_error"] = str(exc)
        return self._finalize_daemon_status(status)

    def bootstrap_daemon(self, *, wait_ready_seconds: float = 8.0) -> Dict[str, Any]:
        """Start local daemon if not already running. Returns daemon status dict."""
        from .engine_host_daemon import DaemonPidFile, start_daemon_background, DEFAULT_DAEMON_PORT
        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        if pid_info.is_alive():
            return {"already_running": True, **self.get_daemon_status()}
        bootstrap_cfg = self._prepare_local_unconfigured_bootstrap()
        result = start_daemon_background(
            port=self._daemon_port_override or DEFAULT_DAEMON_PORT,
            pid_file=Path(pid_file_path) if pid_file_path else None,
            log_file=Path(self._daemon_log_file) if self._daemon_log_file else None,
            wait_ready_seconds=wait_ready_seconds,
        )
        with self._connection_lock:
            self._connection = None  # Force reconnect on next invoke
        return {
            "already_running": False,
            "bootstrap_control_config": dict(bootstrap_cfg or {}) if isinstance(bootstrap_cfg, dict) else None,
            **result,
            **self.get_daemon_status(),
        }

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
            conn = LocalSocketConnection(
                port=port,
                pid_file=Path(pid_file_path) if pid_file_path else None,
                timeout=5.0,
                max_reconnect_attempts=1,
            )
            conn.invoke("__shutdown__", {"shutdown_token": token})
            conn.close()
            with self._connection_lock:
                self._connection = None
            return {"status": "shutdown_sent"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def reset_hosting_access(self) -> Dict[str, Any]:
        """
        Local-only helper: stop local daemon and clear auth state from control config.

        This helper intentionally does not go through daemon RPC/auth surfaces.
        """
        if str(self.get_target().get("mode") or "local") != "local":
            raise ValueError("reset_hosting_access is only valid in local mode")
        from .engine_host_daemon import DaemonPidFile
        from .engine_host_service import EngineHostService

        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        daemon_info = dict(pid_info.read() or {})
        stop_result = self.stop_daemon()
        pid = int(daemon_info.get("pid") or 0)
        if pid > 0 and bool(pid_info.is_alive()):
            try:
                os.kill(pid, getattr(signal, "SIGTERM", 15))
            except Exception as exc:
                stop_result = {"status": "error", "error": str(exc), "forced_kill_attempted": True}
            else:
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    if not pid_info.is_alive():
                        break
                    time.sleep(0.1)
                stop_result = {
                    "status": "terminated" if not pid_info.is_alive() else "terminate_timeout",
                    "forced_kill_attempted": True,
                    "pid": pid,
                }
        if not pid_info.is_alive():
            pid_info.remove()
        with self._connection_lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None
        svc = EngineHostService(control_state_file=self._local_control_state_path())
        reset = svc.reset_hosting_access()
        return {
            "status": "ok",
            "local_helper_only": True,
            "rpc_accessible": False,
            "daemon_stop": stop_result,
            "auth_reset": dict(reset or {}),
            "daemon_status": self.get_daemon_status(),
        }

    def close_connection(self) -> None:
        """Close and discard the current persistent connection."""
        with self._connection_lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None

    def restart_remote_daemon(self, *, wait_seconds: float = 3.0) -> Dict[str, Any]:
        """SSH-exec the remote daemon start command and wait for it to bind.

        Only valid in SSH mode.  Raises ValueError if called on a local channel.
        Returns {"started": bool, "error": Optional[str]}.
        """
        import time as _time

        target = self.get_target()
        if str(target.get("mode") or "local") != "ssh":
            raise ValueError("restart_remote_daemon is only valid in SSH mode")
        ssh_target = str(target.get("target") or "").strip()
        if not ssh_target:
            raise ValueError("SSH target is not set")
        ssh_key = str(self.control_settings.get("control_ssh_key") or "").strip() or None
        known_hosts_line = str(self.control_settings.get("ssh_known_hosts_line") or "").strip() or None

        # Build the daemon-start command (use base CLI without --relay)
        raw_remote = str(
            self.control_settings.get("engine_host_remote_cmd") or ""
        ).strip()
        if not raw_remote or "--relay" in raw_remote:
            raw_remote = "python -m hosting.engine_host_cli"
        daemon_cmd = f"{raw_remote} --daemon --background"

        argv: List[str] = [
            "ssh", "-T",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={max(5, int(self._timeout))}",
        ]
        if known_hosts_line:
            import tempfile as _tempfile, os as _os
            try:
                fd, tmppath = _tempfile.mkstemp(prefix="mp13_kh_", suffix=".txt")
                with _os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(known_hosts_line + "\n")
                argv += ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={tmppath}"]
            except Exception:
                argv += ["-o", "StrictHostKeyChecking=accept-new"]
        else:
            argv += ["-o", "StrictHostKeyChecking=accept-new"]
        if ssh_key:
            argv += ["-i", ssh_key]
        argv.append(ssh_target)
        argv += shlex.split(daemon_cmd)

        try:
            proc = subprocess.run(  # noqa: S603
                argv,
                capture_output=True,
                timeout=float(wait_seconds) + 10.0,
                check=False,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
                logger.warning("restart_remote_daemon SSH exec returned %d: %s", proc.returncode, stderr)
                return {"started": False, "error": f"exit {proc.returncode}: {stderr}"}
        except Exception as exc:
            logger.warning("restart_remote_daemon failed: %s", exc)
            return {"started": False, "error": str(exc)}

        _time.sleep(float(wait_seconds))
        return {"started": True, "error": None}

    def discover_running(self) -> List[Dict[str, Any]]:
        res = self._invoke("discover-running", {})
        return list(res or []) if isinstance(res, list) else []

    def discover_running_progress(
        self,
        *,
        include_reachability: bool = True,
        reachability_timeout_seconds: float = 0.35,
        prune_stale: bool = True,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "discover-running",
            {
                "include_progress": True,
                "include_reachability": bool(include_reachability),
                "reachability_timeout_seconds": float(reachability_timeout_seconds or 0.35),
                "prune_stale": bool(prune_stale),
            },
        )
        return dict(res or {}) if isinstance(res, dict) else {}

    def get_registration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        res = self._invoke("get-registration", {"engine_id": str(engine_id)})
        return dict(res or {}) if isinstance(res, dict) else None

    def spawn_process(self, *, engine_id: str, command: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        res = self._invoke(
            "spawn",
            {
                "engine_id": str(engine_id),
                "command": [str(x) for x in list(command or [])],
                "cwd": str(cwd) if cwd else None,
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

    def claim_engine(
        self,
        engine_id: str,
        *,
        backend_id: Optional[str],
        exclusive: Optional[bool] = None,
        force_override: bool = False,
        force_override_confirmation: Optional[str] = None,
        force_override_reason: Optional[str] = None,
        force_override_emergency: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "engine_id": str(engine_id),
            "backend_id": backend_id,
            "force_override": bool(force_override),
            "force_override_confirmation": force_override_confirmation,
            "force_override_reason": force_override_reason,
            "force_override_emergency": bool(force_override_emergency),
        }
        if exclusive is not None:
            payload["exclusive"] = bool(exclusive)
        res = self._invoke(
            "claim-engine",
            payload,
        )
        return dict(res or {})

    def claim_endpoint(
        self,
        *,
        backend_id: Optional[str],
        exclusive: Optional[bool] = None,
        force_override: bool = False,
        force_override_confirmation: Optional[str] = None,
        force_override_reason: Optional[str] = None,
        force_override_emergency: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "backend_id": backend_id,
            "force_override": bool(force_override),
            "force_override_confirmation": force_override_confirmation,
            "force_override_reason": force_override_reason,
            "force_override_emergency": bool(force_override_emergency),
        }
        if exclusive is not None:
            payload["exclusive"] = bool(exclusive)
        res = self._invoke(
            "claim-endpoint",
            payload,
        )
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

    def claim_resource(
        self,
        resource_kind: str,
        resource_id: str,
        *,
        backend_id: Optional[str],
        exclusive: Optional[bool] = None,
        force_override: bool = False,
        force_override_confirmation: Optional[str] = None,
        force_override_reason: Optional[str] = None,
        force_override_emergency: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "resource_kind": str(resource_kind),
            "resource_id": str(resource_id),
            "backend_id": backend_id,
            "force_override": bool(force_override),
            "force_override_confirmation": force_override_confirmation,
            "force_override_reason": force_override_reason,
            "force_override_emergency": bool(force_override_emergency),
        }
        if exclusive is not None:
            payload["exclusive"] = bool(exclusive)
        res = self._invoke(
            "claim-resource",
            payload,
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

    def start_host_operation(
        self,
        *,
        command: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "op-start",
            {
                "command": str(command or "").strip(),
                "payload": dict(payload or {}),
            },
        )
        return dict(res or {}) if isinstance(res, dict) else {}

    def get_host_operation_status(self, *, operation_id: str) -> Dict[str, Any]:
        res = self._invoke(
            "op-status",
            {
                "operation_id": str(operation_id or "").strip(),
            },
        )
        return dict(res or {}) if isinstance(res, dict) else {}

    def start_connect_from_config(
        self,
        *,
        config_path: str,
        engine_id: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.start_host_operation(
            command="connect-from-config",
            payload={
                "config_path": str(config_path or "default"),
                "engine_id": str(engine_id).strip() if engine_id else None,
                "model_path": str(model_path).strip() if model_path else None,
            },
        )

    def inspect_engine_capabilities(self, *, engine_id: str, endpoint: str = "") -> Dict[str, Any]:
        res = self._invoke(
            "inspect-capabilities",
            {
                "engine_id": str(engine_id or "").strip(),
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

    def proxy_request(
        self,
        *,
        engine_id: str,
        method: str = "GET",
        path: str = "/",
        query: str = "",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-request",
            {
                "engine_id": str(engine_id or "").strip(),
                "method": str(method or "GET"),
                "path": str(path or "/"),
                "query": str(query or ""),
                "headers": dict(headers or {}),
                "body_b64": str(body_b64 or ""),
                "timeout_seconds": float(timeout_seconds or 30.0),
                "max_response_bytes": int(max_response_bytes or 1024 * 1024),
            },
        )
        return dict(res or {})

    def proxy_rpc_call(
        self,
        *,
        engine_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-rpc-call",
            {
                "engine_id": str(engine_id or "").strip(),
                "method": str(method or ""),
                "params": dict(params or {}),
                "timeout_seconds": float(timeout_seconds or 30.0),
            },
        )
        return dict(res or {})

    def proxy_rpc_open(
        self,
        *,
        engine_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: str,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-rpc-open",
            {
                "engine_id": str(engine_id or "").strip(),
                "method": str(method or ""),
                "params": dict(params or {}),
                "request_id": str(request_id or "").strip(),
                "timeout_seconds": float(timeout_seconds or 30.0),
            },
        )
        return dict(res or {})

    def proxy_rpc_send(
        self,
        *,
        engine_id: str,
        stream_id: str,
        message: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-rpc-send",
            {
                "engine_id": str(engine_id or "").strip(),
                "stream_id": str(stream_id or "").strip(),
                "message": dict(message or {}),
                "timeout_seconds": float(timeout_seconds or 30.0),
            },
        )
        return dict(res or {})

    def proxy_rpc_recv(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 2.0,
        max_items: int = 64,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-rpc-recv",
            {
                "engine_id": str(engine_id or "").strip(),
                "stream_id": str(stream_id or "").strip(),
                "timeout_seconds": float(timeout_seconds or 2.0),
                "max_items": int(max_items or 64),
            },
        )
        return dict(res or {})

    def proxy_rpc_close(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-rpc-close",
            {
                "engine_id": str(engine_id or "").strip(),
                "stream_id": str(stream_id or "").strip(),
                "timeout_seconds": float(timeout_seconds or 10.0),
            },
        )
        return dict(res or {})

    def proxy_stream_open(
        self,
        *,
        engine_id: str,
        tool: str = "run-inference",
        arguments: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-stream-open",
            {
                "engine_id": str(engine_id or "").strip(),
                "tool": str(tool or "run-inference"),
                "arguments": dict(arguments or {}),
                "timeout_seconds": float(timeout_seconds or 30.0),
            },
        )
        return dict(res or {})

    def proxy_stream_send(
        self,
        *,
        engine_id: str,
        stream_id: str,
        message: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-stream-send",
            {
                "engine_id": str(engine_id or "").strip(),
                "stream_id": str(stream_id or "").strip(),
                "message": dict(message or {}),
                "timeout_seconds": float(timeout_seconds or 30.0),
            },
        )
        return dict(res or {})

    def proxy_stream_recv(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 2.0,
        max_items: int = 64,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-stream-recv",
            {
                "engine_id": str(engine_id or "").strip(),
                "stream_id": str(stream_id or "").strip(),
                "timeout_seconds": float(timeout_seconds or 2.0),
                "max_items": int(max_items or 64),
            },
        )
        return dict(res or {})

    def proxy_stream_close(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "proxy-stream-close",
            {
                "engine_id": str(engine_id or "").strip(),
                "stream_id": str(stream_id or "").strip(),
                "timeout_seconds": float(timeout_seconds or 10.0),
            },
        )
        return dict(res or {})

    def get_control_config(self) -> Dict[str, Any]:
        res = self._invoke("get-control-config", {})
        return dict(res or {})

    def get_endpoint_mode_effective(self) -> Dict[str, Any]:
        res = self._invoke("get-endpoint-mode-effective", {})
        return dict(res or {})

    def set_endpoint_mode_override(self, mode: Optional[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if mode is not None:
            payload["mode"] = str(mode).strip().lower()
        res = self._invoke("set-endpoint-mode-override", payload)
        return dict(res or {})

    def set_control_config(
        self,
        *,
        ssh_key: Optional[str] = None,
        require_auth: Optional[bool] = None,
        access_profile: Optional[Dict[str, Any]] = None,
        endpoint_mode_default: Optional[str] = None,
        lifecycle_profile: Optional[str] = None,
        lifecycle_policy: Optional[Dict[str, Any]] = None,
        traffic_policy: Optional[Dict[str, Any]] = None,
        engine_traffic_policies: Optional[Dict[str, Dict[str, Any]]] = None,
        claim_acl_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"ssh_key": str(ssh_key).strip() if ssh_key else None}
        if require_auth is not None:
            payload["require_auth"] = bool(require_auth)
        if access_profile is not None:
            payload["access_profile"] = dict(access_profile or {})
        if endpoint_mode_default is not None:
            payload["endpoint_mode_default"] = str(endpoint_mode_default).strip().lower()
        if lifecycle_profile is not None:
            payload["lifecycle_profile"] = str(lifecycle_profile).strip().lower()
        if lifecycle_policy is not None:
            payload["lifecycle_policy"] = dict(lifecycle_policy or {})
        if traffic_policy is not None:
            payload["traffic_policy"] = dict(traffic_policy or {})
        if engine_traffic_policies is not None:
            payload["engine_traffic_policies"] = {
                str(k): dict(v or {}) for k, v in dict(engine_traffic_policies or {}).items()
            }
        if claim_acl_policy is not None:
            payload["claim_acl_policy"] = dict(claim_acl_policy or {})
        res = self._invoke("set-control-config", payload)
        return dict(res or {})

    def get_lifecycle_policy_effective(self) -> Dict[str, Any]:
        res = self._invoke("get-lifecycle-policy-effective", {})
        return dict(res or {})

    def auth_status(self) -> Dict[str, Any]:
        res = self._invoke("auth-status", {})
        return dict(res or {})

    def get_host_metrics(self) -> Dict[str, Any]:
        res = self._invoke("host-metrics", {})
        return dict(res or {})

    def auth_list_keys(self) -> List[Dict[str, Any]]:
        res = self._invoke("auth-list-keys", {})
        return list(res or []) if isinstance(res, list) else []

    def auth_list_sessions(
        self,
        *,
        key_id: Optional[str] = None,
        scope: Optional[str] = None,
        role: Optional[str] = None,
        token_preview_contains: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "auth-list-sessions",
            {
                "key_id": str(key_id).strip() if key_id else None,
                "scope": str(scope).strip() if scope else None,
                "role": str(role).strip() if role else None,
                "token_preview_contains": str(token_preview_contains).strip() if token_preview_contains else None,
                "limit": int(limit or 100),
                "offset": int(offset or 0),
            },
        )
        return dict(res or {}) if isinstance(res, dict) else {}

    def auth_list_issued_tokens(
        self,
        *,
        engine_id: Optional[str] = None,
        resource_kind: Optional[str] = None,
        resource_id: Optional[str] = None,
        backend_id: Optional[str] = None,
        token_preview_contains: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "auth-list-issued-tokens",
            {
                "engine_id": str(engine_id).strip() if engine_id else None,
                "resource_kind": str(resource_kind).strip() if resource_kind else None,
                "resource_id": str(resource_id).strip() if resource_id else None,
                "backend_id": str(backend_id).strip() if backend_id else None,
                "token_preview_contains": str(token_preview_contains).strip() if token_preview_contains else None,
                "limit": int(limit or 100),
                "offset": int(offset or 0),
            },
        )
        return dict(res or {}) if isinstance(res, dict) else {}

    def auth_upsert_key(
        self,
        *,
        key_id: str,
        key_secret: str = "",
        role: str,
        auth_method: str = "shared_secret",
        public_key: str = "",
        allowed_configs: Optional[List[str]] = None,
        allowed_engines: Optional[List[str]] = None,
        disabled: bool = False,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "auth-upsert-key",
            {
                "key_id": str(key_id or ""),
                "key_secret": str(key_secret or ""),
                "role": str(role or ""),
                "auth_method": str(auth_method or "shared_secret"),
                "public_key": str(public_key or ""),
                "allowed_configs": list(allowed_configs or []),
                "allowed_engines": list(allowed_engines or []),
                "disabled": bool(disabled),
            },
        )
        return dict(res or {})

    def auth_list_audit_events(
        self,
        *,
        event_type: Optional[str] = None,
        actor_key_id: Optional[str] = None,
        target_key_id: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "auth-audit-list",
            {
                "event_type": str(event_type) if event_type else None,
                "actor_key_id": str(actor_key_id) if actor_key_id else None,
                "target_key_id": str(target_key_id) if target_key_id else None,
                "result": str(result) if result else None,
                "limit": int(limit),
                "offset": int(offset),
            },
        )
        return dict(res or {})

    def auth_revoke_key(self, key_id: str) -> Dict[str, Any]:
        res = self._invoke("auth-revoke-key", {"key_id": str(key_id or "")})
        return dict(res or {})

    def auth_issue_session(
        self,
        *,
        key_id: str,
        key_secret: str,
        scope: str = "control",
        ttl_seconds: int = 900,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        bind_to_ssh: bool = True,
        adopt: bool = True,
    ) -> Dict[str, Any]:
        binding = self._current_ssh_session_binding() if bind_to_ssh else None
        res = self._invoke(
            "auth-issue-session",
            {
                "key_id": str(key_id or ""),
                "key_secret": str(key_secret or ""),
                "scope": str(scope or "control"),
                "ttl_seconds": int(ttl_seconds or 900),
                "config_paths": list(config_paths or []),
                "engine_ids": list(engine_ids or []),
                "ssh_binding": dict(binding or {}),
            },
        )
        out = dict(res or {})
        token = str(out.get("token") or "").strip()
        if adopt and token:
            self.set_session_token(token)
        return out

    def auth_begin_challenge(
        self,
        *,
        key_id: str,
        scope: str = "control",
        ttl_seconds: int = 120,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        bind_to_ssh: bool = True,
    ) -> Dict[str, Any]:
        binding = self._current_ssh_session_binding() if bind_to_ssh else None
        res = self._invoke(
            "auth-begin-challenge",
            {
                "key_id": str(key_id or ""),
                "scope": str(scope or "control"),
                "ttl_seconds": int(ttl_seconds or 120),
                "config_paths": list(config_paths or []),
                "engine_ids": list(engine_ids or []),
                "ssh_binding": dict(binding or {}),
            },
        )
        return dict(res or {})

    def auth_complete_challenge(
        self,
        *,
        challenge_id: str,
        signature_ssh: str,
        adopt: bool = True,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "auth-complete-challenge",
            {
                "challenge_id": str(challenge_id or ""),
                "signature_ssh": str(signature_ssh or ""),
            },
        )
        out = dict(res or {})
        token = str(out.get("token") or "").strip()
        if adopt and token:
            self.set_session_token(token)
        return out

    def auth_revoke_session(self, token: str) -> Dict[str, Any]:
        res = self._invoke("auth-revoke-session", {"token": str(token or "")})
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
