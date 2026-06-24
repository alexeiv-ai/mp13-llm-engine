"""
Backend-side adapter to interact with engine host.

Primary path: persistent connection to a running EngineHostDaemon.
  - Local mode (no SSH): LocalSocketConnection via local IPC discovered from PID file
  - SSH mode:            SSHRelayConnection via SSH subprocess running --relay-wrapper

Per-command CLI fallback is intentionally restricted to explicit diagnostic
commands. Runtime, config, auth, claim, model, toolbox, and proxy commands must
go through the persistent control channel.

The entire existing public API is preserved; no callers need changes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ._process_utils import hidden_subprocess_kwargs, pid_alive
from .client_realm import resolve_client_profile_control_settings
from .engine_host_connection import CommandError


def _resolved_pid_path(pid_info: Any, pid_file_path: Any) -> Optional[Path]:
    if pid_file_path:
        return Path(pid_file_path)
    path = getattr(pid_info, "path", None)
    return Path(path) if path is not None else None

# Keywords that indicate an expired or invalid session token in daemon error strings.
_SESSION_AUTH_ERROR_KEYWORDS = (
    "session_expired",
    "invalid_session",
    "missing_or_invalid_session_token",
    "session_token_required",
    "session_not_found",
    "invalid_token",
    "expired_token",
)


_SUBPROCESS_FALLBACK_COMMANDS = frozenset(
    {
        # Diagnostic-only command. Lifecycle start/stop/restart use dedicated
        # channel methods rather than the generic per-command CLI path.
        "discover-running",
    }
)

_AUTO_SESSION_CACHE_LOCK = threading.Lock()
_AUTO_SESSION_CACHE: Dict[str, Dict[str, Any]] = {}


def _is_session_auth_error(msg: str) -> bool:
    ml = str(msg or "").lower()
    return any(k in ml for k in _SESSION_AUTH_ERROR_KEYWORDS)


def _exception_is_session_auth_error(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "") or "").strip()
    if code and _is_session_auth_error(code):
        return True
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        reason = str(details.get("reason") or "").strip()
        if reason and _is_session_auth_error(reason):
            return True
    return _is_session_auth_error(str(exc))


def _command_error_message(command: str, exc: Exception) -> str:
    code = str(getattr(exc, "code", "") or "").strip()
    details = getattr(exc, "details", None)
    suffix = f" ({code})" if code and code not in str(exc) else ""
    details_suffix = f" details={details}" if isinstance(details, dict) and details else ""
    return f"persistent daemon control channel failed for '{command}': {exc}{suffix}{details_suffix}"


def _clear_auto_session_cache_for_tests() -> None:
    with _AUTO_SESSION_CACHE_LOCK:
        _AUTO_SESSION_CACHE.clear()


class EngineHostControlChannel:
    """Command-channel wrapper that requires the persistent daemon control path."""

    def __init__(self, control_settings: Optional[Dict[str, Any]] = None):
        self.control_settings: Dict[str, Any] = resolve_client_profile_control_settings(control_settings)
        self._base_cmd: List[str] = []
        self._engines_state_file = self.control_settings.get("engine_host_state_file")
        self._control_state_file = self.control_settings.get("engine_host_control_state_file")
        self._timeout = float(self.control_settings.get("engine_host_timeout_seconds") or 15.0)
        self._session_token: Optional[str] = str(
            self.control_settings.get("engine_host_session_token") or ""
        ).strip() or None
        self._session_token_meta: Dict[str, Any] = {}
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
        if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", raw):
            return Path(raw)
        return Path(raw).expanduser().resolve()

    def _read_local_control_snapshot(self) -> Optional[Dict[str, Any]]:
        if str(self.get_target().get("mode") or "local") != "local":
            return None
        try:
            from .service.host_service import EngineHostService

            svc = EngineHostService(control_state_file=self._local_control_state_path())
            return dict(svc.get_control_config() or {})
        except Exception as exc:
            logger.debug("Failed to read local hosting control snapshot: %s", exc)
            return None

    def _configured_access_root_conflict(self) -> Optional[Dict[str, Any]]:
        selected = self._local_control_state_path()
        if selected is None:
            return None
        try:
            selected_resolved = selected.expanduser().resolve()
        except Exception:
            selected_resolved = selected

        candidates: List[Path] = []
        try:
            from .service.constants import DEFAULT_CONTROL_STATE_FILE

            candidates.append(DEFAULT_CONTROL_STATE_FILE.expanduser().resolve())
        except Exception:
            pass
        try:
            if selected_resolved.name != "access_control.json":
                candidates.append((selected_resolved.parent.parent / "hosting" / "access_control.json").resolve())
        except Exception:
            pass

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate == selected_resolved or not candidate.exists():
                continue
            try:
                from .service.host_service import EngineHostService

                cfg = EngineHostService(control_state_file=candidate).get_control_config()
            except Exception:
                continue
            if int(dict(cfg or {}).get("keys_count") or 0) <= 0:
                continue
            return {
                "selected_control_state_file": str(selected_resolved),
                "configured_control_state_file": str(candidate),
                "configured_require_auth": bool(dict(cfg or {}).get("require_auth", False)),
                "configured_keys_count": int(dict(cfg or {}).get("keys_count") or 0),
                "configured_endpoint_mode_default": str(dict(cfg or {}).get("endpoint_mode_default") or ""),
            }
        return None

    def _should_auto_recover_unreachable_local_daemon(self) -> Dict[str, Any]:
        snapshot = dict(self._read_local_control_snapshot() or {})
        endpoint_mode = str(snapshot.get("endpoint_mode_default") or "").strip().lower()
        lifecycle_profile = str(snapshot.get("lifecycle_profile") or "").strip().lower()
        auto_recover = endpoint_mode == "exclusive" or lifecycle_profile == "foreground_terminal_bound"
        return {
            "auto_recover": bool(auto_recover),
            "endpoint_mode_default": endpoint_mode or None,
            "lifecycle_profile": lifecycle_profile or None,
            "reason": (
                "exclusive_or_foreground_daemon"
                if auto_recover
                else "shared_or_detached_daemon_requires_explicit_force"
            ),
        }

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
        conflict = self._configured_access_root_conflict()
        if conflict:
            raise RuntimeError(
                "Refusing temporary no-auth local daemon bootstrap because configured hosting access "
                f"already exists at {conflict['configured_control_state_file']} "
                f"(selected control state: {conflict['selected_control_state_file']})"
            )
        try:
            from .service.host_service import EngineHostService

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
                    # Remote command defaults to the forced-command wrapper for SSH mode
                    raw_remote = str(
                        self.control_settings.get("engine_host_remote_cmd") or ""
                    ).strip()
                    if not raw_remote or "--relay" not in raw_remote:
                        raw_remote = "python -m hosting.engine_host_cli --relay-wrapper"
                    self._connection = SSHRelayConnection(
                        ssh_target=ssh_target,
                        ssh_key=str(self.control_settings.get("control_ssh_key") or "").strip() or None,
                        remote_cmd=raw_remote,
                        timeout=self._timeout,
                        known_hosts_line=str(self.control_settings.get("ssh_known_hosts_line") or "").strip() or None,
                    )
                return self._connection

            if mode == "local":
                from .daemon import DaemonPidFile, start_daemon_background, DEFAULT_DAEMON_PORT

                pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
                pid_info = DaemonPidFile(pid_file_path)
                pid_path = _resolved_pid_path(pid_info, pid_file_path)

                # If we already have a connection, check it first
                if self._connection is not None and isinstance(self._connection, LocalSocketConnection):
                    if self._connection.is_alive():
                        return self._connection
                    self._connection = None

                port = self._daemon_port_override or pid_info.get_port()
                if port and pid_info.is_alive():
                    conn = LocalSocketConnection(
                        port=port,
                        pid_file=pid_path,
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
                            pid_file=pid_path,
                            log_file=Path(self._daemon_log_file) if self._daemon_log_file else None,
                            engines_state_file=Path(self._engines_state_file) if self._engines_state_file else None,
                            control_state_file=Path(self._control_state_file) if self._control_state_file else None,
                            wait_ready_seconds=wait,
                        )
                        new_port = int(result.get("port") or DEFAULT_DAEMON_PORT)
                        conn = LocalSocketConnection(
                            port=new_port,
                            pid_file=pid_path,
                            timeout=self._timeout,
                        )
                        self._connection = conn
                        return conn
                    except Exception as exc:
                        logger.warning("Auto-bootstrap of local daemon failed: %s", exc)

                return None

            return None

    def _invoke_subprocess(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """Restricted per-command subprocess path for explicit diagnostics only."""
        if str(command or "").strip() not in _SUBPROCESS_FALLBACK_COMMANDS:
            raise RuntimeError(
                f"engine host command '{command}' requires the persistent daemon control channel"
            )
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
            **hidden_subprocess_kwargs(),
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
            raise CommandError(
                msg,
                code=str((out or {}).get("error_code") or "").strip(),
                details=dict((out or {}).get("error_details") or {}),
                result=(out or {}).get("result"),
            )
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

        Tries the persistent connection first. Per-command subprocess fallback
        is restricted to explicit diagnostic commands.
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
            cached = self._get_cached_auto_session()
            if cached:
                self.set_session_token(cached)
            try:
                if not self._session_token:
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
                        self._store_cached_auto_session(token, dict(issued or {}))
            except Exception as exc:
                logger.debug("Auto session issuance failed: %s", exc)

        effective_payload = dict(payload or {})
        ssh_binding = self._current_ssh_session_binding()
        if command in {"auth-issue-session", "auth-begin-challenge"}:
            if ssh_binding and not effective_payload.get("ssh_binding"):
                effective_payload["ssh_binding"] = ssh_binding
        elif ssh_binding:
            effective_payload.setdefault("_ssh_session_binding", ssh_binding)
        if self._session_token and command not in {"auth-issue-session", "auth-begin-challenge", "auth-complete-challenge"}:
            effective_payload.setdefault("session_token", self._session_token)

        conn = self._get_connection()
        if conn is not None:
            try:
                return conn.invoke(command, effective_payload)
            except Exception as exc:
                with self._connection_lock:
                    self._connection = None
                _no_retry_cmds = {"auth-issue-session", "auth-status", "auth-begin-challenge", "auth-complete-challenge"}
                if _exception_is_session_auth_error(exc):
                    if (
                        _retry_on_auth_error
                        and self._session_token
                        and command not in _no_retry_cmds
                    ):
                        logger.info(
                            "Auth error on '%s' (likely expired session); clearing token and retrying: %s",
                            command,
                            exc,
                        )
                        self._session_token = None
                        self.control_settings["engine_host_session_token"] = None
                        self._session_token_meta = {}
                        self._clear_cached_auto_session()
                        return self._invoke(command, payload, allow_auto_session=True, _retry_on_auth_error=False)
                    raise RuntimeError(_command_error_message(command, exc)) from exc
                if str(command or "").strip() not in _SUBPROCESS_FALLBACK_COMMANDS:
                    raise RuntimeError(_command_error_message(command, exc)) from exc
                logger.warning(
                    "Persistent connection failed for diagnostic command '%s': %s. Falling back to subprocess.",
                    command,
                    exc,
                )
        elif str(command or "").strip() not in _SUBPROCESS_FALLBACK_COMMANDS:
            raise RuntimeError(
                f"engine host command '{command}' requires a running persistent daemon control channel"
            )
        try:
            return self._invoke_subprocess(command, effective_payload)
        except RuntimeError as exc:
            # Retry once on session/auth errors: clear stale token so auto-issue fires again.
            _no_retry_cmds = {"auth-issue-session", "auth-status", "auth-begin-challenge", "auth-complete-challenge"}
            if (
                _retry_on_auth_error
                and self._session_token
                and command not in _no_retry_cmds
                and _exception_is_session_auth_error(exc)
            ):
                logger.info(
                    "Auth error on '%s' (likely expired session); clearing token and retrying: %s",
                    command,
                    exc,
                )
                self._session_token = None
                self.control_settings["engine_host_session_token"] = None
                self._session_token_meta = {}
                self._clear_cached_auto_session()
                return self._invoke(command, payload, allow_auto_session=True, _retry_on_auth_error=False)
            raise

    def set_session_token(self, token: Optional[str]) -> None:
        self._session_token = str(token or "").strip() or None
        self.control_settings["engine_host_session_token"] = self._session_token
        if not self._session_token:
            self._session_token_meta = {}

    def get_session_token(self) -> Optional[str]:
        return self._session_token

    def _set_session_token_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        self._session_token_meta = dict(meta or {})

    def _public_key_session_meta_matches(
        self,
        *,
        key_id: str,
        scope: str,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        bind_to_ssh: bool = True,
    ) -> bool:
        meta = dict(self._session_token_meta or {})
        if str(meta.get("auth_method") or "") != "public_key":
            return False
        if str(meta.get("key_id") or "").strip() != str(key_id or "").strip():
            return False
        if str(meta.get("scope") or "").strip().lower() != (str(scope or "control").strip().lower() or "control"):
            return False
        expires_at = float(meta.get("expires_at") or 0.0)
        if expires_at > 0 and time.time() >= expires_at - 5:
            return False
        expected_binding = self._current_ssh_session_binding() if bind_to_ssh else None
        if dict(meta.get("ssh_binding") or {}) != dict(expected_binding or {}):
            return False
        expected_configs = sorted([str(item or "").strip() for item in list(config_paths or []) if str(item or "").strip()])
        expected_engines = sorted([str(item or "").strip() for item in list(engine_ids or []) if str(item or "").strip()])
        return (
            list(meta.get("config_paths") or []) == expected_configs
            and list(meta.get("engine_ids") or []) == expected_engines
        )

    def _auto_session_cache_key(self) -> str:
        binding = self._current_ssh_session_binding() or {}
        payload = {
            "control_state_file": str(self._control_state_file or ""),
            "engine_state_file": str(self._engines_state_file or ""),
            "key_id": str(self._key_id or ""),
            "scope": str(self._session_scope or ""),
            "target": self.get_target(),
            "ssh_binding": binding,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _get_cached_auto_session(self) -> Optional[str]:
        try:
            key = self._auto_session_cache_key()
        except Exception:
            return None
        now = time.time()
        with _AUTO_SESSION_CACHE_LOCK:
            row = dict(_AUTO_SESSION_CACHE.get(key) or {})
            token = str(row.get("token") or "").strip()
            expires_at = float(row.get("expires_at") or 0.0)
            if not token:
                _AUTO_SESSION_CACHE.pop(key, None)
                return None
            if expires_at > 0 and now >= expires_at - 5:
                _AUTO_SESSION_CACHE.pop(key, None)
                return None
            return token

    def _store_cached_auto_session(self, token: str, issued: Dict[str, Any]) -> None:
        tok = str(token or "").strip()
        if not tok:
            return
        try:
            key = self._auto_session_cache_key()
        except Exception:
            return
        expires_at = float(issued.get("expires_at") or 0.0)
        if expires_at <= 0:
            expires_at = time.time() + max(60, int(self._session_ttl_seconds or 900))
        with _AUTO_SESSION_CACHE_LOCK:
            _AUTO_SESSION_CACHE[key] = {"token": tok, "expires_at": expires_at}

    def _clear_cached_auto_session(self) -> None:
        try:
            key = self._auto_session_cache_key()
        except Exception:
            return
        with _AUTO_SESSION_CACHE_LOCK:
            _AUTO_SESSION_CACHE.pop(key, None)

    def _public_key_session_cache_key(
        self,
        *,
        key_id: str,
        scope: str,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        bind_to_ssh: bool = True,
    ) -> str:
        binding = self._current_ssh_session_binding() if bind_to_ssh else None
        payload = {
            "auth_method": "public_key",
            "control_state_file": str(self._control_state_file or ""),
            "engine_state_file": str(self._engines_state_file or ""),
            "key_id": str(key_id or "").strip(),
            "scope": str(scope or "control").strip().lower() or "control",
            "config_paths": sorted([str(item or "").strip() for item in list(config_paths or []) if str(item or "").strip()]),
            "engine_ids": sorted([str(item or "").strip() for item in list(engine_ids or []) if str(item or "").strip()]),
            "target": self.get_target(),
            "ssh_binding": dict(binding or {}),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _get_cached_public_key_session(
        self,
        *,
        key_id: str,
        scope: str,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        bind_to_ssh: bool = True,
    ) -> Optional[str]:
        try:
            key = self._public_key_session_cache_key(
                key_id=key_id,
                scope=scope,
                config_paths=config_paths,
                engine_ids=engine_ids,
                bind_to_ssh=bind_to_ssh,
            )
        except Exception:
            return None
        now = time.time()
        with _AUTO_SESSION_CACHE_LOCK:
            row = dict(_AUTO_SESSION_CACHE.get(key) or {})
            token = str(row.get("token") or "").strip()
            expires_at = float(row.get("expires_at") or 0.0)
            if not token:
                _AUTO_SESSION_CACHE.pop(key, None)
                return None
            if expires_at > 0 and now >= expires_at - 5:
                _AUTO_SESSION_CACHE.pop(key, None)
                return None
            self._set_session_token_meta(row)
            return token

    def _store_cached_public_key_session(
        self,
        token: str,
        issued: Dict[str, Any],
        *,
        key_id: str,
        scope: str,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        bind_to_ssh: bool = True,
    ) -> None:
        tok = str(token or "").strip()
        if not tok:
            return
        try:
            key = self._public_key_session_cache_key(
                key_id=key_id,
                scope=scope,
                config_paths=config_paths,
                engine_ids=engine_ids,
                bind_to_ssh=bind_to_ssh,
            )
        except Exception:
            return
        expires_at = float(issued.get("expires_at") or 0.0)
        if expires_at <= 0:
            expires_at = time.time() + 900
        with _AUTO_SESSION_CACHE_LOCK:
            _AUTO_SESSION_CACHE[key] = {
                "token": tok,
                "expires_at": expires_at,
                "auth_method": "public_key",
                "key_id": str(key_id or "").strip(),
                "scope": str(scope or "control").strip().lower() or "control",
                "config_paths": sorted([str(item or "").strip() for item in list(config_paths or []) if str(item or "").strip()]),
                "engine_ids": sorted([str(item or "").strip() for item in list(engine_ids or []) if str(item or "").strip()]),
                "ssh_binding": dict((self._current_ssh_session_binding() if bind_to_ssh else None) or {}),
            }

    def _clear_cached_public_key_session(
        self,
        *,
        key_id: str,
        scope: str,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        bind_to_ssh: bool = True,
    ) -> None:
        try:
            key = self._public_key_session_cache_key(
                key_id=key_id,
                scope=scope,
                config_paths=config_paths,
                engine_ids=engine_ids,
                bind_to_ssh=bind_to_ssh,
            )
        except Exception:
            return
        with _AUTO_SESSION_CACHE_LOCK:
            _AUTO_SESSION_CACHE.pop(key, None)

    def invoke_control_command(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke a daemon control command through this channel."""
        return self._invoke(str(command or "").strip(), dict(payload or {}))

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
        from .daemon import DaemonPidFile
        from .engine_host_connection import LocalSocketConnection
        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        pid_path = _resolved_pid_path(pid_info, pid_file_path)
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
                pid_file=pid_path,
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
            try:
                auth = conn.invoke("auth-status", payload)
            except Exception as exc:
                status["auth_status_error"] = str(exc)
                conn.close()
                return self._finalize_daemon_status(status)
            conn.close()
            status["auth_status"] = dict(auth or {}) if isinstance(auth, dict) else None
        except Exception as exc:
            status["alive"] = False
            status["reachable"] = False
            status["reachability_error"] = str(exc)
            status["auth_status_error"] = str(exc)
        return self._finalize_daemon_status(status)

    def bootstrap_daemon(self, *, wait_ready_seconds: float = 8.0, recover_unreachable: bool = False) -> Dict[str, Any]:
        """Start local daemon if not already running. Returns daemon status dict."""
        from .daemon import DaemonPidFile, start_daemon_background, DEFAULT_DAEMON_PORT
        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        pid_path = _resolved_pid_path(pid_info, pid_file_path)
        auto_recovery_attempted = False
        auto_recovery_policy: Optional[Dict[str, Any]] = None
        auto_recovery_stop: Optional[Dict[str, Any]] = None
        if pid_info.is_alive():
            status = self.get_daemon_status()
            if bool(status.get("alive") or status.get("reachable")):
                return {"already_running": True, **status}
            recovery_policy = self._should_auto_recover_unreachable_local_daemon()
            if bool(recovery_policy.get("auto_recover")) and bool(recover_unreachable):
                auto_recovery_attempted = True
                auto_recovery_policy = dict(recovery_policy)
                auto_recovery_stop = self.force_stop_daemon(stop_workers=True, stop_orphan_workers=True)
                if pid_info.is_alive():
                    return {
                        "already_running": False,
                        "blocked_by_unreachable_pid": True,
                        "auto_recovery_attempted": True,
                        "auto_recovery_allowed": bool(recovery_policy.get("auto_recover")),
                        "auto_recovery_requires_explicit_request": False,
                        "auto_recovery_policy": recovery_policy,
                        "force_stop": auto_recovery_stop,
                        "error": "existing daemon PID is alive but automatic recovery did not stop it",
                        **self.get_daemon_status(),
                    }
            else:
                return {
                    "already_running": False,
                    "blocked_by_unreachable_pid": True,
                    "auto_recovery_attempted": False,
                    "auto_recovery_allowed": bool(recovery_policy.get("auto_recover")),
                    "auto_recovery_requires_explicit_request": bool(recovery_policy.get("auto_recover")),
                    "auto_recovery_policy": recovery_policy,
                    "error": "existing daemon PID is alive but the local control channel is not reachable",
                    **status,
                }
        bootstrap_cfg = self._prepare_local_unconfigured_bootstrap()
        result = start_daemon_background(
            port=self._daemon_port_override or DEFAULT_DAEMON_PORT,
            pid_file=pid_path,
            log_file=Path(self._daemon_log_file) if self._daemon_log_file else None,
            engines_state_file=Path(self._engines_state_file) if self._engines_state_file else None,
            control_state_file=Path(self._control_state_file) if self._control_state_file else None,
            wait_ready_seconds=wait_ready_seconds,
        )
        with self._connection_lock:
            self._connection = None  # Force reconnect on next invoke
        return {
            "already_running": False,
            "auto_recovery_attempted": auto_recovery_attempted,
            "auto_recovery_policy": auto_recovery_policy,
            "force_stop": auto_recovery_stop,
            "bootstrap_control_config": dict(bootstrap_cfg or {}) if isinstance(bootstrap_cfg, dict) else None,
            **result,
            **self.get_daemon_status(),
        }

    def stop_daemon(self, *, reason: str = "client_requested_shutdown", requested_by: str = "EngineHostControlChannel.stop_daemon") -> Dict[str, Any]:
        """Send graceful shutdown signal to local daemon."""
        from .daemon import DaemonPidFile
        from .engine_host_connection import LocalSocketConnection
        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        pid_path = _resolved_pid_path(pid_info, pid_file_path)
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
                pid_file=pid_path,
                timeout=5.0,
                max_reconnect_attempts=1,
            )
            conn.invoke(
                "__shutdown__",
                {
                    "shutdown_token": token,
                    "shutdown_reason": str(reason or "client_requested_shutdown"),
                    "requested_by": str(requested_by or "EngineHostControlChannel.stop_daemon"),
                },
            )
            conn.close()
            with self._connection_lock:
                self._connection = None
            return {"status": "shutdown_sent"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def _stop_daemon_with_reason(self, *, reason: str, requested_by: str) -> Dict[str, Any]:
        try:
            return self.stop_daemon(reason=reason, requested_by=requested_by)
        except TypeError:
            # Preserve compatibility with tests/embedders that monkeypatch the
            # old no-argument stop_daemon method.
            return self.stop_daemon()

    def _write_local_daemon_report(self, *, event: str, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        try:
            from .daemon.diagnostics import daemon_report_path_for_control_state, write_daemon_report

            write_daemon_report(
                event=event,
                reason=reason,
                actor={
                    "requested_by": "EngineHostControlChannel",
                    "transport": "local_helper",
                    "peer_pid": os.getpid(),
                    "peer_process": {"pid": os.getpid(), "name": Path(sys.argv[0]).name if sys.argv else None},
                },
                details=dict(details or {}),
                path=daemon_report_path_for_control_state(self._local_control_state_path()),
            )
        except Exception:
            logger.debug("Failed to write local daemon diagnostic report", exc_info=True)

    def _list_local_engine_worker_processes(self) -> List[Dict[str, Any]]:
        current_pid = os.getpid()
        rows: List[Dict[str, Any]] = []
        try:
            if sys.platform == "win32":
                script = (
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { $_.CommandLine -match 'hosting\\.engine_worker_ipc' } | "
                    "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
                )
                proc = subprocess.run(  # noqa: S603
                    ["powershell", "-NoProfile", "-Command", script],
                    text=True,
                    capture_output=True,
                    timeout=8.0,
                    check=False,
                    **hidden_subprocess_kwargs(),
                )
                raw = (proc.stdout or "").strip()
                if not raw:
                    return []
                parsed = json.loads(raw)
                items = parsed if isinstance(parsed, list) else [parsed]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    pid = int(item.get("ProcessId") or 0)
                    if pid > 0 and pid != current_pid:
                        rows.append(
                            {
                                "pid": pid,
                                "parent_pid": int(item.get("ParentProcessId") or 0),
                                "command": str(item.get("CommandLine") or ""),
                            }
                        )
                return rows
            proc = subprocess.run(  # noqa: S603
                ["ps", "-eo", "pid=,ppid=,command="],
                text=True,
                capture_output=True,
                timeout=8.0,
                check=False,
                **hidden_subprocess_kwargs(),
            )
            for line in (proc.stdout or "").splitlines():
                if "hosting.engine_worker_ipc" not in line:
                    continue
                parts = line.strip().split(None, 2)
                if len(parts) < 3:
                    continue
                pid = int(parts[0])
                if pid > 0 and pid != current_pid:
                    rows.append({"pid": pid, "parent_pid": int(parts[1]), "command": parts[2]})
        except Exception as exc:
            logger.debug("Failed to list local hosting worker processes: %s", exc)
        return rows

    def force_stop_daemon(
        self,
        *,
        stop_workers: bool = True,
        stop_orphan_workers: bool = True,
        wait_seconds: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Local-only recovery helper: stop registered workers, then terminate the daemon PID.

        This is intentionally stronger than stop_daemon() and should only be used
        from operator recovery flows.
        """
        if str(self.get_target().get("mode") or "local") != "local":
            raise ValueError("force_stop_daemon is only valid in local mode")
        from .daemon import DaemonPidFile
        from .service.host_service import EngineHostService

        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        info = dict(pid_info.read() or {})
        worker_report: Dict[str, Any] = {
            "enabled": bool(stop_workers),
            "registered_attempted": 0,
            "registered_stopped": 0,
            "registered_failed": 0,
            "orphan_scan_enabled": bool(stop_orphan_workers),
            "orphan_attempted": 0,
            "orphan_stopped": 0,
            "orphan_failed": 0,
            "attempted": 0,
            "stopped": 0,
            "failed": 0,
            "results": [],
        }
        registered_pids: set[int] = set()
        if stop_workers:
            svc = EngineHostService(
                engines_state_file=Path(self._engines_state_file) if self._engines_state_file else None,
                control_state_file=self._local_control_state_path(),
            )
            try:
                rows = svc.discover_running(
                    prune_stale=False,
                    include_progress=False,
                    include_reachability=False,
                )
                for row in list(rows or []) if isinstance(rows, list) else []:
                    engine_id = str((row or {}).get("engine_id") or "").strip()
                    if not engine_id:
                        continue
                    pid = int((row or {}).get("pid") or 0)
                    if pid > 0:
                        registered_pids.add(pid)
                    worker_report["registered_attempted"] = int(worker_report.get("registered_attempted") or 0) + 1
                    try:
                        out = svc.shutdown(engine_id, timeout_seconds=2.0)
                        status = str((out or {}).get("status") or "")
                        ok = status in {"stopped", "already_stopped", "not_found", "invalid_pid"}
                        key = "registered_stopped" if ok else "registered_failed"
                        worker_report[key] = int(worker_report.get(key) or 0) + 1
                        worker_report["results"].append({"kind": "registered", "engine_id": engine_id, "pid": pid, "status": status, "ok": ok})
                    except Exception as exc:
                        worker_report["registered_failed"] = int(worker_report.get("registered_failed") or 0) + 1
                        worker_report["results"].append({"kind": "registered", "engine_id": engine_id, "pid": pid, "status": "exception", "ok": False, "error": str(exc)})
            except Exception as exc:
                worker_report["error"] = str(exc)

        if stop_workers and stop_orphan_workers:
            for proc_info in self._list_local_engine_worker_processes():
                worker_pid = int(proc_info.get("pid") or 0)
                if worker_pid <= 0 or worker_pid in registered_pids:
                    continue
                worker_report["orphan_attempted"] = int(worker_report.get("orphan_attempted") or 0) + 1
                try:
                    os.kill(worker_pid, getattr(signal, "SIGTERM", 15))
                    deadline = time.time() + 1.5
                    while time.time() < deadline:
                        if not pid_alive(worker_pid):
                            break
                        time.sleep(0.1)
                    if pid_alive(worker_pid):
                        os.kill(worker_pid, getattr(signal, "SIGKILL", getattr(signal, "SIGTERM", 15)))
                        time.sleep(0.1)
                    alive = pid_alive(worker_pid)
                    ok = not alive
                    key = "orphan_stopped" if ok else "orphan_failed"
                    worker_report[key] = int(worker_report.get(key) or 0) + 1
                    worker_report["results"].append({"kind": "orphan_process", "pid": worker_pid, "status": "stopped" if ok else "stop_failed", "ok": ok})
                except Exception as exc:
                    worker_report["orphan_failed"] = int(worker_report.get("orphan_failed") or 0) + 1
                    worker_report["results"].append({"kind": "orphan_process", "pid": worker_pid, "status": "exception", "ok": False, "error": str(exc)})

        worker_report["attempted"] = int(worker_report.get("registered_attempted") or 0) + int(worker_report.get("orphan_attempted") or 0)
        worker_report["stopped"] = int(worker_report.get("registered_stopped") or 0) + int(worker_report.get("orphan_stopped") or 0)
        worker_report["failed"] = int(worker_report.get("registered_failed") or 0) + int(worker_report.get("orphan_failed") or 0)

        graceful = self._stop_daemon_with_reason(
            reason="force_stop_daemon_graceful_phase",
            requested_by="EngineHostControlChannel.force_stop_daemon",
        )
        pid = int(info.get("pid") or 0)
        terminate_result: Dict[str, Any] = {"pid": pid, "attempted": False, "status": "not_needed"}
        if pid > 0 and pid_info.is_alive():
            terminate_result = {"pid": pid, "attempted": True, "status": "running"}
            self._write_local_daemon_report(
                event="daemon_force_terminate_requested",
                reason="force_stop_daemon_after_graceful_stop_failed",
                details={
                    "pid": pid,
                    "pid_file": str(getattr(pid_info, "path", "") or ""),
                    "graceful_stop": dict(graceful or {}),
                    "worker_shutdown": worker_report,
                },
            )
            try:
                os.kill(pid, getattr(signal, "SIGTERM", 15))
                deadline = time.time() + max(0.1, float(wait_seconds))
                while time.time() < deadline:
                    if not pid_info.is_alive():
                        break
                    time.sleep(0.1)
                if pid_info.is_alive():
                    sigkill = getattr(signal, "SIGKILL", getattr(signal, "SIGTERM", 15))
                    os.kill(pid, sigkill)
                    time.sleep(0.2)
                terminate_result["status"] = "terminated" if not pid_info.is_alive() else "terminate_failed"
            except Exception as exc:
                terminate_result["status"] = "error"
                terminate_result["error"] = str(exc)
        if not pid_info.is_alive():
            if pid > 0 and bool(terminate_result.get("attempted")):
                self._write_local_daemon_report(
                    event="daemon_force_terminated",
                    reason=str(terminate_result.get("status") or "force_stop_daemon_completed"),
                    details={
                        "pid": pid,
                        "pid_file": str(getattr(pid_info, "path", "") or ""),
                        "graceful_stop": dict(graceful or {}),
                        "daemon_terminate": dict(terminate_result or {}),
                    },
                )
            pid_info.remove()
        with self._connection_lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None
        return {
            "status": "ok" if str(terminate_result.get("status") or "") != "terminate_failed" else "error",
            "local_helper_only": True,
            "worker_shutdown": worker_report,
            "graceful_stop": graceful,
            "daemon_terminate": terminate_result,
            "daemon_status": self.get_daemon_status(),
        }

    def force_restart_daemon(self, *, wait_ready_seconds: float = 8.0) -> Dict[str, Any]:
        stop = self.force_stop_daemon(stop_workers=True, stop_orphan_workers=True)
        start = self.bootstrap_daemon(wait_ready_seconds=wait_ready_seconds)
        return {"status": "ok" if (start.get("alive") or start.get("reachable") or start.get("already_running")) else "error", "force_stop": stop, "start": start}

    def reset_hosting_access(self) -> Dict[str, Any]:
        """
        Local-only helper: stop local daemon and clear auth state from control config.

        This helper intentionally does not go through daemon RPC/auth surfaces.
        """
        if str(self.get_target().get("mode") or "local") != "local":
            raise ValueError("reset_hosting_access is only valid in local mode")
        from .daemon import DaemonPidFile
        from .service.host_service import EngineHostService

        pid_file_path = self.control_settings.get("engine_host_daemon_pid_file")
        pid_info = DaemonPidFile(pid_file_path)
        daemon_info = dict(pid_info.read() or {})
        stop_result = self._stop_daemon_with_reason(
            reason="reset_hosting_access",
            requested_by="EngineHostControlChannel.reset_hosting_access",
        )
        pid = int(daemon_info.get("pid") or 0)
        if pid > 0 and bool(pid_info.is_alive()):
            self._write_local_daemon_report(
                event="daemon_force_terminate_requested",
                reason="reset_hosting_access_after_graceful_stop_failed",
                details={
                    "pid": pid,
                    "pid_file": str(getattr(pid_info, "path", "") or ""),
                    "graceful_stop": dict(stop_result or {}),
                },
            )
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
            if pid > 0 and bool(dict(stop_result or {}).get("forced_kill_attempted")):
                self._write_local_daemon_report(
                    event="daemon_force_terminated",
                    reason=str(dict(stop_result or {}).get("status") or "reset_hosting_access_terminate_completed"),
                    details={
                        "pid": pid,
                        "pid_file": str(getattr(pid_info, "path", "") or ""),
                        "daemon_stop": dict(stop_result or {}),
                    },
                )
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
        if not known_hosts_line:
            raise RuntimeError("ssh_known_hosts_line is required for restart_remote_daemon in SSH mode")
        if known_hosts_line:
            import tempfile as _tempfile, os as _os
            try:
                fd, tmppath = _tempfile.mkstemp(prefix="mp13_kh_", suffix=".txt")
                with _os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(known_hosts_line + "\n")
                argv += ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={tmppath}"]
            except Exception:
                raise RuntimeError("strict SSH host-key verification requires writable temporary known_hosts file")
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
                **hidden_subprocess_kwargs(),
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

    def spawn_process(
        self,
        *,
        engine_id: str,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        worker_profile_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "spawn",
            {
                "engine_id": str(engine_id),
                "command": [str(x) for x in list(command or [])],
                "cwd": str(cwd) if cwd else None,
                "env": dict(env or {}),
                "worker_profile_class": str(worker_profile_class or "").strip() or None,
            },
        )
        return dict(res or {})

    def workflow_python_environment_spec(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        python: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-environment-spec",
            {
                "profile": str(profile or "helper").strip() or "helper",
                "environment_name": str(environment_name or "workflow-python-helper").strip() or "workflow-python-helper",
                "python": dict(python or {}),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def workflow_python_prepare_environment(
        self,
        *,
        environment_name: str = "workflow-python-helper",
        python: Optional[Dict[str, Any]] = None,
        package_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-prepare-environment",
            {
                "environment_name": str(environment_name or "workflow-python-helper").strip() or "workflow-python-helper",
                "python": dict(python or {}),
                "package_id": str(package_id or "").strip() or None,
                "workflow_id": str(workflow_id or "").strip() or None,
            },
        )
        return dict(res or {})

    def workflow_python_lock_environment(self, *, environment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self._invoke("workflow-python-lock-environment", {"environment": dict(environment or {})})
        return dict(res or {})

    def workflow_python_verify_environment(self, *, environment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self._invoke("workflow-python-verify-environment", {"environment": dict(environment or {})})
        return dict(res or {})

    def workflow_python_install_environment(
        self,
        *,
        environment: Optional[Dict[str, Any]] = None,
        allow_execution: bool = False,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-install-environment",
            {"environment": dict(environment or {}), "allow_execution": bool(allow_execution)},
        )
        return dict(res or {})

    def workflow_python_verify_install_receipt(self, *, environment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self._invoke("workflow-python-verify-install-receipt", {"environment": dict(environment or {})})
        return dict(res or {})

    def sandbox_state_snapshot(
        self,
        *,
        scope: str,
        workflow_id: str = "",
        instance_id: str = "",
        request_id: str = "",
        prefix: str = "",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "sandbox-state-snapshot",
            {
                "scope": str(scope or "").strip(),
                "workflow_id": str(workflow_id or "").strip() or None,
                "instance_id": str(instance_id or "").strip() or None,
                "request_id": str(request_id or "").strip() or None,
                "prefix": str(prefix or ""),
            },
        )
        return dict(res or {})

    def sandbox_state_restore(
        self,
        *,
        snapshot: Dict[str, Any],
        scope: str = "",
        workflow_id: str = "",
        instance_id: str = "",
        request_id: str = "",
        mode: str = "merge",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "sandbox-state-restore",
            {
                "snapshot": dict(snapshot or {}),
                "scope": str(scope or "").strip() or None,
                "workflow_id": str(workflow_id or "").strip() or None,
                "instance_id": str(instance_id or "").strip() or None,
                "request_id": str(request_id or "").strip() or None,
                "mode": str(mode or "merge").strip() or "merge",
            },
        )
        return dict(res or {})

    def workflow_artifact_recovery_inspect(
        self,
        *,
        request_id: str,
        names: Optional[list[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-artifact-recovery-inspect",
            {
                "request_id": str(request_id or "").strip(),
                "names": list(names or []),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def workflow_artifact_recovery_claim(
        self,
        *,
        request_id: str,
        names: Optional[list[str]] = None,
        target_id: str = "",
        instance_id: str = "",
        patch_absolute_paths: bool = False,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-artifact-recovery-claim",
            {
                "request_id": str(request_id or "").strip(),
                "names": list(names or []),
                "target_id": str(target_id or "").strip(),
                "instance_id": str(instance_id or "").strip(),
                "patch_absolute_paths": bool(patch_absolute_paths),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def workflow_artifact_recovery_cleanup(
        self,
        *,
        request_id: str,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-artifact-recovery-cleanup",
            {
                "request_id": str(request_id or "").strip(),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def ensure_workflow_python(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        environment_key: Optional[str] = None,
        python: Optional[Dict[str, Any]] = None,
        python_executable: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        engine_id: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-ensure",
            {
                "profile": str(profile or "helper").strip() or "helper",
                "environment_name": str(environment_name or "workflow-python-helper").strip() or "workflow-python-helper",
                "environment_key": str(environment_key or "").strip() or None,
                "python": dict(python or {}),
                "python_executable": str(python_executable or "").strip() or None,
                "capacity": max(1, min(int(capacity or 1), 256)),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
                "engine_id": str(engine_id or "").strip() or None,
                "worker_profile_class": str(worker_profile_class or "generic").strip() or "generic",
            },
        )
        return dict(res or {})

    def execute_workflow_python(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        approval_requester_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "profile": str(profile or "helper").strip() or "helper",
            "environment_name": str(environment_name or "workflow-python-helper").strip() or "workflow-python-helper",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request": dict(request or {}),
            "capacity": max(1, min(int(capacity or 1), 256)),
            "sandbox_policy": dict(sandbox_policy or {}) or None,
        }
        if approval_requester_binding is not None:
            payload["approval_requester_binding"] = dict(approval_requester_binding or {})
        res = self._invoke(
            "workflow-python-execute",
            payload,
        )
        return dict(res or {})

    def workflow_python_action_describe(self, *, request: Optional[Dict[str, Any]] = None, include_hidden: bool = False) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-action-describe",
            {"request": dict(request or {}), "include_hidden": bool(include_hidden)},
        )
        return dict(res or {})

    def execute_workflow_python_action(
        self,
        *,
        action_name: str,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        approval_requester_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "action_name": str(action_name or "").strip(),
            "profile": str(profile or "helper").strip() or "helper",
            "environment_name": str(environment_name or "workflow-python-helper").strip() or "workflow-python-helper",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request": dict(request or {}),
            "capacity": max(1, min(int(capacity or 1), 256)),
            "sandbox_policy": dict(sandbox_policy or {}) or None,
        }
        if approval_requester_binding is not None:
            payload["approval_requester_binding"] = dict(approval_requester_binding or {})
        res = self._invoke(
            "workflow-python-action-execute",
            payload,
        )
        return dict(res or {})

    def workflow_python_instance_create(
        self,
        *,
        request: Optional[Dict[str, Any]] = None,
        instance_id: Optional[str] = None,
        replace: bool = False,
        profile: str = "node",
        environment_name: str = "workflow-python-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-instance-create",
            {
                "profile": str(profile or "node").strip() or "node",
                "environment_name": str(environment_name or "workflow-python-node").strip() or "workflow-python-node",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "request": dict(request or {}),
                "instance_id": str(instance_id or "").strip() or None,
                "replace": bool(replace),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def workflow_python_instance_execute(
        self,
        *,
        instance_id: str,
        request: Optional[Dict[str, Any]] = None,
        profile: str = "node",
        environment_name: str = "workflow-python-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        approval_requester_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "instance_id": str(instance_id or "").strip(),
            "profile": str(profile or "node").strip() or "node",
            "environment_name": str(environment_name or "workflow-python-node").strip() or "workflow-python-node",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request": dict(request or {}),
            "capacity": max(1, min(int(capacity or 1), 256)),
            "sandbox_policy": dict(sandbox_policy or {}) or None,
        }
        if approval_requester_binding is not None:
            payload["approval_requester_binding"] = dict(approval_requester_binding or {})
        res = self._invoke(
            "workflow-python-instance-execute",
            payload,
        )
        return dict(res or {})

    def workflow_python_instance_close(self, *, instance_id: str, reason: str = "client_requested") -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-instance-close",
            {"instance_id": str(instance_id or "").strip(), "reason": str(reason or "client_requested")},
        )
        return dict(res or {})

    def workflow_python_instance_list(self) -> Dict[str, Any]:
        res = self._invoke("workflow-python-instance-list", {})
        return dict(res or {})

    def workflow_python_resources(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        python: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-resources",
            {
                "profile": str(profile or "helper").strip() or "helper",
                "environment_name": str(environment_name or "workflow-python-helper").strip() or "workflow-python-helper",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "python": dict(python or {}),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def set_workflow_python_capacity(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        capacity: int,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-set-capacity",
            {
                "profile": str(profile or "helper").strip() or "helper",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "capacity": max(1, min(int(capacity or 1), 256)),
            },
        )
        return dict(res or {})

    def cancel_workflow_python_request(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-cancel-request",
            {
                "profile": str(profile or "helper").strip() or "helper",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "request_id": str(request_id or "").strip(),
            },
        )
        return dict(res or {})

    def workflow_python_request_status(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-request-status",
            {
                "profile": str(profile or "helper").strip() or "helper",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "request_id": str(request_id or "").strip(),
            },
        )
        return dict(res or {})

    def workflow_python_stream_open(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-python-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        python: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        approval_requester_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "profile": str(profile or "node").strip() or "node",
            "environment_name": str(environment_name or "workflow-python-node").strip() or "workflow-python-node",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request": dict(request or {}),
            "python": dict(python or {}),
            "sandbox_policy": dict(sandbox_policy or {}) or None,
            "capacity": max(1, min(int(capacity or 1), 256)),
        }
        if approval_requester_binding is not None:
            payload["approval_requester_binding"] = dict(approval_requester_binding or {})
        res = self._invoke(
            "workflow-python-stream-open",
            payload,
        )
        return dict(res or {})

    def workflow_python_event_subscribe(self, *, stream_id: str, max_items: int = 64) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-event-subscribe",
            {"stream_id": str(stream_id or "").strip(), "max_items": max(1, min(int(max_items or 64), 4096))},
        )
        return dict(res or {})

    def workflow_python_stream_send(self, *, stream_id: str, message: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-python-stream-send",
            {"stream_id": str(stream_id or "").strip(), "message": dict(message or {})},
        )
        return dict(res or {})

    def workflow_python_stream_close(self, *, stream_id: str) -> Dict[str, Any]:
        res = self._invoke("workflow-python-stream-close", {"stream_id": str(stream_id or "").strip()})
        return dict(res or {})

    def workflow_js_environment_spec(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-environment-spec",
            {
                "profile": str(profile or "node").strip() or "node",
                "environment_name": str(environment_name or "workflow-js-node").strip() or "workflow-js-node",
                "node": dict(node or {}),
                "javascript": dict(javascript or {}),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def ensure_workflow_js(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        engine_id: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-ensure",
            {
                "profile": str(profile or "node").strip() or "node",
                "environment_name": str(environment_name or "workflow-js-node").strip() or "workflow-js-node",
                "environment_key": str(environment_key or "").strip() or None,
                "node": dict(node or {}),
                "javascript": dict(javascript or {}),
                "capacity": max(1, min(int(capacity or 1), 256)),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
                "engine_id": str(engine_id or "").strip() or None,
                "worker_profile_class": str(worker_profile_class or "generic").strip() or "generic",
            },
        )
        return dict(res or {})

    def workflow_js_resources(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-resources",
            {
                "profile": str(profile or "node").strip() or "node",
                "environment_name": str(environment_name or "workflow-js-node").strip() or "workflow-js-node",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "node": dict(node or {}),
                "javascript": dict(javascript or {}),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def execute_workflow_js(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        approval_requester_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "profile": str(profile or "node").strip() or "node",
            "environment_name": str(environment_name or "workflow-js-node").strip() or "workflow-js-node",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request": dict(request or {}),
            "node": dict(node or {}),
            "javascript": dict(javascript or {}),
            "capacity": max(1, min(int(capacity or 1), 256)),
            "sandbox_policy": dict(sandbox_policy or {}) or None,
        }
        if approval_requester_binding is not None:
            payload["approval_requester_binding"] = dict(approval_requester_binding or {})
        res = self._invoke(
            "workflow-js-execute",
            payload,
        )
        return dict(res or {})

    def workflow_js_action_describe(self, *, request: Optional[Dict[str, Any]] = None, include_hidden: bool = False) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-action-describe",
            {"request": dict(request or {}), "include_hidden": bool(include_hidden)},
        )
        return dict(res or {})

    def execute_workflow_js_action(
        self,
        *,
        action_name: str,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        approval_requester_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "action_name": str(action_name or "").strip(),
            "profile": str(profile or "node").strip() or "node",
            "environment_name": str(environment_name or "workflow-js-node").strip() or "workflow-js-node",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request": dict(request or {}),
            "node": dict(node or {}),
            "javascript": dict(javascript or {}),
            "capacity": max(1, min(int(capacity or 1), 256)),
            "sandbox_policy": dict(sandbox_policy or {}) or None,
        }
        if approval_requester_binding is not None:
            payload["approval_requester_binding"] = dict(approval_requester_binding or {})
        res = self._invoke(
            "workflow-js-action-execute",
            payload,
        )
        return dict(res or {})

    def workflow_js_instance_create(
        self,
        *,
        request: Optional[Dict[str, Any]] = None,
        instance_id: Optional[str] = None,
        replace: bool = False,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-instance-create",
            {
                "profile": str(profile or "node").strip() or "node",
                "environment_name": str(environment_name or "workflow-js-node").strip() or "workflow-js-node",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "request": dict(request or {}),
                "node": dict(node or {}),
                "javascript": dict(javascript or {}),
                "instance_id": str(instance_id or "").strip() or None,
                "replace": bool(replace),
                "sandbox_policy": dict(sandbox_policy or {}) or None,
            },
        )
        return dict(res or {})

    def workflow_js_instance_execute(
        self,
        *,
        instance_id: str,
        request: Optional[Dict[str, Any]] = None,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        approval_requester_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "instance_id": str(instance_id or "").strip(),
            "profile": str(profile or "node").strip() or "node",
            "environment_name": str(environment_name or "workflow-js-node").strip() or "workflow-js-node",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request": dict(request or {}),
            "node": dict(node or {}),
            "javascript": dict(javascript or {}),
            "capacity": max(1, min(int(capacity or 1), 256)),
            "sandbox_policy": dict(sandbox_policy or {}) or None,
        }
        if approval_requester_binding is not None:
            payload["approval_requester_binding"] = dict(approval_requester_binding or {})
        res = self._invoke(
            "workflow-js-instance-execute",
            payload,
        )
        return dict(res or {})

    def workflow_js_instance_close(self, *, instance_id: str, reason: str = "client_requested") -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-instance-close",
            {"instance_id": str(instance_id or "").strip(), "reason": str(reason or "client_requested")},
        )
        return dict(res or {})

    def workflow_js_instance_list(self) -> Dict[str, Any]:
        res = self._invoke("workflow-js-instance-list", {})
        return dict(res or {})

    def set_workflow_js_capacity(
        self,
        *,
        profile: str = "node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        capacity: int,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-set-capacity",
            {
                "profile": str(profile or "node").strip() or "node",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "capacity": max(1, min(int(capacity or 1), 256)),
            },
        )
        return dict(res or {})

    def cancel_workflow_js_request(
        self,
        *,
        profile: str = "node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-cancel-request",
            {
                "profile": str(profile or "node").strip() or "node",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "request_id": str(request_id or "").strip(),
            },
        )
        return dict(res or {})

    def workflow_js_request_status(
        self,
        *,
        profile: str = "node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-request-status",
            {
                "profile": str(profile or "node").strip() or "node",
                "environment_key": str(environment_key or "").strip() or None,
                "engine_id": str(engine_id or "").strip() or None,
                "request_id": str(request_id or "").strip(),
            },
        )
        return dict(res or {})

    def workflow_js_stream_open(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        approval_requester_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "profile": str(profile or "node").strip() or "node",
            "environment_name": str(environment_name or "workflow-js-node").strip() or "workflow-js-node",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request": dict(request or {}),
            "node": dict(node or {}),
            "javascript": dict(javascript or {}),
            "sandbox_policy": dict(sandbox_policy or {}) or None,
            "capacity": max(1, min(int(capacity or 1), 256)),
        }
        if approval_requester_binding is not None:
            payload["approval_requester_binding"] = dict(approval_requester_binding or {})
        res = self._invoke(
            "workflow-js-stream-open",
            payload,
        )
        return dict(res or {})

    def workflow_js_event_subscribe(self, *, stream_id: str, max_items: int = 64) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-event-subscribe",
            {"stream_id": str(stream_id or "").strip(), "max_items": max(1, min(int(max_items or 64), 4096))},
        )
        return dict(res or {})

    def workflow_js_stream_send(self, *, stream_id: str, message: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self._invoke(
            "workflow-js-stream-send",
            {"stream_id": str(stream_id or "").strip(), "message": dict(message or {})},
        )
        return dict(res or {})

    def workflow_js_stream_close(self, *, stream_id: str) -> Dict[str, Any]:
        res = self._invoke("workflow-js-stream-close", {"stream_id": str(stream_id or "").strip()})
        return dict(res or {})

    def host_capability_session_register(
        self,
        *,
        methods: List[Dict[str, Any]],
        scope: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        provider_kind: str = "client_session",
        visibility: str = "workflow",
        binding: Optional[Dict[str, Any]] = None,
        close_on_client_disconnect: bool = True,
        expires_at_ms: Optional[int] = None,
        allow_override: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": str(session_id or "").strip() or None,
            "provider_kind": str(provider_kind or "client_session").strip() or "client_session",
            "visibility": str(visibility or "workflow").strip() or "workflow",
            "scope": dict(scope or {}),
            "methods": [dict(row or {}) for row in list(methods or [])],
            "binding": dict(binding or {}),
            "close_on_client_disconnect": bool(close_on_client_disconnect),
            "allow_override": bool(allow_override),
        }
        if expires_at_ms is not None:
            payload["expires_at_ms"] = int(expires_at_ms)
        res = self._invoke("host-capability-session-register", payload)
        return dict(res or {})

    @staticmethod
    def known_host_capability_methods(*, include_fs: bool = True, include_http: bool = True) -> List[Dict[str, Any]]:
        from .sandbox.host_api import known_host_capability_method_descriptors

        return known_host_capability_method_descriptors(include_fs=include_fs, include_http=include_http)

    def host_capability_session_register_known_methods(
        self,
        *,
        scope: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        provider_kind: str = "client_session",
        visibility: str = "workflow",
        binding: Optional[Dict[str, Any]] = None,
        close_on_client_disconnect: bool = True,
        expires_at_ms: Optional[int] = None,
        include_fs: bool = True,
        include_http: bool = True,
        allow_override: bool = False,
    ) -> Dict[str, Any]:
        return self.host_capability_session_register(
            methods=self.known_host_capability_methods(include_fs=include_fs, include_http=include_http),
            scope=scope,
            session_id=session_id,
            provider_kind=provider_kind,
            visibility=visibility,
            binding=binding,
            close_on_client_disconnect=close_on_client_disconnect,
            expires_at_ms=expires_at_ms,
            allow_override=allow_override,
        )

    @staticmethod
    def _host_capability_session_matches(
        session: Dict[str, Any],
        *,
        workflow_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        request_id: Optional[str] = None,
        consumer_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        owner: Optional[str] = None,
        visibility: Optional[str] = None,
        method: Optional[str] = None,
        methods: Optional[List[str]] = None,
    ) -> bool:
        scope = dict(session.get("scope") or {})
        provider = dict(session.get("provider") or {})
        if workflow_id is not None and str(scope.get("workflow_id") or "") != str(workflow_id or ""):
            return False
        if instance_id is not None and str(scope.get("instance_id") or "") != str(instance_id or ""):
            return False
        if request_id is not None and str(scope.get("request_id") or "") != str(request_id or ""):
            return False
        if consumer_id is not None and str(scope.get("consumer_id") or "") != str(consumer_id or ""):
            return False
        if provider_id is not None and str(session.get("session_id") or "") != str(provider_id or ""):
            return False
        if owner is not None and str(session.get("owner") or "") != str(owner or ""):
            return False
        if visibility is not None and str(provider.get("visibility") or "") != str(visibility or ""):
            return False
        wanted = {str(item or "").strip() for item in list(methods or []) if str(item or "").strip()}
        if method is not None and str(method or "").strip():
            wanted.add(str(method or "").strip())
        if wanted:
            names = {
                str(row.get("name") or "").strip()
                for row in list(session.get("methods") or [])
                if isinstance(row, dict) and str(row.get("name") or "").strip()
            }
            if not wanted.intersection(names):
                return False
        return True

    def host_capability_session_list(self, *, include_all: bool = False) -> Dict[str, Any]:
        res = self._invoke("host-capability-session-list", {"include_all": bool(include_all)})
        return dict(res or {})

    def host_capability_session_list_filtered(
        self,
        *,
        workflow_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        request_id: Optional[str] = None,
        consumer_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        owner: Optional[str] = None,
        visibility: Optional[str] = None,
        method: Optional[str] = None,
        methods: Optional[List[str]] = None,
        include_all: bool = False,
    ) -> Dict[str, Any]:
        listing = self.host_capability_session_list(include_all=include_all)
        sessions = [
            dict(session or {})
            for session in list(listing.get("sessions") or [])
            if isinstance(session, dict)
            and self._host_capability_session_matches(
                session,
                workflow_id=workflow_id,
                instance_id=instance_id,
                request_id=request_id,
                consumer_id=consumer_id,
                provider_id=provider_id,
                owner=owner,
                visibility=visibility,
                method=method,
                methods=methods,
            )
        ]
        return {"status": "ok", "sessions": sessions, "count": len(sessions)}

    def host_capability_session_close(self, *, session_id: str, force: bool = False) -> Dict[str, Any]:
        res = self._invoke(
            "host-capability-session-close",
            {"session_id": str(session_id or "").strip(), "force": bool(force)},
        )
        return dict(res or {})

    def host_capability_session_close_filtered(
        self,
        *,
        workflow_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        request_id: Optional[str] = None,
        consumer_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        owner: Optional[str] = None,
        visibility: Optional[str] = None,
        method: Optional[str] = None,
        methods: Optional[List[str]] = None,
        include_all: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        listing = self.host_capability_session_list_filtered(
            workflow_id=workflow_id,
            instance_id=instance_id,
            request_id=request_id,
            consumer_id=consumer_id,
            provider_id=provider_id,
            owner=owner,
            visibility=visibility,
            method=method,
            methods=methods,
            include_all=include_all,
        )
        closed: List[Dict[str, Any]] = []
        for session in list(listing.get("sessions") or []):
            sid = str(dict(session or {}).get("session_id") or "").strip()
            if not sid:
                continue
            closed.append(self.host_capability_session_close(session_id=sid, force=force))
        return {"status": "ok", "closed": closed, "count": len(closed)}

    def host_capability_session_upsert(
        self,
        *,
        methods: List[Dict[str, Any]],
        scope: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        provider_kind: str = "client_session",
        visibility: str = "workflow",
        binding: Optional[Dict[str, Any]] = None,
        close_on_client_disconnect: bool = True,
        expires_at_ms: Optional[int] = None,
        allow_override: bool = False,
        replace_workflow_id: Optional[str] = None,
        replace_instance_id: Optional[str] = None,
        replace_request_id: Optional[str] = None,
        replace_consumer_id: Optional[str] = None,
        replace_provider_id: Optional[str] = None,
        replace_owner: Optional[str] = None,
        replace_method: Optional[str] = None,
        replace_methods: Optional[List[str]] = None,
        include_all: bool = False,
        force_close: bool = False,
    ) -> Dict[str, Any]:
        replacement_methods = list(replace_methods or [])
        if replace_method:
            replacement_methods.append(str(replace_method))
        if not replacement_methods:
            replacement_methods = [
                str(row.get("name") or "").strip()
                for row in list(methods or [])
                if isinstance(row, dict) and str(row.get("name") or "").strip()
            ]
        close_result = self.host_capability_session_close_filtered(
            workflow_id=replace_workflow_id if replace_workflow_id is not None else dict(scope or {}).get("workflow_id"),
            instance_id=replace_instance_id if replace_instance_id is not None else dict(scope or {}).get("instance_id"),
            request_id=replace_request_id if replace_request_id is not None else dict(scope or {}).get("request_id"),
            consumer_id=replace_consumer_id if replace_consumer_id is not None else dict(scope or {}).get("consumer_id"),
            provider_id=replace_provider_id if replace_provider_id is not None else session_id,
            owner=replace_owner,
            visibility=visibility,
            methods=replacement_methods,
            include_all=include_all,
            force=force_close,
        )
        register_result = self.host_capability_session_register(
            methods=methods,
            scope=scope,
            session_id=session_id,
            provider_kind=provider_kind,
            visibility=visibility,
            binding=binding,
            close_on_client_disconnect=close_on_client_disconnect,
            expires_at_ms=expires_at_ms,
            allow_override=allow_override,
        )
        return {"status": "ok", "closed": close_result, "registered": register_result}

    def host_capability_session_register_toolbox(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tools_view: Optional[Dict[str, Any]] = None,
        scope: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        visibility: str = "workflow",
        binding: Optional[Dict[str, Any]] = None,
        close_on_client_disconnect: bool = True,
        expires_at_ms: Optional[int] = None,
        namespace: str = "toolbox",
        owner: str = "client",
        allow_override: bool = False,
        upsert: bool = True,
    ) -> Dict[str, Any]:
        from .callable_surface import toolbox_to_host_capability_descriptors

        description = self.toolbox_describe(engine_id=engine_id, toolbox_id=toolbox_id)
        descriptors = toolbox_to_host_capability_descriptors(
            description,
            tools_view=tools_view,
            provider_id=session_id or toolbox_id or engine_id or "toolbox",
            owner=owner,
            visibility=visibility,
            namespace=namespace,
        )
        methods = [descriptor.to_dict() for descriptor in descriptors]
        register = self.host_capability_session_upsert if upsert else self.host_capability_session_register
        binding_payload = dict(binding or {})
        binding_payload.setdefault("transport", "toolbox_harness")
        binding_payload.setdefault("engine_id", str(engine_id or "").strip())
        binding_payload.setdefault("toolbox_id", str(toolbox_id or "").strip())
        if isinstance(tools_view, dict):
            binding_payload.setdefault("tools_view", dict(tools_view or {}))
        return register(
            methods=methods,
            scope=scope,
            session_id=session_id,
            provider_kind="toolbox_session",
            visibility=visibility,
            binding=binding_payload,
            close_on_client_disconnect=close_on_client_disconnect,
            expires_at_ms=expires_at_ms,
            allow_override=allow_override,
        )

    def host_capability_audit_list(
        self,
        *,
        workflow_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        request_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        method: Optional[str] = None,
        approval_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "host-capability-audit-list",
            {
                "workflow_id": str(workflow_id).strip() if workflow_id is not None else None,
                "instance_id": str(instance_id).strip() if instance_id is not None else None,
                "request_id": str(request_id).strip() if request_id is not None else None,
                "provider_id": str(provider_id).strip() if provider_id is not None else None,
                "method": str(method).strip() if method is not None else None,
                "approval_id": str(approval_id).strip() if approval_id is not None else None,
                "since": float(since) if since is not None else None,
                "until": float(until) if until is not None else None,
                "limit": int(limit or 100),
                "offset": int(offset or 0),
            },
        )
        return dict(res or {})

    def shutdown_managed(self, engine_id: str, *, timeout_seconds: float = 8.0) -> Dict[str, Any]:
        res = self._invoke("shutdown", {"engine_id": str(engine_id), "timeout_seconds": float(timeout_seconds)})
        return dict(res or {})

    def ensure_running(self, engine_id: str) -> Dict[str, Any]:
        res = self._invoke("ensure-running", {"engine_id": str(engine_id)})
        return dict(res or {})

    def unload_model(self, engine_id: str, *, timeout_seconds: float = 30.0, shutdown_all: bool = False) -> Dict[str, Any]:
        res = self._invoke(
            "unload-model",
            {
                "engine_id": str(engine_id),
                "timeout_seconds": float(timeout_seconds or 30.0),
                "shutdown_all": bool(shutdown_all),
            },
        )
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

    def connect_from_config(
        self,
        *,
        config_path: str,
        engine_id: Optional[str] = None,
        model_path: Optional[str] = None,
        force_new_worker: bool = False,
        launch_policy: Optional[str] = None,
        target_worker_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "connect-from-config",
            {
                "config_path": str(config_path or "default"),
                "engine_id": str(engine_id).strip() if engine_id else None,
                "model_path": str(model_path).strip() if model_path else None,
                "force_new_worker": bool(force_new_worker),
                "launch_policy": str(launch_policy).strip() if launch_policy else None,
                "target_worker_id": str(target_worker_id).strip() if target_worker_id else None,
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

    def cancel_host_operation(self, *, operation_id: str, reason: str = "") -> Dict[str, Any]:
        res = self._invoke(
            "op-cancel",
            {
                "operation_id": str(operation_id or "").strip(),
                "reason": str(reason or "").strip() or None,
            },
        )
        return dict(res or {}) if isinstance(res, dict) else {}

    def start_connect_from_config(
        self,
        *,
        config_path: str,
        engine_id: Optional[str] = None,
        model_path: Optional[str] = None,
        force_new_worker: bool = False,
        launch_policy: Optional[str] = None,
        target_worker_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.start_host_operation(
            command="connect-from-config",
            payload={
                "config_path": str(config_path or "default"),
                "engine_id": str(engine_id).strip() if engine_id else None,
                "model_path": str(model_path).strip() if model_path else None,
                "force_new_worker": bool(force_new_worker),
                "launch_policy": str(launch_policy).strip() if launch_policy else None,
                "target_worker_id": str(target_worker_id).strip() if target_worker_id else None,
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

    def sandbox_fs_list(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: Optional[str] = None,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "sandbox-fs-list",
            {
                "engine_id": str(engine_id or "").strip(),
                "root_id": str(root_id or "").strip(),
                "relative_path": relative_path,
                "callback_context": dict(callback_context or {}) if isinstance(callback_context, dict) else None,
            },
        )
        return dict(res or {})

    def sandbox_fs_read_text(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        encoding: str = "utf-8",
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "sandbox-fs-read-text",
            {
                "engine_id": str(engine_id or "").strip(),
                "root_id": str(root_id or "").strip(),
                "relative_path": str(relative_path or ""),
                "encoding": str(encoding or "utf-8"),
                "callback_context": dict(callback_context or {}) if isinstance(callback_context, dict) else None,
            },
        )
        return dict(res or {})

    def sandbox_fs_write_text(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        text: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "sandbox-fs-write-text",
            {
                "engine_id": str(engine_id or "").strip(),
                "root_id": str(root_id or "").strip(),
                "relative_path": str(relative_path or ""),
                "text": str(text or ""),
                "encoding": str(encoding or "utf-8"),
                "create_parents": bool(create_parents),
                "callback_context": dict(callback_context or {}) if isinstance(callback_context, dict) else None,
            },
        )
        return dict(res or {})

    def sandbox_fs_mkdir(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        parents: bool = True,
        exist_ok: bool = True,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "sandbox-fs-mkdir",
            {
                "engine_id": str(engine_id or "").strip(),
                "root_id": str(root_id or "").strip(),
                "relative_path": str(relative_path or ""),
                "parents": bool(parents),
                "exist_ok": bool(exist_ok),
                "callback_context": dict(callback_context or {}) if isinstance(callback_context, dict) else None,
            },
        )
        return dict(res or {})

    def sandbox_fs_stat(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: Optional[str] = None,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "sandbox-fs-stat",
            {
                "engine_id": str(engine_id or "").strip(),
                "root_id": str(root_id or "").strip(),
                "relative_path": relative_path,
                "callback_context": dict(callback_context or {}) if isinstance(callback_context, dict) else None,
            },
        )
        return dict(res or {})

    def sandbox_http_fetch(
        self,
        *,
        engine_id: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "sandbox-http-fetch",
            {
                "engine_id": str(engine_id or "").strip(),
                "url": str(url or ""),
                "method": str(method or "GET"),
                "headers": dict(headers or {}),
                "body_b64": str(body_b64 or ""),
                "timeout_seconds": float(timeout_seconds or 30.0),
                "max_response_bytes": int(max_response_bytes or 1024 * 1024),
                "callback_context": dict(callback_context or {}) if isinstance(callback_context, dict) else None,
            },
        )
        return dict(res or {})

    def toolbox_describe(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-describe",
            {
                "engine_id": str(engine_id or "").strip(),
                "toolbox_id": str(toolbox_id or "").strip(),
                "timeout_seconds": float(timeout_seconds or 10.0),
            },
        )
        return dict(res or {})

    def toolbox_gate(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_name: str,
        tools_view: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-gate",
            {
                "engine_id": str(engine_id or "").strip(),
                "toolbox_id": str(toolbox_id or "").strip(),
                "tool_name": str(tool_name or "").strip(),
                "tools_view": dict(tools_view or {}) if isinstance(tools_view, dict) else None,
            },
        )
        return dict(res or {})

    def toolbox_execute(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_call: Dict[str, Any],
        timeout_seconds: float = 30.0,
        tools_view: Optional[Dict[str, Any]] = None,
        callback_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-execute",
            {
                "engine_id": str(engine_id or "").strip(),
                "toolbox_id": str(toolbox_id or "").strip(),
                "tool_call": dict(tool_call or {}),
                "timeout_seconds": float(timeout_seconds or 30.0),
                "tools_view": dict(tools_view or {}) if isinstance(tools_view, dict) else None,
                "callback_binding": dict(callback_binding or {}) if isinstance(callback_binding, dict) else None,
            },
        )
        return dict(res or {})

    def toolbox_cancel(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_name: str = "",
        tool_call_id: str = "",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-cancel",
            {
                "engine_id": str(engine_id or "").strip(),
                "toolbox_id": str(toolbox_id or "").strip(),
                "tool_name": str(tool_name or "").strip(),
                "tool_call_id": str(tool_call_id or "").strip(),
                "timeout_seconds": float(timeout_seconds or 8.0),
                "respawn": bool(respawn),
            },
        )
        return dict(res or {})

    def toolbox_gc(self) -> Dict[str, Any]:
        res = self._invoke("toolbox-gc", {})
        return dict(res or {})

    def toolbox_references(self) -> Dict[str, Any]:
        res = self._invoke("toolbox-references", {})
        return dict(res or {})

    def toolbox_consistency(self) -> Dict[str, Any]:
        res = self._invoke("toolbox-consistency", {})
        return dict(res or {})

    def toolbox_review_snapshot(
        self,
        *,
        toolbox_ids: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-review-snapshot",
            {
                "toolbox_ids": [str(item or "").strip() for item in list(toolbox_ids or []) if str(item or "").strip()],
            },
        )
        return dict(res or {})

    def toolbox_repair(
        self,
        *,
        toolbox_ids: Optional[list[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-repair",
            {
                "toolbox_ids": [str(item or "").strip() for item in list(toolbox_ids or []) if str(item or "").strip()],
                "only_inconsistent": bool(only_inconsistent),
                "details": bool(details),
            },
        )
        return dict(res or {})

    def toolbox_reconcile(
        self,
        *,
        toolbox_ids: Optional[list[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-reconcile",
            {
                "toolbox_ids": [str(item or "").strip() for item in list(toolbox_ids or []) if str(item or "").strip()],
                "only_inconsistent": bool(only_inconsistent),
                "details": bool(details),
            },
        )
        return dict(res or {})

    def toolbox_register_auto(
        self,
        *,
        toolbox_id: str,
        requests: list[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-register-auto",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "requests": [dict(item or {}) for item in list(requests or [])],
                "python_executable": str(python_executable or "").strip() or None,
                "worker_profile_class": str(worker_profile_class or "generic").strip() or "generic",
            },
        )
        return dict(res or {})

    def toolbox_unregister_auto(
        self,
        *,
        toolbox_id: str,
        tool_keys: list[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-unregister-auto",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()],
                "python_executable": str(python_executable or "").strip() or None,
                "worker_profile_class": str(worker_profile_class or "generic").strip() or "generic",
            },
        )
        return dict(res or {})

    def toolbox_register_intrinsics(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: list[str],
        include_guides: bool = False,
        sandbox_profile: Optional[Dict[str, Any]] = None,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-register-intrinsics",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "intrinsic_tool_names": [str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
                "include_guides": bool(include_guides),
                "sandbox_profile": dict(sandbox_profile or {}) or None,
                "python_executable": str(python_executable or "").strip() or None,
                "worker_profile_class": str(worker_profile_class or "generic").strip() or "generic",
            },
        )
        return dict(res or {})

    def toolbox_unregister_intrinsics(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: list[str],
        include_guides: bool = False,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-unregister-intrinsics",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "intrinsic_tool_names": [str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
                "include_guides": bool(include_guides),
                "python_executable": str(python_executable or "").strip() or None,
                "worker_profile_class": str(worker_profile_class or "generic").strip() or "generic",
            },
        )
        return dict(res or {})

    def toolbox_register_manual(
        self,
        *,
        toolbox_id: str,
        requests: list[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-register-manual",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "requests": [dict(item or {}) for item in list(requests or [])],
                "python_executable": str(python_executable or "").strip() or None,
                "worker_profile_class": str(worker_profile_class or "generic").strip() or "generic",
            },
        )
        return dict(res or {})

    def toolbox_unregister_manual(
        self,
        *,
        toolbox_id: str,
        tool_keys: list[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-unregister-manual",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()],
                "python_executable": str(python_executable or "").strip() or None,
                "worker_profile_class": str(worker_profile_class or "generic").strip() or "generic",
            },
        )
        return dict(res or {})

    def toolbox_environment_description_list(self) -> Dict[str, Any]:
        res = self._invoke("toolbox-environment-list", {})
        return dict(res or {})

    def toolbox_environment_description_upsert(
        self,
        *,
        name: str,
        base_env_name: Optional[str] = None,
        extra_packages: Optional[list[str]] = None,
        allow_online_install: bool = False,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-upsert",
            {
                "name": str(name or "").strip(),
                "base_env_name": str(base_env_name or "").strip() or None,
                "extra_packages": [str(item or "").strip() for item in list(extra_packages or []) if str(item or "").strip()],
                "allow_online_install": bool(allow_online_install),
            },
        )
        return dict(res or {})

    def toolbox_environment_description_clone(
        self,
        *,
        source_name: str,
        target_name: str,
        extra_packages: Optional[list[str]] = None,
        allow_online_install: Optional[bool] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-clone",
            {
                "source_name": str(source_name or "").strip(),
                "target_name": str(target_name or "").strip(),
                "extra_packages": [str(item or "").strip() for item in list(extra_packages or []) if str(item or "").strip()] if extra_packages is not None else None,
                "allow_online_install": allow_online_install if allow_online_install is not None else None,
            },
        )
        return dict(res or {})

    def toolbox_environment_resolve_requirements(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-resolve",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "environment_name": str(environment_name or "base").strip() or "base",
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            },
        )
        return dict(res or {})

    def toolbox_environment_apply(
        self,
        *,
        environment_name: str,
        toolbox_ids: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-apply",
            {
                "environment_name": str(environment_name or "base").strip() or "base",
                "toolbox_ids": [str(item or "").strip() for item in list(toolbox_ids or []) if str(item or "").strip()] or None,
            },
        )
        return dict(res or {})

    def toolbox_environment_realize(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-realize",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "environment_name": str(environment_name or "base").strip() or "base",
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            },
        )
        return dict(res or {})

    def toolbox_environment_sync_description(
        self,
        *,
        toolbox_id: str,
        source_environment_name: str,
        target_environment_name: Optional[str] = None,
        tool_keys: Optional[list[str]] = None,
        apply: bool = False,
        realize: bool = False,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-sync",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "source_environment_name": str(source_environment_name or "base").strip() or "base",
                "target_environment_name": str(target_environment_name or "").strip() or None,
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                "apply": bool(apply),
                "realize": bool(realize),
            },
        )
        return dict(res or {})

    def toolbox_environment_prepare_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-prepare-install",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "environment_name": str(environment_name or "base").strip() or "base",
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            },
        )
        return dict(res or {})

    def toolbox_environment_lock_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-lock-install",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "environment_name": str(environment_name or "base").strip() or "base",
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            },
        )
        return dict(res or {})

    def toolbox_environment_verify_install_lock(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-verify-install-lock",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "environment_name": str(environment_name or "base").strip() or "base",
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            },
        )
        return dict(res or {})

    def toolbox_environment_resolve_install_lock(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[list[str]] = None,
        allow_resolution: bool = False,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-resolve-install-lock",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "environment_name": str(environment_name or "base").strip() or "base",
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                "allow_resolution": bool(allow_resolution),
            },
        )
        return dict(res or {})

    def toolbox_environment_verify_install_receipt(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-verify-install-receipt",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "environment_name": str(environment_name or "base").strip() or "base",
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            },
        )
        return dict(res or {})

    def toolbox_environment_execute_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[list[str]] = None,
        allow_execution: bool = False,
    ) -> Dict[str, Any]:
        res = self._invoke(
            "toolbox-environment-execute-install",
            {
                "toolbox_id": str(toolbox_id or "").strip(),
                "environment_name": str(environment_name or "base").strip() or "base",
                "tool_keys": [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                "allow_execution": bool(allow_execution),
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

    def list_live_consumers(self) -> Dict[str, Any]:
        res = self._invoke("list-live-consumers", {})
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

    def auth_validate_session(
        self,
        token: Optional[str] = None,
        *,
        scope: str = "control",
        expected_key_id: Optional[str] = None,
        check_ssh_binding: bool = True,
    ) -> Dict[str, Any]:
        tok = str(token or self.get_session_token() or "").strip()
        binding = self._current_ssh_session_binding() if check_ssh_binding else None
        res = self._invoke(
            "auth-validate-session",
            {
                "token": tok,
                "scope": str(scope or "control"),
                "expected_key_id": str(expected_key_id or "").strip() or None,
                "check_ssh_binding": bool(check_ssh_binding),
                "ssh_binding": dict(binding or {}),
            },
            allow_auto_session=False,
        )
        out = dict(res or {})
        if bool(out.get("valid", False)):
            previous = dict(self._session_token_meta or {})
            self._set_session_token_meta(
                {
                    **previous,
                    "auth_method": str(out.get("auth_method") or previous.get("auth_method") or "").strip(),
                    "key_id": str(out.get("key_id") or previous.get("key_id") or "").strip(),
                    "scope": str(out.get("scope") or "").strip().lower(),
                    "expires_at": float(out.get("expires_at") or 0.0),
                    "ssh_binding": dict(out.get("ssh_binding") or {}),
                    "allowed_configs": list(out.get("allowed_configs") or []),
                    "allowed_engines": list(out.get("allowed_engines") or []),
                }
            )
        return out

    def adopt_session_token(
        self,
        token: str,
        *,
        scope: str = "control",
        expected_key_id: Optional[str] = None,
        check_ssh_binding: bool = True,
    ) -> Dict[str, Any]:
        out = self.auth_validate_session(
            token,
            scope=scope,
            expected_key_id=expected_key_id,
            check_ssh_binding=check_ssh_binding,
        )
        if bool(out.get("valid", False)):
            self.set_session_token(token)
            previous = dict(self._session_token_meta or {})
            self._set_session_token_meta(
                {
                    **previous,
                    "auth_method": str(out.get("auth_method") or previous.get("auth_method") or "").strip(),
                    "key_id": str(out.get("key_id") or previous.get("key_id") or "").strip(),
                    "scope": str(out.get("scope") or "").strip().lower(),
                    "expires_at": float(out.get("expires_at") or 0.0),
                    "ssh_binding": dict(out.get("ssh_binding") or {}),
                    "allowed_configs": list(out.get("allowed_configs") or []),
                    "allowed_engines": list(out.get("allowed_engines") or []),
                }
            )
        return out

    def current_session_status(
        self,
        *,
        scope: str = "control",
        expected_key_id: Optional[str] = None,
        check_ssh_binding: bool = True,
    ) -> Dict[str, Any]:
        tok = self.get_session_token()
        if not tok:
            return {"valid": False, "reason": "no_adopted_session", "ssh_bound": False}
        return self.auth_validate_session(
            tok,
            scope=scope,
            expected_key_id=expected_key_id,
            check_ssh_binding=check_ssh_binding,
        )

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
            self._set_session_token_meta(
                {
                    "auth_method": "shared_secret",
                    "key_id": str(key_id or "").strip(),
                    "scope": str(out.get("scope") or scope or "control").strip().lower() or "control",
                    "expires_at": float(out.get("expires_at") or 0.0),
                }
            )
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
            self._set_session_token_meta(
                {
                    "auth_method": "public_key",
                    "key_id": str(out.get("key_id") or "").strip(),
                    "scope": str(out.get("scope") or "").strip().lower(),
                    "expires_at": float(out.get("expires_at") or 0.0),
                    "ssh_binding": dict(out.get("ssh_binding") or {}),
                }
            )
        return out

    def ensure_public_key_session(
        self,
        *,
        key_id: str,
        scope: str = "control",
        signer: Optional[Any] = None,
        private_key_text: str = "",
        signature_ssh: str = "",
        ttl_seconds: int = 120,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        bind_to_ssh: bool = True,
        adopt: bool = True,
        namespace: str = "engine-host-auth",
        sign_timeout_seconds: float = 30.0,
    ) -> str:
        """
        Return a usable public-key session token, reusing an adopted/cached token
        before falling back to challenge signing.

        GUI/browser clients should prefer this over unconditionally running
        auth-begin-challenge/auth-complete-challenge for every operation.
        """
        kid = str(key_id or "").strip()
        if not kid:
            raise ValueError("key_id is required")
        scope_norm = str(scope or "control").strip().lower() or "control"

        current = self.get_session_token()
        if current:
            try:
                validation = self.auth_validate_session(
                    current,
                    scope=scope_norm,
                    expected_key_id=kid,
                    check_ssh_binding=bind_to_ssh,
                )
            except Exception:
                validation = {"valid": False, "reason": "validation_unavailable"}
            if bool(validation.get("valid", False)) and self._public_key_session_meta_matches(
                    key_id=kid,
                    scope=scope_norm,
                    config_paths=config_paths,
                    engine_ids=engine_ids,
                    bind_to_ssh=bind_to_ssh,
                ):
                return current
            if scope_norm == "control":
                try:
                    if bool(validation.get("valid", False)) and str(validation.get("key_id") or "").strip() == kid:
                        return current
                except Exception:
                    self.set_session_token(None)
            if not bool(validation.get("valid", False)):
                self.set_session_token(None)

        cached = self._get_cached_public_key_session(
            key_id=kid,
            scope=scope_norm,
            config_paths=config_paths,
            engine_ids=engine_ids,
            bind_to_ssh=bind_to_ssh,
        )
        if cached:
            try:
                validation = self.adopt_session_token(
                    cached,
                    scope=scope_norm,
                    expected_key_id=kid,
                    check_ssh_binding=bind_to_ssh,
                )
            except Exception:
                validation = {"valid": False, "reason": "validation_unavailable"}
            if bool(validation.get("valid", False)) and self._public_key_session_meta_matches(
                    key_id=kid,
                    scope=scope_norm,
                    config_paths=config_paths,
                    engine_ids=engine_ids,
                    bind_to_ssh=bind_to_ssh,
                ):
                return cached
            if scope_norm == "control":
                try:
                    if bool(validation.get("valid", False)) and str(validation.get("key_id") or "").strip() == kid:
                        return cached
                except Exception:
                    self.set_session_token(None)
            if not bool(validation.get("valid", False)):
                self._clear_cached_public_key_session(
                    key_id=kid,
                    scope=scope_norm,
                    config_paths=config_paths,
                    engine_ids=engine_ids,
                    bind_to_ssh=bind_to_ssh,
                )
                self.set_session_token(None)

        challenge = self.auth_begin_challenge(
            key_id=kid,
            scope=scope_norm,
            ttl_seconds=int(ttl_seconds or 120),
            config_paths=list(config_paths or []),
            engine_ids=list(engine_ids or []),
            bind_to_ssh=bool(bind_to_ssh),
        )
        challenge_text = str(challenge.get("challenge") or challenge.get("challenge_text") or "")
        signature = str(signature_ssh or "").strip()
        if not signature and signer is not None:
            from .client_realm_api import _coerce_signature_ssh

            signature = _coerce_signature_ssh(
                signer(dict(challenge)),
                expected_challenge_id=str(challenge.get("challenge_id") or ""),
            )
        if not signature and private_key_text:
            from .client_realm_api import sign_client_auth_challenge_with_private_key

            signature = sign_client_auth_challenge_with_private_key(
                private_key_text=private_key_text,
                challenge_text=challenge_text,
                namespace=namespace,
                timeout_seconds=sign_timeout_seconds,
            )
        if not signature:
            raise ValueError("signature_ssh, signer, or private_key_text is required")

        result = self.auth_complete_challenge(
            challenge_id=str(challenge.get("challenge_id") or ""),
            signature_ssh=signature,
            adopt=adopt,
        )
        token = str(result.get("token") or "").strip()
        if not token:
            raise RuntimeError("authentication failed: no token returned")
        if adopt:
            self.set_session_token(token)
            self._set_session_token_meta(
                {
                    "auth_method": "public_key",
                    "key_id": kid,
                    "scope": scope_norm,
                    "expires_at": float(result.get("expires_at") or 0.0),
                    "config_paths": sorted([str(item or "").strip() for item in list(config_paths or []) if str(item or "").strip()]),
                    "engine_ids": sorted([str(item or "").strip() for item in list(engine_ids or []) if str(item or "").strip()]),
                    "ssh_binding": dict((self._current_ssh_session_binding() if bind_to_ssh else None) or {}),
                }
            )
            self._store_cached_public_key_session(
                token,
                result,
                key_id=kid,
                scope=scope_norm,
                config_paths=config_paths,
                engine_ids=engine_ids,
                bind_to_ssh=bind_to_ssh,
            )
        return token

    def auth_revoke_session(self, token: str) -> Dict[str, Any]:
        res = self._invoke("auth-revoke-session", {"token": str(token or "")})
        return dict(res or {})

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        return pid_alive(pid)
