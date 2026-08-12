"""Background daemon launch helpers."""
from __future__ import annotations

import http.client
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._process_utils import hidden_subprocess_kwargs
from .constants import DEFAULT_DAEMON_PORT, DEFAULT_HTTP_INGRESS_PORT
from .paths import _default_http_pid_file
from .pidfile import DaemonPidFile


def start_daemon_background(
    *,
    port: int = DEFAULT_DAEMON_PORT,
    pid_file: Optional[Path] = None,
    log_file: Optional[Path] = None,
    mp13_config_file: Optional[Path] = None,
    wait_ready_seconds: float = 8.0,
) -> Dict[str, Any]:
    """
    Spawn daemon as a detached background process and wait until it is connectable.

    Returns {"pid": N, "port": P, "log_file": ...?} on success.
    Raises RuntimeError if daemon does not become reachable within wait_ready_seconds.
    """
    pid_info = DaemonPidFile(pid_file)
    existing_info = pid_info.read() or {}
    process_alive = getattr(pid_info, "process_alive", None)
    existing_pid_alive = bool(process_alive()) if callable(process_alive) else bool(pid_info.is_alive())
    existing_state = str(existing_info.get("lifecycle_state") or "running").strip().lower()
    if existing_pid_alive and existing_state in {"shutting_down", "stopping"}:
        deadline = time.time() + max(1.0, float(wait_ready_seconds))
        while time.time() < deadline:
            time.sleep(0.15)
            existing_info = pid_info.read() or {}
            process_alive = getattr(pid_info, "process_alive", None)
            existing_pid_alive = bool(process_alive()) if callable(process_alive) else bool(pid_info.is_alive())
            existing_state = str(existing_info.get("lifecycle_state") or "running").strip().lower()
            if not existing_pid_alive:
                break
            if existing_state not in {"shutting_down", "stopping"}:
                break
        if existing_pid_alive and existing_state in {"shutting_down", "stopping"}:
            progress = dict(existing_info.get("shutdown_progress") or {}) if isinstance(existing_info.get("shutdown_progress"), dict) else {}
            requested_at = float(existing_info.get("shutdown_requested_at") or progress.get("shutdown_requested_at") or 0.0)
            age = max(0.0, time.time() - requested_at) if requested_at else None
            stage = str(progress.get("stage") or "").strip() or "unknown"
            reason = str(existing_info.get("shutdown_reason") or progress.get("shutdown_reason") or "").strip() or "unknown"
            raise RuntimeError(
                "Existing engine host daemon is still shutting down; "
                "wait for it to exit before starting another daemon "
                f"(pid={int(existing_info.get('pid') or 0)}, pid_file={pid_info.path}, "
                f"shutdown_reason={reason}, shutdown_stage={stage}, shutdown_age_seconds={age})"
            )
    # Only treat a live PID file as an existing hosting daemon when it has the
    # metadata written by DaemonPidFile.write(). This avoids mistaking minimal
    # test/stale PID-like records for a reusable daemon instance.
    process_alive = getattr(pid_info, "process_alive", None)
    existing_pid_alive = bool(process_alive()) if callable(process_alive) else bool(pid_info.is_alive())
    if existing_pid_alive and (existing_info.get("started_at") or existing_info.get("shutdown_token")):
        actual_port = int(existing_info.get("port") or 0)
        if actual_port:
            try:
                from ..engine_host_connection import LocalSocketConnection

                conn_kwargs: Dict[str, Any] = {
                    "port": actual_port,
                    "timeout": 1.0,
                    "max_reconnect_attempts": 1,
                }
                pid_path = getattr(pid_info, "path", None)
                if pid_path is not None:
                    conn_kwargs["pid_file"] = pid_path
                conn = LocalSocketConnection(**conn_kwargs)
                pong = conn.invoke("__ping__", {})
                conn.close()
                if str(pong) == "pong":
                    out: Dict[str, Any] = {
                        "pid": int(existing_info.get("pid") or 0),
                        "port": int(actual_port),
                        "already_running": True,
                    }
                    if log_file:
                        out["log_file"] = str(log_file)
                    return out
            except Exception as exc:
                raise RuntimeError(
                    "Existing engine host daemon PID is alive but local control is not reachable; "
                    "stop or force-restart it before starting another daemon "
                    f"(pid={int(existing_info.get('pid') or 0)}, pid_file={pid_info.path}, error={exc})"
                ) from exc
        raise RuntimeError(
            "Existing engine host daemon PID is alive but PID metadata is incomplete; "
            f"stop or force-restart it before starting another daemon (pid_file={pid_info.path})"
        )

    argv: List[str] = [
        sys.executable,
        "-m",
        "hosting.engine_host_cli",
        "--daemon",
        "--runtime-profile",
        "detached_user_process",
        "--port",
        str(port),
    ]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if pid_file:
        argv += ["--pid-file", str(pid_file)]
    if mp13_config_file:
        argv += ["--mp13-config-file", str(mp13_config_file)]

    # Build environment with src dir on PYTHONPATH so connectors package is found
    env = dict(os.environ)
    src_root = str(Path(__file__).resolve().parents[2])
    py_path = str(env.get("PYTHONPATH") or "")
    if src_root not in py_path.split(os.pathsep):
        env["PYTHONPATH"] = src_root if not py_path else f"{src_root}{os.pathsep}{py_path}"

    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        kwargs.update(hidden_subprocess_kwargs(detached=True, new_process_group=True))
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(argv, **kwargs)  # noqa: S603
    spawned_pid = int(proc.pid)
    try:
        # On Windows, Popen with DETACHED_PROCESS can leave a stale CPython
        # exception. A subsequent C-level call may raise a spurious SystemError.
        # os.kill() triggers the latent error, allowing us to catch and clear it.
        # proc.poll() and proc.returncode do not. See diag_daemon_tcp_crash.py.
        if sys.platform == "win32":
            os.kill(spawned_pid, 0)
        else:
            proc.poll()
    except Exception:
        pass

    # Poll until PID file appears and daemon responds to a protocol ping.
    deadline = time.time() + max(1.0, float(wait_ready_seconds))
    while time.time() < deadline:
        time.sleep(0.15)
        try:
            if not pid_info.is_alive():
                continue
            actual_port = pid_info.get_port()
            if not actual_port:
                continue
            from ..engine_host_connection import LocalSocketConnection

            ready_conn_kwargs: Dict[str, Any] = {
                "port": actual_port,
                "timeout": 1.0,
                "max_reconnect_attempts": 1,
            }
            pid_path = getattr(pid_info, "path", None)
            if pid_path is not None:
                ready_conn_kwargs["pid_file"] = pid_path
            conn = LocalSocketConnection(**ready_conn_kwargs)
            pong = conn.invoke("__ping__", {})
            conn.close()
            if str(pong) != "pong":
                continue
            info = pid_info.read() or {}
            ready_out: Dict[str, Any] = {"pid": int(info.get("pid") or spawned_pid), "port": actual_port}
            if log_file:
                ready_out["log_file"] = str(log_file)
            return ready_out
        except Exception:
            continue

    raise RuntimeError(
        f"Engine host daemon did not become ready within {wait_ready_seconds}s "
        f"(spawned pid={spawned_pid}, port={port}, log_file={log_file})"
    )


def start_http_ingress_background(
    *,
    port: int = DEFAULT_HTTP_INGRESS_PORT,
    pid_file: Optional[Path] = None,
    log_file: Optional[Path] = None,
    mp13_config_file: Optional[Path] = None,
    wait_ready_seconds: float = 8.0,
) -> Dict[str, Any]:
    """
    Spawn HTTP ingress daemon as a detached background process and wait until healthy.

    Returns {"pid": N, "port": P, "log_file": ...?} on success.
    """
    argv: List[str] = [
        sys.executable,
        "-m",
        "hosting.engine_host_cli",
        "--daemon-http",
        "--http-port",
        str(port),
    ]
    if log_file:
        argv += ["--log-file", str(log_file)]
    if pid_file:
        argv += ["--pid-file", str(pid_file)]
    if mp13_config_file:
        argv += ["--mp13-config-file", str(mp13_config_file)]

    env = dict(os.environ)
    src_root = str(Path(__file__).resolve().parents[2])
    py_path = str(env.get("PYTHONPATH") or "")
    if src_root not in py_path.split(os.pathsep):
        env["PYTHONPATH"] = src_root if not py_path else f"{src_root}{os.pathsep}{py_path}"

    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        kwargs.update(hidden_subprocess_kwargs(detached=True, new_process_group=True))
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(argv, **kwargs)  # noqa: S603
    spawned_pid = int(proc.pid)
    try:
        if sys.platform == "win32":
            os.kill(spawned_pid, 0)
        else:
            proc.poll()
    except Exception:
        pass

    pid_info = DaemonPidFile(pid_file or _default_http_pid_file())
    deadline = time.time() + max(1.0, float(wait_ready_seconds))
    while time.time() < deadline:
        time.sleep(0.15)
        try:
            if not pid_info.is_alive():
                continue
            actual_port = pid_info.get_port()
            if not actual_port:
                continue
            conn = http.client.HTTPConnection("127.0.0.1", actual_port, timeout=1.0)  # type: ignore[name-defined]
            conn.request("GET", "/health")
            resp = conn.getresponse()
            _ = resp.read()
            conn.close()
            if int(resp.status) == 200:
                info = pid_info.read() or {}
                out: Dict[str, Any] = {"pid": int(info.get("pid") or spawned_pid), "port": actual_port}
                if log_file:
                    out["log_file"] = str(log_file)
                return out
        except Exception:
            continue

    raise RuntimeError(
        f"Engine host HTTP ingress daemon did not become ready within {wait_ready_seconds}s "
        f"(spawned pid={spawned_pid}, port={port}, log_file={log_file})"
    )
