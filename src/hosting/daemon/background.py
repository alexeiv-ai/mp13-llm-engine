"""Background daemon launch helpers."""
from __future__ import annotations

import http.client
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .constants import DEFAULT_DAEMON_PORT, DEFAULT_HTTP_INGRESS_PORT
from .paths import _default_http_pid_file
from .pidfile import DaemonPidFile


def start_daemon_background(
    *,
    port: int = DEFAULT_DAEMON_PORT,
    pid_file: Optional[Path] = None,
    log_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
    wait_ready_seconds: float = 8.0,
) -> Dict[str, Any]:
    """
    Spawn daemon as a detached background process and wait until it is connectable.

    Returns {"pid": N, "port": P, "log_file": ...?} on success.
    Raises RuntimeError if daemon does not become reachable within wait_ready_seconds.
    """
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
    if engines_state_file:
        argv += ["--engines-state-file", str(engines_state_file)]
    if control_state_file:
        argv += ["--control-state-file", str(control_state_file)]

    # Build environment with src dir on PYTHONPATH so connectors package is found
    import os as _os
    env = dict(_os.environ)
    src_root = str(Path(__file__).resolve().parents[2])
    py_path = str(env.get("PYTHONPATH") or "")
    if src_root not in py_path.split(_os.pathsep):
        env["PYTHONPATH"] = src_root if not py_path else f"{src_root}{_os.pathsep}{py_path}"

    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
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
    pid_info = DaemonPidFile(pid_file)
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
            if str(pong) != "pong":
                continue
            info = pid_info.read() or {}
            out: Dict[str, Any] = {"pid": int(info.get("pid") or spawned_pid), "port": actual_port}
            if log_file:
                out["log_file"] = str(log_file)
            return out
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
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
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
    if engines_state_file:
        argv += ["--engines-state-file", str(engines_state_file)]
    if control_state_file:
        argv += ["--control-state-file", str(control_state_file)]

    import os as _os

    env = dict(_os.environ)
    src_root = str(Path(__file__).resolve().parents[2])
    py_path = str(env.get("PYTHONPATH") or "")
    if src_root not in py_path.split(_os.pathsep):
        env["PYTHONPATH"] = src_root if not py_path else f"{src_root}{_os.pathsep}{py_path}"

    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
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
