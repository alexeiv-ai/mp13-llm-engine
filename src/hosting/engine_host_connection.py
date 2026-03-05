"""
Persistent connection strategies for EngineHostControlChannel.

Two implementations sharing a common BaseConnection interface:

- LocalSocketConnection: direct TCP socket to 127.0.0.1:<port> (local daemon)
- SSHRelayConnection: persistent SSH subprocess running --relay on the remote host

Both implement:
    invoke(cmd, payload) -> Any   -- send command, return result or raise
    is_alive() -> bool            -- cheap liveness check via __ping__
    close()                       -- tear down connection
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectionError(Exception):  # noqa: A001
    """Raised when a connection to the daemon cannot be established or is lost."""


class BaseConnection:
    def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError

    def is_alive(self) -> bool:
        return False

    def close(self) -> None:
        pass


class LocalSocketConnection(BaseConnection):
    """
    Direct TCP connection to local daemon on 127.0.0.1:<port>.

    Thread-safe: uses a lock so concurrent callers are serialized.
    Reconnects automatically on socket failure (up to max_reconnect_attempts).

    Usage::

        conn = LocalSocketConnection(port=19876)
        result = conn.invoke("discover-running", {})
        conn.close()
    """

    def __init__(
        self,
        *,
        port: int,
        timeout: float = 15.0,
        max_reconnect_attempts: int = 3,
    ):
        self._port = int(port)
        self._timeout = float(timeout or 15.0)
        self._max_reconnect = max(1, int(max_reconnect_attempts or 3))
        self._sock: Optional[socket.socket] = None
        self._file: Optional[Any] = None  # socket.makefile("rb")
        self._seq = 0
        self._lock = threading.Lock()

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._file = None
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self._timeout)
        s.connect(("127.0.0.1", self._port))
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self._sock = s
        self._file = s.makefile("rb")

    def _readline(self) -> str:
        if self._file is None:
            raise ConnectionError("Not connected")
        line = self._file.readline()
        if not line:
            raise ConnectionError("Daemon closed connection")
        return line.decode("utf-8", errors="replace").strip()

    def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        with self._lock:
            self._seq += 1
            seq = self._seq
            request = json.dumps({"seq": seq, "cmd": cmd, "payload": dict(payload or {})}, ensure_ascii=False) + "\n"
            last_exc: Optional[Exception] = None
            for attempt in range(self._max_reconnect):
                try:
                    if self._sock is None:
                        self._connect()
                    assert self._sock is not None
                    self._sock.sendall(request.encode("utf-8"))
                    raw = self._readline()
                    resp = json.loads(raw)
                    if not resp.get("ok"):
                        raise RuntimeError(str(resp.get("error") or f"daemon command '{cmd}' failed"))
                    return resp.get("result")
                except (OSError, BrokenPipeError, ConnectionResetError, ConnectionError) as exc:
                    last_exc = exc
                    self._sock = None
                    self._file = None
                    if attempt < self._max_reconnect - 1:
                        time.sleep(0.2 * (attempt + 1))
            raise ConnectionError(
                f"Failed to reach local daemon on port {self._port}: {last_exc}"
            ) from last_exc

    def is_alive(self) -> bool:
        try:
            result = self.invoke("__ping__")
            return result == "pong"
        except Exception:
            return False

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
                self._file = None


class SSHRelayConnection(BaseConnection):
    """
    Persistent SSH subprocess running `engine_host_cli --relay` on the remote host.

    The relay process on the remote end connects to the remote daemon's TCP socket
    and bridges stdin/stdout to it.  SSH provides the encrypted, authenticated channel.

    Usage::

        conn = SSHRelayConnection(
            ssh_target="user@host",
            ssh_key="/path/to/id_rsa",
        )
        result = conn.invoke("discover-running", {})
        conn.close()
    """

    def __init__(
        self,
        *,
        ssh_target: str,
        ssh_key: Optional[str] = None,
        remote_cmd: str = "python -m hosting.engine_host_cli --relay",
        timeout: float = 15.0,
        max_reconnect_attempts: int = 3,
        known_hosts_line: Optional[str] = None,
    ):
        self._ssh_target = str(ssh_target)
        self._ssh_key = str(ssh_key or "").strip() or None
        self._remote_cmd = str(remote_cmd or "python -m hosting.engine_host_cli --relay")
        self._timeout = float(timeout or 15.0)
        self._max_reconnect = max(1, int(max_reconnect_attempts or 3))
        self._known_hosts_line: Optional[str] = str(known_hosts_line or "").strip() or None
        self._known_hosts_tmpfile: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._seq = 0
        self._lock = threading.Lock()

    def _build_ssh_argv(self) -> List[str]:
        argv: List[str] = [
            "ssh",
            "-T",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={max(5, int(self._timeout))}",
        ]
        if self._known_hosts_line:
            # Write temp known_hosts file for strict checking
            try:
                fd, tmppath = tempfile.mkstemp(prefix="mp13_kh_", suffix=".txt")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(self._known_hosts_line + "\n")
                self._known_hosts_tmpfile = tmppath
                argv += [
                    "-o", "StrictHostKeyChecking=yes",
                    "-o", f"UserKnownHostsFile={tmppath}",
                ]
            except Exception:
                # Fallback: accept-new if temp file creation fails
                argv += ["-o", "StrictHostKeyChecking=accept-new"]
        else:
            argv += ["-o", "StrictHostKeyChecking=accept-new"]
        if self._ssh_key:
            argv += ["-i", self._ssh_key]
        argv.append(self._ssh_target)
        argv += self._remote_cmd.split()
        return argv

    def _spawn(self) -> None:
        argv = self._build_ssh_argv()
        self._proc = subprocess.Popen(  # noqa: S603
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _close_proc(self) -> None:
        p = self._proc
        self._proc = None
        if p is None:
            return
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass
        # Clean up temp known_hosts file
        tmpfile = self._known_hosts_tmpfile
        self._known_hosts_tmpfile = None
        if tmpfile:
            try:
                os.unlink(tmpfile)
            except Exception:
                pass

    def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        with self._lock:
            self._seq += 1
            seq = self._seq
            request = json.dumps({"seq": seq, "cmd": cmd, "payload": dict(payload or {})}, ensure_ascii=False) + "\n"
            last_exc: Optional[Exception] = None
            for attempt in range(self._max_reconnect):
                try:
                    if self._proc is None or self._proc.poll() is not None:
                        self._close_proc()
                        self._spawn()
                    assert self._proc and self._proc.stdin and self._proc.stdout
                    self._proc.stdin.write(request.encode("utf-8"))
                    self._proc.stdin.flush()
                    line = self._proc.stdout.readline()
                    if not line:
                        raise ConnectionError("SSH relay process closed stdout")
                    raw = line.decode("utf-8", errors="replace").strip()
                    resp = json.loads(raw)
                    if not resp.get("ok"):
                        raise RuntimeError(str(resp.get("error") or f"relay command '{cmd}' failed"))
                    return resp.get("result")
                except (OSError, BrokenPipeError, ConnectionError, AssertionError) as exc:
                    last_exc = exc
                    self._close_proc()
                    if attempt < self._max_reconnect - 1:
                        time.sleep(0.3 * (attempt + 1))
            raise ConnectionError(
                f"SSH relay failed for {self._ssh_target}: {last_exc}"
            ) from last_exc

    def is_alive(self) -> bool:
        try:
            result = self.invoke("__ping__")
            return result == "pong"
        except Exception:
            return False

    def close(self) -> None:
        with self._lock:
            self._close_proc()
