from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict


WINDOWS_DETACHED_PROCESS = 0x00000008
WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200
WINDOWS_CREATE_NO_WINDOW = 0x08000000
WINDOWS_SW_HIDE = 0


def pid_alive(pid: int) -> bool:
    try:
        p = int(pid or 0)
    except Exception:
        return False
    if p <= 0:
        return False
    if p == os.getpid():
        return True
    if sys.platform == "win32":
        try:
            return _pid_alive_windows(p)
        except SystemError:
            return True
        except Exception:
            return False
    try:
        os.kill(p, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, SystemError):
        return True
    except Exception:
        return False


def _pid_alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE

    get_exit_code_process = kernel32.GetExitCodeProcess
    get_exit_code_process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_exit_code_process.restype = wintypes.BOOL

    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        err = ctypes.get_last_error()
        return err == ERROR_ACCESS_DENIED
    try:
        exit_code = wintypes.DWORD()
        if not get_exit_code_process(handle, ctypes.byref(exit_code)):
            err = ctypes.get_last_error()
            return err == ERROR_ACCESS_DENIED
        return int(exit_code.value) == STILL_ACTIVE
    finally:
        close_handle(handle)


def hidden_subprocess_kwargs(
    *,
    detached: bool = False,
    new_process_group: bool = False,
) -> Dict[str, Any]:
    """
    Return Popen/run kwargs that prevent transient console windows on Windows.

    CREATE_NO_WINDOW is normally enough for console children, but pairing it
    with STARTF_USESHOWWINDOW/SW_HIDE covers launch paths that briefly create a
    window before honoring creation flags.
    """
    if sys.platform != "win32":
        return {}
    flags = WINDOWS_CREATE_NO_WINDOW
    if detached:
        flags |= WINDOWS_DETACHED_PROCESS
    if new_process_group:
        flags |= WINDOWS_CREATE_NEW_PROCESS_GROUP
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = WINDOWS_SW_HIDE
    return {
        "creationflags": flags,
        "startupinfo": startupinfo,
    }


def _child_pids_posix(pid: int) -> list[int]:
    try:
        proc = subprocess.run(  # noqa: S603
            ["ps", "-eo", "pid=,ppid="],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except Exception:
        return []
    children_by_parent: dict[int, list[int]] = {}
    for line in (proc.stdout or "").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            child = int(parts[0])
            parent = int(parts[1])
        except Exception:
            continue
        children_by_parent.setdefault(parent, []).append(child)
    out: list[int] = []
    stack = list(children_by_parent.get(int(pid), []))
    while stack:
        child = stack.pop()
        if child in out:
            continue
        out.append(child)
        stack.extend(children_by_parent.get(child, []))
    return out


def terminate_process_tree(pid: int, *, timeout_seconds: float = 8.0) -> Dict[str, Any]:
    """
    Best-effort terminate of a process and descendants.

    On Windows this uses taskkill /T so venv launcher processes do not leave the
    real Python worker behind. On POSIX it walks the process table and signals
    descendants before the root PID.
    """
    root = int(pid or 0)
    if root <= 0:
        return {"pid": root, "status": "invalid_pid", "alive": False, "children": []}
    if not pid_alive(root):
        return {"pid": root, "status": "already_stopped", "alive": False, "children": []}

    deadline = time.time() + max(0.1, float(timeout_seconds or 8.0))
    children: list[int] = []
    errors: list[str] = []

    if sys.platform == "win32":
        try:
            proc = subprocess.run(  # noqa: S603
                ["taskkill", "/PID", str(root), "/T", "/F"],
                text=True,
                capture_output=True,
                timeout=max(1.0, float(timeout_seconds or 8.0)),
                check=False,
                **hidden_subprocess_kwargs(),
            )
            if proc.returncode not in (0, 128):
                detail = (proc.stderr or proc.stdout or "").strip()
                if detail:
                    errors.append(detail)
        except Exception as exc:
            errors.append(str(exc))
    else:
        children = _child_pids_posix(root)
        for target in list(reversed(children)) + [root]:
            try:
                os.kill(target, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:
                errors.append(f"SIGTERM {target}: {exc}")
        while time.time() < deadline:
            live = [p for p in [root] + children if pid_alive(p)]
            if not live:
                break
            time.sleep(0.1)
        for target in list(reversed(children)) + [root]:
            if not pid_alive(target):
                continue
            try:
                os.kill(target, getattr(signal, "SIGKILL", signal.SIGTERM))
            except ProcessLookupError:
                pass
            except Exception as exc:
                errors.append(f"SIGKILL {target}: {exc}")

    while time.time() < deadline:
        if not pid_alive(root):
            break
        time.sleep(0.1)
    alive = pid_alive(root)
    return {
        "pid": root,
        "status": "stopped" if not alive else "stop_failed",
        "alive": alive,
        "children": children,
        "errors": errors,
    }
