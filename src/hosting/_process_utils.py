from __future__ import annotations

import os
import subprocess
import sys
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
