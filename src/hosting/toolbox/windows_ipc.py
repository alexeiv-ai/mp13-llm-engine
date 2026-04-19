"""Windows IPC helpers for hosted toolbox callbacks."""
from __future__ import annotations

import ctypes
import multiprocessing.connection as mp_connection
import os
from ctypes import wintypes
from typing import Any


if os.name == "nt":
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _SDDL_REVISION_1 = 1
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


    class _SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        ]

    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    ]
    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    _kernel32.CreateNamedPipeW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_SECURITY_ATTRIBUTES),
    ]
    _kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
    _kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    _kernel32.LocalFree.restype = wintypes.HLOCAL


def _create_windows_low_integrity_pipe(address: str, *, first: bool = False) -> Any:
    if os.name != "nt":
        raise RuntimeError("windows_pipe_only")
    pipe_name = str(address or "").strip()
    if not pipe_name:
        raise ValueError("pipe_name_required")
    sd = wintypes.LPVOID()
    sd_size = wintypes.DWORD(0)
    if not _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        "S:(ML;;NW;;;LW)",
        _SDDL_REVISION_1,
        ctypes.byref(sd),
        ctypes.byref(sd_size),
    ):
        err = ctypes.get_last_error()
        raise OSError(err, f"ConvertStringSecurityDescriptorToSecurityDescriptorW failed for {pipe_name}")
    try:
        security = _SECURITY_ATTRIBUTES(
            nLength=ctypes.sizeof(_SECURITY_ATTRIBUTES),
            lpSecurityDescriptor=sd,
            bInheritHandle=False,
        )
        flags = mp_connection._winapi.PIPE_ACCESS_DUPLEX | mp_connection._winapi.FILE_FLAG_OVERLAPPED
        if first:
            flags |= mp_connection._winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
        handle = _kernel32.CreateNamedPipeW(
            ctypes.c_wchar_p(pipe_name),
            flags,
            mp_connection._winapi.PIPE_TYPE_MESSAGE
            | mp_connection._winapi.PIPE_READMODE_MESSAGE
            | mp_connection._winapi.PIPE_WAIT,
            mp_connection._winapi.PIPE_UNLIMITED_INSTANCES,
            mp_connection.BUFSIZE,
            mp_connection.BUFSIZE,
            mp_connection._winapi.NMPWAIT_WAIT_FOREVER,
            ctypes.byref(security),
        )
        if not handle or int(handle) == -1:
            err = ctypes.get_last_error()
            raise OSError(err, f"CreateNamedPipeW failed for {pipe_name}")
        return int(handle)
    finally:
        _kernel32.LocalFree(sd)
