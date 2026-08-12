from __future__ import annotations

import ctypes
import msvcrt
import os
import subprocess
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

TOKEN_DUPLICATE = 0x0002
TOKEN_ASSIGN_PRIMARY = 0x0001
TOKEN_QUERY = 0x0008
TOKEN_ADJUST_DEFAULT = 0x0080
TOKEN_ADJUST_SESSIONID = 0x0100
TOKEN_ALL_ACCESS_NEEDED = TOKEN_DUPLICATE | TOKEN_ASSIGN_PRIMARY | TOKEN_QUERY | TOKEN_ADJUST_DEFAULT | TOKEN_ADJUST_SESSIONID

DISABLE_MAX_PRIVILEGE = 0x1
SANDBOX_INERT = 0x2
LUA_TOKEN = 0x4
CREATE_NEW_CONSOLE = 0x00000010
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
STARTF_USESHOWWINDOW = 0x00000001
STARTF_USESTDHANDLES = 0x00000100
SW_HIDE = 0
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102

TokenIntegrityLevel = 25
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

SECURITY_MANDATORY_UNTRUSTED_RID = 0x00000000
SECURITY_MANDATORY_LOW_RID = 0x00001000
SECURITY_MANDATORY_MEDIUM_RID = 0x00002000


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Sid", wintypes.LPVOID),
        ("Attributes", wintypes.DWORD),
    ]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class LARGE_INTEGER(ctypes.Structure):
    _fields_ = [("QuadPart", ctypes.c_longlong)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", LARGE_INTEGER),
        ("PerJobUserTimeLimit", LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


@dataclass
class WindowsLaunchResult:
    pid: int
    runtime: Dict[str, Any]


_JOB_HANDLES: Dict[int, int] = {}
_PROCESS_HANDLES: Dict[int, int] = {}


kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
kernel32.LocalFree.restype = wintypes.HLOCAL
kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
kernel32.CreateJobObjectW.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
kernel32.TerminateProcess.restype = wintypes.BOOL
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel32.GetExitCodeProcess.restype = wintypes.BOOL
kernel32.SetInformationJobObject.argtypes = [
    wintypes.HANDLE,
    wintypes.INT,
    wintypes.LPVOID,
    wintypes.DWORD,
]
kernel32.SetInformationJobObject.restype = wintypes.BOOL
kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
advapi32.OpenProcessToken.restype = wintypes.BOOL
advapi32.CreateRestrictedToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.HANDLE),
]
advapi32.CreateRestrictedToken.restype = wintypes.BOOL
advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.LPVOID)]
advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL
advapi32.GetLengthSid.argtypes = [wintypes.LPVOID]
advapi32.GetLengthSid.restype = wintypes.DWORD
advapi32.SetTokenInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD]
advapi32.SetTokenInformation.restype = wintypes.BOOL
advapi32.CreateProcessAsUserW.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.BOOL,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.LPCWSTR,
    ctypes.POINTER(STARTUPINFOW),
    ctypes.POINTER(PROCESS_INFORMATION),
]
advapi32.CreateProcessAsUserW.restype = wintypes.BOOL


def _raise_last_error(prefix: str) -> None:
    err = ctypes.get_last_error()
    raise OSError(err, f"{prefix} failed with WinError {err}")


def _build_env_block(env: Dict[str, str]) -> str:
    items = [f"{k}={v}" for k, v in sorted({str(k): str(v) for k, v in env.items()}.items())]
    return "\x00".join(items) + "\x00\x00"


def _mandatory_level_rid(level: str) -> int:
    raw = str(level or "").strip().lower()
    if raw == "untrusted":
        return SECURITY_MANDATORY_UNTRUSTED_RID
    if raw == "medium":
        return SECURITY_MANDATORY_MEDIUM_RID
    return SECURITY_MANDATORY_LOW_RID


def _make_integrity_sid(level: str) -> wintypes.LPVOID:
    sid = wintypes.LPVOID()
    sid_text = f"S-1-16-{_mandatory_level_rid(level)}"
    if not advapi32.ConvertStringSidToSidW(wintypes.LPCWSTR(sid_text), ctypes.byref(sid)):
        _raise_last_error("ConvertStringSidToSidW")
    return sid


def _create_restricted_token(integrity_level: str) -> wintypes.HANDLE:
    process = kernel32.GetCurrentProcess()
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(process, TOKEN_ALL_ACCESS_NEEDED, ctypes.byref(token)):
        _raise_last_error("OpenProcessToken")
    restricted = wintypes.HANDLE()
    try:
        if not advapi32.CreateRestrictedToken(
            token,
            DISABLE_MAX_PRIVILEGE | LUA_TOKEN | SANDBOX_INERT,
            0,
            None,
            0,
            None,
            0,
            None,
            ctypes.byref(restricted),
        ):
            _raise_last_error("CreateRestrictedToken")
    finally:
        kernel32.CloseHandle(token)
    sid = _make_integrity_sid(integrity_level)
    try:
        tml = TOKEN_MANDATORY_LABEL()
        tml.Label.Sid = sid
        tml.Label.Attributes = 0x20
        size = ctypes.sizeof(TOKEN_MANDATORY_LABEL) + advapi32.GetLengthSid(sid)
        if not advapi32.SetTokenInformation(restricted, TokenIntegrityLevel, ctypes.byref(tml), size):
            _raise_last_error("SetTokenInformation(TokenIntegrityLevel)")
    finally:
        kernel32.LocalFree(sid)
    return restricted


def _create_job_object() -> wintypes.HANDLE:
    hjob = kernel32.CreateJobObjectW(None, None)
    if not hjob:
        _raise_last_error("CreateJobObjectW")
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        hjob,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        kernel32.CloseHandle(hjob)
        _raise_last_error("SetInformationJobObject")
    return hjob


def launch_restricted_worker(
    *,
    argv: List[str],
    cwd: Optional[Path],
    env: Dict[str, str],
    log_path: Path,
    integrity_level: str = "low",
    use_job_object: bool = True,
) -> WindowsLaunchResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"")

    restricted = _create_restricted_token(integrity_level)
    try:
        with open(log_path, "ab", buffering=0) as log_fp, open(os.devnull, "rb", buffering=0) as stdin_fp:
            log_handle = msvcrt.get_osfhandle(log_fp.fileno())
            stdin_handle = msvcrt.get_osfhandle(stdin_fp.fileno())
            os.set_handle_inheritable(log_handle, True)
            os.set_handle_inheritable(stdin_handle, True)
            si = STARTUPINFOW()
            si.cb = ctypes.sizeof(si)
            si.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES
            si.wShowWindow = SW_HIDE
            si.hStdInput = wintypes.HANDLE(stdin_handle)
            si.hStdOutput = wintypes.HANDLE(log_handle)
            si.hStdError = wintypes.HANDLE(log_handle)
            pi = PROCESS_INFORMATION()
            env_block = _build_env_block(env)
            command_line = subprocess.list2cmdline(list(argv or []))
            flags = CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW
            created = advapi32.CreateProcessAsUserW(
                restricted,
                None,
                ctypes.c_wchar_p(command_line),
                None,
                None,
                True,
                flags,
                ctypes.c_wchar_p(env_block),
                ctypes.c_wchar_p(str(cwd) if cwd else None),
                ctypes.byref(si),
                ctypes.byref(pi),
            )
        if not created:
            _raise_last_error("CreateProcessAsUserW")
        if use_job_object:
            hjob = _create_job_object()
            if not kernel32.AssignProcessToJobObject(hjob, pi.hProcess):
                kernel32.CloseHandle(hjob)
                _raise_last_error("AssignProcessToJobObject")
            _JOB_HANDLES[int(pi.dwProcessId)] = int(hjob)
        _PROCESS_HANDLES[int(pi.dwProcessId)] = int(pi.hProcess)
        kernel32.CloseHandle(pi.hThread)
        return WindowsLaunchResult(
            pid=int(pi.dwProcessId),
            runtime={
                "platform": "windows",
                "mode": "restricted_token_low_il_job" if use_job_object else "restricted_token_low_il",
                "integrity_level": str(integrity_level),
                "job_object": bool(use_job_object),
                "log_capture": "stdout_stderr",
            },
        )
    finally:
        kernel32.CloseHandle(restricted)


def wait_for_process_exit(pid: int, timeout_seconds: float = 10.0) -> Optional[int]:
    handle_value = _PROCESS_HANDLES.get(int(pid))
    handle = wintypes.HANDLE(handle_value) if handle_value else kernel32.OpenProcess(
        SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        int(pid),
    )
    if not handle:
        _raise_last_error("OpenProcess")
    try:
        rc = kernel32.WaitForSingleObject(handle, max(0, int(float(timeout_seconds) * 1000.0)))
        if rc == WAIT_TIMEOUT:
            return None
        if rc != WAIT_OBJECT_0:
            _raise_last_error("WaitForSingleObject")
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            _raise_last_error("GetExitCodeProcess")
        return int(exit_code.value)
    finally:
        if handle_value:
            _PROCESS_HANDLES.pop(int(pid), None)
        kernel32.CloseHandle(handle)


def terminate_process(pid: int, exit_code: int = 1) -> None:
    handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | 0x0001, False, int(pid))
    if not handle:
        _raise_last_error("OpenProcess")
    try:
        if not kernel32.TerminateProcess(handle, int(exit_code)):
            _raise_last_error("TerminateProcess")
    finally:
        _PROCESS_HANDLES.pop(int(pid), None)
        job_handle = _JOB_HANDLES.pop(int(pid), None)
        if job_handle:
            kernel32.CloseHandle(wintypes.HANDLE(job_handle))
        kernel32.CloseHandle(handle)
