from __future__ import annotations

import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._process_utils import hidden_subprocess_kwargs
from .policy import WorkerSandboxPolicy

_NORMAL_JOB_HANDLES: Dict[int, int] = {}
_POSIX_LAUNCH_LOCK = threading.Lock()
_POSIX_LAUNCH_QUEUE: Optional[queue.Queue[tuple["WorkerLaunchRequest", queue.Queue[tuple[str, Any]]]]] = None
_POSIX_LAUNCH_THREAD: Optional[threading.Thread] = None


@dataclass
class WorkerLaunchRequest:
    engine_id: str
    command: List[str]
    cwd: Optional[Path]
    env: Dict[str, str]
    # TBD: consider adding a daemon-owned real-time log sink here so worker
    # stdout/stderr can be streamed through control APIs without tailing files.
    log_path: Path
    sandbox_policy: WorkerSandboxPolicy


@dataclass
class WorkerLaunchResult:
    pid: int
    command: List[str]
    persisted_env: Dict[str, str]
    runtime: Dict[str, Any]


def _normal_launch(req: WorkerLaunchRequest) -> WorkerLaunchResult:
    req.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(req.log_path, "wb")
    kwargs = hidden_subprocess_kwargs()
    try:
        proc = subprocess.Popen(  # noqa: S603,S607
            list(req.command),
            cwd=str(req.cwd) if req.cwd else None,
            env=dict(req.env),
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            close_fds=not req.sandbox_policy.process.inherit_parent_handles,
            **kwargs
        )
    finally:
        log_fp.close()
    runtime: Dict[str, Any] = {
        "platform": "windows" if os.name == "nt" else "posix",
        "mode": "plain_subprocess",
        "sandbox_enabled": bool(req.sandbox_policy.enabled),
        "inherit_parent_handles": bool(req.sandbox_policy.process.inherit_parent_handles),
        "close_fds": not req.sandbox_policy.process.inherit_parent_handles,
    }
    if os.name == "nt":
        runtime.update(_attach_windows_kill_on_close_job(proc))
    return WorkerLaunchResult(
        pid=int(proc.pid),
        command=list(req.command),
        persisted_env=dict(req.env),
        runtime=runtime,
    )


def _persistent_posix_launch_loop(
    request_queue: queue.Queue[tuple[WorkerLaunchRequest, queue.Queue[tuple[str, Any]]]],
) -> None:
    """Launch workers from a task that lives for the host process lifetime.

    Linux PR_SET_PDEATHSIG is tied to the parent *thread* that called fork,
    rather than merely to its process. Hosted operations run on short-lived
    threads, so launching directly from them causes otherwise healthy workers
    to receive SIGTERM as soon as the operation completes.
    """
    while True:
        req, reply = request_queue.get()
        try:
            reply.put(("ok", _normal_launch(req)))
        except Exception as exc:
            reply.put(("error", exc))


def _launch_from_persistent_posix_thread(req: WorkerLaunchRequest) -> WorkerLaunchResult:
    global _POSIX_LAUNCH_QUEUE, _POSIX_LAUNCH_THREAD

    with _POSIX_LAUNCH_LOCK:
        if _POSIX_LAUNCH_THREAD is None or not _POSIX_LAUNCH_THREAD.is_alive():
            # Recreate both objects after a fork: inherited Thread objects are
            # not alive in the child and their old queue has no consumer.
            request_queue: queue.Queue[tuple[WorkerLaunchRequest, queue.Queue[tuple[str, Any]]]] = queue.Queue()
            launcher = threading.Thread(
                target=_persistent_posix_launch_loop,
                args=(request_queue,),
                name="mp13-posix-worker-launcher",
                daemon=True,
            )
            launcher.start()
            _POSIX_LAUNCH_QUEUE = request_queue
            _POSIX_LAUNCH_THREAD = launcher
        request_queue = _POSIX_LAUNCH_QUEUE

    if request_queue is None:  # pragma: no cover - guarded by initialization above
        raise RuntimeError("POSIX worker launcher queue was not initialized")
    reply: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
    request_queue.put((req, reply))
    status, value = reply.get()
    if status == "error":
        raise value
    result = value
    result.runtime["parent_task"] = "persistent_worker_launcher"
    return result


def _attach_windows_kill_on_close_job(proc: subprocess.Popen[Any]) -> Dict[str, Any]:
    try:
        from ctypes import wintypes

        from .windows import _create_job_object, kernel32

        process_handle = getattr(proc, "_handle", None)
        if not process_handle:
            return {"job_object": False, "job_object_error": "missing_process_handle"}
        hjob = _create_job_object()
        if not kernel32.AssignProcessToJobObject(hjob, wintypes.HANDLE(int(process_handle))):
            err = getattr(__import__("ctypes"), "get_last_error")()
            kernel32.CloseHandle(hjob)
            return {"job_object": False, "job_object_error": f"AssignProcessToJobObject failed with WinError {err}"}
        _NORMAL_JOB_HANDLES[int(proc.pid)] = int(hjob)
        return {"job_object": True, "job_object_mode": "kill_on_close"}
    except Exception as exc:
        return {"job_object": False, "job_object_error": str(exc)}


def close_worker_job(pid: int) -> Dict[str, Any]:
    if os.name != "nt":
        return {"pid": int(pid or 0), "job_object": False, "closed": False, "reason": "not_windows"}
    target = int(pid or 0)
    handle_value = _NORMAL_JOB_HANDLES.pop(target, None)
    if not handle_value:
        return {"pid": target, "job_object": False, "closed": False, "reason": "not_found"}
    try:
        from ctypes import wintypes

        from .windows import kernel32

        ok = bool(kernel32.CloseHandle(wintypes.HANDLE(int(handle_value))))
        return {"pid": target, "job_object": True, "closed": ok}
    except Exception as exc:
        return {"pid": target, "job_object": True, "closed": False, "error": str(exc)}


def launch_worker_process(req: WorkerLaunchRequest) -> WorkerLaunchResult:
    policy = req.sandbox_policy
    if os.name == "nt" and policy.enabled:
        from .windows import launch_restricted_worker

        launched = launch_restricted_worker(
            argv=list(req.command),
            cwd=req.cwd,
            env=dict(req.env),
            log_path=req.log_path,
            integrity_level=policy.windows.integrity_level,
            use_job_object=bool(policy.windows.job_object),
        )
        return WorkerLaunchResult(
            pid=int(launched.pid),
            command=list(req.command),
            persisted_env=dict(req.env),
            runtime=dict(launched.runtime),
        )
    if os.name != "nt":
        return _launch_from_persistent_posix_thread(req)
    return _normal_launch(req)
