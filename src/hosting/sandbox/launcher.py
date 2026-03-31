from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .policy import WorkerSandboxPolicy


@dataclass
class WorkerLaunchRequest:
    engine_id: str
    command: List[str]
    cwd: Optional[Path]
    env: Dict[str, str]
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
    log_fp = open(req.log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603,S607
            list(req.command),
            cwd=str(req.cwd) if req.cwd else None,
            env=dict(req.env),
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            close_fds=not req.sandbox_policy.process.inherit_parent_handles,
        )
    finally:
        log_fp.close()
    return WorkerLaunchResult(
        pid=int(proc.pid),
        command=list(req.command),
        persisted_env=dict(req.env),
        runtime={
            "platform": "windows" if os.name == "nt" else "posix",
            "mode": "plain_subprocess",
            "sandbox_enabled": bool(req.sandbox_policy.enabled),
            "inherit_parent_handles": bool(req.sandbox_policy.process.inherit_parent_handles),
            "close_fds": not req.sandbox_policy.process.inherit_parent_handles,
        },
    )


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
    return _normal_launch(req)
