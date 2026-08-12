from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import pytest

if os.name != "nt":
    pytest.skip("Windows-only sandbox integration tests", allow_module_level=True)

from hosting.sandbox import WorkerSandboxPolicy
from hosting.sandbox.launcher import WorkerLaunchRequest, launch_worker_process
from hosting.sandbox.windows import terminate_process, wait_for_process_exit


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows-only sandbox integration tests")


def _sandbox_policy() -> WorkerSandboxPolicy:
    return WorkerSandboxPolicy.from_mapping(
        {
            "sandbox": {
                "enabled": True,
                "platform_policy": {
                    "windows": {
                        "restricted_token": True,
                        "integrity_level": "low",
                        "job_object": True,
                    }
                },
            }
        }
    )


def _probe_pipe_hello(*, pipe: str, auth: str, timeout_seconds: float = 30.0) -> tuple[object, object]:
    resp = None
    last_exc = None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            conn = Client(address=pipe, family="AF_PIPE", authkey=auth.encode("utf-8"))
            conn.send({"kind": "hello"})
            resp = conn.recv()
            conn.close()
            break
        except Exception as exc:  # pragma: no cover - retry path is environment timing dependent
            last_exc = exc
            time.sleep(0.1)
    return resp, last_exc


def _shutdown_pipe(*, pipe: str, auth: str) -> None:
    try:
        conn = Client(address=pipe, family="AF_PIPE", authkey=auth.encode("utf-8"))
        conn.send({"kind": "shutdown"})
        conn.recv()
        conn.close()
    except Exception:
        pass


def _wait_or_kill_subprocess(proc: subprocess.Popen[bytes], timeout_seconds: float) -> int:
    try:
        return int(proc.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        proc.kill()
        return int(proc.wait(timeout=5.0))


def _wait_or_kill_pid(pid: int, timeout_seconds: float) -> int:
    code = wait_for_process_exit(pid, timeout_seconds)
    if code is not None:
        return int(code)
    terminate_process(pid, exit_code=1)
    code = wait_for_process_exit(pid, 5.0)
    return int(code if code is not None else 1)


def test_windows_low_il_worker_cannot_modify_medium_integrity_file(tmp_path: Path) -> None:
    target = tmp_path / "protected.txt"
    target.write_text("orig", encoding="utf-8")
    cmd = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "p = Path(sys.argv[1])",
                "try:",
                "    p.write_text('changed', encoding='utf-8')",
                "    raise SystemExit(0)",
                "except Exception:",
                "    raise SystemExit(13)",
            ]
        ),
        str(target),
    ]
    req = WorkerLaunchRequest(
        engine_id="writecheck",
        command=cmd,
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "writecheck.log",
        sandbox_policy=_sandbox_policy(),
    )
    out = launch_worker_process(req)
    code = wait_for_process_exit(out.pid, 5.0)

    assert code == 13
    assert target.read_text(encoding="utf-8") == "orig"


def test_windows_low_il_worker_serves_named_pipe_rpc_with_minimal_helper(tmp_path: Path) -> None:
    pipe = r"\\.\pipe\mp13-sandbox-test-" + secrets.token_hex(8)
    auth = "tok-" + secrets.token_hex(8)
    script = (
        "import sys;"
        "print('low-il-helper-ready', flush=True);"
        "from multiprocessing.connection import Listener;"
        "listener = Listener(address=sys.argv[1], family='AF_PIPE', authkey=sys.argv[2].encode('utf-8'));"
        "conn = listener.accept();"
        "req = conn.recv();"
        "conn.send({'status':'ok','echo':req});"
        "conn.close();"
        "listener.close();"
    )
    req = WorkerLaunchRequest(
        engine_id="ipcmini",
        command=[sys.executable, "-c", script, pipe, auth],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "ipcmini.log",
        sandbox_policy=_sandbox_policy(),
    )
    out = launch_worker_process(req)
    resp = None
    last_exc = None
    for _ in range(50):
        try:
            conn = Client(address=pipe, family="AF_PIPE", authkey=auth.encode("utf-8"))
            conn.send({"kind": "hello"})
            resp = conn.recv()
            conn.close()
            break
        except Exception as exc:  # pragma: no cover - retry path is environment timing dependent
            last_exc = exc
            time.sleep(0.1)
    code = wait_for_process_exit(out.pid, 5.0)

    assert resp == {"status": "ok", "echo": {"kind": "hello"}}, repr(last_exc)
    assert code == 0
    assert "low-il-helper-ready" in (tmp_path / "ipcmini.log").read_text(encoding="utf-8")


@pytest.mark.skipif(
    str(os.environ.get("MP13_RUN_HOSTING_SANDBOX_SENSITIVE") or "").strip() not in {"1", "true", "yes"},
    reason="Enable MP13_RUN_HOSTING_SANDBOX_SENSITIVE=1 for environment-sensitive engine_worker_ipc validation",
)
def test_windows_low_il_engine_worker_ipc_serves_hello_over_named_pipe(tmp_path: Path) -> None:
    pipe = r"\\.\pipe\mp13-sandbox-engine-worker-" + secrets.token_hex(8)
    auth = "tok-" + secrets.token_hex(8)
    model_path = str(os.environ.get("MP13_SANDBOX_ENGINE_MODEL_PATH") or "").strip()
    config_path = str(os.environ.get("MP13_SANDBOX_ENGINE_CONFIG_PATH") or "").strip()
    if not (model_path or config_path):
        pytest.skip(
            "Set MP13_SANDBOX_ENGINE_MODEL_PATH or MP13_SANDBOX_ENGINE_CONFIG_PATH "
            "to run the real engine_worker_ipc Low-IL validation"
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["MP13_ENGINE_HOST_TOKEN"] = auth
    env["MP13_ENGINE_HOST_TOKEN_HEADER"] = "X-MP13-Host-Token"
    if model_path:
        env["MP13_MODEL_PATH"] = model_path
    if config_path:
        env["MP13_ENGINE_CONFIG_PATH"] = config_path
    base_cmd = [sys.executable, "-m", "hosting.engine_worker_ipc", "--ipc-family", "AF_PIPE", "--ipc-address"]

    # Preflight unsandboxed worker startup first. If the real worker cannot
    # start in this environment, the test should not report a sandbox failure.
    plain_pipe = r"\\.\pipe\mp13-plain-engine-worker-" + secrets.token_hex(8)
    plain_log = tmp_path / "engineworker_plain.log"
    with open(plain_log, "ab") as plain_fp:
        proc = subprocess.Popen(  # noqa: S603,S607
            base_cmd + [plain_pipe],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=plain_fp,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    plain_resp, plain_last_exc = _probe_pipe_hello(pipe=plain_pipe, auth=auth)
    _shutdown_pipe(pipe=plain_pipe, auth=auth)
    plain_code = _wait_or_kill_subprocess(proc, 5.0)
    if not isinstance(plain_resp, dict):
        log_preview = plain_log.read_text(encoding="utf-8", errors="replace")[-2000:] if plain_log.exists() else ""
        pytest.skip(
            "Unsandboxed engine_worker_ipc could not start in this environment; "
            f"exit_code={plain_code!r}, last_exc={plain_last_exc!r}, log_tail={log_preview!r}"
        )

    req = WorkerLaunchRequest(
        engine_id="engineworker",
        command=base_cmd + [pipe],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        log_path=tmp_path / "engineworker.log",
        sandbox_policy=_sandbox_policy(),
    )
    out = launch_worker_process(req)
    resp, last_exc = _probe_pipe_hello(pipe=pipe, auth=auth)
    _shutdown_pipe(pipe=pipe, auth=auth)
    code = _wait_or_kill_pid(out.pid, 5.0)

    assert isinstance(resp, dict), repr(last_exc)
    assert str(resp.get("status") or "") == "ok"
    assert str(resp.get("contract") or "").startswith("mp13.worker.rpc.")
    assert code == 0
