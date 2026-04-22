from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hosting.daemon import (
    _secure_path,
    _secure_state_parent_dir,
    _apply_foreground_terminal_disconnect_policy,
    _daemon_local_ipc_endpoint,
    DaemonPidFile,
    EngineHostDaemon,
    start_daemon_background,
    start_http_ingress_background,
)


def test_pid_alive_returns_true_on_system_error(monkeypatch) -> None:
    def _raise_system_error(_pid: int, _sig: int) -> None:
        raise SystemError("simulated_windows_detached_kill_behavior")

    monkeypatch.setattr("hosting._process_utils.sys.platform", "linux")
    monkeypatch.setattr("hosting._process_utils.os.kill", _raise_system_error)
    assert DaemonPidFile._pid_alive(12345) is True


def test_pid_alive_returns_false_on_process_lookup_error(monkeypatch) -> None:
    def _raise_lookup_error(_pid: int, _sig: int) -> None:
        raise ProcessLookupError()

    monkeypatch.setattr("hosting._process_utils.sys.platform", "linux")
    monkeypatch.setattr("hosting._process_utils.os.kill", _raise_lookup_error)
    assert DaemonPidFile._pid_alive(12345) is False


def test_pid_alive_returns_true_on_permission_error(monkeypatch) -> None:
    def _raise_permission_error(_pid: int, _sig: int) -> None:
        raise PermissionError()

    monkeypatch.setattr("hosting._process_utils.sys.platform", "linux")
    monkeypatch.setattr("hosting._process_utils.os.kill", _raise_permission_error)
    assert DaemonPidFile._pid_alive(12345) is True


def test_pid_alive_windows_returns_true_on_system_error(monkeypatch) -> None:
    def _raise_system_error(_pid: int) -> bool:
        raise SystemError("simulated_windows_detached_probe_behavior")

    monkeypatch.setattr("hosting._process_utils.sys.platform", "win32")
    monkeypatch.setattr("hosting._process_utils._pid_alive_windows", _raise_system_error)

    assert DaemonPidFile._pid_alive(12345) is True


def test_start_daemon_background_uses_protocol_ping_for_readiness(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 43210

        def poll(self):
            captured["poll_checked"] = True
            return None

    class _FakePidFile:
        def __init__(self, _path=None):
            return

        def is_alive(self) -> bool:
            return True

        def get_port(self) -> int:
            return 19876

        def read(self):
            return {"pid": 55555, "port": 19876}

    class _FakeConn:
        def __init__(self, *, port: int, timeout: float, max_reconnect_attempts: int):
            captured["port"] = port
            captured["timeout"] = timeout
            captured["max_reconnect_attempts"] = max_reconnect_attempts

        def invoke(self, cmd: str, payload=None):
            captured["cmd"] = cmd
            captured["payload"] = dict(payload or {})
            return "pong"

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(
        "hosting.daemon.background.subprocess.Popen",
        lambda *args, **kwargs: _FakeProc(),
    )
    monkeypatch.setattr("hosting.daemon.background.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.daemon.background.time.sleep", lambda _sec: None)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeConn)

    result = start_daemon_background(port=19876, wait_ready_seconds=1.0)

    assert captured["cmd"] == "__ping__"
    assert captured["payload"] == {}
    assert captured["closed"] is True
    # Windows path uses os.kill(pid, 0) branch; non-Windows uses proc.poll().
    assert captured.get("poll_checked") in {None, True}
    assert result == {"pid": 55555, "port": 19876}


def test_pidfile_read_and_get_port_handle_system_error() -> None:
    class _BadPath:
        def exists(self) -> bool:
            raise SystemError("PurePath.__str__ returned a result with an exception set")

        def read_text(self, encoding: str = "utf-8") -> str:
            return "{}"

    pid = DaemonPidFile()
    pid.path = _BadPath()  # type: ignore[assignment]

    assert pid.read() is None
    assert pid.get_port() is None


def test_pidfile_write_persists_payload(tmp_path: Path) -> None:
    pid = DaemonPidFile(tmp_path / "daemon.pid")
    pid.write(
        pid=1234,
        port=19876,
        shutdown_token="tok",
        transport="local_ipc",
        ipc_family="AF_UNIX",
        ipc_address=str(tmp_path / "daemon.sock"),
    )

    raw = json.loads((tmp_path / "daemon.pid").read_text(encoding="utf-8"))
    assert int(raw["pid"]) == 1234
    assert int(raw["port"]) == 19876
    assert str(raw["shutdown_token"]) == "tok"
    assert str(raw["transport"]) == "local_ipc"
    assert str(raw["ipc_family"]) == "AF_UNIX"


def test_daemon_local_ipc_endpoint_uses_unix_socket_on_posix(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("hosting.daemon.paths.os.name", "posix")

    endpoint = _daemon_local_ipc_endpoint(tmp_path / "daemon.pid")

    assert endpoint["transport"] == "local_ipc"
    assert endpoint["family"] == "AF_UNIX"
    assert str(endpoint["address"]).endswith(".sock")
    assert str(tmp_path.resolve()) in str(endpoint["address"])


def test_daemon_local_ipc_endpoint_uses_named_pipe_on_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("hosting.daemon.paths.os.name", "nt")

    endpoint = _daemon_local_ipc_endpoint(tmp_path / "daemon.pid")

    assert endpoint["transport"] == "local_ipc"
    assert endpoint["family"] == "AF_PIPE"
    assert str(endpoint["address"]).startswith(r"\\.\pipe\mp13-host-daemon-")


def test_secure_state_parent_dir_posix_applies_0700(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr("hosting.daemon.security.os.name", "posix")
    monkeypatch.setattr(
        "hosting.daemon.security.os.chmod",
        lambda path, mode: calls.append((str(path), int(mode))),
    )

    target = tmp_path / "state" / "daemon.pid"
    _secure_state_parent_dir(target)

    assert target.parent.exists()
    assert calls == [(str(target.parent), 0o700)]


def test_secure_path_windows_uses_icacls(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(argv, **_kwargs):
        captured["argv"] = list(argv)
        return _Proc()

    monkeypatch.setattr("hosting.daemon.security.os.name", "nt")
    monkeypatch.setattr("hosting.daemon.security._current_windows_account_name", lambda: "DOMAIN\\user")
    monkeypatch.setattr("hosting.daemon.security.subprocess.run", _fake_run)

    target = tmp_path / "daemon.pid"
    target.write_text("x", encoding="utf-8")
    _secure_path(target)

    argv = list(captured.get("argv") or [])
    assert argv[:3] == ["icacls", str(target), "/inheritance:r"]
    assert "DOMAIN\\user:F" in argv
    assert "SYSTEM:F" in argv
    assert "Administrators:F" in argv


def test_start_daemon_background_retries_after_pidfile_system_error(monkeypatch) -> None:
    captured: dict[str, object] = {"get_port_calls": 0}

    class _FakeProc:
        pid = 43210

        def poll(self):
            captured["poll_checked"] = True
            return None

    class _FakePidFile:
        def __init__(self, _path=None):
            return

        def is_alive(self) -> bool:
            return True

        def get_port(self) -> int:
            captured["get_port_calls"] = int(captured["get_port_calls"] or 0) + 1
            if int(captured["get_port_calls"] or 0) == 1:
                raise SystemError("PurePath.__str__ returned a result with an exception set")
            return 19876

        def read(self):
            return {"pid": 55555, "port": 19876}

    class _FakeConn:
        def __init__(self, *, port: int, timeout: float, max_reconnect_attempts: int):
            return

        def invoke(self, cmd: str, payload=None):
            return "pong"

        def close(self) -> None:
            return

    monkeypatch.setattr(
        "hosting.daemon.background.subprocess.Popen",
        lambda *args, **kwargs: _FakeProc(),
    )
    monkeypatch.setattr("hosting.daemon.background.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.daemon.background.time.sleep", lambda _sec: None)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeConn)

    result = start_daemon_background(port=19876, wait_ready_seconds=1.0)

    # Windows path uses os.kill(pid, 0) branch; non-Windows uses proc.poll().
    assert captured.get("poll_checked") in {None, True}
    assert int(captured["get_port_calls"] or 0) >= 2
    assert result == {"pid": 55555, "port": 19876}


def test_start_http_ingress_background_checks_returncode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 98765

        def poll(self):
            captured["poll_checked"] = True
            return None

    class _FakePidFile:
        def __init__(self, _path=None):
            return

        def is_alive(self) -> bool:
            return True

        def get_port(self) -> int:
            return 19877

        def read(self):
            return {"pid": 77777, "port": 19877}

    class _FakeHTTPConn:
        def __init__(self, host: str, port: int, timeout: float):
            self.host = host
            self.port = port
            self.timeout = timeout

        def request(self, method: str, path: str) -> None:
            return

        def getresponse(self):
            class _Resp:
                status = 200

                def read(self):
                    return b"ok"

            return _Resp()

        def close(self) -> None:
            return

    monkeypatch.setattr(
        "hosting.daemon.background.subprocess.Popen",
        lambda *args, **kwargs: _FakeProc(),
    )
    monkeypatch.setattr("hosting.daemon.background.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.daemon.background.time.sleep", lambda _sec: None)
    monkeypatch.setattr("hosting.daemon.background.http.client.HTTPConnection", _FakeHTTPConn)

    result = start_http_ingress_background(port=19877, wait_ready_seconds=1.0)

    # Windows path uses os.kill(pid, 0) branch; non-Windows uses proc.poll().
    assert captured.get("poll_checked") in {None, True}
    assert result == {"pid": 77777, "port": 19877}


def test_start_daemon_background_omits_log_flag_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {"argv": []}

    class _FakeProc:
        pid = 54321

        def poll(self):
            return None

    class _FakePidFile:
        def __init__(self, _path=None):
            return

        def is_alive(self) -> bool:
            return True

        def get_port(self) -> int:
            return 19876

        def read(self):
            return {"pid": 54321, "port": 19876}

    class _FakeConn:
        def __init__(self, *, port: int, timeout: float, max_reconnect_attempts: int):
            return

        def invoke(self, cmd: str, payload=None):
            return "pong"

        def close(self) -> None:
            return

    def _fake_popen(argv, **_kwargs):
        captured["argv"] = list(argv)
        return _FakeProc()

    monkeypatch.setattr("hosting.daemon.background.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("hosting.daemon.background.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.daemon.background.time.sleep", lambda _sec: None)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeConn)

    result = start_daemon_background(port=19876, wait_ready_seconds=1.0)

    argv = list(captured.get("argv") or [])
    assert "--runtime-profile" in argv
    rp_idx = argv.index("--runtime-profile")
    assert str(argv[rp_idx + 1]) == "detached_user_process"
    assert "--log-file" not in argv
    assert result == {"pid": 54321, "port": 19876}


def test_start_daemon_background_includes_explicit_log_flag(monkeypatch) -> None:
    captured: dict[str, object] = {"argv": []}

    class _FakeProc:
        pid = 54321

        def poll(self):
            return None

    class _FakePidFile:
        def __init__(self, _path=None):
            return

        def is_alive(self) -> bool:
            return True

        def get_port(self) -> int:
            return 19876

        def read(self):
            return {"pid": 54321, "port": 19876}

    class _FakeConn:
        def __init__(self, *, port: int, timeout: float, max_reconnect_attempts: int):
            return

        def invoke(self, cmd: str, payload=None):
            return "pong"

        def close(self) -> None:
            return

    def _fake_popen(argv, **_kwargs):
        captured["argv"] = list(argv)
        return _FakeProc()

    monkeypatch.setattr("hosting.daemon.background.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("hosting.daemon.background.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.daemon.background.time.sleep", lambda _sec: None)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeConn)

    result = start_daemon_background(
        port=19876,
        log_file=Path("C:/tmp/host_daemon.log"),
        wait_ready_seconds=1.0,
    )

    argv = list(captured.get("argv") or [])
    assert "--runtime-profile" in argv
    rp_idx = argv.index("--runtime-profile")
    assert str(argv[rp_idx + 1]) == "detached_user_process"
    assert "--log-file" in argv
    idx = argv.index("--log-file")
    assert Path(argv[idx + 1]) == Path("C:/tmp/host_daemon.log")
    assert result["pid"] == 54321
    assert result["port"] == 19876
    assert Path(str(result.get("log_file") or "")) == Path("C:/tmp/host_daemon.log")


def test_apply_foreground_terminal_disconnect_policy_ignores_sighup_when_configured(monkeypatch, tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        runtime_profile="foreground_terminal_bound",
    )
    daemon.svc.set_control_config(
        lifecycle_profile="foreground_terminal_bound",
        lifecycle_policy={"on_terminal_disconnect": "keep_daemon_running"},
    )
    captured: dict[str, object] = {}

    def _fake_signal(sig, handler):
        captured["sig"] = sig
        captured["handler"] = handler

    monkeypatch.setattr("hosting.daemon.lifecycle.signal.signal", _fake_signal)
    out = _apply_foreground_terminal_disconnect_policy(daemon)
    assert out in {"keep_daemon_running_ignore_sighup", "keep_daemon_running_no_sighup"}
    if "sig" in captured:
        assert captured["handler"] is not None


def test_apply_foreground_terminal_disconnect_policy_noop_for_detached(monkeypatch, tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        runtime_profile="detached_user_process",
    )
    called: dict[str, object] = {}

    def _fake_signal(sig, handler):
        called["sig"] = sig
        called["handler"] = handler

    monkeypatch.setattr("hosting.daemon.lifecycle.signal.signal", _fake_signal)
    out = _apply_foreground_terminal_disconnect_policy(daemon)
    assert out == "not_foreground"
    assert called == {}


def test_execute_shutdown_checkpoints_orders_managed_shutdowns(tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        runtime_profile="detached_user_process",
    )
    shutdown_calls: list[str] = []
    daemon.svc.discover_running = lambda **_kwargs: [  # type: ignore[method-assign]
        {"engine_id": "e1"},
        {"engine_id": "e2"},
    ]

    def _fake_shutdown(engine_id: str, *, timeout_seconds: float = 2.0):
        shutdown_calls.append(str(engine_id))
        if str(engine_id) == "e2":
            return {"status": "stop_failed", "engine_id": "e2", "alive": True}
        return {"status": "stopped", "engine_id": str(engine_id), "alive": False}

    daemon.svc.shutdown = _fake_shutdown  # type: ignore[method-assign]
    report = daemon._execute_shutdown_checkpoints()  # noqa: SLF001
    assert shutdown_calls == ["e1", "e2"]
    assert int(report.get("attempted") or 0) == 2
    assert int(report.get("stopped") or 0) == 1
    assert int(report.get("failed") or 0) == 1
    assert len(list(report.get("results") or [])) == 2


def test_execute_shutdown_checkpoints_handles_discovery_failure(tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
    )

    def _raise_discovery(**_kwargs):
        raise RuntimeError("discover_failed")

    daemon.svc.discover_running = _raise_discovery  # type: ignore[method-assign]
    report = daemon._execute_shutdown_checkpoints()  # noqa: SLF001
    assert str(report.get("status") or "") == "failed"
    assert "discover_failed" in str(report.get("error") or "")


def test_drain_inflight_operations_completes_pending_tasks(tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
    )

    async def _run() -> None:
        async def _short_task() -> None:
            await asyncio.sleep(0.01)

        task = asyncio.create_task(_short_task())
        with daemon._operation_tasks_lock:  # noqa: SLF001
            daemon._operation_tasks.add(task)  # noqa: SLF001
        report = await daemon._drain_inflight_operations(timeout_seconds=1.0)  # noqa: SLF001
        assert int(report.get("pending_before") or 0) == 1
        assert int(report.get("pending_after") or 0) == 0
        assert int(report.get("drained") or 0) == 1
        assert bool(report.get("timed_out")) is False

    asyncio.run(_run())
