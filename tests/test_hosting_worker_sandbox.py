from __future__ import annotations
import base64
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from hosting._process_utils import pid_alive, terminate_process_tree
from hosting.service.host_service import EngineHostService
from hosting.sandbox import (
    BrokeredFilesystem,
    BrokeredFilesystemClient,
    BrokeredHttpClient,
    WorkerLaunchResult,
    WorkerSandboxPolicy,
)
from hosting.sandbox.launcher import WorkerLaunchRequest, launch_worker_process


def test_worker_sandbox_policy_normalizes_nested_shape() -> None:
    policy = WorkerSandboxPolicy.from_mapping(
        {
            "sandbox": {
                "enabled": True,
                "profile": "generic_worker_v1",
                "platform_policy": {
                    "windows": {
                        "restricted_token": True,
                        "integrity_level": "low",
                        "job_object": True,
                    }
                },
                "filesystem": {
                    "rules": [
                        {
                            "path": "C:\\workers\\scratch\\gw1",
                            "access": ["read", "write"],
                            "platform_status": {"windows": "partial", "linux": "supported"},
                        }
                    ]
                },
                "process": {
                    "allow_subprocess": False,
                    "inherit_parent_handles": False,
                },
                "network": {
                    "mode": "brokered_only",
                    "allow_hosts": ["example.com"],
                    "allow_url_prefixes": ["https://example.com/api/"],
                },
                "brokered_io": {"filesystem": True, "http": True, "subprocess": False},
            }
        }
    )

    assert policy.enabled is True
    assert policy.profile == "generic_worker_v1"
    assert len(policy.filesystem_rules) == 1
    assert policy.filesystem_rules[0].access == ["read", "write"]
    assert policy.network.mode == "brokered_only"
    assert policy.network.allow_hosts == ["example.com"]
    assert policy.windows.integrity_level == "low"


def test_brokered_filesystem_denies_traversal_and_allows_root_scoped_io(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "input.txt").write_text("hello", encoding="utf-8")
    broker = BrokeredFilesystem(
        WorkerSandboxPolicy.from_mapping(
            {
                "sandbox": {
                    "enabled": True,
                    "filesystem": {
                        "rules": [
                            {
                                "root_id": "worker_input",
                                "path": str(root),
                                "access": ["read", "write"],
                            }
                        ]
                    },
                }
            }
        )
    )

    read_out = broker.read_text(root_id="worker_input", relative_path="input.txt")
    assert read_out["text"] == "hello"

    write_out = broker.write_text(root_id="worker_input", relative_path="out.txt", text="world")
    assert int(write_out["bytes_written"]) == 5
    assert (root / "out.txt").read_text(encoding="utf-8") == "world"

    try:
        broker.read_text(root_id="worker_input", relative_path="..\\secret.txt")
    except PermissionError as exc:
        assert str(exc) == "path_traversal_denied"
    else:  # pragma: no cover
        raise AssertionError("expected traversal denial")


def test_service_brokered_filesystem_uses_registration_policy(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "control_state.json",
    )
    root = tmp_path / "sandbox_root"
    root.mkdir()
    svc.register_spawned(
        engine_id="worker1",
        pid=1234,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        sandbox_policy={
            "sandbox": {
                "enabled": True,
                "filesystem": {
                    "rules": [
                        {
                            "root_id": "rw",
                            "path": str(root),
                            "access": ["read", "write"],
                        }
                    ]
                },
                "brokered_io": {"filesystem": True, "http": False, "subprocess": False},
            }
        },
    )

    mkdir_out = svc.sandbox_fs_mkdir(engine_id="worker1", root_id="rw", relative_path="nested")
    assert mkdir_out["created"] is True

    svc.sandbox_fs_write_text(engine_id="worker1", root_id="rw", relative_path="nested\\a.txt", text="abc")
    read_out = svc.sandbox_fs_read_text(
        engine_id="worker1",
        root_id="rw",
        relative_path="nested\\a.txt",
        callback_context={"tool_name": "peek", "tool_call_id": "call-fs-1"},
    )
    assert read_out["text"] == "abc"
    assert read_out["callback_context"] == {"tool_name": "peek", "tool_call_id": "call-fs-1"}

    listing = svc.sandbox_fs_list(engine_id="worker1", root_id="rw", relative_path="nested")
    assert [row["name"] for row in listing["entries"]] == ["a.txt"]


def test_worker_side_brokered_filesystem_client_builds_expected_rpc_payloads() -> None:
    calls: list[tuple[str, dict]] = []

    def _invoke(cmd: str, payload: dict) -> dict:
        calls.append((str(cmd), dict(payload)))
        return {"ok": True}

    client = BrokeredFilesystemClient(engine_id="worker1", rpc_invoke=_invoke)
    client.read_text(root_id="input", relative_path="a.txt")
    client.write_text(root_id="rw", relative_path="b.txt", text="hello")
    client.list_dir(root_id="rw", relative_path="nested")
    client.mkdir(root_id="rw", relative_path="nested")
    client.stat(root_id="rw", relative_path="nested\\b.txt")

    assert [cmd for cmd, _ in calls] == [
        "sandbox-fs-read-text",
        "sandbox-fs-write-text",
        "sandbox-fs-list",
        "sandbox-fs-mkdir",
        "sandbox-fs-stat",
    ]
    assert calls[0][1]["engine_id"] == "worker1"
    assert calls[1][1]["text"] == "hello"


def test_service_brokered_http_enforces_allowlist_and_returns_response(monkeypatch) -> None:
    svc = EngineHostService()
    svc._find_registration = lambda _eid: {  # type: ignore[method-assign]
        "engine_id": "worker1",
        "sandbox_policy": {
            "sandbox": {
                "enabled": True,
                "network": {
                    "mode": "brokered_only",
                    "allow_hosts": ["example.com"],
                    "allow_url_prefixes": ["https://example.com/api/"],
                },
                "brokered_io": {"filesystem": True, "http": True, "subprocess": False},
            }
        },
    }

    class _Resp:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int = -1) -> bytes:
            return b'{"ok":true}'

    def _fake_urlopen(req, timeout=0.0):
        assert req.full_url == "https://example.com/api/test"
        assert req.get_method() == "POST"
        assert timeout == 5.0
        return _Resp()

    monkeypatch.setattr("hosting.sandbox.broker_http.urllib.request.urlopen", _fake_urlopen)

    out = svc.sandbox_http_fetch(
        engine_id="worker1",
        url="https://example.com/api/test",
        method="POST",
        headers={"Content-Type": "application/json", "Host": "ignored"},
        body_b64=base64.b64encode(b"{}").decode("ascii"),
        timeout_seconds=5.0,
        callback_context={"tool_name": "http_tool", "tool_call_id": "call-http-1"},
    )

    assert out["engine_id"] == "worker1"
    assert int(out["status_code"]) == 200
    assert base64.b64decode(out["body_b64"]) == b'{"ok":true}'
    assert out["callback_context"] == {"tool_name": "http_tool", "tool_call_id": "call-http-1"}


def test_service_brokered_http_denies_non_allowlisted_url() -> None:
    svc = EngineHostService()
    svc._find_registration = lambda _eid: {  # type: ignore[method-assign]
        "engine_id": "worker1",
        "sandbox_policy": {
            "sandbox": {
                "enabled": True,
                "network": {
                    "mode": "brokered_only",
                    "allow_hosts": ["example.com"],
                    "allow_url_prefixes": ["https://example.com/api/"],
                },
                "brokered_io": {"filesystem": True, "http": True, "subprocess": False},
            }
        },
    }

    try:
        svc.sandbox_http_fetch(engine_id="worker1", url="https://evil.example.net/api/test")
    except PermissionError as exc:
        assert str(exc).startswith("brokered_http_host_not_allowed:")
    else:  # pragma: no cover
        raise AssertionError("expected brokered http denial")


def test_worker_side_brokered_http_client_builds_expected_rpc_payload() -> None:
    calls: list[tuple[str, dict]] = []

    def _invoke(cmd: str, payload: dict) -> dict:
        calls.append((str(cmd), dict(payload)))
        return {"status": "ok"}

    client = BrokeredHttpClient(engine_id="worker1", rpc_invoke=_invoke)
    client.fetch(
        url="https://example.com/api/test",
        method="POST",
        headers={"Content-Type": "application/json"},
        body_b64="e30=",
        timeout_seconds=4.0,
        max_response_bytes=512,
    )

    assert calls == [
        (
            "sandbox-http-fetch",
            {
                "engine_id": "worker1",
                "url": "https://example.com/api/test",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body_b64": "e30=",
                "timeout_seconds": 4.0,
                "max_response_bytes": 512,
            },
        )
    ]


def test_spawn_persists_sandbox_policy_and_runtime(monkeypatch, tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "control_state.json",
    )

    def _fake_launch(req: WorkerLaunchRequest) -> WorkerLaunchResult:
        assert req.sandbox_policy.enabled is True
        return WorkerLaunchResult(
            pid=4321,
            command=list(req.command),
            persisted_env=dict(req.env),
            runtime={"platform": "windows", "mode": "mock_sandbox"},
        )

    monkeypatch.setattr("hosting.service.engines.launch_worker_process", _fake_launch)

    reg = svc.spawn(
        engine_id="worker1",
        command=["python", "-m", "hosting.engine_worker_ipc"],
        sandbox_policy={
            "sandbox": {
                "enabled": True,
                "platform_policy": {"windows": {"restricted_token": True, "integrity_level": "low", "job_object": True}},
            }
        },
    )

    assert int(reg["pid"]) == 4321
    assert dict(reg.get("sandbox_runtime") or {}).get("mode") == "mock_sandbox"
    sandbox_policy = dict(reg.get("sandbox_policy") or {}).get("sandbox") or {}
    assert sandbox_policy.get("enabled") is True
    assert dict(dict(sandbox_policy.get("platform_policy") or {}).get("windows") or {}).get("integrity_level") == "low"


def test_plain_launcher_uses_close_fds_when_parent_handles_disabled(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 9876

    def _fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr("hosting.sandbox.launcher.subprocess.Popen", _fake_popen)

    req = WorkerLaunchRequest(
        engine_id="worker1",
        command=["python", "-m", "hosting.engine_worker_ipc"],
        cwd=tmp_path,
        env={"A": "B"},
        log_path=tmp_path / "worker.log",
        sandbox_policy=WorkerSandboxPolicy.from_mapping({"sandbox": {"enabled": False, "process": {"inherit_parent_handles": False}}}),
    )
    out = launch_worker_process(req)

    assert int(out.pid) == 9876
    assert bool(captured["kwargs"]["close_fds"]) is True


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death signals are task scoped")
def test_posix_worker_survives_the_short_lived_operation_thread(tmp_path: Path) -> None:
    ready = tmp_path / "worker.ready"
    source = (
        "from ctypes import CDLL; from pathlib import Path; import sys, time; "
        "assert CDLL(None, use_errno=True).prctl(1, 15, 0, 0, 0) == 0; "
        "Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
    )
    request = WorkerLaunchRequest(
        engine_id="thread-parent-regression",
        command=[sys.executable, "-c", source, str(ready)],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "worker.log",
        sandbox_policy=WorkerSandboxPolicy.from_mapping({"sandbox": {"enabled": False}}),
    )
    launched: list[WorkerLaunchResult] = []
    errors: list[BaseException] = []

    def _launch() -> None:
        try:
            launched.append(launch_worker_process(request))
        except BaseException as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)

    operation_thread = threading.Thread(target=_launch)
    operation_thread.start()
    operation_thread.join(timeout=10)
    assert not operation_thread.is_alive()
    assert not errors
    assert launched
    pid = int(launched[0].pid)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            if not pid_alive(pid):
                break
            time.sleep(0.02)
        assert ready.exists(), (tmp_path / "worker.log").read_text(encoding="utf-8", errors="replace")

        survival_deadline = time.monotonic() + 0.5
        while time.monotonic() < survival_deadline:
            assert pid_alive(pid)
            time.sleep(0.02)
        assert launched[0].runtime["parent_task"] == "persistent_worker_launcher"
    finally:
        terminate_process_tree(pid)
