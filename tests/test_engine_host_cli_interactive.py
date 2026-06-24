from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import hosting.engine_host_cli_interactive as interactive


class _FakeChannel:
    instances: list["_FakeChannel"] = []

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})
        self.session_token: Optional[str] = None
        self.invocations: list[tuple[str, Dict[str, Any], Optional[str]]] = []
        self.target = {"mode": "ssh" if self.settings.get("engine_host_ssh_target") else "local"}
        self.bootstrap_result = {"alive": True}
        self.stop_result = {"status": "shutdown_sent"}
        _FakeChannel.instances.append(self)

    def set_session_token(self, token: Optional[str]) -> None:
        self.session_token = token

    def get_target(self) -> Dict[str, Any]:
        return dict(self.target)

    def get_daemon_status(self) -> Dict[str, Any]:
        return {"alive": True, "reachable": True, "pid_alive": True}

    def bootstrap_daemon(self, *, wait_ready_seconds: float = 8.0) -> Dict[str, Any]:
        return dict(self.bootstrap_result)

    def stop_daemon(self) -> Dict[str, Any]:
        return dict(self.stop_result)

    def restart_remote_daemon(self) -> Dict[str, Any]:
        return {"started": True}

    def invoke_control_command(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.invocations.append((command, dict(payload), self.session_token))
        if command == "needs-auth":
            raise RuntimeError("session_token_required")
        if command == "auth-failed":
            raise RuntimeError("persistent daemon control channel failed for 'host-metrics': auth_failed")
        return {"command": command, "payload": dict(payload), "session_token": self.session_token}


def test_interactive_api_invoke_uses_engine_host_control_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeChannel.instances.clear()
    monkeypatch.setattr(interactive, "EngineHostControlChannel", _FakeChannel)
    args = argparse.Namespace(
        pid_file=Path("daemon.pid"),
        engines_state_file=Path("engines.json"),
        control_state_file=Path("control.json"),
        engine_host_ssh_target="user@example-host",
        control_ssh_key="C:/keys/id_ed25519",
    )

    out = interactive._api_invoke(args, "host-metrics", {"detail": True}, session_token="tok-1")

    assert out == {"command": "host-metrics", "payload": {"detail": True}, "session_token": "tok-1"}
    assert len(_FakeChannel.instances) == 1
    channel = _FakeChannel.instances[0]
    assert channel.settings["engine_host_ssh_target"] == "user@example-host"
    assert channel.settings["control_ssh_key"] == "C:/keys/id_ed25519"
    assert channel.settings["engine_host_daemon_pid_file"] == "daemon.pid"
    assert channel.invocations == [("host-metrics", {"detail": True}, "tok-1")]


def test_interactive_api_invoke_maps_session_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeChannel.instances.clear()
    monkeypatch.setattr(interactive, "EngineHostControlChannel", _FakeChannel)
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)

    with pytest.raises(PermissionError, match="session_token_required"):
        interactive._api_invoke(args, "needs-auth", {})


def test_interactive_api_invoke_maps_auth_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeChannel.instances.clear()
    monkeypatch.setattr(interactive, "EngineHostControlChannel", _FakeChannel)
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)

    with pytest.raises(PermissionError, match="session_token_required"):
        interactive._api_invoke(args, "auth-failed", {})


def test_background_session_renewer_extends_token_while_menu_can_block(monkeypatch: pytest.MonkeyPatch) -> None:
    renewed = threading.Event()

    class RenewChannel(_FakeChannel):
        def invoke_control_command(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            self.invocations.append((command, dict(payload), self.session_token))
            if command == "auth-validate-session":
                return {"valid": True, "ttl_remaining_seconds": 10, "scope": "control"}
            if command == "auth-renew-session":
                renewed.set()
                return {"status": "ok", "expires_at": time.time() + 900, "scope": "control"}
            return {}

    _FakeChannel.instances.clear()
    monkeypatch.setattr(interactive, "EngineHostControlChannel", RenewChannel)
    monkeypatch.setattr(interactive, "_SESSION_RENEW_CHECK_INTERVAL_SECONDS", 0.01)
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    interactive._set_interactive_session_token(args, "tok-1")

    interactive._ensure_session_renewer(args)
    try:
        assert renewed.wait(1.0)
    finally:
        interactive._stop_session_renewer(args)

    worker = next(inst for inst in _FakeChannel.instances if inst.invocations)
    assert ("auth-renew-session", {"token": "tok-1", "scope": "control", "ttl_seconds": 900}, "tok-1") in worker.invocations


def test_background_session_renewer_pauses_quietly_when_daemon_stops(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class StoppedChannel(_FakeChannel):
        def get_daemon_status(self) -> Dict[str, Any]:
            return {"alive": False, "reachable": False}

    _FakeChannel.instances.clear()
    monkeypatch.setattr(interactive, "EngineHostControlChannel", StoppedChannel)
    monkeypatch.setattr(interactive, "_SESSION_RENEW_CHECK_INTERVAL_SECONDS", 0.01)
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    interactive._set_interactive_session_token(args, "tok-1")

    interactive._ensure_session_renewer(args)
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            thread = getattr(args, "_interactive_session_renew_thread", None)
            if isinstance(thread, threading.Thread) and not thread.is_alive():
                break
            time.sleep(0.01)
    finally:
        interactive._stop_session_renewer(args)

    assert getattr(args, "_interactive_daemon_stopped_notice") is True
    assert "automatic auth refresh is paused" not in capsys.readouterr().out


def test_metrics_auth_error_does_not_print_daemon_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        interactive,
        "_api_invoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("session_token_required")),
    )

    with pytest.raises(PermissionError):
        interactive._show_metrics(args, session_token=None)

    assert "Error fetching metrics" not in capsys.readouterr().out


def test_interactive_lifecycle_uses_channel_helpers(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _FakeChannel.instances.clear()
    monkeypatch.setattr(interactive, "EngineHostControlChannel", _FakeChannel)
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)

    interactive._start_daemon(args)
    interactive._stop_daemon(args)

    out = capsys.readouterr().out
    assert "Daemon started" in out
    assert "Daemon stop signal sent" in out


def test_interactive_extracts_key_id_from_secret_record_metadata() -> None:
    assert interactive._extract_key_id_from_private_key_json(
        {
            "secret_id": "rbac-wrong-private",
            "payload": "PRIVATE",
            "metadata": {"key_id": "admin-main"},
        }
    ) == "admin-main"


def test_interactive_extracts_key_id_from_secret_record_id_fallback() -> None:
    assert interactive._extract_key_id_from_private_key_json(
        {
            "secret_id": "rbac-admin-main-private",
            "payload": "PRIVATE",
            "metadata": {},
        }
    ) == "admin-main"


def test_list_consumers_uses_offline_fallback_when_local_daemon_stopped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        interactive,
        "_offline_local_invoke",
        lambda *_args, **_kwargs: {
            "sessions": [
                {
                    "token_preview": "tok...123",
                    "key_id": "admin-main",
                    "scope": "control",
                    "role": "admin",
                }
            ]
        },
    )

    interactive._list_consumers(args, session_token=None)

    out = capsys.readouterr().out
    assert "offline fallback state" in out
    assert "admin-main" in out
    assert "tok...123" in out


def test_print_live_consumers_marks_current_interactive_cli(capsys: pytest.CaptureFixture[str]) -> None:
    current = "abcdefghijk"
    interactive._print_live_consumers(
        {
            "connections": [
                {
                    "connection_id": "conn-123456",
                    "transport": "local_ipc",
                    "peer_host": "127.0.0.1",
                    "actor_ids": ["key:admin-main"],
                    "session_token_previews": [interactive._get_token_preview(current)],
                    "pid": 1234,
                    "consumer_kind": "interactive_cli",
                    "process": {"name": "python.exe", "parent_pid": 1000},
                    "age_seconds": 2,
                    "idle_seconds": 0,
                    "command_count": 3,
                    "last_command": "list-live-consumers",
                }
            ],
            "actors": [{"actor_id": "key:admin-main", "connection_count": 1}],
        },
        session_token=current,
    )

    out = capsys.readouterr().out
    assert "conn-123456" in out
    assert "local_ipc" in out
    assert "PID: 1234" in out
    assert "interactive_cli" in out
    assert "(this interactive CLI)" in out
    assert "key:admin-main" in out


def test_workflow_pool_active_request_ids_merges_workers() -> None:
    assert interactive._workflow_pool_active_request_ids(
        {
            "active_request_ids": ["req-a"],
            "workers": [
                {"active_request_ids": ["req-a", "req-b"]},
                {"active_request_ids": ["req-c"]},
            ],
        }
    ) == ["req-a", "req-b", "req-c"]


def test_manage_workflow_helpers_prefers_workflow_python_facade(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    invocations: list[tuple[str, Dict[str, Any]]] = []

    def fake_api(_args: argparse.Namespace, cmd: str, payload: Dict[str, Any], session_token: Optional[str] = None) -> Dict[str, Any]:
        invocations.append((cmd, dict(payload)))
        if cmd == "discover-running":
            return {
                "engines": {
                    "wf-py": {
                        "executor_kind": "workflow_python_helper",
                        "environment": {"environment_key": "env-demo"},
                        "process_resources": {
                            "workflow_python_capacity": 2,
                            "workflow_python_active_calls": 1,
                            "workflow_python_process_count": 2,
                        },
                    }
                }
            }
        if cmd == "workflow-python-resources":
            return {
                "status": "ok",
                "engine_id": "wf-py",
                "environment_key": "env-demo",
                "workflow_pool": {
                    "pool_id": "workflow_python/env-demo",
                    "metrics": {
                        "desired_capacity": 2,
                        "worker_count": 1,
                        "active_calls": 1,
                        "available_slots": 1,
                        "saturation_count": 0,
                        "cancellation_count": 0,
                        "error_count": 0,
                        "workers": [{"active_request_ids": ["req-1"]}],
                        "recent_requests": [{"request_id": "req-0", "status": "ok", "lifetime_ms": 12}],
                    },
                },
                "workflow_python_cpu_percent": 7.5,
                "workflow_python_memory_mb": 128.0,
            }
        raise AssertionError(cmd)

    choices = iter(["wf-py", "b"])
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(interactive, "_api_invoke", fake_api)
    monkeypatch.setattr(interactive, "_active_session_token", lambda _args, token: token)
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: next(choices))

    interactive._manage_workflow_runtimes(args, session_token="tok-1")

    assert ("workflow-python-resources", {"engine_id": "wf-py", "profile": "helper", "environment_key": "env-demo"}) in invocations
    out = capsys.readouterr().out
    assert "Environment Key" in out
    assert "workflow_python/env-demo" in out
    assert "req-1" in out
    assert "7.5%" in out
    assert "128.0MB" in out


def test_manage_workflow_helpers_can_ensure_python_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    invocations: list[tuple[str, Dict[str, Any]]] = []

    def fake_api(_args: argparse.Namespace, cmd: str, payload: Dict[str, Any], session_token: Optional[str] = None) -> Dict[str, Any]:
        invocations.append((cmd, dict(payload)))
        if cmd == "discover-running":
            return {
                "engines": {
                    "wf-py": {
                        "executor_kind": "workflow_python_helper",
                        "process_resources": {"workflow_python_capacity": 2},
                    }
                }
            }
        if cmd == "workflow-python-resources":
            return {
                "status": "ok",
                "engine_id": "wf-py",
                "environment_key": "env-demo",
                "workflow_pool": {"pool_id": "workflow_python/env-demo", "metrics": {"desired_capacity": 2}},
            }
        if cmd == "workflow-python-ensure":
            return {
                "status": "ok",
                "engine_id": "wf-py",
                "environment_key": "env-demo",
            }
        raise AssertionError(cmd)

    choices = iter(["wf-py", "e", "b"])
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(interactive, "_api_invoke", fake_api)
    monkeypatch.setattr(interactive, "_active_session_token", lambda _args, token: token)
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: next(choices))

    interactive._manage_workflow_runtimes(args, session_token="tok-1")

    assert ("workflow-python-ensure", {"profile": "helper", "engine_id": "wf-py", "capacity": 2}) in invocations
    assert ("workflow-python-resources", {"engine_id": "wf-py", "profile": "helper", "environment_key": "env-demo"}) in invocations
    assert "runtime ensured" in capsys.readouterr().out


def test_manage_workflow_helpers_can_inspect_request_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    invocations: list[tuple[str, Dict[str, Any]]] = []

    def fake_api(_args: argparse.Namespace, cmd: str, payload: Dict[str, Any], session_token: Optional[str] = None) -> Dict[str, Any]:
        invocations.append((cmd, dict(payload)))
        if cmd == "discover-running":
            return {
                "engines": {
                    "wf-py": {
                        "executor_kind": "workflow_python_helper",
                        "environment": {"environment_key": "env-demo"},
                        "process_resources": {"workflow_python_capacity": 2},
                    }
                }
            }
        if cmd == "workflow-python-resources":
            return {
                "status": "ok",
                "engine_id": "wf-py",
                "environment_key": "env-demo",
                "workflow_pool": {
                    "pool_id": "workflow_python/env-demo",
                    "metrics": {
                        "desired_capacity": 2,
                        "active_request_ids": ["req-1"],
                        "recent_requests": [{"request_id": "req-1", "status": "running"}],
                    },
                },
            }
        if cmd == "workflow-python-request-status":
            return {
                "status": "ok",
                "environment_key": "env-demo",
                "request": {
                    "request_id": "req-1",
                    "status": "running",
                    "stream_event_count": 3,
                    "latest_progress": {"message": "halfway"},
                },
            }
        raise AssertionError(cmd)

    choices = iter(["wf-py", "i", "b"])
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(interactive, "_api_invoke", fake_api)
    monkeypatch.setattr(interactive, "_active_session_token", lambda _args, token: token)
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: next(choices))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    interactive._manage_workflow_runtimes(args, session_token="tok-1")

    assert (
        "workflow-python-request-status",
        {"engine_id": "wf-py", "profile": "helper", "environment_key": "env-demo", "request_id": "req-1"},
    ) in invocations
    out = capsys.readouterr().out
    assert "Request Status" in out
    assert "halfway" in out


def test_manage_workflow_helpers_can_receive_python_stream_events(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    invocations: list[tuple[str, Dict[str, Any]]] = []

    def fake_api(_args: argparse.Namespace, cmd: str, payload: Dict[str, Any], session_token: Optional[str] = None) -> Dict[str, Any]:
        invocations.append((cmd, dict(payload)))
        if cmd == "discover-running":
            return {
                "engines": {
                    "wf-py": {
                        "executor_kind": "workflow_python_helper",
                        "environment": {"environment_key": "env-demo"},
                        "process_resources": {"workflow_python_capacity": 2},
                    }
                }
            }
        if cmd == "workflow-python-resources":
            return {
                "status": "ok",
                "engine_id": "wf-py",
                "environment_key": "env-demo",
                "workflow_pool": {"pool_id": "workflow_python/env-demo", "metrics": {"desired_capacity": 2}},
            }
        if cmd == "workflow-python-event-subscribe":
            return {
                "status": "ok",
                "normalized_events": [
                    {"kind": "started", "request_id": "req-1"},
                    {"kind": "log", "logs": {"output_limit_bytes": 4096, "summary": ""}},
                    {"kind": "artifact", "name": "report", "ref": "@artifacts/a/report.txt", "size_bytes": 12},
                    {"kind": "artifact", "name": "summary", "artifact_kind": "inline", "filename": "summary.txt", "size_bytes": 7},
                    {"kind": "error", "error": {"code": "workflow_python_node_profile_not_implemented"}},
                ],
            }
        raise AssertionError(cmd)

    choices = iter(["wf-py", "v", "b"])
    inputs = iter(["stream-1", "5"])
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(interactive, "_api_invoke", fake_api)
    monkeypatch.setattr(interactive, "_active_session_token", lambda _args, token: token)
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: next(choices))
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    interactive._manage_workflow_runtimes(args, session_token="tok-1")

    assert ("workflow-python-event-subscribe", {"stream_id": "stream-1", "max_items": 5}) in invocations
    out = capsys.readouterr().out
    assert "Events" in out
    assert "@artifacts/a/report.txt" in out
    assert "summary inline summary.txt" in out
    assert "workflow_python_node_profile_not_implemented" in out


def test_manage_workflow_runtimes_supports_js_node_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    invocations: list[tuple[str, Dict[str, Any]]] = []

    def fake_api(_args: argparse.Namespace, cmd: str, payload: Dict[str, Any], session_token: Optional[str] = None) -> Dict[str, Any]:
        invocations.append((cmd, dict(payload)))
        if cmd == "discover-running":
            return {
                "engines": {
                    "wf-js": {
                        "executor_kind": "workflow_js_node",
                        "environment": {"environment_key": "env-js"},
                        "process_resources": {"workflow_js_capacity": 2},
                    }
                }
            }
        if cmd == "workflow-js-resources":
            return {
                "status": "ok",
                "engine_id": "wf-js",
                "environment_key": "env-js",
                "workflow_pool": {
                    "pool_id": "workflow_js/env-js",
                    "metrics": {
                        "desired_capacity": 2,
                        "active_request_ids": ["req-js-1"],
                        "recent_requests": [{"request_id": "req-js-1", "status": "running"}],
                    },
                },
                "node_runtime": {"active_count": 1, "processes": [{"request_id": "req-js-1", "pid": 123, "alive": True}]},
            }
        if cmd == "workflow-js-request-status":
            return {
                "status": "ok",
                "environment_key": "env-js",
                "request": {
                    "request_id": "req-js-1",
                    "status": "running",
                    "stream_event_count": 2,
                    "latest_progress": {"message": "js halfway"},
                },
            }
        raise AssertionError(cmd)

    choices = iter(["wf-js", "i", "b"])
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(interactive, "_api_invoke", fake_api)
    monkeypatch.setattr(interactive, "_active_session_token", lambda _args, token: token)
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: next(choices))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    interactive._manage_workflow_runtimes(args, session_token="tok-1")

    assert ("workflow-js-resources", {"engine_id": "wf-js", "profile": "node", "environment_key": "env-js"}) in invocations
    assert (
        "workflow-js-request-status",
        {"engine_id": "wf-js", "profile": "node", "environment_key": "env-js", "request_id": "req-js-1"},
    ) in invocations
    out = capsys.readouterr().out
    assert "workflow_js/env-js" in out
    assert "js halfway" in out


def test_manage_workflow_runtimes_can_receive_js_stream_events(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    invocations: list[tuple[str, Dict[str, Any]]] = []

    def fake_api(_args: argparse.Namespace, cmd: str, payload: Dict[str, Any], session_token: Optional[str] = None) -> Dict[str, Any]:
        invocations.append((cmd, dict(payload)))
        if cmd == "discover-running":
            return {
                "engines": {
                    "wf-js": {
                        "executor_kind": "workflow_js_node",
                        "environment": {"environment_key": "env-js"},
                        "process_resources": {"workflow_js_capacity": 2},
                    }
                }
            }
        if cmd == "workflow-js-resources":
            return {
                "status": "ok",
                "engine_id": "wf-js",
                "environment_key": "env-js",
                "workflow_pool": {"pool_id": "workflow_js/env-js", "metrics": {"desired_capacity": 2}},
            }
        if cmd == "workflow-js-event-subscribe":
            return {
                "status": "ok",
                "normalized_events": [
                    {"kind": "started", "request_id": "req-js-1"},
                    {"kind": "progress", "message": "js progress"},
                    {"kind": "result", "status": "ok"},
                ],
            }
        raise AssertionError(cmd)

    choices = iter(["wf-js", "v", "b"])
    inputs = iter(["js-stream-1", "5"])
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(interactive, "_api_invoke", fake_api)
    monkeypatch.setattr(interactive, "_active_session_token", lambda _args, token: token)
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: next(choices))
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    interactive._manage_workflow_runtimes(args, session_token="tok-1")

    assert ("workflow-js-event-subscribe", {"stream_id": "js-stream-1", "max_items": 5}) in invocations
    out = capsys.readouterr().out
    assert "Events" in out
    assert "js progress" in out


def test_print_sessions_marks_current_interactive_cli(capsys: pytest.CaptureFixture[str]) -> None:
    current = "abcdefghijk"
    other = "other-session-token"
    interactive._print_sessions(
        {
            "sessions": [
                {
                    "token_preview": interactive._get_token_preview(current),
                    "key_id": "admin-main",
                    "scope": "control",
                    "role": "admin",
                },
                {
                    "token_preview": interactive._get_token_preview(other),
                    "key_id": "backend",
                    "scope": "traffic",
                    "role": "worker_user",
                },
            ]
        },
        session_token=current,
    )

    out = capsys.readouterr().out
    assert "this interactive CLI" in out
    assert "Consumer: interactive CLI" in out
    assert "backend" in out


def test_session_identity_from_list_matches_current_token() -> None:
    token = "abcdefghijk"
    other = "other-session-token"

    key_id, role = interactive._session_identity_from_list(
        {
            "sessions": [
                {"token_preview": interactive._get_token_preview(other), "key_id": "backend", "role": "worker_user"},
                {"token_preview": interactive._get_token_preview(token), "key_id": "admin-main", "role": "admin"},
            ]
        },
        token,
    )

    assert key_id == "admin-main"
    assert role == "admin"


def test_reachability_summary_explains_unavailable_ipc() -> None:
    note = interactive._reachability_summary(
        {
            "alive": True,
            "reachable": False,
            "reachability": {
                "error": "worker IPC endpoint is unavailable for engine 'tools1' at 'pipe'; worker process may not be running"
            },
        }
    )

    assert note is not None
    assert "IPC endpoint unavailable" in note
    assert "stale PID" in note


def test_reachability_summary_treats_spawning_ipc_as_startup() -> None:
    note = interactive._reachability_summary(
        {
            "state": "spawning",
            "alive": True,
            "reachable": False,
            "reachability": {
                "error": "worker IPC endpoint is unavailable for engine 'model1' at 'pipe'; worker process may not be running"
            },
        }
    )

    assert note is not None
    assert "still starting" in note
    assert "stale PID" not in note


def test_worker_status_summary_uses_daemon_resource_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    monkeypatch.setattr(
        interactive,
        "_api_invoke",
        lambda *_args, **_kwargs: {
            "resource_summary": {
                "workers_count": 2,
                "worker_cpu_percent": 12.3,
                "worker_memory_mb": 56.8,
                "worker_gpu_vram_mb": 1024.0,
            }
        },
    )

    summary = interactive._worker_status_summary(args, session_token="tok")

    assert summary["workers_count"] == 2
    assert summary["worker_cpu_percent"] == 12.3
    assert summary["worker_memory_mb"] == 56.8
    assert summary["worker_gpu_vram_mb"] == 1024.0


def test_resource_formatters_show_na_for_unknown_values() -> None:
    assert interactive._format_percent_or_na(None) == "N/A"
    assert interactive._format_mb_or_na(None) == "N/A"
    assert interactive._format_gb_from_mb_or_na(None) == "N/A"
    assert interactive._format_gb_from_mb_or_na(5120.0) == "5.0GB"
    assert interactive._resource_bits({"cpu_percent": None, "memory_mb": None, "gpu_vram_mb": None}) == [
        "cpu=N/A",
        "rss=N/A",
        "vram=N/A",
    ]
    assert interactive._resource_bits({"gpu_vram_mb": None, "gpu_vram_pending": True}) == ["vram=pending"]
    assert interactive._resource_bits({"gpu_vram_mb": 5120.0}) == ["vram=5.0GB"]
    assert interactive._resource_bits(
        {
            "workflow_python_capacity": 3,
            "workflow_python_active_process_count": 1,
            "workflow_python_process_count": 2,
            "workflow_python_cpu_percent": 7.5,
            "workflow_python_memory_mb": 128.0,
        }
    ) == ["py_nodes=1/2/3", "py_cpu=7.5%", "py_rss=128.0MB"]
    assert interactive._resource_bits(
        {
            "workflow_js_capacity": 4,
            "workflow_js_active_node_process_count": 2,
            "workflow_js_node_process_count": 3,
            "workflow_js_node_cpu_percent": 9.25,
            "workflow_js_node_memory_mb": 256.0,
        }
    ) == ["js_nodes=2/3/4", "js_cpu=9.2%", "js_rss=256.0MB"]


def test_operator_resource_kind_labels_all_worker_kinds() -> None:
    assert interactive._operator_resource_kind(
        {
            "worker_profile_class": "model",
            "command": ["python", "-m", "hosting.engine_worker_ipc"],
            "env": {"MP13_MODEL_PATH": "C:/models/demo"},
        }
    ) == "model instance"
    assert interactive._operator_resource_kind(
        {
            "worker_profile_class": "model",
            "sandbox_policy": {"sandbox": {"enabled": True}},
        }
    ) == "sandboxed model instance"
    assert interactive._operator_resource_kind({"worker_profile_class": "generic"}) == "generic worker"
    assert interactive._operator_resource_kind(
        {
            "worker_profile_class": "generic",
            "sandbox_policy": {"sandbox": {"enabled": True}},
        }
    ) == "sandboxed worker"
    assert interactive._operator_resource_kind(
        {
            "executor_kind": "toolbox_executor",
            "tool_access": {},
        }
    ) == "tools worker"
    assert interactive._operator_resource_kind(
        {
            "command": ["python", "-m", "hosting.toolbox_executor_ipc"],
            "sandbox_policy": {"sandbox": {"enabled": True}},
        }
    ) == "tools sandbox"
    assert interactive._operator_resource_kind({"executor_kind": "workflow_python_helper"}) == "workflow python worker"
    assert interactive._operator_resource_kind(
        {
            "command": ["python", "-m", "hosting.workflow_python_helper_ipc"],
            "sandbox_policy": {"sandbox": {"enabled": True}},
        }
    ) == "workflow python sandbox"
    assert interactive._operator_resource_kind({"executor_kind": "workflow_js_node"}) == "workflow js node worker"
    assert interactive._operator_resource_kind(
        {
            "command": ["python", "-m", "hosting.workflow_js_node_worker_ipc"],
            "sandbox_policy": {"sandbox": {"enabled": True}},
        }
    ) == "workflow js node sandbox"


def test_python_runtime_rows_show_daemon_and_engine_python() -> None:
    rows = interactive._python_runtime_rows(
        {
            "daemon_python_executable": "C:/daemon/python.exe",
            "engine_python_executable": "C:/engine/python.exe",
            "mp13_engine_python_env": "C:/engine/python.exe",
        }
    )

    assert rows == [
        ("Daemon Python", "C:/daemon/python.exe"),
        ("Engine Python", "C:/engine/python.exe (MP13_ENGINE_PYTHON=C:/engine/python.exe)"),
    ]


def test_python_runtime_rows_show_engine_python_default_source() -> None:
    rows = interactive._python_runtime_rows(
        {
            "daemon_python_executable": "C:/daemon/python.exe",
            "engine_python_executable": "C:/daemon/python.exe",
            "mp13_engine_python_env": None,
        }
    )

    assert rows == [
        ("Daemon Python", "C:/daemon/python.exe"),
        ("Engine Python", "C:/daemon/python.exe (MP13_ENGINE_PYTHON unset; using daemon Python)"),
    ]


def test_configured_model_path_from_config_row_reads_common_fields(tmp_path: Path) -> None:
    cfg = tmp_path / "demo.json"
    cfg.write_text('{"engine_params":{"base_model_path":"C:/models/demo"}}', encoding="utf-8")

    assert interactive._configured_model_path_from_config_row({"path": str(cfg)}) == "C:/models/demo"


def test_config_uses_generic_worker_detects_spawn_command(tmp_path: Path) -> None:
    cfg = tmp_path / "generic.json"
    cfg.write_text('{"spawn":{"command":["python","worker.py"]}}', encoding="utf-8")

    assert interactive._config_uses_generic_worker({"path": str(cfg)}) is True


def test_print_progress_snapshot_prefers_percent(capsys: pytest.CaptureFixture[str]) -> None:
    line = interactive._print_progress_snapshot(
        {"progress_percent": 74, "progress_text": "Loading model weights (74%)."}
    )

    out = capsys.readouterr().out
    assert "[###############.....]" in out
    assert "74%" in out
    assert "Loading model weights" in out
    assert "74%" in line


def test_print_progress_snapshot_without_percent_uses_zero_bar(capsys: pytest.CaptureFixture[str]) -> None:
    line = interactive._print_progress_snapshot({"progress_text": "Starting model load"})

    out = capsys.readouterr().out
    assert "[....................]" in out
    assert "0%" in out
    assert "?" not in out
    assert "Starting model load" in line


def test_operation_failure_message_reads_service_result() -> None:
    snap = {
        "status": "completed",
        "result": {
            "status": "failed",
            "reason": "worker_not_ready",
            "message": "worker RPC did not become ready",
        },
    }

    assert interactive._operation_failure_message(snap) == "worker RPC did not become ready"


def test_offline_read_authenticates_and_retries_when_token_required(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    calls: list[Optional[str]] = []

    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: True)

    def fake_invoke(_args: argparse.Namespace, _cmd: str, _payload: Dict[str, Any], *, session_token: Optional[str] = None) -> Any:
        calls.append(session_token)
        if not session_token:
            raise PermissionError("session_token_required")
        return [{"engine_id": "worker1", "state": "running", "kind": "engine"}]

    monkeypatch.setattr(interactive, "_offline_local_invoke", fake_invoke)
    monkeypatch.setattr(interactive, "_local_authenticate", lambda _args: "tok-123")

    token = interactive._list_engines(args, session_token=None)

    out = capsys.readouterr().out
    assert "protected by hosting auth" in out
    assert "worker1" in out
    assert calls == [None, "tok-123"]
    assert token == "tok-123"


def test_engine_details_uses_offline_fallback_when_local_daemon_stopped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        interactive,
        "_offline_local_invoke",
        lambda *_args, **_kwargs: [
            {
                "engine_id": "worker1",
                "state": "running",
                "kind": "engine",
                "pid": 123,
            }
        ],
    )
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: "worker1")

    interactive._engine_details(args, session_token=None)

    out = capsys.readouterr().out
    assert "offline fallback state" in out
    assert "worker1" in out
    assert "Raw State Info" in out


def test_list_engines_derives_labels_for_older_daemon_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        interactive,
        "_api_invoke",
        lambda *_args, **_kwargs: [
            {
                "engine_id": "model1",
                "pid": 123,
                "alive": True,
                "reachable": True,
                "command": ["python", "-m", "hosting.engine_worker_ipc"],
                "env": {"MP13_MODEL_PATH": "C:/models/demo"},
                "worker_profile_class": "model",
            },
            {
                "engine_id": "tools1",
                "pid": 456,
                "alive": True,
                "reachable": False,
                "command": ["python", "-m", "hosting.toolbox_executor_ipc"],
                "env": {"MP13_TOOLBOX_EXECUTOR_ENGINE_ID": "tools1"},
                "executor_kind": "toolbox_executor",
                "sandbox_policy": {"sandbox": {"enabled": True}},
            },
            {
                "engine_id": "wf-py",
                "pid": 789,
                "alive": True,
                "reachable": True,
                "executor_kind": "workflow_python_helper",
                "process_resources": {
                    "workflow_python_capacity": 3,
                    "workflow_python_active_process_count": 1,
                    "workflow_python_process_count": 2,
                    "workflow_python_cpu_percent": 7.5,
                    "workflow_python_memory_mb": 128.0,
                },
            },
            {
                "engine_id": "wf-js",
                "pid": 987,
                "alive": False,
                "reachable": False,
                "executor_kind": "workflow_js_node",
                "process_resources": {
                    "workflow_js_capacity": 4,
                    "workflow_js_active_node_process_count": 2,
                    "workflow_js_node_process_count": 3,
                    "workflow_js_node_cpu_percent": 9.25,
                    "workflow_js_node_memory_mb": 256.0,
                },
            },
        ],
    )

    interactive._list_engines(args, session_token=None)

    out = capsys.readouterr().out
    assert "model instance" in out
    assert "tools sandbox" in out
    assert "workflow python worker" in out
    assert "workflow js node worker" in out
    assert "pid=123" in out
    assert "reachable=no" in out
    assert "py_nodes=1/2/3" in out
    assert "py_cpu=7.5%" in out
    assert "py_rss=128.0MB" in out
    assert "js_nodes=2/3/4" in out
    assert "js_cpu=9.2%" in out
    assert "js_rss=256.0MB" in out
    assert "[stopped]" in out


def test_kill_resource_is_unavailable_when_local_daemon_stopped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    monkeypatch.setattr(interactive, "_can_use_offline_local_fallback", lambda *_args, **_kwargs: True)

    interactive._kill_resource(args, session_token=None)

    out = capsys.readouterr().out
    assert "Kill/disconnect actions require a running daemon" in out


def test_local_recovery_menu_can_update_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    choices = iter(["u", "b"])
    monkeypatch.setattr(interactive, "_target_mode", lambda _args: "local")
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: next(choices))
    monkeypatch.setattr(interactive, "_local_authenticate", lambda _args: "tok-123")

    token = interactive._local_recovery_menu(args, session_token=None)

    assert token == "tok-123"


def test_local_recovery_menu_rejects_remote_target(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    monkeypatch.setattr(interactive, "_target_mode", lambda _args: "ssh")

    token = interactive._local_recovery_menu(args, session_token="tok-existing")

    assert token == "tok-existing"
    assert "not available for remote targets" in capsys.readouterr().out


def test_revoke_local_session_uses_numbered_selection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    revoked: list[str] = []

    class FakeService:
        def auth_list_sessions(self) -> Dict[str, Any]:
            return {
                "sessions": [
                    {
                        "token_preview": "tok...123",
                        "key_id": "admin-main",
                        "scope": "control",
                        "role": "admin",
                    }
                ]
            }

        def auth_revoke_session(self, token: str) -> Dict[str, Any]:
            revoked.append(token)
            return {"revoked": True, "token": token}

    monkeypatch.setattr(interactive, "_offline_service", lambda _args: FakeService())
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: "tok...123")
    monkeypatch.setattr(interactive, "_confirm_local_mutation", lambda _prompt: True)

    interactive._revoke_local_session(args)

    assert revoked == ["tok...123"]
    assert "revoked" in capsys.readouterr().out


def test_revoke_local_key_uses_numbered_selection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)
    revoked: list[str] = []

    class FakeService:
        def auth_list_keys(self) -> list[Dict[str, Any]]:
            return [{"key_id": "admin-main", "role": "admin", "auth_method": "public_key", "disabled": False}]

        def auth_revoke_key(self, key_id: str) -> Dict[str, Any]:
            revoked.append(key_id)
            return {"revoked": True, "key_id": key_id}

    monkeypatch.setattr(interactive, "_offline_service", lambda _args: FakeService())
    monkeypatch.setattr(interactive, "_prompt_menu", lambda *_args, **_kwargs: "admin-main")
    monkeypatch.setattr(interactive, "_confirm_local_mutation", lambda _prompt: True)

    interactive._revoke_local_key(args)

    assert revoked == ["admin-main"]
    assert "admin-main" in capsys.readouterr().out
