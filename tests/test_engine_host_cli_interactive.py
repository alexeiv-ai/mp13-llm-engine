from __future__ import annotations

import argparse
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
        ],
    )

    interactive._list_engines(args, session_token=None)

    out = capsys.readouterr().out
    assert "model instance" in out
    assert "tools sandbox" in out
    assert "pid=123" in out
    assert "reachable=no" in out


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
