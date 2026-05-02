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


def test_interactive_lifecycle_uses_channel_helpers(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _FakeChannel.instances.clear()
    monkeypatch.setattr(interactive, "EngineHostControlChannel", _FakeChannel)
    args = argparse.Namespace(pid_file=None, engines_state_file=None, control_state_file=None)

    interactive._start_daemon(args)
    interactive._stop_daemon(args)

    out = capsys.readouterr().out
    assert "Daemon started" in out
    assert "Daemon stop signal sent" in out
