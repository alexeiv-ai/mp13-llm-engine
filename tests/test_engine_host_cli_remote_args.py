from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pytest

from hosting import engine_host_cli


class _FakeChannel:
    instances: list["_FakeChannel"] = []

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})
        self.invocations: list[tuple[str, Dict[str, Any]]] = []
        _FakeChannel.instances.append(self)

    def invoke_control_command(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.invocations.append((command, dict(payload or {})))
        return {"command": command, "payload": dict(payload or {}), "target": self.settings.get("engine_host_ssh_target")}

    def reset_hosting_access(self) -> Dict[str, Any]:
        return {"status": "ok"}

    def force_stop_daemon(self, *, stop_workers: bool = True) -> Dict[str, Any]:
        return {"status": "ok", "stop_workers": stop_workers}

    def force_restart_daemon(self) -> Dict[str, Any]:
        return {"status": "ok", "restarted": True}


def test_cli_remote_target_routes_noninteractive_command_through_channel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _FakeChannel.instances.clear()
    monkeypatch.setattr("hosting.engine_host_channel.EngineHostControlChannel", _FakeChannel)

    rc = engine_host_cli.main(
        [
            "--ssh-target",
            "user@example-host",
            "--control-ssh-key",
            "C:/keys/id_ed25519",
            "--ssh-known-hosts-line",
            "example-host ssh-ed25519 AAAATEST",
            "host-metrics",
        ]
    )

    assert rc == 0
    assert len(_FakeChannel.instances) == 1
    channel = _FakeChannel.instances[0]
    assert channel.settings["engine_host_ssh_target"] == "user@example-host"
    assert channel.settings["control_ssh_key"] == "C:/keys/id_ed25519"
    assert channel.settings["ssh_known_hosts_line"] == "example-host ssh-ed25519 AAAATEST"
    assert channel.invocations == [("host-metrics", {})]
    assert '"ok": true' in capsys.readouterr().out


def test_cli_remote_target_includes_subcommand_selectors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _FakeChannel.instances.clear()
    monkeypatch.setattr("hosting.engine_host_channel.EngineHostControlChannel", _FakeChannel)

    rc = engine_host_cli.main(["--ssh-target", "user@example-host", "shutdown", "--engine-id", "worker1"])

    assert rc == 0
    assert _FakeChannel.instances[0].invocations == [("shutdown", {"engine_id": "worker1"})]
    assert '"engine_id": "worker1"' in capsys.readouterr().out


def test_cli_rejects_remote_reset_hosting_access(capsys: pytest.CaptureFixture[str]) -> None:
    rc = engine_host_cli.main(["--ssh-target", "user@example-host", "reset-hosting-access"])

    assert rc == 2
    assert "local-only" in capsys.readouterr().out


def test_cli_exposes_local_force_restart_daemon(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _FakeChannel.instances.clear()
    monkeypatch.setattr("hosting.engine_host_channel.EngineHostControlChannel", _FakeChannel)

    rc = engine_host_cli.main(["force-restart-daemon"])

    assert rc == 0
    assert '"restarted": true' in capsys.readouterr().out


def test_cli_rejects_remote_force_stop_daemon(capsys: pytest.CaptureFixture[str]) -> None:
    rc = engine_host_cli.main(["--ssh-target", "user@example-host", "force-stop-daemon"])

    assert rc == 2
    assert "local-only" in capsys.readouterr().out


def test_cli_local_workflow_python_resources_uses_facade_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: Dict[str, Any] = {}

    class FakeService:
        def __init__(self, **kwargs: Any) -> None:
            seen["init"] = dict(kwargs)

        def authorize_command(self, command: str, payload: Dict[str, Any]) -> None:
            seen["authorized"] = (command, dict(payload))

        def workflow_python_resources(self, **kwargs: Any) -> Dict[str, Any]:
            seen["resources"] = dict(kwargs)
            return {"status": "ok", "environment_key": kwargs.get("environment_key")}

    monkeypatch.setattr(engine_host_cli, "_try_daemon_invoke", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(engine_host_cli, "EngineHostService", FakeService)

    rc = engine_host_cli.main(
        [
            "--payload-json",
            json.dumps(
                {
                    "profile": "helper",
                    "environment_key": "env-demo",
                    "engine_id": "wf-py",
                    "python": {"import_allowlist": ["json"]},
                    "sandbox_policy": {"sandbox": {"enabled": True}},
                }
            ),
            "workflow-python-resources",
        ]
    )

    assert rc == 0
    assert seen["resources"] == {
        "profile": "helper",
        "environment_name": "workflow-python-helper",
        "environment_key": "env-demo",
        "engine_id": "wf-py",
        "python": {"import_allowlist": ["json"]},
        "sandbox_policy": {"sandbox": {"enabled": True}},
    }
    assert '"environment_key": "env-demo"' in capsys.readouterr().out


def test_cli_local_workflow_python_capacity_and_cancel_use_facade(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, Dict[str, Any]]] = []

    class FakeService:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def authorize_command(self, command: str, payload: Dict[str, Any]) -> None:
            calls.append(("authorize", {"command": command, **dict(payload)}))

        def set_workflow_python_capacity(self, **kwargs: Any) -> Dict[str, Any]:
            calls.append(("capacity", dict(kwargs)))
            return {"status": "ok", "capacity": kwargs.get("capacity")}

        def cancel_workflow_python_request(self, **kwargs: Any) -> Dict[str, Any]:
            calls.append(("cancel", dict(kwargs)))
            return {"status": "ok", "request_id": kwargs.get("request_id")}

    monkeypatch.setattr(engine_host_cli, "_try_daemon_invoke", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(engine_host_cli, "EngineHostService", FakeService)

    resize_rc = engine_host_cli.main(
        [
            "--payload-json",
            json.dumps({"environment_key": "env-demo", "engine_id": "wf-py", "capacity": 7}),
            "workflow-python-set-capacity",
        ]
    )
    cancel_rc = engine_host_cli.main(
        [
            "--payload-json",
            json.dumps({"environment_key": "env-demo", "engine_id": "wf-py", "request_id": "req-1"}),
            "workflow-python-cancel-request",
        ]
    )

    assert resize_rc == 0
    assert cancel_rc == 0
    assert ("capacity", {"profile": "helper", "environment_key": "env-demo", "engine_id": "wf-py", "capacity": 7}) in calls
    assert ("cancel", {"profile": "helper", "environment_key": "env-demo", "engine_id": "wf-py", "request_id": "req-1"}) in calls
    out = capsys.readouterr().out
    assert '"capacity": 7' in out
    assert '"request_id": "req-1"' in out
