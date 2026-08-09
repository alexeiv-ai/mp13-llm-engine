from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from hosting.daemon.local_ipc import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel
from hosting import engine_host_cli
from hosting.service.host_service import EngineHostService
from hosting.toolbox.hosted_ref import HostedToolBoxRef


class _Connection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def invoke(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.calls.append((command, dict(payload or {})))
        return {"command": command}

    def is_alive(self) -> bool:
        return True

    def close(self) -> None:
        return None


def test_definition_channel_forwards_exact_commands_and_payloads() -> None:
    connection = _Connection()
    channel = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    channel._get_connection = lambda: connection  # type: ignore[method-assign]
    channel.set_session_token("token-1")

    definition = {"contract": "hosting.toolbox.definition", "toolbox_id": "tb"}
    channel.toolbox_get_definition(toolbox_id="tb")
    channel.toolbox_plan_definition(definition=definition, ttl_ms=42)
    channel.toolbox_approve_definition_plan(plan_id="plan-1")
    channel.toolbox_apply_definition(
        definition=definition,
        plan_id="plan-1",
        request_id="request-1",
        dependency_approval_ref="opaque-approval",
    )

    assert [command for command, _ in connection.calls] == [
        "toolbox-get-definition",
        "toolbox-plan-definition",
        "toolbox-approve-definition-plan",
        "toolbox-apply-definition",
    ]
    assert all(payload["session_token"] == "token-1" for _, payload in connection.calls)
    assert connection.calls[1][1]["ttl_ms"] == 42
    assert connection.calls[3][1]["dependency_approval_ref"] == "opaque-approval"


def test_hosted_reference_exposes_only_definition_and_template_consumer_helpers() -> None:
    class Host:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Dict[str, Any]]] = []

        def __getattr__(self, name: str):
            def invoke(**payload: Any) -> Dict[str, Any]:
                self.calls.append((name, dict(payload)))
                return {"method": name}

            return invoke

    host = Host()
    ref = HostedToolBoxRef(toolbox_id="tb", host=host)
    definition = {"toolbox_id": "tb"}

    ref.get_definition()
    ref.plan_definition(definition, ttl_ms=99)
    ref.approve_definition_plan(plan_id="plan-1")
    ref.apply_definition(
        definition=definition,
        plan_id="plan-1",
        request_id="request-1",
        dependency_approval_ref="approval-1",
    )
    ref.list_environment_templates()
    ref.describe_environment_template(template_id="core", template_digest="digest-1")

    assert [name for name, _ in host.calls] == [
        "toolbox_get_definition",
        "toolbox_plan_definition",
        "toolbox_approve_definition_plan",
        "toolbox_apply_definition",
        "toolbox_template_list",
        "toolbox_template_describe",
    ]
    assert host.calls[3][1]["request_id"] == "request-1"
    assert not hasattr(type(ref), "publish_template")


def test_definition_role_separation() -> None:
    definition_commands = {
        "toolbox-get-definition",
        "toolbox-plan-definition",
        "toolbox-approve-definition-plan",
        "toolbox-apply-definition",
    }
    for role in ("worker_user", "config_editor", "admin"):
        assert definition_commands <= EngineHostService._commands_allowed_for_role(role)  # noqa: SLF001
    diagnostics = EngineHostService._commands_allowed_for_role("diagnostic_user")  # noqa: SLF001
    assert "toolbox-get-definition" in diagnostics
    assert not (definition_commands - {"toolbox-get-definition"}) & diagnostics
    worker = EngineHostService._commands_allowed_for_role("worker_user")  # noqa: SLF001
    assert "toolbox-template-publish" not in worker


def test_daemon_dispatch_preserves_actor_and_opaque_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
    )
    calls: list[Dict[str, Any]] = []

    def apply(**payload: Any) -> Dict[str, Any]:
        calls.append(dict(payload))
        return {"status": "accepted"}

    monkeypatch.setattr(daemon.svc, "toolbox_apply_definition", apply)
    response = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            json.dumps(
                {
                    "seq": 1,
                    "cmd": "toolbox-apply-definition",
                    "payload": {
                        "definition": {"toolbox_id": "tb"},
                        "plan_id": "plan-1",
                        "request_id": "request-1",
                        "dependency_approval_ref": {"forged": True},
                    },
                }
            ),
            peer_host="127.0.0.1",
            transport="local_ipc",
        )
    )
    assert response["ok"] is True
    assert calls[0]["owner_actor_id"] == "backend:unknown"
    assert calls[0]["authority_id"] == calls[0]["owner_actor_id"]
    assert calls[0]["dependency_approval_ref"] == {"forged": True}


def test_remote_cli_routes_definition_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Dict[str, Any]]] = []

    class FakeRemoteChannel:
        def __init__(self, _settings=None):
            pass

        def invoke_control_command(self, command: str, payload=None):
            calls.append((command, dict(payload or {})))
            return {"status": "ok"}

    monkeypatch.setattr("hosting.engine_host_channel.EngineHostControlChannel", FakeRemoteChannel)
    payloads = {
        "toolbox-get-definition": {"toolbox_id": "tb"},
        "toolbox-plan-definition": {"definition": {"toolbox_id": "tb"}},
        "toolbox-approve-definition-plan": {"plan_id": "plan-1"},
        "toolbox-apply-definition": {
            "definition": {"toolbox_id": "tb"},
            "plan_id": "plan-1",
            "request_id": "request-1",
        },
    }
    for command, payload in payloads.items():
        monkeypatch.setattr("sys.stdin.read", lambda value=json.dumps(payload): value)
        assert engine_host_cli.main(
            ["--ssh-target", "user@example.test", "--payload-stdin", command]
        ) == 0

    assert [command for command, _ in calls] == list(payloads)
    assert all(payload == payloads[command] for command, payload in calls)
    assert capsys.readouterr().out.count('"ok": true') == 4
