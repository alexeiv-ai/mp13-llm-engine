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
from tests.hosting_v3_fixtures import write_hosting_configuration


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
    channel.toolbox_plan_definition(definition=definition, request_id="plan-1", ttl_ms=42)
    channel.toolbox_plan_tool_changes(
        toolbox_id="tb",
        expected_revision="sha256:" + "a" * 64,
        changes=[{
            "change_id": "remove-old",
            "kind": "remove",
            "target_tool_key": "pkg.tools:Old",
            "request_kind": None,
            "request": None,
        }],
        request_id="changes-1",
    )
    channel.toolbox_revise_definition_plan(
        plan_id="plan-1",
        decisions=[{
            "change_id": "remove-old",
            "decision": "exclude",
            "denied_import_roots": [],
        }],
        request_id="revise-1",
    )
    channel.toolbox_confirm_definition_plan(
        plan_id="plan-1", environment_choices=[], request_id="confirm-1"
    )
    channel.toolbox_approve_confirmed_definition_plan(
        confirmation_ref="confirmation-1"
    )
    channel.toolbox_apply_definition(
        plan_id="plan-1",
        confirmation_ref="confirmation-1",
        request_id="request-1",
        dependency_approval_ref="opaque-approval",
    )
    channel.toolbox_prepare_definition_candidate(
        plan_id="plan-1",
        confirmation_ref="confirmation-1",
        request_id="candidate-1",
        dependency_approval_ref="opaque-approval",
        requested_lifetime_ms=600_000,
    )
    channel.toolbox_get_definition_candidate(candidate_ref="candidate-1")
    channel.toolbox_renew_definition_candidate(
        candidate_ref="candidate-1", requested_lifetime_ms=900_000, request_id="renew-1"
    )
    channel.toolbox_execute_definition_candidate(
        candidate_ref="candidate-1",
        tool_call={"name": "Fetch", "arguments": {}},
        execution_request_id="execute-1",
        timeout_seconds=12.0,
        tools_view={"allowed_tool_names": ["Fetch"]},
        callback_binding={"callback_id": "callback-1"},
        host_api_approval={"approval_id": "approval-1"},
    )
    channel.toolbox_publish_definition_candidate(
        candidate_ref="candidate-1", request_id="publish-1"
    )
    channel.toolbox_discard_definition_candidate(
        candidate_ref="candidate-2", request_id="discard-1"
    )

    assert [command for command, _ in connection.calls] == [
        "toolbox-get-definition",
        "op-start",
        "op-start",
        "op-start",
        "op-start",
        "toolbox-approve-confirmed-definition-plan",
        "op-start",
        "op-start",
        "toolbox-get-definition-candidate",
        "toolbox-renew-definition-candidate",
        "toolbox-execute-definition-candidate",
        "op-start",
        "op-start",
    ]
    assert all(payload["session_token"] == "token-1" for _, payload in connection.calls)
    assert connection.calls[1][1]["payload"]["ttl_ms"] == 42
    assert connection.calls[2][1]["command"] == "toolbox-plan-tool-changes"
    assert connection.calls[3][1]["command"] == "toolbox-revise-definition-plan"
    assert connection.calls[6][1]["payload"]["dependency_approval_ref"] == "opaque-approval"
    assert connection.calls[7][1]["command"] == "toolbox-prepare-definition-candidate"
    assert connection.calls[7][1]["payload"]["requested_lifetime_ms"] == 600_000
    assert connection.calls[8][1]["candidate_ref"] == "candidate-1"
    assert connection.calls[9][1]["requested_lifetime_ms"] == 900_000
    assert connection.calls[10][1]["execution_request_id"] == "execute-1"
    assert connection.calls[10][1]["host_api_approval"]["approval_id"] == "approval-1"
    assert connection.calls[11][1]["command"] == "toolbox-publish-definition-candidate"
    assert connection.calls[12][1]["command"] == "toolbox-discard-definition-candidate"


def test_operation_watch_emits_changed_snapshots_and_stops_at_terminal() -> None:
    channel = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    snapshots = iter(
        [
            {"updated_at_ms": 1, "lifecycle": "queued"},
            {"updated_at_ms": 1, "lifecycle": "queued"},
            {"updated_at_ms": 2, "lifecycle": "running"},
            {"updated_at_ms": 3, "lifecycle": "terminal_success"},
        ]
    )
    channel.get_host_operation_status = (  # type: ignore[method-assign]
        lambda **_kwargs: next(snapshots)
    )

    changed = channel.watch_host_operation(
        operation_id="op-test", timeout_seconds=1, poll_interval_seconds=0.01
    )

    assert [item["updated_at_ms"] for item in changed] == [1, 2, 3]


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
    ref.plan_definition(definition, request_id="plan-1", ttl_ms=99)
    ref.plan_tool_changes(
        [{
            "change_id": "remove-old", "kind": "remove",
            "target_tool_key": "pkg.tools:Old", "request_kind": None, "request": None,
        }],
        expected_revision="sha256:" + "a" * 64,
        request_id="changes-1",
    )
    ref.revise_definition_plan(
        plan_id="plan-1",
        decisions=[{
            "change_id": "remove-old", "decision": "exclude",
            "denied_import_roots": [],
        }],
        request_id="revise-1",
    )
    ref.confirm_definition_plan(
        plan_id="plan-1", environment_choices=[], request_id="confirm-1"
    )
    ref.apply_definition(
        plan_id="plan-1",
        confirmation_ref="confirmation-1",
        request_id="request-1",
        dependency_approval_ref="approval-1",
    )
    ref.list_environment_templates()
    ref.describe_environment_template(template_id="core", template_digest="digest-1")

    assert [name for name, _ in host.calls] == [
        "toolbox_get_definition",
        "toolbox_plan_definition",
        "toolbox_plan_tool_changes",
        "toolbox_revise_definition_plan",
        "toolbox_confirm_definition_plan",
        "toolbox_apply_definition",
        "toolbox_template_list",
        "toolbox_template_describe",
    ]
    assert host.calls[5][1]["request_id"] == "request-1"
    assert not hasattr(type(ref), "publish_template")


def test_definition_role_separation() -> None:
    definition_commands = {
        "toolbox-get-definition",
        "toolbox-plan-definition",
        "toolbox-plan-tool-changes",
        "toolbox-revise-definition-plan",
        "toolbox-confirm-definition-plan",
        "toolbox-apply-definition",
    }
    for role in ("worker_user", "config_editor", "admin"):
        assert definition_commands <= EngineHostService._commands_allowed_for_role(role)  # noqa: SLF001
    diagnostics = EngineHostService._commands_allowed_for_role("diagnostic_user")  # noqa: SLF001
    assert "toolbox-get-definition" in diagnostics
    assert not (definition_commands - {"toolbox-get-definition"}) & diagnostics
    worker = EngineHostService._commands_allowed_for_role("worker_user")  # noqa: SLF001
    assert "toolbox-template-publish" not in worker
    approval_command = "toolbox-approve-confirmed-definition-plan"
    assert approval_command not in worker
    assert approval_command not in EngineHostService._commands_allowed_for_role("config_editor")  # noqa: SLF001
    assert approval_command in EngineHostService._commands_allowed_for_role("dependency_approver")  # noqa: SLF001


def test_daemon_dispatch_preserves_actor_and_opaque_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = write_hosting_configuration(tmp_path)
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        mp13_config_file=config_file,
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
                        "cmd": "op-start",
                        "payload": {
                            "command": "toolbox-apply-definition",
                            "payload": {
                                "plan_id": "plan-1",
                                "confirmation_ref": "confirmation-1",
                                "request_id": "request-1",
                                "dependency_approval_ref": {"forged": True},
                            },
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
        "toolbox-plan-definition": {"request_id": "plan-1", "definition": {"toolbox_id": "tb"}},
        "toolbox-plan-tool-changes": {
            "toolbox_id": "tb",
            "expected_revision": "sha256:" + "a" * 64,
            "changes": [],
            "request_id": "changes-1",
            "operator_details": False,
        },
        "toolbox-revise-definition-plan": {
            "plan_id": "plan-1",
            "decisions": [],
            "request_id": "revise-1",
            "operator_details": False,
        },
        "toolbox-approve-confirmed-definition-plan": {"confirmation_ref": "confirmation-1"},
        "toolbox-apply-definition": {
            "plan_id": "plan-1",
            "confirmation_ref": "confirmation-1",
            "request_id": "request-1",
        },
        "toolbox-prepare-definition-candidate": {
            "plan_id": "plan-1",
            "confirmation_ref": "confirmation-1",
            "request_id": "candidate-1",
            "dependency_approval_ref": None,
            "requested_lifetime_ms": 600_000,
        },
        "toolbox-get-definition-candidate": {"candidate_ref": "candidate-1"},
        "toolbox-renew-definition-candidate": {
            "candidate_ref": "candidate-1",
            "requested_lifetime_ms": 900_000,
            "request_id": "renew-1",
        },
        "toolbox-execute-definition-candidate": {
            "candidate_ref": "candidate-1",
            "tool_call": {"name": "Fetch", "arguments": {}},
            "execution_request_id": "execute-1",
            "timeout_seconds": 12.0,
            "tools_view": None,
            "callback_binding": None,
            "host_api_approval": None,
        },
        "toolbox-publish-definition-candidate": {
            "candidate_ref": "candidate-1", "request_id": "publish-1",
        },
        "toolbox-discard-definition-candidate": {
            "candidate_ref": "candidate-2", "request_id": "discard-1",
        },
    }
    for command, payload in payloads.items():
        monkeypatch.setattr("sys.stdin.read", lambda value=json.dumps(payload): value)
        assert engine_host_cli.main(
            ["--ssh-target", "user@example.test", "--payload-stdin", command]
        ) == 0

    assert [command for command, _ in calls] == [
        "toolbox-get-definition",
        "op-start",
        "op-start",
        "op-start",
        "toolbox-approve-confirmed-definition-plan",
        "op-start",
        "op-start",
        "toolbox-get-definition-candidate",
        "toolbox-renew-definition-candidate",
        "toolbox-execute-definition-candidate",
        "op-start",
        "op-start",
    ]
    assert calls[1][1] == {
        "command": "toolbox-plan-definition",
        "payload": payloads["toolbox-plan-definition"],
    }
    assert calls[2][1] == {
        "command": "toolbox-plan-tool-changes",
        "payload": payloads["toolbox-plan-tool-changes"],
    }
    assert calls[3][1] == {
        "command": "toolbox-revise-definition-plan",
        "payload": payloads["toolbox-revise-definition-plan"],
    }
    assert calls[5][1] == {
        "command": "toolbox-apply-definition",
        "payload": payloads["toolbox-apply-definition"],
    }
    assert calls[0][1] == payloads["toolbox-get-definition"]
    assert calls[4][1] == payloads["toolbox-approve-confirmed-definition-plan"]
    assert capsys.readouterr().out.count('"ok": true') == len(payloads)
