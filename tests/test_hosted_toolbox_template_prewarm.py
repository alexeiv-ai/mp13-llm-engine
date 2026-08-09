from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import pytest

from hosting.daemon import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel
from hosting.operation_contract import HostedExecutionKind, HostedOperationLifecycle
from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_materialization import (
    ToolboxTemplateMaterializationError,
    ToolboxTemplateMaterializationReceipt,
    derived_environment_digest,
)


SIGNATURE = "A" * 86


def _digest(char: str) -> str:
    return f"sha256:{char * 64}"


def _template() -> dict[str, Any]:
    return {
        "template_id": "core",
        "python_requires": ">=3.12,<3.13",
        "python_abis": ["cp312"],
        "runtime_kind": "toolbox_python",
        "worker_protocol_version": "1.0",
        "platforms": ["win_amd64"],
        "locked_distributions": [{"name": "hosting-runtime", "version": "1.0", "extras": []}],
        "exposed_import_roots": ["hosting", "mp13_engine"],
        "lock_digest": _digest("a"),
        "parent_worker_artifact_digest": _digest("b"),
        "isolation_policy_version": "1.0",
        "provenance": {
            "source": "release",
            "revision": "1",
            "manifest_digest": _digest("c"),
            "signing_key_id": "release-key-1",
        },
    }


def _artifact() -> dict[str, Any]:
    return {
        "source_id": "release-artifacts",
        "filename": "hosting-runtime.whl",
        "sha256": _digest("e"),
        "size_bytes": 1234,
    }


class VerifiedMaterializer:
    def __init__(self) -> None:
        self.calls = 0
        self.entered = threading.Event()

    def materialize(self, *, catalog_entry: Mapping[str, Any], python_abi: str, platform: str, progress):
        self.calls += 1
        self.entered.set()
        progress("artifact_verification", "artifacts_verified", 1, 1, "All locked artifacts were verified.", True)
        progress("environment_build", "environment_built", 1, 1, "The isolated environment was built.", True)
        progress("import_probe", "imports_verified", 2, 2, "All declared import roots were probed.", False)
        artifact_digests = tuple(sorted(item["sha256"] for item in catalog_entry["artifacts"]))
        return ToolboxTemplateMaterializationReceipt(
            template_id=catalog_entry["template_id"],
            template_digest=catalog_entry["template_digest"],
            python_abi=python_abi,
            platform=platform,
            environment_digest=derived_environment_digest(
                template_digest=catalog_entry["template_digest"],
                python_abi=python_abi,
                platform=platform,
                artifact_digests=artifact_digests,
            ),
            artifact_digests=artifact_digests,
            verified_import_roots=tuple(sorted(catalog_entry["template"]["exposed_import_roots"])),
            verified_at_ms=int(time.time() * 1000),
            verifier="test-verifier-v1",
        )


class FailingMaterializer:
    def materialize(self, **_kwargs):
        raise ToolboxTemplateMaterializationError(
            "offline_artifact_unavailable",
            "A locked artifact is unavailable from configured offline sources.",
        )


def _service(tmp_path: Path, materializer) -> EngineHostService:
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "access_control.json",
        toolbox_template_materializer=materializer,
    )
    service.toolbox_template_publish(
        template=_template(),
        artifact_references=[_artifact()],
        manifest_signature=SIGNATURE,
        activate=True,
        actor_id="admin:test",
    )
    return service


def _start(service: EngineHostService, request_id: str = "prewarm-1") -> dict[str, Any]:
    return service.toolbox_template_prewarm(
        template_id="core",
        python_abi="cp312",
        platform="win_amd64",
        request_id=request_id,
        owner_actor_id="admin:test",
    )


def _terminal(service: EngineHostService, started: Mapping[str, Any]) -> dict[str, Any]:
    operation_id = started["operation"]["operation_id"]
    return service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=operation_id, timeout_seconds=5
    )


def test_prewarm_returns_durable_ref_and_advertises_only_verified_receipt(tmp_path: Path) -> None:
    materializer = VerifiedMaterializer()
    service = _service(tmp_path, materializer)
    before = service.toolbox_template_describe(template_id="core")
    assert before["materialization"] == "not_materialized"
    started = _start(service)
    assert started["operation"]["execution_kind"] == HostedExecutionKind.TOOLBOX_TEMPLATE_PREWARM.value
    assert started["operation"]["selector"] == {"kind": "template_id", "id": "core"}
    terminal = _terminal(service, started)
    assert terminal["lifecycle"] == HostedOperationLifecycle.TERMINAL_SUCCESS.value
    assert terminal["progress"]["code"] == "verification_receipt_committed"
    assert terminal["progress"]["cancellable"] is False
    assert terminal["result"]["code"] == "template_materialization_verified"
    assert service.toolbox_template_describe(template_id="core")["materialization"] == "ready"

    restarted = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    assert restarted.toolbox_template_describe(template_id="core")["user_projection"]["state"] == "ready"


def test_failed_or_incomplete_verification_never_advertises_ready(tmp_path: Path) -> None:
    service = _service(tmp_path, FailingMaterializer())
    terminal = _terminal(service, _start(service))
    assert terminal["lifecycle"] == HostedOperationLifecycle.TERMINAL_FAILURE.value
    assert terminal["reason"] == "offline_artifact_unavailable"
    assert terminal["result"] == {
        "status": "error",
        "code": "offline_artifact_unavailable",
        "summary": "A locked artifact is unavailable from configured offline sources.",
    }
    assert service.toolbox_template_describe(template_id="core")["materialization"] == "not_materialized"


def test_prewarm_request_is_idempotent_and_changed_target_conflicts(tmp_path: Path) -> None:
    materializer = VerifiedMaterializer()
    service = _service(tmp_path, materializer)
    first = _start(service)
    terminal = _terminal(service, first)
    replay = _start(service)
    assert replay["operation"] == terminal["operation"]
    assert replay["lifecycle"] == HostedOperationLifecycle.TERMINAL_SUCCESS.value
    assert materializer.calls == 1
    with pytest.raises(ValueError, match="template_target_unsupported"):
        service.toolbox_template_prewarm(
            template_id="core",
            python_abi="cp312",
            platform="manylinux_2_28_x86_64",
            request_id="prewarm-1",
            owner_actor_id="admin:test",
        )


def test_default_host_materializer_fails_closed_with_terminal_diagnostic(tmp_path: Path) -> None:
    service = _service(tmp_path, None)
    terminal = _terminal(service, _start(service))
    assert terminal["reason"] == "template_materializer_unconfigured"
    assert terminal["result"]["code"] == "template_materializer_unconfigured"
    assert service.toolbox_template_describe(template_id="core")["materialization"] == "not_materialized"


def test_role_separation_and_channel_payload() -> None:
    for role in ["worker_user", "config_editor", "diagnostic_user"]:
        assert "toolbox-template-prewarm" not in EngineHostService._commands_allowed_for_role(role)  # noqa: SLF001
    assert "toolbox-template-prewarm" in EngineHostService._commands_allowed_for_role("admin")  # noqa: SLF001

    calls: list[tuple[str, dict[str, Any]]] = []

    class Connection:
        def invoke(self, command: str, payload: dict[str, Any]):
            calls.append((command, payload))
            return {"operation": {}}

        def close(self):
            return None

    channel = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    channel._get_connection = lambda: Connection()  # type: ignore[method-assign]
    channel.set_session_token("admin-token")
    channel.toolbox_template_prewarm(
        template_id="core",
        template_digest=_digest("f"),
        python_abi="cp312",
        platform="win_amd64",
        request_id="prewarm-remote-1",
    )
    assert calls == [
        (
            "toolbox-template-prewarm",
            {
                "template_id": "core",
                "template_digest": _digest("f"),
                "python_abi": "cp312",
                "platform": "win_amd64",
                "request_id": "prewarm-remote-1",
                "session_token": "admin-token",
            },
        )
    ]


def test_daemon_dispatch_runs_target_host_service_method(tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    daemon.svc._toolbox_template_materializer = VerifiedMaterializer()  # noqa: SLF001
    daemon.svc.toolbox_template_publish(
        template=_template(),
        artifact_references=[_artifact()],
        manifest_signature=SIGNATURE,
        activate=True,
    )
    response = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            json.dumps(
                {
                    "seq": 1,
                    "cmd": "toolbox-template-prewarm",
                    "payload": {
                        "template_id": "core",
                        "python_abi": "cp312",
                        "platform": "win_amd64",
                        "request_id": "daemon-prewarm-1",
                    },
                }
            ),
            peer_host="127.0.0.1",
            transport="local_ipc",
        )
    )
    assert response["ok"] is True
    assert response["result"]["operation"]["execution_kind"] == "toolbox_template_prewarm"
    assert _terminal(daemon.svc, response["result"])["lifecycle"] == "terminal_success"
