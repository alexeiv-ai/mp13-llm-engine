from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from hosting import engine_host_cli
from hosting.daemon import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel
from hosting.model_runtime_contract import (
    MODEL_RUNTIME_STATUS_FIELDS,
    ModelRuntimeIdentity,
    ModelRuntimeStatus,
    reject_model_runtime_selection,
)
from hosting.sandbox.python_runtime import HostedPythonRuntimeManager
from hosting.service.host_service import EngineHostService
from hosting.toolbox.bundle_models import ToolboxBundleFile
from hosting.toolbox.dependency_analysis import (
    analyze_toolbox_bundle_imports,
    resolve_toolbox_dependencies,
    select_toolbox_environment_template,
)
from hosting.toolbox.dependency_policy import (
    ToolboxDependencyPolicyError,
    ToolboxDependencyPolicy,
    validate_toolbox_dependency_policy,
)
from hosting_toolbox_test_catalog import realized_test_catalog


def _digest(char: str) -> str:
    return f"sha256:{char * 64}"


def _identity(*, verified: bool = True) -> ModelRuntimeIdentity:
    return ModelRuntimeIdentity(
        python_abi="cp312",
        platform="win_amd64",
        engine_artifact_digest=_digest("a"),
        complete_lock_digest=_digest("b"),
        optional_package_set="cuda-12.6",
        materialization_revision="model-runtime-2026.08.08",
        verified=verified,
        updated_at_ms=1786233600000,
    )


def _service(tmp_path: Path, *, verified: bool = True) -> EngineHostService:
    return EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "access_control.json",
        model_runtime_identity=_identity(verified=verified),
    )


def test_model_runtime_status_is_exact_bounded_and_read_only(tmp_path: Path) -> None:
    service = _service(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    status = service.model_runtime_status()
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert set(status) == MODEL_RUNTIME_STATUS_FIELDS
    assert ModelRuntimeStatus(**status).to_dict() == status
    assert status == {
        "state": "ready",
        "code": "model_runtime_ready",
        "summary": "The exclusive model runtime is verified and ready for authorized model operations.",
        "python_abi": "cp312",
        "platform": "win_amd64",
        "engine_artifact_digest": _digest("a"),
        "complete_lock_digest": _digest("b"),
        "optional_package_set": "cuda-12.6",
        "materialization_revision": "model-runtime-2026.08.08",
        "updated_at_ms": 1786233600000,
    }
    assert before == after
    serialized = json.dumps(status)
    for forbidden in ["environment_name", "environment_key", "venv", "python_executable", "package_path", "activation"]:
        assert forbidden not in serialized


def test_unconfigured_and_unverified_statuses_remain_bounded(tmp_path: Path) -> None:
    unconfigured = EngineHostService(
        engines_state_file=tmp_path / "a" / "engines.json",
        control_state_file=tmp_path / "a" / "access_control.json",
    ).model_runtime_status()
    assert set(unconfigured) == MODEL_RUNTIME_STATUS_FIELDS
    assert unconfigured["state"] == "unavailable"
    assert unconfigured["code"] == "model_runtime_unconfigured"
    degraded = _service(tmp_path / "b", verified=False).model_runtime_status()
    assert degraded["state"] == "degraded"
    assert degraded["code"] == "model_runtime_verification_failed"


@pytest.mark.parametrize(
    "payload",
    [
        {"model_runtime": {"id": "installed"}},
        {"template_id": "model-runtime"},
        {"environment_name": "C:/venvs/model-runtime"},
        {"python_executable": "C:/venvs/model/python.exe"},
        {"runtime_kind": "model"},
        {"worker_profile_class": "model"},
    ],
)
def test_generic_selection_guard_rejects_every_model_selector(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="model_runtime_selection_denied"):
        reject_model_runtime_selection(payload)


def test_healthy_installed_model_cannot_be_selected_by_planner_or_custom_builder(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.model_runtime_status()["state"] == "ready"
    with pytest.raises(ValueError, match="model_runtime_selection_denied"):
        service.resolve_hosted_template_environment(
            consumer_kind="toolbox",
            files=[{"relative_path": "main.py", "content": "import json\n"}],
            python_abi="cp312",
            platform="win_amd64",
            allowed_template_ids=["model-runtime"],
        )

    manager = HostedPythonRuntimeManager(tmp_path)
    model_environment = tmp_path / "venvs" / "model-runtime"
    model_environment.mkdir(parents=True)
    with pytest.raises(ValueError, match="model_runtime_selection_denied"):
        manager.environment_spec(
            environment_name=str(model_environment),
            profile="helper",
            python_policy={"python_executable": str(model_environment / "python.exe")},
        )
    with pytest.raises(PermissionError, match="model_runtime_selection_denied"):
        service.authorize_command(
            "workflow-python-prepare-environment",
            {"environment_name": str(model_environment), "python": {}},
        )
    with pytest.raises(PermissionError, match="model_runtime_selection_denied"):
        service.authorize_command(
            "toolbox-template-publish",
            {"template": {"runtime_kind": "model"}},
        )


def test_dependency_payload_cannot_smuggle_model_runtime_authority() -> None:
    template = realized_test_catalog().release("core").template
    dependencies = resolve_toolbox_dependencies(
        analyze_toolbox_bundle_imports(
            [ToolboxBundleFile(relative_path="main.py", content="import json\n")]
        )
    )
    selection = select_toolbox_environment_template(
        dependencies, [template], python_abi="cp312", platform="win_amd64"
    )
    policy = ToolboxDependencyPolicy(
        revision=_digest("c"),
        allowed_template_ids=("core",),
        allowed_targets=("cp312-win_amd64",),
        package_allowlist=(),
        package_denylist=(),
        allow_custom=False,
        custom_requires_approval=True,
        online_resolution_allowed=False,
        allowed_index_origins=(),
    )
    with pytest.raises(ToolboxDependencyPolicyError) as denied:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            policy,
            python_abi="cp312",
            platform="win_amd64",
            dependency_payload={"mode": "auto", "model_runtime": "installed"},
        )
    assert denied.value.code == "dependency_payload_authority_forbidden"
    assert "model_runtime" in denied.value.summary


def test_status_command_roles_channel_and_daemon_projection(tmp_path: Path) -> None:
    for role in ["admin", "config_editor", "worker_user", "model_user_with_model_control", "model_user", "diagnostic_user"]:
        assert "model-runtime-status" in EngineHostService._commands_allowed_for_role(role)  # noqa: SLF001

    calls: list[tuple[str, dict[str, Any]]] = []

    class Connection:
        def invoke(self, command: str, payload: dict[str, Any]):
            calls.append((command, payload))
            return _service(tmp_path / "channel").model_runtime_status()

        def close(self):
            return None

    channel = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    channel._get_connection = lambda: Connection()  # type: ignore[method-assign]
    channel.set_session_token("status-token")
    assert channel.model_runtime_status()["state"] == "ready"
    assert calls == [("model-runtime-status", {"session_token": "status-token"})]

    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "daemon-engines.json",
        control_state_file=tmp_path / "daemon-access-control.json",
    )
    daemon.svc._model_runtime_identity = _identity()  # noqa: SLF001
    response = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            json.dumps({"seq": 1, "cmd": "model-runtime-status", "payload": {}}),
            peer_host="127.0.0.1",
            transport="local_ipc",
        )
    )
    assert response["ok"] is True
    assert set(response["result"]) == MODEL_RUNTIME_STATUS_FIELDS
    assert response["result"]["state"] == "ready"


def test_ssh_cli_routes_read_only_model_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class RemoteChannel:
        def __init__(self, _settings=None):
            pass

        def invoke_control_command(self, command: str, payload=None):
            calls.append((command, dict(payload or {})))
            return ModelRuntimeStatus(
                state="unavailable",
                code="model_runtime_unconfigured",
                summary="The exclusive model runtime is not configured on this host.",
                python_abi=None,
                platform=None,
                engine_artifact_digest=None,
                complete_lock_digest=None,
                optional_package_set=None,
                materialization_revision=None,
                updated_at_ms=1,
            ).to_dict()

    monkeypatch.setattr("hosting.engine_host_channel.EngineHostControlChannel", RemoteChannel)
    assert engine_host_cli.main(
        ["--ssh-target", "admin@example.test", "model-runtime-status"]
    ) == 0
    assert calls == [("model-runtime-status", {})]
    assert '"state": "unavailable"' in capsys.readouterr().out
