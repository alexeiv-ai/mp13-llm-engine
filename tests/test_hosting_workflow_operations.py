from __future__ import annotations

from pathlib import Path

import pytest

from hosting.operation_contract import hosted_execution_fingerprint
from hosting.service.host_service import EngineHostService
from tests.hosting_v3_fixtures import hosting_configuration


def _service(tmp_path: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        hosting_configuration=hosting_configuration(tmp_path),
    )


def _stub_environment_specs(service: EngineHostService, monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "workflow_python_environment_spec",
        lambda **_kwargs: {"environment_key": "python-env", "environment": {"kind": "python"}},
    )
    monkeypatch.setattr(
        service,
        "workflow_js_environment_spec",
        lambda **_kwargs: {"environment_key": "js-env", "environment": {"kind": "javascript"}},
    )


@pytest.mark.parametrize(
    ("runtime", "execute_name", "runtime_name", "request_payload", "expected_kind"),
    [
        (
            "python",
            "execute_workflow_python",
            "_execute_workflow_python_runtime",
            {"request_id": "request-python", "python": {"execution_mode": "snippet"}, "source": "result = 7"},
            "workflow_python",
        ),
        (
            "javascript",
            "execute_workflow_js",
            "_execute_workflow_js_runtime",
            {"request_id": "request-js", "javascript": {"execution_mode": "snippet"}, "source": "return 7"},
            "workflow_js",
        ),
    ],
)
def test_workflow_execute_attach_replay_and_conflict_share_canonical_contract(
    tmp_path: Path,
    monkeypatch,
    runtime: str,
    execute_name: str,
    runtime_name: str,
    request_payload: dict,
    expected_kind: str,
) -> None:
    service = _service(tmp_path)
    _stub_environment_specs(service, monkeypatch)
    dispatches: list[dict] = []

    def execute_runtime(**kwargs):
        dispatches.append(kwargs)
        return {"status": "ok", "ok": True, "output": {"answer": 7}, "request_id": request_payload["request_id"]}

    monkeypatch.setattr(service, runtime_name, execute_runtime)
    execute = getattr(service, execute_name)
    first = execute(request=request_payload)
    replay = execute(request=request_payload)
    changed = {**request_payload, "source": "changed"}
    conflict = execute(request=changed)

    assert len(dispatches) == 1
    assert first["contract"] == replay["contract"] == "hosting.operation_status"
    assert first["lifecycle"] == replay["lifecycle"] == "terminal_success"
    assert first["operation"] == replay["operation"]
    assert first["operation"]["execution_kind"] == expected_kind
    assert first["operation"]["selector"]["kind"] == "engine_id"
    assert first["result"]["output"]["answer"] == 7
    assert conflict["api_status"] == "error"
    assert conflict["lifecycle"] == "idempotency_conflict"


@pytest.mark.parametrize(
    ("runtime", "request_payload", "runtime_method"),
    [
        ("python", {"request_id": "replay-python"}, "_execute_workflow_python_runtime"),
        ("javascript", {"request_id": "replay-js"}, "_execute_workflow_js_runtime"),
    ],
)
def test_workflow_terminal_replay_survives_service_recreation_without_runtime_dispatch(
    tmp_path: Path, monkeypatch, runtime: str, request_payload: dict, runtime_method: str
) -> None:
    first = _service(tmp_path)
    _stub_environment_specs(first, monkeypatch)
    monkeypatch.setattr(first, runtime_method, lambda **_kwargs: {"status": "ok", "ok": True, "output": 7})
    execute_name = "execute_workflow_python" if runtime == "python" else "execute_workflow_js"
    terminal = getattr(first, execute_name)(request=request_payload)

    repository_key = str((tmp_path / "state" / "hosted_operations.json").resolve())
    EngineHostService._operation_repositories.pop(repository_key, None)
    recreated = _service(tmp_path)
    _stub_environment_specs(recreated, monkeypatch)
    dispatches: list[dict] = []
    monkeypatch.setattr(recreated, runtime_method, lambda **kwargs: dispatches.append(kwargs) or {"status": "ok"})
    replay = getattr(recreated, execute_name)(request=request_payload)

    assert replay["lifecycle"] == "terminal_success"
    assert replay["operation"] == terminal["operation"]
    assert replay["result"]["output"] == 7
    assert dispatches == []


@pytest.mark.parametrize("runtime", ["python", "javascript"])
def test_generic_workflow_cancel_routes_from_stored_ref_only(tmp_path: Path, monkeypatch, runtime: str) -> None:
    service = _service(tmp_path)
    execution_kind = "workflow_python" if runtime == "python" else "workflow_js"
    engine_id = f"workflow-{runtime}-engine"
    prepared = service._hosted_operations.prepare(
        owner_actor_id="actor:a",
        execution_kind=execution_kind,
        selector={"kind": "engine_id", "id": engine_id},
        namespace=f"{execution_kind}:{engine_id}",
        request_id=f"cancel-{runtime}",
        fingerprint=hosted_execution_fingerprint({"runtime": runtime}),
        metadata={
            "runtime": runtime,
            "engine_id": engine_id,
            "environment_key": f"{runtime}-env",
            "profile": "node",
        },
    )
    operation_id = prepared["status"]["operation"]["operation_id"]
    service._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
    cancel_method = "_cancel_workflow_python_runtime" if runtime == "python" else "_cancel_workflow_js_runtime"
    calls: list[dict] = []
    monkeypatch.setattr(
        service,
        cancel_method,
        lambda **kwargs: calls.append(kwargs) or {"status": "ok", "canceled": True},
    )

    canceled = service.hosted_operation_cancel(
        ref=prepared["status"]["operation"], reason="workspace_unload", owner_actor_id="actor:a"
    )
    assert canceled["lifecycle"] == "terminal_cancellation"
    assert canceled["reason"] == "workspace_unload"
    assert calls == [
        {
            "profile": "node",
            "environment_key": f"{runtime}-env",
            "engine_id": engine_id,
            "request_id": f"cancel-{runtime}",
        }
    ]


def test_workflow_prepare_conflict_and_replay_do_not_start_worker_or_sandbox(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    _stub_environment_specs(service, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        service,
        "_execute_workflow_python_runtime",
        lambda **_kwargs: calls.append("dispatch") or {"status": "ok", "ok": True},
    )
    request = {"request_id": "no-restart", "source": "result = 1"}
    first = service.execute_workflow_python(request=request)
    replay = service.execute_workflow_python(request=request)
    conflict = service.execute_workflow_python(request={**request, "source": "result = 2"})

    assert first["lifecycle"] == replay["lifecycle"] == "terminal_success"
    assert conflict["lifecycle"] == "idempotency_conflict"
    assert calls == ["dispatch"]
