from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import pytest

from hosting.service.host_service import EngineHostService
from hosting.toolbox.bundle_models import ResolvedToolboxProfileSpec, ToolboxDefinitionSpec
from hosting.toolbox.dependency_policy import ToolboxDependencyPolicy
from hosting.toolbox.hosted_ref import HostedToolBoxRef
from hosting.toolbox.identity import identity_digest
from test_hosting_toolbox_definition_resolution import _service_with_verified_closure
from test_hosting_toolbox_definition_service import _custom_policy


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _definition(toolbox_id: str, *, source_value: str = "v1") -> dict:
    return {
        "contract": "hosting.toolbox.definition",
        "toolbox_id": toolbox_id,
        "expected_revision": None,
        "auto_requests": [
            {
                "files": [
                    {
                        "relative_path": "pkg/shared.py",
                        "content": f"def Shared():\n    return {source_value!r}\n",
                    }
                ],
                "module_name": "pkg.shared",
                "callable_name": "Shared",
                "dependency": {
                    "mode": "auto",
                    "template_id": None,
                    "declared_imports": [],
                    "package_requirements": [],
                },
                "sandbox_policy": {"sandbox": {"enabled": True}},
                "activate": True,
                "hidden": False,
                "non_restartable": False,
                "guide_content": None,
                "guide_description": None,
                "callback_signature": None,
                "concurrency": None,
            }
        ],
        "manual_requests": [],
        "intrinsics": {
            "names": [],
            "include_guides": False,
            "sandbox_policy": {"sandbox": {"enabled": True}},
        },
    }


def _service(tmp_path: Path) -> EngineHostService:
    return _service_with_verified_closure(tmp_path)[0]


def _custom_service(tmp_path: Path) -> EngineHostService:
    return _service_with_verified_closure(tmp_path, policy=_custom_policy())[0]


def _custom_definition(toolbox_id: str) -> dict:
    definition = _definition(toolbox_id)
    request = definition["auto_requests"][0]
    request["files"][0]["content"] = "import requests\ndef Shared():\n    return requests.__name__\n"
    request["dependency"] = {
        "mode": "auto",
        "template_id": None,
        "declared_imports": ["requests"],
        "package_requirements": ["requests==2.32.5"],
    }
    return definition


def _plan_result(service: EngineHostService, definition: dict, request_id: str, ttl_ms=900000) -> dict:
    started = service.toolbox_plan_definition(
        definition=definition, request_id=request_id,
        owner_actor_id="actor:a", authority_id="workspace:a", ttl_ms=ttl_ms,
    )
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"], timeout_seconds=10
    )
    assert terminal["lifecycle"] == "terminal_success"
    return terminal["result"]


def _confirmation_result(service: EngineHostService, plan: dict, request_id: str) -> dict:
    choices = [{
        "environment_id": item["environment_id"],
        "alternative_id": item["preferred_alternative_id"],
        "accept_package_changes": True,
    } for item in plan["environment_mutations"]]
    started = service.toolbox_confirm_definition_plan(
        plan_id=plan["plan_id"], environment_choices=choices, request_id=request_id,
        owner_actor_id="actor:a", authority_id="workspace:a",
    )
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"], timeout_seconds=10
    )
    assert terminal["lifecycle"] == "terminal_success"
    return terminal["result"]


def _install_toolbox(service: EngineHostService, toolbox_id: str, engine_id: str, character: str) -> None:
    definition = ToolboxDefinitionSpec.from_dict(_definition(toolbox_id))
    profile = ResolvedToolboxProfileSpec(
        environment_key=_digest(character),
        template_id="core",
        template_lock_digest=_digest("f"),
        custom_resolved_lock_digest=None,
        sandbox_policy={"sandbox": {"enabled": True}},
        assigned_tool_keys=("pkg.shared:Shared",),
        resolved_import_roots=(),
    )
    reference = f"toolbox:{toolbox_id}:{profile.profile_id}:{definition.revision}"
    service._toolbox_state_v2.publish(
        toolbox_id=toolbox_id,
        expected_revision=None,
        definition=definition.to_dict(),
        profiles={
            profile.profile_id: {
                "profile": profile.to_dict(),
                "manifest_hash": _digest(character),
                "engine_id": engine_id,
                "tool_names": ["Shared"],
                "environment_reference": reference,
                "resolved_environment": {},
            }
        },
        tool_routes={
            "Shared": {
                "profile_id": profile.profile_id,
                "engine_id": engine_id,
                "non_restartable": False,
            }
        },
        environment_references=[reference],
        published_at_ms=1,
    )
    service.register_spawned(
        engine_id=engine_id,
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        routing_state="active",
        bundle={
            "toolbox_id": toolbox_id,
            "resolved_profile_id": profile.profile_id,
            "sandbox_profile_id": profile.profile_id,
            "definition_revision": definition.revision,
            "manifest_hash": _digest(character),
        },
        environment={"environment_key": profile.environment_key},
        tool_access={"allowed_tool_names": ["Shared"], "advertised_tool_names": ["Shared"]},
    )


def test_plan_definition_expiry_and_authoritative_pin_changes_fail_closed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    definition = _definition("approval")
    plan = _plan_result(service, definition, "plan-pins")
    with pytest.raises(PermissionError, match="toolbox_confirmation_not_found"):
        service.toolbox_apply_definition(
            plan_id=plan["plan_id"],
            confirmation_ref="confirmation_fabricated",
            request_id="fabricated-confirmation",
            owner_actor_id="actor:a",
            authority_id="workspace:a",
        )
    confirmation = _confirmation_result(service, plan, "confirm-pins")

    original_context = service._toolbox_definition_planning_context

    def changed_context():
        context = original_context()
        context["policy"] = dataclasses.replace(
            context["policy"], revision=identity_digest("test.changed.policy", {"v": 2})
        )
        return context

    service._toolbox_definition_planning_context = changed_context  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="toolbox_definition_plan_pins_changed"):
        service.toolbox_apply_definition(
            plan_id=plan["plan_id"],
            confirmation_ref=confirmation["confirmation_ref"],
            request_id="changed-policy",
            owner_actor_id="actor:a",
            authority_id="workspace:a",
        )

    service = _service(tmp_path / "expiry")
    expiring = _plan_result(service, _definition("expiry"), "plan-expiry", ttl_ms=1)
    time.sleep(0.01)
    with pytest.raises(ValueError, match="toolbox_definition_plan_expired"):
        service.toolbox_confirm_definition_plan(
            plan_id=expiring["plan_id"],
            environment_choices=[], request_id="confirm-expired",
            owner_actor_id="actor:a",
            authority_id="workspace:a",
        )


def test_custom_approval_is_bound_to_exact_plan_definition_and_delta(tmp_path: Path) -> None:
    service = _custom_service(tmp_path)
    first = _custom_definition("custom-one")
    second = _custom_definition("custom-two")
    first_plan = _plan_result(service, first, "plan-first")
    second_plan = _plan_result(service, second, "plan-second")
    first_confirmation = _confirmation_result(service, first_plan, "confirm-first")
    second_confirmation = _confirmation_result(service, second_plan, "confirm-second")
    approval = service.toolbox_approve_confirmed_definition_plan(
        confirmation_ref=first_confirmation["confirmation_ref"],
        approver_actor_id="approver:dependencies",
        dependency_approver_authorized=True,
    )

    with pytest.raises(PermissionError, match="dependency_approval_invalid"):
        service.toolbox_apply_definition(
            plan_id=second_plan["plan_id"],
            confirmation_ref=second_confirmation["confirmation_ref"],
            request_id="wrong-plan-delta",
            dependency_approval_ref=approval["approval_ref"],
            owner_actor_id="actor:a",
            authority_id="workspace:a",
        )


def test_multi_toolbox_references_execute_concurrently_and_keep_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    _install_toolbox(service, "one", "engine-one", "a")
    _install_toolbox(service, "two", "engine-two", "b")
    barrier = threading.Barrier(2)

    def fake_ipc_call(*, reg, payload, **_kwargs):
        barrier.wait(timeout=3)
        toolbox_id = reg["bundle"]["toolbox_id"]
        return {
            "status": "ok",
            "tool_call": {
                **dict(payload["params"]["tool_call"]),
                "result": toolbox_id,
            },
        }

    monkeypatch.setattr(service, "_ipc_call", fake_ipc_call)
    refs = {name: HostedToolBoxRef(toolbox_id=name, host=service) for name in ("one", "two")}
    results: dict[str, dict] = {}

    def execute(name: str) -> None:
        results[name] = refs[name].execute(
            tool_name="Shared",
            arguments={},
            execution_request_id=f"execute-{name}",
        )

    workers = [threading.Thread(target=execute, args=(name,)) for name in refs]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()

    assert results["one"]["operation"]["selector"] == {"kind": "toolbox_id", "id": "one"}
    assert results["two"]["operation"]["selector"] == {"kind": "toolbox_id", "id": "two"}
    assert results["one"]["result"]["tool_call"]["result"] == "one"
    assert results["two"]["result"]["tool_call"]["result"] == "two"
    for result in results.values():
        serialized = str(result)
        for forbidden in ("engine-one", "engine-two", "environment_key", "profile_id", "hosted_pool"):
            assert forbidden not in serialized
        assert result["result"]["user_projection"]["code"] == "toolbox_execution_succeeded"

    before_two = service._toolbox_state_v2.get("two")
    current_one = service._toolbox_state_v2.get("one")
    updated = ToolboxDefinitionSpec.from_dict(
        {**_definition("one", source_value="v2"), "expected_revision": current_one["active_revision"]}
    )
    service._toolbox_state_v2.publish(
        toolbox_id="one",
        expected_revision=current_one["active_revision"],
        definition=updated.to_dict(),
        profiles=current_one["profiles"],
        tool_routes=current_one["tool_routes"],
        environment_references=current_one["environment_references"],
        published_at_ms=2,
    )
    assert service._toolbox_state_v2.get("two") == before_two


def test_consumer_describe_projection_hides_runtime_placement(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_toolbox(service, "projection", "engine-secret", "c")

    described = HostedToolBoxRef(toolbox_id="projection", host=service).describe()

    assert described["user_projection"] == {
        "state": "ready",
        "code": "toolbox_runtime_ready",
        "summary": "The toolbox runtime is ready.",
    }
    serialized = str(described)
    for forbidden in (
        "engine-secret",
        "engine_ids",
        "sandbox_profile_ids",
        "environment_key",
        "profile_id",
        "hosted_pools",
        "installer",
    ):
        assert forbidden not in serialized
