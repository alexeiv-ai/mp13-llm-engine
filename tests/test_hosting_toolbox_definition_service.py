from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from hosting.service.host_service import EngineHostService
from hosting.toolbox.dependency_policy import ToolboxDependencyPolicy
from hosting.toolbox.identity import identity_digest
from hosting_toolbox_test_catalog import realized_test_catalog
from test_hosting_toolbox_definition_resolution import _service_with_verified_closure


def _dependency(*, imports=(), requirements=()):
    return {
        "mode": "auto",
        "template_id": None,
        "declared_imports": list(imports),
        "package_requirements": list(requirements),
    }


def _definition(*, toolbox_id="demo", tool_name="Alpha", expected_revision=None, dependency=None, source=None):
    return {
        "contract": "hosting.toolbox.definition",
        "toolbox_id": toolbox_id,
        "expected_revision": expected_revision,
        "auto_requests": [
            {
                "files": [
                    {
                        "relative_path": "pkg/tool.py",
                        "content": source or f"def {tool_name}():\n    return 1\n",
                    }
                ],
                "module_name": "pkg.tool",
                "callable_name": tool_name,
                "dependency": dependency or _dependency(),
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


def _service(tmp_path: Path, *, policy: ToolboxDependencyPolicy | None = None) -> EngineHostService:
    service, _template = _service_with_verified_closure(tmp_path, policy=policy)
    return service


def _custom_policy() -> ToolboxDependencyPolicy:
    shipped = realized_test_catalog()
    python_abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform = "win_amd64" if os.name == "nt" else "manylinux_2_28_x86_64"
    payload = {
        "allowed_template_ids": tuple(item.template_id for item in shipped.templates),
        "allowed_targets": (f"{python_abi}-{platform}",),
        "package_allowlist": ("requests",),
        "package_denylist": (),
        "allow_custom": True,
        "custom_requires_approval": True,
        "online_resolution_allowed": False,
        "allowed_index_origins": (),
    }
    return ToolboxDependencyPolicy(
        revision=identity_digest("test.toolbox.policy", payload),
        **payload,
    )


def test_authoritative_read_is_side_effect_free_and_plan_is_actor_owned(tmp_path: Path) -> None:
    service = _service(tmp_path)
    before = set(tmp_path.rglob("*"))

    snapshot = service.toolbox_get_definition(
        toolbox_id="demo", owner_actor_id="actor:a", authority_id="workspace:a"
    )

    assert snapshot["active_revision"] is None
    assert snapshot["definition"]["expected_revision"] is None
    assert snapshot["active_tools"] == []
    assert set(tmp_path.rglob("*")) == before

    plan = service.toolbox_plan_definition(
        definition=_definition(), owner_actor_id="actor:a", authority_id="workspace:a"
    )
    assert plan["contract"] == "hosting.toolbox.definition_plan.v2"
    assert plan["can_apply"] is False
    assert plan["confirmation_required"] is True
    assert plan["approval_required"] is False
    assert "profile_id" not in str(plan)
    assert "environment_key" not in str(plan)
    with pytest.raises(PermissionError, match="toolbox_definition_plan_not_found"):
        service.toolbox_apply_definition(
            definition=_definition(),
            plan_id=plan["plan_id"],
            request_id="apply-a",
            owner_actor_id="actor:b",
            authority_id="workspace:a",
        )
    assert service._read_engines() == []
    assert not (tmp_path / "state" / "toolbox_sandboxes_v2.json").exists()


def test_apply_returns_immediately_reuses_request_and_rolls_out_once(tmp_path: Path) -> None:
    service = _service(tmp_path)
    definition = _definition()
    plan = service.toolbox_plan_definition(
        definition=definition, owner_actor_id="actor:a", authority_id="workspace:a"
    )
    entered = threading.Event()
    release = threading.Event()
    dispatches: list[str] = []

    def fake_apply(*, draft, profile_changes, operation_id):
        dispatches.append(operation_id)
        service._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        entered.set()
        assert release.wait(2)
        return service._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle="terminal_success",
            envelope={
                "contract": "hosting.toolbox.definition_apply_result",
                "status": "ok",
                "code": "definition_apply_succeeded",
                "active_revision": draft.definition.revision,
            },
        )

    service._apply_resolved_toolbox_definition = fake_apply  # type: ignore[method-assign]
    started = service.toolbox_apply_definition(
        definition=definition,
        plan_id=plan["plan_id"],
        request_id="apply-stable",
        owner_actor_id="actor:a",
        authority_id="workspace:a",
    )
    assert entered.wait(2)
    duplicate = service.toolbox_apply_definition(
        definition=definition,
        plan_id=plan["plan_id"],
        request_id="apply-stable",
        owner_actor_id="actor:a",
        authority_id="workspace:a",
    )

    assert started["operation"] == duplicate["operation"]
    assert started["lifecycle"] == "queued"
    assert duplicate["lifecycle"] == "running"
    assert len(dispatches) == 1
    release.set()
    deadline = time.time() + 2
    terminal = duplicate
    while time.time() < deadline:
        terminal = service.hosted_operation_status(
            ref=started["operation"], owner_actor_id="actor:a"
        )
        if terminal["lifecycle"] == "terminal_success":
            break
        time.sleep(0.01)
    assert terminal["lifecycle"] == "terminal_success"
    recovered = service.hosted_operation_resolve_request(
        execution_kind="toolbox_definition_apply",
        selector={"kind": "toolbox_id", "id": "demo"},
        request_id="apply-stable",
        owner_actor_id="actor:a",
    )
    assert recovered == terminal


def test_custom_delta_requires_exact_parent_approval_and_consumption_is_request_bound(tmp_path: Path) -> None:
    service = _service(tmp_path, policy=_custom_policy())
    definition = _definition(
        dependency=_dependency(imports=("requests",), requirements=("requests==2.32.5",)),
        source="import requests\ndef Alpha():\n    return requests.__name__\n",
    )
    plan = service.toolbox_plan_definition(
        definition=definition, owner_actor_id="actor:a", authority_id="workspace:a"
    )
    assert plan["approval_required"] is True
    with pytest.raises(PermissionError, match="toolbox_definition_plan_not_found"):
        service.toolbox_approve_definition_plan(
            plan_id=plan["plan_id"], owner_actor_id="actor:b", authority_id="workspace:a"
        )
    approval = service.toolbox_approve_definition_plan(
        plan_id=plan["plan_id"], owner_actor_id="actor:a", authority_id="workspace:a"
    )
    assert approval["approval_ref"].startswith("approval_")
    with pytest.raises(ValueError, match="dependency_approval_ref_must_be_opaque_string"):
        service.toolbox_apply_definition(
            definition=definition,
            plan_id=plan["plan_id"],
            request_id="custom-1",
            dependency_approval_ref={"approved": True},  # type: ignore[arg-type]
            owner_actor_id="actor:a",
            authority_id="workspace:a",
        )
    entered = threading.Event()
    release = threading.Event()

    def fake_apply(*, draft, profile_changes, operation_id):
        service._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        entered.set()
        assert release.wait(2)
        return service._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle="terminal_success",
            envelope={"status": "ok", "active_revision": draft.definition.revision},
        )

    service._apply_resolved_toolbox_definition = fake_apply  # type: ignore[method-assign]
    started = service.toolbox_apply_definition(
        definition=definition,
        plan_id=plan["plan_id"],
        request_id="custom-1",
        dependency_approval_ref=approval["approval_ref"],
        owner_actor_id="actor:a",
        authority_id="workspace:a",
    )
    assert entered.wait(2)
    duplicate = service.toolbox_apply_definition(
        definition=definition,
        plan_id=plan["plan_id"],
        request_id="custom-1",
        dependency_approval_ref=approval["approval_ref"],
        owner_actor_id="actor:a",
        authority_id="workspace:a",
    )
    assert duplicate["operation"] == started["operation"]
    with pytest.raises(PermissionError, match="dependency_approval_invalid"):
        service.toolbox_apply_definition(
            definition=definition,
            plan_id=plan["plan_id"],
            request_id="custom-2",
            dependency_approval_ref=approval["approval_ref"],
            owner_actor_id="actor:a",
            authority_id="workspace:a",
        )
    release.set()
