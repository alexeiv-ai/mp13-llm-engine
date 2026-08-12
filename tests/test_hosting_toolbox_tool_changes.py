from __future__ import annotations

import pytest

from hosting.operation_contract import HostedOperationLifecycle
from hosting.service.toolbox_runtime import ToolboxRuntimeMixin
from hosting.toolbox.bundle_models import ToolboxDefinitionSpec
from hosting.toolbox.tool_changes import (
    ToolboxToolChange,
    deterministic_definition_changes,
    merge_toolbox_tool_changes,
)


def _auto(name: str, *, value: int = 1, module: str = "pkg.tools") -> dict:
    return {
        "files": [{"relative_path": f"pkg/{name.lower()}.py", "content": f"def {name}():\n    return {value}\n"}],
        "module_name": module,
        "callable_name": name,
        "dependency": {"mode": "auto", "template_id": None, "declared_imports": [], "package_requirements": []},
        "sandbox_policy": {"sandbox": {"enabled": True}},
        "activate": True,
        "hidden": False,
        "non_restartable": False,
        "guide_content": None,
        "guide_description": None,
        "callback_signature": None,
        "concurrency": None,
    }


def _manual(name: str, *, advertised: str | None = None) -> dict:
    return {
        "files": [{"relative_path": f"pkg/{name.lower()}.py", "content": f"def {name}():\n    return 1\n"}],
        "module_name": "pkg.manual",
        "callable_name": name,
        "tool_definition": {
            "type": "function",
            "function": {
                "name": advertised or name,
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        "dependency": {"mode": "template", "template_id": "core", "declared_imports": [], "package_requirements": []},
        "sandbox_policy": {"sandbox": {"enabled": True}},
        "hidden": False,
        "non_restartable": False,
        "callback_signature": None,
        "concurrency": None,
    }


def _definition(*, autos=(), manuals=(), expected_revision=None, intrinsics=()) -> ToolboxDefinitionSpec:
    return ToolboxDefinitionSpec.from_dict({
        "contract": "hosting.toolbox.definition",
        "toolbox_id": "demo",
        "expected_revision": expected_revision,
        "auto_requests": list(autos),
        "manual_requests": list(manuals),
        "intrinsics": {
            "names": list(intrinsics),
            "include_guides": False,
            "sandbox_policy": {"sandbox": {"enabled": True}},
        },
    })


def _change(change_id: str, kind: str, target, request_kind=None, request=None):
    return {
        "change_id": change_id,
        "kind": kind,
        "target_tool_key": target,
        "request_kind": request_kind,
        "request": request,
    }


def test_atomic_batch_merges_add_update_rename_and_remove() -> None:
    active = _definition(
        autos=(_auto("Alpha"), _auto("Beta"), _auto("Gamma")),
        manuals=(_manual("ManualOne"),),
    )
    revision = active.revision
    proposed, normalized = merge_toolbox_tool_changes(
        toolbox_id="demo",
        expected_revision=revision,
        active_revision=revision,
        active_definition=active,
        changes=(
            _change("update-alpha", "update", "pkg.tools:Alpha", "auto", _auto("Alpha", value=2)),
            _change("rename-beta", "rename", "pkg.tools:Beta", "auto", _auto("Delta")),
            _change("remove-gamma", "remove", "pkg.tools:Gamma"),
            _change("add-epsilon", "add", None, "auto", _auto("Epsilon")),
            _change("update-manual", "update", "manual:pkg.manual:ManualOne", "manual", _manual("ManualOne", advertised="ManualOneV2")),
        ),
    )

    assert proposed.expected_revision == revision
    assert {item.stable_key for item in proposed.auto_requests} == {
        "pkg.tools:Alpha", "pkg.tools:Delta", "pkg.tools:Epsilon"
    }
    assert proposed.auto_requests[0].to_dict() == _auto("Alpha", value=2)
    assert proposed.manual_requests[0].advertised_name == "ManualOneV2"
    assert [item.change_id for item in normalized] == sorted(item.change_id for item in normalized)
    assert next(item for item in normalized if item.change_id == "rename-beta").to_dict() == {
        "change_id": "rename-beta",
        "kind": "rename",
        "prior_tool_key": "pkg.tools:Beta",
        "tool_key": "pkg.tools:Delta",
        "request_kind": "auto",
    }


def test_atomic_rename_allows_two_keys_to_swap() -> None:
    active = _definition(autos=(_auto("Alpha"), _auto("Beta")))
    proposed, _normalized = merge_toolbox_tool_changes(
        toolbox_id="demo",
        expected_revision=active.revision,
        active_revision=active.revision,
        active_definition=active,
        changes=(
            _change("a-to-b", "rename", "pkg.tools:Alpha", "auto", _auto("Beta", value=10)),
            _change("b-to-a", "rename", "pkg.tools:Beta", "auto", _auto("Alpha", value=20)),
        ),
    )
    assert {item.callable_name: item.to_dict()["files"][0]["content"] for item in proposed.auto_requests} == {
        "Alpha": "def Alpha():\n    return 20\n",
        "Beta": "def Beta():\n    return 10\n",
    }


def test_stale_revision_and_result_conflict_fail_before_merge() -> None:
    active = _definition(autos=(_auto("Alpha"), _auto("Beta")))
    with pytest.raises(ValueError, match="tool_change_revision_conflict"):
        merge_toolbox_tool_changes(
            toolbox_id="demo",
            expected_revision="sha256:" + "f" * 64,
            active_revision=active.revision,
            active_definition=active,
            changes=(_change("remove", "remove", "pkg.tools:Alpha"),),
        )
    with pytest.raises(ValueError, match="tool_change_result_conflict"):
        merge_toolbox_tool_changes(
            toolbox_id="demo",
            expected_revision=active.revision,
            active_revision=active.revision,
            active_definition=active,
            changes=(_change("rename", "rename", "pkg.tools:Alpha", "auto", _auto("Beta")),),
        )


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ((), "tool_change_count_invalid"),
        ((_change("same", "remove", "pkg.tools:Alpha"), _change("same", "remove", "pkg.tools:Beta")), "tool_change_id_duplicate"),
        ((_change("one", "remove", "pkg.tools:Alpha"), _change("two", "update", "pkg.tools:Alpha", "auto", _auto("Alpha"))), "tool_change_target_duplicate"),
        ((_change("missing", "remove", "pkg.tools:Missing"),), "tool_change_target_not_found"),
        ((_change("update", "update", "pkg.tools:Alpha", "auto", _auto("Renamed")),), "tool_change_update_key_changed"),
        ((_change("rename", "rename", "pkg.tools:Alpha", "auto", _auto("Alpha", value=2)),), "tool_change_rename_key_unchanged"),
        ((_change("wrong-kind", "update", "pkg.tools:Alpha", "manual", _manual("Alpha")),), "tool_change_request_kind_conflict"),
    ],
)
def test_invalid_batches_fail_closed(changes, code: str) -> None:
    active = _definition(autos=(_auto("Alpha"), _auto("Beta")))
    with pytest.raises(ValueError, match=code):
        merge_toolbox_tool_changes(
            toolbox_id="demo",
            expected_revision=active.revision,
            active_revision=active.revision,
            active_definition=active,
            changes=changes,
        )


def test_change_payload_is_strict_for_add_and_remove() -> None:
    with pytest.raises(ValueError, match="tool_change_fields_invalid"):
        ToolboxToolChange.from_dict({**_change("add", "add", None, "auto", _auto("Alpha")), "extra": True})
    with pytest.raises(ValueError, match="tool_change_add_target_invalid"):
        ToolboxToolChange.from_dict(_change("add", "add", "pkg.tools:Alpha", "auto", _auto("Alpha")))
    with pytest.raises(ValueError, match="tool_change_remove_request_invalid"):
        ToolboxToolChange.from_dict(_change("remove", "remove", "pkg.tools:Alpha", "auto", _auto("Alpha")))


def test_complete_definition_change_ids_are_stable_and_cover_intrinsics() -> None:
    active = _definition(autos=(_auto("Alpha"),), intrinsics=("Clock",))
    proposed = _definition(
        autos=(_auto("Alpha", value=2), _auto("Beta")),
        expected_revision=active.revision,
        intrinsics=("Search",),
    )
    first = deterministic_definition_changes(active, proposed)
    second = deterministic_definition_changes(active, proposed)

    assert first == second
    assert len(first) == 4
    assert all(item.change_id.startswith("host:sha256:") and len(item.change_id) == 76 for item in first)
    assert {(item.kind, item.prior_tool_key, item.tool_key) for item in first} == {
        ("update", "pkg.tools:Alpha", "pkg.tools:Alpha"),
        ("add", None, "pkg.tools:Beta"),
        ("remove", "intrinsic:Clock", None),
        ("add", None, "intrinsic:Search"),
    }


def test_service_worker_merges_against_authoritative_revision_before_planning() -> None:
    active = _definition(autos=(_auto("Alpha"),))

    class Operations:
        def __init__(self) -> None:
            self.claimed = []

        def mark_dispatch_claimed(self, *, operation_id):
            self.claimed.append(operation_id)

        def finish(self, *, operation_id, lifecycle, envelope):
            return {
                "operation_id": operation_id,
                "lifecycle": lifecycle.value,
                "result": envelope,
            }

    class Service(ToolboxRuntimeMixin):
        def __init__(self) -> None:
            self._hosted_operations = Operations()
            self.planned = None

        def toolbox_get_definition(self, *, toolbox_id, **_kwargs):
            assert toolbox_id == "demo"
            return {
                "active_revision": active.revision,
                "definition": active.to_dict(),
            }

        def _build_toolbox_definition_plan(self, **kwargs):
            self.planned = kwargs
            return {
                "contract": "hosting.toolbox.definition_plan.v2",
                "proposal_kind": kwargs["proposal_kind"],
                "changes": [item.to_dict() for item in kwargs["normalized_changes"]],
            }

    service = Service()
    result = service._run_toolbox_tool_changes_plan(
        operation_id="op-1",
        toolbox_id="demo",
        expected_revision=active.revision,
        changes=[_change("rename-alpha", "rename", "pkg.tools:Alpha", "auto", _auto("Beta"))],
        operator_details=False,
        owner_actor_id="actor:one",
        authority_id="workspace:one",
        ttl_ms=60_000,
    )

    assert service._hosted_operations.claimed == ["op-1"]
    assert result["lifecycle"] == HostedOperationLifecycle.TERMINAL_SUCCESS.value
    assert result["result"]["proposal_kind"] == "tool_changes"
    assert result["result"]["changes"][0]["change_id"] == "rename-alpha"
    assert ToolboxDefinitionSpec.from_dict(service.planned["definition"]).auto_requests[0].stable_key == "pkg.tools:Beta"


def test_service_worker_returns_stable_conflict_without_calling_planner() -> None:
    active = _definition(autos=(_auto("Alpha"),))

    class Operations:
        def mark_dispatch_claimed(self, **_kwargs):
            return None

        def finish(self, *, operation_id, lifecycle, envelope):
            return {"lifecycle": lifecycle.value, "result": envelope}

    class Service(ToolboxRuntimeMixin):
        _hosted_operations = Operations()

        def toolbox_get_definition(self, **_kwargs):
            return {"active_revision": active.revision, "definition": active.to_dict()}

        def _build_toolbox_definition_plan(self, **_kwargs):
            raise AssertionError("stale batch reached planner")

    result = Service()._run_toolbox_tool_changes_plan(
        operation_id="op-2",
        toolbox_id="demo",
        expected_revision="sha256:" + "f" * 64,
        changes=[_change("remove-alpha", "remove", "pkg.tools:Alpha")],
        operator_details=False,
        owner_actor_id="actor:one",
        authority_id="workspace:one",
        ttl_ms=60_000,
    )
    assert result == {
        "lifecycle": HostedOperationLifecycle.TERMINAL_FAILURE.value,
        "result": {
            "contract": "hosting.toolbox.definition_plan_failure.v1",
            "status": "failed",
            "code": "tool_change_conflict",
        },
    }
