from __future__ import annotations

import threading
from pathlib import Path

import pytest

from hosting.operation_contract import hosted_execution_fingerprint
from hosting.service.host_service import EngineHostService
from hosting.toolbox.bundle_models import (
    ResolvedToolboxProfileSpec,
    ToolboxBundleAutoTool,
    ToolboxBundleFile,
    ToolboxBundleSpec,
    ToolboxDefinitionSpec,
)
from hosting.toolbox.definition_planner import ToolboxDefinitionPlanDraft
from hosting.toolbox.orchestration import ToolboxSandboxOrchestrator


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _draft(tool_name: str | None, character: str, expected_revision: str | None) -> ToolboxDefinitionPlanDraft:
    auto_requests = []
    if tool_name:
        auto_requests.append(
            {
                "files": [{"relative_path": "pkg/tool.py", "content": f"def {tool_name}():\n    return 1\n"}],
                "module_name": "pkg.tool",
                "callable_name": tool_name,
                "dependency": {
                    "mode": "auto",
                    "template_id": None,
                    "declared_imports": [],
                    "package_requirements": [],
                },
                "sandbox_policy": {"sandbox": {"enabled": True}},
                "activate": True,
                "hidden": False,
                "non_restartable": True,
                "guide_content": None,
                "guide_description": None,
                "callback_signature": None,
                "concurrency": None,
            }
        )
    definition = ToolboxDefinitionSpec.from_dict(
        {
            "contract": "hosting.toolbox.definition",
            "toolbox_id": "demo",
            "expected_revision": expected_revision,
            "auto_requests": auto_requests,
            "manual_requests": [],
            "intrinsics": {
                "names": [],
                "include_guides": False,
                "sandbox_policy": {"sandbox": {"enabled": True}},
            },
        }
    )
    if not tool_name:
        return ToolboxDefinitionPlanDraft(definition=definition, profiles=(), bundles=(), custom_environment_count=0)
    profile = ResolvedToolboxProfileSpec(
        environment_key=_digest(character),
        template_id="core",
        template_lock_digest=_digest(character),
        custom_resolved_lock_digest=None,
        sandbox_policy={"sandbox": {"enabled": True}},
        assigned_tool_keys=(f"auto:pkg.tool:{tool_name}",),
        resolved_import_roots=(),
    )
    bundle = ToolboxBundleSpec(
        bundle_id=f"demo-{tool_name.lower()}",
        toolbox_id="demo",
        files=[ToolboxBundleFile(relative_path="pkg/tool.py", content=f"def {tool_name}():\n    return 1\n")],
        auto_tools=[
            ToolboxBundleAutoTool(
                module_name="pkg.tool",
                callable_name=tool_name,
                non_restartable=True,
            )
        ],
        dependency_lock_hash=profile.effective_lock_digest,
        resolved_profile=profile,
    )
    return ToolboxDefinitionPlanDraft(
        definition=definition,
        profiles=(profile,),
        bundles=(bundle,),
        custom_environment_count=0,
    )


def _service(tmp_path: Path) -> EngineHostService:
    service = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    service._require_toolbox_executor_registration = (  # type: ignore[method-assign]
        lambda engine_id, *, command_label: service.get_registration(engine_id)
    )
    return service


def _prepare(service: EngineHostService, draft: ToolboxDefinitionPlanDraft, request_id: str) -> str:
    prepared = service._hosted_operations.prepare(
        owner_actor_id="actor:a",
        execution_kind="toolbox_definition_apply",
        selector={"kind": "toolbox_id", "id": "demo"},
        namespace="toolbox-definition:demo",
        request_id=request_id,
        fingerprint=hosted_execution_fingerprint(
            {"definition_revision": draft.definition.revision, "request_id": request_id}
        ),
        metadata={"toolbox_id": "demo"},
    )
    return prepared["status"]["operation"]["operation_id"]


class _FakeOrchestrator:
    def __init__(self, service: EngineHostService):
        self.service = service
        self.spawned = 0

    build_resolved_assignments = staticmethod(ToolboxSandboxOrchestrator.build_resolved_assignments)

    def spawn_resolved_assignments(self, *, toolbox_id, definition_revision, assignments):
        for assignment in assignments:
            if assignment.classification == "reused":
                continue
            self.spawned += 1
            engine_id = f"candidate-{self.spawned}"
            manifest = assignment.bundle_spec.manifest_payload()
            assignment.registration = self.service.register_spawned(
                engine_id=engine_id,
                pid=1234,
                command=["python", "worker.py"],
                executor_kind="toolbox_executor",
                routing_state="candidate",
                bundle={
                    "toolbox_id": toolbox_id,
                    "sandbox_profile_id": assignment.resolved_profile.profile_id,
                    "resolved_profile_id": assignment.resolved_profile.profile_id,
                    "manifest_hash": manifest["manifest_hash"],
                    "definition_revision": definition_revision,
                },
                environment={"environment_key": assignment.resolved_profile.environment_key},
                tool_access={
                    "allowed_tool_names": sorted(
                        item["name"] for item in [*manifest["tools"], *manifest["auto_tools"]]
                    )
                },
            )
        return list(assignments)


def _install_fake_rollout(service: EngineHostService) -> _FakeOrchestrator:
    orchestrator = _FakeOrchestrator(service)
    service._toolbox_rollout_orchestrator_factory = lambda: orchestrator  # type: ignore[attr-defined]
    service._ensure_toolbox_assignments_ready = (  # type: ignore[method-assign]
        lambda assignments, timeout_seconds=8.0: {
            item.registration["engine_id"]: {"ready": True}
            for item in assignments
            if item.registration
        }
    )
    return orchestrator


def _changes(draft: ToolboxDefinitionPlanDraft, classification: str, active_profile_id=None):
    return [
        {
            "classification": classification,
            "active_profile_id": active_profile_id,
            "proposed_profile_id": draft.profiles[0].profile_id,
            "changed_fields": [] if classification != "replaced" else ["manifest_hash"],
        }
    ]


def test_apply_publishes_routes_then_drains_old_and_keeps_terminal_result_user_safe(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_fake_rollout(service)
    first = _draft("Alpha", "a", None)
    first_id = _prepare(service, first, "first")
    first_result = service._apply_resolved_toolbox_definition(
        draft=first,
        profile_changes=_changes(first, "added"),
        operation_id=first_id,
    )
    first_engine = service._toolbox_state_v2.get("demo")["tool_routes"]["Alpha"]["engine_id"]
    assert first_result["lifecycle"] == "terminal_success"
    assert service._route_toolbox_registration(
        toolbox_id="demo", tool_name="Alpha", command_label="test"
    )["engine_id"] == first_engine

    second = _draft("Beta", "b", first.definition.revision)
    second_id = _prepare(service, second, "second")
    second_result = service._apply_resolved_toolbox_definition(
        draft=second,
        profile_changes=_changes(second, "replaced", first.profiles[0].profile_id),
        operation_id=second_id,
    )
    snapshot = service._toolbox_state_v2.get("demo")

    assert snapshot["active_revision"] == second.definition.revision
    assert set(snapshot["tool_routes"]) == {"Beta"}
    assert service.get_registration(first_engine) is None
    assert service._route_toolbox_registration(
        toolbox_id="demo", tool_name="Beta", command_label="test"
    )["routing_state"] == "active"
    with pytest.raises(PermissionError, match="tool_not_allowed:Alpha"):
        service._route_toolbox_registration(toolbox_id="demo", tool_name="Alpha", command_label="test")
    terminal_text = str(second_result["result"])
    assert "engine_id" not in terminal_text
    assert "profile_id" not in terminal_text
    assert "environment_key" not in terminal_text
    with pytest.raises(PermissionError, match="toolbox_operator_details_denied"):
        service.toolbox_definition_apply_operator_details(
            operation_id=second_id, operator_authorized=False
        )
    details = service.toolbox_definition_apply_operator_details(
        operation_id=second_id, operator_authorized=True
    )
    assert details["candidate_engine_ids"]


def test_candidate_warmup_and_failed_readiness_leave_old_routes_untouched(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_fake_rollout(service)
    first = _draft("Alpha", "a", None)
    service._apply_resolved_toolbox_definition(
        draft=first,
        profile_changes=_changes(first, "added"),
        operation_id=_prepare(service, first, "first"),
    )
    old_snapshot = service._toolbox_state_v2.get("demo")
    second = _draft("Beta", "b", first.definition.revision)
    warmup = threading.Event()
    release = threading.Event()

    def fail_readiness(assignments, timeout_seconds=8.0):
        warmup.set()
        assert release.wait(2)
        raise RuntimeError("candidate_not_ready")

    service._ensure_toolbox_assignments_ready = fail_readiness  # type: ignore[method-assign]
    results: list[dict] = []
    worker = threading.Thread(
        target=lambda: results.append(
            service._apply_resolved_toolbox_definition(
                draft=second,
                profile_changes=_changes(second, "replaced", first.profiles[0].profile_id),
                operation_id=_prepare(service, second, "second"),
            )
        )
    )
    worker.start()
    assert warmup.wait(2)
    assert service._route_toolbox_registration(
        toolbox_id="demo", tool_name="Alpha", command_label="test"
    )["engine_id"] == old_snapshot["tool_routes"]["Alpha"]["engine_id"]
    with pytest.raises(PermissionError):
        service._route_toolbox_registration(toolbox_id="demo", tool_name="Beta", command_label="test")
    release.set()
    worker.join(2)

    assert results[0]["lifecycle"] == "terminal_failure"
    assert service._toolbox_state_v2.get("demo") == old_snapshot
    assert all(row["routing_state"] != "candidate" for row in service._toolbox_executor_registrations("demo"))


def test_published_replacement_marks_busy_old_worker_retired_without_killing_inflight_work(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_fake_rollout(service)
    first = _draft("Alpha", "a", None)
    service._apply_resolved_toolbox_definition(
        draft=first,
        profile_changes=_changes(first, "added"),
        operation_id=_prepare(service, first, "first"),
    )
    old_engine = service._toolbox_state_v2.get("demo")["tool_routes"]["Alpha"]["engine_id"]
    service._toolbox_runtime_base = lambda: type(  # type: ignore[method-assign]
        "BusyRuntime",
        (),
        {"resources": staticmethod(lambda environment_key: {"metrics": {"active_calls": 1}})},
    )()
    second = _draft("Beta", "b", first.definition.revision)

    result = service._apply_resolved_toolbox_definition(
        draft=second,
        profile_changes=_changes(second, "replaced", first.profiles[0].profile_id),
        operation_id=_prepare(service, second, "second"),
    )

    assert result["lifecycle"] == "terminal_success"
    assert result["result"]["rollout_summary"]["drain_pending_profiles"] == 1
    assert service.get_registration(old_engine)["routing_state"] == "retired"
    assert service._toolbox_state_v2.get("demo")["tool_routes"]["Beta"]["non_restartable"] is True


def test_prepublication_cancel_cleans_candidate_and_empty_definition_is_valid_revision(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_fake_rollout(service)
    first = _draft("Alpha", "a", None)
    service._apply_resolved_toolbox_definition(
        draft=first,
        profile_changes=_changes(first, "added"),
        operation_id=_prepare(service, first, "first"),
    )
    old_snapshot = service._toolbox_state_v2.get("demo")
    second = _draft("Beta", "b", first.definition.revision)
    second_id = _prepare(service, second, "second")
    warmup = threading.Event()
    release = threading.Event()

    def block_readiness(assignments, timeout_seconds=8.0):
        warmup.set()
        assert release.wait(2)
        return {item.registration["engine_id"]: {"ready": True} for item in assignments if item.registration}

    service._ensure_toolbox_assignments_ready = block_readiness  # type: ignore[method-assign]
    results: list[dict] = []
    worker = threading.Thread(
        target=lambda: results.append(
            service._apply_resolved_toolbox_definition(
                draft=second,
                profile_changes=_changes(second, "replaced", first.profiles[0].profile_id),
                operation_id=second_id,
            )
        )
    )
    worker.start()
    assert warmup.wait(2)
    record = service._hosted_operations.get_by_operation_id(second_id)
    canceled = service.hosted_operation_cancel(ref=record["operation"], owner_actor_id="actor:a")
    release.set()
    worker.join(2)

    assert canceled["lifecycle"] == results[0]["lifecycle"] == "terminal_cancellation"
    assert service._toolbox_state_v2.get("demo") == old_snapshot
    assert all(row["routing_state"] != "candidate" for row in service._toolbox_executor_registrations("demo"))

    service._ensure_toolbox_assignments_ready = lambda assignments, timeout_seconds=8.0: {}  # type: ignore[method-assign]
    empty = _draft(None, "c", first.definition.revision)
    empty_result = service._apply_resolved_toolbox_definition(
        draft=empty,
        profile_changes=[
            {
                "classification": "removed",
                "active_profile_id": first.profiles[0].profile_id,
                "proposed_profile_id": None,
                "changed_fields": [],
            }
        ],
        operation_id=_prepare(service, empty, "empty"),
    )
    snapshot = service._toolbox_state_v2.get("demo")
    assert empty_result["lifecycle"] == "terminal_success"
    assert snapshot["active_revision"] == empty.definition.revision
    assert snapshot["profiles"] == snapshot["tool_routes"] == {}
    assert len(snapshot["rollout_history"]) == 2
    described = service.toolbox_describe(toolbox_id="demo")
    assert described["all_registered_tool_names"] == []
