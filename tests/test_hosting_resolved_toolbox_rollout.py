from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hosting.service.errors import ToolboxRolloutError
from hosting.service.host_service import EngineHostService
from hosting.toolbox.bundle_models import (
    ResolvedToolboxProfileSpec,
    ToolboxBundleAutoTool,
    ToolboxBundleFile,
    ToolboxBundleSpec,
)
from hosting.toolbox.orchestration import ToolboxSandboxOrchestrator
from hosting.toolbox.staging import ToolboxBundleStager


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _profile(character: str, tool_name: str) -> ResolvedToolboxProfileSpec:
    return ResolvedToolboxProfileSpec(
        environment_key=_digest(character),
        template_id="core",
        template_lock_digest=_digest(character),
        custom_resolved_lock_digest=None,
        sandbox_policy={"sandbox": {"enabled": True}},
        assigned_tool_keys=(f"auto:pkg.tools:{tool_name}",),
        resolved_import_roots=("json",),
    )


def _bundle(toolbox_id: str, profile: ResolvedToolboxProfileSpec, tool_name: str) -> ToolboxBundleSpec:
    return ToolboxBundleSpec(
        bundle_id=f"{toolbox_id}-{tool_name.lower()}",
        toolbox_id=toolbox_id,
        files=[ToolboxBundleFile(relative_path="pkg/tools.py", content=f"def {tool_name}():\n    return 1\n")],
        auto_tools=[ToolboxBundleAutoTool(module_name="pkg.tools", callable_name=tool_name)],
        dependency_lock_hash=profile.effective_lock_digest,
        resolved_profile=profile,
    )


def _service(tmp_path: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )


def test_resolved_rollout_skips_reused_and_spawns_added_as_candidate(tmp_path: Path) -> None:
    reused = _profile("a", "Alpha")
    added = _profile("b", "Beta")
    receipt_root = tmp_path / "environment"
    receipt_root.mkdir()
    (receipt_root / "verification-receipt.json").write_text(
        json.dumps(
            {
                "contract": "hosting.toolbox.hermetic_environment_receipt.v1",
                "state": "verified",
                "environment_key": added.environment_key,
            }
        ),
        encoding="utf-8",
    )

    class Service:
        _toolbox_required_python_abi = "cp312"
        _toolbox_required_platform = "win_amd64"
        engines_state_file = tmp_path / "engines.json"
        control_state_file = tmp_path / "access.json"

        def __init__(self) -> None:
            self.materialized: list[dict] = []
            self.spawned: list[dict] = []

        def materialize_toolbox_environment_for_bundle(self, **kwargs):
            self.materialized.append(kwargs)
            return SimpleNamespace(
                environment_key=added.environment_key,
                environment_root=str(receipt_root),
                python_executable=str(receipt_root / "python.exe"),
                resolved=SimpleNamespace(
                    complete_lock_digest=added.effective_lock_digest,
                    runtime_artifact_digest=_digest("c"),
                ),
            )

        def spawn(self, **kwargs):
            self.spawned.append(kwargs)
            return {**kwargs, "pid": 1234}

    service = Service()
    orchestrator = ToolboxSandboxOrchestrator(
        service=service,
        stager=ToolboxBundleStager(tmp_path / "host"),
    )
    assignments = orchestrator.build_resolved_assignments(
        toolbox_id="demo",
        profiles=(reused, added),
        bundles=(_bundle("demo", reused, "Alpha"), _bundle("demo", added, "Beta")),
        profile_changes=(
            {
                "classification": "reused",
                "active_profile_id": reused.profile_id,
                "proposed_profile_id": reused.profile_id,
                "changed_fields": [],
            },
            {
                "classification": "added",
                "active_profile_id": None,
                "proposed_profile_id": added.profile_id,
                "changed_fields": [],
            },
        ),
    )

    result = orchestrator.spawn_resolved_assignments(
        toolbox_id="demo",
        definition_revision=_digest("d"),
        assignments=assignments,
    )

    reused_assignment = next(item for item in result if item.classification == "reused")
    added_assignment = next(item for item in result if item.classification == "added")
    assert reused_assignment.staged_bundle is reused_assignment.registration is None
    assert added_assignment.staged_bundle is not None
    assert added_assignment.registration["routing_state"] == "candidate"
    assert len(service.materialized) == len(service.spawned) == 1
    assert service.spawned[0]["bundle"]["resolved_profile_id"] == added.profile_id
    assert service.spawned[0]["environment"]["environment_key"] == added.environment_key


def test_candidate_registration_is_never_selected_by_scan_routing(tmp_path: Path) -> None:
    service = _service(tmp_path)
    common = {
        "pid": 1234,
        "command": ["python", "worker.py"],
        "executor_kind": "toolbox_executor",
        "bundle": {"toolbox_id": "demo", "sandbox_profile_id": "profile"},
        "tool_access": {"allowed_tool_names": ["Alpha"]},
    }
    service.register_spawned(engine_id="active", routing_state="active", **common)
    service.register_spawned(engine_id="candidate", routing_state="candidate", **common)
    service._require_toolbox_executor_registration = (  # type: ignore[method-assign]
        lambda engine_id, *, command_label: service.get_registration(engine_id)
    )

    routed = service._route_toolbox_registration(
        toolbox_id="demo", tool_name="Alpha", command_label="toolbox-execute"
    )

    assert routed["engine_id"] == "active"


def test_candidate_readiness_requires_exact_inventory_metadata_and_receipt(tmp_path: Path) -> None:
    service = _service(tmp_path)
    profile = _profile("e", "Alpha")
    environment_root = tmp_path / "verified-env"
    environment_root.mkdir()
    (environment_root / "verification-receipt.json").write_text(
        json.dumps(
            {
                "contract": "hosting.toolbox.hermetic_environment_receipt.v1",
                "state": "verified",
                "environment_key": profile.environment_key,
            }
        ),
        encoding="utf-8",
    )
    registration = service.register_spawned(
        engine_id="candidate",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        routing_state="candidate",
        bundle={
            "toolbox_id": "demo",
            "sandbox_profile_id": profile.profile_id,
            "resolved_profile_id": profile.profile_id,
            "bundle_revision": "revision",
        },
        environment={
            "venv_path": str(environment_root),
            "environment_key": profile.environment_key,
            "verification_receipt_contract": "hosting.toolbox.hermetic_environment_receipt.v1",
            "verification_state": "verified",
        },
        tool_access={"allowed_tool_names": ["Alpha"]},
    )
    assignment = SimpleNamespace(
        toolbox_id="demo", resolved_profile=profile, registration=registration
    )
    descriptions = [{"all_registered_tool_names": ["Alpha"]}]
    service._wait_for_toolbox_executor_ready = (  # type: ignore[method-assign]
        lambda engine_id, *, timeout_seconds: descriptions[-1]
    )

    ready = service._ensure_toolbox_assignments_ready([assignment])
    assert ready["candidate"]["tool_inventory_ok"] is True
    assert ready["candidate"]["install_receipt_verification_status"] == "ok"

    descriptions.append({"all_registered_tool_names": ["Alpha", "Unexpected"]})
    with pytest.raises(ToolboxRolloutError) as exc:
        service._ensure_toolbox_assignments_ready([assignment])
    assert exc.value.code == "toolbox_candidate_inventory_mismatch"

    descriptions[-1] = {"all_registered_tool_names": ["Alpha"]}
    (environment_root / "verification-receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ToolboxRolloutError) as receipt_exc:
        service._ensure_toolbox_assignments_ready([assignment])
    assert receipt_exc.value.code == "toolbox_environment_receipt_unverified"
