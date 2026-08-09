from __future__ import annotations

from pathlib import Path

from hosting.service.host_service import EngineHostService
from hosting.toolbox.bundle_models import (
    ResolvedToolboxProfileSpec,
    ToolboxBundleAutoTool,
    ToolboxBundleFile,
    ToolboxBundleSpec,
    ToolboxDefinitionSpec,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _install_active(service: EngineHostService, *, include_registration: bool = True) -> str:
    definition = ToolboxDefinitionSpec.from_dict(
        {
            "contract": "hosting.toolbox.definition",
            "toolbox_id": "demo",
            "expected_revision": None,
            "auto_requests": [
                {
                    "files": [{"relative_path": "demo.py", "content": "def Alpha(): return 1\n"}],
                    "module_name": "demo",
                    "callable_name": "Alpha",
                    "dependency": {
                        "mode": "auto",
                        "template_id": None,
                        "declared_imports": [],
                        "package_requirements": [],
                    },
                    "sandbox_policy": {},
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
            "intrinsics": {"names": [], "include_guides": False, "sandbox_policy": {}},
        }
    )
    profile = ResolvedToolboxProfileSpec(
        environment_key=_digest("a"),
        template_id="core",
        template_lock_digest=_digest("b"),
        custom_resolved_lock_digest=None,
        sandbox_policy={},
        assigned_tool_keys=("auto:demo:Alpha",),
        resolved_import_roots=(),
    )
    bundle = ToolboxBundleSpec(
        bundle_id="demo-alpha",
        toolbox_id="demo",
        files=[ToolboxBundleFile(relative_path="demo.py", content="def Alpha(): return 1\n")],
        auto_tools=[ToolboxBundleAutoTool(module_name="demo", callable_name="Alpha")],
        dependency_lock_hash=profile.effective_lock_digest,
        resolved_profile=profile,
    )
    manifest_hash = bundle.manifest_payload()["manifest_hash"]
    if not manifest_hash.startswith("sha256:"):
        manifest_hash = f"sha256:{manifest_hash}"
    engine_id = "active-alpha"
    service._toolbox_state_v2.publish(
        toolbox_id="demo",
        expected_revision=None,
        definition=definition.to_dict(),
        profiles={
            profile.profile_id: {
                "profile": profile.to_dict(),
                "manifest_hash": manifest_hash,
                "engine_id": engine_id,
                "tool_names": ["Alpha"],
                "environment_reference": f"toolbox:demo:{profile.profile_id}:{definition.revision}",
            }
        },
        tool_routes={
            "Alpha": {
                "profile_id": profile.profile_id,
                "engine_id": engine_id,
                "non_restartable": False,
            }
        },
        environment_references=[f"toolbox:demo:{profile.profile_id}:{definition.revision}"],
        published_at_ms=1,
    )
    if include_registration:
        service.register_spawned(
            engine_id=engine_id,
            pid=1234,
            command=["python", "worker.py"],
            executor_kind="toolbox_executor",
            routing_state="active",
            bundle={
                "toolbox_id": "demo",
                "sandbox_profile_id": profile.profile_id,
                "resolved_profile_id": profile.profile_id,
                "manifest_hash": manifest_hash,
                "definition_revision": definition.revision,
            },
            environment={"environment_key": profile.environment_key},
            tool_access={"allowed_tool_names": ["Alpha"]},
        )
    return engine_id


def _service(tmp_path: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=tmp_path / "managed.json",
        control_state_file=tmp_path / "control.json",
    )


def test_route_based_references_consistency_and_review(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_active(service)

    references = service.toolbox_references()
    consistency = service.toolbox_consistency()
    review = service.toolbox_review_snapshot(toolbox_ids=["demo"])

    assert references["contract"] == "hosting.toolbox.references.v2"
    assert references["summary"]["active_registration_count"] == 1
    assert consistency["consistent"] is True
    assert review["recommended_action"] == "observe"
    assert review["toolboxes"]["demo"]["tool_names"] == ["Alpha"]


def test_missing_active_registration_requires_definition_reapply(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_active(service, include_registration=False)

    consistency = service.toolbox_consistency()
    repair = service.toolbox_repair(toolbox_ids=["demo"])

    assert consistency["issues"][0]["issue"] == "missing_active_registration"
    assert repair["reapply_required"][0]["issue"] == "definition_reapply_required"
    assert repair["repaired_toolbox_ids"] == []


def test_repair_restores_route_state_from_active_snapshot(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    engine_id = _install_active(service)
    service.set_toolbox_registration_routing_states({engine_id: "retired"})
    monkeypatch.setattr(service, "recover_toolbox_definition_rollouts", lambda: {"status": "ok"})

    repair = service.toolbox_repair(toolbox_ids=["demo"])

    assert repair["reactivated_engine_ids"] == [engine_id]
    assert service.get_registration(engine_id)["routing_state"] == "active"
    assert service.toolbox_consistency()["consistent"] is True


def test_gc_removes_only_unreferenced_candidate_and_retired_workers(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path)
    active = _install_active(service)
    for engine_id, state in (("candidate-orphan", "candidate"), ("retired-old", "retired")):
        service.register_spawned(
            engine_id=engine_id,
            pid=2345,
            command=["python", "worker.py"],
            executor_kind="toolbox_executor",
            routing_state=state,
            bundle={"toolbox_id": "demo", "sandbox_profile_id": "old"},
            environment={},
            tool_access={"allowed_tool_names": ["Old"]},
        )
    removed: list[str] = []
    monkeypatch.setattr(service, "recover_toolbox_definition_rollouts", lambda: {"status": "ok"})
    monkeypatch.setattr(service, "_retire_toolbox_registration", removed.append)

    result = service.toolbox_gc()

    assert result["removed_engine_ids"] == ["candidate-orphan", "retired-old"]
    assert removed == ["candidate-orphan", "retired-old"]
    assert active not in removed


def test_gc_removes_orphaned_bundle_but_preserves_active_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path)
    active_engine = _install_active(service)
    bundles_root = service.hosting_root / "toolbox_bundles"
    active_root = bundles_root / "active"
    orphan_root = bundles_root / "orphan"
    active_root.mkdir(parents=True)
    orphan_root.mkdir(parents=True)
    (active_root / "manifest.json").write_text("{}", encoding="utf-8")
    (orphan_root / "manifest.json").write_text("{}", encoding="utf-8")
    registration = service.get_registration(active_engine)
    registration["bundle"]["bundle_root"] = str(active_root)
    service._write_engines(
        [registration if row["engine_id"] == active_engine else row for row in service._read_engines()]
    )
    monkeypatch.setattr(service, "recover_toolbox_definition_rollouts", lambda: {"status": "ok"})

    result = service.toolbox_gc()

    assert result["removed_bundle_roots"] == [str(orphan_root.resolve())]
    assert active_root.is_dir()
    assert not orphan_root.exists()
