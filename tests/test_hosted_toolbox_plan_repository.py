from __future__ import annotations

import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

from hosting.service.toolbox_plans import (
    AtomicJsonCompleteToolboxDefinitionPlanRepository,
    AtomicJsonToolboxDefinitionPlanRepository,
)
from hosting.toolbox.bundle_models import (
    ToolboxDefinitionSpec,
    ToolboxDependencyEdgeSpec,
    ToolboxEnvironmentMutationSpec,
    ToolboxExactArtifactSpec,
    ToolboxPackageMutationSpec,
    ToolboxPlanPins,
    ToolboxResolutionAlternativeSpec,
    ToolboxToolMutationSpec,
)
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.definition_planner import (
    classify_toolbox_profiles,
    plan_toolbox_definition,
    profile_snapshots_from_draft,
)
from hosting_toolbox_test_catalog import realized_test_catalog


CATALOG_REVISION = "sha256:" + "c" * 64
POLICY_REVISION = "sha256:" + "d" * 64
HOST_CONFIG_REVISION = "sha256:" + "1" * 64
SOURCE_SET_REVISION = "sha256:" + "2" * 64


def _auto(name: str, *, body: str | None = None, sandbox: dict | None = None) -> dict:
    return {
        "files": [
            {
                "relative_path": f"pkg/{name.lower()}.py",
                "content": body or f"def {name}():\n    return 1\n",
            }
        ],
        "module_name": f"pkg.{name.lower()}",
        "callable_name": name,
        "dependency": {
            "mode": "auto",
            "template_id": None,
            "declared_imports": [],
            "package_requirements": [],
        },
        "sandbox_policy": sandbox or {"sandbox": {"enabled": True}},
        "activate": True,
        "hidden": False,
        "non_restartable": False,
        "guide_content": None,
        "guide_description": None,
        "callback_signature": None,
        "concurrency": None,
    }


def _definition(toolbox_id: str, requests: list[dict], *, expected_revision: str | None = None) -> dict:
    return {
        "contract": "hosting.toolbox.definition",
        "toolbox_id": toolbox_id,
        "expected_revision": expected_revision,
        "auto_requests": requests,
        "manual_requests": [],
        "intrinsics": {
            "names": [],
            "include_guides": False,
            "sandbox_policy": {"sandbox": {"enabled": True}},
        },
    }


def _draft(definition: dict):
    shipped = realized_test_catalog()
    return plan_toolbox_definition(
        definition,
        templates=shipped.templates,
        python_abi=f"cp{sys.version_info.major}{sys.version_info.minor}",
        platform="win_amd64" if os.name == "nt" else "manylinux_2_28_x86_64",
        runtime_identity={"version": "3.12.7", "artifact_digest": "sha256:" + "a" * 64},
    )


def _complete_inputs(definition: dict):
    draft = _draft(definition)
    active = ToolboxDefinitionSpec.from_dict(_definition(definition["toolbox_id"], []))
    artifact = ToolboxExactArtifactSpec(
        import_roots=("demo_pkg",),
        distribution="demo-pkg",
        dependency_reason="direct",
        version="1.0.0",
        wheel_filename="demo_pkg-1.0.0-py3-none-any.whl",
        artifact_digest="sha256:" + "3" * 64,
        compatibility_tags=("py3-none-any",),
        provenance="signed-test",
        source_id="release",
    )
    mutation = ToolboxPackageMutationSpec(
        distribution="demo-pkg",
        mutation="addition",
        dependency_reason="direct",
        from_version=None,
        to_version="1.0.0",
    )
    alternative_payload = {"artifact": artifact.to_dict(), "mutation": mutation.to_dict()}
    alternative = ToolboxResolutionAlternativeSpec(
        alternative_id=identity_digest("test.toolbox.complete.alternative.v1", alternative_payload),
        source_id="release",
        source_origin="https://packages.example.invalid/simple",
        lock_digest=identity_digest("test.toolbox.complete.lock.v1", alternative_payload),
        artifacts=(artifact,),
        package_mutations=(mutation,),
    )
    tool_key = draft.definition.auto_requests[0].stable_key
    environment = ToolboxEnvironmentMutationSpec(
        environment_id=identity_digest("test.toolbox.complete.environment.v1", tool_key),
        tool_mutations=(ToolboxToolMutationSpec(tool_key, "added"),),
        base_template_id=draft.profiles[0].template_id,
        base_template_revision="sha256:" + "4" * 64,
        alternatives=(alternative,),
        preferred_alternative_id=alternative.alternative_id,
        alternatives_truncated=False,
        confirmation_required=True,
        dependency_approval_required=True,
        dependency_edges=(ToolboxDependencyEdgeSpec(tool_key, (), ("demo-pkg",)),),
    )
    pins = ToolboxPlanPins(
        active_definition_revision=None,
        target="cp312-win_amd64" if os.name == "nt" else "cp312-manylinux_2_28_x86_64",
        catalog_revision=CATALOG_REVISION,
        host_config_revision=HOST_CONFIG_REVISION,
        dependency_policy_revision=POLICY_REVISION,
        source_set_revision=SOURCE_SET_REVISION,
    )
    return draft, active, pins, (environment,)


def _create_in_process(path: str, definition: dict, now_ms: int, queue) -> None:
    try:
        record = AtomicJsonToolboxDefinitionPlanRepository(Path(path)).create(
            _draft(definition),
            active_profiles=(),
            catalog_revision=CATALOG_REVISION,
            package_policy_revision=POLICY_REVISION,
            now_ms=now_ms,
            ttl_ms=60_000,
        )
        queue.put({"ok": True, "plan_id": record.plan_id})
    except Exception as exc:  # pragma: no cover - asserted through child output
        queue.put({"ok": False, "error": type(exc).__name__, "detail": str(exc)})


def _create_complete_in_process(path: str, definition: dict, now_ms: int, queue) -> None:
    try:
        draft, active, pins, environments = _complete_inputs(definition)
        record = AtomicJsonCompleteToolboxDefinitionPlanRepository(Path(path)).create(
            draft,
            active_definition=active,
            pins=pins,
            environment_mutations=environments,
            active_profiles=(),
            now_ms=now_ms,
            ttl_ms=60_000,
            owner_actor_id="actor:one",
            authority_id="workspace:one",
        )
        queue.put({"ok": True, "plan_id": record.plan_id})
    except Exception as exc:  # pragma: no cover - asserted through child output
        queue.put({"ok": False, "error": type(exc).__name__, "detail": str(exc)})


def test_profile_classification_covers_reused_replaced_added_and_removed() -> None:
    original = _draft(_definition("demo", [_auto("Alpha")]))
    active = profile_snapshots_from_draft(original)
    assert [item["classification"] for item in classify_toolbox_profiles(original, active)] == ["reused"]

    source_changed = _draft(
        _definition("demo", [_auto("Alpha", body="def Alpha():\n    return 2\n")])
    )
    changed = classify_toolbox_profiles(source_changed, active)
    assert changed[0]["classification"] == "replaced"
    assert changed[0]["changed_fields"] == ["manifest_hash"]

    policy_changed = _draft(
        _definition(
            "demo",
            [_auto("Alpha", sandbox={"sandbox": {"enabled": True, "network": {"mode": "disabled"}}})],
        )
    )
    policy = classify_toolbox_profiles(policy_changed, active)
    assert policy[0]["classification"] == "replaced"
    assert "sandbox_policy_digest" in policy[0]["changed_fields"]

    unrelated = _draft(_definition("demo", [_auto("Beta")]))
    assert [item["classification"] for item in classify_toolbox_profiles(unrelated, active)] == [
        "added",
        "removed",
    ]


def test_plan_record_is_pinned_strict_and_restart_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "plans.json"
    draft = _draft(_definition("demo", [_auto("Alpha")]))
    repository = AtomicJsonToolboxDefinitionPlanRepository(path)
    record = repository.create(
        draft,
        active_profiles=(),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_000,
        ttl_ms=60_000,
    )

    recovered = AtomicJsonToolboxDefinitionPlanRepository(path).get(record.plan_id, now_ms=2_000)
    assert recovered == record
    assert recovered.definition_revision == draft.definition.revision
    assert recovered.expected_revision is None
    assert recovered.catalog_revision == CATALOG_REVISION
    assert recovered.package_policy_revision == POLICY_REVISION
    assert recovered.profile_changes[0]["classification"] == "added"


def test_plan_repository_can_atomically_invalidate_unused_plans(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionPlanRepository(tmp_path / "plans.json")
    record = repository.create(
        _draft(_definition("demo", [_auto("Alpha")])),
        active_profiles=(),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_000,
        ttl_ms=60_000,
    )

    assert repository.invalidate_all() == 1
    assert repository.invalidate_all() == 0
    with pytest.raises(ValueError, match="toolbox_definition_plan_not_found"):
        repository.get(record.plan_id, now_ms=2_000)


def test_plan_identity_changes_with_every_authoritative_pin(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionPlanRepository(tmp_path / "plans.json")
    first_draft = _draft(_definition("demo", [_auto("Alpha")]))
    expected = first_draft.definition.revision
    replace_draft = _draft(_definition("demo", [_auto("Alpha")], expected_revision=expected))
    first = repository.create(
        first_draft,
        active_profiles=(),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_000,
        ttl_ms=60_000,
    )
    expected_changed = repository.create(
        replace_draft,
        active_profiles=profile_snapshots_from_draft(first_draft),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_001,
        ttl_ms=60_000,
    )
    catalog_changed = repository.create(
        first_draft,
        active_profiles=(),
        catalog_revision="sha256:" + "e" * 64,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_002,
        ttl_ms=60_000,
    )
    policy_changed = repository.create(
        first_draft,
        active_profiles=(),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision="sha256:" + "f" * 64,
        now_ms=1_003,
        ttl_ms=60_000,
    )
    assert len({first.plan_id, expected_changed.plan_id, catalog_changed.plan_id, policy_changed.plan_id}) == 4


def test_expired_plan_is_removed_and_cannot_be_recovered(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionPlanRepository(tmp_path / "plans.json")
    record = repository.create(
        _draft(_definition("demo", [_auto("Alpha")])),
        active_profiles=(),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_000,
        ttl_ms=100,
    )
    with pytest.raises(ValueError, match="toolbox_definition_plan_expired"):
        repository.get(record.plan_id, now_ms=1_100)
    with pytest.raises(ValueError, match="toolbox_definition_plan_not_found"):
        repository.get(record.plan_id, now_ms=1_101)
    assert repository.list(now_ms=1_101) == ()


def test_plan_create_is_idempotent_without_refreshing_expiry(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionPlanRepository(tmp_path / "plans.json")
    draft = _draft(_definition("demo", [_auto("Alpha")]))
    first = repository.create(
        draft,
        active_profiles=(),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_000,
        ttl_ms=100,
    )
    second = repository.create(
        draft,
        active_profiles=(),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_050,
        ttl_ms=100,
    )
    assert second == first
    assert second.expires_at_ms == 1_100


def test_two_processes_persist_distinct_plans_without_lost_update(tmp_path: Path) -> None:
    path = tmp_path / "plans.json"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_create_in_process,
            args=(str(path), _definition(f"demo-{index}", [_auto(f"Tool{index}")]), 1_000 + index, queue),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]

    assert all(item["ok"] for item in results), results
    assert len({item["plan_id"] for item in results}) == 2
    assert len(AtomicJsonToolboxDefinitionPlanRepository(path).list(now_ms=2_000)) == 2


@pytest.mark.parametrize("raw", ["{", "[]", '{"contract":"wrong","plans":{}}'])
def test_corrupt_or_wrong_contract_state_fails_closed(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "plans.json"
    path.write_text(raw, encoding="utf-8")
    repository = AtomicJsonToolboxDefinitionPlanRepository(path)
    with pytest.raises(ValueError, match="toolbox_plan_state"):
        repository.list(now_ms=1_000)


def test_planning_and_persistence_do_not_stage_or_start_workers(tmp_path: Path) -> None:
    repository = AtomicJsonToolboxDefinitionPlanRepository(tmp_path / "state" / "plans.json")
    repository.create(
        _draft(_definition("demo", [_auto("Alpha")])),
        active_profiles=(),
        catalog_revision=CATALOG_REVISION,
        package_policy_revision=POLICY_REVISION,
        now_ms=1_000,
        ttl_ms=60_000,
    )
    assert not (tmp_path / "toolbox_bundles").exists()
    assert not (tmp_path / "managed_engines.json").exists()


def test_complete_plan_roundtrips_every_pin_offer_artifact_and_edge(tmp_path: Path) -> None:
    draft, active, pins, environments = _complete_inputs(
        _definition("complete", [_auto("Alpha")])
    )
    path = tmp_path / "complete-plans.json"
    repository = AtomicJsonCompleteToolboxDefinitionPlanRepository(path)
    created = repository.create(
        draft,
        active_definition=active,
        pins=pins,
        environment_mutations=environments,
        active_profiles=(),
        now_ms=1_000,
        ttl_ms=60_000,
        owner_actor_id="actor:one",
        authority_id="workspace:one",
    )
    duplicate = repository.create(
        draft,
        active_definition=active,
        pins=pins,
        environment_mutations=environments,
        active_profiles=(),
        now_ms=1_100,
        ttl_ms=60_000,
        owner_actor_id="actor:one",
        authority_id="workspace:one",
    )
    recovered = AtomicJsonCompleteToolboxDefinitionPlanRepository(path).get(
        created.plan_id, now_ms=2_000
    )

    assert duplicate == recovered == created
    assert recovered.pins == pins
    assert recovered.environment_mutations == environments
    artifact = recovered.environment_mutations[0].alternatives[0].artifacts[0]
    assert artifact.artifact_digest == "sha256:" + "3" * 64
    assert recovered.profile_changes[0]["classification"] == "added"
    assert recovered.expires_at_ms == 61_000
    assert repository.list(now_ms=2_000) == (created,)
    assert repository.invalidate_all() == 1
    assert repository.invalidate_all() == 0


def test_complete_plan_identity_changes_with_pin_and_offered_artifact(tmp_path: Path) -> None:
    draft, active, pins, environments = _complete_inputs(
        _definition("complete", [_auto("Alpha")])
    )
    repository = AtomicJsonCompleteToolboxDefinitionPlanRepository(tmp_path / "plans.json")

    def create(current_pins, current_environments, now_ms):
        return repository.create(
            draft,
            active_definition=active,
            pins=current_pins,
            environment_mutations=current_environments,
            active_profiles=(),
            now_ms=now_ms,
            ttl_ms=60_000,
            owner_actor_id="actor:one",
            authority_id="workspace:one",
        )

    baseline = create(pins, environments, 1_000)
    changed_pins = ToolboxPlanPins(
        **{**pins.to_dict(), "source_set_revision": "sha256:" + "5" * 64}
    )
    pin_changed = create(changed_pins, environments, 1_001)
    environment = environments[0]
    alternative = environment.alternatives[0]
    artifact = alternative.artifacts[0]
    changed_artifact = ToolboxExactArtifactSpec.from_dict(
        {**artifact.to_dict(), "artifact_digest": "sha256:" + "6" * 64}
    )
    changed_alternative = ToolboxResolutionAlternativeSpec(
        **{
            **alternative.__dict__,
            "alternative_id": identity_digest(
                "test.toolbox.changed.alternative.v1", changed_artifact.to_dict()
            ),
            "lock_digest": identity_digest(
                "test.toolbox.changed.lock.v1", changed_artifact.to_dict()
            ),
            "artifacts": (changed_artifact,),
        }
    )
    changed_environment = ToolboxEnvironmentMutationSpec(
        **{
            **environment.__dict__,
            "alternatives": (changed_alternative,),
            "preferred_alternative_id": changed_alternative.alternative_id,
        }
    )
    artifact_changed = create(pins, (changed_environment,), 1_002)

    assert len({baseline.plan_id, pin_changed.plan_id, artifact_changed.plan_id}) == 3


def test_complete_plan_processes_do_not_lose_distinct_records(tmp_path: Path) -> None:
    path = tmp_path / "complete-plans.json"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_create_complete_in_process,
            args=(
                str(path),
                _definition(f"complete-{index}", [_auto(f"Tool{index}")]),
                1_000 + index,
                queue,
            ),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]

    assert all(item["ok"] for item in results), results
    assert len({item["plan_id"] for item in results}) == 2
    assert len(AtomicJsonCompleteToolboxDefinitionPlanRepository(path).list(now_ms=2_000)) == 2
