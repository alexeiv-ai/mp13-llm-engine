from __future__ import annotations

import copy
import os
import sys

import pytest

from hosting.toolbox.definition_planner import (
    ToolboxEnvironmentConfirmationChoice,
    classify_toolbox_profiles,
    plan_toolbox_definition,
    profile_snapshots_from_draft,
    reduce_toolbox_confirmation,
)
from hosting.toolbox.bundle_models import (
    ToolboxDependencyEdgeSpec,
    ToolboxDefinitionSpec,
    ToolboxEnvironmentMutationSpec,
    ToolboxExactArtifactSpec,
    ToolboxPackageMutationSpec,
    ToolboxResolutionAlternativeSpec,
    ToolboxToolMutationSpec,
)
from hosting.toolbox.identity import identity_digest
from hosting_toolbox_test_catalog import realized_test_catalog


MANUAL_KEY = "manual:pkg.manualecho:ManualEcho"


def _dependency() -> dict:
    return {
        "mode": "auto",
        "template_id": None,
        "declared_imports": [],
        "package_requirements": [],
    }


def _auto(name: str, *, source: str | None = None) -> dict:
    return {
        "files": [
            {
                "relative_path": f"pkg/{name.lower()}.py",
                "content": source or f"def {name}():\n    return {name!r}\n",
            }
        ],
        "module_name": f"pkg.{name.lower()}",
        "callable_name": name,
        "dependency": _dependency(),
        "sandbox_policy": {
            "sandbox": {"enabled": True, "network": {"mode": "disabled"}}
        },
        "activate": True,
        "hidden": False,
        "non_restartable": False,
        "guide_content": None,
        "guide_description": None,
        "callback_signature": None,
        "concurrency": None,
    }


def _manual(name: str) -> dict:
    return {
        "files": [
            {
                "relative_path": f"pkg/{name.lower()}.py",
                "content": f"def {name}():\n    return {name!r}\n",
            }
        ],
        "module_name": f"pkg.{name.lower()}",
        "callable_name": name,
        "tool_definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": "manual",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        "dependency": _dependency(),
        "sandbox_policy": {
            "sandbox": {
                "enabled": True,
                "network": {"mode": "disabled"},
                "process": {"allow_subprocess": False},
            }
        },
        "hidden": False,
        "non_restartable": False,
        "callback_signature": None,
        "concurrency": None,
    }


def _definition(*, autos=(), manuals=(), intrinsics=("scriptable_calculator",)) -> dict:
    return {
        "contract": "hosting.toolbox.definition",
        "toolbox_id": "mixed",
        "expected_revision": None,
        "auto_requests": list(autos),
        "manual_requests": list(manuals),
        "intrinsics": {
            "names": list(intrinsics),
            "include_guides": False,
            "sandbox_policy": {
                "sandbox": {
                    "enabled": True,
                    "network": {"mode": "disabled"},
                    "filesystem": {"default_access": "deny", "rules": []},
                }
            },
        },
    }


def _plan(definition: dict):
    return plan_toolbox_definition(
        definition,
        templates=realized_test_catalog().templates,
        python_abi=f"cp{sys.version_info.major}{sys.version_info.minor}",
        platform="win_amd64" if os.name == "nt" else "manylinux_2_28_x86_64",
        runtime_identity={
            "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "artifact_digest": "sha256:" + "a" * 64,
        },
    )


def _bundle_by_key(draft) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for profile, bundle in zip(draft.profiles, draft.bundles, strict=True):
        manifest = bundle.manifest_payload()
        for key in profile.assigned_tool_keys:
            out[key] = {
                "profile_id": profile.profile_id,
                "environment_key": profile.environment_key,
                "resolved_import_roots": profile.resolved_import_roots,
                "manifest": manifest,
            }
    return out


def _alternative(
    name: str,
    *,
    mutations: tuple[ToolboxPackageMutationSpec, ...],
) -> ToolboxResolutionAlternativeSpec:
    artifact = ToolboxExactArtifactSpec(
        import_roots=("demo_pkg",),
        distribution="demo-pkg",
        dependency_reason="direct",
        version="1.0.0",
        wheel_filename="demo_pkg-1.0.0-py3-none-any.whl",
        artifact_digest="sha256:" + "a" * 64,
        compatibility_tags=("py3-none-any",),
        provenance="signed-test",
        source_id="release",
    )
    payload = {
        "name": name,
        "mutations": [item.to_dict() for item in mutations],
    }
    return ToolboxResolutionAlternativeSpec(
        alternative_id=identity_digest("test.toolbox.alternative.v1", payload),
        source_id="release",
        source_origin="https://packages.example.invalid/simple",
        lock_digest=identity_digest("test.toolbox.lock.v1", payload),
        artifacts=(artifact,),
        package_mutations=mutations,
    )


def _offer(
    name: str,
    *,
    tools: tuple[ToolboxToolMutationSpec, ...],
    mutations: tuple[ToolboxPackageMutationSpec, ...],
    required_tools: dict[str, tuple[str, ...]] | None = None,
) -> ToolboxEnvironmentMutationSpec:
    alternative = _alternative(name, mutations=mutations)
    required = required_tools or {}
    return ToolboxEnvironmentMutationSpec(
        environment_id=identity_digest("test.toolbox.environment.v1", {"name": name}),
        tool_mutations=tools,
        base_template_id="core",
        base_template_revision="sha256:" + "b" * 64,
        alternatives=(alternative,),
        preferred_alternative_id=alternative.alternative_id,
        alternatives_truncated=False,
        confirmation_required=any(item.mutation != "removal" for item in mutations),
        dependency_approval_required=True,
        dependency_edges=tuple(
            ToolboxDependencyEdgeSpec(
                tool_key=item.tool_key,
                required_tool_keys=required.get(item.tool_key, ()),
                required_distributions=("demo-pkg",),
            )
            for item in tools
        ),
    )


def test_mixed_definition_category_updates_preserve_unchanged_profiles() -> None:
    base_definition = _definition(autos=[_auto("Alpha")], manuals=[_manual("ManualEcho")])
    base = _plan(base_definition)
    active = profile_snapshots_from_draft(base)
    base_inventory = _bundle_by_key(base)
    assert len(base.profiles) == 3
    assert set(base_inventory) == {
        "pkg.alpha:Alpha",
        MANUAL_KEY,
        "intrinsic:scriptable_calculator",
    }

    code_update = copy.deepcopy(base_definition)
    code_update["auto_requests"][0]["files"][0]["content"] = "def Alpha():\n    return 'v2'\n"
    updated = _plan(code_update)
    changes = classify_toolbox_profiles(updated, active)
    assert [item["classification"] for item in changes].count("replaced") == 1
    assert [item["classification"] for item in changes].count("reused") == 2
    updated_inventory = _bundle_by_key(updated)
    for key in (MANUAL_KEY, "intrinsic:scriptable_calculator"):
        assert updated_inventory[key] == base_inventory[key]
    assert updated_inventory["pkg.alpha:Alpha"]["environment_key"] == base_inventory["pkg.alpha:Alpha"]["environment_key"]

    add_remove = _definition(
        autos=[_auto("Alpha"), _auto("Beta")],
        manuals=[],
        intrinsics=("scriptable_calculator",),
    )
    combined = _plan(add_remove)
    combined_changes = classify_toolbox_profiles(combined, active)
    assert {item["classification"] for item in combined_changes} == {
        "reused",
        "replaced",
        "removed",
    }
    combined_inventory = _bundle_by_key(combined)
    assert MANUAL_KEY not in combined_inventory
    assert combined_inventory["intrinsic:scriptable_calculator"] == base_inventory[
        "intrinsic:scriptable_calculator"
    ]

    intrinsic_update = _plan(
        _definition(
            autos=[_auto("Alpha")],
            manuals=[_manual("ManualEcho")],
            intrinsics=("scriptable_calculator", "symbolic_algebra"),
        )
    )
    intrinsic_changes = classify_toolbox_profiles(intrinsic_update, active)
    assert [item["classification"] for item in intrinsic_changes].count("reused") == 2
    assert [item["classification"] for item in intrinsic_changes].count("replaced") == 1
    intrinsic_inventory = _bundle_by_key(intrinsic_update)
    for key in ("pkg.alpha:Alpha", MANUAL_KEY):
        assert intrinsic_inventory[key] == base_inventory[key]
    assert "intrinsic:symbolic_algebra" in intrinsic_inventory

    empty = _plan(_definition(autos=[], manuals=[], intrinsics=()))
    assert empty.profiles == empty.bundles == ()
    assert {item["classification"] for item in classify_toolbox_profiles(empty, active)} == {
        "removed"
    }


def test_conflicting_active_profile_identity_and_missing_import_fail_before_rollout() -> None:
    draft = _plan(_definition(autos=[_auto("Alpha")], manuals=[_manual("ManualEcho")]))
    snapshot = profile_snapshots_from_draft(draft)[0]
    with pytest.raises(ValueError, match="active_profile_snapshot_duplicate"):
        classify_toolbox_profiles(draft, [snapshot, snapshot])

    missing = _definition(
        autos=[
            _auto(
                "Missing",
                source="import definitely_missing_host_package\ndef Missing():\n    return 1\n",
            )
        ],
        manuals=[],
        intrinsics=(),
    )
    with pytest.raises(Exception, match="dependency_unresolved"):
        _plan(missing)


def test_confirmation_decline_skips_new_preserves_update_and_applies_removal() -> None:
    active_payload = _definition(
        autos=[
            _auto("Update", source="def Update():\n    return 'old'\n"),
            _auto("Remove"),
        ],
        manuals=[],
        intrinsics=(),
    )
    active = ToolboxDefinitionSpec.from_dict(active_payload)
    proposed_payload = _definition(
        autos=[
            _auto("Update", source="def Update():\n    return 'new'\n"),
            _auto("Add"),
        ],
        manuals=[],
        intrinsics=(),
    )
    proposed_payload["expected_revision"] = active.revision
    proposed = ToolboxDefinitionSpec.from_dict(proposed_payload)
    package_change = ToolboxPackageMutationSpec(
        distribution="demo-pkg",
        mutation="transition",
        dependency_reason="direct",
        from_version="0.9.0",
        to_version="1.0.0",
    )
    removal = ToolboxPackageMutationSpec(
        distribution="old-pkg",
        mutation="removal",
        dependency_reason="direct",
        from_version="2.0.0",
        to_version=None,
    )
    changed = _offer(
        "changed",
        tools=(
            ToolboxToolMutationSpec("pkg.add:Add", "added"),
            ToolboxToolMutationSpec("pkg.update:Update", "updated"),
        ),
        mutations=(package_change,),
    )
    removed = _offer(
        "removed",
        tools=(ToolboxToolMutationSpec("pkg.remove:Remove", "removed"),),
        mutations=(removal,),
    )

    result = reduce_toolbox_confirmation(
        active_definition=active,
        proposed_definition=proposed,
        environment_mutations=(changed, removed),
        choices=(
            ToolboxEnvironmentConfirmationChoice(
                changed.environment_id, changed.preferred_alternative_id, False
            ),
            ToolboxEnvironmentConfirmationChoice(
                removed.environment_id, removed.preferred_alternative_id, False
            ),
        ),
    )

    assert result.accepted_tool_keys == ()
    assert result.preserved_active_tool_keys == ("pkg.update:Update",)
    assert result.removed_tool_keys == ("pkg.remove:Remove",)
    assert [item["reason"] for item in result.skipped_tools] == [
        "package_changes_declined",
        "package_changes_declined",
    ]
    assert [item.callable_name for item in result.effective_definition.auto_requests] == [
        "Update"
    ]
    assert result.effective_definition.auto_requests[0].files[0].content.endswith("'old'\n")
    assert [item["distribution"] for item in result.package_mutations] == ["old-pkg"]
    assert result.dependency_approval_required is False


def test_confirmation_propagates_shared_environment_skip_and_rejects_unoffered_choice() -> None:
    active = ToolboxDefinitionSpec.from_dict(_definition(autos=[], manuals=[], intrinsics=()))
    proposed_payload = _definition(
        autos=[_auto("Producer"), _auto("Consumer")],
        manuals=[],
        intrinsics=(),
    )
    proposed = ToolboxDefinitionSpec.from_dict(proposed_payload)
    addition = ToolboxPackageMutationSpec(
        distribution="demo-pkg",
        mutation="addition",
        dependency_reason="direct",
        from_version=None,
        to_version="1.0.0",
    )
    producer = _offer(
        "producer",
        tools=(ToolboxToolMutationSpec("pkg.producer:Producer", "added"),),
        mutations=(addition,),
    )
    consumer = _offer(
        "consumer",
        tools=(ToolboxToolMutationSpec("pkg.consumer:Consumer", "added"),),
        mutations=(),
        required_tools={"pkg.consumer:Consumer": ("pkg.producer:Producer",)},
    )
    choices = (
        ToolboxEnvironmentConfirmationChoice(
            producer.environment_id, producer.preferred_alternative_id, False
        ),
        ToolboxEnvironmentConfirmationChoice(
            consumer.environment_id, consumer.preferred_alternative_id, True
        ),
    )

    result = reduce_toolbox_confirmation(
        active_definition=active,
        proposed_definition=proposed,
        environment_mutations=(producer, consumer),
        choices=choices,
    )

    assert result.effective_definition.auto_requests == ()
    assert {item["tool_key"]: item["reason"] for item in result.skipped_tools} == {
        "pkg.consumer:Consumer": "shared_environment_incomplete",
        "pkg.producer:Producer": "package_changes_declined",
    }
    bad = list(choices)
    bad[0] = ToolboxEnvironmentConfirmationChoice(
        producer.environment_id, "sha256:" + "f" * 64, True
    )
    with pytest.raises(ValueError, match="toolbox_confirmation_alternative_not_offered"):
        reduce_toolbox_confirmation(
            active_definition=active,
            proposed_definition=proposed,
            environment_mutations=(producer, consumer),
            choices=bad,
        )


def test_plan_offer_models_reject_source_secrets_and_more_than_three_alternatives() -> None:
    mutation = ToolboxPackageMutationSpec(
        distribution="demo-pkg",
        mutation="addition",
        dependency_reason="direct",
        from_version=None,
        to_version="1.0.0",
    )
    alternative = _alternative("one", mutations=(mutation,))
    assert ToolboxResolutionAlternativeSpec.from_dict(alternative.to_dict()) == alternative
    leaked = alternative.to_dict()
    leaked["source_origin"] = "https://user:secret@packages.example.invalid/simple?token=x"
    with pytest.raises(ValueError, match="toolbox_plan_source_origin_invalid"):
        ToolboxResolutionAlternativeSpec.from_dict(leaked)
    with pytest.raises(ValueError, match="toolbox_plan_alternatives_invalid"):
        ToolboxEnvironmentMutationSpec(
            environment_id="sha256:" + "1" * 64,
            tool_mutations=(ToolboxToolMutationSpec("pkg.add:Add", "added"),),
            base_template_id="core",
            base_template_revision="sha256:" + "2" * 64,
            alternatives=(alternative, alternative, alternative, alternative),
            preferred_alternative_id=alternative.alternative_id,
            alternatives_truncated=True,
            confirmation_required=True,
            dependency_approval_required=True,
            dependency_edges=(
                ToolboxDependencyEdgeSpec("pkg.add:Add", (), ("demo-pkg",)),
            ),
        )
