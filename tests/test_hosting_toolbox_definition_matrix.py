from __future__ import annotations

import copy
import os
import sys

import pytest

from hosting.toolbox.definition_planner import (
    classify_toolbox_profiles,
    plan_toolbox_definition,
    profile_snapshots_from_draft,
)
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
