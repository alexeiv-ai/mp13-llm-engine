from __future__ import annotations

import copy
import os
import sys

import pytest

from hosting.toolbox.bundle_models import ToolboxDefinitionSpec
from hosting.toolbox.definition_planner import plan_toolbox_definition
from hosting.toolbox.shipped_templates import load_shipped_toolbox_catalog


def _dependency(*, mode: str = "auto", template_id: str | None = None, imports=(), requirements=()):
    return {
        "mode": mode,
        "template_id": template_id,
        "declared_imports": list(imports),
        "package_requirements": list(requirements),
    }


def _auto(name: str, *, source: str = "", dependency=None, sandbox=None):
    module = name.lower()
    return {
        "files": [{"relative_path": f"pkg/{module}.py", "content": source or f"def {name}():\n    return 1\n"}],
        "module_name": f"pkg.{module}",
        "callable_name": name,
        "dependency": dependency or _dependency(),
        "sandbox_policy": sandbox or {"sandbox": {"enabled": True}},
        "activate": True,
        "hidden": False,
        "non_restartable": False,
        "guide_content": None,
        "guide_description": None,
        "callback_signature": None,
        "concurrency": None,
    }


def _manual(advertised: str, *, callable_name: str | None = None, dependency=None, sandbox=None):
    callable_name = callable_name or advertised
    return {
        "files": [{"relative_path": f"pkg/{callable_name.lower()}.py", "content": f"def {callable_name}():\n    return 1\n"}],
        "module_name": f"pkg.{callable_name.lower()}",
        "callable_name": callable_name,
        "tool_definition": {
            "type": "function",
            "function": {"name": advertised, "description": "demo", "parameters": {"type": "object", "properties": {}}},
        },
        "dependency": dependency or _dependency(),
        "sandbox_policy": sandbox or {"sandbox": {"enabled": True}},
        "hidden": False,
        "non_restartable": False,
        "callback_signature": None,
        "concurrency": None,
    }


def _definition(*, toolbox_id="demo", autos=(), manuals=(), intrinsics=()):
    return {
        "contract": "hosting.toolbox.definition",
        "toolbox_id": toolbox_id,
        "expected_revision": None,
        "auto_requests": list(autos),
        "manual_requests": list(manuals),
        "intrinsics": {
            "names": list(intrinsics),
            "include_guides": False,
            "sandbox_policy": {"sandbox": {"enabled": True}},
        },
    }


def _plan(definition):
    shipped = load_shipped_toolbox_catalog()
    return plan_toolbox_definition(
        definition,
        templates=shipped.templates,
        python_abi=f"cp{sys.version_info.major}{sys.version_info.minor}",
        platform="win_amd64" if os.name == "nt" else "manylinux_2_28_x86_64",
        runtime_identity={"version": "3.12.7", "artifact_digest": "sha256:" + "a" * 64},
    )


def test_definition_parser_is_strict_at_every_versioned_request_boundary() -> None:
    payload = _definition(autos=[_auto("Alpha")])
    model = ToolboxDefinitionSpec.from_dict(payload)
    assert model.to_dict() == payload

    unknown_definition = {**payload, "runtime": {}}
    with pytest.raises(ValueError, match="toolbox_definition_unknown_fields:runtime"):
        ToolboxDefinitionSpec.from_dict(unknown_definition)

    unknown_request = copy.deepcopy(payload)
    unknown_request["auto_requests"][0]["sandbox_profile"] = {"environment_name": "local"}
    with pytest.raises(ValueError, match="toolbox_auto_request_v2_unknown_fields:sandbox_profile"):
        ToolboxDefinitionSpec.from_dict(unknown_request)

    unknown_dependency = copy.deepcopy(payload)
    unknown_dependency["auto_requests"][0]["dependency"]["allow_resolution"] = True
    with pytest.raises(ValueError, match="toolbox_dependency_unknown_fields:allow_resolution"):
        ToolboxDefinitionSpec.from_dict(unknown_dependency)


def test_definition_canonicalization_sorts_requests_files_and_dependency_lists() -> None:
    alpha = _auto("Alpha", dependency=_dependency(imports=("json", "collections")))
    alpha["files"].append({"relative_path": "pkg/z.py", "content": "Z = 1\n"})
    beta = _auto("Beta")
    payload = _definition(autos=[beta, alpha])
    model = ToolboxDefinitionSpec.from_dict(payload)
    expected_revision = model.revision
    reordered = copy.deepcopy(payload)
    reordered["auto_requests"].reverse()
    reordered["auto_requests"][1]["files"].reverse()
    reordered["auto_requests"][1]["dependency"]["declared_imports"].reverse()
    reordered["expected_revision"] = "sha256:" + "f" * 64

    assert ToolboxDefinitionSpec.from_dict(reordered).revision == expected_revision
    assert [item.stable_key for item in model.auto_requests] == sorted(item.stable_key for item in model.auto_requests)


@pytest.mark.parametrize(
    "definition",
    [
        _definition(autos=[_auto("Shared")], manuals=[_manual("Shared", callable_name="Other")]),
        _definition(manuals=[_manual("symbolic_algebra")], intrinsics=["symbolic_algebra"]),
    ],
)
def test_duplicate_advertised_names_are_rejected_within_one_toolbox(definition) -> None:
    with pytest.raises(ValueError, match="toolbox_definition_duplicate_advertised_name"):
        _plan(definition)


def test_same_advertised_name_in_separate_toolboxes_is_valid() -> None:
    first = _plan(_definition(toolbox_id="one", autos=[_auto("Shared")]))
    second = _plan(_definition(toolbox_id="two", autos=[_auto("Shared")]))
    assert first.definition.toolbox_id == "one"
    assert second.definition.toolbox_id == "two"


def test_grouping_occurs_after_resolution_and_ignores_raw_import_subset() -> None:
    plan = _plan(
        _definition(
            autos=[
                _auto("JsonTool", source="import json\ndef JsonTool():\n    return json.dumps({})\n"),
                _auto("MathTool", source="import math\ndef MathTool():\n    return math.sqrt(4)\n"),
            ]
        )
    )
    assert len(plan.profiles) == 1
    assert plan.profiles[0].template_id == "core"
    assert plan.profiles[0].assigned_tool_keys == ("pkg.jsontool:JsonTool", "pkg.mathtool:MathTool")
    assert len(plan.bundles) == 1


def test_sandbox_policy_remains_a_profile_boundary() -> None:
    plan = _plan(
        _definition(
            autos=[
                _auto("Alpha"),
                _auto("Beta", sandbox={"sandbox": {"enabled": True, "network": {"mode": "disabled"}}}),
            ]
        )
    )
    assert len(plan.profiles) == 2


def test_custom_request_emits_custom_lock_environment_and_resolved_manifest() -> None:
    plan = _plan(
        _definition(
            autos=[
                _auto(
                    "Fetch",
                    source="import requests\ndef Fetch():\n    return requests.__version__\n",
                    dependency=_dependency(
                        mode="custom",
                        template_id="core",
                        requirements=("requests==2.32.5",),
                    ),
                )
            ]
        )
    )
    profile = plan.profiles[0]
    manifest = plan.bundles[0].manifest_payload()
    assert plan.custom_environment_count == 1
    assert profile.custom_resolved_lock_digest is not None
    assert manifest["dependency_lock_hash"] == profile.custom_resolved_lock_digest
    assert manifest["resolved_profile"] == profile.to_dict()


def test_explicit_template_fails_when_source_requires_a_custom_delta() -> None:
    definition = _definition(
        autos=[
            _auto(
                "Fetch",
                source="import requests\ndef Fetch():\n    return 1\n",
                dependency=_dependency(mode="template", template_id="core", requirements=("requests==2.32.5",)),
            )
        ]
    )
    with pytest.raises(ValueError, match="definition_explicit_template_incomplete"):
        _plan(definition)


def test_conflicting_normalized_file_content_is_rejected_before_resolution() -> None:
    alpha = _auto("Alpha")
    beta = _auto("Beta")
    beta["files"][0] = {"relative_path": "PKG/ALPHA.PY", "content": "different\n"}
    with pytest.raises(ValueError, match="toolbox_definition_file_conflict"):
        _plan(_definition(autos=[alpha, beta]))


def test_empty_definition_is_a_valid_revision_with_no_profiles() -> None:
    plan = _plan(_definition())
    assert plan.definition.revision.startswith("sha256:")
    assert plan.profiles == ()
    assert plan.bundles == ()
