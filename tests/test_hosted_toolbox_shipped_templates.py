from __future__ import annotations

import time
import tomllib
from pathlib import Path
from typing import Any, Mapping

import pytest

from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_materialization import (
    ToolboxTemplateMaterializationReceipt,
    derived_environment_digest,
)
from hosting.toolbox.bundle_models import ToolboxBundleFile
from hosting.toolbox.catalog import ToolboxEnvironmentTemplateSpec
from hosting.toolbox.dependency_analysis import (
    analyze_toolbox_bundle_imports,
    resolve_toolbox_dependencies,
    select_toolbox_environment_template,
)
from hosting.toolbox.shipped_templates import (
    SHIPPED_CATALOG_RESOURCE,
    SHIPPED_TEMPLATE_IDS,
    compute_only_sandbox_policy,
    compute_only_worker_policy,
    load_shipped_toolbox_catalog,
)
from mp13_engine.mp13_intrinsics_metadata import intrinsic_dependency_metadata


ROOT = Path(__file__).resolve().parents[1]


def _poetry_versions() -> dict[str, str]:
    with (ROOT / "poetry.lock").open("rb") as handle:
        payload = tomllib.load(handle)
    return {item["name"]: item["version"] for item in payload["package"]}


class ImportProbeMaterializer:
    def materialize(self, *, catalog_entry: Mapping[str, Any], python_abi: str, platform: str, progress):
        artifacts = tuple(sorted(item["sha256"] for item in catalog_entry["artifacts"]))
        roots = tuple(sorted(catalog_entry["template"]["exposed_import_roots"]))
        progress("artifact_verification", "shipped_lock_verified", 1, 1, "The shipped lock resource was verified.", True)
        progress("environment_build", "shipped_environment_available", 1, 1, "The isolated environment was materialized.", True)
        locked = {item["name"] for item in catalog_entry["template"]["locked_distributions"]}
        root_distributions = {
            "hosting": "mp13-engine",
            "mp13_engine": "mp13-engine",
            "mpmath": "mpmath",
            "numexpr": "numexpr",
            "numpy": "numpy",
            "packaging": "packaging",
            "pydantic": "pydantic",
            "sympy": "sympy",
        }
        for index, root in enumerate(roots, start=1):
            assert root_distributions[root] in locked
            progress("import_probe", "shipped_import_probe", index, len(roots), f"Verified import root {root}.", False)
        return ToolboxTemplateMaterializationReceipt(
            template_id=catalog_entry["template_id"],
            template_digest=catalog_entry["template_digest"],
            python_abi=python_abi,
            platform=platform,
            environment_digest=derived_environment_digest(
                template_digest=catalog_entry["template_digest"],
                python_abi=python_abi,
                platform=platform,
                artifact_digests=artifacts,
            ),
            artifact_digests=artifacts,
            verified_import_roots=roots,
            verified_at_ms=int(time.time() * 1000),
            verifier="shipped-import-probe-test-v1",
        )


def test_shipped_catalog_contains_exact_independent_complete_locks() -> None:
    catalog = load_shipped_toolbox_catalog()
    assert catalog.resource == SHIPPED_CATALOG_RESOURCE
    assert tuple(item.template.template_id for item in catalog.releases) == SHIPPED_TEMPLATE_IDS
    versions = _poetry_versions()
    core = catalog.release("core").template
    compute = catalog.release("py-compute").template
    core_lock = {item.name: item.version for item in core.locked_distributions}
    compute_lock = {item.name: item.version for item in compute.locked_distributions}
    assert set(core_lock) == {
        "annotated-types", "mp13-engine", "packaging", "pydantic",
        "pydantic-core", "typing-extensions", "typing-inspection",
    }
    assert set(compute_lock) == set(core_lock) | {"mpmath", "numexpr", "numpy", "sympy"}
    assert compute_lock is not core_lock
    for name, version in core_lock.items():
        if name != "mp13-engine":
            assert versions[name] == version
    for name, version in compute_lock.items():
        if name != "mp13-engine":
            assert versions[name] == version
    assert core.lock_digest != compute.lock_digest
    assert core.parent_worker_artifact_digest == compute.parent_worker_artifact_digest


def test_compute_lock_covers_all_intrinsic_metadata_exactly() -> None:
    compute = load_shipped_toolbox_catalog().release("py-compute").template
    locked = {item.name: item.version for item in compute.locked_distributions}
    dependencies = intrinsic_dependency_metadata(
        ["scriptable_calculator", "symbolic_algebra"]
    )
    assert set(dependencies["import_roots"]) <= set(compute.exposed_import_roots)
    for requirement in dependencies["package_requirements"]:
        name, version = requirement.split("==", 1)
        assert locked[name] == version


def test_compute_only_preset_is_exact_and_enforceable() -> None:
    assert compute_only_sandbox_policy() == {
        "policy_id": "compute-only",
        "sandbox_required": True,
        "filesystem_read_roots": [],
        "filesystem_write_roots": [],
        "artifact_roots": [],
        "network": False,
        "subprocess": False,
        "brokered_io": {"filesystem": False, "http": False, "subprocess": False},
        "host_api_permissions": [],
    }
    worker = compute_only_worker_policy()
    assert worker.summary() == {
        **worker.summary(),
        "enabled": True,
        "profile": "compute-only",
        "filesystem_rules_count": 0,
        "artifact_roots": {},
        "brokered_filesystem": False,
        "brokered_http": False,
        "allow_subprocess": False,
        "inherit_parent_handles": False,
        "network_mode": "disabled",
    }


def test_package_metadata_cannot_assert_sandbox_capability() -> None:
    payload = load_shipped_toolbox_catalog().release("core").template.to_dict()
    payload["locked_distributions"][0]["sandbox"] = {"network": True}
    with pytest.raises(ValueError, match="locked_distribution_unknown_fields:sandbox"):
        ToolboxEnvironmentTemplateSpec.from_dict(payload)


def test_planner_selects_core_then_py_compute_as_smallest_exact_template() -> None:
    templates = load_shipped_toolbox_catalog().templates
    stdlib = resolve_toolbox_dependencies(
        analyze_toolbox_bundle_imports(
            [ToolboxBundleFile(relative_path="tool.py", content="import json\n")]
        )
    )
    selected = select_toolbox_environment_template(
        stdlib, templates, python_abi="cp312", platform="win_amd64"
    )
    assert selected.mode == "template"
    assert selected.template.template_id == "core"

    compute = resolve_toolbox_dependencies(
        analyze_toolbox_bundle_imports(
            [ToolboxBundleFile(relative_path="tool.py", content="import numexpr\nimport numpy\nimport sympy\n")]
        )
    )
    selected = select_toolbox_environment_template(
        compute, templates, python_abi="cp312", platform="win_amd64"
    )
    assert selected.mode == "template"
    assert selected.template.template_id == "py-compute"


def test_normal_setup_publishes_prewarms_and_gates_bounded_readiness(tmp_path: Path) -> None:
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "access_control.json",
        toolbox_template_materializer=ImportProbeMaterializer(),
        toolbox_required_python_abi="cp312",
        toolbox_required_platform="win_amd64",
    )
    before = service.hosting_setup_summary()["toolbox_environment_catalog"]
    assert before["status"] == "degraded"
    assert before["code"] == "required_template_missing"
    started = service.initialize_shipped_toolbox_templates(
        python_abi="cp312",
        platform="win_amd64",
        request_id_prefix="normal-setup-1",
    )
    assert [item["template_id"] for item in started["published"]] == list(SHIPPED_TEMPLATE_IDS)
    for operation in started["operations"]:
        terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
            operation_id=operation["operation"]["operation_id"],
            timeout_seconds=10,
        )
        assert terminal["lifecycle"] == "terminal_success"
    status = service.hosting_setup_summary()["toolbox_environment_catalog"]
    assert status["status"] == "ready"
    assert status["code"] == "required_templates_ready"
    assert status["diagnostics"] == []
    assert all(item["ready"] for item in status["templates"])
    serialized = str(status)
    assert "filename" not in serialized
    assert "artifact_source" not in serialized
    assert "environment_digest" not in serialized
