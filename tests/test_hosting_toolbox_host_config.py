from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import pytest

from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_materialization import (
    ToolboxTemplateMaterializationReceipt,
    derived_environment_digest,
)
from hosting.toolbox.host_project_config import (
    ToolboxHostProjectConfiguration,
)
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.target import detect_current_toolbox_target
from hosting_toolbox_test_catalog import realized_test_catalog


TARGET = detect_current_toolbox_target()
BUILTIN_IDS = ("core", "py-compute")


class VerifiedMaterializer:
    def materialize(
        self,
        *,
        catalog_entry: Mapping[str, Any],
        python_abi: str,
        platform: str,
        progress,
    ) -> ToolboxTemplateMaterializationReceipt:
        artifacts = tuple(sorted(item["sha256"] for item in catalog_entry["artifacts"]))
        roots = tuple(sorted(catalog_entry["template"]["exposed_import_roots"]))
        progress("artifact_verification", "artifacts_verified", 1, 1, "Artifacts verified.", True)
        progress("environment_build", "environment_built", 1, 1, "Environment built.", True)
        progress("import_probe", "imports_verified", len(roots), len(roots), "Imports verified.", False)
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
            verifier="host-project-config-test-v1",
        )


def _configuration() -> dict[str, Any]:
    return {
        "builtins": [
            {
                "template_id": "core",
                "imports": ["hosting", "mp13_engine", "packaging", "pydantic"],
                "package_requirements": [],
                "sandbox_policy": "compute-only",
                "required": True,
                "prewarm": True,
                "provenance": "parent-release",
            },
            {
                "template_id": "py-compute",
                "imports": [
                    "hosting", "mp13_engine", "mpmath", "numexpr", "numpy",
                    "packaging", "pydantic", "sympy",
                ],
                "package_requirements": ["numpy", "sympy", "numexpr", "mpmath"],
                "sandbox_policy": "compute-only",
                "required": True,
                "prewarm": True,
                "provenance": "parent-release",
            },
        ],
        "sources": [
            {
                "source_id": "parent-release-resources",
                "kind": "airgap_store",
                "origin": "airgap://parent-release-resources",
                "credential_ref": None,
                "allowed_package_namespaces": ["*"],
                "priority": 100,
                "trust_key_ids": ["parent-release-toolbox-v1"],
                "maximum_download_bytes": 536_870_912,
            }
        ],
        "resolution": {
            "mode": "air_gapped",
            "timeout_seconds": 300,
            "maximum_bytes": 536_870_912,
            "maximum_artifacts": 256,
            "allowed_redirect_origins": [],
            "wheel_only": True,
        },
        "retention": {
            "artifact_cache_grace_seconds": 604_800,
            "maximum_cache_bytes": 10_737_418_240,
            "maximum_cache_artifacts": 4096,
            "protected_digests": [],
            "remove_unreferenced_custom_revisions_on_apply": False,
        },
    }


def _service(root: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=root / "engines.json",
        control_state_file=root / "access_control.json",
        toolbox_template_materializer=VerifiedMaterializer(),
        toolbox_host_project_configuration=_configuration(),
    )


def test_host_project_configuration_is_strict_revisioned_and_current_target() -> None:
    config = ToolboxHostProjectConfiguration.from_dict(_configuration())
    assert tuple(item.template_id for item in config.builtins) == BUILTIN_IDS
    assert config.target == TARGET
    assert config.config_revision.startswith("sha256:")
    assert config.source_set_revision.startswith("sha256:")
    assert ToolboxHostProjectConfiguration.from_dict(config.to_dict()) == config

    invalid = _configuration() | {"required_target": TARGET.name}
    with pytest.raises(ValueError, match="unknown_fields:required_target"):
        ToolboxHostProjectConfiguration.from_dict(invalid)
    with pytest.raises(ValueError, match="unknown_fields"):
        ToolboxHostProjectConfiguration.from_dict(_configuration() | {"toolbox_id": "mutable"})
    reordered = _configuration()
    reordered["sources"] = [
        {**reordered["sources"][0], "priority": 1},
        {
            **reordered["sources"][0],
            "source_id": "higher-priority",
            "origin": "airgap://higher-priority",
            "priority": 2,
        },
    ]
    with pytest.raises(ValueError, match="source_priority_order_invalid"):
        ToolboxHostProjectConfiguration.from_dict(reordered)


def test_configured_intents_remain_unpublished_until_exact_resolution(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    assert service._toolbox_startup["status"] == "not_ready"  # noqa: SLF001
    assert service._toolbox_startup["closures"] == []  # noqa: SLF001
    assert service._toolbox_startup["published"] == []  # noqa: SLF001
    assert service._toolbox_startup["operations"] == []  # noqa: SLF001
    assert service._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    summary = service.hosting_setup_summary()
    assert summary["toolbox_readiness"]["status"] == "degraded"
    assert summary["toolbox_readiness"]["code"] == "required_template_missing"
    parsed = ToolboxHostProjectConfiguration.from_dict(_configuration())
    assert summary["toolbox_host_project"] == parsed.public_dict()
    assert "credential_ref" not in str(summary["toolbox_host_project"])
    serialized = str(summary["toolbox_readiness"])
    assert "artifact_source" not in serialized
    assert "python_executable" not in serialized
    assert "installer" not in serialized


def test_airgap_source_rejects_paths_credentials_and_https_mix() -> None:
    configured = _configuration()
    configured["sources"] = [
        {**configured["sources"][0], "origin": "C:/packages"}
    ]
    with pytest.raises(ValueError, match="package_source_origin_invalid"):
        ToolboxHostProjectConfiguration.from_dict(configured)

    configured = _configuration()
    configured["sources"] = [
        {**configured["sources"][0], "credential_ref": "secret:airgap"}
    ]
    with pytest.raises(ValueError, match="credential_ref_forbidden"):
        ToolboxHostProjectConfiguration.from_dict(configured)


def test_realized_shipped_catalog_and_lock_resources_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src/hosting/toolbox/shipped_templates.py").exists()
    resources = root / "src/hosting/resources/toolbox_templates"
    assert not (resources / "catalog.json").exists()
    assert not list(resources.glob("*.lock.json"))


def test_admin_immutable_template_replacement_survives_restart(tmp_path: Path) -> None:
    service = _service(tmp_path)
    shipped = realized_test_catalog()
    release = shipped.release("core")
    replacement = release.template.to_dict()
    replacement["provenance"] = {
        **replacement["provenance"],
        "revision": "2026.08.08.2",
        "manifest_digest": identity_digest(
            "hosting.toolbox.test.replacement.v1", {"template_id": "core", "version": 2}
        ),
    }
    published = service.toolbox_template_publish(
        template=replacement,
        artifact_references=[release.artifact_reference()],
        manifest_signature=release.manifest_signature,
        activate=True,
        actor_id="admin:replacement-test",
    )
    assert published["outcome"] == "published_and_activated"
    operation = service.toolbox_template_prewarm(
        template_id="core",
        template_digest=published["template_digest"],
        python_abi=TARGET.python_abi,
        platform=TARGET.platform,
        request_id="admin-replacement-core-v2",
        owner_actor_id="admin:replacement-test",
    )
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=operation["operation"]["operation_id"], timeout_seconds=10
    )
    assert terminal["lifecycle"] == "terminal_success"
    assert service.toolbox_required_template_status(
        python_abi=TARGET.python_abi, platform=TARGET.platform
    )["templates"][0]["template_digest"] == published["template_digest"]
    service.close()

    restarted = _service(tmp_path)
    state = restarted._toolbox_template_catalog.read()  # noqa: SLF001
    core_entries = [item for item in state["entries"] if item["template_id"] == "core"]
    assert len(core_entries) == 1
    assert state["active"]["core"] == published["template_digest"]
    status = restarted.hosting_setup_summary()["toolbox_readiness"]
    assert status["status"] == "degraded"
    assert status["templates"][0]["template_digest"] == published["template_digest"]
