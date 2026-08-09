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
    standard_toolbox_host_project_configuration,
    validate_toolbox_sandbox_policies,
)
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.shipped_templates import (
    SHIPPED_TEMPLATE_IDS,
    compute_only_sandbox_policy,
    load_shipped_toolbox_catalog,
)


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
    return standard_toolbox_host_project_configuration(target="cp312-win_amd64")


def _service(root: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=root / "engines.json",
        control_state_file=root / "access_control.json",
        toolbox_template_materializer=VerifiedMaterializer(),
        toolbox_environment_catalog=_configuration(),
        toolbox_sandbox_policies={"compute_only": compute_only_sandbox_policy()},
    )


def _wait_startup(service: EngineHostService) -> None:
    for operation in service._toolbox_startup["operations"]:  # noqa: SLF001
        terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
            operation_id=operation["operation"]["operation_id"],
            timeout_seconds=10,
        )
        assert terminal["lifecycle"] == "terminal_success"


def test_host_project_configuration_is_exact_and_compute_only() -> None:
    config = ToolboxHostProjectConfiguration.from_dict(_configuration())
    assert config.required_template_ids == SHIPPED_TEMPLATE_IDS
    assert config.target == ("cp312", "win_amd64")
    assert validate_toolbox_sandbox_policies(
        {"compute_only": compute_only_sandbox_policy()}
    )["compute_only"]["policy_id"] == "compute-only"

    invalid = _configuration() | {"required_template_ids": ["core"]}
    with pytest.raises(ValueError, match="required_template_ids_invalid"):
        ToolboxHostProjectConfiguration.from_dict(invalid)
    with pytest.raises(ValueError, match="unknown_fields"):
        ToolboxHostProjectConfiguration.from_dict(_configuration() | {"toolbox_id": "mutable"})
    widened = compute_only_sandbox_policy() | {"network": True}
    with pytest.raises(ValueError, match="compute_only_policy_invalid"):
        validate_toolbox_sandbox_policies({"compute_only": widened})


def test_configured_startup_publishes_prewarms_and_reports_bounded_readiness(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    assert [item["template_id"] for item in service._toolbox_startup["published"]] == list(  # noqa: SLF001
        SHIPPED_TEMPLATE_IDS
    )
    _wait_startup(service)
    summary = service.hosting_setup_summary()
    assert summary["toolbox_environment_catalog"]["status"] == "ready"
    assert summary["toolbox_host_project"] == {
        "resource": _configuration()["resource"],
        "required_template_ids": list(SHIPPED_TEMPLATE_IDS),
        "required_target": "cp312-win_amd64",
        "prewarm_required": True,
        "compute_only_policy_id": "compute-only",
    }
    serialized = str(summary["toolbox_environment_catalog"])
    assert "artifact_source" not in serialized
    assert "python_executable" not in serialized
    assert "installer" not in serialized


def test_admin_immutable_template_replacement_survives_restart(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _wait_startup(service)
    shipped = load_shipped_toolbox_catalog()
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
        python_abi="cp312",
        platform="win_amd64",
        request_id="admin-replacement-core-v2",
        owner_actor_id="admin:replacement-test",
    )
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=operation["operation"]["operation_id"], timeout_seconds=10
    )
    assert terminal["lifecycle"] == "terminal_success"
    assert service.toolbox_required_template_status(
        python_abi="cp312", platform="win_amd64"
    )["templates"][0]["template_digest"] == published["template_digest"]
    service.close()

    restarted = _service(tmp_path)
    _wait_startup(restarted)
    state = restarted._toolbox_template_catalog.read()  # noqa: SLF001
    core_entries = [item for item in state["entries"] if item["template_id"] == "core"]
    assert len(core_entries) == 2
    assert state["active"]["core"] == published["template_digest"]
    status = restarted.hosting_setup_summary()["toolbox_environment_catalog"]
    assert status["status"] == "ready"
    assert status["templates"][0]["template_digest"] == published["template_digest"]
