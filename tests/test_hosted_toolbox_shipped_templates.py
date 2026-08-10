from __future__ import annotations

from pathlib import Path

import pytest

from hosting.daemon import EngineHostDaemon
from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.sandbox_policies import (
    compute_only_sandbox_policy,
    compute_only_worker_policy,
)
from hosting.toolbox.target import detect_current_toolbox_target


def _configuration() -> dict[str, object]:
    return {
        "builtins": [
            {
                "template_id": template_id,
                "imports": imports,
                "package_requirements": requirements,
                "sandbox_policy": "compute-only",
                "required": True,
                "prewarm": True,
                "provenance": "parent-release",
            }
            for template_id, imports, requirements in (
                ("core", ["hosting", "mp13_engine", "packaging", "pydantic"], []),
                (
                    "py-compute",
                    ["hosting", "mp13_engine", "mpmath", "numexpr", "numpy", "packaging", "pydantic", "sympy"],
                    ["numpy", "sympy", "numexpr", "mpmath"],
                ),
            )
        ],
        "sources": [
            {
                "source_id": "release-airgap",
                "kind": "airgap_store",
                "origin": "airgap://release-airgap",
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


def _policy() -> dict[str, object]:
    target = detect_current_toolbox_target()
    body = {
        "allowed_template_ids": ["core", "py-compute"],
        "allowed_targets": [target.name],
        "package_allowlist": [],
        "package_denylist": [],
        "allow_custom": False,
        "custom_requires_approval": True,
        "online_resolution_allowed": False,
        "allowed_index_origins": [],
    }
    return {"revision": identity_digest("hosting.toolbox.test.policy.v1", body), **body}


def test_compute_only_policy_is_independent_of_realized_templates() -> None:
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
    assert worker.enabled is True
    assert worker.network.mode == "disabled"
    assert worker.process.allow_subprocess is False


def test_builtin_intent_contains_no_resolved_lock_or_artifact() -> None:
    parsed = ToolboxHostProjectConfiguration.from_dict(_configuration())
    serialized = parsed.to_dict()
    forbidden = {"locked_distributions", "lock_digest", "artifacts", "filename"}
    assert all(not (forbidden & set(item)) for item in serialized["builtins"])
    assert [item.template_id for item in parsed.builtins] == ["core", "py-compute"]


def test_realized_shipped_resources_and_loader_are_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src/hosting/toolbox/shipped_templates.py").exists()
    resources = root / "src/hosting/resources/toolbox_templates"
    assert not (resources / "catalog.json").exists()
    assert not list(resources.glob("*.lock.json"))


def test_normal_daemon_does_not_publish_intent_before_exact_resolution(tmp_path: Path) -> None:
    source = tmp_path / "airgap"
    source.mkdir()
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=_configuration(),
        toolbox_artifact_sources={"release-airgap": source},
        toolbox_dependency_policy=_policy(),
    )

    assert daemon.svc._hermetic_toolbox_environment_builder is not None  # noqa: SLF001
    assert daemon.svc._toolbox_startup["status"] == "not_ready"  # noqa: SLF001
    assert daemon.svc._toolbox_startup["closures"] == []  # noqa: SLF001
    assert daemon.svc._toolbox_startup["diagnostics"][0]["code"] == (  # noqa: SLF001
        "required_template_requirements_missing"
    )
    assert daemon.svc._toolbox_startup["published"] == []  # noqa: SLF001
    assert daemon.svc._toolbox_startup["operations"] == []  # noqa: SLF001
    assert daemon.svc._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    readiness = daemon.svc.hosting_setup_summary()["toolbox_readiness"]
    assert readiness["status"] == "degraded"
    assert readiness["code"] == "required_template_missing"
    with pytest.raises(ValueError, match="toolbox_builtins_not_ready"):
        daemon.svc._toolbox_definition_planning_context()  # noqa: SLF001
