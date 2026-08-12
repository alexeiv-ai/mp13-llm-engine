from __future__ import annotations

from pathlib import Path

from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration
from hosting.toolbox.sandbox_policies import (
    compute_only_sandbox_policy,
    compute_only_worker_policy,
)


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
