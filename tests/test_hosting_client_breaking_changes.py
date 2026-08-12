from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "src" / "hosting" / "HOSTING_CLIENT_BREAKING_CHANGES.md"


def _text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_handoff_is_an_actionable_dependent_consumer_guide() -> None:
    text = _text()
    for section in [
        "# Dependent consumer migration guide: hosting control v3",
        "## 1. Required consumer changes at a glance",
        "## 2. Gate the daemon and capabilities",
        "## 3. Replace startup and configuration integration",
        "## 4. Preserve the complete authentication result",
        "## 5. Adopt generic package and environment contracts",
        "## 6. Replace readiness handling",
        "## 7. Persist and resolve v3 durable operations",
        "## 8. Implement atomic tool-change review",
        "## 9. Implement candidate validation",
        "## 10. Remove old state and compatibility behavior",
        "## 11. Consumer implementation checklist",
        "## 12. Required verification",
        "## 13. Adoption receipt",
    ]:
        assert section in text

    for required in [
        "hosting.control.v3",
        "4d01307f664366c3149bef539aaa1b4e3f98a82f",
        "package_artifact_ingress_v1",
        "package_locks_v1",
        "environment_management_v1",
        "environment_references_v1",
        "environment_execution_leases_v1",
        "toolbox_tool_changes_v1",
        "toolbox_definition_candidates_v1",
        "engine_host_mp13_config_file",
        "hosting.setup.v1",
        "hosting.operation_status.v3",
        "hosting.toolbox.definition_plan.v2",
        "hosting.toolbox.confirmation_receipt.v1",
        "hosting.toolbox.definition_candidate.v1",
    ]:
        assert required in text


def test_handoff_compares_removed_and_replacement_consumer_surfaces() -> None:
    text = _text()
    for old, new in [
        ("toolbox-artifact-upload-begin", "package-artifact-upload-begin"),
        ("toolbox-template-list", "environment-template-list"),
        ("toolbox-environment-remove", "environment-remove"),
        ("toolbox-gc", "hosting-gc"),
        ("toolbox_configuration_missing", "hosting_configuration_missing"),
        ("toolbox_source_binding_invalid", "package_source_invalid"),
    ]:
        assert f"`{old}`" in text
        assert f"`{new}`" in text

    for dependent_path in [
        "src/backend/app/factory.py",
        "src/backend/platform/hosting/daemon_contract.py",
        "src/backend/platform/hosting/hosting_admin.py",
        "src/backend/platform/toolboxes/definition_coordinator.py",
        "src/ui/web/static/js/features/chat/CapabilityToolsController.js",
        "src/ui/web/static/js/features/chat/CapabilityToolsPanel.js",
    ]:
        assert dependent_path in text


def test_handoff_excludes_parent_execution_ledger_and_internal_navigation() -> None:
    text = _text()
    for forbidden in [
        "hosting_access_plan.md",
        "## Handoff gate",
        "## Ownership and rollout",
        "Parent navigation:",
        "src/hosting/client_realm_api.py",
        "src/hosting/engine_host_channel.py",
        "contract_major",
        "hosting_contract_major_unsupported",
        "R0 contract freeze",
        "R9.6",
    ]:
        assert forbidden not in text
