from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "src" / "hosting" / "HOSTED_TOOLBOX_CONTRACT.md"
HANDOFF = ROOT / "src" / "hosting" / "HOSTING_CLIENT_BREAKING_CHANGES.md"
ACCESS = ROOT / "src" / "hosting" / "HOSTING_ACCESS.md"
OLD_OPERATION_CONTRACT = ROOT / "src" / "hosting" / "HOSTING_OPERATION_CONTRACT.md"


def _contract_text() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def test_contract_has_frozen_public_sections_and_limits() -> None:
    text = _contract_text()
    required = [
        "## Validation limits",
        "## Canonical identities",
        "## ToolboxDefinitionSpec",
        "## ToolboxAutoAssignmentRequestV2",
        "## ToolboxManualAssignmentRequestV2",
        "## ToolboxDependencyRequest",
        "## Environment template descriptor",
        "## Deployment administration policy",
        "## Initial environment catalog",
        "## Cross-worker use of core",
        "## Model runtime boundary",
        "## Planning",
        "## Dependency approval references",
        "## Authoritative read",
        "## Apply and durable operation behavior",
        "## Public client surface",
        "## Actor authorization",
        "## User and operator projections",
        "## Stable error codes",
        "## Client algorithm",
    ]
    assert all(section in text for section in required)
    for frozen_limit in [
        "32 MiB",
        "Auto requests per definition | 512",
        "Manual requests per definition | 512",
        "Files per request | 256",
        "All file content in one definition | 24 MiB",
        "User diagnostics returned | 64",
        "Rollout history entries in a read snapshot | 32",
    ]:
        assert frozen_limit in text


def test_contract_json_examples_are_valid() -> None:
    blocks = re.findall(r"```json\n(.*?)\n```", _contract_text(), flags=re.DOTALL)
    assert len(blocks) >= 7
    for block in blocks:
        assert isinstance(json.loads(block), dict)


def test_contract_freezes_strict_fields_scope_and_codes() -> None:
    text = _contract_text()
    prose = " ".join(text.split())
    for field in [
        '"contract": "hosting.toolbox.definition"',
        '"expected_revision"',
        '"auto_requests"',
        '"manual_requests"',
        '"intrinsics"',
        '"dependency"',
        '"mode"',
        '"template_id"',
        '"declared_imports"',
        '"package_requirements"',
        '"user_projection"',
        '"contract": "hosting.toolbox.definition_apply_result"',
    ]:
        assert field in text
    for code in [
        "definition_invalid",
        "definition_too_large",
        "duplicate_stable_key",
        "duplicate_tool_name",
        "revision_conflict",
        "dependency_unresolved",
        "dependency_approval_required",
        "dependency_approval_invalid",
        "plan_expired",
        "plan_stale",
        "request_id_conflict",
        "state_corrupt",
        "apply_publication_committed",
    ]:
        assert f"`{code}`" in text
    assert "Name uniqueness is per toolbox." in text
    assert "The same advertised name is valid in different toolboxes." in prose
    assert "Applying an empty complete definition" in prose
    for signature in [
        "get_definition(*, operator_details: bool = False)",
        "plan_definition(definition: dict, *, operator_details: bool = False)",
        "approve_definition_plan(*, plan_id: str)",
        "dependency_approval_ref: str | None = None",
    ]:
        assert signature in text


def test_contract_contains_only_supported_vocabulary() -> None:
    text = _contract_text().lower()
    forbidden = [
        "register_",
        "unregister_",
        "environment_descriptions",
        "version-1",
        "migration",
        "deprecated behavior",
        "compatibility",
    ]
    assert not {item for item in forbidden if item in text}


def test_handoff_specifies_pending_corrective_migration() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    prose = " ".join(text.split())
    assert "[Hosted Toolbox Definition Contract](HOSTED_TOOLBOX_CONTRACT.md)" in text
    assert "[Hosting Access §11.6]" in text
    assert "Status: migration required before corrective hosting rollout (2026-08-10)" in text
    assert "mp13-docs" in prose
    assert "125d20f232bf5b755d18c1b23bc1e4b8929edf21" in text
    for required in [
        "## Adoption gate and pins",
        "## Removed target and host-configuration contract",
        "## Removed toolbox mutation commands and fields",
        "## Exact replacement request sequence",
        "## Retry, watch, confirmation, and recovery logic",
        "## Dependent code, configuration, tests, and documentation",
        '"command": "toolbox-confirm-definition-plan"',
        '"confirmation_ref": "opaque-confirmation-ref"',
        "toolbox-approve-confirmed-definition-plan",
        "shared_environment_incomplete",
    ]:
        assert required in text
    assert "adoption is prohibited" in prose
    assert "No pending client-breaking-change action remains" not in prose


def test_generic_operation_contract_is_consolidated_into_hosting_access() -> None:
    access = ACCESS.read_text(encoding="utf-8")
    toolbox = CONTRACT.read_text(encoding="utf-8")
    assert not OLD_OPERATION_CONTRACT.exists()
    for section in [
        "### 11.6 Durable hosted operation and capability contract",
        "#### 11.6.1 Operation identity and idempotency",
        "#### 11.6.2 Lifecycle, progress, and terminal results",
        "#### 11.6.3 Fingerprints and request recovery",
        "#### 11.6.4 Repository and restart behavior",
        "#### 11.6.5 Authorization",
        "#### 11.6.6 Provider sessions, callbacks, and capability authority",
    ]:
        assert section in access
    for required in [
        "HostedOperationRef.from_dict",
        "(owner_actor_id, receipt_namespace, request_id)",
        "hosted_operation_resolve_request",
        "result_omission",
        "provider_call_id",
        "owner_authority_id",
        "on_transport_loss",
    ]:
        assert required in access
    assert "[Hosting Access §11.6](HOSTING_ACCESS.md#116-durable-hosted-operation-and-capability-contract)" in toolbox


def test_contract_freezes_deployment_administration_policy() -> None:
    text = _contract_text()
    for role in [
        "toolbox_consumer",
        "toolbox_dependency_approver",
        "hosting_template_admin",
        "hosting_auditor",
    ]:
        assert f"`{role}`" in text
    for method in [
        "toolbox-template-list",
        "toolbox-template-describe",
        "toolbox-template-publish",
        "toolbox-template-deprecate",
        "toolbox-template-revoke",
        "toolbox-template-prewarm",
    ]:
        assert f"`{method}`" in text
    for required in [
        "The signature algorithm is `ed25519`.",
        "Online index resolution is denied by default.",
        "CPython 3.12 on `win_amd64`, `win_arm64`",
        "`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`",
        "`macosx_11_0_arm64`",
        "`packaging.tags.sys_tags()`",
        "300 seconds per artifact fetch",
        "1,800 seconds for one environment materialization",
        "3,600 seconds for one prewarm or",
        "Template lifecycle is `active`, `deprecated`, or `revoked`.",
        "seven-day grace period",
        "configurable from one to 90 days",
    ]:
        assert required in text


def test_contract_freezes_initial_environment_catalog_and_config() -> None:
    text = _contract_text()
    prose = " ".join(text.split())
    for required in [
        "exactly two stable logical template IDs: `core` and `py-compute`",
        "Logical IDs have no version suffix.",
        "signed complete manifests with complete distribution locks",
        "parent worker artifact digest, and isolation policy version",
        "pinned NumPy, SymPy, NumExpr",
        "independently materialized lock",
        "built-in `sandbox_policy` reference is `compute-only`",
        "`toolbox_host_project_configuration`",
        "`required_template_missing`",
        "`required_template_signature_invalid`",
        "`required_template_lock_invalid`",
        "`required_template_artifact_unavailable`",
        "`required_template_materialization_failed`",
        "`required_template_probe_failed`",
        "`compute_only_policy_unenforceable`",
        "Selection always chooses the smallest allowed complete template.",
    ]:
        assert required in prose
    for key in [
        "`builtins`",
        "`sources`",
        "`resolution`",
        "`retention`",
        "`package_requirements`",
        "`credential_ref`",
        "`source_set_revision`",
        "`toolbox_artifact_sources`",
        "`toolbox_dependency_policy`",
        "`artifact_cache_grace_seconds`",
        "`remove_unreferenced_custom_revisions_on_apply`",
    ]:
        assert key in text
    policy_block = next(
        json.loads(block)
        for block in re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL)
        if '"policy_id": "compute-only"' in block
    )
    assert policy_block == {
        "policy_id": "compute-only",
        "sandbox_required": True,
        "filesystem_read_roots": [],
        "filesystem_write_roots": [],
        "artifact_roots": [],
        "network": False,
        "subprocess": False,
        "brokered_io": {
            "filesystem": False,
            "http": False,
            "subprocess": False,
        },
        "host_api_permissions": [],
    }


def test_contract_keeps_cross_worker_core_consumers_separate() -> None:
    text = _contract_text()
    for required in [
        "standard-library-only toolbox functions",
        "`workflow_python(profile=node)`",
        "Python workflow helper workers",
        "This is environment reuse, not worker or protocol unification.",
        "worker-pool identity",
        "Pools do not exchange live interpreters",
        "do not expose a generic Python execution endpoint",
    ]:
        assert required in text


def test_contract_freezes_exclusive_model_runtime_boundary() -> None:
    text = _contract_text()
    prose = " ".join(text.split())
    for required in [
        "root `pyproject.toml`",
        "administrator-configured optional model package set",
        "The exact host configuration namespace is `model_runtime`",
        "Only authenticated model operations may activate this runtime",
        "It is not a generic interpreter or arbitrary-code route.",
        "`ModelRuntimeStatus`",
        "model authorization, resource, network, data-access, and secret policies",
        "does not cause toolbox catalog fallback",
    ]:
        assert required in prose
    for key in [
        "`project_resource`",
        "`lock_resource`",
        "`optional_package_set`",
        "`required_target`",
        "`engine_artifact_digest`",
        "`readiness_required`",
    ]:
        assert key in text
    for field in [
        "`state`",
        "`code`",
        "`summary`",
        "`python_abi`",
        "`platform`",
        "`complete_lock_digest`",
        "`materialization_revision`",
        "`updated_at_ms`",
    ]:
        assert field in text
