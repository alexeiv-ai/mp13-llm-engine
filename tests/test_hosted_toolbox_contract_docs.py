from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "src" / "hosting" / "HOSTED_TOOLBOX_CONTRACT.md"
HANDOFF = ROOT / "src" / "hosting" / "HOSTING_CLIENT_BREAKING_CHANGES.md"


def _contract_text() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def test_contract_has_frozen_public_sections_and_limits() -> None:
    text = _contract_text()
    required = [
        "## Validation limits",
        "## ToolboxDefinitionSpec",
        "## ToolboxAutoAssignmentRequestV2",
        "## ToolboxManualAssignmentRequestV2",
        "## ToolboxDependencyRequest",
        "## Environment template descriptor",
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
        "deprecated",
        "compatibility",
    ]
    assert not {item for item in forbidden if item in text}


def test_handoff_links_contract_and_keeps_client_removal_instructions() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "[Hosted Toolbox Definition Contract](HOSTED_TOOLBOX_CONTRACT.md)" in text
    assert "### Required dependent-project logic change" in text
    assert "### Deprecated behavior to remove from dependents" in text
    assert "retire_toolbox_daemon_registration()" in text
    assert "_run_environment_checks()" in text
