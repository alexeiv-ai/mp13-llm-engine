from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hosting.engine_host_channel import EngineHostControlChannel
from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_state_v2 import AtomicJsonToolboxStateV2Repository
from hosting.toolbox.bundle_models import ToolboxDefinitionSpec
from hosting.toolbox.hosted_ref import HostedToolBoxRef
from hosting.toolbox.target import detect_current_toolbox_target
from tests.hosting_v3_fixtures import hosting_configuration


def test_removed_public_payload_and_state_shapes_are_rejected() -> None:
    empty = AtomicJsonToolboxStateV2Repository._payload({})  # noqa: SLF001
    with pytest.raises(ValueError, match="toolbox_state_v2_fields_invalid"):
        AtomicJsonToolboxStateV2Repository._validate_state(  # noqa: SLF001
            {**empty, "environment_descriptions": {}},
        )
    with pytest.raises(ValueError, match="toolbox_state_v2_contract_invalid"):
        AtomicJsonToolboxStateV2Repository._validate_state(  # noqa: SLF001
            {**empty, "version": 1},
        )

    definition = {
        "contract": "hosting.toolbox.definition",
        "toolbox_id": "demo",
        "expected_revision": None,
        "auto_requests": [],
        "manual_requests": [],
        "intrinsics": {"names": [], "include_guides": False, "sandbox_policy": {}},
    }
    for field, value in (
        ("environment_name", "base"),
        ("required_imports", ["numpy"]),
        ("profile_id", "consumer-profile"),
        ("python_executable", "python"),
    ):
        with pytest.raises(ValueError, match="toolbox_definition_unknown_fields"):
            ToolboxDefinitionSpec.from_dict({**definition, field: value})
    with pytest.raises(ValueError, match="legacy_toolbox_runtime_selector_rejected"):
        HostedToolBoxRef.from_dict(
            {"toolbox_id": "demo", "worker_profile_class": "legacy"}, host=object()
        )


def test_supported_public_surfaces_have_no_mutation_or_install_sequence() -> None:
    forbidden_prefixes = (
        "register_",
        "unregister_",
        "add_",
        "remove_",
        "environment_description",
        "prepare_environment",
        "lock_environment",
        "execute_environment_install",
    )
    for target in (HostedToolBoxRef, EngineHostControlChannel):
        public = {
            name
            for name, member in inspect.getmembers(target)
            if not name.startswith("_") and callable(member)
        }
        if target is EngineHostControlChannel:
            public = {name for name in public if name.startswith("toolbox_")}
        assert not {
            name for name in public if any(name.startswith(prefix) for prefix in forbidden_prefixes)
        }
    execute_parameters = set(inspect.signature(EngineHostService.toolbox_execute).parameters)
    assert not {
        "environment_name",
        "required_imports",
        "profile_id",
        "python_executable",
    } & execute_parameters


def test_only_explicit_release_archival_can_reference_version_one_state() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "hosting"
    allowed = {
        root / "service" / "host_service.py",
        root / "service" / "toolbox_state_cutover.py",
    }
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "toolbox_sandboxes.json" in text and path not in allowed:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_unconfigured_toolbox_readiness_uses_generic_hosting_code(tmp_path: Path) -> None:
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        hosting_configuration=hosting_configuration(tmp_path),
    )
    target = detect_current_toolbox_target()

    readiness = service.toolbox_required_template_status(
        python_abi=target.python_abi,
        platform=target.platform,
    )

    assert readiness["code"] == "hosting_configuration_missing"
    assert readiness["diagnostics"][0]["code"] == "hosting_configuration_missing"
