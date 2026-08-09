from __future__ import annotations

import inspect
import os
from dataclasses import replace
from pathlib import Path

import pytest

from hosting.toolbox.bundle_models import ToolboxEnvironmentSpec
from hosting.toolbox.catalog import ToolboxLockedDistributionSpec
from hosting.toolbox.environment import RuntimeEnvironmentManager, ToolboxEnvironmentManager
from hosting.toolbox.hermetic_environment import (
    HermeticToolboxEnvironmentResolver,
    ResolvedToolboxEnvironmentInput,
)
from hosting.toolbox.orchestration import ToolboxSandboxOrchestrator


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _resolved(**overrides: object) -> ResolvedToolboxEnvironmentInput:
    values: dict[str, object] = {
        "template_id": "core",
        "template_digest": _digest("1"),
        "runtime_version": "3.12.8",
        "runtime_artifact_digest": _digest("2"),
        "python_abi": "cp312",
        "platform": "win_amd64",
        "complete_lock_digest": _digest("3"),
        "complete_lock": (
            ToolboxLockedDistributionSpec(name="mp13-engine", version="0.9.0"),
            ToolboxLockedDistributionSpec(name="pydantic", version="2.12.5"),
        ),
        "custom_resolved_lock_digest": None,
        "isolation_policy_version": "toolbox-isolation-v1",
        "resolved_import_roots": ("hosting", "pydantic"),
    }
    values.update(overrides)
    return ResolvedToolboxEnvironmentInput(**values)  # type: ignore[arg-type]


def test_resolved_environment_input_is_strict_and_round_trips() -> None:
    resolved = _resolved()
    assert ResolvedToolboxEnvironmentInput.from_dict(resolved.to_dict()) == resolved

    unknown = {**resolved.to_dict(), "environment_name": "local-env"}
    with pytest.raises(ValueError, match="resolved_toolbox_environment_unknown_fields:environment_name"):
        ResolvedToolboxEnvironmentInput.from_dict(unknown)

    missing = resolved.to_dict()
    del missing["complete_lock_digest"]
    with pytest.raises(ValueError, match="resolved_toolbox_environment_missing_fields:complete_lock_digest"):
        ResolvedToolboxEnvironmentInput.from_dict(missing)


def test_environment_key_uses_only_frozen_runtime_lock_and_isolation_identity() -> None:
    resolved = _resolved()
    expected = "sha256:893229045bba67ef4c04f21facae7fd4f67a17c292ec65503c8c06b493cc4505"
    assert resolved.environment_key == expected

    changes = (
        {"runtime_version": "3.12.9"},
        {"runtime_artifact_digest": _digest("4")},
        {"python_abi": "cp313"},
        {"platform": "manylinux_2_28_x86_64"},
        {"complete_lock_digest": _digest("5")},
        {"custom_resolved_lock_digest": _digest("6")},
        {"isolation_policy_version": "toolbox-isolation-v2"},
    )
    for change in changes:
        assert replace(resolved, **change).environment_key != expected


def test_template_labels_and_per_function_import_subsets_do_not_change_key() -> None:
    resolved = _resolved()
    assert replace(
        resolved,
        template_id="renamed-logical-template",
        template_digest=_digest("7"),
        resolved_import_roots=("json",),
    ).environment_key == resolved.environment_key


def test_environment_resolver_uses_only_digest_addressed_toolbox_cache(tmp_path: Path) -> None:
    resolved = _resolved()
    spec = HermeticToolboxEnvironmentResolver(tmp_path).environment_spec(resolved)
    expected_root = (tmp_path / "toolbox_environment_cache" / resolved.environment_key.removeprefix("sha256:")).resolve()
    expected_python = expected_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert Path(spec.environment_root) == expected_root
    assert Path(spec.python_executable) == expected_python
    assert "core" not in str(expected_root)


def test_toolbox_interpreter_selection_never_returns_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ToolboxEnvironmentManager(tmp_path)
    spec = ToolboxEnvironmentSpec(
        venv_key="strict",
        venv_path=str(tmp_path / "strict"),
        python_executable=str(tmp_path / "strict" / "Scripts" / "python.exe"),
    )
    monkeypatch.setattr(manager, "ensure_environment", lambda candidate: candidate)
    assert manager.toolbox_runtime_python_executable(spec) == spec.python_executable

    runtime_manager = RuntimeEnvironmentManager(tmp_path)
    monkeypatch.setattr(runtime_manager, "ensure_environment", lambda candidate: candidate)
    monkeypatch.setattr(
        runtime_manager,
        "read_environment_metadata",
        lambda _candidate: {"realization": {"planned_packages": ["demo==1"]}},
    )
    assert runtime_manager.runtime_python_executable(
        spec, bootstrap_python_executable="C:/bootstrap/python.exe"
    ) == "C:/bootstrap/python.exe"


def test_orchestrator_has_no_environment_description_or_bootstrap_selection() -> None:
    source = inspect.getsource(ToolboxSandboxOrchestrator.spawn_assignments)
    assert "toolbox_environment_description" not in source
    assert ".runtime_python_executable(" not in source
    assert ".toolbox_runtime_python_executable(" in source
    assert "fallback_python_executable" not in source
