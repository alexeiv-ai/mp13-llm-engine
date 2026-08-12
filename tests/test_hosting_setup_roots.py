from __future__ import annotations

import json
from pathlib import Path

import pytest

from hosting.hosting_setup_api import (
    _atomic_write_json,
    _hosting_configuration_file,
    _journal_file,
    _recover_root_update,
    apply_local_hosting_setup,
    plan_local_hosting_setup,
    reset_local_hosting_setup,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _base_config() -> dict:
    return {
        "category_dirs": {
            "hosting_root_dir": "@config/old-hosting",
            "packages_root_dir": "@config/old-packages",
            "environments_root_dir": "@config/old-environments",
        },
        "unrelated": {"preserved": True},
    }


def _hosting_config(event_limit: int = 100) -> dict:
    return {
        "contract": "hosting.configuration.v3",
        "control": {
            "authentication": {},
            "roles": {},
            "session_policy": {},
            "audit": {"event_limit": event_limit},
        },
        "package_management": {
            "artifact_root": "@packages/artifacts",
            "lock_root": "@packages/locks",
            "sources": {},
            "credentials": {},
            "dependency_policy": {},
            "verification": {"hash_algorithm": "sha256"},
        },
        "environment_management": {
            "environment_root": "@environments",
            "scratch_root": "@hosting/scratch",
            "retention": {},
            "cache": {},
        },
    }


def _request(config_file: Path) -> dict:
    return {
        "contract": "hosting.setup.v1",
        "mp13_config_file": config_file,
        "roots": {
            "hosting_root_dir": "@config/new-hosting",
            "packages_root_dir": "@config/new-packages",
            "environments_root_dir": "@config/new-environments",
        },
    }


def test_root_plan_is_no_write_and_apply_preserves_unrelated_configuration(tmp_path: Path) -> None:
    config_file = tmp_path / "config" / "mp13_config.json"
    hosting_file = _hosting_configuration_file(config_file)
    _write(config_file, _base_config())
    _write(hosting_file, _hosting_config())

    plan = plan_local_hosting_setup(_request(config_file))

    assert plan["status"] == "planned"
    assert plan["would_write"] is False
    assert plan["ok"] is True
    assert json.loads(config_file.read_text(encoding="utf-8")) == _base_config()

    applied = apply_local_hosting_setup(
        {
            **_request(config_file),
            "confirm": True,
            "expected_config_revision": plan["config_revision"],
            "expected_hosting_revision": plan["hosting_revision"],
        }
    )

    stored = json.loads(config_file.read_text(encoding="utf-8"))
    assert applied["status"] == "applied"
    assert applied["journal_state"] == "committed"
    assert stored["unrelated"] == {"preserved": True}
    assert stored["category_dirs"] == _request(config_file)["roots"]
    assert json.loads(hosting_file.read_text(encoding="utf-8"))["control"]["audit"] == {"event_limit": 100}
    assert not _journal_file(config_file).exists()
    for value in applied["resolved_roots"].values():
        assert Path(value).is_dir()


def test_root_apply_requires_local_confirmation_and_matching_revisions(tmp_path: Path) -> None:
    config_file = tmp_path / "mp13_config.json"
    _write(config_file, _base_config())
    request = _request(config_file)

    with pytest.raises(PermissionError, match="confirm=True"):
        apply_local_hosting_setup(request)
    with pytest.raises(PermissionError, match="host-local"):
        plan_local_hosting_setup({**request, "host_local": False})
    with pytest.raises(RuntimeError, match="config_revision_conflict"):
        apply_local_hosting_setup({**request, "confirm": True, "expected_config_revision": "sha256:" + "0" * 64})


def test_root_plan_rejects_nonempty_destination_without_override(tmp_path: Path) -> None:
    config_file = tmp_path / "mp13_config.json"
    _write(config_file, _base_config())
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    request = _request(config_file)
    request["roots"] = {**request["roots"], "packages_root_dir": "@config/occupied"}

    plan = plan_local_hosting_setup(request)

    package_check = next(row for row in plan["preflight"] if row["root"] == "packages_root_dir")
    assert package_check["nonempty"] is True
    assert package_check["ok"] is False
    with pytest.raises(PermissionError, match="preflight_failed"):
        apply_local_hosting_setup({**request, "confirm": True})


def test_root_reset_uses_shared_defaults_without_deleting_data(tmp_path: Path) -> None:
    config_file = tmp_path / "mp13_config.json"
    _write(config_file, _base_config())

    result = reset_local_hosting_setup(
        {"operation": "reset", "mp13_config_file": config_file, "confirm": True}
    )

    stored = json.loads(config_file.read_text(encoding="utf-8"))
    assert result["status"] == "reset"
    assert result["packages_environments_preserved"] is True
    assert stored["category_dirs"]["hosting_root_dir"] == "@home/.mp13-llm/hosting"


@pytest.mark.parametrize(
    "phase,expected,current_is_target",
    [
        ("prepared", "discarded_prepared", False),
        ("top_level_written", "rolled_back_top_level", False),
        ("hosting_written", "completed_target", True),
        ("committed", "completed_target", True),
    ],
)
def test_root_update_recovery_is_idempotent(
    tmp_path: Path, phase: str, expected: str, current_is_target: bool
) -> None:
    config_file = tmp_path / "mp13_config.json"
    hosting_file = _hosting_configuration_file(config_file)
    previous_top = _base_config()
    previous_hosting = _hosting_config(100)
    target_top = {**previous_top, "revision": "new"}
    target_hosting = _hosting_config(200)
    _write(config_file, target_top if phase != "prepared" else previous_top)
    _write(hosting_file, target_hosting if phase in {"hosting_written", "committed"} else previous_hosting)
    _atomic_write_json(
        _journal_file(config_file),
        {
            "contract": "hosting.setup.journal.v1",
            "phase": phase,
            "top_path": str(config_file),
            "hosting_path": str(hosting_file),
            "previous_top": previous_top,
            "previous_hosting": previous_hosting,
            "target_top": target_top,
            "target_hosting": target_hosting,
            "write_hosting": True,
        },
    )

    assert _recover_root_update(config_file) == expected
    assert json.loads(config_file.read_text(encoding="utf-8")) == (target_top if current_is_target else previous_top)
    assert _recover_root_update(config_file) is None
