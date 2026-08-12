from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from hosting.daemon.background import start_daemon_background
from hosting.daemon.local_ipc import EngineHostDaemon
from hosting.hosting_configuration import HostingConfigurationError, load_hosting_configuration
from hosting.service.host_service import EngineHostService


def _write_configuration(tmp_path: Path) -> Path:
    config_file = tmp_path / "mp13_config.json"
    config_file.write_text(
        json.dumps(
            {
                "category_dirs": {
                    "hosting_root_dir": "@config/data-hosting",
                    "packages_root_dir": "@config/data-packages",
                    "environments_root_dir": "@config/data-environments",
                }
            }
        ),
        encoding="utf-8",
    )
    authority = tmp_path / "hosting" / "hosting_config.json"
    authority.parent.mkdir()
    authority.write_text(
        json.dumps(
            {
                "contract": "hosting.configuration.v3",
                "control": {
                    "authentication": {},
                    "roles": {},
                    "session_policy": {},
                    "audit": {},
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
        ),
        encoding="utf-8",
    )
    return config_file


def test_service_starts_with_only_unified_configuration_and_separates_state(tmp_path: Path) -> None:
    config_file = _write_configuration(tmp_path)
    authority = tmp_path / "hosting" / "hosting_config.json"
    original = authority.read_bytes()
    service = EngineHostService(hosting_configuration=load_hosting_configuration(config_file))

    service.auth_upsert_key(key_id="admin", role="admin", key_secret="secret")

    assert service.get_control_config()["daemon_version"] == "3.0.0"
    assert authority.read_bytes() == original
    assert (tmp_path / "data-hosting" / "keyring" / "keys.json").exists()
    assert not (tmp_path / "data-hosting" / "access_control.json").exists()


def test_old_file_alone_fails_with_precise_missing_error(tmp_path: Path) -> None:
    config_file = tmp_path / "mp13_config.json"
    config_file.write_text("{}", encoding="utf-8")
    (tmp_path / "hosting").mkdir()
    (tmp_path / "hosting" / "access_control.json").write_text("{}", encoding="utf-8")

    with pytest.raises(HostingConfigurationError) as captured:
        load_hosting_configuration(config_file)

    assert captured.value.code == "hosting_configuration_missing"


def test_startup_signatures_reject_removed_inputs() -> None:
    service_parameters = inspect.signature(EngineHostService).parameters
    daemon_parameters = inspect.signature(EngineHostDaemon).parameters
    background_parameters = inspect.signature(start_daemon_background).parameters
    for removed in (
        "control_state_file",
        "toolbox_host_project_configuration",
        "toolbox_artifact_sources",
        "toolbox_trust_public_keys",
        "toolbox_source_credentials",
        "toolbox_dependency_policy",
    ):
        assert removed not in service_parameters
        assert removed not in daemon_parameters
        assert removed not in background_parameters
    assert "mp13_config_file" in daemon_parameters
    assert "mp13_config_file" in background_parameters


def test_daemon_validates_configuration_before_initializing_pid_or_listener(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(HostingConfigurationError, match="mp13_configuration_missing"):
        EngineHostDaemon(mp13_config_file=missing, pid_file=tmp_path / "daemon.pid")
    assert not (tmp_path / "daemon.pid").exists()
