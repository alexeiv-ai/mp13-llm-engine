from __future__ import annotations

import json
from pathlib import Path

import pytest

from hosting.hosting_configuration import (
    HostingConfigurationError,
    HostingConfigurationRepository,
    parse_hosting_configuration,
)
from mp13_engine.mp13_config_paths import resolve_config_paths


def _resolver(tmp_path: Path):
    config_file = tmp_path / "config" / "mp13_config.json"
    config = {
        "category_dirs": {
            "hosting_root_dir": "@config/hosting-data",
            "packages_root_dir": "@config/packages-data",
            "environments_root_dir": "@config/environments-data",
        }
    }
    return resolve_config_paths(config, cwd=tmp_path, config_path=config_file, project_root=tmp_path)[1]


def _minimal() -> dict:
    return {
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


def test_minimal_configuration_is_immutable_and_preserves_logical_paths(tmp_path: Path) -> None:
    config = parse_hosting_configuration(_minimal(), _resolver(tmp_path))
    assert config.contract == "hosting.configuration.v3"
    assert config.package_management["artifact_root"] == "@packages/artifacts"
    assert config.resolved_paths["artifact_root"].endswith("packages-data\\artifacts") or config.resolved_paths["artifact_root"].endswith("packages-data/artifacts")
    with pytest.raises(TypeError):
        config.control["authentication"] = {}  # type: ignore[index]


def test_full_configuration_and_sanitized_inspection(tmp_path: Path) -> None:
    payload = _minimal()
    payload["control"] = {
        "authentication": {"require_auth": True, "connectivity_mode": "truly_remote", "endpoint_mode": "shared", "ssh_key_ref": "keyring:admin"},
        "roles": {"admin": {"permissions": ["*"]}},
        "session_policy": {"ttl_seconds": 300, "idle_timeout_seconds": 60, "max_sessions_per_key": 2},
        "audit": {"event_limit": 100, "retention_seconds": 3600},
    }
    payload["package_management"]["credentials"] = {"private-index": "SENTINEL_SECRET"}
    payload["package_management"]["sources"] = {
        "private": {"kind": "https", "location": "https://example.invalid/simple?token=SENTINEL_TOKEN", "credential_ref": "private-index", "enabled": True, "priority": 1}
    }
    payload["package_management"]["dependency_policy"] = {"allow_prereleases": False, "allow_sdists": False, "max_dependencies": 25}
    payload["environment_management"]["retention"] = {"unused_seconds": 60, "receipt_seconds": 120}
    payload["environment_management"]["cache"] = {"enabled": True, "max_bytes": 1024}
    config = parse_hosting_configuration(payload, _resolver(tmp_path))
    remote = json.dumps(config.inspect(local_admin=False))
    assert "SENTINEL_SECRET" not in remote
    assert "SENTINEL_TOKEN" not in remote
    assert "resolved_paths" not in remote
    assert "resolved_paths" in config.inspect(local_admin=True)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value.update(contract="hosting.configuration.v2"), "hosting_configuration_unsupported"),
        (lambda value: value["control"].update(password="SENTINEL_SECRET"), "hosting_configuration_key_unknown"),
        (lambda value: value["control"].update(authentication=[]), "hosting_configuration_type_invalid"),
        (lambda value: value["package_management"].update(artifact_root="@unknown/place"), "hosting_configuration_path_invalid"),
        (
            lambda value: value["package_management"]["sources"].update(
                private={"kind": "https", "location": "https://example.invalid", "credential_ref": "missing"}
            ),
            "hosting_configuration_credential_policy_conflict",
        ),
        (
            lambda value: value["control"].update(
                authentication={"require_auth": False, "connectivity_mode": "truly_remote"}
            ),
            "hosting_configuration_policy_conflict",
        ),
    ],
)
def test_rejections_are_stable_and_do_not_leak_values(tmp_path: Path, mutate, code: str) -> None:
    payload = _minimal()
    mutate(payload)
    with pytest.raises(HostingConfigurationError) as captured:
        parse_hosting_configuration(payload, _resolver(tmp_path))
    assert captured.value.code == code
    assert "SENTINEL_SECRET" not in str(captured.value)


def test_repository_is_the_atomic_validation_boundary(tmp_path: Path) -> None:
    path = tmp_path / "config" / "hosting" / "hosting_config.json"
    repository = HostingConfigurationRepository(path, _resolver(tmp_path))
    written = repository.write(_minimal())
    loaded = repository.read()
    assert loaded.revision == written.revision
    assert path.read_text(encoding="utf-8").endswith("\n")
    bad = _minimal()
    bad["package_management"]["verification"]["hash_algorithm"] = "md5"
    with pytest.raises(HostingConfigurationError):
        repository.write(bad)
    assert repository.read().revision == written.revision
