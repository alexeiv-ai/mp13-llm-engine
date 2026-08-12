from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hosting.hosting_configuration import parse_hosting_configuration
from hosting.packages import PackageError, PackagePolicy, PackageSource
from hosting.service.auth import AuthMixin
from hosting.service.host_service import EngineHostService
from mp13_engine.mp13_config_paths import resolve_config_paths


def _service(tmp_path: Path, *, credential: bool = True, max_bytes: int = 1024 * 1024) -> EngineHostService:
    _, resolver = resolve_config_paths(
        {
            "category_dirs": {
                "hosting_root_dir": "@config/host",
                "packages_root_dir": "@config/packages",
                "environments_root_dir": "@config/environments",
            }
        },
        cwd=tmp_path,
        config_path=tmp_path / "mp13_config.json",
        project_root=tmp_path,
    )
    payload = {
        "contract": "hosting.configuration.v3",
        "control": {"authentication": {}, "roles": {}, "session_policy": {}, "audit": {}},
        "package_management": {
            "artifact_root": "@packages/artifacts",
            "lock_root": "@packages/locks",
            "sources": {
                "ingress": {
                    "kind": "upload",
                    "locator": "@packages/artifacts",
                    "credential_ref": "upload-key",
                    "enabled": True,
                    "priority": 100,
                }
            },
            "credentials": {"upload-key": "SENTINEL_CREDENTIAL"} if credential else {},
            "dependency_policy": {
                "policy_id": "default",
                "revision": 1,
                "allowed_source_ids": ["ingress"],
                "allowed_platforms": ["win_amd64", "linux_x86_64"],
                "allowed_runtimes": ["python"],
                "max_artifact_bytes": max_bytes,
                "require_sha256": True,
                "optional_verifier": None,
            },
            "verification": {"hash_algorithm": "sha256"},
        },
        "environment_management": {
            "environment_root": "@environments",
            "scratch_root": "@hosting/scratch",
            "retention": {},
            "cache": {},
        },
    }
    return EngineHostService(hosting_configuration=parse_hosting_configuration(payload, resolver))


def _encoded(content: bytes) -> str:
    return base64.urlsafe_b64encode(content).decode("ascii").rstrip("=")


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def test_neutral_models_are_strict_and_verifier_is_optional() -> None:
    source = PackageSource.from_dict(
        {
            "contract": "hosting.package_source.v1",
            "source_id": "internal",
            "kind": "local",
            "locator": "@packages/artifacts",
            "credential_ref": None,
            "enabled": True,
            "priority": 1,
        }
    )
    policy = PackagePolicy.from_dict(
        {
            "contract": "hosting.package_policy.v1",
            "policy_id": "default",
            "revision": 1,
            "allowed_source_ids": ["internal"],
            "allowed_platforms": ["win_amd64"],
            "allowed_runtimes": ["python"],
            "max_artifact_bytes": 100,
            "require_sha256": True,
            "optional_verifier": None,
        }
    )
    assert source.source_id == "internal"
    assert policy.optional_verifier is None
    with pytest.raises(ValueError, match="fields_invalid"):
        PackageSource.from_dict({**source.to_dict(), "trust_key_ids": []})


def test_server_role_policy_authorizes_new_commands_and_rejects_old_family() -> None:
    worker = AuthMixin._commands_allowed_for_role("worker_user")
    diagnostic = AuthMixin._commands_allowed_for_role("diagnostic_user")
    approver = AuthMixin._commands_allowed_for_role("dependency_approver")
    assert "package-artifact-upload-begin" in worker
    assert "package-artifact-upload-begin" not in diagnostic
    assert "package-lock-create" in approver
    assert not any(command.startswith("toolbox-artifact-upload-") for command in worker)


def test_service_configures_toolbox_materialization_from_generic_roots(tmp_path: Path) -> None:
    service = _service(tmp_path)
    builder = service._hermetic_toolbox_environment_builder
    assert builder is not None
    assert builder.environments_root == Path(service.hosting_configuration.resolved_paths["environment_root"]) / "content"
    assert set(builder.artifact_sources) == {"ingress"}


def test_authentication_methods_with_equal_role_receive_equal_policy(tmp_path: Path) -> None:
    service = _service(tmp_path)
    shared = service.auth_upsert_key(
        key_id="shared-admin", key_secret="secret", role="admin", auth_method="shared_secret"
    )
    public = service.auth_upsert_key(
        key_id="public-admin", role="admin", auth_method="public_key",
        public_key="ssh-ed25519 AAAATest public-admin",
    )
    assert shared["role"] == public["role"] == "admin"
    assert AuthMixin._commands_allowed_for_role(shared["role"]) == AuthMixin._commands_allowed_for_role(public["role"])


def test_daemon_local_import_rehashes_and_creates_generic_lock(tmp_path: Path) -> None:
    service = _service(tmp_path)
    content = b"toolbox-resolved-wheel"
    local = tmp_path / "resolved.whl"
    local.write_bytes(content)
    artifact = service._package_manager.import_verified_file(
        source_id="ingress", path=local, expected_digest=_digest(content),
        actor_id="service:toolbox", request_id="toolbox-plan-1",
    )
    lock = service.package_lock_create(
        lock_id="toolbox-profile-1", revision=1, runtime_kind="python", platform="win_amd64",
        artifacts=[artifact],
        dependencies=[{"name": "demo", "version": "1.0", "artifact_id": artifact["artifact_id"]}],
    )
    assert lock["contract"] == "hosting.package_lock.v1"
    assert lock["artifacts"][0]["artifact_id"] == _digest(content)
    with pytest.raises(PackageError, match="hash_mismatch"):
        service._package_manager.import_verified_file(
            source_id="ingress", path=local, expected_digest=_digest(b"different"),
            actor_id="service:toolbox", request_id="toolbox-plan-2",
        )


def test_upload_is_ordered_bounded_idempotent_and_content_addressed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    content = b"daemon-owned-content"
    begin = service.package_artifact_upload_begin(
        actor_id="key:worker",
        source_id="ingress",
        total_size=len(content),
        expected_digest=_digest(content),
        request_id="request-1",
    )
    repeated = service.package_artifact_upload_begin(
        actor_id="key:worker",
        source_id="ingress",
        total_size=len(content),
        expected_digest=_digest(content),
        request_id="request-1",
    )
    assert repeated["upload_id"] == begin["upload_id"]
    with pytest.raises(PackageError, match="package_upload_chunk_invalid"):
        service.package_artifact_upload_chunk(
            actor_id="key:worker", upload_id=begin["upload_id"], chunk_index=1, offset=0, chunk_base64url=_encoded(content)
        )
    chunk = service.package_artifact_upload_chunk(
        actor_id="key:worker", upload_id=begin["upload_id"], chunk_index=0, offset=0, chunk_base64url=_encoded(content)
    )
    assert chunk["received_bytes"] == len(content)
    assert service.package_artifact_upload_chunk(
        actor_id="key:worker", upload_id=begin["upload_id"], chunk_index=0, offset=0, chunk_base64url=_encoded(content)
    )["received_bytes"] == len(content)
    result = service.package_artifact_upload_commit(
        actor_id="key:worker", upload_id=begin["upload_id"], request_id="commit-1"
    )
    assert result["artifact_id"] == _digest(content)
    assert result["receipt"]["verification"] is None
    assert "SENTINEL_CREDENTIAL" not in json.dumps(result)
    assert not any(Path(service.hosting_configuration.resolved_paths["scratch_root"]).rglob("*.part"))
    audit = (tmp_path / "host" / "audit" / "package_events.jsonl").read_text(encoding="utf-8")
    assert result["artifact_id"] in audit
    assert "SENTINEL_CREDENTIAL" not in audit
    assert str(tmp_path) not in audit


def test_hash_mismatch_never_becomes_resolvable(tmp_path: Path) -> None:
    service = _service(tmp_path)
    content = b"actual"
    begin = service.package_artifact_upload_begin(
        actor_id="key:worker", source_id="ingress", total_size=len(content), expected_digest=_digest(b"expected"), request_id="request-2"
    )
    service.package_artifact_upload_chunk(
        actor_id="key:worker", upload_id=begin["upload_id"], chunk_index=0, offset=0, chunk_base64url=_encoded(content)
    )
    with pytest.raises(PackageError, match="package_artifact_hash_mismatch"):
        service.package_artifact_upload_commit(actor_id="key:worker", upload_id=begin["upload_id"], request_id="commit-2")
    artifact = Path(service.hosting_configuration.resolved_paths["artifact_root"]) / "sha256" / _digest(content).split(":")[1]
    assert not artifact.exists()
    assert service.package_artifact_upload_status(actor_id="key:worker", upload_id=begin["upload_id"])["state"] == "quarantined"


def test_upload_permission_boundary_size_cancel_and_disconnect_recovery(tmp_path: Path) -> None:
    service = _service(tmp_path, max_bytes=4)
    with pytest.raises(PackageError, match="package_upload_bounds_exceeded"):
        service.package_artifact_upload_begin(actor_id="key:worker", source_id="ingress", total_size=5, expected_digest=None, request_id="oversize")
    begin = service.package_artifact_upload_begin(actor_id="key:worker", source_id="ingress", total_size=4, expected_digest=None, request_id="disconnect")
    service.package_artifact_upload_chunk(actor_id="key:worker", upload_id=begin["upload_id"], chunk_index=0, offset=0, chunk_base64url=_encoded(b"ab"))
    restarted = _service(tmp_path, max_bytes=4)
    assert restarted.package_artifact_upload_status(actor_id="key:worker", upload_id=begin["upload_id"])["received_bytes"] == 2
    with pytest.raises(PackageError, match="package_upload_not_found"):
        restarted.package_artifact_upload_status(actor_id="key:other", upload_id=begin["upload_id"])
    assert restarted.package_artifact_upload_cancel(actor_id="key:worker", upload_id=begin["upload_id"], request_id="cancel-1") == {
        "upload_id": begin["upload_id"], "state": "cancelled"
    }
    assert not any(Path(restarted.hosting_configuration.resolved_paths["scratch_root"]).rglob("*.part"))


def test_restart_and_concurrent_commit_are_single_effect(tmp_path: Path) -> None:
    first = _service(tmp_path)
    content = b"restart-safe"
    begin = first.package_artifact_upload_begin(
        actor_id="key:worker", source_id="ingress", total_size=len(content), expected_digest=None, request_id="request-3"
    )
    first.package_artifact_upload_chunk(
        actor_id="key:worker", upload_id=begin["upload_id"], chunk_index=0, offset=0, chunk_base64url=_encoded(content)
    )
    restarted = _service(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _index: restarted.package_artifact_upload_commit(
                    actor_id="key:worker", upload_id=begin["upload_id"], request_id="commit-3"
                ),
                range(2),
            )
        )
    assert results[0] == results[1]
    assert len(list((tmp_path / "packages" / "artifacts" / "sha256").iterdir())) == 1


def test_source_credentials_policy_and_deterministic_offline_lock(tmp_path: Path) -> None:
    missing = _service(tmp_path / "missing")
    missing._package_manager._credentials.clear()  # simulate unavailable keyring provider
    with pytest.raises(PackageError, match="package_credential_unavailable"):
        missing.package_artifact_upload_begin(actor_id="key:worker", source_id="ingress", total_size=1, expected_digest=None, request_id="request-4")

    service = _service(tmp_path / "ready")
    content = b"locked"
    begin = service.package_artifact_upload_begin(actor_id="key:worker", source_id="ingress", total_size=len(content), expected_digest=None, request_id="request-5")
    service.package_artifact_upload_chunk(actor_id="key:worker", upload_id=begin["upload_id"], chunk_index=0, offset=0, chunk_base64url=_encoded(content))
    artifact = service.package_artifact_upload_commit(actor_id="key:worker", upload_id=begin["upload_id"], request_id="commit-5")
    request = {
        "lock_id": "lock-1",
        "revision": 1,
        "runtime_kind": "python",
        "platform": "win_amd64",
        "artifacts": [{"artifact_id": artifact["artifact_id"], "size_bytes": len(content), "source_id": "ingress"}],
        "dependencies": [{"name": "demo", "version": "1.0", "artifact_id": artifact["artifact_id"]}],
    }
    first = service.package_lock_create(**request)
    second = _service(tmp_path / "ready").package_lock_create(**request)
    assert second == first
    assert "credential" not in json.dumps(first).lower()
    with pytest.raises(PackageError, match="package_policy_rejected"):
        service.package_lock_create(**{**request, "platform": "macos_arm64"})
