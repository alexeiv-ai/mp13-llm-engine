from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from packaging.utils import parse_wheel_filename

from hosting.operation_contract import (
    HostedExecutionKind,
    HostedOperationSelector,
    hosted_execution_fingerprint,
)
from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_artifact_store import BUNDLE_CONTRACT, SIGNATURE_CONTRACT
from hosting.service.toolbox_artifact_uploads import (
    MAX_CHUNK_BYTES,
    AtomicToolboxArtifactUploadRepository,
    ToolboxArtifactUploadError,
)
from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration


def _configuration() -> ToolboxHostProjectConfiguration:
    return ToolboxHostProjectConfiguration.from_dict(
        {
            "builtins": [
                {
                    "template_id": "core",
                    "imports": ["hosting"],
                    "package_requirements": ["mp13-engine==13.0.0"],
                    "sandbox_policy": "compute-only",
                    "required": True,
                    "prewarm": True,
                    "provenance": "upload-test",
                }
            ],
            "sources": [
                {
                    "source_id": "offline-release",
                    "kind": "airgap_store",
                    "origin": "airgap://offline-release",
                    "credential_ref": None,
                    "allowed_package_namespaces": ["*"],
                    "priority": 100,
                    "trust_key_ids": ["release-key"],
                    "maximum_download_bytes": 2 * 1024 * 1024,
                }
            ],
            "resolution": {
                "mode": "air_gapped",
                "timeout_seconds": 60,
                "maximum_bytes": 2 * 1024 * 1024,
                "maximum_artifacts": 32,
                "allowed_redirect_origins": [],
                "wheel_only": True,
            },
            "retention": {
                "artifact_cache_grace_seconds": 60,
                "maximum_cache_bytes": 4 * 1024 * 1024,
                "maximum_cache_artifacts": 64,
                "protected_digests": [],
                "remove_unreferenced_custom_revisions_on_apply": False,
            },
        }
    )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _signed_bundle(
    path: Path,
    configuration: ToolboxHostProjectConfiguration,
    private: Ed25519PrivateKey,
) -> bytes:
    import io

    wheel_output = io.BytesIO()
    filename = "mp13_engine-13.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_output, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr("hosting/__init__.py", "")
        wheel.writestr("mp13_engine/__init__.py", "")
        wheel.writestr(
            "mp13_engine-13.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: mp13-engine\nVersion: 13.0.0\n"
            "Requires-Python: >=3.12,<3.13\n\n",
        )
        wheel.writestr(
            "mp13_engine-13.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr("mp13_engine-13.0.0.dist-info/RECORD", "")
    wheel_bytes = wheel_output.getvalue()
    _name, _version, _build, tags = parse_wheel_filename(filename)
    manifest = {
        "contract": BUNDLE_CONTRACT,
        "bundle_id": "uploaded-runtime",
        "source_id": "offline-release",
        "source_set_revision": configuration.source_set_revision,
        "target": {
            "name": configuration.target.name,
            "python_abi": configuration.target.python_abi,
            "platform": configuration.target.platform,
        },
        "signing_key_id": "release-key",
        "wheels": [
            {
                "distribution": "mp13-engine",
                "version": "13.0.0",
                "filename": filename,
                "size_bytes": len(wheel_bytes),
                "sha256": _digest(wheel_bytes),
                "tags": sorted(str(item) for item in tags),
                "provenance": "upload-test",
            }
        ],
    }
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    signature = {
        "contract": SIGNATURE_CONTRACT,
        "algorithm": "ed25519",
        "key_id": "release-key",
        "signature": _b64(private.sign(raw)),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", raw)
        archive.writestr(
            "signature.json",
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode(),
        )
        archive.writestr(f"wheels/{filename}", wheel_bytes)
    return path.read_bytes()


def _service(tmp_path: Path, *, public_key: str) -> EngineHostService:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    return EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=_configuration().to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_trust_public_keys={"release-key": public_key},
    )


def test_begin_and_ordered_chunks_are_process_safe_idempotent_and_restartable(
    tmp_path: Path,
) -> None:
    content = b"first-second"
    repository = AtomicToolboxArtifactUploadRepository(
        tmp_path / "uploads", configuration=_configuration()
    )
    started = repository.begin(
        owner_actor_id="admin:one",
        request_id="upload-request-one",
        source_id="offline-release",
        total_size=len(content),
        archive_sha256=_digest(content),
    )
    repeated = repository.begin(
        owner_actor_id="admin:one",
        request_id="upload-request-one",
        source_id="offline-release",
        total_size=len(content),
        archive_sha256=_digest(content),
    )
    assert repeated["upload_id"] == started["upload_id"]
    assert "owner_actor_id" not in started
    assert "fingerprint" not in started

    first = repository.append_chunk(
        owner_actor_id="admin:one",
        upload_id=started["upload_id"],
        chunk_index=0,
        offset=0,
        chunk_base64url=_b64(b"first-"),
    )
    duplicate = repository.append_chunk(
        owner_actor_id="admin:one",
        upload_id=started["upload_id"],
        chunk_index=0,
        offset=0,
        chunk_base64url=_b64(b"first-"),
    )
    assert duplicate["received_size"] == first["received_size"] == 6

    restarted = AtomicToolboxArtifactUploadRepository(
        tmp_path / "uploads", configuration=_configuration()
    )
    complete = restarted.append_chunk(
        owner_actor_id="admin:one",
        upload_id=started["upload_id"],
        chunk_index=1,
        offset=6,
        chunk_base64url=_b64(b"second"),
    )
    assert complete["received_size"] == complete["total_size"] == len(content)
    assert restarted._stage_path(started["upload_id"]).read_bytes() == content  # noqa: SLF001


def test_begin_conflict_and_chunk_order_or_content_conflicts_are_bounded(tmp_path: Path) -> None:
    repository = AtomicToolboxArtifactUploadRepository(
        tmp_path / "uploads", configuration=_configuration()
    )
    started = repository.begin(
        owner_actor_id="admin:one",
        request_id="same-request",
        source_id="offline-release",
        total_size=6,
        archive_sha256=_digest(b"abcdef"),
    )
    with pytest.raises(ToolboxArtifactUploadError) as begin_conflict:
        repository.begin(
            owner_actor_id="admin:one",
            request_id="same-request",
            source_id="offline-release",
            total_size=6,
            archive_sha256=_digest(b"ABCDEF"),
        )
    assert begin_conflict.value.code == "artifact_upload_conflict"

    with pytest.raises(ToolboxArtifactUploadError) as out_of_order:
        repository.append_chunk(
            owner_actor_id="admin:one",
            upload_id=started["upload_id"],
            chunk_index=1,
            offset=0,
            chunk_base64url=_b64(b"abc"),
        )
    assert out_of_order.value.code == "artifact_upload_chunk_invalid"
    repository.append_chunk(
        owner_actor_id="admin:one",
        upload_id=started["upload_id"],
        chunk_index=0,
        offset=0,
        chunk_base64url=_b64(b"abc"),
    )
    with pytest.raises(ToolboxArtifactUploadError) as chunk_conflict:
        repository.append_chunk(
            owner_actor_id="admin:one",
            upload_id=started["upload_id"],
            chunk_index=0,
            offset=0,
            chunk_base64url=_b64(b"xyz"),
        )
    assert chunk_conflict.value.code == "artifact_upload_conflict"


def test_chunk_bound_is_checked_before_base64_decode_allocation(tmp_path: Path) -> None:
    repository = AtomicToolboxArtifactUploadRepository(
        tmp_path / "uploads", configuration=_configuration()
    )
    started = repository.begin(
        owner_actor_id="admin:one",
        request_id="bounded",
        source_id="offline-release",
        total_size=2 * 1024 * 1024,
        archive_sha256=_digest(b"declared-only"),
    )
    oversized = "A" * (((MAX_CHUNK_BYTES * 4 + 2) // 3) + 1)

    with pytest.raises(ToolboxArtifactUploadError) as captured:
        repository.append_chunk(
            owner_actor_id="admin:one",
            upload_id=started["upload_id"],
            chunk_index=0,
            offset=0,
            chunk_base64url=oversized,
        )

    assert captured.value.code == "artifact_upload_chunk_invalid"
    assert repository.status(
        owner_actor_id="admin:one", upload_id=started["upload_id"]
    )["received_size"] == 0


def test_expiry_and_cancel_remove_only_untrusted_staged_bytes(tmp_path: Path) -> None:
    now = [1000.0]
    repository = AtomicToolboxArtifactUploadRepository(
        tmp_path / "uploads", configuration=_configuration(), clock=lambda: now[0]
    )
    expiring = repository.begin(
        owner_actor_id="admin:one",
        request_id="expires",
        source_id="offline-release",
        total_size=3,
        archive_sha256=_digest(b"abc"),
    )
    repository.append_chunk(
        owner_actor_id="admin:one",
        upload_id=expiring["upload_id"],
        chunk_index=0,
        offset=0,
        chunk_base64url=_b64(b"abc"),
    )
    now[0] += 901
    expired = repository.status(
        owner_actor_id="admin:one", upload_id=expiring["upload_id"]
    )
    assert expired["state"] == "expired"
    assert not repository._stage_path(expiring["upload_id"]).exists()  # noqa: SLF001

    current = repository.begin(
        owner_actor_id="admin:one",
        request_id="cancel",
        source_id="offline-release",
        total_size=3,
        archive_sha256=_digest(b"xyz"),
    )
    canceled = repository.cancel(
        owner_actor_id="admin:one", upload_id=current["upload_id"]
    )
    repeated = repository.cancel(
        owner_actor_id="admin:one", upload_id=current["upload_id"]
    )
    assert canceled["state"] == repeated["state"] == "canceled"
    assert not repository._stage_path(current["upload_id"]).exists()  # noqa: SLF001


def test_staging_never_creates_or_changes_verified_artifact_store(tmp_path: Path) -> None:
    repository = AtomicToolboxArtifactUploadRepository(
        tmp_path / "uploads", configuration=_configuration()
    )
    started = repository.begin(
        owner_actor_id="admin:one",
        request_id="untrusted-only",
        source_id="offline-release",
        total_size=3,
        archive_sha256=_digest(b"abc"),
    )
    repository.append_chunk(
        owner_actor_id="admin:one",
        upload_id=started["upload_id"],
        chunk_index=0,
        offset=0,
        chunk_base64url=_b64(b"abc"),
    )

    assert not (tmp_path / "toolbox_artifact_store" / "index.json").exists()
    with pytest.raises(ToolboxArtifactUploadError) as hidden:
        repository.status(
            owner_actor_id="admin:another", upload_id=started["upload_id"]
        )
    assert hidden.value.code == "artifact_upload_not_found"


def test_durable_commit_verifies_signed_bundle_and_is_idempotent_on_same_operation(
    tmp_path: Path,
) -> None:
    private = Ed25519PrivateKey.generate()
    public = _b64(private.public_key().public_bytes_raw())
    bundle = _signed_bundle(tmp_path / "upload.zip", _configuration(), private)
    service = _service(tmp_path, public_key=public)
    started = service.toolbox_artifact_upload_begin(
        source_id="offline-release",
        total_size=len(bundle),
        archive_sha256=_digest(bundle),
        request_id="begin-one",
        owner_actor_id="admin:one",
    )
    service.toolbox_artifact_upload_chunk(
        upload_id=started["upload_id"],
        chunk_index=0,
        offset=0,
        chunk_base64url=_b64(bundle),
        owner_actor_id="admin:one",
    )

    committed = service.toolbox_artifact_upload_commit(
        upload_id=started["upload_id"],
        request_id="commit-one",
        owner_actor_id="admin:one",
    )
    duplicate = service.toolbox_artifact_upload_commit(
        upload_id=started["upload_id"],
        request_id="commit-one",
        owner_actor_id="admin:one",
    )
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=committed["operation"]["operation_id"], timeout_seconds=20
    )

    assert duplicate["operation"]["operation_id"] == committed["operation"]["operation_id"]
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["code"] == "artifact_upload_committed"
    assert terminal["progress"]["phase"] == "cleanup"
    assert service._toolbox_artifact_store.read()["bundles"]["uploaded-runtime"]  # noqa: SLF001
    upload = service.toolbox_artifact_upload_status(
        upload_id=started["upload_id"], owner_actor_id="admin:one"
    )
    assert upload["state"] == "committed"
    assert not service._toolbox_artifact_upload_repository._stage_path(  # noqa: SLF001
        started["upload_id"]
    ).exists()
    with pytest.raises(ToolboxArtifactUploadError) as conflict:
        service.toolbox_artifact_upload_commit(
            upload_id=started["upload_id"],
            request_id="different-commit",
            owner_actor_id="admin:one",
        )
    assert conflict.value.code == "artifact_upload_conflict"


def test_commit_digest_failure_is_terminal_bounded_and_does_not_publish_cas(
    tmp_path: Path,
) -> None:
    private = Ed25519PrivateKey.generate()
    public = _b64(private.public_key().public_bytes_raw())
    bundle = _signed_bundle(tmp_path / "upload.zip", _configuration(), private)
    service = _service(tmp_path, public_key=public)
    started = service.toolbox_artifact_upload_begin(
        source_id="offline-release",
        total_size=len(bundle),
        archive_sha256="sha256:" + "0" * 64,
        request_id="begin-bad-digest",
        owner_actor_id="admin:one",
    )
    service.toolbox_artifact_upload_chunk(
        upload_id=started["upload_id"],
        chunk_index=0,
        offset=0,
        chunk_base64url=_b64(bundle),
        owner_actor_id="admin:one",
    )
    committing = service.toolbox_artifact_upload_commit(
        upload_id=started["upload_id"],
        request_id="commit-bad-digest",
        owner_actor_id="admin:one",
    )
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=committing["operation"]["operation_id"], timeout_seconds=20
    )

    assert terminal["lifecycle"] == "terminal_failure"
    assert terminal["result"]["code"] == "artifact_upload_chunk_invalid"
    assert service._toolbox_artifact_store.read()["bundles"] == {}  # noqa: SLF001
    assert service.toolbox_artifact_upload_status(
        upload_id=started["upload_id"], owner_actor_id="admin:one"
    )["state"] == "failed"
    assert not service._toolbox_artifact_upload_repository._stage_path(  # noqa: SLF001
        started["upload_id"]
    ).exists()


def test_restart_reconciles_committed_checkpoint_on_same_operation(tmp_path: Path) -> None:
    private = Ed25519PrivateKey.generate()
    public = _b64(private.public_key().public_bytes_raw())
    configuration = _configuration()
    bundle = _signed_bundle(tmp_path / "upload.zip", configuration, private)
    first = _service(tmp_path, public_key=public)
    started = first.toolbox_artifact_upload_begin(
        source_id="offline-release",
        total_size=len(bundle),
        archive_sha256=_digest(bundle),
        request_id="begin-restart",
        owner_actor_id="admin:one",
    )
    first.toolbox_artifact_upload_chunk(
        upload_id=started["upload_id"],
        chunk_index=0,
        offset=0,
        chunk_base64url=_b64(bundle),
        owner_actor_id="admin:one",
    )
    upload = first._toolbox_artifact_upload_repository.reserve_commit(  # noqa: SLF001
        owner_actor_id="admin:one",
        upload_id=started["upload_id"],
        request_id="commit-restart",
    )
    fingerprint = hosted_execution_fingerprint(
        {
            "execution_kind": HostedExecutionKind.TOOLBOX_ARTIFACT_IMPORT.value,
            "upload_id": upload["upload_id"],
            "source_id": upload["source_id"],
            "config_revision": upload["config_revision"],
            "source_set_revision": upload["source_set_revision"],
            "target": upload["target"],
            "archive_sha256": upload["archive_sha256"],
            "total_size": upload["total_size"],
        }
    )
    prepared = first._hosted_operations.prepare(  # noqa: SLF001
        owner_actor_id="admin:one",
        execution_kind=HostedExecutionKind.TOOLBOX_ARTIFACT_IMPORT,
        selector=HostedOperationSelector(kind="upload_id", id=upload["upload_id"]),
        namespace=f"toolbox_artifact_import:{upload['upload_id']}",
        request_id="commit-restart",
        fingerprint=fingerprint,
        metadata={
            "source_id": upload["source_id"],
            "config_revision": upload["config_revision"],
            "source_set_revision": upload["source_set_revision"],
            "target": upload["target"],
        },
    )
    operation_id = prepared["status"]["operation"]["operation_id"]
    first._toolbox_artifact_upload_repository.bind_commit_operation(  # noqa: SLF001
        owner_actor_id="admin:one",
        upload_id=upload["upload_id"],
        request_id="commit-restart",
        operation_id=operation_id,
    )
    first._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)  # noqa: SLF001
    imported = first._toolbox_artifact_store.import_signed_bundle(  # noqa: SLF001
        first._toolbox_artifact_upload_repository._stage_path(upload["upload_id"]),  # noqa: SLF001
        configuration=configuration,
        trust_public_keys={"release-key": public},
        expected_source_id="offline-release",
    )
    result = {
        "status": "ok",
        "code": "artifact_upload_committed",
        "upload_id": upload["upload_id"],
        "bundle_id": imported["bundle_id"],
        "manifest_digest": imported["manifest_digest"],
        "artifact_digests": imported["artifact_digests"],
    }
    first._toolbox_artifact_upload_repository.finish_commit(  # noqa: SLF001
        owner_actor_id="admin:one",
        upload_id=upload["upload_id"],
        operation_id=operation_id,
        success=True,
        result=result,
    )

    repository_key = str((first.hosting_root / "state" / "hosted_operations.json").resolve())
    EngineHostService._operation_repositories.pop(repository_key, None)  # noqa: SLF001
    restarted = _service(tmp_path, public_key=public)
    reconciled = restarted.toolbox_artifact_upload_commit(
        upload_id=upload["upload_id"],
        request_id="commit-restart",
        owner_actor_id="admin:one",
    )

    assert reconciled["operation"]["operation_id"] == operation_id
    assert reconciled["lifecycle"] == "terminal_success"
    assert reconciled["result"] == result


def test_artifact_upload_commands_are_admin_only() -> None:
    commands = {
        "toolbox-artifact-upload-begin",
        "toolbox-artifact-upload-chunk",
        "toolbox-artifact-upload-status",
        "toolbox-artifact-upload-cancel",
        "toolbox-artifact-upload-commit",
    }
    assert commands <= EngineHostService._commands_allowed_for_role("admin")  # noqa: SLF001
    for role in (
        "config_editor",
        "worker_user",
        "diagnostic_user",
        "model_user",
        "model_user_with_model_control",
    ):
        assert not commands & EngineHostService._commands_allowed_for_role(role)  # noqa: SLF001
