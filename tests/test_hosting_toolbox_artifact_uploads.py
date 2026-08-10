from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

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
