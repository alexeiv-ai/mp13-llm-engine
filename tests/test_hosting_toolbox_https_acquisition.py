from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hosting.service.toolbox_artifact_store import AtomicToolboxArtifactStore
from hosting.service.toolbox_https_acquisition import (
    ToolboxHttpsAcquisitionError,
    ToolboxHttpsArtifactAcquirer,
)
from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _configuration(*, maximum_bytes: int = 1_000_000) -> ToolboxHostProjectConfiguration:
    return ToolboxHostProjectConfiguration.from_dict(
        {
            "builtins": [
                {
                    "template_id": "core",
                    "imports": ["alpha"],
                    "package_requirements": ["alpha==1.0"],
                    "sandbox_policy": "compute-only",
                    "required": True,
                    "prewarm": True,
                    "provenance": "online-test",
                }
            ],
            "sources": [
                {
                    "source_id": "approved-index",
                    "kind": "https_index",
                    "origin": "https://packages.example/simple/",
                    "credential_ref": "secret:index",
                    "allowed_package_namespaces": ["*"],
                    "priority": 100,
                    "trust_key_ids": ["packages-key"],
                    "maximum_download_bytes": maximum_bytes,
                }
            ],
            "resolution": {
                "mode": "online",
                "timeout_seconds": 30,
                "maximum_bytes": maximum_bytes,
                "maximum_artifacts": 16,
                "allowed_redirect_origins": ["https://artifacts.example"],
                "wheel_only": True,
            },
            "retention": {
                "artifact_cache_grace_seconds": 60,
                "maximum_cache_bytes": 2_000_000,
                "maximum_cache_artifacts": 32,
                "protected_digests": [],
                "remove_unreferenced_custom_revisions_on_apply": False,
            },
        }
    )


def _wheel() -> bytes:
    import io

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("alpha/__init__.py", "VALUE = 1\n")
        archive.writestr(
            "alpha-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: alpha\nVersion: 1.0\nRequires-Python: >=3.12\n\n",
        )
        archive.writestr(
            "alpha-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
    return output.getvalue()


class _Response:
    def __init__(self, status: int, body: bytes = b"", *, headers: dict | None = None):
        self.status_code = status
        self.body = body
        self.headers = dict(headers or {})
        self.closed = False

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.body

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, responses: dict[str, list[_Response]]):
        self.responses = responses
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses[url].pop(0)


def _fixture(tmp_path: Path, *, bad_signature: bool = False, redirect_origin: str = "artifacts.example"):
    configuration = _configuration()
    private = Ed25519PrivateKey.generate()
    public = {
        "packages-key": _b64(private.public_key().public_bytes_raw()),
    }
    wheel = _wheel()
    digest = hashlib.sha256(wheel).hexdigest()
    filename = "alpha-1.0-py3-none-any.whl"
    metadata = json.dumps(
        {
            "meta": {"api-version": "1.0"},
            "name": "alpha",
            "files": [
                {
                    "filename": filename,
                    "url": f"https://{redirect_origin}/files/{filename}",
                    "hashes": {"sha256": digest},
                    "size": len(wheel),
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    signature = private.sign(metadata)
    if bad_signature:
        signature = bytes([signature[0] ^ 1]) + signature[1:]
    metadata_url = "https://packages.example/simple/alpha/"
    artifact_url = f"https://{redirect_origin}/files/{filename}"
    session = _Session(
        {
            metadata_url: [
                _Response(
                    200,
                    metadata,
                    headers={
                        "Content-Length": str(len(metadata)),
                        "X-MP13-Signing-Key-Id": "packages-key",
                        "X-MP13-Signature": _b64(signature),
                    },
                )
            ],
            artifact_url: [
                _Response(200, wheel, headers={"Content-Length": str(len(wheel))})
            ],
        }
    )
    store = AtomicToolboxArtifactStore(tmp_path / "store")
    acquirer = ToolboxHttpsArtifactAcquirer(
        configuration,
        artifact_store=store,
        trust_public_keys=public,
        source_credentials={"secret:index": "Bearer top-secret"},
        session=session,
    )
    return acquirer, store, session, digest


def test_signed_pep691_wheel_is_verified_and_atomically_indexed_in_shared_cas(
    tmp_path: Path,
) -> None:
    acquirer, store, session, digest = _fixture(tmp_path)

    metadata = acquirer.fetch_project_metadata(
        source_id="approved-index", project_name="Alpha"
    )
    imported = acquirer.acquire_wheel(metadata=metadata, artifact=metadata["files"][0])

    assert imported["sha256"] == f"sha256:{digest}"
    assert store.object_path(imported["sha256"]).read_bytes() == _wheel()
    assert list(store.source_artifacts("approved-index")) == [
        "alpha-1.0-py3-none-any.whl"
    ]
    assert len(store.read()["https_manifests"]) == 1
    assert all(call["headers"]["Authorization"] == "Bearer top-secret" for call in session.calls)
    assert "top-secret" not in str(metadata)


def test_https_metadata_requires_configured_ed25519_signature(tmp_path: Path) -> None:
    acquirer, store, _session, _digest = _fixture(tmp_path, bad_signature=True)

    with pytest.raises(ToolboxHttpsAcquisitionError) as captured:
        acquirer.fetch_project_metadata(source_id="approved-index", project_name="alpha")

    assert captured.value.code == "https_source_signature_invalid"
    assert store.read()["objects"] == {}


def test_https_artifact_origin_outside_redirect_allowlist_is_denied(tmp_path: Path) -> None:
    acquirer, store, _session, _digest = _fixture(
        tmp_path, redirect_origin="evil.example"
    )

    with pytest.raises(ToolboxHttpsAcquisitionError) as captured:
        acquirer.fetch_project_metadata(source_id="approved-index", project_name="alpha")

    assert captured.value.code == "https_source_metadata_invalid"
    assert store.read()["objects"] == {}


def test_https_credentials_are_exact_and_never_part_of_public_metadata(tmp_path: Path) -> None:
    configuration = _configuration()
    private = Ed25519PrivateKey.generate()
    public = {"packages-key": _b64(private.public_key().public_bytes_raw())}

    with pytest.raises(ToolboxHttpsAcquisitionError) as captured:
        ToolboxHttpsArtifactAcquirer(
            configuration,
            artifact_store=AtomicToolboxArtifactStore(tmp_path / "store"),
            trust_public_keys=public,
            source_credentials={"unexpected": "secret"},
            session=_Session({}),
        )

    assert captured.value.code == "https_source_credentials_invalid"


def test_https_wheel_digest_mismatch_never_changes_cas_index(tmp_path: Path) -> None:
    acquirer, store, session, _digest = _fixture(tmp_path)
    metadata = acquirer.fetch_project_metadata(
        source_id="approved-index", project_name="alpha"
    )
    artifact_url = metadata["files"][0]["url"]
    response = session.responses[artifact_url][0]
    response.body = bytes([response.body[0] ^ 1]) + response.body[1:]

    with pytest.raises(ToolboxHttpsAcquisitionError) as captured:
        acquirer.acquire_wheel(metadata=metadata, artifact=metadata["files"][0])

    assert captured.value.code == "https_source_artifact_invalid"
    assert store.read()["objects"] == {}
    assert store.read()["https_manifests"] == {}


def test_https_metadata_redirect_to_unapproved_origin_is_denied_before_credentials_move(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    private = Ed25519PrivateKey.generate()
    public = {"packages-key": _b64(private.public_key().public_bytes_raw())}
    original = "https://packages.example/simple/alpha/"
    session = _Session(
        {
            original: [
                _Response(302, headers={"Location": "https://evil.example/simple/alpha/"})
            ]
        }
    )
    acquirer = ToolboxHttpsArtifactAcquirer(
        configuration,
        artifact_store=AtomicToolboxArtifactStore(tmp_path / "store"),
        trust_public_keys=public,
        source_credentials={"secret:index": "Bearer top-secret"},
        session=session,
    )

    with pytest.raises(ToolboxHttpsAcquisitionError) as captured:
        acquirer.fetch_project_metadata(source_id="approved-index", project_name="alpha")

    assert captured.value.code == "https_source_redirect_denied"
    assert [call["url"] for call in session.calls] == [original]
