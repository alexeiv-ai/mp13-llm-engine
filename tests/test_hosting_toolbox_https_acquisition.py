from __future__ import annotations

import base64
import hashlib
import json
import time
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
from hosting.toolbox.builtin_resolver import AirgapBuiltinWheelResolver
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.target import detect_current_toolbox_target


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
                "timeout_seconds": 60,
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


def _package_wheel(
    distribution: str,
    version: str,
    *,
    packages: tuple[str, ...],
    requires: tuple[str, ...] = (),
) -> bytes:
    import io

    normalized = distribution.replace("-", "_")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for package in packages:
            archive.writestr(f"{package}/__init__.py", f"NAME = {package!r}\n")
        metadata = [
            "Metadata-Version: 2.1",
            f"Name: {distribution}",
            f"Version: {version}",
            "Requires-Python: >=3.12",
        ]
        metadata.extend(f"Requires-Dist: {item}" for item in requires)
        archive.writestr(
            f"{normalized}-{version}.dist-info/METADATA",
            "\n".join(metadata) + "\n\n",
        )
        archive.writestr(
            f"{normalized}-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{normalized}-{version}.dist-info/RECORD", "")
    return output.getvalue()


def _online_runtime_configuration() -> ToolboxHostProjectConfiguration:
    value = _configuration().to_dict()
    value["builtins"] = [
        {
            "template_id": "core",
            "imports": ["hosting", "mp13_engine"],
            "package_requirements": ["mp13-engine==13.0.0"],
            "sandbox_policy": "compute-only",
            "required": True,
            "prewarm": True,
            "provenance": "online-runtime-test",
        }
    ]
    return ToolboxHostProjectConfiguration.from_dict(value)


def _online_policy() -> dict:
    body = {
        "allowed_template_ids": ["core"],
        "allowed_targets": [detect_current_toolbox_target().name],
        "package_allowlist": ["alpha", "mp13-engine"],
        "package_denylist": [],
        "allow_custom": False,
        "custom_requires_approval": True,
        "online_resolution_allowed": True,
        "allowed_index_origins": ["https://packages.example"],
    }
    return {"revision": identity_digest("test.online.policy.v1", body), **body}


def _signed_project_response(
    private: Ed25519PrivateKey,
    *,
    project: str,
    filename: str,
    wheel: bytes,
) -> _Response:
    raw = json.dumps(
        {
            "meta": {"api-version": "1.0"},
            "name": project,
            "files": [
                {
                    "filename": filename,
                    "url": f"https://artifacts.example/files/{filename}",
                    "hashes": {"sha256": hashlib.sha256(wheel).hexdigest()},
                    "size": len(wheel),
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _Response(
        200,
        raw,
        headers={
            "Content-Length": str(len(raw)),
            "X-MP13-Signing-Key-Id": "packages-key",
            "X-MP13-Signature": _b64(private.sign(raw)),
        },
    )


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


def test_signed_pep503_html_hash_and_size_use_same_verified_cas_boundary(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    private = Ed25519PrivateKey.generate()
    public = {"packages-key": _b64(private.public_key().public_bytes_raw())}
    wheel = _wheel()
    digest = hashlib.sha256(wheel).hexdigest()
    filename = "alpha-1.0-py3-none-any.whl"
    raw = (
        "<!doctype html><html><body>"
        f'<a href="https://artifacts.example/files/{filename}#sha256={digest}" '
        f'data-size="{len(wheel)}">{filename}</a>'
        "</body></html>"
    ).encode()
    session = _Session(
        {
            "https://packages.example/simple/alpha/": [
                _Response(
                    200,
                    raw,
                    headers={
                        "Content-Length": str(len(raw)),
                        "X-MP13-Signing-Key-Id": "packages-key",
                        "X-MP13-Signature": _b64(private.sign(raw)),
                    },
                )
            ],
            f"https://artifacts.example/files/{filename}": [
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

    metadata = acquirer.fetch_project_metadata(
        source_id="approved-index", project_name="alpha"
    )
    imported = acquirer.acquire_wheel(metadata=metadata, artifact=metadata["files"][0])

    assert imported["sha256"] == f"sha256:{digest}"
    assert store.object_path(imported["sha256"]).is_file()


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


def test_normal_daemon_discovers_transitive_https_closure_and_publishes_from_cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _online_runtime_configuration()
    private = Ed25519PrivateKey.generate()
    public = {"packages-key": _b64(private.public_key().public_bytes_raw())}
    runtime = _package_wheel(
        "mp13-engine",
        "13.0.0",
        packages=("hosting", "mp13_engine"),
        requires=("alpha==1.0",),
    )
    alpha = _package_wheel("alpha", "1.0", packages=("alpha",))
    runtime_name = "mp13_engine-13.0.0-py3-none-any.whl"
    alpha_name = "alpha-1.0-py3-none-any.whl"
    session = _Session(
        {
            "https://packages.example/simple/mp13-engine/": [
                _signed_project_response(
                    private,
                    project="mp13-engine",
                    filename=runtime_name,
                    wheel=runtime,
                )
            ],
            "https://packages.example/simple/alpha/": [
                _signed_project_response(
                    private, project="alpha", filename=alpha_name, wheel=alpha
                )
            ],
            f"https://artifacts.example/files/{runtime_name}": [
                _Response(200, runtime, headers={"Content-Length": str(len(runtime))})
            ],
            f"https://artifacts.example/files/{alpha_name}": [
                _Response(200, alpha, headers={"Content-Length": str(len(alpha))})
            ],
        }
    )
    monkeypatch.setattr(
        "hosting.service.toolbox_https_acquisition.requests.Session", lambda: session
    )

    started_at = time.monotonic()
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={},
        toolbox_dependency_policy=_online_policy(),
        toolbox_trust_public_keys=public,
        toolbox_source_credentials={"secret:index": "Bearer top-secret"},
    )
    elapsed = time.monotonic() - started_at
    operation = daemon.svc._toolbox_setup_operation  # noqa: SLF001
    terminal = daemon.svc._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=operation["operation"]["operation_id"], timeout_seconds=60
    )

    assert elapsed < 2
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["code"] == "toolbox_setup_ready"
    assert daemon.svc.hosting_setup_summary()["toolbox_readiness"]["status"] == "ready"
    catalog = daemon.svc._toolbox_template_catalog.read()  # noqa: SLF001
    assert catalog["entries"][0]["template"]["provenance"]["source"] == (
        "signed-https:approved-index"
    )
    assert {
        item["sha256"] for item in catalog["entries"][0]["artifacts"]
    } == {
        f"sha256:{hashlib.sha256(runtime).hexdigest()}",
        f"sha256:{hashlib.sha256(alpha).hexdigest()}",
    }
    assert "top-secret" not in str(terminal)
    assert "top-secret" not in str(daemon.svc.hosting_setup_summary())


def test_normal_daemon_missing_transitive_https_wheel_stays_not_ready_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _online_runtime_configuration()
    private = Ed25519PrivateKey.generate()
    public = {"packages-key": _b64(private.public_key().public_bytes_raw())}
    runtime = _package_wheel(
        "mp13-engine",
        "13.0.0",
        packages=("hosting", "mp13_engine"),
        requires=("alpha==1.0",),
    )
    runtime_name = "mp13_engine-13.0.0-py3-none-any.whl"
    session = _Session(
        {
            "https://packages.example/simple/mp13-engine/": [
                _signed_project_response(
                    private,
                    project="mp13-engine",
                    filename=runtime_name,
                    wheel=runtime,
                )
            ],
            f"https://artifacts.example/files/{runtime_name}": [
                _Response(200, runtime, headers={"Content-Length": str(len(runtime))})
            ],
            "https://packages.example/simple/alpha/": [_Response(404)],
        }
    )
    monkeypatch.setattr(
        "hosting.service.toolbox_https_acquisition.requests.Session", lambda: session
    )
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={},
        toolbox_dependency_policy=_online_policy(),
        toolbox_trust_public_keys=public,
        toolbox_source_credentials={"secret:index": "Bearer top-secret"},
    )
    operation = daemon.svc._toolbox_setup_operation  # noqa: SLF001
    terminal = daemon.svc._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=operation["operation"]["operation_id"], timeout_seconds=20
    )

    assert terminal["lifecycle"] == "terminal_failure"
    assert terminal["result"]["code"] == "https_source_candidate_missing"
    assert daemon.svc._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    assert daemon.svc.hosting_setup_summary()["toolbox_readiness"]["status"] == "degraded"


def test_online_and_airgap_sources_produce_identical_exact_lock_and_artifact_digests(
    tmp_path: Path,
) -> None:
    online = _online_runtime_configuration()
    private = Ed25519PrivateKey.generate()
    public = {"packages-key": _b64(private.public_key().public_bytes_raw())}
    runtime = _package_wheel(
        "mp13-engine",
        "13.0.0",
        packages=("hosting", "mp13_engine"),
        requires=("alpha==1.0",),
    )
    alpha = _package_wheel("alpha", "1.0", packages=("alpha",))
    runtime_name = "mp13_engine-13.0.0-py3-none-any.whl"
    alpha_name = "alpha-1.0-py3-none-any.whl"
    session = _Session(
        {
            "https://packages.example/simple/mp13-engine/": [
                _signed_project_response(
                    private,
                    project="mp13-engine",
                    filename=runtime_name,
                    wheel=runtime,
                )
            ],
            "https://packages.example/simple/alpha/": [
                _signed_project_response(
                    private, project="alpha", filename=alpha_name, wheel=alpha
                )
            ],
            f"https://artifacts.example/files/{runtime_name}": [
                _Response(200, runtime, headers={"Content-Length": str(len(runtime))})
            ],
            f"https://artifacts.example/files/{alpha_name}": [
                _Response(200, alpha, headers={"Content-Length": str(len(alpha))})
            ],
        }
    )
    store = AtomicToolboxArtifactStore(tmp_path / "online-store")
    acquisition = ToolboxHttpsArtifactAcquirer(
        online,
        artifact_store=store,
        trust_public_keys=public,
        source_credentials={"secret:index": "Bearer top-secret"},
        session=session,
    ).discover_and_acquire(("mp13-engine==13.0.0",))
    online_result = AirgapBuiltinWheelResolver(
        online,
        artifact_sources={},
        verified_artifacts=acquisition["verified_artifacts"],
    ).resolve()

    airgap_value = online.to_dict()
    airgap_value["sources"] = [
        {
            **airgap_value["sources"][0],
            "kind": "airgap_store",
            "origin": "airgap://approved-index",
            "credential_ref": None,
        }
    ]
    airgap_value["resolution"] = {
        **airgap_value["resolution"],
        "mode": "air_gapped",
        "allowed_redirect_origins": [],
    }
    airgap = ToolboxHostProjectConfiguration.from_dict(airgap_value)
    wheelhouse = tmp_path / "airgap"
    wheelhouse.mkdir()
    (wheelhouse / runtime_name).write_bytes(runtime)
    (wheelhouse / alpha_name).write_bytes(alpha)
    airgap_result = AirgapBuiltinWheelResolver(
        airgap,
        artifact_sources={"approved-index": wheelhouse},
    ).resolve()

    assert online_result.status == airgap_result.status == "resolved"
    assert online_result.closures[0].lock_digest == airgap_result.closures[0].lock_digest
    assert [
        item.sha256 for item in online_result.closures[0].locked_artifacts
    ] == [item.sha256 for item in airgap_result.closures[0].locked_artifacts]
from hosting.daemon import EngineHostDaemon
