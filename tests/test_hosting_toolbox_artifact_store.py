from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from packaging.utils import parse_wheel_filename

from hosting.service.toolbox_artifact_store import (
    BUNDLE_CONTRACT,
    SIGNATURE_CONTRACT,
    AtomicToolboxArtifactStore,
    ToolboxArtifactBundleError,
)
from hosting.daemon import EngineHostDaemon
from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.target import detect_current_toolbox_target


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _configuration() -> ToolboxHostProjectConfiguration:
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
                    "provenance": "test-release",
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
                    "maximum_download_bytes": 10_000_000,
                }
            ],
            "resolution": {
                "mode": "air_gapped",
                "timeout_seconds": 60,
                "maximum_bytes": 10_000_000,
                "maximum_artifacts": 10,
                "allowed_redirect_origins": [],
                "wheel_only": True,
            },
            "retention": {
                "artifact_cache_grace_seconds": 60,
                "maximum_cache_bytes": 20_000_000,
                "maximum_cache_artifacts": 100,
                "protected_digests": [],
                "remove_unreferenced_custom_revisions_on_apply": False,
            },
        }
    )


def _wheel(name: str, version: str, *, metadata_name: str | None = None, requires=()) -> bytes:
    import io

    output = io.BytesIO()
    distribution = name.replace("-", "_")
    dist_info = f"{distribution}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {metadata_name or name}",
        f"Version: {version}",
        "Requires-Python: >=3.12,<3.13",
    ]
    metadata.extend(f"Requires-Dist: {item}" for item in requires)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{distribution}/__init__.py", "")
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: mp13-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return output.getvalue()


def _wheel_row(filename: str, content: bytes, *, distribution: str, version: str) -> dict:
    _name, _version, _build, tags = parse_wheel_filename(filename)
    return {
        "distribution": distribution,
        "version": version,
        "filename": filename,
        "size_bytes": len(content),
        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "tags": sorted(str(item) for item in tags),
        "provenance": "test-release",
    }


def _bundle(
    path: Path,
    configuration: ToolboxHostProjectConfiguration,
    private_key: Ed25519PrivateKey,
    *,
    bundle_id: str = "bundle-one",
    include_beta: bool = True,
    bad_signature: bool = False,
    extra_entry: str | None = None,
    target_override: dict | None = None,
    bad_digest: bool = False,
    metadata_mismatch: bool = False,
    compression_bomb: bool = False,
) -> None:
    alpha = _wheel(
        "alpha", "1.0", metadata_name="wrong" if metadata_mismatch else None,
        requires=("beta==2.0",),
    )
    beta = _wheel("beta", "2.0")
    wheels = [_wheel_row("alpha-1.0-py3-none-any.whl", alpha, distribution="alpha", version="1.0")]
    contents = {"alpha-1.0-py3-none-any.whl": alpha}
    if include_beta:
        wheels.append(_wheel_row("beta-2.0-py3-none-any.whl", beta, distribution="beta", version="2.0"))
        contents["beta-2.0-py3-none-any.whl"] = beta
    if bad_digest:
        wheels[0]["sha256"] = "sha256:" + "0" * 64
    target = configuration.target
    manifest = {
        "contract": BUNDLE_CONTRACT,
        "bundle_id": bundle_id,
        "source_id": "offline-release",
        "source_set_revision": configuration.source_set_revision,
        "target": target_override or {
            "name": target.name,
            "python_abi": target.python_abi,
            "platform": target.platform,
        },
        "signing_key_id": "release-key",
        "wheels": wheels,
    }
    manifest_raw = _canonical(manifest)
    signature = private_key.sign(manifest_raw)
    if bad_signature:
        signature = bytes([signature[0] ^ 1]) + signature[1:]
    signature_raw = _canonical(
        {
            "contract": SIGNATURE_CONTRACT,
            "algorithm": "ed25519",
            "key_id": "release-key",
            "signature": _b64(signature),
        }
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_raw)
        archive.writestr("signature.json", signature_raw)
        for filename, content in contents.items():
            archive.writestr(f"wheels/{filename}", content)
        if extra_entry:
            archive.writestr(extra_entry, b"unexpected")
        if compression_bomb:
            archive.writestr("wheels/bomb.bin", b"0" * 1_000_000)


def _keys() -> tuple[Ed25519PrivateKey, dict[str, str]]:
    private = Ed25519PrivateKey.generate()
    return private, {"release-key": _b64(private.public_key().public_bytes_raw())}


def _policy() -> dict:
    target = detect_current_toolbox_target()
    body = {
        "allowed_template_ids": ["core"],
        "allowed_targets": [target.name],
        "package_allowlist": [],
        "package_denylist": [],
        "allow_custom": False,
        "custom_requires_approval": True,
        "online_resolution_allowed": False,
        "allowed_index_origins": [],
    }
    return {"revision": identity_digest("test.bundle.policy.v1", body), **body}


def test_signed_bundle_import_is_atomic_idempotent_and_restart_readable(tmp_path: Path) -> None:
    configuration = _configuration()
    private, public = _keys()
    bundle = tmp_path / "bundle.zip"
    _bundle(bundle, configuration, private)
    store = AtomicToolboxArtifactStore(tmp_path / "store")

    imported = store.import_signed_bundle(bundle, configuration=configuration, trust_public_keys=public)
    repeated = store.import_signed_bundle(bundle, configuration=configuration, trust_public_keys=public)

    assert imported["status"] == "imported"
    assert repeated["status"] == "already_imported"
    recovered = AtomicToolboxArtifactStore(tmp_path / "store").read()
    assert set(recovered["objects"]) == set(imported["artifact_digests"])
    for digest in imported["artifact_digests"]:
        assert AtomicToolboxArtifactStore(tmp_path / "store").object_path(digest).is_file()


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"bad_signature": True}, "artifact_bundle_signature_invalid"),
        ({"extra_entry": "extra.txt"}, "artifact_bundle_archive_invalid"),
        ({"extra_entry": "../escape.whl"}, "artifact_bundle_archive_invalid"),
        ({"compression_bomb": True}, "artifact_bundle_archive_invalid"),
        ({"bad_digest": True}, "artifact_bundle_artifact_invalid"),
        ({"metadata_mismatch": True}, "artifact_bundle_artifact_invalid"),
        ({"include_beta": False}, "artifact_bundle_closure_incomplete"),
    ],
)
def test_invalid_bundle_matrix_leaves_index_unchanged(tmp_path: Path, changes: dict, code: str) -> None:
    configuration = _configuration()
    private, public = _keys()
    store = AtomicToolboxArtifactStore(tmp_path / "store")
    valid = tmp_path / "valid.zip"
    _bundle(valid, configuration, private)
    store.import_signed_bundle(valid, configuration=configuration, trust_public_keys=public)
    before = store.index_path.read_bytes()
    invalid = tmp_path / "invalid.zip"
    _bundle(invalid, configuration, private, bundle_id="bundle-invalid", **changes)

    with pytest.raises(ToolboxArtifactBundleError) as captured:
        store.import_signed_bundle(invalid, configuration=configuration, trust_public_keys=public)

    assert captured.value.code == code
    assert str(tmp_path) not in captured.value.summary
    assert store.index_path.read_bytes() == before


def test_bundle_target_and_symlink_entries_fail_closed(tmp_path: Path) -> None:
    configuration = _configuration()
    private, public = _keys()
    store = AtomicToolboxArtifactStore(tmp_path / "store")
    wrong_target = tmp_path / "wrong-target.zip"
    target = detect_current_toolbox_target()
    _bundle(
        wrong_target,
        configuration,
        private,
        target_override={"name": target.name, "python_abi": target.python_abi, "platform": "wrong"},
    )
    with pytest.raises(ToolboxArtifactBundleError) as captured:
        store.import_signed_bundle(wrong_target, configuration=configuration, trust_public_keys=public)
    assert captured.value.code == "artifact_bundle_target_invalid"

    symlink = tmp_path / "symlink.zip"
    _bundle(symlink, configuration, private)
    rewritten = tmp_path / "rewritten.zip"
    with zipfile.ZipFile(symlink) as source, zipfile.ZipFile(rewritten, "w") as destination:
        for info in source.infolist():
            destination.writestr(info, source.read(info))
        link = zipfile.ZipInfo("wheels/link.whl")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        destination.writestr(link, "alpha-1.0-py3-none-any.whl")
    with pytest.raises(ToolboxArtifactBundleError) as captured:
        store.import_signed_bundle(rewritten, configuration=configuration, trust_public_keys=public)
    assert captured.value.code == "artifact_bundle_archive_invalid"


def test_normal_daemon_discovers_signed_bundles_and_resolves_only_verified_cas_objects(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _bundle(source / "release.zip", configuration, private)

    def construct() -> EngineHostDaemon:
        return EngineHostDaemon(
            pid_file=tmp_path / "daemon.pid",
            engines_state_file=tmp_path / "engines.json",
            control_state_file=tmp_path / "control.json",
            toolbox_host_project_configuration=configuration.to_dict(),
            toolbox_artifact_sources={"offline-release": source},
            toolbox_dependency_policy=_policy(),
            toolbox_trust_public_keys=public,
        )

    daemon = construct()
    assert daemon.svc._toolbox_startup["status"] == "resolved"  # noqa: SLF001
    assert [item["template_id"] for item in daemon.svc._toolbox_startup["closures"]] == ["core"]  # noqa: SLF001
    assert daemon.svc._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    assert str(source) not in str(daemon.svc._toolbox_startup)  # noqa: SLF001
    assert next(iter(public.values())) not in str(daemon.svc.hosting_setup_summary())
    daemon.svc.close()

    restarted = construct()
    assert restarted.svc._toolbox_startup["status"] == "resolved"  # noqa: SLF001
    assert len(restarted.svc._toolbox_artifact_store.read()["bundles"]) == 1  # noqa: SLF001


def test_normal_daemon_invalid_bundle_is_degraded_without_catalog_publication(tmp_path: Path) -> None:
    configuration = _configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _bundle(source / "bad.zip", configuration, private, bad_signature=True)

    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )

    readiness = daemon.svc.hosting_setup_summary()["toolbox_readiness"]
    assert readiness["status"] == "degraded"
    assert readiness["code"] == "artifact_bundle_signature_invalid"
    assert daemon.svc._toolbox_startup["closures"] == []  # noqa: SLF001
    assert daemon.svc._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    assert str(source) not in str(readiness)


@pytest.mark.parametrize(
    "trust_keys",
    [
        {},
        {"release-key": "not-base64"},
        {"release-key": _b64(bytes(32)), "extra-key": _b64(bytes([1]) * 32)},
    ],
)
def test_normal_daemon_rejects_missing_malformed_or_extra_trust_key_bindings(
    tmp_path: Path, trust_keys: dict[str, str]
) -> None:
    configuration = _configuration()
    source = tmp_path / "read-only-source"
    source.mkdir()
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=trust_keys,
    )

    readiness = daemon.svc.hosting_setup_summary()["toolbox_readiness"]
    assert readiness["status"] == "unavailable"
    assert readiness["code"] == "toolbox_configuration_invalid"
    assert not any(value in str(readiness) for value in trust_keys.values())


def test_normal_daemon_never_treats_unsigned_raw_wheels_as_verified_source_content(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    _private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    (source / "alpha-1.0-py3-none-any.whl").write_bytes(
        _wheel("alpha", "1.0", requires=("beta==2.0",))
    )
    (source / "beta-2.0-py3-none-any.whl").write_bytes(_wheel("beta", "2.0"))

    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )

    assert daemon.svc._toolbox_startup["status"] == "not_ready"  # noqa: SLF001
    assert daemon.svc._toolbox_startup["closures"] == []  # noqa: SLF001
    assert daemon.svc._toolbox_startup["diagnostics"][0]["code"] == (  # noqa: SLF001
        "required_template_source_unavailable"
    )
