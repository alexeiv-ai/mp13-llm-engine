from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import time
import zipfile
from functools import lru_cache
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
from hosting.service.toolbox_catalog import AtomicJsonToolboxTemplateCatalog
from hosting.service.host_service import EngineHostService
from hosting.operation_contract import (
    HostedExecutionKind,
    HostedOperationSelector,
    hosted_execution_fingerprint,
)
from hosting.daemon import EngineHostDaemon
from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.target import detect_current_toolbox_target


def _wait_daemon_setup(daemon: EngineHostDaemon, *, timeout_seconds: float = 60) -> dict:
    operation = daemon.svc._toolbox_setup_operation  # noqa: SLF001
    return daemon.svc._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=operation["operation"]["operation_id"],
        timeout_seconds=timeout_seconds,
    )


def _manual_resolved_service(
    tmp_path: Path,
    *,
    configuration: ToolboxHostProjectConfiguration,
    source: Path,
    public: dict[str, str],
) -> EngineHostService:
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )
    service._toolbox_startup = service._resolve_configured_toolbox_startup(  # noqa: SLF001
        progress=lambda *_args: None
    )
    return service


def _prepare_unstarted_setup(service: EngineHostService) -> dict:
    configuration = service._toolbox_host_project_config  # noqa: SLF001
    return service._hosted_operations.prepare(  # noqa: SLF001
        owner_actor_id="system:toolbox-setup",
        execution_kind=HostedExecutionKind.TOOLBOX_SETUP,
        selector=HostedOperationSelector(kind="host_scope", id="toolbox-host"),
        namespace="toolbox_setup:toolbox-host",
        request_id=(
            f"toolbox-setup-{configuration.config_revision.removeprefix('sha256:')}"
        ),
        fingerprint=hosted_execution_fingerprint(
            {
                "execution_kind": "toolbox_setup",
                "host_scope": "toolbox-host",
                "config_revision": configuration.config_revision,
                "source_set_revision": configuration.source_set_revision,
                "target": configuration.target.name,
            }
        ),
        metadata={
            "config_revision": configuration.config_revision,
            "source_set_revision": configuration.source_set_revision,
            "target": configuration.target.name,
        },
    )


def _drop_operation_repository(service: EngineHostService) -> None:
    path = str((service.hosting_root / "state" / "hosted_operations.json").resolve())
    EngineHostService._operation_repositories.pop(path, None)  # noqa: SLF001


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


@lru_cache(maxsize=None)
def _wheel(
    name: str,
    version: str,
    *,
    metadata_name: str | None = None,
    requires=(),
    packages: tuple[str, ...] | None = None,
) -> bytes:
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
        for package in packages or (distribution,):
            archive.writestr(f"{package}/__init__.py", "")
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


def _runtime_configuration(*, two_templates: bool = False) -> ToolboxHostProjectConfiguration:
    payload = _configuration().to_dict()
    payload["builtins"][0]["imports"] = ["hosting", "mp13_engine"]
    payload["builtins"][0]["package_requirements"] = ["mp13-engine==0.9.0"]
    if two_templates:
        payload["builtins"].append(
            {**payload["builtins"][0], "template_id": "py-compute"}
        )
    return ToolboxHostProjectConfiguration.from_dict(payload)


def _runtime_bundle(
    path: Path,
    configuration: ToolboxHostProjectConfiguration,
    private_key: Ed25519PrivateKey,
    *,
    packages: tuple[str, ...] = ("hosting", "mp13_engine"),
) -> None:
    payload = json.dumps(configuration.to_dict(), sort_keys=True, separators=(",", ":"))
    path.write_bytes(_runtime_bundle_bytes(payload, private_key.private_bytes_raw(), packages))


@lru_cache(maxsize=None)
def _runtime_bundle_bytes(
    configuration_payload: str,
    private_key_raw: bytes,
    packages: tuple[str, ...],
) -> bytes:
    configuration = ToolboxHostProjectConfiguration.from_dict(json.loads(configuration_payload))
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_raw)
    content = _wheel(
        "mp13-engine", "0.9.0", packages=packages
    )
    filename = "mp13_engine-0.9.0-py3-none-any.whl"
    manifest = {
        "contract": BUNDLE_CONTRACT,
        "bundle_id": "runtime-bundle",
        "source_id": "offline-release",
        "source_set_revision": configuration.source_set_revision,
        "target": {
            "name": configuration.target.name,
            "python_abi": configuration.target.python_abi,
            "platform": configuration.target.platform,
        },
        "signing_key_id": "release-key",
        "wheels": [
            _wheel_row(
                filename,
                content,
                distribution="mp13-engine",
                version="0.9.0",
            )
        ],
    }
    manifest_raw = _canonical(manifest)
    signature_raw = _canonical(
        {
            "contract": SIGNATURE_CONTRACT,
            "algorithm": "ed25519",
            "key_id": "release-key",
            "signature": _b64(private_key.sign(manifest_raw)),
        }
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_raw)
        archive.writestr("signature.json", signature_raw)
        archive.writestr(f"wheels/{filename}", content)
    return output.getvalue()


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


def test_admin_constructs_inactive_template_from_exact_verified_base(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration()
    private, public = _keys()
    source = tmp_path / "airgap"
    source.mkdir()
    _runtime_bundle(source / "runtime.zip", configuration, private)
    service = _manual_resolved_service(
        tmp_path, configuration=configuration, source=source, public=public
    )
    prepared = service.prepare_configured_toolbox_templates()
    published = service.publish_prepared_configured_toolbox_templates(prepared)
    base_digest = published["templates"][0]["template_digest"]

    started = service.toolbox_template_construct(
        template_id="team-core",
        base_template_digest=base_digest,
        imports=["hosting"],
        package_requirements=[],
        request_id="construct-team-core-1",
        owner_actor_id="admin:test",
    )
    duplicate = service.toolbox_template_construct(
        template_id="team-core",
        base_template_digest=base_digest,
        imports=["hosting"],
        package_requirements=[],
        request_id="construct-team-core-1",
        owner_actor_id="admin:test",
    )
    assert duplicate["operation"]["operation_id"] == started["operation"]["operation_id"]
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"], timeout_seconds=60
    )
    recovered = service.hosted_operation_resolve_request(
        execution_kind="toolbox_template_construct",
        selector={"kind": "template_id", "id": "team-core"},
        request_id="construct-team-core-1",
        owner_actor_id="admin:test",
    )

    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["code"] == "template_constructed_inactive"
    assert terminal["result"]["base_template_digest"] == base_digest
    assert recovered["operation"]["operation_id"] == started["operation"]["operation_id"]
    custom_digest = terminal["result"]["template_digest"]
    catalog = service._toolbox_template_catalog.read()  # noqa: SLF001
    custom = next(item for item in catalog["entries"] if item["template_digest"] == custom_digest)
    assert custom["lifecycle"] == "inactive"
    assert "team-core" not in catalog["active"]

    activated = service.toolbox_template_activate(
        template_id="team-core", template_digest=custom_digest, actor_id="admin:test"
    )
    assert activated["active_revision"] is True
    assert service._toolbox_template_catalog.read()["active"]["team-core"] == custom_digest  # noqa: SLF001


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
    terminal = _wait_daemon_setup(daemon)
    assert terminal["lifecycle"] == "terminal_failure"
    assert terminal["result"]["code"] == "required_template_runtime_artifact_missing"
    assert daemon.svc._toolbox_startup["status"] == "resolved"  # noqa: SLF001
    assert [item["template_id"] for item in daemon.svc._toolbox_startup["closures"]] == ["core"]  # noqa: SLF001
    assert daemon.svc._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    assert str(source) not in str(daemon.svc._toolbox_startup)  # noqa: SLF001
    assert next(iter(public.values())) not in str(daemon.svc.hosting_setup_summary())
    daemon.svc.close()

    restarted = construct()
    replayed = _wait_daemon_setup(restarted)
    assert replayed["lifecycle"] == "terminal_failure"
    assert replayed["result"]["code"] == "required_template_runtime_artifact_missing"
    assert len(restarted.svc._toolbox_artifact_store.read()["bundles"]) == 1  # noqa: SLF001
    assert restarted.svc._toolbox_startup["status"] == "pending"  # noqa: SLF001
    assert restarted.svc._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001


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

    terminal = _wait_daemon_setup(daemon)
    assert terminal["lifecycle"] == "terminal_failure"
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

    terminal = _wait_daemon_setup(daemon)
    assert terminal["lifecycle"] == "terminal_failure"
    assert daemon.svc._toolbox_startup["status"] == "not_ready"  # noqa: SLF001
    assert daemon.svc._toolbox_startup["closures"] == []  # noqa: SLF001
    assert daemon.svc._toolbox_startup["diagnostics"][0]["code"] == (  # noqa: SLF001
        "required_template_source_unavailable"
    )
    assert terminal["result"]["code"] == "required_template_source_unavailable"
    assert str(source) not in str(terminal["result"])
    assert daemon.svc._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001


def test_prepublication_candidate_build_uses_exact_cas_path_and_real_import_probes(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _runtime_bundle(source / "runtime.zip", configuration, private)
    service = _manual_resolved_service(
        tmp_path, configuration=configuration, source=source, public=public
    )
    assert service._toolbox_startup["status"] == "resolved"  # noqa: SLF001

    prepared = service.prepare_configured_toolbox_templates()

    assert prepared["status"] == "prepared"
    assert len(prepared["candidates"]) == 1
    candidate = prepared["candidates"][0]
    assert candidate["template"]["provenance"]["source"] == "signed-airgap:offline-release"
    assert candidate["template"]["parent_worker_artifact_digest"] == (
        candidate["artifact_references"][0]["sha256"]
    )
    assert candidate["receipt"]["verified_import_roots"] == ["hosting", "mp13_engine"]
    assert service._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    assert service._toolbox_materialization_receipts.get(  # noqa: SLF001
        template_digest=candidate["template_digest"],
        python_abi=configuration.target.python_abi,
        platform=configuration.target.platform,
    ) is None


def test_prepublication_candidate_failure_leaves_catalog_and_receipts_empty(tmp_path: Path) -> None:
    configuration = _runtime_configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _runtime_bundle(source / "runtime.zip", configuration, private)
    service = _manual_resolved_service(
        tmp_path, configuration=configuration, source=source, public=public
    )
    artifact = next(
        iter(service._toolbox_verified_artifacts["offline-release"].values())  # noqa: SLF001
    )
    artifact.write_bytes(b"corrupt-after-resolution")

    with pytest.raises(Exception, match="environment_artifact_verification_failed"):
        service.prepare_configured_toolbox_templates()

    assert service._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    assert not service._toolbox_materialization_receipts.path.exists()  # noqa: SLF001


def test_prepublication_import_probe_failure_releases_reference_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _runtime_bundle(
        source / "runtime.zip", configuration, private, packages=("mp13_engine",)
    )
    service = _manual_resolved_service(
        tmp_path, configuration=configuration, source=source, public=public
    )

    with pytest.raises(Exception, match="environment_import_probe_failed"):
        service.prepare_configured_toolbox_templates()

    assert service._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    assert not service._toolbox_materialization_receipts.path.exists()  # noqa: SLF001
    references = service._hermetic_toolbox_environment_builder.references_path  # noqa: SLF001
    assert not references.exists() or json.loads(references.read_text())["environments"] == {}


def test_candidate_provenance_requires_one_unambiguous_signed_bundle(tmp_path: Path) -> None:
    configuration = _configuration()
    private, public = _keys()
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _bundle(first, configuration, private, bundle_id="bundle-first")
    _bundle(second, configuration, private, bundle_id="bundle-second")
    store = AtomicToolboxArtifactStore(tmp_path / "store")
    imported = store.import_signed_bundle(
        first, configuration=configuration, trust_public_keys=public
    )
    store.import_signed_bundle(second, configuration=configuration, trust_public_keys=public)

    with pytest.raises(ValueError, match="artifact_evidence_ambiguous"):
        store.bundle_evidence_for_artifacts(set(imported["artifact_digests"]))


def test_complete_prepared_batch_publishes_atomically_and_is_restart_idempotent(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration(two_templates=True)
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _runtime_bundle(source / "runtime.zip", configuration, private)

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
    terminal = _wait_daemon_setup(daemon)

    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["code"] == "toolbox_setup_ready"
    assert {item["template_id"] for item in terminal["result"]["templates"]} == {
        "core", "py-compute"
    }
    catalog = daemon.svc._toolbox_template_catalog.read()  # noqa: SLF001
    assert set(catalog["active"]) == {"core", "py-compute"}
    assert len(catalog["entries"]) == 2
    assert daemon.svc.hosting_setup_summary()["toolbox_readiness"]["status"] == "ready"
    daemon.svc.close()

    restarted = construct()
    replayed = _wait_daemon_setup(restarted)
    assert replayed["operation"]["operation_id"] == terminal["operation"]["operation_id"]
    assert restarted.svc.hosting_setup_summary()["toolbox_readiness"]["status"] == "ready"
    assert len(restarted.svc._toolbox_template_catalog.read()["entries"]) == 2  # noqa: SLF001


def test_catalog_batch_failure_rolls_back_new_receipt_and_candidate_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _runtime_configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _runtime_bundle(source / "runtime.zip", configuration, private)
    service = _manual_resolved_service(
        tmp_path, configuration=configuration, source=source, public=public
    )
    prepared = service.prepare_configured_toolbox_templates()

    monkeypatch.setattr(
        AtomicJsonToolboxTemplateCatalog,
        "publish_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected-batch-failure")),
    )
    with pytest.raises(RuntimeError, match="injected-batch-failure"):
        service.publish_prepared_configured_toolbox_templates(prepared)

    assert service._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
    candidate = prepared["candidates"][0]
    assert service._toolbox_materialization_receipts.get(  # noqa: SLF001
        template_digest=candidate["template_digest"],
        python_abi=configuration.target.python_abi,
        platform=configuration.target.platform,
    ) is None
    references = service._hermetic_toolbox_environment_builder.references_path  # noqa: SLF001
    assert json.loads(references.read_text())["environments"] == {}


def test_system_setup_operation_returns_immediately_is_idempotent_and_publishes_readiness(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _runtime_bundle(source / "runtime.zip", configuration, private)
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )

    started_at = time.monotonic()
    started = daemon.svc._toolbox_setup_operation  # noqa: SLF001
    duplicate = daemon.svc.toolbox_setup_start()
    elapsed = time.monotonic() - started_at

    assert elapsed < 2
    assert duplicate["operation"]["operation_id"] == started["operation"]["operation_id"]
    assert started["operation"]["execution_kind"] == "toolbox_setup"
    assert started["operation"]["selector"] == {"kind": "host_scope", "id": "toolbox-host"}
    terminal = daemon.svc._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"], timeout_seconds=60
    )
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["code"] == "toolbox_setup_ready"
    assert terminal["progress"]["phase"] == "publication"
    assert terminal["progress"]["cancellable"] is False
    assert daemon.svc.hosting_setup_summary()["toolbox_readiness"]["status"] == "ready"
    recovered = daemon.svc.hosted_operation_resolve_request(
        execution_kind="toolbox_setup",
        selector={"kind": "host_scope", "id": "toolbox-host"},
        request_id=started["operation"]["request_id"],
        owner_actor_id="system:toolbox-setup",
    )
    assert recovered["operation"]["operation_id"] == started["operation"]["operation_id"]


def test_normal_daemon_constructor_does_not_wait_for_bundle_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _runtime_configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _runtime_bundle(source / "runtime.zip", configuration, private)
    release = threading.Event()
    entered = threading.Event()
    original = AtomicToolboxArtifactStore.import_signed_bundle

    def delayed_import(self, *args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AtomicToolboxArtifactStore, "import_signed_bundle", delayed_import)
    started_at = time.monotonic()
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )
    elapsed = time.monotonic() - started_at

    assert elapsed < 2
    assert entered.wait(2)
    summary = daemon.svc.hosting_setup_summary()
    assert summary["toolbox_readiness"]["code"] == "toolbox_setup_in_progress"
    assert summary["toolbox_setup_operation"]["operation"] == (
        daemon.svc._toolbox_setup_operation["operation"]  # noqa: SLF001
    )
    release.set()
    assert _wait_daemon_setup(daemon)["lifecycle"] == "terminal_success"


def test_restart_redispatches_interrupted_before_dispatch_on_same_canonical_record(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration()
    private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    _runtime_bundle(source / "runtime.zip", configuration, private)
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )
    queued = _prepare_unstarted_setup(service)
    operation_id = queued["status"]["operation"]["operation_id"]
    _drop_operation_repository(service)

    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )
    terminal = _wait_daemon_setup(daemon)

    assert terminal["operation"]["operation_id"] == operation_id
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["code"] == "toolbox_setup_ready"


def test_restart_terminally_reconciles_interrupted_after_dispatch_without_parallel_record(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration()
    _private, public = _keys()
    source = tmp_path / "read-only-source"
    source.mkdir()
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )
    queued = _prepare_unstarted_setup(service)
    operation_id = queued["status"]["operation"]["operation_id"]
    service._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)  # noqa: SLF001
    _drop_operation_repository(service)

    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"offline-release": source},
        toolbox_dependency_policy=_policy(),
        toolbox_trust_public_keys=public,
    )
    terminal = _wait_daemon_setup(daemon)

    assert terminal["operation"]["operation_id"] == operation_id
    assert terminal["lifecycle"] == "terminal_failure"
    assert terminal["result"]["code"] == "toolbox_setup_interrupted_after_dispatch"
    assert daemon.svc._toolbox_template_catalog.read()["entries"] == []  # noqa: SLF001
