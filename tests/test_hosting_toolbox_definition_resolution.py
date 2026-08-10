from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from packaging.utils import parse_wheel_filename

from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_artifact_store import BUNDLE_CONTRACT, SIGNATURE_CONTRACT
from hosting.service.toolbox_definition_resolution import ConfiguredToolboxPlanResolver
from hosting.toolbox.catalog import (
    ToolboxEnvironmentTemplateSpec,
    ToolboxLockedDistributionSpec,
    ToolboxTemplateProvenance,
)
from hosting.toolbox.definition_planner import (
    build_toolbox_environment_mutations,
    plan_toolbox_definition,
)
from hosting.toolbox.identity import identity_digest


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _wheel(distribution: str, version: str, import_root: str, *, requires=()) -> tuple[str, bytes]:
    canonical = distribution.replace("-", "_")
    filename = f"{canonical}-{version}-py3-none-any.whl"
    output = io.BytesIO()
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {distribution}\n"
        f"Version: {version}\n"
        "Requires-Python: >=3.12,<3.13\n"
        + "".join(f"Requires-Dist: {item}\n" for item in requires)
        + "\n"
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{import_root}/__init__.py", "")
        archive.writestr(f"{canonical}-{version}.dist-info/METADATA", metadata)
        archive.writestr(
            f"{canonical}-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{canonical}-{version}.dist-info/RECORD", "")
    return filename, output.getvalue()


def _configuration() -> dict:
    return {
        "builtins": [
            {
                "template_id": "core",
                "imports": ["packaging"],
                "package_requirements": ["packaging==26.0"],
                "sandbox_policy": "compute-only",
                "required": True,
                "prewarm": True,
                "provenance": "definition-resolution-test",
            }
        ],
        "sources": [
            {
                "source_id": "release",
                "kind": "airgap_store",
                "origin": "airgap://release",
                "credential_ref": None,
                "allowed_package_namespaces": ["*"],
                "priority": 10,
                "trust_key_ids": ["release-key"],
                "maximum_download_bytes": 16 * 1024 * 1024,
            }
        ],
        "resolution": {
            "mode": "air_gapped",
            "timeout_seconds": 60,
            "maximum_bytes": 16 * 1024 * 1024,
            "maximum_artifacts": 32,
            "allowed_redirect_origins": [],
            "wheel_only": True,
        },
        "retention": {
            "artifact_cache_grace_seconds": 60,
            "maximum_cache_bytes": 64 * 1024 * 1024,
            "maximum_cache_artifacts": 128,
            "protected_digests": [],
            "remove_unreferenced_custom_revisions_on_apply": False,
        },
    }


def _definition() -> dict:
    return {
        "contract": "hosting.toolbox.definition",
        "toolbox_id": "custom-demo",
        "expected_revision": None,
        "auto_requests": [
            {
                "files": [
                    {
                        "relative_path": "pkg/fetch.py",
                        "content": "import requests\ndef Fetch():\n    return requests.__version__\n",
                    }
                ],
                "module_name": "pkg.fetch",
                "callable_name": "Fetch",
                "dependency": {
                    "mode": "custom",
                    "template_id": "core",
                    "declared_imports": [],
                    "package_requirements": ["requests==2.32.5"],
                },
                "sandbox_policy": {"sandbox": {"enabled": True}},
                "activate": True,
                "hidden": False,
                "non_restartable": False,
                "guide_content": None,
                "guide_description": None,
                "callback_signature": None,
                "concurrency": None,
            }
        ],
        "manual_requests": [],
        "intrinsics": {
            "names": [],
            "include_guides": False,
            "sandbox_policy": {"sandbox": {"enabled": True}},
        },
    }


def _service_with_verified_closure(tmp_path: Path, *, policy=None):
    private = Ed25519PrivateKey.generate()
    public = _b64(private.public_key().public_bytes_raw())
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=_configuration(),
        toolbox_artifact_sources={"release": source},
        toolbox_trust_public_keys={"release-key": public},
        toolbox_dependency_policy=policy,
    )
    wheels = [
        _wheel("packaging", "26.0", "packaging"),
        _wheel("requests", "2.32.5", "requests", requires=("urllib3==2.0.0",)),
        _wheel("urllib3", "2.0.0", "urllib3"),
    ]
    configuration = service._toolbox_host_project_config  # noqa: SLF001
    manifest = {
        "contract": BUNDLE_CONTRACT,
        "bundle_id": "definition-resolution",
        "source_id": "release",
        "source_set_revision": configuration.source_set_revision,
        "target": {
            "name": configuration.target.name,
            "python_abi": configuration.target.python_abi,
            "platform": configuration.target.platform,
        },
        "signing_key_id": "release-key",
        "wheels": [],
    }
    for filename, content in wheels:
        name, version, _build, tags = parse_wheel_filename(filename)
        manifest["wheels"].append(
            {
                "distribution": str(name).replace("_", "-"),
                "version": str(version),
                "filename": filename,
                "size_bytes": len(content),
                "sha256": _digest(content),
                "tags": sorted(str(item) for item in tags),
                "provenance": "definition-resolution-test",
            }
        )
    manifest["wheels"] = sorted(manifest["wheels"], key=lambda item: item["filename"])
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    signature = {
        "contract": SIGNATURE_CONTRACT,
        "algorithm": "ed25519",
        "key_id": "release-key",
        "signature": _b64(private.sign(raw)),
    }
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", raw)
        archive.writestr(
            "signature.json",
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode(),
        )
        for filename, content in wheels:
            archive.writestr(f"wheels/{filename}", content)
    imported = service._toolbox_artifact_store.import_signed_bundle(  # noqa: SLF001
        bundle,
        configuration=configuration,
        trust_public_keys={"release-key": public},
        expected_source_id="release",
    )
    packaging_row = next(item for item in manifest["wheels"] if item["distribution"] == "packaging")
    template = ToolboxEnvironmentTemplateSpec(
        template_id="core",
        python_requires=">=3.12,<3.13",
        python_abis=(configuration.target.python_abi,),
        runtime_kind="toolbox_python",
        worker_protocol_version="1.0",
        platforms=(configuration.target.platform,),
        locked_distributions=(ToolboxLockedDistributionSpec("packaging", "26.0"),),
        exposed_import_roots=("packaging",),
        lock_digest=identity_digest("test.definition.base.lock.v1", packaging_row),
        parent_worker_artifact_digest=packaging_row["sha256"],
        isolation_policy_version="compute-only-v1",
        provenance=ToolboxTemplateProvenance(
            source="signed-airgap:release",
            revision=imported["bundle_id"],
            manifest_digest=imported["manifest_digest"],
            signing_key_id="release-key",
        ),
    )
    service.toolbox_template_publish(
        template=template.to_dict(),
        artifact_references=[
            {
                "source_id": "release",
                "filename": packaging_row["filename"],
                "sha256": packaging_row["sha256"],
                "size_bytes": packaging_row["size_bytes"],
            }
        ],
        manifest_signature="s" * 64,
        activate=True,
        actor_id="test:definition-resolution",
    )
    return service, template


def test_configured_resolver_builds_exact_direct_transitive_verified_cas_offer(
    tmp_path: Path,
) -> None:
    service, template = _service_with_verified_closure(tmp_path)
    configuration = service._toolbox_host_project_config  # noqa: SLF001
    draft = plan_toolbox_definition(
        _definition(),
        templates=(template,),
        python_abi=configuration.target.python_abi,
        platform=configuration.target.platform,
        runtime_identity={
            "version": "3.12.7",
            "artifact_digest": template.parent_worker_artifact_digest,
        },
    )
    resolver = ConfiguredToolboxPlanResolver(
        configuration=configuration,
        artifact_store=service._toolbox_artifact_store,  # noqa: SLF001
        catalog_state=service._toolbox_template_catalog.read(),  # noqa: SLF001
    )

    candidates = resolver.candidates_for_draft(draft)
    offers = build_toolbox_environment_mutations(
        active_definition=service.toolbox_get_definition(toolbox_id="custom-demo")[
            "definition"
        ],
        draft=draft,
        candidates=candidates,
        dependency_approval_required=True,
    )

    assert len(candidates) == len(offers) == 1
    alternative = offers[0].alternatives[0]
    assert {item.distribution for item in alternative.artifacts} == {
        "packaging",
        "requests",
        "urllib3",
    }
    assert {item.distribution: item.dependency_reason for item in alternative.artifacts} == {
        "packaging": "template_runtime",
        "requests": "direct",
        "urllib3": "transitive",
    }
    assert alternative.source_ids == ("release",)
    assert alternative.source_origins == ("airgap://release",)
    assert all(".mp13" not in str(item.to_dict()) for item in alternative.artifacts)
    assert offers[0].confirmation_required is True
    assert offers[0].dependency_approval_required is True
