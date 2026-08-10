from __future__ import annotations

import base64
import asyncio
import hashlib
import io
import json
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from packaging.utils import parse_wheel_filename

from hosting.service.host_service import EngineHostService
from hosting.daemon.local_ipc import EngineHostDaemon
from hosting.service.toolbox_artifact_store import BUNDLE_CONTRACT, SIGNATURE_CONTRACT
from hosting.service.toolbox_definition_resolution import ConfiguredToolboxPlanResolver
from hosting.toolbox.catalog import (
    ToolboxEnvironmentTemplateSpec,
    ToolboxLockedDistributionSpec,
    ToolboxTemplateProvenance,
)
from hosting.toolbox.definition_planner import (
    ActiveToolboxEnvironmentResolution,
    ToolboxDefinitionPlanDraft,
    build_toolbox_environment_mutations,
    plan_toolbox_definition,
)
from hosting.toolbox.orchestration import ToolboxSandboxOrchestrator
from hosting.toolbox.staging import ToolboxBundleStager
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
    source.mkdir(parents=True, exist_ok=True)
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


def test_removed_custom_packages_recompute_to_exact_builtin_closure(
    tmp_path: Path,
) -> None:
    service, template = _service_with_verified_closure(tmp_path)
    configuration = service._toolbox_host_project_config  # noqa: SLF001
    custom_draft = plan_toolbox_definition(
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
    active_candidate = resolver.candidates_for_draft(custom_draft)[0]
    active_environment = ActiveToolboxEnvironmentResolution(
        environment_id=custom_draft.profiles[0].profile_id,
        tool_keys=custom_draft.profiles[0].assigned_tool_keys,
        base_template_id=active_candidate.base_template_id,
        base_template_revision=active_candidate.base_template_revision,
        source_ids=active_candidate.source_ids,
        source_origins=active_candidate.source_origins,
        lock_digest=active_candidate.lock_digest,
        artifacts=active_candidate.artifacts,
    )
    contracted_definition = json.loads(json.dumps(_definition()))
    contracted_definition["expected_revision"] = custom_draft.definition.revision
    request = contracted_definition["auto_requests"][0]
    request["files"] = [{
        "relative_path": "pkg/fetch.py",
        "content": "def Fetch():\n    return 'base-only'\n",
    }]
    request["dependency"] = {
        "mode": "template",
        "template_id": "core",
        "declared_imports": [],
        "package_requirements": [],
    }
    contracted_draft = plan_toolbox_definition(
        contracted_definition,
        templates=(template,),
        python_abi=configuration.target.python_abi,
        platform=configuration.target.platform,
        runtime_identity={
            "version": "3.12.7",
            "artifact_digest": template.parent_worker_artifact_digest,
        },
    )
    contracted_candidates = resolver.candidates_for_draft(contracted_draft)
    offers = build_toolbox_environment_mutations(
        active_definition=custom_draft.definition,
        draft=contracted_draft,
        candidates=contracted_candidates,
        active_environments=(active_environment,),
        dependency_approval_required=True,
    )

    assert contracted_draft.custom_environment_count == 0
    assert contracted_draft.profiles[0].custom_resolved_lock_digest is None
    assert {item.distribution for item in contracted_candidates[0].artifacts} == {
        "packaging"
    }
    mutations = offers[0].alternatives[0].package_mutations
    assert {(item.distribution, item.mutation) for item in mutations} == {
        ("requests", "removal"),
        ("urllib3", "removal"),
    }
    assert offers[0].confirmation_required is False
    assert offers[0].dependency_approval_required is False


def test_confirmed_custom_closure_flows_through_orchestration_to_real_builder(
    tmp_path: Path,
) -> None:
    from test_hosting_toolbox_definition_service import _custom_policy

    service, _template = _service_with_verified_closure(tmp_path, policy=_custom_policy())
    started = service.toolbox_plan_definition(
        definition=_definition(), request_id="plan-builder",
        owner_actor_id="actor:a", authority_id="workspace:a",
    )
    planned = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"], timeout_seconds=10
    )["result"]
    choices = [{
        "environment_id": item["environment_id"],
        "alternative_id": item["preferred_alternative_id"],
        "accept_package_changes": True,
    } for item in planned["environment_mutations"]]
    confirmation = service.toolbox_confirm_definition_plan(
        plan_id=planned["plan_id"], environment_choices=choices,
        request_id="confirm-builder", owner_actor_id="actor:a", authority_id="workspace:a",
    )
    confirmed = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=confirmation["operation"]["operation_id"], timeout_seconds=10
    )["result"]
    receipt = service._toolbox_confirmations.get(  # noqa: SLF001
        confirmed["confirmation_ref"], owner_actor_id="actor:a",
        authority_id="workspace:a", now_ms=0,
    )
    draft = ToolboxDefinitionPlanDraft.from_persisted_dict(receipt.confirmed_draft)
    orchestrator = ToolboxSandboxOrchestrator(
        service=service, stager=ToolboxBundleStager(service.hosting_root)
    )
    assignments = orchestrator.build_resolved_assignments(
        toolbox_id=draft.definition.toolbox_id,
        profiles=draft.profiles,
        bundles=draft.bundles,
        profile_changes=[{
            "classification": "added", "active_profile_id": None,
            "proposed_profile_id": draft.profiles[0].profile_id,
            "changed_fields": [],
        }],
    )
    service.spawn = lambda **kwargs: {  # type: ignore[method-assign]
        "engine_id": kwargs["engine_id"],
        "bundle": dict(kwargs["bundle"]),
        "environment": dict(kwargs["environment"]),
    }

    spawned = orchestrator.spawn_resolved_assignments(
        toolbox_id=draft.definition.toolbox_id,
        definition_revision=draft.definition.revision,
        assignments=assignments,
        resolved_environments=receipt.resolved_environments,
    )

    assert spawned[0].registration is not None
    assert spawned[0].registration["environment"]["environment_key"] == draft.profiles[0].environment_key
    environment_root = Path(spawned[0].registration["environment"]["venv_path"])
    receipt_payload = json.loads(
        (environment_root / "verification-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt_payload["resolved"]["custom_resolved_lock_digest"] == draft.profiles[0].custom_resolved_lock_digest


def test_corrupt_confirmed_artifact_fails_before_atomic_publication(tmp_path: Path) -> None:
    from test_hosting_toolbox_definition_service import _custom_policy

    service, _template = _service_with_verified_closure(tmp_path, policy=_custom_policy())
    started = service.toolbox_plan_definition(
        definition=_definition(), request_id="plan-corrupt",
        owner_actor_id="actor:a", authority_id="workspace:a",
    )
    plan = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"], timeout_seconds=10
    )["result"]
    choices = [{
        "environment_id": item["environment_id"],
        "alternative_id": item["preferred_alternative_id"],
        "accept_package_changes": True,
    } for item in plan["environment_mutations"]]
    confirmation_started = service.toolbox_confirm_definition_plan(
        plan_id=plan["plan_id"], environment_choices=choices,
        request_id="confirm-corrupt", owner_actor_id="actor:a", authority_id="workspace:a",
    )
    confirmation = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=confirmation_started["operation"]["operation_id"], timeout_seconds=10
    )["result"]
    approval = service.toolbox_approve_confirmed_definition_plan(
        confirmation_ref=confirmation["confirmation_ref"],
        approver_actor_id="approver:dependencies",
        dependency_approver_authorized=True,
    )
    receipt = service._toolbox_confirmations.get(  # noqa: SLF001
        confirmation["confirmation_ref"], owner_actor_id="actor:a",
        authority_id="workspace:a", now_ms=0,
    )
    resolved = next(iter(receipt.resolved_environments.values()))
    corrupt = resolved["locked_artifacts"][-1]
    service._toolbox_artifact_store.object_path(corrupt["sha256"]).write_bytes(b"corrupt")  # noqa: SLF001
    references = service.hosting_root / "toolbox_environment_cache" / "references.json"

    applied = service.toolbox_apply_definition(
        plan_id=plan["plan_id"],
        confirmation_ref=confirmation["confirmation_ref"],
        dependency_approval_ref=approval["approval_ref"],
        request_id="apply-corrupt",
        owner_actor_id="actor:a",
        authority_id="workspace:a",
    )
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=applied["operation"]["operation_id"], timeout_seconds=30
    )

    assert terminal["lifecycle"] == "terminal_failure"
    assert service._toolbox_state_v2.get("custom-demo") is None  # noqa: SLF001
    assert service._toolbox_executor_registrations("custom-demo") == []  # noqa: SLF001
    if references.exists():
        assert json.loads(references.read_text(encoding="utf-8"))["environments"] == {}


def test_authenticated_daemon_recovers_one_multi_tool_plan_and_confirmation(
    tmp_path: Path,
) -> None:
    service, _template = _service_with_verified_closure(tmp_path)
    service.auth_upsert_key(
        key_id="consumer", key_secret="consumer-secret", role="worker_user",
        auth_method="shared_secret",
    )
    service.set_control_config(
        require_auth=True, access_profile={"connectivity_mode": "local_only"}
    )
    token = service.auth_issue_session(
        key_id="consumer", key_secret="consumer-secret", scope="control"
    )["token"]
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "unused-engines.json",
        control_state_file=tmp_path / "unused-control.json",
    )
    daemon.svc = service
    definition = _definition()
    definition["toolbox_id"] = "multi-daemon"
    definition["auto_requests"][0]["files"][0]["content"] = "def Fetch():\n    return 1\n"
    definition["auto_requests"][0]["dependency"] = {
        "mode": "auto", "template_id": None,
        "declared_imports": [], "package_requirements": [],
    }
    second = json.loads(json.dumps(definition["auto_requests"][0]))
    second["files"] = [{"relative_path": "pkg/second.py", "content": "def Second():\n    return 2\n"}]
    second["module_name"] = "pkg.second"
    second["callable_name"] = "Second"
    definition["auto_requests"].append(second)
    plan_request = {
        "seq": 1,
        "cmd": "op-start",
        "payload": {
            "session_token": token,
            "command": "toolbox-plan-definition",
            "payload": {"request_id": "multi-plan", "definition": definition},
        },
    }

    first = asyncio.run(daemon._dispatch(  # noqa: SLF001
        json.dumps(plan_request), peer_host="127.0.0.1", transport="local_ipc"
    ))
    duplicate = asyncio.run(daemon._dispatch(  # noqa: SLF001
        json.dumps({**plan_request, "seq": 2}),
        peer_host="127.0.0.1", transport="local_ipc",
    ))
    assert first["ok"] is duplicate["ok"] is True
    assert first["result"]["operation"] == duplicate["result"]["operation"]
    planned = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=first["result"]["operation"]["operation_id"], timeout_seconds=10
    )["result"]
    offered_tools = {
        item["tool_key"]
        for offer in planned["environment_mutations"]
        for item in offer["tool_mutations"]
    }
    assert offered_tools == {"pkg.fetch:Fetch", "pkg.second:Second"}
    choices = [{
        "environment_id": offer["environment_id"],
        "alternative_id": offer["preferred_alternative_id"],
        "accept_package_changes": True,
    } for offer in planned["environment_mutations"]]
    confirmed = asyncio.run(daemon._dispatch(  # noqa: SLF001
        json.dumps({
            "seq": 3,
            "cmd": "op-start",
            "payload": {
                "session_token": token,
                "command": "toolbox-confirm-definition-plan",
                "payload": {
                    "request_id": "multi-confirm",
                    "plan_id": planned["plan_id"],
                    "environment_choices": choices,
                },
            },
        }),
        peer_host="127.0.0.1", transport="local_ipc",
    ))
    terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=confirmed["result"]["operation"]["operation_id"], timeout_seconds=10
    )
    assert terminal["lifecycle"] == "terminal_success"
    assert set(terminal["result"]["accepted_tool_keys"]) == offered_tools
    assert not daemon._operations_state_file.exists()  # noqa: SLF001
