from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from packaging.utils import parse_wheel_filename

from hosting.daemon.local_ipc import EngineHostDaemon
from hosting.service.toolbox_artifact_store import BUNDLE_CONTRACT, SIGNATURE_CONTRACT
from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration
from hosting.toolbox.identity import identity_digest


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _configuration() -> ToolboxHostProjectConfiguration:
    return ToolboxHostProjectConfiguration.from_dict(
        {
            "builtins": [
                {
                    "template_id": "core",
                    "imports": ["hosting", "mp13_engine"],
                    "package_requirements": ["mp13-engine==0.9.0"],
                    "sandbox_policy": "compute-only",
                    "required": True,
                    "prewarm": True,
                    "provenance": "r7-acceptance",
                }
            ],
            "sources": [
                {
                    "source_id": "r7-release",
                    "kind": "airgap_store",
                    "origin": "airgap://r7-release",
                    "credential_ref": None,
                    "allowed_package_namespaces": ["*"],
                    "priority": 100,
                    "trust_key_ids": ["r7-key"],
                    "maximum_download_bytes": 128 * 1024 * 1024,
                }
            ],
            "resolution": {
                "mode": "air_gapped",
                "timeout_seconds": 120,
                "maximum_bytes": 128 * 1024 * 1024,
                "maximum_artifacts": 16,
                "allowed_redirect_origins": [],
                "wheel_only": True,
            },
            "retention": {
                "artifact_cache_grace_seconds": 1,
                "maximum_cache_bytes": 256 * 1024 * 1024,
                "maximum_cache_artifacts": 64,
                "protected_digests": [],
                "remove_unreferenced_custom_revisions_on_apply": False,
            },
        }
    )


def _add_tree(archive: zipfile.ZipFile, source: Path, destination: str) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(source).as_posix()
        if destination in {"hosting", "mp13_engine"} and relative == "__init__.py":
            continue
        archive.write(path, f"{destination}/{relative}")


def _add_installed_module(archive: zipfile.ZipFile, module_name: str) -> None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise AssertionError(f"R7 runtime fixture dependency is unavailable: {module_name}")
    locations = list(spec.submodule_search_locations or [])
    if locations:
        _add_tree(archive, Path(locations[0]), module_name)
        return
    origin = Path(str(spec.origin or ""))
    if not origin.is_file():
        raise AssertionError(f"R7 runtime fixture module has no file: {module_name}")
    archive.write(origin, origin.name)


def _runtime_wheel(configuration: ToolboxHostProjectConfiguration) -> tuple[str, bytes]:
    tag = next(
        item
        for item in configuration.target.compatible_tags
        if item.startswith(f"{configuration.target.python_abi}-{configuration.target.python_abi}-")
    )
    filename = f"mp13_engine-0.9.0-{tag}.whl"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _add_tree(archive, ROOT / "src" / "hosting", "hosting")
        _add_tree(archive, ROOT / "src" / "mp13_engine", "mp13_engine")
        archive.writestr("hosting/__init__.py", "\"\"\"R7 hermetic worker runtime.\"\"\"\n")
        archive.writestr("mp13_engine/__init__.py", "\"\"\"R7 hermetic worker runtime.\"\"\"\n")
        for module_name in (
            "_cffi_backend",
            "annotated_types",
            "cffi",
            "cryptography",
            "certifi",
            "charset_normalizer",
            "idna",
            "packaging",
            "pydantic",
            "pydantic_core",
            "requests",
            "typing_inspection",
            "typing_extensions",
            "urllib3",
        ):
            _add_installed_module(archive, module_name)
        dist_info = "mp13_engine-0.9.0.dist-info"
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: mp13-engine\nVersion: 0.9.0\n"
            "Requires-Python: >=3.12,<3.13\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: mp13-r7-acceptance\n"
            f"Root-Is-Purelib: false\nTag: {tag}\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return filename, output.getvalue()


def _addon_wheel() -> tuple[str, bytes]:
    filename = "requests-2.32.5-py3-none-any.whl"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("requests/__init__.py", "__version__ = 'addon-ok'\n")
        dist_info = "requests-2.32.5.dist-info"
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: requests\nVersion: 2.32.5\n"
            "Requires-Python: >=3.12,<3.13\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: mp13-r7-acceptance\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return filename, output.getvalue()


def _write_signed_source(
    source: Path,
    configuration: ToolboxHostProjectConfiguration,
) -> dict[str, str]:
    private_key = Ed25519PrivateKey.generate()
    wheels = [_runtime_wheel(configuration), _addon_wheel()]
    rows: list[dict[str, Any]] = []
    for filename, content in wheels:
        name, version, _build, tags = parse_wheel_filename(filename)
        rows.append(
            {
                "distribution": str(name).replace("_", "-"),
                "version": str(version),
                "filename": filename,
                "size_bytes": len(content),
                "sha256": _digest(content),
                "tags": sorted(str(item) for item in tags),
                "provenance": "r7-acceptance",
            }
        )
    manifest = {
        "contract": BUNDLE_CONTRACT,
        "bundle_id": "r7-acceptance-bundle",
        "source_id": "r7-release",
        "source_set_revision": configuration.source_set_revision,
        "target": {
            "name": configuration.target.name,
            "python_abi": configuration.target.python_abi,
            "platform": configuration.target.platform,
        },
        "signing_key_id": "r7-key",
        "wheels": sorted(rows, key=lambda item: item["filename"]),
    }
    manifest_raw = _canonical(manifest)
    signature = {
        "contract": SIGNATURE_CONTRACT,
        "algorithm": "ed25519",
        "key_id": "r7-key",
        "signature": _b64(private_key.sign(manifest_raw)),
    }
    source.mkdir(parents=True)
    with zipfile.ZipFile(source / "r7-release.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_raw)
        archive.writestr("signature.json", _canonical(signature))
        for filename, content in wheels:
            archive.writestr(f"wheels/{filename}", content)
    return {"r7-key": _b64(private_key.public_key().public_bytes_raw())}


def _policy(configuration: ToolboxHostProjectConfiguration) -> dict[str, Any]:
    body = {
        "allowed_template_ids": ["core"],
        "allowed_targets": [configuration.target.name],
        "package_allowlist": ["requests"],
        "package_denylist": [],
        "allow_custom": True,
        "custom_requires_approval": True,
        "online_resolution_allowed": False,
        "allowed_index_origins": [],
    }
    return {"revision": identity_digest("hosting.toolbox.r7.policy.v1", body), **body}


def _tool(
    *,
    module_name: str,
    callable_name: str,
    source: str,
    dependency: dict[str, Any],
) -> dict[str, Any]:
    return {
        "files": [{"relative_path": module_name.replace(".", "/") + ".py", "content": source}],
        "module_name": module_name,
        "callable_name": callable_name,
        "dependency": dependency,
        "sandbox_policy": {"sandbox": {"enabled": True}},
        "activate": True,
        "hidden": False,
        "non_restartable": False,
        "guide_content": None,
        "guide_description": None,
        "callback_signature": None,
        "concurrency": None,
    }


def _definition(*, expected_revision: str | None, include_addon: bool) -> dict[str, Any]:
    requests = [
        _tool(
            module_name="r7.base",
            callable_name="Base",
            source="def Base():\n    return 'base-ok'\n",
            dependency={
                "mode": "template",
                "template_id": "core",
                "declared_imports": [],
                "package_requirements": [],
            },
        )
    ]
    if include_addon:
        requests.append(
            _tool(
                module_name="r7.addon",
                callable_name="Addon",
                source=(
                    "import requests\n"
                    "def Addon():\n"
                    "    return requests.__version__\n"
                ),
                dependency={
                    "mode": "custom",
                    "template_id": "core",
                    "declared_imports": ["requests"],
                    "package_requirements": ["requests==2.32.5"],
                },
            )
        )
    return {
        "contract": "hosting.toolbox.definition",
        "toolbox_id": "r7-e2e",
        "expected_revision": expected_revision,
        "auto_requests": requests,
        "manual_requests": [],
        "intrinsics": {
            "names": [],
            "include_guides": False,
            "sandbox_policy": {"sandbox": {"enabled": True}},
        },
    }


def _daemon(
    root: Path,
    *,
    configuration: ToolboxHostProjectConfiguration,
    source: Path,
    public_keys: dict[str, str],
) -> EngineHostDaemon:
    return EngineHostDaemon(
        pid_file=root / "daemon.pid",
        engines_state_file=root / "engines.json",
        control_state_file=root / "control.json",
        toolbox_host_project_configuration=configuration.to_dict(),
        toolbox_artifact_sources={"r7-release": source},
        toolbox_dependency_policy=_policy(configuration),
        toolbox_trust_public_keys=public_keys,
    )


def _tokens(daemon: EngineHostDaemon) -> dict[str, str]:
    result: dict[str, str] = {}
    secrets: dict[str, str] = {}
    for key_id, role in (
        ("consumer", "worker_user"),
        ("approver", "dependency_approver"),
        ("administrator", "admin"),
    ):
        secret = f"{key_id}-secret"
        daemon.svc.auth_upsert_key(
            key_id=key_id,
            key_secret=secret,
            role=role,
            auth_method="shared_secret",
        )
        secrets[key_id] = secret
    daemon.svc.set_control_config(
        require_auth=True,
        access_profile={"connectivity_mode": "local_only"},
    )
    for key_id, secret in secrets.items():
        result[key_id] = daemon.svc.auth_issue_session(
            key_id=key_id,
            key_secret=secret,
            scope="control",
        )["token"]
    return result


def _dispatch(
    daemon: EngineHostDaemon,
    *,
    seq: int,
    command: str,
    payload: dict[str, Any],
    token: str,
) -> dict[str, Any]:
    response = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            json.dumps(
                {
                    "seq": seq,
                    "cmd": command,
                    "payload": {"session_token": token, **payload},
                }
            ),
            peer_host="127.0.0.1",
            transport="local_ipc",
        )
    )
    assert response["ok"] is True, response
    return dict(response["result"])


def _start(
    daemon: EngineHostDaemon,
    *,
    seq: int,
    command: str,
    payload: dict[str, Any],
    token: str,
) -> dict[str, Any]:
    return _dispatch(
        daemon,
        seq=seq,
        command="op-start",
        payload={"command": command, "payload": payload},
        token=token,
    )


def _terminal(daemon: EngineHostDaemon, started: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
    return daemon.svc._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"],
        timeout_seconds=timeout,
    )


def _choices(plan: dict[str, Any], *, accept_custom: bool) -> list[dict[str, Any]]:
    choices = []
    for offer in plan["environment_mutations"]:
        custom = any(
            item["distribution"] == "requests"
            for alternative in offer["alternatives"]
            for item in alternative["package_mutations"]
        )
        choices.append(
            {
                "environment_id": offer["environment_id"],
                "alternative_id": offer["preferred_alternative_id"],
                "accept_package_changes": accept_custom or not custom,
            }
        )
    return choices


def _plan_confirm(
    daemon: EngineHostDaemon,
    *,
    token: str,
    definition: dict[str, Any],
    prefix: str,
    seq: int,
    accept_custom: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = _start(
        daemon,
        seq=seq,
        command="toolbox-plan-definition",
        payload={"request_id": f"{prefix}-plan", "definition": definition},
        token=token,
    )
    duplicate = _start(
        daemon,
        seq=seq + 1,
        command="toolbox-plan-definition",
        payload={"request_id": f"{prefix}-plan", "definition": definition},
        token=token,
    )
    assert duplicate["operation"] == started["operation"]
    planned = _terminal(daemon, started)
    assert planned["lifecycle"] == "terminal_success", json.dumps(planned, sort_keys=True)
    plan = dict(planned["result"])
    confirmed_started = _start(
        daemon,
        seq=seq + 2,
        command="toolbox-confirm-definition-plan",
        payload={
            "request_id": f"{prefix}-confirm",
            "plan_id": plan["plan_id"],
            "environment_choices": _choices(plan, accept_custom=accept_custom),
        },
        token=token,
    )
    confirmed = _terminal(daemon, confirmed_started)
    assert confirmed["lifecycle"] == "terminal_success", json.dumps(confirmed, sort_keys=True)
    return plan, dict(confirmed["result"])


def _apply(
    daemon: EngineHostDaemon,
    *,
    token: str,
    plan: dict[str, Any],
    confirmation: dict[str, Any],
    prefix: str,
    seq: int,
    approval_ref: str = "",
) -> dict[str, Any]:
    payload = {
        "request_id": f"{prefix}-apply",
        "plan_id": plan["plan_id"],
        "confirmation_ref": confirmation["confirmation_ref"],
    }
    if approval_ref:
        payload["dependency_approval_ref"] = approval_ref
    terminal = _terminal(
        daemon,
        _start(
            daemon,
            seq=seq,
            command="toolbox-apply-definition",
            payload=payload,
            token=token,
        ),
    )
    assert terminal["lifecycle"] == "terminal_success", json.dumps(terminal, sort_keys=True)
    return terminal


def _execute(
    daemon: EngineHostDaemon,
    *,
    token: str,
    tool_name: str,
    request_id: str,
    seq: int,
) -> dict[str, Any]:
    started = _dispatch(
        daemon,
        seq=seq,
        command="toolbox-execute",
        payload={
            "toolbox_id": "r7-e2e",
            "execution_request_id": request_id,
            "tool_call": {"id": request_id, "name": tool_name, "arguments": {}},
        },
        token=token,
    )
    duplicate = _dispatch(
        daemon,
        seq=seq + 1,
        command="toolbox-execute",
        payload={
            "toolbox_id": "r7-e2e",
            "execution_request_id": request_id,
            "tool_call": {"id": request_id, "name": tool_name, "arguments": {}},
        },
        token=token,
    )
    assert duplicate["operation"] == started["operation"]
    terminal = _terminal(daemon, started)
    assert terminal["lifecycle"] == "terminal_success", json.dumps(
        terminal, indent=2, sort_keys=True
    )
    return terminal


def test_r7_real_daemon_no_double_acceptance(tmp_path: Path) -> None:
    configuration = _configuration()
    source = tmp_path / "signed-source"
    public_keys = _write_signed_source(source, configuration)
    daemon = _daemon(
        tmp_path,
        configuration=configuration,
        source=source,
        public_keys=public_keys,
    )
    setup = _terminal(daemon, daemon.svc._toolbox_setup_operation, timeout=180)  # noqa: SLF001
    assert setup["lifecycle"] == "terminal_success", setup
    assert daemon.svc.hosting_setup_summary()["toolbox_readiness"]["status"] == "ready"
    tokens = _tokens(daemon)

    proposed = _definition(expected_revision=None, include_addon=True)
    declined_plan, declined = _plan_confirm(
        daemon,
        token=tokens["consumer"],
        definition=proposed,
        prefix="declined",
        seq=10,
        accept_custom=False,
    )
    assert any(
        item["tool_key"] == "r7.addon:Addon" and item["reason"] == "package_changes_declined"
        for item in declined["skipped_tools"]
    )
    assert "r7.base:Base" in declined["accepted_tool_keys"]
    declined_approval = _dispatch(
        daemon,
        seq=13,
        command="toolbox-approve-confirmed-definition-plan",
        payload={"confirmation_ref": declined["confirmation_ref"]},
        token=tokens["approver"],
    )
    _apply(
        daemon,
        token=tokens["consumer"],
        plan=declined_plan,
        confirmation=declined,
        prefix="declined",
        seq=14,
        approval_ref=declined_approval["approval_ref"],
    )
    base_result = _execute(
        daemon,
        token=tokens["consumer"],
        tool_name="Base",
        request_id="base-execution",
        seq=15,
    )
    assert "base-ok" in json.dumps(base_result["result"])

    current = daemon.svc.toolbox_get_definition(toolbox_id="r7-e2e")["active_revision"]
    accepted_plan, accepted = _plan_confirm(
        daemon,
        token=tokens["consumer"],
        definition=_definition(expected_revision=current, include_addon=True),
        prefix="accepted",
        seq=20,
        accept_custom=True,
    )
    assert accepted["dependency_approval_required"] is True
    approval = _dispatch(
        daemon,
        seq=23,
        command="toolbox-approve-confirmed-definition-plan",
        payload={"confirmation_ref": accepted["confirmation_ref"]},
        token=tokens["approver"],
    )
    _apply(
        daemon,
        token=tokens["consumer"],
        plan=accepted_plan,
        confirmation=accepted,
        prefix="accepted",
        seq=24,
        approval_ref=approval["approval_ref"],
    )
    addon_result = _execute(
        daemon,
        token=tokens["consumer"],
        tool_name="Addon",
        request_id="addon-execution",
        seq=25,
    )
    assert addon_result["result"]["user_projection"]["code"] == "toolbox_execution_succeeded"

    active_snapshot = daemon.svc._toolbox_state_v2.get("r7-e2e")  # noqa: SLF001
    addon_profile_id = active_snapshot["tool_routes"]["Addon"]["profile_id"]
    addon_reference = active_snapshot["profiles"][addon_profile_id]["environment_reference"]
    references = daemon.svc._environment_manager.list_references(limit=500)["references"]  # noqa: SLF001
    custom_environment = next(
        row["environment_id"]
        for row in references
        if row["reference_id"] == addon_reference
    )
    current = active_snapshot["active_revision"]
    removal_plan, removal = _plan_confirm(
        daemon,
        token=tokens["consumer"],
        definition=_definition(expected_revision=current, include_addon=False),
        prefix="remove",
        seq=30,
        accept_custom=True,
    )
    _apply(
        daemon,
        token=tokens["consumer"],
        plan=removal_plan,
        confirmation=removal,
        prefix="remove",
        seq=33,
    )
    assert daemon.svc.toolbox_get_definition(toolbox_id="r7-e2e")["active_tools"] == ["Base"]

    guarded_remove = _terminal(
        daemon,
        _start(
            daemon,
            seq=34,
            command="toolbox-environment-remove",
            payload={"environment_id": custom_environment, "request_id": "remove-custom-environment"},
            token=tokens["administrator"],
        ),
    )
    assert guarded_remove["lifecycle"] in {"terminal_success", "terminal_failure"}
    if guarded_remove["lifecycle"] == "terminal_failure":
        assert guarded_remove["result"]["code"] == "environment_remove_referenced"

    semantic_revision = daemon.svc.toolbox_get_definition(toolbox_id="r7-e2e")["active_revision"]
    for registration in daemon.svc._toolbox_executor_registrations("r7-e2e"):  # noqa: SLF001
        daemon.svc.shutdown(registration["engine_id"], timeout_seconds=5)
    daemon.svc.close()

    restarted = _daemon(
        tmp_path,
        configuration=configuration,
        source=source,
        public_keys=public_keys,
    )
    restarted_setup = _terminal(
        restarted, restarted.svc._toolbox_setup_operation, timeout=180  # noqa: SLF001
    )
    assert restarted_setup["lifecycle"] == "terminal_success", restarted_setup
    restarted_tokens = _tokens(restarted)
    healed_plan, healed = _plan_confirm(
        restarted,
        token=restarted_tokens["consumer"],
        definition=_definition(expected_revision=semantic_revision, include_addon=False),
        prefix="heal",
        seq=40,
        accept_custom=True,
    )
    healed_apply = _apply(
        restarted,
        token=restarted_tokens["consumer"],
        plan=healed_plan,
        confirmation=healed,
        prefix="heal",
        seq=43,
    )
    assert healed_apply["result"]["active_revision"] == semantic_revision
    assert restarted.svc._toolbox_executor_registrations("r7-e2e")  # noqa: SLF001

    gc_terminal = _terminal(
        restarted,
        _start(
            restarted,
            seq=44,
            command="hosting-gc",
            payload={"request_id": "r7-gc"},
            token=restarted_tokens["administrator"],
        ),
    )
    assert gc_terminal["lifecycle"] == "terminal_success", gc_terminal
    recovered = restarted.svc.hosted_operation_resolve_request(
        execution_kind="toolbox_maintenance",
        selector={"kind": "host_scope", "id": "toolbox-host"},
        request_id="r7-gc",
        owner_actor_id="key:administrator",
    )
    assert recovered["operation"] == gc_terminal["operation"]
    restarted.svc.close()
