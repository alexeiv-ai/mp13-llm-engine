from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from hosting import engine_host_cli
from hosting.daemon import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel
from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_catalog import (
    AtomicJsonToolboxTemplateCatalog,
    ToolboxTemplateArtifactReference,
)
from hosting.toolbox.catalog import ToolboxEnvironmentTemplateSpec
from tests.hosting_v3_fixtures import hosting_configuration, write_hosting_configuration


ROOT = Path(__file__).resolve().parents[1]
SIGNATURE = "A" * 86


def _digest(char: str) -> str:
    return f"sha256:{char * 64}"


def _template_payload(
    template_id: str = "core", *, revision: str = "1", lock_char: str = "a"
) -> dict[str, Any]:
    return {
        "template_id": template_id,
        "python_requires": ">=3.12,<3.13",
        "python_abis": ["cp312"],
        "runtime_kind": "toolbox_python",
        "worker_protocol_version": "1.0",
        "platforms": ["win_amd64"],
        "locked_distributions": [
            {"name": "hosting-runtime", "version": "1.0", "extras": []}
        ],
        "exposed_import_roots": ["hosting", "mp13_engine"],
        "lock_digest": _digest(lock_char),
        "parent_worker_artifact_digest": _digest("b"),
        "isolation_policy_version": "1.0",
        "provenance": {
            "source": "release",
            "revision": revision,
            "evidence_digest": _digest(revision[-1]),
            "verifier_id": "release-key-1",
        },
    }


def _artifact(char: str = "e") -> ToolboxTemplateArtifactReference:
    return ToolboxTemplateArtifactReference(
        source_id="release-artifacts",
        filename=f"hosting-runtime-{char}.whl",
        sha256=_digest(char),
        size_bytes=1234,
    )


def _environment_template(revision: int = 1) -> dict[str, Any]:
    return {
        "contract": "hosting.environment_template.v1",
        "template_id": "core",
        "revision": revision,
        "runtime_kind": "python",
        "builder_id": "python-manifest-v1",
        "package_lock_id": "core-lock",
        "platforms": ["win_amd64"],
        "state": "draft",
    }


def _repo(tmp_path: Path) -> AtomicJsonToolboxTemplateCatalog:
    return AtomicJsonToolboxTemplateCatalog(tmp_path / "catalog.json", clock=lambda: 1000.0)


def _publish(
    repo: AtomicJsonToolboxTemplateCatalog,
    payload: dict[str, Any] | None = None,
    *,
    artifact: ToolboxTemplateArtifactReference | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    published = repo.publish_inactive(
        template=ToolboxEnvironmentTemplateSpec.from_dict(payload or _template_payload()),
        artifacts=(artifact or _artifact(),),
        verification_evidence=SIGNATURE,
        actor_id="admin:test",
    )
    if activate:
        current = repo.read()["active"].get(published["template_id"])
        activated = (
            repo.replace(
                template_id=published["template_id"],
                expected_active_digest=current,
                replacement_digest=published["template_digest"],
                actor_id="admin:test",
            )
            if current is not None and current != published["template_digest"]
            else repo.activate(
                template_id=published["template_id"],
                template_digest=published["template_digest"],
                actor_id="admin:test",
            )
        )
        return {**published, **activated, "template_digest": published["template_digest"]}
    return published


def test_catalog_publishes_generic_template_without_verification_evidence(
    tmp_path: Path,
) -> None:
    payload = _template_payload()
    payload["provenance"]["verifier_id"] = None
    published = _repo(tmp_path).publish_inactive(
        template=ToolboxEnvironmentTemplateSpec.from_dict(payload),
        artifacts=(_artifact(),),
        actor_id="admin:test",
    )
    entry = _repo(tmp_path).read()["entries"][0]
    assert entry["template_digest"] == published["template_digest"]
    assert entry["verification_evidence"] is None


def test_artifact_reference_and_signature_are_strict() -> None:
    assert ToolboxTemplateArtifactReference.from_dict(_artifact().to_dict()) == _artifact()
    row = _artifact().to_dict()
    row["path"] = "C:/secret"
    with pytest.raises(ValueError, match="unknown_fields"):
        ToolboxTemplateArtifactReference.from_dict(row)
    with pytest.raises(ValueError, match="filename_invalid"):
        ToolboxTemplateArtifactReference(
            source_id="source",
            filename="../artifact.whl",
            sha256=_digest("a"),
            size_bytes=1,
        )


def test_publish_is_idempotent_immutable_and_restart_safe(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _publish(repo)
    second = _publish(repo)
    assert first["template_digest"] == second["template_digest"]
    assert second["outcome"] == "idempotent"
    restarted = _repo(tmp_path).read()
    assert len(restarted["entries"]) == 1
    assert restarted["active"] == {"core": first["template_digest"]}
    with pytest.raises(ValueError, match="immutable_publish_conflict"):
        _publish(repo, artifact=_artifact("f"))


def test_multiple_revisions_and_lifecycle_update_active_pointer(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _publish(repo)
    second = _publish(
        repo,
        _template_payload(revision="2", lock_char="c"),
        artifact=_artifact("f"),
    )
    state = repo.read()
    assert len(state["entries"]) == 2
    assert state["active"] == {"core": second["template_digest"]}
    deprecated = repo.set_lifecycle(
        template_id="core",
        template_digest=second["template_digest"],
        lifecycle="deprecated",
        actor_id="admin:test",
    )
    assert deprecated["entry"]["lifecycle"] == "deprecated"
    assert repo.read()["active"] == {}
    revoked = repo.set_lifecycle(
        template_id="core",
        template_digest=first["template_digest"],
        lifecycle="revoked",
        actor_id="admin:test",
    )
    assert revoked["entry"]["lifecycle"] == "revoked"
    with pytest.raises(ValueError, match="transition_invalid"):
        repo.set_lifecycle(
            template_id="core",
            template_digest=first["template_digest"],
            lifecycle="deprecated",
            actor_id="admin:test",
        )


def test_corrupt_or_digest_mismatched_state_fails_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    payload = json.loads(repo.path.read_text(encoding="utf-8"))
    payload["entries"][0]["template"]["runtime_kind"] = "model"
    repo.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        repo.read()
    repo.path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="state_corrupt"):
        repo.read()


def test_consumer_projection_is_bounded_and_audit_is_redacted(tmp_path: Path) -> None:
    svc = EngineHostService(
        hosting_configuration=hosting_configuration(tmp_path),
    )
    published = svc._toolbox_template_catalog.publish_inactive(  # noqa: SLF001
        template=ToolboxEnvironmentTemplateSpec.from_dict(_template_payload()),
        artifacts=(_artifact(),),
        verification_evidence=SIGNATURE,
        actor_id="admin:test",
    )
    svc.toolbox_template_activate(
        template_id="core", template_digest=published["template_digest"], actor_id="admin:test"
    )
    listed = svc.toolbox_template_list()
    descriptor = listed["templates"][0]
    assert descriptor == svc.toolbox_template_describe(template_id="core")
    assert descriptor["template_digest"] == published["template_digest"]
    serialized = json.dumps(descriptor)
    for secret_field in [
        "locked_distributions",
        "artifacts",
        "verification_evidence",
        "published_by",
        "filename",
        "source_id",
    ]:
        assert secret_field not in serialized
    state = svc._toolbox_template_catalog.read()  # noqa: SLF001
    assert state["audit"][-1] == {
        "at_ms": state["audit"][-1]["at_ms"],
        "actor_id": "admin:test",
        "action": "activate",
        "template_id": "core",
        "template_digest": published["template_digest"],
        "outcome": "activated",
    }
    assert SIGNATURE not in json.dumps(state["audit"])


def test_service_describe_requires_active_or_exact_revision(tmp_path: Path) -> None:
    svc = EngineHostService(
        hosting_configuration=hosting_configuration(tmp_path),
    )
    published = svc._toolbox_template_catalog.publish_inactive(  # noqa: SLF001
        template=ToolboxEnvironmentTemplateSpec.from_dict(_template_payload()),
        artifacts=(_artifact(),),
        verification_evidence=SIGNATURE,
        actor_id="admin:test",
    )
    with pytest.raises(ValueError, match="active_revision_not_found"):
        svc.toolbox_template_describe(template_id="core")
    exact = svc.toolbox_template_describe(
        template_id="core", template_digest=published["template_digest"]
    )
    assert exact["active_revision"] is False


def test_multi_process_publish_preserves_both_revisions(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    script = r"""
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from hosting.service.toolbox_catalog import AtomicJsonToolboxTemplateCatalog, ToolboxTemplateArtifactReference
from hosting.toolbox.catalog import ToolboxEnvironmentTemplateSpec
def digest(c): return 'sha256:' + c * 64
name, char, state_path = sys.argv[1:]
template = ToolboxEnvironmentTemplateSpec.from_dict({
 'template_id': name, 'python_requires': '>=3.12,<3.13', 'python_abis': ['cp312'],
 'runtime_kind': 'toolbox_python', 'worker_protocol_version': '1.0', 'platforms': ['win_amd64'],
 'locked_distributions': [{'name': 'hosting-runtime', 'version': '1.0', 'extras': []}],
 'exposed_import_roots': ['hosting'], 'lock_digest': digest(char),
 'parent_worker_artifact_digest': digest('b'), 'isolation_policy_version': '1.0',
 'provenance': {'source': 'test', 'revision': char, 'evidence_digest': digest(char), 'verifier_id': 'key-1'}})
artifact = ToolboxTemplateArtifactReference(source_id='source', filename=name + '.whl', sha256=digest(char), size_bytes=10)
repo = AtomicJsonToolboxTemplateCatalog(Path(state_path))
published = repo.publish_inactive(template=template, artifacts=(artifact,), verification_evidence='A'*86, actor_id='admin:test')
repo.activate(template_id=name, template_digest=published['template_digest'], actor_id='admin:test')
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, name, char, str(path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for name, char in [("core", "a"), ("py-compute", "c")]
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, f"{stdout}\n{stderr}"
    state = AtomicJsonToolboxTemplateCatalog(path).read()
    assert {item["template_id"] for item in state["entries"]} == {"core", "py-compute"}


class _FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def invoke(self, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((command, dict(payload or {})))
        return {}

    def is_alive(self) -> bool:
        return True

    def close(self) -> None:
        return None


def test_channel_forwards_exact_catalog_payloads() -> None:
    connection = _FakeConnection()
    channel = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    channel._get_connection = lambda: connection  # type: ignore[method-assign]
    channel.set_session_token("token-1")
    channel.environment_template_list()
    channel.environment_template_describe(template_id="core", revision=1)
    channel.environment_template_construct(template=_environment_template())
    channel.environment_template_activate(template_id="core", revision=1)
    channel.environment_template_replace(template=_environment_template(revision=2))
    channel.environment_template_deprecate(template_id="core", revision=1)
    channel.environment_template_revoke(template_id="core", revision=1)
    assert [item[0] for item in connection.calls] == [
        "environment-template-list",
        "environment-template-describe",
        "environment-template-construct",
        "environment-template-activate",
        "environment-template-replace",
        "environment-template-deprecate",
        "environment-template-revoke",
    ]
    assert all(payload["session_token"] == "token-1" for _, payload in connection.calls)
    assert connection.calls[2][1]["template"] == _environment_template()


def test_role_separation_allows_consumer_reads_and_admin_mutation() -> None:
    for role in ["worker_user", "config_editor", "diagnostic_user"]:
        allowed = EngineHostService._commands_allowed_for_role(role)  # noqa: SLF001
        assert {"environment-template-list", "environment-template-describe"} <= allowed
        assert "environment-template-construct" not in allowed
        assert "environment-template-activate" not in allowed
        assert "environment-template-replace" not in allowed
        assert "environment-template-deprecate" not in allowed
        assert "environment-template-revoke" not in allowed
        assert "environment-template-prewarm" not in allowed
    admin = EngineHostService._commands_allowed_for_role("admin")  # noqa: SLF001
    assert {
        "environment-template-list",
        "environment-template-describe",
        "environment-template-construct",
        "environment-template-activate",
        "environment-template-replace",
        "environment-template-deprecate",
        "environment-template-revoke",
        "environment-template-prewarm",
    } <= admin


def test_authenticated_command_policy_enforces_catalog_role_separation(tmp_path: Path) -> None:
    svc = EngineHostService(
        hosting_configuration=hosting_configuration(tmp_path, require_auth=True),
    )
    svc.auth_upsert_key(
        key_id="admin", key_secret="admin-secret", role="admin", auth_method="shared_secret"
    )
    svc.auth_upsert_key(
        key_id="worker",
        key_secret="worker-secret",
        role="worker_user",
        auth_method="shared_secret",
    )
    worker = svc.auth_issue_session(
        key_id="worker", key_secret="worker-secret", scope="control"
    )["token"]
    admin = svc.auth_issue_session(
        key_id="admin", key_secret="admin-secret", scope="control"
    )["token"]
    svc.authorize_command("environment-template-list", {"session_token": worker})
    svc.authorize_command("environment-template-describe", {"session_token": worker})
    with pytest.raises(PermissionError, match="insufficient_role"):
        svc.authorize_command("environment-template-construct", {"session_token": worker})
    svc.authorize_command("environment-template-construct", {"session_token": admin})
    svc.authorize_command("environment-template-activate", {"session_token": admin})
    svc.authorize_command("environment-template-replace", {"session_token": admin})
    svc.authorize_command("environment-template-deprecate", {"session_token": admin})
    svc.authorize_command("environment-template-revoke", {"session_token": admin})


def test_daemon_dispatch_and_remote_cli_route_catalog_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        mp13_config_file=write_hosting_configuration(tmp_path),
    )
    response = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            json.dumps(
                {
                    "seq": 1,
                    "cmd": "environment-template-list",
                    "payload": {},
                }
            ),
            peer_host="127.0.0.1",
            transport="local_ipc",
        )
    )
    assert response["ok"] is True
    assert response["result"]["templates"] == []

    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeRemoteChannel:
        def __init__(self, _settings=None):
            pass

        def invoke_control_command(self, command: str, payload=None):
            calls.append((command, dict(payload or {})))
            return {"status": "ok"}

    monkeypatch.setattr("hosting.engine_host_channel.EngineHostControlChannel", FakeRemoteChannel)
    rc = engine_host_cli.main(
        ["--ssh-target", "user@example.test", "environment-template-list"]
    )
    assert rc == 0
    assert calls == [("environment-template-list", {})]
    assert '"ok": true' in capsys.readouterr().out


def test_daemon_op_start_routes_template_construction_to_canonical_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        mp13_config_file=write_hosting_configuration(tmp_path),
    )
    calls: list[dict[str, Any]] = []

    def construct(**kwargs):
        calls.append(dict(kwargs))
        return {"contract": "hosting.operation_status", "lifecycle": "queued"}

    monkeypatch.setattr(daemon.svc, "environment_template_construct", construct)
    response = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            json.dumps(
                {
                    "seq": 1,
                    "cmd": "op-start",
                    "payload": {
                        "command": "environment-template-construct",
                        "payload": {"template": _environment_template()},
                    },
                }
            ),
            peer_host="127.0.0.1",
            transport="local_ipc",
        )
    )
    assert response["ok"] is True
    assert calls[0]["template"] == _environment_template()


def test_activate_and_replace_are_exact_compare_and_swap_transitions(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _publish(repo, activate=False)
    activated = repo.activate(
        template_id="core",
        template_digest=first["template_digest"],
        actor_id="admin:test",
    )
    assert activated["entry"]["lifecycle"] == "active"
    second = _publish(
        repo,
        _template_payload(revision="2", lock_char="c"),
        artifact=_artifact("f"),
        activate=False,
    )
    with pytest.raises(ValueError, match="active_revision_conflict"):
        repo.replace(
            template_id="core",
            expected_active_digest=_digest("9"),
            replacement_digest=second["template_digest"],
            actor_id="admin:test",
        )
    replaced = repo.replace(
        template_id="core",
        expected_active_digest=first["template_digest"],
        replacement_digest=second["template_digest"],
        actor_id="admin:test",
    )
    state = repo.read()
    assert replaced["outcome"] == "replaced"
    assert state["active"] == {"core": second["template_digest"]}
    assert next(item for item in state["entries"] if item["template_digest"] == first["template_digest"])["lifecycle"] == "deprecated"
