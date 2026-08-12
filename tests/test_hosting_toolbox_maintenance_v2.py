from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from hosting import engine_host_cli
from hosting.daemon import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel
from hosting.service.host_service import EngineHostService
from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration
from tests.hosting_v3_fixtures import hosting_configuration, write_hosting_configuration
from hosting.toolbox.bundle_models import (
    ResolvedToolboxProfileSpec,
    ToolboxBundleAutoTool,
    ToolboxBundleFile,
    ToolboxBundleSpec,
    ToolboxDefinitionSpec,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _install_active(service: EngineHostService, *, include_registration: bool = True) -> str:
    definition = ToolboxDefinitionSpec.from_dict(
        {
            "contract": "hosting.toolbox.definition",
            "toolbox_id": "demo",
            "expected_revision": None,
            "auto_requests": [
                {
                    "files": [{"relative_path": "demo.py", "content": "def Alpha(): return 1\n"}],
                    "module_name": "demo",
                    "callable_name": "Alpha",
                    "dependency": {
                        "mode": "auto",
                        "template_id": None,
                        "declared_imports": [],
                        "package_requirements": [],
                    },
                    "sandbox_policy": {},
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
            "intrinsics": {"names": [], "include_guides": False, "sandbox_policy": {}},
        }
    )
    profile = ResolvedToolboxProfileSpec(
        environment_key=_digest("a"),
        template_id="core",
        template_lock_digest=_digest("b"),
        custom_resolved_lock_digest=None,
        sandbox_policy={},
        assigned_tool_keys=("auto:demo:Alpha",),
        resolved_import_roots=(),
    )
    bundle = ToolboxBundleSpec(
        bundle_id="demo-alpha",
        toolbox_id="demo",
        files=[ToolboxBundleFile(relative_path="demo.py", content="def Alpha(): return 1\n")],
        auto_tools=[ToolboxBundleAutoTool(module_name="demo", callable_name="Alpha")],
        dependency_lock_hash=profile.effective_lock_digest,
        resolved_profile=profile,
    )
    manifest_hash = bundle.manifest_payload()["manifest_hash"]
    if not manifest_hash.startswith("sha256:"):
        manifest_hash = f"sha256:{manifest_hash}"
    engine_id = "active-alpha"
    service._toolbox_state_v2.publish(
        toolbox_id="demo",
        expected_revision=None,
        definition=definition.to_dict(),
        profiles={
            profile.profile_id: {
                "profile": profile.to_dict(),
                "manifest_hash": manifest_hash,
                "engine_id": engine_id,
                "tool_names": ["Alpha"],
                "environment_reference": f"toolbox:demo:{profile.profile_id}:{definition.revision}",
                "resolved_environment": {},
            }
        },
        tool_routes={
            "Alpha": {
                "profile_id": profile.profile_id,
                "engine_id": engine_id,
                "non_restartable": False,
            }
        },
        environment_references=[f"toolbox:demo:{profile.profile_id}:{definition.revision}"],
        published_at_ms=1,
    )
    if include_registration:
        service.register_spawned(
            engine_id=engine_id,
            pid=1234,
            command=["python", "worker.py"],
            executor_kind="toolbox_executor",
            routing_state="active",
            bundle={
                "toolbox_id": "demo",
                "sandbox_profile_id": profile.profile_id,
                "resolved_profile_id": profile.profile_id,
                "manifest_hash": manifest_hash,
                "definition_revision": definition.revision,
            },
            environment={"environment_key": profile.environment_key},
            tool_access={"allowed_tool_names": ["Alpha"]},
        )
    return engine_id


def _service(tmp_path: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=tmp_path / "managed.json",
        hosting_configuration=hosting_configuration(tmp_path),
    )


def _wait_maintenance(service: EngineHostService, started: dict) -> dict:
    return service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"], timeout_seconds=10
    )


def _configuration(*, protected: tuple[str, ...] = ()) -> dict:
    return {
        "builtins": [{
            "template_id": "core",
            "imports": ["packaging"],
            "package_requirements": [],
            "sandbox_policy": "compute-only",
            "required": True,
            "prewarm": False,
            "provenance": "maintenance-test",
        }],
        "sources": [{
            "source_id": "release",
            "kind": "airgap_store",
            "origin": "airgap://release",
            "credential_ref": None,
            "allowed_package_namespaces": ["*"],
            "priority": 1,
            "trust_key_ids": ["release-key"],
            "maximum_download_bytes": 1024 * 1024,
        }],
        "resolution": {
            "mode": "air_gapped",
            "timeout_seconds": 60,
            "maximum_bytes": 1024 * 1024,
            "maximum_artifacts": 16,
            "allowed_redirect_origins": [],
            "wheel_only": True,
        },
        "retention": {
            "artifact_cache_grace_seconds": 60,
            "maximum_cache_bytes": 1024 * 1024,
            "maximum_cache_artifacts": 16,
            "protected_digests": list(protected),
            "remove_unreferenced_custom_revisions_on_apply": False,
        },
    }


class _RemovalManager:
    def __init__(self, *, references: dict[str, dict[str, int]] | None = None, result: str = "removed"):
        self.references = {str(key): dict(value) for key, value in dict(references or {}).items()}
        self.result = result
        self.removed: list[str] = []

    def list_references(self, *, cursor: str = "", limit: int = 500):
        rows = [
            {
                "contract": "hosting.environment_reference.v1",
                "reference_id": reference_id,
                "environment_id": environment_id,
                "consumer_kind": "toolbox",
                "consumer_id": "demo",
                "revision": 1,
                "acquired_at_ms": acquired_at_ms,
                "released_at_ms": None,
            }
            for environment_id, references in self.references.items()
            for reference_id, acquired_at_ms in references.items()
        ]
        return {"references": rows, "next_cursor": None}

    def remove(self, *, environment_id: str) -> dict:
        if self.references.get(environment_id):
            raise RuntimeError("environment_references_present")
        self.removed.append(environment_id)
        return {"environment_id": environment_id, "state": self.result}

    def gc(self) -> dict:
        return {"removed_environment_ids": [], "removed_count": 0}


def _configured_service(tmp_path: Path, *, protected: tuple[str, ...] = ()) -> EngineHostService:
    service = EngineHostService(
        engines_state_file=tmp_path / "managed.json",
        hosting_configuration=hosting_configuration(tmp_path),
    )
    service._toolbox_host_project_config = ToolboxHostProjectConfiguration.from_dict(  # noqa: SLF001
        _configuration(protected=protected)
    )
    return service


def _wait_remove(service: EngineHostService, started: dict) -> dict:
    return service._hosted_operations.wait_for_terminal(  # noqa: SLF001
        operation_id=started["operation"]["operation_id"], timeout_seconds=5
    )


def test_environment_remove_is_durable_idempotent_and_reports_progress(tmp_path: Path) -> None:
    service = _configured_service(tmp_path)
    builder = _RemovalManager()
    service._environment_manager_instance = builder  # type: ignore[attr-defined]
    digest = _digest("d")

    first = service.toolbox_environment_remove(
        environment_digest=digest, request_id="remove-one", owner_actor_id="admin:a"
    )
    second = service.toolbox_environment_remove(
        environment_digest=digest, request_id="remove-one", owner_actor_id="admin:a"
    )
    terminal = _wait_remove(service, first)

    assert first["operation"] == second["operation"]
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["status"] == "removed"
    assert terminal["result"]["environment_digest"] == digest
    assert terminal["progress"]["phase"] == "cleanup"
    assert builder.removed == [digest]


def test_environment_remove_blocks_active_candidate_and_builder_references(tmp_path: Path) -> None:
    service = _configured_service(tmp_path)
    digest = _digest("a")
    builder = _RemovalManager(references={digest: {"candidate:one": 1}})
    service._environment_manager_instance = builder  # type: ignore[attr-defined]
    service.register_spawned(
        engine_id="candidate-one",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        routing_state="candidate",
        bundle={"toolbox_id": "demo"},
        environment={"environment_key": digest},
        tool_access={"allowed_tool_names": []},
    )

    started = service.toolbox_environment_remove(
        environment_digest=digest, request_id="remove-blocked", owner_actor_id="admin:a"
    )
    terminal = _wait_remove(service, started)

    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["status"] == "blocked"
    assert terminal["result"]["blocking_reference_kinds"] == ["candidate", "reference"]
    assert builder.removed == []


def test_environment_remove_blocks_protected_digest_and_reports_already_absent(tmp_path: Path) -> None:
    protected = _digest("b")
    service = _configured_service(tmp_path, protected=(protected,))
    service._environment_manager_instance = _RemovalManager(result="already_absent")  # type: ignore[attr-defined]

    blocked = _wait_remove(
        service,
        service.toolbox_environment_remove(
            environment_digest=protected, request_id="remove-protected", owner_actor_id="admin:a"
        ),
    )
    absent_digest = _digest("c")
    absent = _wait_remove(
        service,
        service.toolbox_environment_remove(
            environment_digest=absent_digest, request_id="remove-absent", owner_actor_id="admin:a"
        ),
    )

    assert blocked["result"]["status"] == "blocked"
    assert blocked["result"]["blocking_reference_kinds"] == ["protected"]
    assert absent["result"]["status"] == "already_absent"


def test_environment_remove_is_admin_only() -> None:
    command = "environment-remove"
    assert command in EngineHostService._commands_allowed_for_role("admin")  # noqa: SLF001
    for role in ("config_editor", "worker_user", "diagnostic_user", "dependency_approver"):
        assert command not in EngineHostService._commands_allowed_for_role(role)  # noqa: SLF001


def test_environment_remove_checks_unexpired_plan_confirmation_and_operation_references(
    tmp_path: Path, monkeypatch
) -> None:
    service = _configured_service(tmp_path)
    digest = _digest("d")
    service._environment_manager_instance = _RemovalManager()  # type: ignore[attr-defined]

    class _Plan:
        def to_dict(self):
            return {"environment_key": digest}

    monkeypatch.setattr(
        service._toolbox_definition_plans,
        "list",
        lambda *, now_ms: (_Plan(),),
    )
    monkeypatch.setattr(
        service._toolbox_confirmations,
        "_read",
        lambda: {"receipts": {"receipt": {"expires_at_ms": 9_999_999_999_999, "environment_key": digest}}},
    )
    monkeypatch.setattr(
        service._hosted_operations,
        "active_records",
        lambda **kwargs: [{
            "operation": {"operation_id": "other-operation"},
            "metadata": {"environment_digest": digest},
        }],
    )

    terminal = _wait_remove(
        service,
        service.toolbox_environment_remove(
            environment_digest=digest, request_id="remove-persisted-blockers", owner_actor_id="admin:a"
        ),
    )

    assert terminal["result"]["blocking_reference_kinds"] == ["plan", "confirmation", "operation"]


def test_route_based_references_consistency_and_review(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_active(service)

    references = service.toolbox_references()
    consistency = service.toolbox_consistency()
    review = service.toolbox_review_snapshot(toolbox_ids=["demo"])

    assert references["contract"] == "hosting.toolbox.references.v2"
    assert references["summary"]["active_registration_count"] == 1
    assert consistency["consistent"] is True
    assert review["recommended_action"] == "observe"
    assert review["toolboxes"]["demo"]["tool_names"] == ["Alpha"]


def test_missing_active_registration_requires_definition_reapply(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _install_active(service, include_registration=False)

    consistency = service.toolbox_consistency()
    repair = service._toolbox_repair_now(toolbox_ids=["demo"])  # noqa: SLF001

    assert consistency["issues"][0]["issue"] == "missing_active_registration"
    assert repair["reapply_required"][0]["issue"] == "definition_reapply_required"
    assert repair["repaired_toolbox_ids"] == []


def test_repair_restores_route_state_from_active_snapshot(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    engine_id = _install_active(service)
    service.set_toolbox_registration_routing_states({engine_id: "retired"})
    monkeypatch.setattr(service, "recover_toolbox_definition_rollouts", lambda: {"status": "ok"})

    repair = service._toolbox_repair_now(toolbox_ids=["demo"])  # noqa: SLF001

    assert repair["reactivated_engine_ids"] == [engine_id]
    assert service.get_registration(engine_id)["routing_state"] == "active"
    assert service.toolbox_consistency()["consistent"] is True


def test_gc_removes_only_unreferenced_candidate_and_retired_workers(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path)
    active = _install_active(service)
    for engine_id, state in (("candidate-orphan", "candidate"), ("retired-old", "retired")):
        service.register_spawned(
            engine_id=engine_id,
            pid=2345,
            command=["python", "worker.py"],
            executor_kind="toolbox_executor",
            routing_state=state,
            bundle={"toolbox_id": "demo", "sandbox_profile_id": "old"},
            environment={},
            tool_access={"allowed_tool_names": ["Old"]},
        )
    removed: list[str] = []
    monkeypatch.setattr(service, "recover_toolbox_definition_rollouts", lambda: {"status": "ok"})
    monkeypatch.setattr(service, "_retire_toolbox_registration", removed.append)

    result = service._toolbox_gc_now()  # noqa: SLF001

    assert result["removed_engine_ids"] == ["candidate-orphan", "retired-old"]
    assert removed == ["candidate-orphan", "retired-old"]
    assert active not in removed


def test_gc_removes_orphaned_bundle_but_preserves_active_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path)
    active_engine = _install_active(service)
    bundles_root = service.hosting_root / "toolbox_bundles"
    active_root = bundles_root / "active"
    orphan_root = bundles_root / "orphan"
    active_root.mkdir(parents=True)
    orphan_root.mkdir(parents=True)
    (active_root / "manifest.json").write_text("{}", encoding="utf-8")
    (orphan_root / "manifest.json").write_text("{}", encoding="utf-8")
    registration = service.get_registration(active_engine)
    registration["bundle"]["bundle_root"] = str(active_root)
    service._write_engines(
        [registration if row["engine_id"] == active_engine else row for row in service._read_engines()]
    )
    monkeypatch.setattr(service, "recover_toolbox_definition_rollouts", lambda: {"status": "ok"})

    result = service._toolbox_gc_now()  # noqa: SLF001

    assert result["removed_bundle_roots"] == [str(orphan_root.resolve())]
    assert active_root.is_dir()
    assert not orphan_root.exists()


def test_hosted_maintenance_is_durable_idempotent_and_recoverable(tmp_path: Path) -> None:
    service = _service(tmp_path)
    started = service.toolbox_gc(request_id="gc-1", owner_actor_id="admin:test")
    duplicate = service.toolbox_gc(request_id="gc-1", owner_actor_id="admin:test")
    assert duplicate["operation"]["operation_id"] == started["operation"]["operation_id"]
    terminal = _wait_maintenance(service, started)
    recovered = service.hosted_operation_resolve_request(
        execution_kind="toolbox_maintenance",
        selector={"kind": "host_scope", "id": "toolbox-host"},
        request_id="gc-1",
        owner_actor_id="admin:test",
    )
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["operation"]["execution_kind"] == "toolbox_maintenance"
    assert terminal["operation"]["selector"] == {"kind": "host_scope", "id": "toolbox-host"}
    assert terminal["result"]["action"] == "gc"
    assert terminal["result"]["maintenance_result"]["contract"] == "hosting.toolbox.gc.v2"
    assert terminal["progress"]["phase"] == "cleanup"
    assert recovered["operation"]["operation_id"] == started["operation"]["operation_id"]


def test_hosted_maintenance_request_conflict_and_pre_dispatch_cancel_are_immediate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(threading.Thread, "start", lambda _self: None)
    started = service.toolbox_repair(
        request_id="maintenance-1",
        toolbox_ids=["demo"],
        owner_actor_id="admin:test",
        apply=True,
        mutation_authorized=True,
    )
    conflict = service.toolbox_repair(
        request_id="maintenance-1",
        toolbox_ids=["other"],
        owner_actor_id="admin:test",
        apply=True,
        mutation_authorized=True,
    )
    canceled = service.hosted_operation_cancel(
        ref=started["operation"],
        owner_actor_id="admin:test",
        reason="test_cancel",
    )
    assert conflict["lifecycle"] == "idempotency_conflict"
    assert canceled["lifecycle"] == "terminal_cancellation"
    assert canceled["result"]["code"] == "toolbox_maintenance_canceled_before_mutation"


def test_interrupted_maintenance_reuses_the_same_operation_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    real_start = threading.Thread.start
    monkeypatch.setattr(threading.Thread, "start", lambda _self: None)
    started = service.toolbox_gc(request_id="gc-restart", owner_actor_id="admin:test")
    operation_id = started["operation"]["operation_id"]
    service._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)  # noqa: SLF001
    repository_path = str((service.hosting_root / "state" / "hosted_operations.json").resolve())
    service.close()
    EngineHostService._operation_repositories.pop(repository_path, None)  # noqa: SLF001
    restarted = _service(tmp_path)
    monkeypatch.setattr(threading.Thread, "start", real_start)

    resumed = restarted.toolbox_gc(
        request_id="gc-restart", owner_actor_id="admin:test"
    )
    terminal = _wait_maintenance(restarted, resumed)
    assert resumed["operation"]["operation_id"] == operation_id
    assert terminal["lifecycle"] == "terminal_success"
    assert terminal["result"]["action"] == "gc"


class _MaintenanceConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, command: str, payload=None):
        self.calls.append((command, dict(payload or {})))
        return {}

    def is_alive(self) -> bool:
        return True

    def close(self) -> None:
        return None


def test_channel_and_daemon_require_op_start_for_mutating_maintenance(tmp_path: Path) -> None:
    connection = _MaintenanceConnection()
    channel = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    channel._get_connection = lambda: connection  # type: ignore[method-assign]
    channel.toolbox_gc(request_id="gc-channel")
    channel.toolbox_repair(request_id="repair-channel", toolbox_ids=["demo"])
    channel.toolbox_reconcile(request_id="reconcile-channel", details=True)
    assert [item[0] for item in connection.calls] == ["op-start", "op-start", "op-start"]
    assert [item[1]["command"] for item in connection.calls] == [
        "toolbox-gc", "toolbox-repair", "toolbox-reconcile"
    ]

    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "daemon-engines.json",
        mp13_config_file=write_hosting_configuration(tmp_path),
    )
    raw = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            json.dumps({"seq": 1, "cmd": "toolbox-gc", "payload": {"request_id": "raw"}}),
            peer_host="127.0.0.1",
            transport="local_ipc",
        )
    )
    assert raw["ok"] is False
    assert raw["error_code"] == "operation_wrapper_required"
    wrapped = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            json.dumps(
                {
                    "seq": 2,
                    "cmd": "op-start",
                    "payload": {
                        "command": "toolbox-gc",
                        "payload": {"request_id": "wrapped-gc"},
                    },
                }
            ),
            peer_host="127.0.0.1",
            transport="local_ipc",
        )
    )
    assert wrapped["ok"] is True
    assert wrapped["result"]["operation"]["execution_kind"] == "toolbox_maintenance"
    assert _wait_maintenance(daemon.svc, wrapped["result"])["lifecycle"] == "terminal_success"


def test_remote_cli_wraps_mutating_maintenance_in_op_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeRemoteChannel:
        def __init__(self, _settings=None):
            pass

        def invoke_control_command(self, command: str, payload=None):
            calls.append((command, dict(payload or {})))
            return {"status": "ok"}

    monkeypatch.setattr(
        "hosting.engine_host_channel.EngineHostControlChannel", FakeRemoteChannel
    )
    monkeypatch.setattr("sys.stdin.read", lambda: json.dumps({"request_id": "gc-cli"}))
    assert engine_host_cli.main(
        ["--ssh-target", "admin@example.test", "--payload-stdin", "toolbox-gc"]
    ) == 0
    assert calls == [
        (
            "op-start",
            {"command": "toolbox-gc", "payload": {"request_id": "gc-cli"}},
        )
    ]
    assert '"ok": true' in capsys.readouterr().out
