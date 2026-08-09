from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from hosting.daemon.pidfile import DaemonPidFile
from hosting.engine_host_cli import main as engine_host_cli_main
from hosting.operation_contract import hosted_execution_fingerprint
from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_state_cutover import ToolboxStateArchiveError
from hosting.service.toolbox_state_v2 import (
    AtomicJsonToolboxStateV2Repository,
    LegacyToolboxStateError,
    ToolboxRevisionConflictError,
)
from hosting.toolbox.bundle_models import ResolvedToolboxProfileSpec, ToolboxDefinitionSpec


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _definition(tool_name: str, expected_revision: str | None = None) -> ToolboxDefinitionSpec:
    return ToolboxDefinitionSpec.from_dict(
        {
            "contract": "hosting.toolbox.definition",
            "toolbox_id": "demo",
            "expected_revision": expected_revision,
            "auto_requests": [
                {
                    "files": [{"relative_path": "pkg/tool.py", "content": f"def {tool_name}():\n    return 1\n"}],
                    "module_name": "pkg.tool",
                    "callable_name": tool_name,
                    "dependency": {
                        "mode": "auto",
                        "template_id": None,
                        "declared_imports": [],
                        "package_requirements": [],
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
    )


def _runtime(character: str, tool_name: str, engine_id: str):
    profile = ResolvedToolboxProfileSpec(
        environment_key=_digest(character),
        template_id="core",
        template_lock_digest=_digest(character),
        custom_resolved_lock_digest=None,
        sandbox_policy={"sandbox": {"enabled": True}},
        assigned_tool_keys=(f"auto:pkg.tool:{tool_name}",),
        resolved_import_roots=(),
    )
    profiles = {
        profile.profile_id: {
            "profile": profile.to_dict(),
            "manifest_hash": _digest("f"),
            "engine_id": engine_id,
            "tool_names": [tool_name],
            "environment_reference": f"toolbox:demo:{profile.profile_id}:revision",
        }
    }
    routes = {
        tool_name: {
            "profile_id": profile.profile_id,
            "engine_id": engine_id,
            "non_restartable": False,
        }
    }
    return profile, profiles, routes


def _publish(repository: AtomicJsonToolboxStateV2Repository, character: str, tool_name: str, engine_id: str, expected=None):
    definition = _definition(tool_name, expected)
    _profile, profiles, routes = _runtime(character, tool_name, engine_id)
    return repository.publish(
        toolbox_id="demo",
        expected_revision=expected,
        definition=definition.to_dict(),
        profiles=profiles,
        tool_routes=routes,
        environment_references=[next(iter(profiles.values()))["environment_reference"]],
        published_at_ms=int(time.time() * 1000),
    )


def _concurrent_publish(path: str, character: str, tool_name: str, queue) -> None:
    try:
        snapshot = _publish(AtomicJsonToolboxStateV2Repository(Path(path)), character, tool_name, f"engine-{character}")
        queue.put(("ok", snapshot["active_revision"]))
    except Exception as exc:
        queue.put((type(exc).__name__, str(exc)))


def test_state_reader_fails_closed_on_legacy_corruption_version_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "toolbox_sandboxes_v2.json"
    legacy = tmp_path / "toolbox_sandboxes.json"
    legacy.write_text('{"version":1,"toolboxes":{}}', encoding="utf-8")
    repository = AtomicJsonToolboxStateV2Repository(path, legacy_path=legacy)
    with pytest.raises(LegacyToolboxStateError, match="toolbox_state_v1_unsupported"):
        repository.read()

    legacy.unlink()
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="toolbox_state_v2_corrupt"):
        repository.read()
    path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="toolbox_state_v2_fields_invalid"):
        repository.read()
    path.unlink()
    repository.initialize_empty()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state_digest"] = _digest("0")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="toolbox_state_v2_digest_mismatch"):
        repository.read()


def test_publish_is_process_safe_cas_and_interrupted_replace_preserves_old_state(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "toolbox_sandboxes_v2.json"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [
        context.Process(target=_concurrent_publish, args=(str(path), "a", "Alpha", queue)),
        context.Process(target=_concurrent_publish, args=(str(path), "b", "Beta", queue)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0
    outcomes = [queue.get(timeout=2), queue.get(timeout=2)]
    assert sorted(item[0] for item in outcomes) == ["ToolboxRevisionConflictError", "ok"]
    repository = AtomicJsonToolboxStateV2Repository(path)
    before = path.read_bytes()
    active_revision = repository.get("demo")["active_revision"]

    monkeypatch.setattr(
        "hosting.service.toolbox_state_v2._replace_with_bounded_retries",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("interrupted replace")),
    )
    with pytest.raises(OSError, match="interrupted replace"):
        _publish(repository, "c", "Gamma", "engine-c", expected=active_revision)
    assert path.read_bytes() == before
    assert repository.get("demo")["active_revision"] == active_revision


def test_rollout_recovery_resolves_pre_and_post_publication_crash_points(tmp_path: Path) -> None:
    service = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    service._require_toolbox_executor_registration = lambda engine_id, *, command_label: service.get_registration(engine_id)  # type: ignore[method-assign]
    candidate = service.register_spawned(
        engine_id="candidate-before",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        routing_state="candidate",
        bundle={"toolbox_id": "demo", "resolved_profile_id": "profile-before"},
        environment={"environment_key": _digest("a")},
    )
    prepared = service._hosted_operations.prepare(
        owner_actor_id="actor:a",
        execution_kind="toolbox_definition_apply",
        selector={"kind": "toolbox_id", "id": "demo"},
        namespace="toolbox-definition:demo",
        request_id="before",
        fingerprint=hosted_execution_fingerprint({"request": "before"}),
        metadata={
            "toolbox_id": "demo",
            "definition_revision": _digest("d"),
            "candidate_engine_ids": [candidate["engine_id"]],
        },
    )
    before_id = prepared["status"]["operation"]["operation_id"]
    service._hosted_operations.mark_dispatch_claimed(operation_id=before_id)
    service._hosted_operations.update_progress(
        operation_id=before_id,
        progress={
            "phase": "warmup",
            "code": "warmup",
            "completed_units": 0,
            "total_units": 1,
            "updated_at_ms": int(time.time() * 1000),
            "summary": "Warmup.",
            "cancellable": True,
        },
    )
    recovered_before = service.recover_toolbox_definition_rollouts()
    assert recovered_before["removed_candidate_engine_ids"] == ["candidate-before"]
    assert service._hosted_operations.get_by_operation_id(before_id)["lifecycle"] == "terminal_failure"

    definition = _definition("Alpha")
    profile, profiles, routes = _runtime("b", "Alpha", "candidate-after")
    service.register_spawned(
        engine_id="candidate-after",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        routing_state="candidate",
        bundle={"toolbox_id": "demo", "resolved_profile_id": profile.profile_id},
        environment={"environment_key": profile.environment_key},
    )
    service.register_spawned(
        engine_id="old-active",
        pid=1234,
        command=["python", "worker.py"],
        executor_kind="toolbox_executor",
        routing_state="active",
        bundle={"toolbox_id": "demo", "resolved_profile_id": "old-profile"},
        environment={"environment_key": _digest("e")},
    )
    service._toolbox_state_v2.publish(
        toolbox_id="demo",
        expected_revision=None,
        definition=definition.to_dict(),
        profiles=profiles,
        tool_routes=routes,
        environment_references=[next(iter(profiles.values()))["environment_reference"]],
        published_at_ms=int(time.time() * 1000),
    )
    after = service._hosted_operations.prepare(
        owner_actor_id="actor:a",
        execution_kind="toolbox_definition_apply",
        selector={"kind": "toolbox_id", "id": "demo"},
        namespace="toolbox-definition:demo",
        request_id="after",
        fingerprint=hosted_execution_fingerprint({"request": "after"}),
        metadata={"toolbox_id": "demo", "definition_revision": definition.revision},
    )
    after_id = after["status"]["operation"]["operation_id"]
    service._hosted_operations.mark_dispatch_claimed(operation_id=after_id)
    service._hosted_operations.update_progress(
        operation_id=after_id,
        progress={
            "phase": "publication",
            "code": "publication",
            "completed_units": 0,
            "total_units": 1,
            "updated_at_ms": int(time.time() * 1000),
            "summary": "Publication.",
            "cancellable": False,
        },
    )
    recovered_after = service.recover_toolbox_definition_rollouts()
    assert recovered_after["activated_engine_ids"] == ["candidate-after"]
    assert "old-active" in recovered_after["retired_engine_ids"]
    assert service.get_registration("old-active") is None
    assert service._hosted_operations.get_by_operation_id(after_id)["lifecycle"] == "terminal_success"


def test_archive_v1_validates_digest_moves_payload_and_initializes_empty_v2(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()
    state_root = root / "state"
    state_root.mkdir()
    state_file = state_root / "toolbox_sandboxes.json"
    state_file.write_text(json.dumps({"version": 1, "toolboxes": {}, "environment_descriptions": {}}), encoding="utf-8")
    bundle_file = root / "toolbox_bundles" / "demo" / "revision" / "manifest.json"
    bundle_file.parent.mkdir(parents=True)
    bundle_file.write_text('{"bundle":"demo"}', encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(state_file.read_bytes()).hexdigest()
    monkeypatch.setenv("MP13_RELEASE_COMMIT", "a" * 40)

    result = EngineHostService.toolbox_state_archive_v1(
        hosting_root=str(root),
        expected_state_sha256=digest,
        acknowledge_version_1_archive=True,
    )

    assert result["status"] == "ok"
    assert not state_file.exists()
    assert not (root / "toolbox_bundles").exists()
    archive = Path(result["archive_root"])
    assert (archive / "payload" / "state" / "toolbox_sandboxes.json").is_file()
    assert (archive / "payload" / "toolbox_bundles" / "demo" / "revision" / "manifest.json").is_file()
    inventory = json.loads((archive / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["parent_release_commit"] == "a" * 40
    assert {item["source_relative_path"] for item in inventory["files"]} == {
        "state/toolbox_sandboxes.json",
        "toolbox_bundles/demo/revision/manifest.json",
    }
    v2 = AtomicJsonToolboxStateV2Repository(state_root / "toolbox_sandboxes_v2.json").read()
    assert v2["version"] == 2 and v2["toolboxes"] == {}


def test_archive_v1_refuses_running_daemon_or_digest_mismatch_without_moving_state(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()
    state_root = root / "state"
    state_root.mkdir()
    state_file = state_root / "toolbox_sandboxes.json"
    state_file.write_text('{"version":1,"toolboxes":{}}', encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(state_file.read_bytes()).hexdigest()
    monkeypatch.setenv("MP13_RELEASE_COMMIT", "b" * 40)
    pid_file = DaemonPidFile(state_root / "daemon.pid")
    pid_file.write(pid=os.getpid(), port=0, shutdown_token="test")
    with pytest.raises(ToolboxStateArchiveError, match="toolbox_archive_daemon_running"):
        EngineHostService.toolbox_state_archive_v1(
            hosting_root=str(root),
            expected_state_sha256=digest,
            acknowledge_version_1_archive=True,
        )
    pid_file.remove()
    with pytest.raises(ToolboxStateArchiveError, match="toolbox_archive_state_digest_mismatch"):
        EngineHostService.toolbox_state_archive_v1(
            hosting_root=str(root),
            expected_state_sha256=_digest("0"),
            acknowledge_version_1_archive=True,
        )
    assert state_file.is_file()
    assert not (state_root / "toolbox_sandboxes_v2.json").exists()


def test_archive_v1_cli_is_local_and_dispatches_exact_payload(tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path.resolve()
    state_root = root / "state"
    state_root.mkdir()
    state_file = state_root / "toolbox_sandboxes.json"
    state_file.write_text('{"version":1,"toolboxes":{}}', encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(state_file.read_bytes()).hexdigest()
    monkeypatch.setenv("MP13_RELEASE_COMMIT", "c" * 40)
    payload = json.dumps(
        {
            "hosting_root": str(root),
            "expected_state_sha256": digest,
            "acknowledge_version_1_archive": True,
        }
    )

    exit_code = engine_host_cli_main(
        ["--payload-json", payload, "toolbox-state-archive-v1"]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["ok"] is True
    assert output["result"]["status"] == "ok"
