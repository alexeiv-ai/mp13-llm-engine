from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hosting.environments import (
    EnvironmentError,
    EnvironmentManager,
    EnvironmentRequest,
    EnvironmentTemplate,
)
from hosting.service.auth import AuthMixin


REVISION = "sha256:" + "a" * 64
LOCK_DIGEST = "sha256:" + "b" * 64


class RecordingBuilder:
    builder_id = "test-builder-v1"
    runtime_kind = "python"

    def __init__(self) -> None:
        self.calls = 0
        self.guard = threading.Lock()

    def build(self, *, request, destination: Path, package_lock):
        with self.guard:
            self.calls += 1
        time.sleep(0.02)
        (destination / "ready.json").write_text(json.dumps({"lock": package_lock["lock_digest"]}), encoding="utf-8")
        return {"builder_id": self.builder_id}


def _manager(tmp_path: Path, *, retention_seconds: int = 0):
    lock_root = tmp_path / "packages" / "locks"
    lock_root.mkdir(parents=True)
    (lock_root / f"{LOCK_DIGEST.split(':')[1]}.json").write_text(
        json.dumps({"contract": "hosting.package_lock.v1", "lock_id": "lock-1", "lock_digest": LOCK_DIGEST, "artifacts": []}),
        encoding="utf-8",
    )
    builder = RecordingBuilder()
    manager = EnvironmentManager(
        environment_root=tmp_path / "environments",
        scratch_root=tmp_path / "hosting" / "scratch",
        package_lock_root=lock_root,
        configuration_revision=REVISION,
        builders={builder.builder_id: builder},
        retention_seconds=retention_seconds,
    )
    manager.put_template(EnvironmentTemplate.from_dict({
        "contract": "hosting.environment_template.v1", "template_id": "base", "revision": 1,
        "runtime_kind": "python", "builder_id": builder.builder_id, "package_lock_id": "lock-1",
        "platforms": ["win_amd64"], "state": "active",
    }))
    return manager, builder


def _request(*, request_id: str, consumer_kind: str, consumer_id: str, revision: int = 1, template_revision: int = 1):
    return EnvironmentRequest.from_dict({
        "contract": "hosting.environment_request.v1", "request_id": request_id,
        "consumer_kind": consumer_kind, "consumer_id": consumer_id, "revision": revision,
        "template_id": "base", "template_revision": template_revision,
        "package_lock_digest": LOCK_DIGEST, "runtime_kind": "python", "platform": "win_amd64",
        "configuration_revision": REVISION,
    })


def test_contracts_are_strict_and_worker_neutral() -> None:
    request = _request(request_id="one", consumer_kind="toolbox", consumer_id="tb-1")
    assert request.consumer_kind == "toolbox"
    with pytest.raises(ValueError, match="fields_invalid"):
        EnvironmentRequest.from_dict({**request.to_dict(), "toolbox_id": "legacy"})


def test_toolbox_and_other_consumers_share_one_content_environment(tmp_path: Path) -> None:
    manager, builder = _manager(tmp_path)
    first = manager.ensure(_request(request_id="one", consumer_kind="toolbox", consumer_id="tb-1"))
    second = manager.ensure(_request(request_id="two", consumer_kind="workflow-python", consumer_id="wf-1"))
    assert first["receipt"]["environment_id"] == second["receipt"]["environment_id"]
    assert first["reference"]["consumer_kind"] == "toolbox"
    assert second["reference"]["consumer_kind"] == "workflow-python"
    assert builder.calls == 1


def test_concurrent_same_key_coalesces_and_reference_release_is_idempotent(tmp_path: Path) -> None:
    manager, builder = _manager(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: manager.ensure(_request(request_id=f"r{i}", consumer_kind="worker", consumer_id=f"c{i}")), range(2)))
    assert builder.calls == 1
    assert len({row["receipt"]["environment_id"] for row in results}) == 1
    reference_id = results[0]["reference"]["reference_id"]
    assert manager.release(reference_id=reference_id) == manager.release(reference_id=reference_id)


def test_references_busy_and_retention_guard_removal(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path, retention_seconds=60)
    result = manager.ensure(_request(request_id="one", consumer_kind="worker", consumer_id="one"))
    environment_id = result["receipt"]["environment_id"]
    with pytest.raises(EnvironmentError, match="environment_referenced"):
        manager.remove(environment_id=environment_id)
    manager.release(reference_id=result["reference"]["reference_id"])
    manager.execution_begin(environment_id=environment_id, execution_id="execution-1")
    with pytest.raises(EnvironmentError, match="environment_active"):
        manager.remove(environment_id=environment_id)
    assert manager.execution_end(execution_id="execution-1")["state"] == "ended"
    with pytest.raises(EnvironmentError, match="environment_retained"):
        manager.remove(environment_id=environment_id)
    assert manager.remove(environment_id=environment_id, force_retention=True)["state"] == "removed"


def test_state_version_rejection_and_reference_pagination(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    for index in range(3):
        manager.ensure(_request(request_id=f"r{index}", consumer_kind="worker", consumer_id=f"c{index}"))
    first = manager.list_references(limit=2)
    assert len(first["references"]) == 2
    assert first["next_cursor"]
    second = manager.list_references(cursor=first["next_cursor"], limit=2)
    assert len(second["references"]) == 1
    state_path = tmp_path / "environments" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["contract"] = "hosting.environment_state.v1"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(EnvironmentError, match="environment_state_invalid"):
        manager.list_templates()


def test_legacy_roots_and_incomplete_builds_are_not_discovered(tmp_path: Path) -> None:
    for name in ("toolbox_venvs", "runtime_envs", "toolbox_environment_cache"):
        (tmp_path / name / "fake").mkdir(parents=True)
    manager, builder = _manager(tmp_path)
    result = manager.ensure(_request(request_id="one", consumer_kind="worker", consumer_id="one"))
    assert builder.calls == 1
    assert "environments" in str(tmp_path / "environments")
    digest = result["receipt"]["environment_id"].split(":", 1)[1]
    assert (tmp_path / "environments" / "content" / digest).is_dir()


def test_public_role_commands_use_only_generic_environment_names() -> None:
    all_commands = set().union(*(AuthMixin._commands_allowed_for_role(role) for role in (
        "admin", "worker_user", "diagnostic_user", "dependency_approver", "config_editor"
    )))
    assert "environment-template-list" in all_commands
    assert "environment-remove" in all_commands
    assert not any(command.startswith("toolbox-template-") for command in all_commands)
    assert "toolbox-environment-remove" not in all_commands
