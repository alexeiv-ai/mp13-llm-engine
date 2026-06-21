from __future__ import annotations

from pathlib import Path
import base64
import hashlib
import io
import shutil
import sys
import threading
import time
import zipfile

import pytest
from hosting._process_utils import terminate_process_tree
from hosting.service.host_service import EngineHostService
from hosting.daemon.local_ipc import EngineHostDaemon
from hosting.sandbox.workflow_python_contract import build_workflow_python_node_snippet_request
from hosting.sandbox.workflow_python_node_runtime import WorkflowPythonNodeRuntimeRegistry


def test_spawn_workflow_python_helper_uses_existing_spawn_model(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    seen = {}

    def fake_spawn(self, **kwargs):
        seen.update(kwargs)
        return {
            "engine_id": kwargs["engine_id"],
            "command": list(kwargs["command"]),
            "env": dict(kwargs["env"]),
            "worker_profile_class": kwargs["worker_profile_class"],
            "sandbox_policy": dict(kwargs["sandbox_policy"]),
            "executor_kind": kwargs["executor_kind"],
            "capabilities": dict(kwargs["capabilities"]),
        }

    monkeypatch.setattr(EngineHostService, "spawn", fake_spawn)

    out = svc.spawn_workflow_python_helper(engine_id="wf-py", python_executable="python-custom", capacity=3)

    assert out["engine_id"] == "wf-py"
    assert out["command"][-1] == "hosting.workflow_python_helper_ipc"
    assert out["env"]["MP13_WORKER_CONTRACT"] == "hosting.workflow_helper.worker.v1"
    assert out["env"]["MP13_WORKFLOW_PYTHON"] == "python-custom"
    assert out["env"]["MP13_WORKFLOW_PYTHON_HELPER_CAPACITY"] == "3"
    assert out["worker_profile_class"] == "generic"
    assert out["executor_kind"] == "workflow_python_helper"
    assert out["workflow_runtime_kind"] == "workflow_python"
    assert out["workflow_profile"] == "helper"
    assert out["environment_key"]
    assert out["workflow_ensure"]["outcome"] == "spawned"
    assert out["sandbox_policy"]["sandbox"]["profile"] == "workflow_python_helper_v1"
    assert out["sandbox_policy"]["sandbox"]["network"]["mode"] == "disabled"
    assert out["sandbox_policy"]["sandbox"]["brokered_io"] == {
        "filesystem": False,
        "http": False,
        "subprocess": False,
    }
    assert seen["capabilities"]["workflow_python_helper"] is True
    assert seen["capabilities"]["capacity"] == 3


def test_daemon_spawn_preserves_worker_profile_class() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.kwargs = None

        def spawn(self, **kwargs):
            self.kwargs = dict(kwargs)
            return {"status": "ok", "worker_profile_class": kwargs.get("worker_profile_class")}

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    out = daemon._call_service(
        "spawn",
        {
            "engine_id": "worker-demo",
            "command": ["python", "-m", "hosting.engine_worker_ipc"],
            "worker_profile_class": "generic",
        },
    )

    assert out["worker_profile_class"] == "generic"
    assert fake.kwargs["worker_profile_class"] == "generic"


def test_daemon_dispatches_workflow_python_facade() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.calls = []

        def workflow_python_environment_spec(self, **kwargs):
            self.calls.append(("spec", dict(kwargs)))
            return {"status": "ok", "environment_key": "env-key"}

        def ensure_workflow_python(self, **kwargs):
            self.calls.append(("ensure", dict(kwargs)))
            return {"status": "ok", "engine_id": "wf-py"}

        def execute_workflow_python(self, **kwargs):
            self.calls.append(("execute", dict(kwargs)))
            return {"status": "ok", "ok": True}

        def workflow_python_resources(self, **kwargs):
            self.calls.append(("resources", dict(kwargs)))
            return {"status": "ok"}

        def set_workflow_python_capacity(self, **kwargs):
            self.calls.append(("set_capacity", dict(kwargs)))
            return {"status": "ok", "capacity": kwargs["capacity"]}

        def cancel_workflow_python_request(self, **kwargs):
            self.calls.append(("cancel", dict(kwargs)))
            return {"status": "ok", "request_id": kwargs["request_id"]}

        def workflow_python_stream_open(self, **kwargs):
            self.calls.append(("stream_open", dict(kwargs)))
            return {"status": "ok", "stream_id": "stream-1"}

        def workflow_python_event_subscribe(self, **kwargs):
            self.calls.append(("event_subscribe", dict(kwargs)))
            return {"status": "ok", "batch": {"frames": []}, "normalized_events": []}

        def workflow_python_stream_send(self, **kwargs):
            self.calls.append(("stream_send", dict(kwargs)))
            return {"status": "ok", "accepted": True}

        def workflow_python_stream_close(self, **kwargs):
            self.calls.append(("stream_close", dict(kwargs)))
            return {"status": "ok", "closed": True}

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    assert daemon._call_service("workflow-python-environment-spec", {"profile": "helper"})["environment_key"] == "env-key"
    assert daemon._call_service("workflow-python-ensure", {"engine_id": "wf-py"})["engine_id"] == "wf-py"
    assert daemon._call_service("workflow-python-execute", {"request": {"request_id": "req-1"}})["ok"] is True
    assert daemon._call_service("workflow-python-resources", {"engine_id": "wf-py"})["status"] == "ok"
    assert daemon._call_service("workflow-python-set-capacity", {"engine_id": "wf-py", "capacity": 5})["capacity"] == 5
    assert daemon._call_service("workflow-python-cancel-request", {"engine_id": "wf-py", "request_id": "req-1"})["request_id"] == "req-1"
    assert daemon._call_service("workflow-python-stream-open", {"profile": "node", "request": {"request_id": "req-node"}})["stream_id"] == "stream-1"
    assert daemon._call_service("workflow-python-event-subscribe", {"stream_id": "stream-1", "max_items": 2})["normalized_events"] == []
    assert daemon._call_service("workflow-python-stream-send", {"stream_id": "stream-1", "message": {"action": "cancel"}})["accepted"] is True
    assert daemon._call_service("workflow-python-stream-close", {"stream_id": "stream-1"})["closed"] is True

    assert [name for name, _ in fake.calls] == [
        "spec",
        "ensure",
        "execute",
        "resources",
        "set_capacity",
        "cancel",
        "stream_open",
        "event_subscribe",
        "stream_send",
        "stream_close",
    ]
    assert fake.calls[-4][1]["profile"] == "node"
    assert fake.calls[-4][1]["environment_name"] == "workflow-python-node"
    assert fake.calls[-4][1]["request"] == {"request_id": "req-node"}
    assert fake.calls[-3][1] == {"stream_id": "stream-1", "max_items": 2}
    assert fake.calls[-2][1] == {"stream_id": "stream-1", "message": {"action": "cancel"}}
    assert fake.calls[-1][1] == {"stream_id": "stream-1"}


def test_daemon_dispatches_workflow_js_facade() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.calls = []

        def workflow_js_environment_spec(self, **kwargs):
            self.calls.append(("spec", dict(kwargs)))
            return {"status": "ok", "environment_key": "env-js"}

        def ensure_workflow_js(self, **kwargs):
            self.calls.append(("ensure", dict(kwargs)))
            return {"status": "ok", "engine_id": kwargs.get("engine_id")}

        def execute_workflow_js(self, **kwargs):
            self.calls.append(("execute", dict(kwargs)))
            return {"status": "ok", "ok": True}

        def workflow_js_resources(self, **kwargs):
            self.calls.append(("resources", dict(kwargs)))
            return {"status": "ok"}

        def set_workflow_js_capacity(self, **kwargs):
            self.calls.append(("set_capacity", dict(kwargs)))
            return {"status": "ok", "capacity": kwargs["capacity"]}

        def cancel_workflow_js_request(self, **kwargs):
            self.calls.append(("cancel", dict(kwargs)))
            return {"status": "ok", "request_id": kwargs["request_id"]}

        def workflow_js_request_status(self, **kwargs):
            self.calls.append(("status", dict(kwargs)))
            return {"status": "ok", "request_id": kwargs["request_id"]}

        def workflow_js_stream_open(self, **kwargs):
            self.calls.append(("stream_open", dict(kwargs)))
            return {"status": "ok", "stream_id": "js-stream-1"}

        def workflow_js_event_subscribe(self, **kwargs):
            self.calls.append(("event_subscribe", dict(kwargs)))
            return {"status": "ok", "batch": {"frames": []}, "normalized_events": []}

        def workflow_js_stream_send(self, **kwargs):
            self.calls.append(("stream_send", dict(kwargs)))
            return {"status": "ok", "accepted": True}

        def workflow_js_stream_close(self, **kwargs):
            self.calls.append(("stream_close", dict(kwargs)))
            return {"status": "ok", "closed": True}

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    assert daemon._call_service("workflow-js-environment-spec", {"profile": "node", "javascript": {"host_api": {"enabled": True}}})["environment_key"] == "env-js"
    assert daemon._call_service("workflow-js-ensure", {"engine_id": "wf-js"})["engine_id"] == "wf-js"
    assert daemon._call_service("workflow-js-execute", {"request": {"request_id": "req-1"}})["ok"] is True
    assert daemon._call_service("workflow-js-resources", {"engine_id": "wf-js"})["status"] == "ok"
    assert daemon._call_service("workflow-js-set-capacity", {"engine_id": "wf-js", "capacity": 5})["capacity"] == 5
    assert daemon._call_service("workflow-js-cancel-request", {"engine_id": "wf-js", "request_id": "req-1"})["request_id"] == "req-1"
    assert daemon._call_service("workflow-js-request-status", {"engine_id": "wf-js", "request_id": "req-1"})["request_id"] == "req-1"
    assert daemon._call_service("workflow-js-stream-open", {"profile": "node", "request": {"request_id": "req-js-stream"}})["stream_id"] == "js-stream-1"
    assert daemon._call_service("workflow-js-event-subscribe", {"stream_id": "js-stream-1", "max_items": 2})["normalized_events"] == []
    assert daemon._call_service("workflow-js-stream-send", {"stream_id": "js-stream-1", "message": {"action": "cancel"}})["accepted"] is True
    assert daemon._call_service("workflow-js-stream-close", {"stream_id": "js-stream-1"})["closed"] is True

    assert [name for name, _ in fake.calls] == [
        "spec",
        "ensure",
        "execute",
        "resources",
        "set_capacity",
        "cancel",
        "status",
        "stream_open",
        "event_subscribe",
        "stream_send",
        "stream_close",
    ]
    assert fake.calls[0][1]["profile"] == "node"
    assert fake.calls[0][1]["javascript"] == {"host_api": {"enabled": True}}
    assert fake.calls[-4][1]["profile"] == "node"
    assert fake.calls[-4][1]["environment_name"] == "workflow-js-node"
    assert fake.calls[-4][1]["request"] == {"request_id": "req-js-stream"}
    assert fake.calls[-3][1] == {"stream_id": "js-stream-1", "max_items": 2}
    assert fake.calls[-2][1] == {"stream_id": "js-stream-1", "message": {"action": "cancel"}}
    assert fake.calls[-1][1] == {"stream_id": "js-stream-1"}


def test_workflow_js_facade_reserves_environment_keyed_node_pool(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    out = svc.ensure_workflow_js(
        profile="node",
        node={"runtime_hash": "quickjs-demo"},
        capacity=3,
    )

    assert out["status"] == "ok"
    assert out["outcome"] == "ready"
    assert out["engine_id"].startswith("workflow-js-")
    assert out["environment_key"]

    resources = svc.workflow_js_resources(
        profile="node",
        node={"runtime_hash": "quickjs-demo"},
    )
    assert resources["workflow_pool"]["metrics"]["desired_capacity"] == 3
    assert resources["workflow_pool"]["metrics"]["worker_count"] == 1


def test_workflow_js_facade_isolates_node_pools_by_policy(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    first = svc.ensure_workflow_js(
        profile="node",
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_js_node_v1"}},
    )
    second = svc.ensure_workflow_js(
        profile="node",
        sandbox_policy={"sandbox": {"enabled": False, "profile": "workflow_js_node_v1"}},
    )

    assert first["environment_key"] != second["environment_key"]
    pools = svc._workflow_python_pool_registry().resources()["pools"]
    assert set(pools.keys()) == {
        f"workflow_js/{first['environment_key']}",
        f"workflow_js/{second['environment_key']}",
    }

def test_workflow_python_helper_resources_include_child_process_metrics(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    def fake_proxy_rpc_call(**kwargs):
        assert kwargs["engine_id"] == "wf-py"
        assert kwargs["method"] == "worker.resources"
        return {
            "result": {
                "status": "ok",
                "capacity": 2,
                "active_calls": 1,
                "pool": {
                    "process_count": 1,
                    "active_process_count": 1,
                    "idle_process_count": 0,
                    "processes": [
                        {
                            "pid": 4321,
                            "alive": True,
                            "busy": True,
                            "active_request_id": "req-live",
                            "request_count": 3,
                        }
                    ],
                },
            }
        }

    monkeypatch.setattr(svc, "proxy_rpc_call", fake_proxy_rpc_call)
    monkeypatch.setattr(svc, "_process_resource_snapshot", lambda pid: {"pid": pid, "cpu_percent": 12.5, "memory_mb": 64.0})

    out = svc.workflow_python_helper_resources(engine_id="wf-py")

    pool = out["pool"]
    assert pool["active_request_ids"] == ["req-live"]
    assert pool["cpu_percent"] == 12.5
    assert pool["memory_mb"] == 64.0
    assert pool["processes"][0]["resources"]["pid"] == 4321


def test_workflow_python_helper_proxy_realizes_runtime_environment(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    svc.spawn(
        engine_id="wf-py-runtime",
        command=["python", "-m", "hosting.workflow_python_helper_ipc"],
        worker_profile_class="generic",
        executor_kind="workflow_python_helper",
        sandbox_policy={"sandbox": {"enabled": False, "profile": "workflow_python_helper_v1"}},
        capabilities={"workflow_python_helper": True},
    )
    captured = {}

    def fake_ipc_call(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "result": {"ok": True, "result": None}}

    monkeypatch.setattr(svc, "_ipc_call", fake_ipc_call)

    try:
        out = svc.proxy_rpc_call(
            engine_id="wf-py-runtime",
            method="execute_workflow_python_helper",
            params={
                "request_id": "req-runtime",
                "module_source": "def condition(input):\n    return None\n",
                "module_sha256": "abc123",
                "package_id": "pkg-demo",
                "workflow_id": "wf-demo",
                "package_source_digest": "sha256:digest",
                "source_path": "helpers/condition.py",
                "operation": "condition",
                "export_name": "condition",
                "payload": {},
                "python": {
                    "import_allowlist": ["json"],
                    "package_pins": {"demo": "1.0.0"},
                    "environment_name": "workflow-python-helper",
                },
            },
        )
    finally:
        svc.shutdown("wf-py-runtime", timeout_seconds=5.0)

    rpc_params = captured["payload"]["params"]
    assert "_workflow_python_facade_execute" not in rpc_params
    assert out["workflow_runtime_kind"] == "workflow_python"
    assert out["environment_key"]
    assert out["workflow_execute"]["metrics"]["request"]["request_id"] == "req-runtime"
    python_runtime = rpc_params["python"]
    assert python_runtime["environment_name"] == "workflow-python-helper"
    assert python_runtime["python_executable"]
    assert python_runtime["python_source"] in {"bootstrap", "venv"}
    assert python_runtime["runtime_environment"]["venv_key"]


def test_workflow_python_helper_spawn_and_rpc_round_trip(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    reg = svc.spawn_workflow_python_helper(
        engine_id="wf-py-roundtrip",
        python_executable=None,
        sandbox_policy={
            "sandbox": {
                "enabled": False,
                "profile": "workflow_python_helper_v1",
                "network": {"mode": "disabled"},
                "brokered_io": {"filesystem": False, "http": False, "subprocess": False},
            }
        },
    )
    source = "def condition(input):\n    return {'accepted': input['value'] == 7}\n"
    try:
        svc._wait_for_worker_rpc_ready(reg, timeout_seconds=5.0, poll_interval_seconds=0.1)
        hello = svc.proxy_rpc_call(engine_id="wf-py-roundtrip", method="rpc.describe", params={})
        assert hello["executor_kind"] == "workflow_python_helper"
        out = svc.proxy_rpc_call(
            engine_id="wf-py-roundtrip",
            method="execute_workflow_python_helper",
            params={
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg-demo",
                "workflow_id": "config/demo",
                "package_source_digest": "sha256:digest",
                "source_path": "helpers/condition.py",
                "export_name": "condition",
                "operation": "condition",
                "payload": {"value": 7},
                "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
            },
        )
        result = out["result"]
        assert result["ok"] is True
        assert result["result"] == {"accepted": True}
        assert result["runtime"]["sandbox_profile"] == "workflow_python_helper_v1"
        persisted = svc.get_registration("wf-py-roundtrip")
        assert persisted is not None
        assert persisted["executor_kind"] == "workflow_python_helper"
        assert persisted["sandbox_policy"]["sandbox"]["profile"] == "workflow_python_helper_v1"
        assert persisted["sandbox_runtime"]
        ensured = svc.ensure_running("wf-py-roundtrip")
        assert ensured["status"] in {"ok", "already_running", "running"}
    finally:
        svc.shutdown(str(reg.get("engine_id") or "wf-py-roundtrip"), timeout_seconds=5.0)


def test_workflow_python_environment_spec_facade(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    out = svc.workflow_python_environment_spec(
        profile="helper",
        python={"import_allowlist": ["json"], "package_pins": {"demo": "1.0.0"}},
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_python_helper_v1"}},
    )

    assert out["status"] == "ok"
    assert out["environment_key"]
    assert out["environment"]["environment_root_kind"] == "runtime_envs"
    assert out["environment"]["required_imports"] == ["json"]


def test_ensure_workflow_python_helper_spawns_environment_keyed_worker(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    seen = {}

    monkeypatch.setattr(svc, "get_registration", lambda _engine_id: None)

    def fake_spawn(**kwargs):
        seen.update(kwargs)
        return {"status": "ok", "engine_id": kwargs["engine_id"]}

    monkeypatch.setattr(svc, "_spawn_workflow_python_helper_worker", fake_spawn)
    monkeypatch.setattr(svc, "workflow_python_helper_resources", lambda **_kwargs: {"status": "ok"})

    out = svc.ensure_workflow_python(
        profile="helper",
        python={"import_allowlist": ["json"], "package_pins": {}},
        capacity=3,
    )

    assert out["status"] == "ok"
    assert out["outcome"] == "spawned"
    assert out["engine_id"].startswith("workflow-python-")
    assert out["environment_key"]
    assert seen["engine_id"] == out["engine_id"]
    assert seen["capacity"] == 3

    resources = svc.workflow_python_resources(
        profile="helper",
        python={"import_allowlist": ["json"], "package_pins": {}},
    )
    assert resources["workflow_pool"]["metrics"]["desired_capacity"] == 3
    assert resources["workflow_pool"]["metrics"]["worker_count"] == 1
    assert resources["workflow_pool"]["metrics"]["available_slots"] == 3


def test_ensure_workflow_python_annotates_existing_registration(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    svc.register_spawned(
        engine_id="wf-py-existing",
        pid=0,
        command=["python", "-m", "hosting.workflow_python_helper_ipc"],
        worker_profile_class="generic",
        executor_kind="workflow_python_helper",
        capabilities={"workflow_python_helper": True},
    )
    monkeypatch.setattr(svc, "ensure_running", lambda _engine_id: {"status": "already_running"})

    out = svc.ensure_workflow_python(
        profile="helper",
        engine_id="wf-py-existing",
        python={"import_allowlist": ["json"], "package_pins": {}},
    )

    assert out["status"] == "ok"
    reg = svc.get_registration("wf-py-existing")
    assert reg is not None
    assert reg["environment"]["environment_key"] == out["environment_key"]
    assert reg["environment"]["workflow_runtime_kind"] == "workflow_python"
    assert reg["capabilities"]["workflow_python"] is True
    assert reg["capabilities"]["environment_key"] == out["environment_key"]


def test_ensure_workflow_python_rejects_environment_key_mismatch(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    out = svc.ensure_workflow_python(
        profile="helper",
        environment_key="wrong-key",
        python={"import_allowlist": ["json"], "package_pins": {}},
    )

    assert out["status"] == "error"
    assert out["reason"] == "environment_key_mismatch"
    assert out["derived_environment_key"] != "wrong-key"


def test_workflow_python_facade_isolates_pools_by_environment_key(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    spawned = []

    monkeypatch.setattr(svc, "get_registration", lambda _engine_id: None)
    monkeypatch.setattr(svc, "workflow_python_helper_resources", lambda **_kwargs: {"status": "ok"})

    def fake_spawn(**kwargs):
        spawned.append(dict(kwargs))
        return {"status": "ok", "engine_id": kwargs["engine_id"]}

    monkeypatch.setattr(svc, "_spawn_workflow_python_helper_worker", fake_spawn)

    first = svc.ensure_workflow_python(
        profile="helper",
        python={"import_allowlist": ["json"]},
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_python_helper_v1"}},
    )
    second = svc.ensure_workflow_python(
        profile="helper",
        python={"import_allowlist": ["json"]},
        sandbox_policy={"sandbox": {"enabled": False, "profile": "workflow_python_helper_v1"}},
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert first["environment_key"] != second["environment_key"]
    assert first["engine_id"] != second["engine_id"]
    pools = svc._workflow_python_pool_registry().resources()["pools"]
    assert sorted(pools.keys()) == [
        f"workflow_python/{first['environment_key']}",
        f"workflow_python/{second['environment_key']}",
    ]
    assert [row["engine_id"] for row in spawned] == [first["engine_id"], second["engine_id"]]


def test_workflow_python_capacity_and_cancel_infer_environment_key_from_registration(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    svc.register_spawned(
        engine_id="wf-py-existing",
        pid=0,
        command=["python", "-m", "hosting.workflow_python_helper_ipc"],
        worker_profile_class="generic",
        executor_kind="workflow_python_helper",
        capabilities={"workflow_python_helper": True},
    )
    monkeypatch.setattr(svc, "ensure_running", lambda _engine_id: {"status": "already_running"})
    monkeypatch.setattr(svc, "set_workflow_python_helper_capacity", lambda **kwargs: {"status": "ok", "capacity": kwargs["capacity"]})
    monkeypatch.setattr(svc, "cancel_workflow_python_helper_request", lambda **kwargs: {"status": "ok", "canceled": True, "request_id": kwargs["request_id"]})

    ensured = svc.ensure_workflow_python(
        profile="helper",
        engine_id="wf-py-existing",
        python={"import_allowlist": ["json"], "package_pins": {}},
    )

    resized = svc.set_workflow_python_capacity(
        profile="helper",
        engine_id="wf-py-existing",
        capacity=5,
    )
    canceled = svc.cancel_workflow_python_request(
        profile="helper",
        engine_id="wf-py-existing",
        request_id="req-missing",
    )

    assert resized["environment_key"] == ensured["environment_key"]
    assert canceled["environment_key"] == ensured["environment_key"]
    pool = svc._workflow_python_pool_registry().get(svc._workflow_python_pool_key(ensured["environment_key"]))
    assert pool is not None
    assert pool.resources()["metrics"]["desired_capacity"] == 5
    assert canceled["workflow_pool_cancel"]["status"] == "not_found"


def test_old_python_helper_resource_alias_reports_workflow_pool_for_annotated_registration(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    svc.register_spawned(
        engine_id="wf-py-existing",
        pid=0,
        command=["python", "-m", "hosting.workflow_python_helper_ipc"],
        worker_profile_class="generic",
        executor_kind="workflow_python_helper",
        capabilities={"workflow_python_helper": True},
    )
    monkeypatch.setattr(svc, "ensure_running", lambda _engine_id: {"status": "already_running"})
    monkeypatch.setattr(svc, "proxy_rpc_call", lambda **_kwargs: {"result": {"status": "ok", "capacity": 2, "pool": {}}})

    ensured = svc.ensure_workflow_python(
        profile="helper",
        engine_id="wf-py-existing",
        python={"import_allowlist": ["json"]},
        capacity=2,
    )
    out = svc.workflow_python_helper_resources(engine_id="wf-py-existing")

    assert out["workflow_runtime_kind"] == "workflow_python"
    assert out["workflow_profile"] == "helper"
    assert out["environment_key"] == ensured["environment_key"]
    assert out["workflow_pool"]["metrics"]["desired_capacity"] == 2


def test_old_python_helper_capacity_and_cancel_alias_update_workflow_pool(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    svc.register_spawned(
        engine_id="wf-py-existing",
        pid=0,
        command=["python", "-m", "hosting.workflow_python_helper_ipc"],
        worker_profile_class="generic",
        executor_kind="workflow_python_helper",
        capabilities={"workflow_python_helper": True},
    )
    monkeypatch.setattr(svc, "ensure_running", lambda _engine_id: {"status": "already_running"})

    def fake_proxy_rpc_call(**kwargs):
        if kwargs["method"] == "workflow_python_helper.set_capacity":
            return {"result": {"status": "ok", "capacity": kwargs["params"]["capacity"], "pool": {}}}
        if kwargs["method"] == "workflow_python_helper.cancel_request":
            return {"result": {"status": "ok", "canceled": True, "request_id": kwargs["params"]["request_id"]}}
        return {"result": {"status": "ok", "pool": {}}}

    monkeypatch.setattr(svc, "proxy_rpc_call", fake_proxy_rpc_call)

    ensured = svc.ensure_workflow_python(
        profile="helper",
        engine_id="wf-py-existing",
        python={"import_allowlist": ["json"]},
        capacity=2,
    )
    resized = svc.set_workflow_python_helper_capacity(engine_id="wf-py-existing", capacity=4)
    canceled = svc.cancel_workflow_python_helper_request(engine_id="wf-py-existing", request_id="req-missing")

    assert resized["environment_key"] == ensured["environment_key"]
    assert resized["workflow_pool"]["metrics"]["desired_capacity"] == 4
    assert canceled["environment_key"] == ensured["environment_key"]
    assert canceled["workflow_pool_cancel"]["status"] == "not_found"


def test_execute_workflow_python_helper_facade_uses_existing_rpc(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    calls = {}

    monkeypatch.setattr(
        svc,
        "ensure_workflow_python",
        lambda **kwargs: {
            "status": "ok",
            "engine_id": "workflow-python-demo",
            "environment_key": "env-demo",
        },
    )

    def fake_proxy_rpc_call(**kwargs):
        calls.update(kwargs)
        return {"status": "ok", "result": {"ok": True, "result": {"accepted": True}}}

    monkeypatch.setattr(svc, "proxy_rpc_call", fake_proxy_rpc_call)

    out = svc.execute_workflow_python(
        profile="helper",
        request={
            "module_source": "def condition(input):\n    return {'accepted': True}\n",
            "module_sha256": "demo",
            "operation": "condition",
            "export_name": "condition",
            "payload": {},
            "limits": {"timeout_ms": 1000},
        },
    )

    assert out["status"] == "ok"
    assert out["ok"] is True
    assert out["output"] == {"accepted": True}
    assert out["metrics"]["request"]["status"] == "ok"
    assert out["metrics"]["workflow_pool"]["metrics"]["active_calls"] == 0
    assert out["metrics"]["workflow_pool"]["metrics"]["recent_requests"][0]["request_id"] == "workflow-python-sync"
    assert calls["engine_id"] == "workflow-python-demo"
    assert calls["method"] == "execute_workflow_python_helper"


def test_execute_workflow_js_facade_uses_quickjs_node_runtime(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "exports.run = function(input, api) { return { output: { accepted: input.value === 7 }, state_patch: { seen: true } }; };"

    out = svc.execute_workflow_js(
        profile="node",
        request={
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg-demo",
            "workflow_id": "wf-demo",
            "package_source_digest": "sha256:digest",
            "export_name": "run",
            "payload": {"value": 7},
            "limits": {"timeout_ms": 1000},
        },
    )

    assert out["status"] == "ok"
    assert out["ok"] is True
    assert out["output"] == {"accepted": True}
    assert out["state_patch"] == {"seen": True}
    assert out["metrics"]["request"]["status"] == "ok"
    assert out["metrics"]["workflow_pool"]["metrics"]["active_calls"] == 0
    assert out["metrics"]["workflow_pool"]["metrics"]["recent_requests"][0]["request_id"] == "workflow-js-sync"


def test_execute_workflow_js_rejects_missing_contract_fields(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "exports.run = function() { return {output: true}; };"

    out = svc.execute_workflow_js(
        profile="node",
        request={
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "payload": {},
        },
    )

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_js_node_invalid_request"
    assert "package_id" in out["error"]["message"]
    assert "workflow_id" in out["error"]["message"]
    assert "package_source_digest" in out["error"]["message"]


def test_execute_workflow_js_node_reads_and_writes_declared_artifacts(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    write_source = """
exports.run = function(input, api) {
  api.fs.writeText("seed", "", "seed text");
  return {output: {written: true}};
};
"""
    read_source = """
exports.run = function(input, api) {
  const text = api.fs.readText("seed", "");
  api.fs.writeText("report", "", text.toUpperCase());
  return {output: {text: text}};
};
"""

    written = svc.execute_workflow_js(
        profile="node",
        request={
            "request_id": "req-js-artifact-write",
            "module_source": write_source,
            "module_sha256": hashlib.sha256(write_source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "artifact_outputs": [{"name": "seed", "filename": "seed.txt", "media_type": "text/plain"}],
        },
    )
    read = svc.execute_workflow_js(
        profile="node",
        request={
            "request_id": "req-js-artifact-read",
            "module_source": read_source,
            "module_sha256": hashlib.sha256(read_source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "artifact_inputs": [{"name": "seed", "ref": written["artifacts"][0]["ref"]}],
            "artifact_outputs": [{"name": "report", "filename": "report.txt", "media_type": "text/plain"}],
        },
    )

    assert written["status"] == "ok"
    assert written["artifacts"][0]["ref"].startswith("@artifacts/")
    assert read["status"] == "ok"
    assert read["output"] == {"text": "seed text"}
    assert read["artifact_store"]["status"] == "ok"
    assert read["artifacts"][0]["filename"] == "report.txt"
    assert read["artifacts"][0]["size_bytes"] == len("SEED TEXT")


def test_execute_workflow_js_node_artifact_fs_list_stat_and_mkdir(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = """
exports.run = function(input, api) {
  api.fs.mkdir("report", "nested");
  api.fs.writeText("report", "nested/a.txt", "alpha");
  const listing = api.fs.list("report", "nested");
  const stat = api.fs.stat("report", "nested/a.txt");
  return {output: {names: listing.entries.map(x => x.name), type: stat.type, size: stat.size}};
};
"""

    out = svc.execute_workflow_js(
        profile="node",
        request={
            "request_id": "req-js-artifact-fs",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "artifact_outputs": [{"name": "report", "path_mask": "*.txt", "recursive": True, "media_type": "text/plain"}],
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"names": ["a.txt"], "type": "file", "size": len("alpha")}


def test_execute_workflow_js_node_enforces_artifact_root_permissions(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    write_seed = """
exports.run = function(input, api) {
  api.fs.writeText("seed", "", "seed text");
  return {output: true};
};
"""
    write_input = """
exports.run = function(input, api) {
  api.fs.writeText("seed", "changed.txt", "should fail");
  return {output: true};
};
"""
    escape_output = """
exports.run = function(input, api) {
  api.fs.writeText("report", "../escape.txt", "should fail");
  return {output: true};
};
"""
    seed = svc.execute_workflow_js(
        profile="node",
        request={
            "request_id": "req-js-root-seed",
            "module_source": write_seed,
            "module_sha256": hashlib.sha256(write_seed.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "artifact_outputs": [{"name": "seed", "filename": "seed.txt", "media_type": "text/plain"}],
        },
    )
    input_write = svc.execute_workflow_js(
        profile="node",
        request={
            "request_id": "req-js-root-readonly",
            "module_source": write_input,
            "module_sha256": hashlib.sha256(write_input.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "artifact_inputs": [{"name": "seed", "ref": seed["artifacts"][0]["ref"]}],
        },
    )
    escaped = svc.execute_workflow_js(
        profile="node",
        request={
            "request_id": "req-js-root-escape",
            "module_source": escape_output,
            "module_sha256": hashlib.sha256(escape_output.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "artifact_outputs": [{"name": "report", "filename": "report.txt", "media_type": "text/plain"}],
        },
    )

    assert seed["status"] == "ok"
    assert input_write["status"] == "error"
    assert input_write["error"]["code"] == "host_call_failed"
    assert "artifact_root_unavailable:seed" in input_write["error"]["message"]
    assert escaped["status"] == "error"
    assert escaped["error"]["code"] == "host_call_failed"
    assert "artifact_path_escape" in escaped["error"]["message"]


def test_execute_workflow_js_node_brokered_http_allowed_and_denied(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = """
exports.run = function(input, api) {
  const response = api.http.fetch("https://example.test/data", {method: "POST", body_b64: "ZGF0YQ=="});
  return {output: {status_code: response.status_code, body_b64: response.body_b64}};
};
"""
    calls = []

    def fake_fetch(self, **kwargs):
        calls.append(dict(kwargs))
        return {
            "url": kwargs["url"],
            "status_code": 201,
            "headers": {"content-type": "text/plain"},
            "body_b64": "b2s=",
            "body_size": 2,
            "truncated": False,
        }

    monkeypatch.setattr("hosting.service.workflow_helpers.BrokeredHttpClient.fetch", fake_fetch)

    allowed = svc.execute_workflow_js(
        profile="node",
        sandbox_policy={
            "sandbox": {
                "enabled": True,
                "network": {"mode": "brokered_only"},
                "brokered_io": {"http": True},
            }
        },
        request={
            "request_id": "req-js-http-allowed",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
        },
    )
    denied = svc.execute_workflow_js(
        profile="node",
        request={
            "request_id": "req-js-http-denied",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
        },
    )

    assert allowed["status"] == "ok"
    assert allowed["output"] == {"status_code": 201, "body_b64": "b2s="}
    assert calls[0]["url"] == "https://example.test/data"
    assert calls[0]["method"] == "POST"
    assert denied["status"] == "error"
    assert denied["error"]["code"] == "host_call_failed"
    assert "unsupported_host_method:http.fetch" in denied["error"]["message"]


def test_workflow_js_node_resources_report_terminal_metrics(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    env = svc.workflow_js_environment_spec(profile="node", environment_name="workflow-js-node", javascript={})
    environment_key = str(env["environment_key"])

    def request(source: str, request_id: str, *, timeout_ms: int = 1000) -> dict:
        return {
            "request_id": request_id,
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "limits": {"timeout_ms": timeout_ms, "output_limit_bytes": 1024},
        }

    success_source = "exports.run = function() { return {output: {ok: true}}; };"
    error_source = "exports.run = function() { throw new Error('metric error'); };"
    timeout_source = "exports.run = function() { while (true) {} };"
    cancel_result: dict[str, object] = {}

    success = svc.execute_workflow_js(profile="node", request=request(success_source, "req-js-metrics-ok"))
    error = svc.execute_workflow_js(profile="node", request=request(error_source, "req-js-metrics-error"))
    timeout = svc.execute_workflow_js(profile="node", request=request(timeout_source, "req-js-metrics-timeout", timeout_ms=50))

    def run_cancel() -> None:
        cancel_result.update(
            svc.execute_workflow_js(
                profile="node",
                request=request(timeout_source, "req-js-metrics-cancel", timeout_ms=5000),
            )
        )

    thread = threading.Thread(target=run_cancel, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        status = svc.workflow_js_request_status(
            profile="node",
            environment_key=environment_key,
            request_id="req-js-metrics-cancel",
        )
        if dict(status.get("request") or {}).get("status") == "running":
            break
        time.sleep(0.05)
    canceled = svc.cancel_workflow_js_request(
        profile="node",
        environment_key=environment_key,
        request_id="req-js-metrics-cancel",
    )
    thread.join(timeout=5.0)
    resources = svc.workflow_js_resources(profile="node", environment_key=environment_key)
    metrics = resources["workflow_pool"]["metrics"]
    recent = {row["request_id"]: row["status"] for row in metrics["recent_requests"]}

    assert success["status"] == "ok"
    assert error["status"] == "error"
    assert timeout["error"]["code"] == "workflow_sandbox_timeout"
    assert canceled["canceled"] is True
    assert cancel_result["status"] == "canceled"
    assert metrics["active_calls"] == 0
    assert metrics["error_count"] >= 1
    assert metrics["timeout_count"] >= 1
    assert metrics["cancellation_count"] >= 1
    assert recent["req-js-metrics-ok"] == "ok"
    assert recent["req-js-metrics-error"] == "error"
    assert recent["req-js-metrics-timeout"] == "timeout"
    assert recent["req-js-metrics-cancel"] == "canceled"


def test_workflow_js_stream_emits_terminal_events_and_artifacts(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = """
exports.run = function(input, api) {
  console.log("stream start");
  api.progress({step: "write"});
  api.fs.writeText("report", "", "stream artifact");
  return {output: {done: true}};
};
"""

    opened = svc.workflow_js_stream_open(
        profile="node",
        request={
            "request_id": "req-js-stream-ok",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "limits": {"stream_max_events": 32},
            "artifact_outputs": [{"name": "report", "filename": "stream.txt", "media_type": "text/plain"}],
        },
    )
    normalized_events: list[dict] = []
    deadline = time.time() + 10.0
    received = {}
    while time.time() < deadline:
        received = svc.workflow_js_event_subscribe(stream_id=opened["stream_id"], max_items=8)
        normalized_events.extend(list(received.get("normalized_events") or []))
        if any(row["kind"] == "done" for row in normalized_events):
            break
        time.sleep(0.05)
    status = svc.workflow_js_request_status(
        profile="node",
        environment_key=opened["environment_key"],
        request_id="req-js-stream-ok",
    )
    svc.workflow_js_stream_close(stream_id=opened["stream_id"])

    event_types = [row["kind"] for row in normalized_events]
    assert opened["status"] == "ok"
    assert "started" in event_types
    assert "stdout" in event_types
    assert "host_call" in event_types
    assert "progress" in event_types
    assert "artifact" in event_types
    assert "result" in event_types
    assert event_types[-1] == "done"
    assert status["request"]["status"] == "ok"
    assert status["request"]["stream_event_count"] >= len(normalized_events)
    assert any(row.get("filename") == "stream.txt" for row in normalized_events if row["kind"] == "artifact")
    assert any(row.get("kind") == "host_call" and row.get("call_id") for row in normalized_events)


def test_workflow_js_stream_retention_reports_dropped_events(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = """
exports.run = function(input, api) {
  for (let i = 0; i < 8; i++) {
    api.progress({step: i});
  }
  return {output: {done: true}};
};
"""

    opened = svc.workflow_js_stream_open(
        profile="node",
        request={
            "request_id": "req-js-stream-retention",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "limits": {"stream_max_events": 3},
        },
    )
    deadline = time.time() + 10.0
    received = {}
    while time.time() < deadline:
        received = svc.workflow_js_event_subscribe(stream_id=opened["stream_id"], max_items=10)
        if bool(received.get("closed")):
            break
        time.sleep(0.05)
    svc.workflow_js_stream_close(stream_id=opened["stream_id"])

    assert opened["status"] == "ok"
    loss = dict(dict(received.get("batch") or {}).get("loss") or {})
    assert sum(int(value or 0) for value in loss.values()) > 0
    assert len(dict(received.get("batch") or {}).get("frames") or []) <= 3


def test_workflow_js_stream_cancel_routes_to_worker_cancel(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "exports.run = function() { while (true) {} };"

    opened = svc.workflow_js_stream_open(
        profile="node",
        request={
            "request_id": "req-js-stream-cancel",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "limits": {"timeout_ms": 5000},
        },
    )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        status = svc.workflow_js_request_status(
            profile="node",
            environment_key=opened["environment_key"],
            request_id="req-js-stream-cancel",
        )
        if dict(status.get("request") or {}).get("status") == "running":
            break
        time.sleep(0.05)

    sent = svc.workflow_js_stream_send(stream_id=opened["stream_id"], message={"action": "cancel"})
    events: list[dict] = []
    deadline = time.time() + 10.0
    while time.time() < deadline:
        received = svc.workflow_js_event_subscribe(stream_id=opened["stream_id"], max_items=8)
        events.extend(list(received.get("normalized_events") or []))
        if any(row["kind"] == "done" for row in events):
            break
        time.sleep(0.05)
    status = svc.workflow_js_request_status(
        profile="node",
        environment_key=opened["environment_key"],
        request_id="req-js-stream-cancel",
    )
    svc.workflow_js_stream_close(stream_id=opened["stream_id"])

    event_types = [row["kind"] for row in events]
    assert sent["accepted"] is True
    assert sent["worker_cancel"]["canceled"] is True
    assert "canceled" in event_types
    assert event_types[-1] == "done"
    assert status["request"]["status"] == "canceled"


def test_execute_workflow_python_node_returns_contract_envelope(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'accepted': payload['value'] == 7}, 'state_patch': {'seen': True}}\n"

    try:
        out = svc.execute_workflow_python(
            profile="node",
            engine_id="wf-node",
            request={
                "request_id": "req-node",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {"value": 7},
                "limits": {"output_limit_bytes": 1024},
            },
        )
    finally:
        svc.shutdown("wf-node", timeout_seconds=5.0)

    assert out["status"] == "ok"
    assert out["ok"] is True
    assert out["profile"] == "node"
    assert out["request_id"] == "req-node"
    assert out["output"] == {"accepted": True}
    assert out["state_patch"] == {"seen": True}
    assert "progress" in out["contract"]["stream_event_types"]


def test_execute_workflow_python_node_keeps_and_reaps_warm_child_process(tmp_path: Path) -> None:
    from hosting.sandbox.workflow_python_node_runtime import _ACTIVE_NODE_PROCS

    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'done': True}}\n"
    baseline = len(list(_ACTIVE_NODE_PROCS))

    try:
        out = svc.execute_workflow_python(
            profile="node",
            request={
                "request_id": "req-node-reap",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
            },
        )

        assert out["status"] == "ok"
        assert len(list(_ACTIVE_NODE_PROCS)) == baseline + 1
    finally:
        svc._workflow_python_node_runtime_registry().shutdown()
    assert len(list(_ACTIVE_NODE_PROCS)) == baseline


def test_execute_workflow_python_node_does_not_call_helper_proxy(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'value': payload['value']}}\n"

    def fail_proxy(**_kwargs):
        raise AssertionError("node execution should not use execute_workflow_python_helper")

    monkeypatch.setattr(svc, "proxy_rpc_call", fail_proxy)

    out = svc.execute_workflow_python(
        profile="node",
        engine_id="wf-node-direct",
        request={
            "request_id": "req-node-direct",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {"value": 11},
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"value": 11}


def test_execute_workflow_python_node_enforces_import_allowlist(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "import math\n\ndef run(payload):\n    return {'output': math.sqrt(payload['value'])}\n"
    base_request = {
        "request_id": "req-node-import",
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg",
        "workflow_id": "wf",
        "package_source_digest": "digest",
        "operation": "run",
        "payload": {"value": 9},
    }

    denied = svc.execute_workflow_python(profile="node", engine_id="wf-node-import-denied", request=base_request)
    wrong_allowlist = svc.execute_workflow_python(
        profile="node",
        engine_id="wf-node-import-wrong",
        request={**base_request, "request_id": "req-node-import-wrong", "python": {"import_allowlist": ["json"]}},
    )
    allowed = svc.execute_workflow_python(
        profile="node",
        engine_id="wf-node-import-allowed",
        request={**base_request, "request_id": "req-node-import-allowed", "python": {"import_allowlist": ["math"]}},
    )

    assert denied["status"] == "error"
    assert denied["error"]["code"] == "workflow_sandbox_runtime_error"
    assert "import" in denied["error"]["message"].lower()
    assert wrong_allowlist["status"] == "error"
    assert wrong_allowlist["error"]["code"] == "workflow_sandbox_runtime_error"
    assert "math" in wrong_allowlist["error"]["message"]
    assert allowed["status"] == "ok"
    assert allowed["output"] == 3.0


def test_execute_workflow_python_node_rejects_environment_key_mismatch(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'ok': True}}\n"

    out = svc.execute_workflow_python(
        profile="node",
        environment_key="wrong-node-key",
        request={
            "request_id": "req-node-env-mismatch",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
        },
    )

    assert out["status"] == "error"
    assert out["reason"] == "environment_key_mismatch"
    assert out["derived_environment_key"] != "wrong-node-key"


def test_execute_workflow_python_node_rejects_dependency_execution_without_verified_environment(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'ok': True}}\n"

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-unverified-env",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"package_pins": {"demo-dependency": "1.0.0"}},
        },
    )

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_python_environment_not_prepared"
    assert out["error"]["detail"]["environment_key"]
    assert out["error"]["detail"]["install_status"]["install_plan_status"] == "missing"


def test_execute_workflow_python_node_rejects_uv_execution_without_verified_environment(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'ok': True}}\n"

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-unverified-uv-env",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"uv": {"pyproject_toml": "[project]\nname='demo'\nversion='0.0.0'\n"}},
        },
    )

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_python_environment_not_prepared"
    assert out["error"]["detail"]["install_status"]["uv_install_plan_status"] == "missing"


def test_execute_workflow_python_node_rejects_dependency_execution_without_install_receipt(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    python = {"package_pins": {"demo-dependency": "1.0.0"}}
    spec = svc.workflow_python_environment_spec(
        profile="node",
        environment_name="workflow-python-node",
        python=python,
    )
    prepared = svc.workflow_python_prepare_environment(
        environment_name="workflow-python-node",
        python=python,
        package_id="pkg",
        workflow_id="wf",
    )
    locked = svc.workflow_python_lock_environment(environment=dict(prepared["environment"]))
    source = "def run(payload):\n    return {'output': {'ok': True}}\n"

    out = svc.execute_workflow_python(
        profile="node",
        environment_key=str(spec["environment_key"]),
        request={
            "request_id": "req-node-missing-receipt",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": python,
        },
    )

    assert locked["status"] == "ok"
    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_python_environment_unverified"
    assert out["error"]["detail"]["install_status"]["install_receipt_verification_status"] == "missing"


def test_execute_workflow_python_node_uses_selected_verified_dependency_runtime(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    selected = {"python_executable": sys.executable, "python_source": "venv"}
    calls = []
    real_runtime_manager = svc._workflow_python_runtime_manager()

    class FakeRuntimeManager:
        def environment_spec(self, **kwargs):
            return real_runtime_manager.environment_spec(**kwargs)

        def select_runtime_python(self, **kwargs):
            calls.append(dict(kwargs))
            return selected

    monkeypatch.setattr(
        svc,
        "workflow_python_verify_install_receipt",
        lambda **_kwargs: {
            "status": "ok",
            "install_status": {
                "install_plan_status": "ok",
                "install_execution_status": "ok",
                "install_receipt_status": "ok",
                "install_receipt_verification_status": "ok",
            },
        },
    )
    monkeypatch.setattr(svc, "_workflow_python_runtime_manager", lambda: FakeRuntimeManager())
    source = "def run(payload):\n    return {'output': {'ok': True}}\n"

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-verified-env",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"package_pins": {"demo-dependency": "1.0.0"}},
        },
    )

    assert out["status"] == "ok"
    assert calls
    assert calls[0]["environment"]["environment_name"] == "workflow-python-node"
    assert out["audit"]["runtime"]["python_executable"] == sys.executable


def test_execute_workflow_python_node_uses_selected_verified_uv_runtime(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    selected = {"python_executable": sys.executable, "python_source": "uv"}
    calls = []
    real_runtime_manager = svc._workflow_python_runtime_manager()

    class FakeRuntimeManager:
        def environment_spec(self, **kwargs):
            return real_runtime_manager.environment_spec(**kwargs)

        def select_runtime_python(self, **kwargs):
            calls.append(dict(kwargs))
            return selected

    monkeypatch.setattr(
        svc,
        "workflow_python_verify_install_receipt",
        lambda **_kwargs: {
            "status": "ok",
            "install_status": {
                "uv_install_plan_status": "planned",
                "uv_install_execution_status": "ok",
                "uv_install_receipt_status": "ok",
                "uv_install_receipt_verification_status": "ok",
            },
        },
    )
    monkeypatch.setattr(svc, "_workflow_python_runtime_manager", lambda: FakeRuntimeManager())
    source = "def run(payload):\n    return {'output': {'ok': True}}\n"

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-verified-uv-env",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"uv": {"pyproject_toml": "[project]\nname='demo'\nversion='0.0.0'\n"}},
        },
    )

    assert out["status"] == "ok"
    assert calls
    assert calls[0]["environment"]["environment_name"] == "workflow-python-node"
    assert out["audit"]["runtime"]["python_executable"] == sys.executable


def test_execute_workflow_python_node_uses_separate_pools_for_incompatible_identities(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'value': payload['value']}}\n"
    base_request = {
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg",
        "workflow_id": "wf",
        "package_source_digest": "digest",
        "operation": "run",
    }

    first = svc.execute_workflow_python(
        profile="node",
        request={**base_request, "request_id": "req-node-identity-a", "payload": {"value": "a"}},
    )
    second = svc.execute_workflow_python(
        profile="node",
        request={
            **base_request,
            "request_id": "req-node-identity-b",
            "payload": {"value": "b"},
            "python": {"import_allowlist": ["math"]},
        },
    )
    pools = svc._workflow_python_pool_registry().resources()["pools"]

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert first["environment_key"] != second["environment_key"]
    assert f"workflow_python/{first['environment_key']}" in pools
    assert f"workflow_python/{second['environment_key']}" in pools


def test_execute_workflow_python_node_runs_same_code_concurrently_by_capacity(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "import time\n\ndef run(payload):\n    progress({'slot': payload['slot']})\n    time.sleep(0.35)\n    return {'output': {'slot': payload['slot']}}\n"
    base_request = {
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg",
        "workflow_id": "wf",
        "package_source_digest": "digest",
        "operation": "run",
        "python": {"import_allowlist": ["time"]},
        "limits": {"timeout_ms": 2000},
    }
    results: dict[str, dict] = {}

    def call(slot: str) -> None:
        results[slot] = svc.execute_workflow_python(
            profile="node",
            capacity=2,
            request={**base_request, "request_id": f"req-node-concurrent-{slot}", "payload": {"slot": slot}},
        )

    first = threading.Thread(target=call, args=("a",))
    second = threading.Thread(target=call, args=("b",))
    started_at = time.monotonic()
    first.start()
    second.start()
    deadline = time.time() + 2.0
    active_snapshot = {}
    while time.time() < deadline:
        resources = svc.workflow_python_resources(profile="node", python={"import_allowlist": ["time"]})
        if int(resources.get("workflow_python_active_calls") or 0) >= 2 and int(resources.get("workflow_python_active_process_count") or 0) >= 2:
            active_snapshot = resources
            break
        time.sleep(0.02)
    first.join(timeout=3.0)
    second.join(timeout=3.0)
    elapsed = time.monotonic() - started_at

    assert active_snapshot["workflow_python_active_calls"] == 2
    assert active_snapshot["workflow_python_active_process_count"] == 2
    assert results["a"]["status"] == "ok"
    assert results["b"]["status"] == "ok"
    assert {results["a"]["output"]["slot"], results["b"]["output"]["slot"]} == {"a", "b"}
    assert elapsed < 1.3


def test_execute_workflow_python_node_routes_different_jobs_through_same_capacity_pool(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source_a = "import time\n\ndef run(payload):\n    time.sleep(0.25)\n    return {'output': {'job': 'a'}}\n"
    source_b = "import time\n\ndef run(payload):\n    time.sleep(0.25)\n    return {'output': {'job': 'b'}}\n"

    def request(source: str, request_id: str) -> dict:
        return {
            "request_id": request_id,
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"import_allowlist": ["time"]},
            "limits": {"timeout_ms": 2000},
        }

    results: dict[str, dict] = {}
    threads = [
        threading.Thread(target=lambda: results.update(a=svc.execute_workflow_python(profile="node", capacity=2, request=request(source_a, "req-node-route-a")))),
        threading.Thread(target=lambda: results.update(b=svc.execute_workflow_python(profile="node", capacity=2, request=request(source_b, "req-node-route-b")))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)
    resources = svc.workflow_python_resources(profile="node", environment_key=results["a"]["environment_key"])

    assert results["a"]["environment_key"] == results["b"]["environment_key"]
    assert results["a"]["output"] == {"job": "a"}
    assert results["b"]["output"] == {"job": "b"}
    assert resources["workflow_pool"]["metrics"]["desired_capacity"] == 2
    recent_request_ids = {row["request_id"] for row in resources["workflow_pool"]["metrics"]["recent_requests"][-2:]}
    assert recent_request_ids == {"req-node-route-a", "req-node-route-b"}


def test_execute_workflow_python_node_reuses_warm_worker_for_compatible_sequential_requests(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "import os\n\ndef run(payload):\n    return {'output': {'pid': os.getpid(), 'value': payload['value']}}\n"
    base_request = {
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg",
        "workflow_id": "wf",
        "package_source_digest": "digest",
        "operation": "run",
        "python": {"import_allowlist": ["os"]},
        "limits": {"timeout_ms": 2000},
    }

    first = svc.execute_workflow_python(
        profile="node",
        request={**base_request, "request_id": "req-node-warm-a", "payload": {"value": "a"}},
    )
    resources = svc.workflow_python_resources(profile="node", python={"import_allowlist": ["os"]})
    second = svc.execute_workflow_python(
        profile="node",
        request={**base_request, "request_id": "req-node-warm-b", "payload": {"value": "b"}},
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert first["output"]["pid"] == second["output"]["pid"]
    assert resources["workflow_python_idle_process_count"] == 1
    assert resources["workflow_python_active_process_count"] == 0
    svc._workflow_python_node_runtime_registry().shutdown()


def test_execute_workflow_python_node_routes_edited_code_to_new_revision_worker(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source_a = "import os\n\ndef run(payload):\n    return {'output': {'pid': os.getpid(), 'version': 'a'}}\n"
    source_b = "import os\n\ndef run(payload):\n    return {'output': {'pid': os.getpid(), 'version': 'b'}}\n"

    def request(source: str, request_id: str) -> dict:
        return {
            "request_id": request_id,
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"import_allowlist": ["os"]},
            "limits": {"timeout_ms": 2000},
        }

    first = svc.execute_workflow_python(profile="node", capacity=1, request=request(source_a, "req-node-revision-a"))
    second = svc.execute_workflow_python(profile="node", capacity=1, request=request(source_b, "req-node-revision-b"))
    resources = svc.workflow_python_resources(profile="node", environment_key=first["environment_key"])

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert first["output"]["version"] == "a"
    assert second["output"]["version"] == "b"
    assert first["output"]["pid"] != second["output"]["pid"]
    assert resources["workflow_python_idle_process_count"] == 1
    svc._workflow_python_node_runtime_registry().shutdown()


def test_execute_workflow_python_node_trims_idle_warm_workers_on_capacity_shrink(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "import time\n\ndef run(payload):\n    time.sleep(0.2)\n    return {'output': {'slot': payload['slot']}}\n"
    base_request = {
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg",
        "workflow_id": "wf",
        "package_source_digest": "digest",
        "operation": "run",
        "python": {"import_allowlist": ["time"]},
        "limits": {"timeout_ms": 2000},
    }
    results: dict[str, dict] = {}

    def run(slot: str) -> None:
        results[slot] = svc.execute_workflow_python(
            profile="node",
            capacity=2,
            request={**base_request, "request_id": f"req-node-trim-{slot}", "payload": {"slot": slot}},
        )

    threads = [threading.Thread(target=run, args=(slot,)) for slot in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20.0)

    assert all(not thread.is_alive() for thread in threads)
    assert set(results) == {"a", "b"}
    assert results["a"]["status"] == "ok"
    assert results["b"]["status"] == "ok"
    environment_key = results["a"]["environment_key"]
    resources = svc.workflow_python_resources(profile="node", environment_key=environment_key)
    assert resources["workflow_python_idle_process_count"] == 2

    resized = svc.set_workflow_python_capacity(profile="node", environment_key=environment_key, capacity=1)

    assert resized["status"] == "ok"
    assert resized["capacity"] == 1
    assert resized["node_runtime_trim"]["stopped_count"] == 1
    resources_after = svc.workflow_python_resources(profile="node", environment_key=environment_key)
    assert resources_after["workflow_python_idle_process_count"] == 1
    svc._workflow_python_node_runtime_registry().shutdown()


def test_execute_workflow_python_node_recycles_idle_worker_when_policy_identity_changes(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "import os\n\ndef run(payload):\n    return {'output': {'pid': os.getpid()}}\n"
    request = {
        "module_source": source,
        "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "package_id": "pkg",
        "workflow_id": "wf",
        "package_source_digest": "digest",
        "operation": "run",
        "payload": {},
        "python": {"import_allowlist": ["os"]},
        "limits": {"timeout_ms": 2000},
    }

    first = svc.execute_workflow_python(
        profile="node",
        environment_name="workflow-python-node-recycle-policy",
        request={**request, "request_id": "req-node-recycle-policy-a"},
    )
    first_resources = svc.workflow_python_resources(profile="node", environment_key=first["environment_key"])
    second = svc.execute_workflow_python(
        profile="node",
        environment_name="workflow-python-node-recycle-policy",
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_python_node_recycle_policy_v1"}},
        request={**request, "request_id": "req-node-recycle-policy-b"},
    )
    old_resources = svc.workflow_python_resources(profile="node", environment_key=first["environment_key"])

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert first["environment_key"] != second["environment_key"]
    assert first["output"]["pid"] != second["output"]["pid"]
    assert first_resources["workflow_python_idle_process_count"] == 1
    assert second["metrics"]["node_runtime_recycle"]["stopped_count"] == 1
    assert second["metrics"]["node_runtime_recycle"]["stopped"][0]["reason"] == "environment_identity_changed"
    assert old_resources["workflow_python_idle_process_count"] == 0
    svc._workflow_python_node_runtime_registry().shutdown()


def test_workflow_python_node_resources_recycle_unhealthy_idle_workers(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "import os\n\ndef run(payload):\n    return {'output': {'pid': os.getpid()}}\n"
    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-recycle-unhealthy",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"import_allowlist": ["os"]},
            "limits": {"timeout_ms": 2000},
        },
    )
    environment_key = str(out["environment_key"])
    pid = int(out["output"]["pid"])

    terminate_process_tree(pid, timeout_seconds=2.0)
    deadline = time.time() + 5.0
    resources = {}
    while time.time() < deadline:
        resources = svc.workflow_python_resources(profile="node", environment_key=environment_key)
        if dict(resources.get("node_runtime") or {}).get("recycle", {}).get("stopped_count"):
            break
        time.sleep(0.05)

    recycle = dict(dict(resources.get("node_runtime") or {}).get("recycle") or {})
    assert out["status"] == "ok"
    assert recycle["stopped_count"] == 1
    assert recycle["stopped"][0]["reason"] == "unhealthy"
    assert resources["workflow_python_idle_process_count"] == 0
    svc._workflow_python_node_runtime_registry().shutdown()


def test_execute_workflow_python_node_does_not_promote_untrusted_artifact_refs(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    return {\n"
        "        'output': {'path': '/tmp/report.csv'},\n"
        "        'artifacts': [{'path': '/tmp/report.csv', 'name': 'report'}],\n"
        "    }\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        engine_id="wf-node-artifacts",
        request={
            "request_id": "req-node-artifacts",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"path": "/tmp/report.csv"}
    assert out["artifacts"] == []
    assert out["artifact_store"]["status"] == "unavailable"


def test_execute_workflow_python_node_collects_declared_output_artifact(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    f = open(artifact_outputs['report'], 'w')\n"
        "    f.write('hello artifact')\n"
        "    f.close()\n"
        "    return {'output': {'done': True}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-output-artifact",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_outputs": [{"name": "report", "filename": "report.txt", "media_type": "text/plain"}],
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"done": True}
    assert out["artifact_store"]["status"] == "ok"
    assert out["artifacts"] == [
        {
            "name": "report",
            "kind": "ref",
            "ref": out["artifacts"][0]["ref"],
            "filename": "report.txt",
            "media_type": "text/plain",
            "size_bytes": len("hello artifact"),
            "encoding": None,
        }
    ]
    assert out["artifacts"][0]["ref"].startswith("@artifacts/")


def test_execute_workflow_python_node_reads_declared_input_artifact_ref(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    write_source = (
        "def run(payload):\n"
        "    f = open(artifact_outputs['seed'], 'w')\n"
        "    f.write('seed text')\n"
        "    f.close()\n"
        "    return {'output': {'written': True}}\n"
    )
    read_source = (
        "def run(payload):\n"
        "    f = open(artifact_inputs['seed'], 'r')\n"
        "    text = f.read()\n"
        "    f.close()\n"
        "    return {'output': {'text': text}}\n"
    )

    written = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-input-artifact-write",
            "module_source": write_source,
            "module_sha256": hashlib.sha256(write_source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_outputs": [{"name": "seed", "filename": "seed.txt", "media_type": "text/plain"}],
        },
    )
    read = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-input-artifact-read",
            "module_source": read_source,
            "module_sha256": hashlib.sha256(read_source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_inputs": [{"name": "seed", "ref": written["artifacts"][0]["ref"]}],
        },
    )

    assert written["status"] == "ok"
    assert read["status"] == "ok"
    assert read["output"] == {"text": "seed text"}


def test_execute_workflow_python_node_rejects_undeclared_artifact_file_access(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    f = open('outside.txt', 'w')\n"
        "    f.write('nope')\n"
        "    f.close()\n"
        "    return {'output': {'done': True}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-artifact-outside",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_outputs": [{"name": "report", "filename": "report.txt", "media_type": "text/plain"}],
        },
    )

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_sandbox_runtime_error"
    assert "artifact output path not allowed" in out["error"]["message"]
    assert out["artifacts"] == []


def test_execute_workflow_python_node_reads_inline_input_artifact(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    f = open(artifact_inputs['seed'], 'r')\n"
        "    text = f.read()\n"
        "    f.close()\n"
        "    return {'output': {'text': text}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-inline-input-artifact",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_inputs": [
                {
                    "name": "seed",
                    "kind": "inline",
                    "filename": "seed.txt",
                    "text": "inline seed",
                    "media_type": "text/plain",
                    "encoding": "utf-8",
                    "max_bytes": 1024,
                    "ttl": "advisory",
                }
            ],
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"text": "inline seed"}


def test_execute_workflow_python_node_host_api_reads_and_writes_artifact_roots(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    described = host.describe()\n"
        "    seed = host.fs_read_text('seed')['text']\n"
        "    host.fs_mkdir('report', 'nested')\n"
        "    written = host.fs_write_text('report', 'nested/report.txt', seed.upper())\n"
        "    stat = host.fs_stat('report', 'nested/report.txt')\n"
        "    schemas = {row['name']: row['args_schema'] for row in described['method_descriptions']}\n"
        "    return {'output': {'methods': described['methods'], 'schemas': schemas, 'transport': described['transport'], 'seed': seed, 'bytes': written['bytes'], 'size': stat['size']}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-host-api-artifacts",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_inputs": [
                {
                    "name": "seed",
                    "kind": "inline",
                    "filename": "seed.txt",
                    "text": "host api seed",
                    "media_type": "text/plain",
                }
            ],
            "artifact_outputs": [{"name": "report", "path_mask": "*.txt", "media_type": "text/plain", "recursive": True}],
        },
    )

    assert out["status"] == "ok"
    assert "host.describe" in out["output"]["methods"]
    assert "fs.write_text" in out["output"]["schemas"]
    assert out["output"]["schemas"]["fs.write_text"]["properties"]["root_id"]["type"] == "string"
    assert out["output"]["schemas"]["fs.write_text"]["properties"]["text"]["type"] == "string"
    assert out["output"]["transport"]["async_capable"] is True
    assert out["output"]["transport"]["host_call_id"] is True
    assert out["output"]["transport"]["out_of_order_responses"] is True
    assert out["output"]["seed"] == "host api seed"
    assert out["output"]["bytes"] == len("HOST API SEED")
    assert out["output"]["size"] == len("HOST API SEED")
    assert out["artifacts"][0]["ref"]


def test_workflow_python_node_host_api_matches_out_of_order_responses() -> None:
    registry = WorkflowPythonNodeRuntimeRegistry()
    completed: list[str] = []
    completed_lock = threading.Lock()
    source = (
        "import threading\n"
        "import time\n"
        "def run(payload):\n"
        "    results = {}\n"
        "    errors = []\n"
        "    def invoke(name, delay):\n"
        "        try:\n"
        "            results[name] = host.call('test.delay', {'name': name, 'delay': delay})\n"
        "        except Exception as exc:\n"
        "            errors.append(str(exc))\n"
        "    slow = threading.Thread(target=invoke, args=('slow', 0.20))\n"
        "    fast = threading.Thread(target=invoke, args=('fast', 0.01))\n"
        "    slow.start()\n"
        "    time.sleep(0.02)\n"
        "    fast.start()\n"
        "    slow.join()\n"
        "    fast.join()\n"
        "    if errors:\n"
        "        raise Exception(';'.join(errors))\n"
        "    return {'output': results}\n"
    )

    def dispatch(call: dict) -> dict:
        args = dict(call.get("arguments") or {})
        time.sleep(float(args.get("delay") or 0.0))
        name = str(args.get("name") or "")
        with completed_lock:
            completed.append(name)
        return {"name": name, "host_call_id": str(call.get("host_call_id") or "")}

    try:
        out = registry.execute(
            {
                "request_id": "req-node-host-api-out-of-order",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "python": {"import_allowlist": ["threading", "time"]},
                "limits": {"timeout_ms": 5000},
            },
            python_executable=sys.executable,
            host_dispatcher=dispatch,
        )
    finally:
        registry.shutdown()

    assert out["ok"] is True
    assert completed == ["fast", "slow"]
    assert out["output"]["slow"]["name"] == "slow"
    assert out["output"]["fast"]["name"] == "fast"
    assert out["output"]["slow"]["host_call_id"] != out["output"]["fast"]["host_call_id"]


def test_execute_workflow_python_node_host_api_rejects_input_writes(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    host.fs_write_text('seed', 'seed.txt', 'mutated')\n"
        "    return {'output': {'ok': True}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-host-api-reject-input-write",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_inputs": [
                {
                    "name": "seed",
                    "kind": "inline",
                    "filename": "seed.txt",
                    "text": "immutable",
                    "media_type": "text/plain",
                }
            ],
        },
    )

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_sandbox_runtime_error"
    assert "artifact_root_unavailable:seed" in out["error"]["detail"]["message"]


def test_execute_workflow_python_node_host_api_respects_disabled_fs_namespace(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    describe_source = (
        "def run(payload):\n"
        "    described = host.describe()\n"
        "    return {'output': {'methods': described['methods'], 'policy': described['policy']}}\n"
    )
    read_source = (
        "def run(payload):\n"
        "    return {'output': host.fs_read_text('seed')}\n"
    )
    sandbox_policy = {"sandbox": {"host_api": {"namespaces": {"fs": False}}}}
    artifact_inputs = [
        {
            "name": "seed",
            "kind": "inline",
            "filename": "seed.txt",
            "text": "disabled fs",
            "media_type": "text/plain",
        }
    ]

    described = svc.execute_workflow_python(
        profile="node",
        sandbox_policy=sandbox_policy,
        request={
            "request_id": "req-node-host-api-fs-disabled-describe",
            "module_source": describe_source,
            "module_sha256": hashlib.sha256(describe_source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_inputs": artifact_inputs,
        },
    )
    rejected = svc.execute_workflow_python(
        profile="node",
        sandbox_policy=sandbox_policy,
        request={
            "request_id": "req-node-host-api-fs-disabled-read",
            "module_source": read_source,
            "module_sha256": hashlib.sha256(read_source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_inputs": artifact_inputs,
        },
    )

    assert described["status"] == "ok"
    assert described["output"]["methods"] == ["host.describe"]
    assert described["output"]["policy"]["artifact_fs"] is False
    assert described["output"]["policy"]["namespaces"]["fs"] is False
    assert rejected["status"] == "error"
    assert rejected["error"]["code"] == "workflow_sandbox_runtime_error"
    assert "unsupported_host_method:fs.read_text" in rejected["error"]["detail"]["message"]


def test_execute_workflow_python_node_host_api_http_fetch_uses_broker_policy(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    described = host.describe()\n"
        "    fetched = host.http_fetch('https://example.com/api/node', method='POST', headers={'Content-Type': 'application/json'}, body_b64='e30=', timeout_seconds=5.0)\n"
        "    return {'output': {'methods': described['methods'], 'policy': described['policy'], 'status_code': fetched['status_code'], 'body_b64': fetched['body_b64']}}\n"
    )
    sandbox_policy = {
        "sandbox": {
            "enabled": True,
            "network": {
                "mode": "brokered_only",
                "allow_hosts": ["example.com"],
                "allow_url_prefixes": ["https://example.com/api/"],
            },
            "brokered_io": {"filesystem": False, "http": True, "subprocess": False},
        }
    }

    class _Resp:
        status = 201
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int = -1) -> bytes:
            return b'{"node":true}'

    def _fake_urlopen(req, timeout=0.0):
        assert req.full_url == "https://example.com/api/node"
        assert req.get_method() == "POST"
        assert timeout == 5.0
        assert req.headers["Content-type"] == "application/json"
        return _Resp()

    monkeypatch.setattr("hosting.sandbox.broker_http.urllib.request.urlopen", _fake_urlopen)

    out = svc.execute_workflow_python(
        profile="node",
        sandbox_policy=sandbox_policy,
        request={
            "request_id": "req-node-host-api-http-fetch",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
        },
    )

    assert out["status"] == "ok"
    assert "http.fetch" in out["output"]["methods"]
    assert out["output"]["policy"]["http"] is True
    assert out["output"]["policy"]["namespaces"]["http"] is True
    assert out["output"]["status_code"] == 201
    assert base64.b64decode(out["output"]["body_b64"]) == b'{"node":true}'


def test_execute_workflow_python_node_host_api_http_fetch_requires_broker_policy(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    describe_source = (
        "def run(payload):\n"
        "    described = host.describe()\n"
        "    return {'output': {'methods': described['methods'], 'policy': described['policy']}}\n"
    )
    fetch_source = (
        "def run(payload):\n"
        "    return {'output': host.http_fetch('https://example.com/api/node')}\n"
    )
    sandbox_policy = {
        "sandbox": {
            "enabled": True,
            "host_api": {"namespaces": {"http": True}},
            "network": {
                "mode": "disabled",
                "allow_hosts": ["example.com"],
                "allow_url_prefixes": ["https://example.com/api/"],
            },
            "brokered_io": {"filesystem": False, "http": True, "subprocess": False},
        }
    }

    described = svc.execute_workflow_python(
        profile="node",
        sandbox_policy=sandbox_policy,
        request={
            "request_id": "req-node-host-api-http-disabled-describe",
            "module_source": describe_source,
            "module_sha256": hashlib.sha256(describe_source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
        },
    )
    rejected = svc.execute_workflow_python(
        profile="node",
        sandbox_policy=sandbox_policy,
        request={
            "request_id": "req-node-host-api-http-disabled-fetch",
            "module_source": fetch_source,
            "module_sha256": hashlib.sha256(fetch_source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
        },
    )

    assert described["status"] == "ok"
    assert "http.fetch" not in described["output"]["methods"]
    assert described["output"]["policy"]["http"] is False
    assert described["output"]["policy"]["namespaces"]["http"] is False
    assert rejected["status"] == "error"
    assert rejected["error"]["code"] == "workflow_sandbox_runtime_error"
    assert "unsupported_host_method:http.fetch" in rejected["error"]["detail"]["message"]


def test_execute_workflow_python_node_collects_declared_inline_output_artifact(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    return {\n"
        "        'output': {'done': True},\n"
        "        'artifacts': [{'name': 'summary', 'text': 'inline output', 'media_type': 'text/plain'}],\n"
        "    }\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-inline-output-artifact",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_outputs": [{"name": "summary", "kind": "inline", "filename": "summary.txt", "media_type": "text/plain"}],
        },
    )

    assert out["status"] == "ok"
    assert out["artifacts"] == [
        {
            "name": "summary",
            "kind": "inline",
            "filename": "summary.txt",
            "media_type": "text/plain",
            "encoding": "utf-8",
            "size_bytes": len("inline output"),
            "text": "inline output",
        }
    ]


def test_execute_workflow_python_node_uses_policy_artifact_root_refs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    seed = project_root / "seed.txt"
    seed.write_text("project seed", encoding="utf-8")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    f = open(artifact_inputs['seed'], 'r')\n"
        "    text = f.read()\n"
        "    f.close()\n"
        "    out = open(artifact_outputs['report'], 'w')\n"
        "    out.write(text.upper())\n"
        "    out.close()\n"
        "    return {'output': {'done': True}}\n"
    )
    sandbox_policy = {"sandbox": {"artifact_roots": {"project": str(project_root)}}}

    out = svc.execute_workflow_python(
        profile="node",
        sandbox_policy=sandbox_policy,
        request={
            "request_id": "req-node-policy-artifact-root",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_inputs": [{"name": "seed", "ref": "@project/seed.txt"}],
            "artifact_outputs": [{"name": "report", "ref": "@project/out/report.txt", "filename": "ignored.txt", "media_type": "text/plain"}],
        },
    )

    assert out["status"] == "ok"
    assert out["artifacts"][0]["ref"] == "@project/out/report.txt"
    assert (project_root / "out" / "report.txt").read_text(encoding="utf-8") == "PROJECT SEED"


def test_execute_workflow_python_node_reads_recursive_masked_input_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "data" / "nested").mkdir(parents=True)
    (project_root / "data" / "a.txt").write_text("a", encoding="utf-8")
    (project_root / "data" / "nested" / "b.txt").write_text("b", encoding="utf-8")
    (project_root / "data" / "skip.bin").write_text("skip", encoding="utf-8")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "import os\n\n"
        "def run(payload):\n"
        "    found = []\n"
        "    for root, dirs, files in os.walk(artifact_inputs['dataset']):\n"
        "        for name in files:\n"
        "            rel = os.path.relpath(os.path.join(root, name), artifact_inputs['dataset']).replace('\\\\', '/')\n"
        "            found.append(rel)\n"
        "    return {'output': {'files': sorted(found)}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        sandbox_policy={"sandbox": {"artifact_roots": {"project": str(project_root)}}},
        request={
            "request_id": "req-node-masked-inputs",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"import_allowlist": ["os"]},
            "artifact_inputs": [{"name": "dataset", "ref": "@project/data", "path_mask": "*.txt", "recursive": True}],
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"files": ["a.txt", "nested/b.txt"]}


def test_execute_workflow_python_node_collects_recursive_masked_output_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "import os\n\n"
        "def run(payload):\n"
        "    root = artifact_outputs['reports']\n"
        "    os.makedirs(os.path.join(root, 'nested'), exist_ok=True)\n"
        "    open(os.path.join(root, 'a.txt'), 'w').write('a')\n"
        "    open(os.path.join(root, 'nested', 'b.txt'), 'w').write('b')\n"
        "    open(os.path.join(root, 'skip.bin'), 'w').write('skip')\n"
        "    return {'output': {'done': True}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        sandbox_policy={"sandbox": {"artifact_roots": {"project": str(project_root)}}},
        request={
            "request_id": "req-node-masked-outputs",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"import_allowlist": ["os"]},
            "artifact_outputs": [{"name": "reports", "ref": "@project/out", "path_mask": "*.txt", "recursive": True, "media_type": "text/plain"}],
        },
    )

    assert out["status"] == "ok"
    assert [row["relative_path"] for row in out["artifacts"]] == ["a.txt", "nested/b.txt"]
    assert [row["ref"] for row in out["artifacts"]] == ["@project/out/a.txt", "@project/out/nested/b.txt"]
    assert (project_root / "out" / "a.txt").read_text(encoding="utf-8") == "a"
    assert (project_root / "out" / "nested" / "b.txt").read_text(encoding="utf-8") == "b"


def test_execute_workflow_python_node_reads_inline_zip_input_artifact(tmp_path: Path) -> None:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.txt", "a")
        zf.writestr("nested/b.txt", "b")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "import os\n\n"
        "def run(payload):\n"
        "    found = []\n"
        "    for root, dirs, files in os.walk(artifact_inputs['project']):\n"
        "        for name in files:\n"
        "            found.append(os.path.relpath(os.path.join(root, name), artifact_inputs['project']).replace('\\\\', '/'))\n"
        "    return {'output': {'files': sorted(found)}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-inline-zip-input",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"import_allowlist": ["os"]},
            "artifact_inputs": [
                {
                    "name": "project",
                    "kind": "inline",
                    "filename": "project.zip",
                    "base64": base64.b64encode(raw.getvalue()).decode("ascii"),
                    "media_type": "application/zip",
                    "encoding": "zip",
                }
            ],
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"files": ["a.txt", "nested/b.txt"]}


def test_execute_workflow_python_node_runs_snippet_without_export(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "progress({'phase': 'snippet'})\nresult = {'output': {'value': payload['value'] + 1}, 'state_patch': {'mode': 'snippet'}}\n"

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-snippet",
            "execution_mode": "snippet",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {"value": 4},
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"value": 5}
    assert out["state_patch"] == {"mode": "snippet"}
    assert out["progress"] == {"phase": "snippet"}


def test_execute_workflow_python_node_runs_snippet_builder_request(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    request = build_workflow_python_node_snippet_request(
        request_id="req-node-snippet-builder",
        source="result = {'output': {'value': payload['value'] * 2}}\n",
        package_id="pkg",
        workflow_id="wf",
        payload={"value": 6},
    )

    out = svc.execute_workflow_python(profile="node", request=request)

    assert out["status"] == "ok"
    assert out["output"] == {"value": 12}
    assert out["audit"]["module_sha256"] == request["module_sha256"]


def test_execute_workflow_python_node_runs_multi_module_project_from_ref(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "src" / "pkg").mkdir(parents=True)
    (project_root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "src" / "pkg" / "util.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (project_root / "src" / "pkg" / "runner.py").write_text(
        "from pkg.util import add\n\n"
        "def run(payload):\n"
        "    return {'output': {'value': add(payload['a'], payload['b'])}}\n",
        encoding="utf-8",
    )
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    out = svc.execute_workflow_python(
        profile="node",
        sandbox_policy={"sandbox": {"artifact_roots": {"project": str(project_root)}}},
        request={
            "request_id": "req-node-project",
            "execution_mode": "project",
            "module_source": "",
            "module_sha256": hashlib.sha256(b"").hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "project-digest",
            "project": {"ref": "@project/src", "entrypoint": "pkg.runner", "callable": "run"},
            "payload": {"a": 2, "b": 3},
        },
    )

    assert out["status"] == "ok"
    assert out["output"] == {"value": 5}
    assert out["audit"]["package_source_digest"] == "project-digest"


def test_execute_workflow_python_node_project_imports_cannot_escape_staged_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "src" / "pkg").mkdir(parents=True)
    (project_root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "src" / "pkg" / "runner.py").write_text(
        "import outside_helper\n\n"
        "def run(payload):\n"
        "    return {'output': outside_helper.VALUE}\n",
        encoding="utf-8",
    )
    (project_root / "outside_helper.py").write_text("VALUE = 9\n", encoding="utf-8")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    out = svc.execute_workflow_python(
        profile="node",
        sandbox_policy={"sandbox": {"artifact_roots": {"project": str(project_root)}}},
        request={
            "request_id": "req-node-project-escape",
            "execution_mode": "project",
            "module_source": "",
            "module_sha256": hashlib.sha256(b"").hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "project-digest",
            "project": {"ref": "@project/src", "entrypoint": "pkg.runner", "callable": "run"},
            "payload": {},
        },
    )

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_sandbox_runtime_error"
    assert "outside_helper" in out["error"]["message"]


def test_execute_workflow_python_node_exports_many_outputs_as_inline_zip_without_takeover(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "import os\n\n"
        "def run(payload):\n"
        "    root = artifact_outputs['bundle']\n"
        "    os.makedirs(os.path.join(root, 'pkg'), exist_ok=True)\n"
        "    open(os.path.join(root, 'pkg', 'a.py'), 'w').write('A = 1')\n"
        "    open(os.path.join(root, 'pkg', 'b.py'), 'w').write('B = 2')\n"
        "    return {'output': {'done': True}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        sandbox_policy={"sandbox": {"artifact_roots": {"project": str(project_root)}}},
        request={
            "request_id": "req-node-inline-zip-output",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"import_allowlist": ["os"]},
            "artifact_outputs": [
                {
                    "name": "bundle",
                    "ref": "@project/producer-owned",
                    "path_mask": "*.py",
                    "recursive": True,
                    "export_inline_zip": True,
                    "filename": "bundle.zip",
                }
            ],
        },
    )

    assert out["status"] == "ok"
    assert out["artifacts"][0]["kind"] == "inline"
    assert out["artifacts"][0]["media_type"] == "application/zip"
    assert out["artifacts"][0]["ownership"] == "producer"
    assert not (project_root / "producer-owned").exists()
    data = base64.b64decode(out["artifacts"][0]["base64"])
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        assert sorted(zf.namelist()) == ["pkg/a.py", "pkg/b.py"]


def test_execute_workflow_python_node_host_takeover_copies_ref_outputs_to_host_store(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    open(artifact_outputs['report'], 'w').write('owned by host')\n"
        "    return {'output': {'done': True}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        sandbox_policy={"sandbox": {"artifact_roots": {"project": str(project_root)}}},
        request={
            "request_id": "req-node-host-takeover",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_outputs": [
                {
                    "name": "report",
                    "ref": "@project/worker/report.txt",
                    "filename": "report.txt",
                    "host_takeover": True,
                    "media_type": "text/plain",
                }
            ],
        },
    )

    assert out["status"] == "ok"
    assert out["artifacts"][0]["ref"].startswith("@artifacts/")
    assert out["artifacts"][0]["ownership"] == "host"
    assert out["artifacts"][0]["host_takeover"] is True
    assert not (project_root / "worker" / "report.txt").exists()


def test_execute_workflow_python_node_rejects_non_alias_artifact_refs(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'done': True}}\n"

    out = svc.execute_workflow_python(
        profile="node",
        request={
            "request_id": "req-node-bad-artifact-ref",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "artifact_inputs": [{"name": "seed", "ref": "workflow-artifact://old/seed.txt"}],
        },
    )

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_python_artifact_error"


def test_workflow_python_node_stream_returns_pending_worker_events(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': {'accepted': payload['value'] == 7}, 'progress': {'message': 'finished'}}\n"
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            request={
                "request_id": "req-node-stream",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {"value": 7},
                "limits": {"output_limit_bytes": 1024},
            },
        )
        events = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            received = svc.workflow_python_event_subscribe(stream_id=opened["stream_id"], max_items=8)
            events.extend(list(received.get("normalized_events") or []))
            if any(dict(row or {}).get("kind") == "done" for row in events):
                break
            time.sleep(0.05)
        status = svc.workflow_python_request_status(
            profile="node",
            environment_key=opened["environment_key"],
            request_id="req-node-stream",
        )
        closed = svc.workflow_python_stream_close(stream_id=opened["stream_id"])
    finally:
        svc.shutdown(str(opened.get("engine_id") or "workflow-python-node"), timeout_seconds=5.0)

    assert opened["status"] == "ok"
    assert [row["kind"] for row in events] == ["started", "log", "progress", "result", "done"]
    assert events[1]["logs"]["output_limit_bytes"] == 1024
    assert events[2]["message"] == "finished"
    assert events[3]["output"] == {"accepted": True}
    assert status["request"]["status"] == "ok"
    assert closed["closed"] is True


def test_workflow_python_node_stream_emits_runtime_progress_and_stdout(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    progress({'message': 'halfway'})\n    print('node stdout')\n    return {'output': {'done': True}}\n"
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            request={
                "request_id": "req-node-stream-progress",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"output_limit_bytes": 1024},
            },
        )
        events = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            received = svc.workflow_python_event_subscribe(stream_id=opened["stream_id"], max_items=8)
            events.extend(list(received.get("normalized_events") or []))
            if any(dict(row or {}).get("kind") == "done" for row in events):
                break
            time.sleep(0.05)
    finally:
        if opened:
            svc.workflow_python_stream_close(stream_id=str(opened.get("stream_id") or ""))

    event_types = [row["kind"] for row in events]
    assert event_types.index("progress") < event_types.index("result")
    assert any(row["kind"] == "stdout" and "node stdout" in row["text"] for row in events)


def test_workflow_python_node_stream_emits_opt_in_heartbeats_for_long_running_request(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "import time\n\ndef run(payload):\n    time.sleep(0.35)\n    return {'output': {'done': True}}\n"
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            request={
                "request_id": "req-node-stream-heartbeat",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "python": {"import_allowlist": ["time"]},
                "limits": {"timeout_ms": 2000, "output_limit_bytes": 1024, "heartbeat_interval_ms": 100},
            },
        )
        events = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            received = svc.workflow_python_event_subscribe(stream_id=opened["stream_id"], max_items=8)
            events.extend(list(received.get("normalized_events") or []))
            if any(dict(row or {}).get("kind") == "done" for row in events):
                break
            time.sleep(0.05)
        status = svc.workflow_python_request_status(
            profile="node",
            environment_key=str(opened["environment_key"]),
            request_id="req-node-stream-heartbeat",
        )
    finally:
        if opened:
            svc.workflow_python_stream_close(stream_id=str(opened.get("stream_id") or ""))

    heartbeats = [row for row in events if row["kind"] == "heartbeat"]
    assert opened["status"] == "ok"
    assert heartbeats
    assert heartbeats[0]["request_id"] == "req-node-stream-heartbeat"
    assert heartbeats[0]["status"] == "running"
    assert any(row["kind"] == "result" for row in events)
    assert status["request"]["status"] == "ok"
    assert status["request"]["stream_event_count"] >= len(heartbeats)


def test_workflow_python_node_stream_reports_bounded_retention_drops(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    for i in range(5):\n        progress({'value': i})\n    return {'output': {'done': True}}\n"
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            request={
                "request_id": "req-node-stream-retention",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"output_limit_bytes": 1024, "stream_max_events": 3},
            },
        )
        received = {}
        deadline = time.time() + 5.0
        while time.time() < deadline:
            received = svc.workflow_python_event_subscribe(stream_id=opened["stream_id"], max_items=10)
            if any(dict(row or {}).get("kind") == "done" for row in list(received.get("normalized_events") or [])):
                break
            time.sleep(0.05)
        status = svc.workflow_python_request_status(
            profile="node",
            environment_key=str(opened["environment_key"]),
            request_id="req-node-stream-retention",
        )
    finally:
        if opened:
            svc.workflow_python_stream_close(stream_id=str(opened.get("stream_id") or ""))

    assert opened["status"] == "ok"
    loss = dict(dict(received.get("batch") or {}).get("loss") or {})
    assert sum(int(value or 0) for value in loss.values()) > 0
    assert len(dict(received.get("batch") or {}).get("frames") or []) <= 3
    assert list(received.get("normalized_events") or [])[-1]["kind"] == "done"
    assert status["request"]["stream_event_count"] >= len(received.get("normalized_events") or [])


def test_workflow_python_node_stream_does_not_emit_untrusted_artifact_events(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    return {\n"
        "        'output': {'value': 1},\n"
        "        'artifacts': [{'ref': '../other-run/output'}],\n"
        "    }\n"
    )
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            request={
                "request_id": "req-node-stream-artifacts",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"output_limit_bytes": 1024},
            },
        )
        events = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            received = svc.workflow_python_event_subscribe(stream_id=opened["stream_id"], max_items=8)
            events.extend(list(received.get("normalized_events") or []))
            if any(dict(row or {}).get("kind") == "done" for row in events):
                break
            time.sleep(0.05)
    finally:
        if opened:
            svc.workflow_python_stream_close(stream_id=str(opened.get("stream_id") or ""))

    assert "artifact" not in [row["kind"] for row in events]
    result_events = [row for row in events if row["kind"] == "result"]
    assert result_events
    assert result_events[0]["artifacts"] == []


def test_workflow_python_node_stream_emits_declared_output_artifact_event(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    f = open(artifact_outputs['report'], 'w')\n"
        "    f.write('stream artifact')\n"
        "    f.close()\n"
        "    return {'output': {'done': True}}\n"
    )
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            request={
                "request_id": "req-node-stream-output-artifact",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"output_limit_bytes": 1024},
                "artifact_outputs": [{"name": "report", "filename": "stream.txt", "media_type": "text/plain"}],
            },
        )
        events = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            received = svc.workflow_python_event_subscribe(stream_id=opened["stream_id"], max_items=8)
            events.extend(list(received.get("normalized_events") or []))
            if any(dict(row or {}).get("kind") == "done" for row in events):
                break
            time.sleep(0.05)
    finally:
        if opened:
            svc.workflow_python_stream_close(stream_id=str(opened.get("stream_id") or ""))

    event_types = [row["kind"] for row in events]
    assert "artifact" in event_types
    assert event_types.index("artifact") < event_types.index("result")
    artifact_events = [row for row in events if row["kind"] == "artifact"]
    result_events = [row for row in events if row["kind"] == "result"]
    assert artifact_events[0]["ref"].startswith("@artifacts/")
    assert artifact_events[0]["filename"] == "stream.txt"
    assert result_events[0]["artifacts"][0]["ref"] == artifact_events[0]["ref"]
    assert result_events[0]["artifacts"][0]["filename"] == artifact_events[0]["filename"]


def test_execute_workflow_python_node_reports_structured_runtime_error(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    raise ValueError('bad node')\n"

    try:
        out = svc.execute_workflow_python(
            profile="node",
            engine_id="wf-node-error",
            request={
                "request_id": "req-node-error",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"timeout_ms": 1000, "output_limit_bytes": 1024},
            },
        )
    finally:
        svc.shutdown("wf-node-error", timeout_seconds=5.0)

    assert out["status"] == "error"
    assert out["ok"] is False
    assert out["error"]["code"] == "workflow_sandbox_runtime_error"
    assert "bad node" in out["error"]["message"]
    assert out["metrics"]["request"]["status"] == "error"


def test_execute_workflow_python_node_reports_timeout(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    while True:\n        pass\n"

    try:
        out = svc.execute_workflow_python(
            profile="node",
            engine_id="wf-node-timeout",
            request={
                "request_id": "req-node-timeout",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"timeout_ms": 50, "output_limit_bytes": 1024},
            },
        )
    finally:
        svc.shutdown("wf-node-timeout", timeout_seconds=5.0)

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_sandbox_timeout"
    assert out["metrics"]["request"]["reason"] == "workflow_sandbox_timeout"


def test_execute_workflow_python_node_reports_output_limit(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    return {'output': 'x' * 256}\n"

    out = svc.execute_workflow_python(
        profile="node",
        engine_id="wf-node-output-limit",
        request={
            "request_id": "req-node-output-limit",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "limits": {"timeout_ms": 1000, "output_limit_bytes": 32},
        },
    )

    assert out["status"] == "error"
    assert out["error"]["code"] == "workflow_sandbox_output_limit_exceeded"
    assert out["error"]["detail"]["output_limit_bytes"] == 32
    assert out["metrics"]["request"]["status"] == "error"


def test_execute_workflow_python_node_truncates_stdout_and_stderr_logs(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = (
        "def run(payload):\n"
        "    print('o' * 80)\n"
        "    import sys\n"
        "    print('e' * 80, file=sys.stderr)\n"
        "    return {'output': {'ok': True}}\n"
    )

    out = svc.execute_workflow_python(
        profile="node",
        engine_id="wf-node-truncate",
        request={
            "request_id": "req-node-truncate",
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "python": {"import_allowlist": ["sys"]},
            "limits": {"timeout_ms": 1000, "output_limit_bytes": 16},
        },
    )

    assert out["status"] == "ok"
    assert out["logs"]["stdout_truncated"] is True
    assert out["logs"]["stderr_truncated"] is True
    assert len(out["logs"]["stdout"].encode("utf-8")) == 16
    assert len(out["logs"]["stderr"].encode("utf-8")) == 16


def test_workflow_python_stream_cancel_routes_to_worker_cancel(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    base = svc._workflow_python_stream_base()
    opened = base.stream_open(
        environment_key="env-node-cancel",
        request_id="req-node-cancel",
        profile="node",
        factory=lambda key, cap: svc._workflow_python_worker_slot(
            engine_id="wf-node-cancel",
            environment_key=key,
            capacity=cap,
        ),
    )
    calls = []

    def fake_cancel(**kwargs):
        calls.append(dict(kwargs))
        return {"status": "ok", "canceled": True, "request_id": kwargs["request_id"]}

    monkeypatch.setattr(svc, "cancel_workflow_python_request", fake_cancel)

    out = svc.workflow_python_stream_send(
        stream_id=str(opened["stream_id"]),
        message={"action": "cancel", "reason": "test_cancel"},
    )

    assert out["accepted"] is True
    assert out["worker_cancel"]["canceled"] is True
    assert calls == [
        {
            "profile": "node",
            "environment_key": "env-node-cancel",
            "request_id": "req-node-cancel",
        }
    ]


def test_cancel_workflow_python_node_interrupts_active_execution(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    while True:\n        pass\n"
    env = svc.workflow_python_environment_spec(profile="node", environment_name="workflow-python-node", python={})
    environment_key = str(env["environment_key"])
    result: dict[str, object] = {}

    def run_node() -> None:
        result.update(
            svc.execute_workflow_python(
                profile="node",
                environment_name="workflow-python-node",
                request={
                    "request_id": "req-node-active-cancel",
                    "module_source": source,
                    "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    "package_id": "pkg",
                    "workflow_id": "wf",
                    "package_source_digest": "digest",
                    "operation": "run",
                    "payload": {},
                    "limits": {"timeout_ms": 5000, "output_limit_bytes": 1024},
                },
            )
        )

    thread = threading.Thread(target=run_node, daemon=True)
    thread.start()
    saw_running = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        status = svc.workflow_python_request_status(
            profile="node",
            environment_key=environment_key,
            request_id="req-node-active-cancel",
        )
        if dict(status.get("request") or {}).get("status") == "running":
            saw_running = True
            break
        time.sleep(0.05)

    canceled = svc.cancel_workflow_python_request(
        profile="node",
        environment_key=environment_key,
        request_id="req-node-active-cancel",
    )
    thread.join(timeout=5.0)

    assert saw_running is True
    assert canceled["canceled"] is True
    assert thread.is_alive() is False
    assert result["status"] == "canceled"
    assert dict(result["metrics"])["request"]["status"] == "canceled"


def test_workflow_python_node_stream_cancel_interrupts_active_execution(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    while True:\n        pass\n"
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            request={
                "request_id": "req-node-stream-active-cancel",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"timeout_ms": 5000, "output_limit_bytes": 1024},
            },
        )
        saw_running = False
        deadline = time.time() + 5.0
        while time.time() < deadline:
            status = svc.workflow_python_request_status(
                profile="node",
                environment_key=str(opened["environment_key"]),
                request_id="req-node-stream-active-cancel",
            )
            if dict(status.get("request") or {}).get("status") == "running":
                saw_running = True
                break
            time.sleep(0.05)

        sent = svc.workflow_python_stream_send(
            stream_id=str(opened["stream_id"]),
            message={"action": "cancel", "reason": "test_cancel"},
        )
        events = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            received = svc.workflow_python_event_subscribe(stream_id=str(opened["stream_id"]), max_items=8)
            events.extend(list(received.get("normalized_events") or []))
            if any(dict(row or {}).get("kind") == "done" for row in events):
                break
            time.sleep(0.05)
        status = svc.workflow_python_request_status(
            profile="node",
            environment_key=str(opened["environment_key"]),
            request_id="req-node-stream-active-cancel",
        )
    finally:
        if opened:
            svc.workflow_python_stream_close(stream_id=str(opened.get("stream_id") or ""))

    event_types = [row["kind"] for row in events]
    done_events = [row for row in events if row["kind"] == "done"]
    assert saw_running is True
    assert sent["accepted"] is True
    assert sent["worker_cancel"]["canceled"] is True
    assert "canceled" in event_types
    assert done_events and done_events[-1]["status"] == "canceled"
    assert status["request"]["status"] == "canceled"


def test_workflow_python_node_stream_reports_capacity_pressure_and_recovers_after_cancel(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    source = "def run(payload):\n    while True:\n        pass\n"
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            capacity=1,
            request={
                "request_id": "req-node-stream-pressure-active",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"timeout_ms": 5000, "output_limit_bytes": 1024},
            },
        )
        environment_key = str(opened["environment_key"])
        saw_running = False
        deadline = time.time() + 5.0
        while time.time() < deadline:
            status = svc.workflow_python_request_status(
                profile="node",
                environment_key=environment_key,
                request_id="req-node-stream-pressure-active",
            )
            if dict(status.get("request") or {}).get("status") == "running":
                saw_running = True
                break
            time.sleep(0.05)

        saturated = svc.workflow_python_stream_open(
            profile="node",
            capacity=1,
            environment_key=environment_key,
            request={
                "request_id": "req-node-stream-pressure-blocked",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"timeout_ms": 5000, "output_limit_bytes": 1024},
            },
        )
        canceled = svc.workflow_python_stream_send(
            stream_id=str(opened["stream_id"]),
            message={"action": "cancel", "reason": "capacity_pressure_test_done"},
        )
        deadline = time.time() + 5.0
        terminal_status = {}
        while time.time() < deadline:
            terminal_status = svc.workflow_python_request_status(
                profile="node",
                environment_key=environment_key,
                request_id="req-node-stream-pressure-active",
            )
            if dict(terminal_status.get("request") or {}).get("status") == "canceled":
                break
            time.sleep(0.05)
        resources = svc.workflow_python_resources(profile="node", environment_key=environment_key)
    finally:
        if opened:
            svc.workflow_python_stream_close(stream_id=str(opened.get("stream_id") or ""))

    metrics = resources["workflow_pool"]["metrics"]
    assert opened["status"] == "ok"
    assert saw_running is True
    assert saturated["status"] == "error"
    assert saturated["reason"] == "capacity_exceeded"
    assert canceled["accepted"] is True
    assert canceled["worker_cancel"]["canceled"] is True
    assert terminal_status["request"]["status"] == "canceled"
    assert metrics["active_calls"] == 0
    assert metrics["saturation_count"] >= 1
    assert metrics["cancellation_count"] >= 1


def test_workflow_python_node_resources_report_terminal_metrics(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    env = svc.workflow_python_environment_spec(profile="node", environment_name="workflow-python-node", python={})
    environment_key = str(env["environment_key"])

    def request(source: str, request_id: str, *, timeout_ms: int = 1000) -> dict:
        return {
            "request_id": request_id,
            "module_source": source,
            "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
            "limits": {"timeout_ms": timeout_ms, "output_limit_bytes": 1024},
        }

    success_source = "def run(payload):\n    return {'output': {'ok': True}}\n"
    error_source = "def run(payload):\n    raise ValueError('metric error')\n"
    timeout_source = "def run(payload):\n    while True:\n        pass\n"
    cancel_result: dict[str, object] = {}

    success = svc.execute_workflow_python(profile="node", request=request(success_source, "req-node-metrics-ok"))
    error = svc.execute_workflow_python(profile="node", request=request(error_source, "req-node-metrics-error"))
    timeout = svc.execute_workflow_python(profile="node", request=request(timeout_source, "req-node-metrics-timeout", timeout_ms=50))

    def run_cancel() -> None:
        cancel_result.update(
            svc.execute_workflow_python(
                profile="node",
                request=request(timeout_source, "req-node-metrics-cancel", timeout_ms=5000),
            )
        )

    thread = threading.Thread(target=run_cancel, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        status = svc.workflow_python_request_status(
            profile="node",
            environment_key=environment_key,
            request_id="req-node-metrics-cancel",
        )
        if dict(status.get("request") or {}).get("status") == "running":
            break
        time.sleep(0.05)
    canceled = svc.cancel_workflow_python_request(
        profile="node",
        environment_key=environment_key,
        request_id="req-node-metrics-cancel",
    )
    thread.join(timeout=5.0)
    resources = svc.workflow_python_resources(profile="node", environment_key=environment_key)
    metrics = resources["workflow_pool"]["metrics"]
    recent = {row["request_id"]: row["status"] for row in metrics["recent_requests"]}

    assert success["status"] == "ok"
    assert error["status"] == "error"
    assert timeout["error"]["code"] == "workflow_sandbox_timeout"
    assert canceled["canceled"] is True
    assert cancel_result["status"] == "canceled"
    assert metrics["active_calls"] == 0
    assert metrics["error_count"] >= 1
    assert metrics["timeout_count"] >= 1
    assert metrics["cancellation_count"] >= 1
    assert recent["req-node-metrics-ok"] == "ok"
    assert recent["req-node-metrics-error"] == "error"
    assert recent["req-node-metrics-timeout"] == "timeout"
    assert recent["req-node-metrics-cancel"] == "canceled"


def test_workflow_python_node_resources_include_active_process_metrics(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    monkeypatch.setattr(svc, "_process_resource_snapshot", lambda pid: {"pid": pid, "cpu_percent": 3.5, "memory_mb": 9.25})
    source = "def run(payload):\n    while True:\n        pass\n"
    opened = {}

    try:
        opened = svc.workflow_python_stream_open(
            profile="node",
            request={
                "request_id": "req-node-active-resources",
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg",
                "workflow_id": "wf",
                "package_source_digest": "digest",
                "operation": "run",
                "payload": {},
                "limits": {"timeout_ms": 5000, "output_limit_bytes": 1024},
            },
        )
        deadline = time.time() + 5.0
        resources = {}
        while time.time() < deadline:
            resources = svc.workflow_python_resources(profile="node", environment_key=str(opened["environment_key"]))
            if dict(resources.get("node_runtime") or {}).get("processes"):
                break
            time.sleep(0.05)
    finally:
        if opened:
            svc.workflow_python_stream_send(
                stream_id=str(opened.get("stream_id") or ""),
                message={"action": "cancel", "reason": "test_done"},
            )
            svc.workflow_python_stream_close(stream_id=str(opened.get("stream_id") or ""))

    runtime = dict(resources.get("node_runtime") or {})
    processes = list(runtime.get("processes") or [])
    assert runtime["active_count"] >= 1
    assert runtime["cpu_percent"] == 3.5
    assert runtime["memory_mb"] == 9.2
    assert processes[0]["request_id"] == "req-node-active-resources"
    assert processes[0]["pid"] > 0
    assert processes[0]["resources"]["memory_mb"] == 9.25
