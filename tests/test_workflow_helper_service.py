from __future__ import annotations

from pathlib import Path
import hashlib
import shutil

import pytest
from hosting.service.host_service import EngineHostService
from hosting.daemon.local_ipc import EngineHostDaemon


def test_spawn_workflow_js_helper_uses_existing_spawn_model(tmp_path: Path, monkeypatch) -> None:
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

    out = svc.spawn_workflow_js_helper(engine_id="wf-js", node_executable="node-custom", capacity=3)

    assert out["engine_id"] == "wf-js"
    assert out["command"][-1] == "hosting.workflow_js_helper_ipc"
    assert out["env"]["MP13_WORKER_CONTRACT"] == "hosting.workflow_helper.worker.v1"
    assert out["env"]["MP13_WORKFLOW_JS_NODE"] == "node-custom"
    assert out["env"]["MP13_WORKFLOW_JS_HELPER_CAPACITY"] == "3"
    assert out["worker_profile_class"] == "generic"
    assert out["executor_kind"] == "workflow_js_helper"
    assert out["sandbox_policy"]["sandbox"]["profile"] == "workflow_js_helper_v1"
    assert out["sandbox_policy"]["sandbox"]["network"]["mode"] == "disabled"
    assert out["sandbox_policy"]["sandbox"]["brokered_io"] == {
        "filesystem": False,
        "http": False,
        "subprocess": False,
    }
    assert seen["capabilities"]["workflow_js_helper"] is True
    assert seen["capabilities"]["capacity"] == 3


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


def test_daemon_dispatches_spawn_workflow_js_helper() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.kwargs = None

        def spawn_workflow_js_helper(self, **kwargs):
            self.kwargs = dict(kwargs)
            return {"status": "ok", "executor_kind": "workflow_js_helper"}

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    out = daemon._call_service(
        "spawn-workflow-js-helper",
        {
            "engine_id": "wf-js",
            "node_executable": "node-demo",
            "capacity": 4,
            "worker_profile_class": "generic",
            "sandbox_policy": {"sandbox": {"enabled": True}},
        },
    )

    assert out["executor_kind"] == "workflow_js_helper"
    assert fake.kwargs == {
        "engine_id": "wf-js",
        "node_executable": "node-demo",
        "capacity": 4,
        "worker_profile_class": "generic",
        "sandbox_policy": {"sandbox": {"enabled": True}},
    }


def test_daemon_dispatches_spawn_workflow_python_helper() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.kwargs = None

        def spawn_workflow_python_helper(self, **kwargs):
            self.kwargs = dict(kwargs)
            return {"status": "ok", "executor_kind": "workflow_python_helper"}

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    out = daemon._call_service(
        "spawn-workflow-python-helper",
        {
            "engine_id": "wf-py",
            "python_executable": "python-demo",
            "capacity": 4,
            "worker_profile_class": "generic",
            "sandbox_policy": {"sandbox": {"enabled": True}},
        },
    )

    assert out["executor_kind"] == "workflow_python_helper"
    assert fake.kwargs == {
        "engine_id": "wf-py",
        "python_executable": "python-demo",
        "capacity": 4,
        "worker_profile_class": "generic",
        "sandbox_policy": {"sandbox": {"enabled": True}},
    }


def test_daemon_dispatches_workflow_js_helper_resources_and_capacity() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.calls = []

        def workflow_js_helper_resources(self, **kwargs):
            self.calls.append(("resources", dict(kwargs)))
            return {"status": "ok", "capacity": 2}

        def set_workflow_js_helper_capacity(self, **kwargs):
            self.calls.append(("set_capacity", dict(kwargs)))
            return {"status": "ok", "capacity": kwargs["capacity"]}

        def cancel_workflow_js_helper_request(self, **kwargs):
            self.calls.append(("cancel", dict(kwargs)))
            return {"status": "ok", "request_id": kwargs["request_id"], "canceled": True}

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    resources = daemon._call_service("workflow-js-helper-resources", {"engine_id": "wf-js"})
    resized = daemon._call_service("workflow-js-helper-set-capacity", {"engine_id": "wf-js", "capacity": 6})
    canceled = daemon._call_service("workflow-js-helper-cancel-request", {"engine_id": "wf-js", "request_id": "req-1"})

    assert resources["capacity"] == 2
    assert resized["capacity"] == 6
    assert canceled["canceled"] is True
    assert fake.calls == [
        ("resources", {"engine_id": "wf-js"}),
        ("set_capacity", {"engine_id": "wf-js", "capacity": 6}),
        ("cancel", {"engine_id": "wf-js", "request_id": "req-1"}),
    ]


def test_daemon_dispatches_workflow_python_helper_resources_and_capacity() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.calls = []

        def workflow_python_helper_resources(self, **kwargs):
            self.calls.append(("resources", dict(kwargs)))
            return {"status": "ok", "capacity": 2}

        def set_workflow_python_helper_capacity(self, **kwargs):
            self.calls.append(("set_capacity", dict(kwargs)))
            return {"status": "ok", "capacity": kwargs["capacity"]}

        def cancel_workflow_python_helper_request(self, **kwargs):
            self.calls.append(("cancel", dict(kwargs)))
            return {"status": "ok", "request_id": kwargs["request_id"], "canceled": True}

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    resources = daemon._call_service("workflow-python-helper-resources", {"engine_id": "wf-py"})
    resized = daemon._call_service("workflow-python-helper-set-capacity", {"engine_id": "wf-py", "capacity": 6})
    canceled = daemon._call_service("workflow-python-helper-cancel-request", {"engine_id": "wf-py", "request_id": "req-1"})

    assert resources["capacity"] == 2
    assert resized["capacity"] == 6
    assert canceled["canceled"] is True
    assert fake.calls == [
        ("resources", {"engine_id": "wf-py"}),
        ("set_capacity", {"engine_id": "wf-py", "capacity": 6}),
        ("cancel", {"engine_id": "wf-py", "request_id": "req-1"}),
    ]


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

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    assert daemon._call_service("workflow-python-environment-spec", {"profile": "helper"})["environment_key"] == "env-key"
    assert daemon._call_service("workflow-python-ensure", {"engine_id": "wf-py"})["engine_id"] == "wf-py"
    assert daemon._call_service("workflow-python-execute", {"request": {"request_id": "req-1"}})["ok"] is True
    assert daemon._call_service("workflow-python-resources", {"engine_id": "wf-py"})["status"] == "ok"
    assert daemon._call_service("workflow-python-set-capacity", {"engine_id": "wf-py", "capacity": 5})["capacity"] == 5
    assert daemon._call_service("workflow-python-cancel-request", {"engine_id": "wf-py", "request_id": "req-1"})["request_id"] == "req-1"

    assert [name for name, _ in fake.calls] == ["spec", "ensure", "execute", "resources", "set_capacity", "cancel"]


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

        def workflow_js_resources(self, **kwargs):
            self.calls.append(("resources", dict(kwargs)))
            return {"status": "ok"}

        def set_workflow_js_capacity(self, **kwargs):
            self.calls.append(("set_capacity", dict(kwargs)))
            return {"status": "ok", "capacity": kwargs["capacity"]}

        def cancel_workflow_js_request(self, **kwargs):
            self.calls.append(("cancel", dict(kwargs)))
            return {"status": "ok", "request_id": kwargs["request_id"]}

    fake = FakeService()
    daemon = EngineHostDaemon.__new__(EngineHostDaemon)
    daemon.svc = fake

    assert daemon._call_service("workflow-js-environment-spec", {"profile": "helper"})["environment_key"] == "env-js"
    assert daemon._call_service("workflow-js-ensure", {"engine_id": "wf-js"})["engine_id"] == "wf-js"
    assert daemon._call_service("workflow-js-resources", {"engine_id": "wf-js"})["status"] == "ok"
    assert daemon._call_service("workflow-js-set-capacity", {"engine_id": "wf-js", "capacity": 5})["capacity"] == 5
    assert daemon._call_service("workflow-js-cancel-request", {"engine_id": "wf-js", "request_id": "req-1"})["request_id"] == "req-1"

    assert [name for name, _ in fake.calls] == ["spec", "ensure", "resources", "set_capacity", "cancel"]


def test_workflow_js_helper_resources_include_normalized_pool_aliases(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    def fake_proxy_rpc_call(**kwargs):
        return {
            "result": {
                "status": "ok",
                "capacity": 2,
                "active_calls": 1,
                "node_pool": {
                    "capacity": 2,
                    "node_process_count": 1,
                    "active_node_process_count": 1,
                    "idle_node_process_count": 0,
                    "node_processes": [{"pid": 4321, "alive": True, "busy": True, "active_request_id": "req-live"}],
                },
            }
        }

    monkeypatch.setattr(svc, "proxy_rpc_call", fake_proxy_rpc_call)
    monkeypatch.setattr(svc, "_process_resource_snapshot", lambda pid: {"pid": pid, "cpu_percent": 1.0, "memory_mb": 2.0})

    out = svc.workflow_js_helper_resources(engine_id="wf-js")

    assert out["pool"]["process_count"] == 1
    assert out["pool"]["active_process_count"] == 1
    assert out["pool"]["active_request_ids"] == ["req-live"]


def test_workflow_js_facade_spawns_environment_keyed_worker(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    seen = {}

    monkeypatch.setattr(svc, "get_registration", lambda _engine_id: None)

    def fake_spawn(**kwargs):
        seen.update(kwargs)
        return {"status": "ok", "engine_id": kwargs["engine_id"]}

    monkeypatch.setattr(svc, "spawn_workflow_js_helper", fake_spawn)
    monkeypatch.setattr(svc, "workflow_js_helper_resources", lambda **_kwargs: {"status": "ok"})

    out = svc.ensure_workflow_js(
        profile="helper",
        node={"node_executable": "node-demo"},
        capacity=3,
    )

    assert out["status"] == "ok"
    assert out["outcome"] == "spawned"
    assert out["engine_id"].startswith("workflow-js-")
    assert out["environment_key"]
    assert seen["engine_id"] == out["engine_id"]
    assert seen["capacity"] == 3

    resources = svc.workflow_js_resources(
        profile="helper",
        node={"node_executable": "node-demo"},
    )
    assert resources["workflow_pool"]["metrics"]["desired_capacity"] == 3
    assert resources["workflow_pool"]["metrics"]["worker_count"] == 1


def test_workflow_js_facade_isolates_pools_by_policy(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    monkeypatch.setattr(svc, "get_registration", lambda _engine_id: None)
    monkeypatch.setattr(svc, "spawn_workflow_js_helper", lambda **kwargs: {"status": "ok", "engine_id": kwargs["engine_id"]})

    first = svc.ensure_workflow_js(
        profile="helper",
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_js_helper_v1"}},
    )
    second = svc.ensure_workflow_js(
        profile="helper",
        sandbox_policy={"sandbox": {"enabled": False, "profile": "workflow_js_helper_v1"}},
    )

    assert first["environment_key"] != second["environment_key"]
    pools = svc._workflow_python_pool_registry().resources()["pools"]
    assert sorted(pools.keys()) == [
        f"workflow_js/{first['environment_key']}",
        f"workflow_js/{second['environment_key']}",
    ]


def test_old_js_helper_resource_alias_reports_workflow_pool_for_annotated_registration(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    svc.register_spawned(
        engine_id="wf-js-existing",
        pid=0,
        command=["python", "-m", "hosting.workflow_js_helper_ipc"],
        worker_profile_class="generic",
        executor_kind="workflow_js_helper",
        capabilities={"workflow_js_helper": True},
    )
    monkeypatch.setattr(svc, "ensure_running", lambda _engine_id: {"status": "already_running"})
    monkeypatch.setattr(svc, "proxy_rpc_call", lambda **_kwargs: {"result": {"status": "ok", "capacity": 2, "node_pool": {}}})

    ensured = svc.ensure_workflow_js(profile="helper", engine_id="wf-js-existing", capacity=2)
    out = svc.workflow_js_helper_resources(engine_id="wf-js-existing")

    assert out["workflow_runtime_kind"] == "workflow_js"
    assert out["workflow_profile"] == "helper"
    assert out["environment_key"] == ensured["environment_key"]
    assert out["workflow_pool"]["metrics"]["desired_capacity"] == 2


def test_old_js_helper_capacity_and_cancel_alias_update_workflow_pool(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    svc.register_spawned(
        engine_id="wf-js-existing",
        pid=0,
        command=["python", "-m", "hosting.workflow_js_helper_ipc"],
        worker_profile_class="generic",
        executor_kind="workflow_js_helper",
        capabilities={"workflow_js_helper": True},
    )
    monkeypatch.setattr(svc, "ensure_running", lambda _engine_id: {"status": "already_running"})

    def fake_proxy_rpc_call(**kwargs):
        if kwargs["method"] == "workflow_js_helper.set_capacity":
            return {"result": {"status": "ok", "capacity": kwargs["params"]["capacity"], "node_pool": {}}}
        if kwargs["method"] == "workflow_js_helper.cancel_request":
            return {"result": {"status": "ok", "canceled": True, "request_id": kwargs["params"]["request_id"]}}
        return {"result": {"status": "ok", "node_pool": {}}}

    monkeypatch.setattr(svc, "proxy_rpc_call", fake_proxy_rpc_call)

    ensured = svc.ensure_workflow_js(profile="helper", engine_id="wf-js-existing", capacity=2)
    resized = svc.set_workflow_js_helper_capacity(engine_id="wf-js-existing", capacity=4)
    canceled = svc.cancel_workflow_js_helper_request(engine_id="wf-js-existing", request_id="req-missing")

    assert resized["environment_key"] == ensured["environment_key"]
    assert resized["workflow_pool"]["metrics"]["desired_capacity"] == 4
    assert canceled["environment_key"] == ensured["environment_key"]
    assert canceled["workflow_pool_cancel"]["status"] == "not_found"


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


def test_workflow_js_helper_resources_include_child_process_metrics(tmp_path: Path, monkeypatch) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    def fake_proxy_rpc_call(**kwargs):
        assert kwargs["engine_id"] == "wf-js"
        assert kwargs["method"] == "worker.resources"
        return {
            "result": {
                "status": "ok",
                "capacity": 2,
                "active_calls": 1,
                "node_pool": {
                    "capacity": 2,
                    "node_process_count": 1,
                    "active_node_process_count": 1,
                    "idle_node_process_count": 0,
                    "node_processes": [
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

    out = svc.workflow_js_helper_resources(engine_id="wf-js")

    pool = out["node_pool"]
    assert pool["active_request_ids"] == ["req-live"]
    assert pool["node_cpu_percent"] == 12.5
    assert pool["node_memory_mb"] == 64.0
    assert pool["node_processes"][0]["resources"]["pid"] == 4321


def test_workflow_js_helper_spawn_and_rpc_round_trip(tmp_path: Path) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node executable is not available")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    reg = svc.spawn_workflow_js_helper(
        engine_id="wf-js-roundtrip",
        node_executable=node,
        sandbox_policy={
            "sandbox": {
                "enabled": False,
                "profile": "workflow_js_helper_v1",
                "network": {"mode": "disabled"},
                "brokered_io": {"filesystem": False, "http": False, "subprocess": False},
            }
        },
    )
    source = "export function condition(input) { return { accepted: input.value === 7 }; }"
    try:
        svc._wait_for_worker_rpc_ready(reg, timeout_seconds=5.0, poll_interval_seconds=0.1)
        hello = svc.proxy_rpc_call(engine_id="wf-js-roundtrip", method="rpc.describe", params={})
        assert hello["executor_kind"] == "workflow_js_helper"
        out = svc.proxy_rpc_call(
            engine_id="wf-js-roundtrip",
            method="execute_workflow_js_helper",
            params={
                "module_source": source,
                "module_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "package_id": "pkg-demo",
                "workflow_id": "config/demo",
                "package_source_digest": "sha256:digest",
                "export_name": "condition",
                "operation": "condition",
                "payload": {"value": 7},
                "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
            },
        )
        result = out["result"]
        assert result["ok"] is True
        assert result["result"] == {"accepted": True}
        assert result["runtime"]["sandbox_profile"] == "workflow_js_helper_v1"
        persisted = svc.get_registration("wf-js-roundtrip")
        assert persisted is not None
        assert persisted["executor_kind"] == "workflow_js_helper"
        assert persisted["sandbox_policy"]["sandbox"]["profile"] == "workflow_js_helper_v1"
        assert persisted["sandbox_runtime"]
        ensured = svc.ensure_running("wf-js-roundtrip")
        assert ensured["status"] in {"ok", "already_running", "running"}
    finally:
        svc.shutdown(str(reg.get("engine_id") or "wf-js-roundtrip"), timeout_seconds=5.0)


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


def test_execute_workflow_python_node_returns_contract_envelope(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    out = svc.execute_workflow_python(
        profile="node",
        environment_key="env-node",
        engine_id="wf-node",
        request={
            "request_id": "req-node",
            "module_source": "def run(payload):\n    return payload\n",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
        },
    )

    assert out["status"] == "error"
    assert out["ok"] is False
    assert out["profile"] == "node"
    assert out["reason"] == "workflow_python_node_profile_not_implemented"
    assert out["environment_key"] == "env-node"
    assert out["request_id"] == "req-node"
    assert "progress" in out["contract"]["stream_event_types"]


def test_workflow_python_node_stream_returns_pending_worker_events(tmp_path: Path) -> None:
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    opened = svc.workflow_python_stream_open(
        profile="node",
        request={
            "request_id": "req-node-stream",
            "module_source": "def run(payload):\n    return payload\n",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
        },
    )
    received = svc.workflow_python_stream_recv(stream_id=opened["stream_id"], max_items=8)
    status = svc.workflow_python_request_status(
        profile="helper",
        environment_key=opened["environment_key"],
        request_id="req-node-stream",
    )
    closed = svc.workflow_python_stream_close(stream_id=opened["stream_id"])

    assert opened["status"] == "ok"
    assert [row["type"] for row in received["events"]] == ["started", "error", "done"]
    assert received["events"][1]["payload"]["error"]["code"] == "workflow_python_node_profile_not_implemented"
    assert status["request"]["status"] == "error"
    assert closed["closed"] is True
