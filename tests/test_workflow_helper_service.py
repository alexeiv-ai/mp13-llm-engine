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
        svc.proxy_rpc_call(
            engine_id="wf-py-runtime",
            method="execute_workflow_python_helper",
            params={
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
