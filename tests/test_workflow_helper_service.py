from __future__ import annotations

from pathlib import Path
import hashlib
import shutil

import pytest
from hosting.service.host_service import EngineHostService


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

    out = svc.spawn_workflow_js_helper(engine_id="wf-js", node_executable="node-custom")

    assert out["engine_id"] == "wf-js"
    assert out["command"][-1] == "hosting.workflow_js_helper_ipc"
    assert out["env"]["MP13_WORKER_CONTRACT"] == "hosting.workflow_helper.worker.v1"
    assert out["env"]["MP13_WORKFLOW_JS_NODE"] == "node-custom"
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
