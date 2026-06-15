from __future__ import annotations

from pathlib import Path

from hosting.sandbox.js_runtime import HostedJsRuntimeBase


def test_js_runtime_base_derives_environment_identity(tmp_path: Path) -> None:
    base = HostedJsRuntimeBase(tmp_path)

    left = base.environment_spec(
        environment_name="workflow-js-node",
        profile="node",
        javascript_policy={
            "runtime_hash": "quickjs-a",
            "allowed_host_modules": ["fs", "fs"],
            "package_pins": {"demo": "1.0.0"},
        },
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_js_node_v1"}},
    )
    right = base.environment_spec(
        environment_name="workflow-js-node",
        profile="node",
        javascript_policy={
            "runtime_hash": "quickjs-b",
            "allowed_host_modules": ["fs"],
            "package_pins": {"demo": "1.0.0"},
        },
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_js_node_v1"}},
    )

    assert left["status"] == "ok"
    assert left["environment_key"] != right["environment_key"]
    assert left["environment"]["workflow_runtime_kind"] == "workflow_js"
    assert left["environment"]["required_imports"] == ["fs"]
    assert left["environment"]["package_pins"] == {"demo": "1.0.0"}
    assert left["environment"]["install_status"] == "not_applicable"


def test_js_runtime_base_sits_above_process_pool_base(tmp_path: Path) -> None:
    base = HostedJsRuntimeBase(tmp_path)
    env = base.environment_spec(javascript_policy={"runtime_hash": "quickjs"})["environment_key"]
    capacity = base.set_capacity(env, capacity=3)

    assert base.sandbox_kind == "workflow_js"
    assert capacity["workflow_pool"]["pool_id"] == f"workflow_js/{env}"
    assert capacity["workflow_pool"]["metrics"]["desired_capacity"] == 3
