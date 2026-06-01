from __future__ import annotations

from pathlib import Path

from hosting.sandbox.python_runtime import HostedPythonRuntimeManager


def _policy():
    return {
        "import_allowlist": ["json", "math", "json"],
        "package_pins": {"demo": "1.0.0"},
    }


def test_workflow_python_environment_spec_uses_runtime_envs_and_stable_key(tmp_path: Path) -> None:
    manager = HostedPythonRuntimeManager(tmp_path)

    left = manager.environment_spec(
        environment_name="workflow-python-helper",
        profile="helper",
        python_policy=_policy(),
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_python_helper_v1"}},
    )
    right = manager.environment_spec(
        environment_name="workflow-python-helper",
        profile="helper",
        python_policy={"package_pins": {"demo": "1.0.0"}, "import_allowlist": ["json", "math"]},
        sandbox_policy={"sandbox": {"profile": "workflow_python_helper_v1", "enabled": True}},
    )

    assert left["environment_key"] == right["environment_key"]
    env = left["environment"]
    assert env["environment_root_kind"] == "runtime_envs"
    assert env["environment_consumer_kind"] == "workflow_python_helper"
    assert env["environment_name"] == "workflow-python-helper"
    assert env["required_imports"] == ["json", "math"]
    assert str(env["venv_path"]).startswith(str(tmp_path / "runtime_envs"))


def test_workflow_python_environment_key_changes_with_sandbox_policy(tmp_path: Path) -> None:
    manager = HostedPythonRuntimeManager(tmp_path)

    enabled = manager.environment_spec(
        python_policy=_policy(),
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_python_helper_v1"}},
    )
    disabled = manager.environment_spec(
        python_policy=_policy(),
        sandbox_policy={"sandbox": {"enabled": False, "profile": "workflow_python_helper_v1"}},
    )

    assert enabled["environment_key"] != disabled["environment_key"]


def test_workflow_python_realize_and_prepare_lock_verify(tmp_path: Path) -> None:
    manager = HostedPythonRuntimeManager(tmp_path)

    realized = manager.realize_environment(python_policy=_policy(), package_id="pkg", workflow_id="wf")
    env = realized["environment"]
    assert Path(env["venv_path"], "pyvenv.cfg").exists()
    assert realized["metadata"]["realization"]["planned_packages"] == ["demo==1.0.0"]
    assert realized["metadata"]["realization"]["allow_online_install"] is False

    prepared = manager.prepare_install(python_policy=_policy(), package_id="pkg", workflow_id="wf")
    assert prepared["metadata"]["install_plan"]["planned_packages"] == ["demo==1.0.0"]
    assert prepared["metadata"]["install_plan"]["can_execute_online"] is False

    locked = manager.lock_install(environment=prepared["environment"])
    assert locked["metadata"]["install_lock"]["status"] == "locked"

    verified = manager.verify_install_lock(environment=prepared["environment"])
    assert verified["metadata"]["install_lock_verification"]["status"] == "ok"


def test_workflow_python_selects_bootstrap_until_dependency_env_verified(tmp_path: Path) -> None:
    manager = HostedPythonRuntimeManager(tmp_path)
    prepared = manager.prepare_install(python_policy=_policy())

    selected = manager.select_runtime_python(
        environment=prepared["environment"],
        bootstrap_python_executable="python-bootstrap",
    )

    assert selected["python_executable"] == "python-bootstrap"
    assert selected["python_source"] == "bootstrap"


def test_workflow_python_no_package_env_selects_venv(tmp_path: Path) -> None:
    manager = HostedPythonRuntimeManager(tmp_path)
    realized = manager.realize_environment(python_policy={"import_allowlist": [], "package_pins": {}})

    selected = manager.select_runtime_python(
        environment=realized["environment"],
        bootstrap_python_executable="python-bootstrap",
    )

    assert selected["python_executable"] != "python-bootstrap"
    assert selected["python_source"] == "venv"
