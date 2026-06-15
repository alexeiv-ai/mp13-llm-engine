from __future__ import annotations

from pathlib import Path
import subprocess

from hosting.sandbox.python_runtime import HostedPythonRuntimeBase, HostedPythonRuntimeManager


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


def test_python_runtime_base_sits_above_process_pool_base(tmp_path: Path) -> None:
    base = HostedPythonRuntimeBase(tmp_path)

    key_spec = base.environment_key_spec(
        environment_name="workflow-python-helper",
        profile="helper",
        python_policy=_policy(),
        sandbox_policy={"sandbox": {"enabled": True, "profile": "workflow_python_helper_v1"}},
    )
    capacity = base.set_capacity(key_spec.short_key(), capacity=2)

    assert base.sandbox_kind == "workflow_python"
    assert key_spec.to_dict()["runtime"]["runtime_kind"] == "workflow_python"
    assert capacity["workflow_pool"]["pool_id"] == f"workflow_python/{key_spec.short_key()}"
    assert capacity["workflow_pool"]["metrics"]["desired_capacity"] == 2


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


def test_workflow_python_environment_key_changes_with_artifact_roots(tmp_path: Path) -> None:
    manager = HostedPythonRuntimeManager(tmp_path)

    left = manager.environment_spec(
        profile="node",
        python_policy=_policy(),
        sandbox_policy={"sandbox": {"artifact_roots": {"project": str(tmp_path / "project-a")}}},
    )
    right = manager.environment_spec(
        profile="node",
        python_policy=_policy(),
        sandbox_policy={"sandbox": {"artifact_roots": {"project": str(tmp_path / "project-b")}}},
    )

    assert left["environment_key"] != right["environment_key"]
    assert left["environment_identity"]["sandbox_policy_hash"] != right["environment_identity"]["sandbox_policy_hash"]


def test_workflow_python_environment_key_changes_with_python_runtime_identity(tmp_path: Path) -> None:
    manager = HostedPythonRuntimeManager(tmp_path)

    default_runtime = manager.environment_spec(python_policy=_policy())
    custom_runtime = manager.environment_spec(
        python_policy={**_policy(), "bootstrap_python_executable": "python-custom"}
    )
    same_custom_runtime = manager.environment_spec(
        python_policy={**_policy(), "python_executable": "python-custom"}
    )

    assert default_runtime["environment_key"] != custom_runtime["environment_key"]
    assert custom_runtime["environment_key"] == same_custom_runtime["environment_key"]
    assert "workflow-python-helper-v1:" in custom_runtime["environment_identity"]["runtime"]["runtime_hash"]


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
    assert prepared["install_status"]["install_plan_status"] == "planned"
    assert prepared["install_status"]["install_lock_status"] == "missing"

    locked = manager.lock_install(environment=prepared["environment"])
    assert locked["metadata"]["install_lock"]["status"] == "locked"
    assert locked["install_status"]["install_lock_status"] == "locked"
    assert locked["install_status"]["install_lock_hash"]

    verified = manager.verify_install_lock(environment=prepared["environment"])
    assert verified["metadata"]["install_lock_verification"]["status"] == "ok"
    assert verified["install_status"]["install_lock_verification_status"] == "ok"

    receipt = manager.verify_install_receipt(environment=prepared["environment"])
    assert receipt["install_status"]["install_receipt_verification_status"] == "missing"
    assert receipt["install_status"]["reason"] in {"install_receipt_missing", "install_lock_missing"}


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


def test_workflow_python_runtime_gc_removes_unreferenced_runtime_envs(tmp_path: Path) -> None:
    manager = HostedPythonRuntimeManager(tmp_path)
    keep = manager.realize_environment(
        environment_name="workflow-python-helper",
        python_policy={"package_pins": {"keep": "1.0.0"}},
    )["environment"]
    stale = manager.realize_environment(
        environment_name="workflow-python-helper",
        python_policy={"package_pins": {"stale": "1.0.0"}},
    )["environment"]

    dry = manager.gc_runtime_environments(
        referenced_environment_keys=[str(keep["venv_key"])],
        dry_run=True,
    )
    assert str(stale["venv_key"]) in dry["stale_environment_keys"]
    assert Path(stale["venv_path"]).exists()

    out = manager.gc_runtime_environments(
        referenced_environment_keys=[str(keep["venv_key"])],
    )

    assert out["removed_environment_keys"] == [str(stale["venv_key"])]
    assert Path(keep["venv_path"]).exists()
    assert not Path(stale["venv_path"]).exists()


def test_workflow_python_environment_spec_reports_missing_uv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("hosting.sandbox.python_runtime.shutil.which", lambda _name: None)
    manager = HostedPythonRuntimeManager(tmp_path)

    out = manager.environment_spec(
        profile="node",
        python_policy={
            "uv": {
                "pyproject_toml": "[project]\nname='demo'\n",
                "uv_lock": "lock-data",
                "dependency_groups": ["dev"],
            }
        },
    )

    assert out["environment"]["uv"]["enabled"] is True
    assert out["environment"]["uv"]["available"] is False
    assert out["environment_identity"]["dependency_lock_hash"]


def test_workflow_python_environment_spec_reports_uv_version(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(["uv", "--version"], 0, stdout="uv 0.7.1\n", stderr="")

    monkeypatch.setattr("hosting.sandbox.python_runtime.shutil.which", lambda _name: "C:/Tools/uv.exe")
    monkeypatch.setattr("hosting.sandbox.python_runtime.subprocess.run", fake_run)
    manager = HostedPythonRuntimeManager(tmp_path)

    out = manager.environment_spec(profile="node", python_policy={"uv": {"enabled": True}})

    assert out["environment"]["uv"]["available"] is True
    assert out["environment"]["uv"]["resolved_executable"] == "C:/Tools/uv.exe"
    assert out["environment"]["uv"]["version"] == "uv 0.7.1"


def test_workflow_python_prepare_install_adds_deterministic_uv_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("hosting.sandbox.python_runtime.shutil.which", lambda _name: "C:/Tools/uv.exe")
    monkeypatch.setattr(
        "hosting.sandbox.python_runtime.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["uv", "--version"], 0, stdout="uv 0.7.1\n", stderr=""),
    )
    manager = HostedPythonRuntimeManager(tmp_path)
    policy = {
        "uv": {
            "pyproject_toml": "[project]\nname='demo'\n",
            "uv_lock": "lock-data",
            "dependency_groups": ["dev", "test"],
        }
    }

    left = manager.prepare_install(python_policy=policy)
    right = manager.prepare_install(python_policy=policy)

    assert left["metadata"]["uv_install_plan"]["status"] == "planned"
    assert left["metadata"]["uv_install_plan"]["allow_execution"] is False
    assert left["install_status"]["uv_install_plan_status"] == "planned"
    assert left["install_status"]["uv_plan_hash"] == right["install_status"]["uv_plan_hash"]
