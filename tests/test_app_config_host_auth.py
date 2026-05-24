from __future__ import annotations

import builtins
import json
import importlib.util
import sys
import types
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "app" / "config.py"
_SPEC = importlib.util.spec_from_file_location("app_config_module", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load config module from {_MODULE_PATH}")
app_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(app_config)


def _run_main(argv: list[str]) -> int:
    old = list(sys.argv)
    try:
        sys.argv = argv
        return int(app_config.main())
    finally:
        sys.argv = old


def test_host_auth_status_and_upsert_key(tmp_path: Path, capsys) -> None:
    control_state = tmp_path / "access_control.json"

    rc = _run_main(
        [
            "mp13config",
            "--host-control-state-file",
            str(control_state),
            "--host-auth-upsert-key",
            "mgmt1",
            "--host-auth-role",
            "admin",
            "--host-auth-secret",
            "secret1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["key_id"] == "mgmt1"

    rc = _run_main(
        [
            "mp13config",
            "--host-control-state-file",
            str(control_state),
            "--host-auth-status",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert int(payload["keys_count"]) >= 1


def test_host_auth_issue_session_prints_remote_shared_secret_guidance(tmp_path: Path, capsys) -> None:
    control_state = tmp_path / "access_control.json"

    rc = _run_main(
        [
            "mp13config",
            "--host-control-state-file",
            str(control_state),
            "--host-auth-upsert-key",
            "mgmt1",
            "--host-auth-role",
            "admin",
            "--host-auth-secret",
            "secret1",
        ]
    )
    assert rc == 0
    _ = capsys.readouterr()

    payload = json.loads(control_state.read_text(encoding="utf-8"))
    cfg = dict(payload.get("control_config") or {})
    cfg["require_auth"] = True
    cfg["access_profile"] = {"connectivity_mode": "ssh_tunnel_only"}
    payload["control_config"] = cfg
    control_state.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rc = _run_main(
        [
            "mp13config",
            "--host-control-state-file",
            str(control_state),
            "--host-auth-issue-session",
            "mgmt1",
            "--host-auth-secret",
            "secret1",
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "local_only for shared-secret keys" in out


def test_runtime_status_flags_declared_but_missing_torch_dependency(monkeypatch) -> None:
    def fake_version(package_name: str) -> str:
        assert package_name == "mp13-engine"
        return "0.9.0"

    def fake_requires(package_name: str) -> list[str]:
        assert package_name == "mp13-engine"
        return ["torch ==2.9.1+cu126"]

    def fake_find_spec(name: str):
        assert name == "torch"
        return None

    monkeypatch.setattr(app_config.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(app_config.importlib.metadata, "requires", fake_requires)
    monkeypatch.setattr(app_config.importlib.util, "find_spec", fake_find_spec)

    status = app_config._torch_runtime_status()

    assert status["dependency"]["package_installed"] is True
    assert status["dependency"]["declares_dependency"] is True
    assert status["dependency"]["dependency_spec_found"] is False
    assert status["dependency"]["installed_without_dependency"] is True
    assert status["torch_imported"] is False
    assert status["cuda_available"] is False
    assert status["gpu_access"] is False


def test_runtime_status_reports_cuda_devices_without_real_torch(monkeypatch) -> None:
    class FakeProps:
        total_memory = 24 * 1024 ** 3

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Test GPU"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return (8, 9)

        @staticmethod
        def get_device_properties(index: int) -> FakeProps:
            assert index == 0
            return FakeProps()

    fake_torch = types.SimpleNamespace(
        __version__="2.9.1+cu126",
        version=types.SimpleNamespace(cuda="12.6"),
        cuda=FakeCuda(),
    )

    monkeypatch.setattr(app_config.importlib.metadata, "version", lambda package_name: "0.9.0")
    monkeypatch.setattr(app_config.importlib.metadata, "requires", lambda package_name: ["torch ==2.9.1+cu126"])
    monkeypatch.setattr(app_config.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(app_config.importlib, "import_module", lambda name: fake_torch)

    status = app_config._torch_runtime_status()

    assert status["dependency"]["installed_without_dependency"] is False
    assert status["torch_imported"] is True
    assert status["torch_version"] == "2.9.1+cu126"
    assert status["torch_cuda_version"] == "12.6"
    assert status["cuda_available"] is True
    assert status["cuda_device_count"] == 1
    assert status["gpu_access"] is True
    assert status["cuda_devices"] == [
        {
            "index": 0,
            "name": "Test GPU",
            "capability": "8.9",
            "total_memory_gb": 24.0,
        }
    ]


def test_runtime_status_cli_prints_json(monkeypatch, capsys) -> None:
    payload = {
        "dependency": {"installed_without_dependency": True},
        "torch_imported": False,
        "cuda_available": False,
    }
    monkeypatch.setattr(app_config, "_torch_runtime_status", lambda: payload)

    rc = _run_main(["mp13config", "--runtime-status"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_missing_torch_dependency_blocks_engine_config_routes(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        app_config,
        "_torch_runtime_status",
        lambda: {
            "dependency": {
                "package_installed": True,
                "declares_dependency": True,
                "dependency_spec_found": False,
            },
            "gpu_access": False,
        },
    )

    rc = _run_main(["mp13config", "--config", str(tmp_path / "cfg.json"), "--init"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "installed without torch dependencies" in captured.err
    assert "hosting_config.py" in captured.err


def test_no_gpu_init_writes_cpu_safe_values(monkeypatch, tmp_path: Path, capsys) -> None:
    target = tmp_path / "cfg.json"
    monkeypatch.setattr(
        app_config,
        "_torch_runtime_status",
        lambda: {
            "dependency": {
                "package_installed": True,
                "declares_dependency": True,
                "dependency_spec_found": True,
            },
            "torch_imported": True,
            "gpu_access": False,
        },
    )

    rc = _run_main(["mp13config", "--config", str(target), "--init"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "no GPU was detected" in captured.err
    payload = json.loads(target.read_text(encoding="utf-8"))
    engine_params = payload["engine_params"]
    assert engine_params["device_map"] == "cpu"
    assert engine_params["base_model_dtype"] == "float32"
    assert engine_params["quantize_bits"] == "none"
    assert engine_params["use_torch_compile"] is False
    assert engine_params["static_kv_cache"] is False
    assert engine_params["use_separate_stream"] is False
    assert payload["training_params"]["trainer_precision"] == "fp32"


def test_no_gpu_reconfigure_rejects_gpu_dependent_param(monkeypatch, tmp_path: Path, capsys) -> None:
    target = tmp_path / "cfg.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        app_config,
        "_torch_runtime_status",
        lambda: {
            "dependency": {
                "package_installed": True,
                "declares_dependency": True,
                "dependency_spec_found": True,
            },
            "torch_imported": True,
            "gpu_access": False,
        },
    )

    rc = _run_main(
        [
            "mp13config",
            "--config",
            str(target),
            "--reconfigure",
            "engine_params.device_map=cuda:0",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "Cannot change GPU-dependent parameter 'engine_params.device_map'" in captured.out


def test_no_gpu_interactive_marks_gpu_dependent_fields_unavailable(monkeypatch, tmp_path: Path, capsys) -> None:
    responses = iter(["2", "21", "^"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    config, _, should_save = app_config._interactive_config(
        tmp_path / "source.json",
        tmp_path / "target.json",
        existing={},
        defaults=app_config._build_template_config(),
        cpu_only=True,
    )

    out = capsys.readouterr().out
    assert should_save is False
    assert config == {}
    assert "GPU-dependent fields are unavailable" in out
    assert "engine_params.device_map is GPU-dependent" in out
