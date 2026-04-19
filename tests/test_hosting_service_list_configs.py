from __future__ import annotations

import tempfile
from pathlib import Path

from hosting.engine_host_service import EngineHostService


def test_list_configs_uses_lightweight_module_discovery(monkeypatch) -> None:
    base = Path(tempfile.gettempdir())
    default_cfg = Path(__file__).resolve()
    cfg_store = Path(__file__).resolve().parent

    svc = EngineHostService(
        engines_state_file=base / "managed_engines.json",
        control_state_file=base / "access_control.json",
    )

    monkeypatch.setattr(EngineHostService, "_default_config_path", lambda self: default_cfg)
    monkeypatch.setattr(EngineHostService, "_config_store_dir", lambda self: cfg_store)
    monkeypatch.setattr(EngineHostService, "_merge_default_and_selected_config", lambda self, _selector: {})
    monkeypatch.setattr(EngineHostService, "_engine_python_executable", lambda self: "python")
    monkeypatch.setattr(
        EngineHostService,
        "_check_module_discoverable",
        staticmethod(lambda _python, _module: (True, "")),
    )

    def _fail_if_called(_python: str, _module: str):
        raise AssertionError("heavy import checker must not be used by list-configs")

    monkeypatch.setattr(EngineHostService, "_check_module_available", staticmethod(_fail_if_called))

    rows = svc.list_engine_configs()
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert bool(rows[0].get("has_spawn_command")) is True


def test_connect_from_config_emits_progress_events(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "default.json"
    cfg_path.write_text("{}", encoding="utf-8")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    monkeypatch.setattr(EngineHostService, "_resolve_json_config_path", lambda self, _selector: cfg_path)
    monkeypatch.setattr(EngineHostService, "_merge_default_and_selected_config", lambda self, _selector: {})
    monkeypatch.setattr(
        EngineHostService,
        "_build_engine_spawn_spec",
        lambda self, **_kwargs: {
            "command": ["python", "-m", "hosting.engine_worker_ipc"],
            "cwd": None,
            "env": {},
            "worker_auth_token": "tok",
            "worker_auth_header": "X-MP13-Host-Token",
            "worker_ipc_family": "AF_INET",
            "worker_ipc_address": "127.0.0.1:12345",
        },
    )
    monkeypatch.setattr(
        EngineHostService,
        "spawn",
        lambda self, **kwargs: {"engine_id": str(kwargs.get("engine_id") or "worker1"), "pid": 1234},
    )

    out = svc.connect_from_config(config_path="default", model_path="C:/models/foo")
    assert str(out.get("status") or "") == "ok"
    assert str(out.get("stage") or "") == "completed"
    events = list(out.get("progress_events") or [])
    assert len(events) >= 3
    assert any(str(x.get("stage") or "").startswith("connect.") for x in events)


def test_connect_from_config_generic_profile_spawns_without_model(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "generic.json"
    cfg_path.write_text("{}", encoding="utf-8")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    monkeypatch.setattr(EngineHostService, "_resolve_json_config_path", lambda self, _selector: cfg_path)
    monkeypatch.setattr(
        EngineHostService,
        "_merge_default_and_selected_config",
        lambda self, _selector: {
            "worker_kind": "generic",
            "worker_command": ["python", "-c", "print('hello')"],
        },
    )
    monkeypatch.setattr(
        EngineHostService,
        "spawn",
        lambda self, **kwargs: {"engine_id": str(kwargs.get("engine_id") or "worker1"), "pid": 4321},
    )

    out = svc.connect_from_config(config_path="generic")
    assert str(out.get("status") or "") == "ok"
    assert str(out.get("worker_class") or "") == "generic"
    assert out.get("model_path") is None
    events = list(out.get("progress_events") or [])
    assert any(str(x.get("status") or "") == "skipped" for x in events)


def test_connect_from_config_generic_profile_requires_worker_command(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "generic_bad.json"
    cfg_path.write_text("{}", encoding="utf-8")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )

    monkeypatch.setattr(EngineHostService, "_resolve_json_config_path", lambda self, _selector: cfg_path)
    monkeypatch.setattr(
        EngineHostService,
        "_merge_default_and_selected_config",
        lambda self, _selector: {"worker_kind": "generic"},
    )

    out = svc.connect_from_config(config_path="generic_bad")
    assert str(out.get("status") or "") == "failed"
    assert str(out.get("reason") or "") == "generic_worker_command_missing"
    assert str(out.get("worker_class") or "") == "generic"
