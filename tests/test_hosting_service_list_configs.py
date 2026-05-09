from __future__ import annotations

import tempfile
from pathlib import Path

from hosting.service.host_service import EngineHostService


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


def test_list_configs_validates_config_selectors_not_paths(tmp_path: Path, monkeypatch) -> None:
    default_cfg = tmp_path / "default.json"
    default_cfg.write_text("{}", encoding="utf-8")
    cfg_store = tmp_path / "configs"
    cfg_store.mkdir()
    named_cfg = cfg_store / "granite-2b.json"
    named_cfg.write_text("{}", encoding="utf-8")

    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    seen_selectors = []

    def _merge(self, selector):
        selector = str(selector)
        seen_selectors.append(selector)
        assert not Path(selector).is_absolute()
        assert "\\" not in selector
        assert "/" not in selector
        return {}

    monkeypatch.setattr(EngineHostService, "_default_config_path", lambda self: default_cfg)
    monkeypatch.setattr(EngineHostService, "_config_store_dir", lambda self: cfg_store)
    monkeypatch.setattr(EngineHostService, "_merge_default_and_selected_config", _merge)
    monkeypatch.setattr(EngineHostService, "_engine_python_executable", lambda self: "python")
    monkeypatch.setattr(
        EngineHostService,
        "_check_module_discoverable",
        staticmethod(lambda _python, _module: (True, "")),
    )

    rows = svc.list_engine_configs()

    assert seen_selectors == ["default", "granite-2b"]
    assert [row["name"] for row in rows] == ["default", "granite-2b"]
    assert all(row.get("connect_reason") is None for row in rows)


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


def test_connect_from_config_waits_for_model_worker_rpc_ready(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "default.json"
    cfg_path.write_text("{}", encoding="utf-8")
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    ready_calls = []

    monkeypatch.setattr(EngineHostService, "_resolve_json_config_path", lambda self, _selector: cfg_path)
    monkeypatch.setattr(
        EngineHostService,
        "_merge_default_and_selected_config",
        lambda self, _selector: {"worker_ready_timeout_seconds": 12},
    )
    monkeypatch.setattr(
        EngineHostService,
        "_build_engine_spawn_spec",
        lambda self, **_kwargs: {
            "command": ["python", "-m", "hosting.engine_worker_ipc"],
            "cwd": None,
            "env": {},
            "worker_auth_token": "tok",
            "worker_auth_header": "X-MP13-Host-Token",
            "worker_ipc_family": "AF_PIPE",
            "worker_ipc_address": r"\\.\pipe\mp13-test-worker",
        },
    )
    monkeypatch.setattr(
        EngineHostService,
        "spawn",
        lambda self, **kwargs: {
            "engine_id": str(kwargs.get("engine_id") or "worker1"),
            "pid": 1234,
            "worker_transport": "ipc",
            "worker_ipc_family": "AF_PIPE",
            "worker_ipc_address": r"\\.\pipe\mp13-test-worker",
            "worker_auth_token": "tok",
        },
    )

    def _fake_wait(self, reg, *, timeout_seconds=600.0, poll_interval_seconds=0.5):
        ready_calls.append((dict(reg), timeout_seconds, poll_interval_seconds))
        return {"status": "ok", "attempts": 3, "ready_at": 123.0, "worker": {"status": "ok"}}

    monkeypatch.setattr(EngineHostService, "_wait_for_worker_rpc_ready", _fake_wait)

    out = svc.connect_from_config(config_path="default", model_path="C:/models/foo")

    assert str(out.get("status") or "") == "ok"
    assert str(out.get("stage") or "") == "completed"
    assert ready_calls
    assert ready_calls[0][1] == 12
    assert dict(out.get("worker_ready") or {})["attempts"] == 3
    events = list(out.get("progress_events") or [])
    assert any(
        str(x.get("stage") or "") == "connect.worker_ready" and str(x.get("status") or "") == "completed"
        for x in events
    )


def test_connect_from_config_reuses_reachable_worker_for_same_model(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "default.json"
    cfg_path.write_text("{}", encoding="utf-8")
    alt_cfg = tmp_path / "alt.json"
    alt_cfg.write_text("{}", encoding="utf-8")
    model_path = tmp_path / "models" / "demo"
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    spawned = []

    monkeypatch.setattr(EngineHostService, "_resolve_json_config_path", lambda self, selector: alt_cfg if selector == "alt" else cfg_path)
    monkeypatch.setattr(EngineHostService, "_merge_default_and_selected_config", lambda self, _selector: {})
    monkeypatch.setattr(EngineHostService, "_engine_python_executable", lambda self: "python")
    monkeypatch.setattr(EngineHostService, "_pid_alive", staticmethod(lambda _pid: True))
    monkeypatch.setattr(
        EngineHostService,
        "_probe_registration_reachability",
        lambda self, _reg, **_kwargs: {"reachable": True, "status": "ok"},
    )
    def _fake_spawn(self, **kwargs):
        spawned.append(kwargs)
        return self.register_spawned(
            engine_id=str(kwargs.get("engine_id") or "model-demo"),
            pid=1234,
            command=list(kwargs.get("command") or ["python"]),
            cwd=kwargs.get("cwd"),
            env=dict(kwargs.get("env") or {}),
            worker_auth_token="tok",
            worker_auth_header="X-MP13-Host-Token",
            worker_ipc_family="AF_PIPE",
            worker_ipc_address=r"\\.\pipe\mp13-test-worker",
            worker_profile_class="model",
        )

    monkeypatch.setattr(EngineHostService, "spawn", _fake_spawn)
    monkeypatch.setattr(
        EngineHostService,
        "_build_engine_spawn_spec",
        lambda self, **_kwargs: {
            "command": ["python", "-m", "hosting.engine_worker_ipc"],
            "cwd": None,
            "env": {},
            "worker_auth_token": "tok",
            "worker_auth_header": "X-MP13-Host-Token",
            "worker_ipc_family": "AF_PIPE",
            "worker_ipc_address": r"\\.\pipe\mp13-test-worker",
        },
    )
    monkeypatch.setattr(
        EngineHostService,
        "_wait_for_worker_rpc_ready",
        lambda self, _reg, **_kwargs: {"status": "ok", "attempts": 1, "worker": {"status": "ok"}},
    )

    first = svc.connect_from_config(config_path="default", model_path=str(model_path))
    assert first["status"] == "ok"
    assert first["spawned"] is True
    assert len(spawned) == 1

    second = svc.connect_from_config(config_path="alt", model_path=str(model_path))
    assert second["status"] == "attached"
    assert second["reconciled"] is True
    assert second["spawned"] is False
    assert second["worker_id"] == first["worker_id"]
    assert second["model_instance_id"] == first["model_instance_id"]
    assert len(spawned) == 1

    reg = svc.get_registration(str(second["engine_id"]))
    assert reg is not None
    assert str(reg.get("_route_model_instance_id") or "") == first["model_instance_id"]

    third = svc.connect_from_config(config_path="alt", model_path=str(model_path))
    assert third["status"] == "reused"
    assert third["engine_id"] == second["engine_id"]
    assert len(spawned) == 1


def test_connect_from_config_force_new_worker_bypasses_reuse(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "default.json"
    cfg_path.write_text("{}", encoding="utf-8")
    model_path = tmp_path / "models" / "demo"
    svc = EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    spawned = []

    monkeypatch.setattr(EngineHostService, "_resolve_json_config_path", lambda self, _selector: cfg_path)
    monkeypatch.setattr(EngineHostService, "_merge_default_and_selected_config", lambda self, _selector: {})
    monkeypatch.setattr(EngineHostService, "_engine_python_executable", lambda self: "python")
    monkeypatch.setattr(EngineHostService, "_pid_alive", staticmethod(lambda _pid: True))
    monkeypatch.setattr(
        EngineHostService,
        "_probe_registration_reachability",
        lambda self, _reg, **_kwargs: {"reachable": True, "status": "ok"},
    )
    def _fake_spawn(self, **kwargs):
        spawned.append(kwargs)
        return self.register_spawned(
            engine_id=str(kwargs.get("engine_id") or f"model-demo-{len(spawned)}"),
            pid=1234 + len(spawned),
            command=list(kwargs.get("command") or ["python"]),
            cwd=kwargs.get("cwd"),
            env=dict(kwargs.get("env") or {}),
            worker_auth_token="tok",
            worker_auth_header="X-MP13-Host-Token",
            worker_ipc_family="AF_PIPE",
            worker_ipc_address=rf"\\.\pipe\mp13-test-worker-{len(spawned)}",
            worker_profile_class="model",
        )

    monkeypatch.setattr(EngineHostService, "spawn", _fake_spawn)
    monkeypatch.setattr(
        EngineHostService,
        "_build_engine_spawn_spec",
        lambda self, **_kwargs: {
            "command": ["python", "-m", "hosting.engine_worker_ipc"],
            "cwd": None,
            "env": {},
            "worker_auth_token": "tok",
            "worker_auth_header": "X-MP13-Host-Token",
            "worker_ipc_family": "AF_PIPE",
            "worker_ipc_address": r"\\.\pipe\mp13-test-worker",
        },
    )
    monkeypatch.setattr(
        EngineHostService,
        "_wait_for_worker_rpc_ready",
        lambda self, _reg, **_kwargs: {"status": "ok", "attempts": 1, "worker": {"status": "ok"}},
    )

    first = svc.connect_from_config(config_path="default", model_path=str(model_path))
    second = svc.connect_from_config(config_path="default", model_path=str(model_path), force_new_worker=True)

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert second["spawned"] is True
    assert second["worker_id"] != first["worker_id"]
    assert len(spawned) == 2


def test_connect_from_config_fails_when_model_worker_rpc_not_ready(tmp_path: Path, monkeypatch) -> None:
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
            "worker_ipc_family": "AF_PIPE",
            "worker_ipc_address": r"\\.\pipe\mp13-test-worker",
        },
    )
    monkeypatch.setattr(
        EngineHostService,
        "spawn",
        lambda self, **kwargs: {
            "engine_id": str(kwargs.get("engine_id") or "worker1"),
            "pid": 1234,
            "worker_transport": "ipc",
            "worker_ipc_family": "AF_PIPE",
            "worker_ipc_address": r"\\.\pipe\mp13-test-worker",
            "worker_auth_token": "tok",
        },
    )
    monkeypatch.setattr(
        EngineHostService,
        "_wait_for_worker_rpc_ready",
        lambda self, reg, **_kwargs: (_ for _ in ()).throw(TimeoutError("worker RPC did not become ready")),
    )

    out = svc.connect_from_config(config_path="default", model_path="C:/models/foo")

    assert str(out.get("status") or "") == "failed"
    assert str(out.get("reason") or "") == "worker_not_ready"
    assert "worker RPC did not become ready" in str(out.get("message") or "")
    events = list(out.get("progress_events") or [])
    assert any(
        str(x.get("stage") or "") == "connect.worker_ready" and str(x.get("status") or "") == "failed"
        for x in events
    )


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
