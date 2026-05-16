from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import pytest

from hosting.service.host_service import EngineHostService


def _make_service(tmp_path: Path) -> EngineHostService:
    return EngineHostService(
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )


def test_resolve_model_path_from_config_value_uses_models_root(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    cfg_path = tmp_path / "backend" / "configs" / "granite-2b.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("{}", encoding="utf-8")
    models_root = tmp_path / "models"
    model_dir = models_root / "granite-3.3-2b-instruct"
    model_dir.mkdir(parents=True)
    svc._resolve_json_config_path = lambda _config_path: cfg_path  # type: ignore[method-assign]

    resolved = svc._resolve_model_path_from_config_value(
        "granite-3.3-2b-instruct",
        config_path="granite-2b",
        cfg={"category_dirs": {"models_root_dir": str(models_root)}},
    )

    assert resolved == str(model_dir.resolve())


def test_resolve_model_path_from_config_value_does_not_use_process_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(tmp_path)
    cfg_path = tmp_path / "backend" / "configs" / "granite-2b.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("{}", encoding="utf-8")
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    svc._resolve_json_config_path = lambda _config_path: cfg_path  # type: ignore[method-assign]

    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.setattr(EngineHostService, "_service_project_root", staticmethod(lambda: None))

    resolved = svc._resolve_model_path_from_config_value(
        "./local-model",
        config_path="granite-2b",
        cfg={},
    )

    assert resolved == str((cfg_path.parent / "local-model").resolve())


def _install_ipc_http_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(
        self,
        *,
        reg,
        engine_id: str,
        method: str,
        path: str,
        query: str,
        headers: dict[str, str],
        body_b64: str,
        timeout_seconds: float,
    ) -> dict[str, object]:
        body = b'{"ok":true}' if str(path).startswith("/health") else b"not found"
        status = 200 if str(path).startswith("/health") else 404
        return {
            "engine_id": str(engine_id),
            "endpoint": "ipc://local",
            "url": f"ipc://{engine_id}{path}",
            "status_code": status,
            "headers": {"content-type": "application/json" if status == 200 else "text/plain"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "body_size": len(body),
            "truncated": False,
        }

    monkeypatch.setattr(EngineHostService, "_proxy_request_via_ipc", _stub)


def test_auth_bootstrap_and_session_enforcement(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)

    # Bootstrap admin key and require auth.
    upsert = svc.auth_upsert_key(key_id="mgmt1", key_secret="secret1", role="admin")
    assert upsert["key_id"] == "mgmt1"
    cfg = svc.set_control_config(require_auth=True)
    assert cfg["require_auth"] is True

    with pytest.raises(PermissionError):
        svc.authorize_command("discover-running", {})

    issued = svc.auth_issue_session(key_id="mgmt1", key_secret="secret1", scope="control", ttl_seconds=300)
    token = str(issued["token"])
    svc.authorize_command("discover-running", {"session_token": token})


def test_reset_hosting_access_clears_only_auth_state(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(key_id="mgmt1", key_secret="secret1", role="admin")
    svc.set_control_config(
        require_auth=True,
        access_profile={"connectivity_mode": "local_only"},
        endpoint_mode_default="shared",
    )
    issued = svc.auth_issue_session(key_id="mgmt1", key_secret="secret1", scope="control", ttl_seconds=300)
    assert str(issued.get("token") or "")

    out = svc.reset_hosting_access()

    assert out["status"] == "ok"
    assert out["cleared_keys"] == 1
    assert out["cleared_sessions"] == 1
    cfg = svc.get_control_config()
    assert cfg["require_auth"] is True
    assert str(cfg["endpoint_mode_default"] or "") == "shared"
    assert cfg["keys_count"] == 0
    assert cfg["sessions_count"] == 0


def test_discover_running_adds_operator_state_and_kind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="model1",
        pid=123,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    svc.register_spawned(
        engine_id="tools1",
        pid=456,
        command=["python", "-m", "hosting.toolbox_executor_ipc"],
        env={"MP13_TOOLBOX_EXECUTOR_ENGINE_ID": "tools1"},
        sandbox_policy={"sandbox": {"enabled": True, "profile": "generic_worker_v1"}},
        executor_kind="toolbox_executor",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {"reachable": True},
    )

    rows = {row["engine_id"]: row for row in svc.discover_running()}

    assert rows["model1"]["state"] == "running"
    assert rows["model1"]["kind"] == "model instance"
    assert rows["tools1"]["state"] == "running"
    assert rows["tools1"]["kind"] == "tools sandbox"
    assert rows["tools1"]["sandbox"]["enabled"] is True


def test_discover_running_adds_worker_reported_gpu_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="model1",
        pid=123,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {"reachable": True},
    )
    monkeypatch.setattr(
        svc,
        "_process_resource_snapshot",
        lambda _pid: {"pid": 123, "cpu_percent": 1.0, "memory_mb": 2.0, "gpu_vram_mb": None},
    )

    def _fake_ipc_call(*, reg, payload, timeout_seconds):
        assert payload["method"] == "worker.resources"
        return {
            "status": "ok",
            "result": {
                "data": {
                    "gpu_info": [
                        {
                            "device_id": 0,
                            "memory_allocated_gb": 1.25,
                            "memory_reserved_gb": 2.5,
                            "memory_total_gb": 24.0,
                        }
                    ]
                }
            },
        }

    monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

    row = svc.discover_running()[0]

    resources = dict(row["process_resources"])
    assert resources["gpu_vram_mb"] == 2560.0
    assert resources["gpu_allocated_mb"] == 1280.0
    assert resources["gpu_devices"] == ["cuda:0"]
    assert resources["gpu_vram_source"] == "worker_torch_cuda"


def test_discover_running_uses_worker_ipc_reported_pid_for_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="model1",
        pid=111,
        command=["C:/venv/Scripts/python.exe", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {
            "reachable": True,
            "transport": "ipc",
            "probe": "hello",
            "worker_pid": 222,
            "worker_executable": "C:/Python/Python312/python.exe",
            "worker_prefix": "C:/venv",
        },
    )
    monkeypatch.setattr(
        svc,
        "_process_resource_snapshot",
        lambda pid: {"pid": pid, "cpu_percent": 1.0, "memory_mb": float(pid), "gpu_vram_mb": None},
    )
    monkeypatch.setattr(svc, "_query_worker_reported_resources", lambda _item: {})

    row = svc.discover_running()[0]

    assert row["pid"] == 222
    assert row["launcher_pid"] == 111
    assert row["pid_identity"]["reason"] == "worker_ipc_reported_pid"
    assert row["process_resources"]["pid"] == 222
    assert row["process_resources"]["memory_mb"] == 222.0


def test_resource_summary_keeps_unknown_gpu_vram_as_none() -> None:
    summary = EngineHostService._resource_summary_from_rows(
        [
            {"pid": 123, "cpu_percent": 1.0, "memory_mb": 2.0, "gpu_vram_mb": None},
            {"pid": 456, "cpu_percent": None, "memory_mb": None},
        ]
    )

    assert summary["worker_gpu_vram_mb"] is None
    assert summary["worker_gpu_allocated_mb"] is None


def test_resource_summary_reports_pending_gpu_vram() -> None:
    summary = EngineHostService._resource_summary_from_rows(
        [
            {"pid": 123, "cpu_percent": 1.0, "memory_mb": 2.0, "gpu_vram_mb": None, "gpu_vram_pending": True},
        ]
    )

    assert summary["worker_gpu_vram_mb"] is None
    assert summary["worker_gpu_vram_pending"] is True


def test_host_metrics_reports_daemon_and_engine_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    monkeypatch.setenv("MP13_ENGINE_PYTHON", "C:/engine/python.exe")
    monkeypatch.setattr(svc, "_registered_worker_resource_rows", lambda: [])

    metrics = svc.get_host_metrics()

    assert metrics["daemon_python_executable"]
    assert metrics["engine_python_executable"] == "C:/engine/python.exe"
    assert metrics["mp13_engine_python_env"] == "C:/engine/python.exe"


def test_discover_running_uses_worker_state_gpu_memory_when_detailed_gpu_info_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="model1",
        pid=123,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {"reachable": True},
    )
    monkeypatch.setattr(
        svc,
        "_process_resource_snapshot",
        lambda _pid: {"pid": 123, "cpu_percent": 1.0, "memory_mb": 2.0, "gpu_vram_mb": None},
    )

    def _fake_ipc_call(*, reg, payload, timeout_seconds):
        assert payload["method"] == "worker.resources"
        return {
            "status": "ok",
            "result": {
                "data": {
                    "gpu_info": "CUDA details unavailable",
                    "current_gpu_mem_allocated_gb": 4.75,
                    "current_gpu_mem_reserved_gb": 5.0,
                }
            },
        }

    monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

    resources = dict(svc.discover_running()[0]["process_resources"])

    assert resources["gpu_vram_mb"] == 5120.0
    assert resources["gpu_allocated_mb"] == 4864.0
    assert resources["gpu_vram_source"] == "worker_state_gpu_memory"


def test_discover_running_accepts_worker_state_gpu_memory_mb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="model1",
        pid=123,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {"reachable": True},
    )
    monkeypatch.setattr(
        svc,
        "_process_resource_snapshot",
        lambda _pid: {"pid": 123, "cpu_percent": 1.0, "memory_mb": 2.0, "gpu_vram_mb": None},
    )

    def _fake_ipc_call(*, reg, payload, timeout_seconds):
        assert payload["method"] == "worker.resources"
        return {
            "status": "ok",
            "result": {
                "data": {
                    "gpu_info": "CUDA details unavailable",
                    "current_gpu_mem_allocated_mb": 4864.0,
                    "current_gpu_mem_reserved_mb": 5120.0,
                }
            },
        }

    monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

    resources = dict(svc.discover_running()[0]["process_resources"])

    assert resources["gpu_vram_mb"] == 5120.0
    assert resources["gpu_allocated_mb"] == 4864.0
    assert resources["gpu_vram_source"] == "worker_state_gpu_memory"


def test_discover_running_marks_gpu_resource_probe_pending_until_next_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="model1",
        pid=123,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {"reachable": True},
    )
    monkeypatch.setattr(
        svc,
        "_process_resource_snapshot",
        lambda _pid: {"pid": 123, "cpu_percent": 1.0, "memory_mb": 2.0, "gpu_vram_mb": None},
    )
    calls: list[float] = []

    def _fake_ipc_call(*, reg, payload, timeout_seconds):
        assert payload["method"] == "worker.resources"
        calls.append(float(timeout_seconds))
        if len(calls) == 1:
            raise TimeoutError("status probe timed out")
        return {
            "status": "ok",
            "result": {
                "data": {
                    "gpu_info": [
                        {
                            "device_id": 0,
                            "memory_allocated_gb": 1.0,
                            "memory_reserved_gb": 2.0,
                        }
                    ]
                }
            },
        }

    monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

    first = svc.discover_running()[0]
    first_resources = dict(first["process_resources"])

    assert calls == [1.0]
    assert first_resources["gpu_vram_mb"] is None
    assert first_resources["gpu_vram_pending"] is True
    assert first["worker_resource_probe"]["status"] == "pending"
    assert first["worker_resource_probe"]["method"] == "worker.resources"

    second = svc.discover_running()[0]
    resources = dict(second["process_resources"])

    assert calls == [1.0, 1.0]
    assert resources["gpu_vram_mb"] == 2048.0
    assert second["worker_resource_probe"]["status"] == "ok"


def test_discover_running_marks_worker_resources_pending_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="model1",
        pid=123,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {"reachable": True},
    )
    monkeypatch.setattr(
        svc,
        "_process_resource_snapshot",
        lambda _pid: {"pid": 123, "cpu_percent": 1.0, "memory_mb": 2.0, "gpu_vram_mb": None},
    )

    def _fake_ipc_call(*, reg, payload, timeout_seconds):
        assert payload["method"] == "worker.resources"
        return {
            "status": "ok",
            "result": {
                "status": "pending",
                "message": "torch_module_not_loaded",
                "data": {"gpu_vram_pending": True},
            },
        }

    monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

    row = svc.discover_running()[0]
    resources = dict(row["process_resources"])

    assert resources["gpu_vram_pending"] is True
    assert row["worker_resource_probe"] == {
        "status": "pending",
        "method": "worker.resources",
        "message": "torch_module_not_loaded",
    }


def test_discover_running_prunes_registration_when_pid_was_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    expected = str(tmp_path / "venv" / "Scripts" / "python.exe")
    actual = str(tmp_path / "Windows" / "System32" / "svchost.exe")
    svc.register_spawned(
        engine_id="stale-tools",
        pid=1700,
        command=[expected, "-m", "hosting.toolbox_executor_ipc"],
        env={"MP13_TOOLBOX_EXECUTOR_ENGINE_ID": "stale-tools"},
        executor_kind="toolbox_executor",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(svc, "_process_image_path", lambda _pid: actual)

    assert svc.discover_running() == []
    assert svc._read_engines() == []


def test_discover_running_keeps_starting_model_with_missing_ipc_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="loading-model",
        pid=1700,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    rows = svc._read_engines()
    rows[0]["spawned_at"] = time.time() - 120.0
    svc._write_engines(rows)
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {
            "reachable": False,
            "transport": "ipc",
            "probe": "hello",
            "error": "worker IPC endpoint is unavailable for engine 'loading-model' at 'pipe'; worker process may not be running",
        },
    )

    row = svc.discover_running()[0]

    assert row["state"] == "spawning"
    assert row["alive"] is True
    assert svc.get_registration("loading-model") is not None


def test_discover_running_retries_transient_missing_ipc_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="ready-model",
        pid=1700,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(svc, "_process_resource_snapshot", lambda _pid: {})
    calls = {"count": 0}

    def _flaky_ipc_call(*, reg, payload, timeout_seconds):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError(
                "worker IPC endpoint is unavailable for engine 'ready-model' at 'pipe'; worker process may not be running"
            )
        return {"status": "ok", "pid": 1700, "executable": "python", "prefix": "venv"}

    monkeypatch.setattr(svc, "_ipc_call", _flaky_ipc_call)
    monkeypatch.setattr(svc, "_query_worker_reported_resources", lambda _item: {})

    row = svc.discover_running(reachability_timeout_seconds=0.35)[0]

    assert calls["count"] == 2
    assert row["reachable"] is True
    assert row["state"] == "running"
    assert row["reachability"]["attempts"] == 2


def test_discover_running_uses_ready_worker_pid_for_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="ready-model",
        pid=1700,
        command=["python", "-m", "hosting.engine_worker_ipc"],
        env={"MP13_MODEL_PATH": "C:/models/demo"},
        worker_profile_class="model",
    )
    rows = svc._read_engines()
    rows[0]["worker_pid"] = 2700
    rows[0]["worker_ready_at"] = time.time()
    svc._write_engines(rows)
    monkeypatch.setattr(svc, "_pid_alive", lambda pid: int(pid) in {1700, 2700})
    monkeypatch.setattr(
        svc,
        "_process_resource_snapshot",
        lambda pid: {"memory_mb": 1882.6} if int(pid) == 2700 else {"memory_mb": 4.1},
    )
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {
            "reachable": False,
            "transport": "ipc",
            "probe": "hello",
            "error": "worker IPC endpoint is unavailable for engine 'ready-model' at 'pipe'; worker process may not be running",
        },
    )

    row = svc.discover_running()[0]

    assert row["pid"] == 2700
    assert row["launcher_pid"] == 1700
    assert row["process_resources"]["memory_mb"] == 1882.6
    assert row["state"] == "unreachable"


def test_discover_running_prunes_old_registration_with_missing_ipc_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="stale-tools",
        pid=1700,
        command=["python", "-m", "hosting.toolbox_executor_ipc"],
        env={"MP13_TOOLBOX_EXECUTOR_ENGINE_ID": "stale-tools"},
        executor_kind="toolbox_executor",
    )
    rows = svc._read_engines()
    rows[0]["spawned_at"] = time.time() - 120.0
    svc._write_engines(rows)
    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        svc,
        "_probe_registration_reachability",
        lambda _item, *, timeout_seconds=0.35: {
            "reachable": False,
            "transport": "ipc",
            "probe": "hello",
            "error": "worker IPC endpoint is unavailable for engine 'stale-tools' at 'pipe'; worker process may not be running",
        },
    )

    assert svc.discover_running() == []
    assert svc._read_engines() == []


def test_traffic_scope_engine_allowlist_enforced(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(
        key_id="traffic1",
        key_secret="secret1",
        role="model_user",
        allowed_engines=["worker_a"],
    )
    svc.set_control_config(require_auth=True)

    issued = svc.auth_issue_session(
        key_id="traffic1",
        key_secret="secret1",
        scope="traffic",
        engine_ids=["worker_a"],
        ttl_seconds=300,
    )
    token = str(issued["token"])

    # Allowed engine.
    svc.authorize_command("proxy-request", {"session_token": token, "engine_id": "worker_a"})

    # Disallowed engine.
    with pytest.raises(PermissionError):
        svc.authorize_command("proxy-request", {"session_token": token, "engine_id": "worker_b"})


def test_config_selector_restriction_blocks_path_traversal(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    with pytest.raises(ValueError):
        svc.models_from_config("../secrets")

    with pytest.raises(ValueError):
        svc.connect_from_config(config_path="C:\\windows\\system32\\x.json")


def test_engine_shutdown_terminates_process_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="worker_tree",
        pid=12345,
        command=["python", "-m", "hosting.engine_worker_ipc"],
    )
    alive = {"value": True}
    captured: dict[str, object] = {}

    monkeypatch.setattr(svc, "_pid_alive", lambda _pid: bool(alive["value"]))

    def _fake_terminate(pid: int, *, timeout_seconds: float = 8.0):
        captured["pid"] = pid
        captured["timeout_seconds"] = timeout_seconds
        alive["value"] = False
        return {"pid": pid, "status": "stopped", "alive": False, "children": [23456]}

    monkeypatch.setattr("hosting.service.engines.terminate_process_tree", _fake_terminate)

    out = svc.shutdown("worker_tree", timeout_seconds=1.25)

    assert out["status"] == "stopped"
    assert captured == {"pid": 12345, "timeout_seconds": 1.25}
    assert out["termination"]["children"] == [23456]
    assert svc.get_registration("worker_tree") is None


def test_proxy_request_policy_and_metrics_ring_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(tmp_path)
    _install_ipc_http_stub(monkeypatch)
    svc.register_spawned(
        engine_id="worker1",
        pid=os.getpid(),
        command=["python", "-m", "hosting.engine_worker_ipc"],
    )

    # Restrict to GET /health path.
    svc.set_control_config(
        traffic_policy={
            "allowed_methods": ["GET"],
            "allowed_path_prefixes": ["/health"],
        }
    )

    with pytest.raises(PermissionError):
        svc.proxy_request(engine_id="worker1", method="POST", path="/health")

    with pytest.raises(PermissionError):
        svc.proxy_request(engine_id="worker1", method="GET", path="/other")

    ok = svc.proxy_request(engine_id="worker1", method="GET", path="/health")
    assert int(ok["status_code"]) == 200
    assert ok["truncated"] is False
    payload = base64.b64decode(str(ok["body_b64"]))
    assert b'"ok":true' in payload

    metrics = svc.get_host_metrics()
    assert metrics["require_auth"] is False
    assert metrics["auth_status_error"] is None
    proxy = dict(metrics.get("proxy") or {})
    assert int(proxy.get("total") or 0) >= 1
    assert int(proxy.get("ok") or 0) >= 1
    assert int(proxy.get("inflight_total") or 0) == 0
    recent = list(proxy.get("recent_requests") or [])
    assert len(recent) >= 1
    last = dict(recent[-1])
    assert str(last.get("engine_id") or "") == "worker1"
    assert str(last.get("method") or "") == "GET"
    assert str(last.get("path") or "") == "/health"
    assert int(last.get("status_code") or 0) == 200
    assert str(last.get("outcome") or "") == "ok"


def test_per_engine_traffic_policy_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(tmp_path)
    _install_ipc_http_stub(monkeypatch)
    svc.register_spawned(
        engine_id="worker1",
        pid=os.getpid(),
        command=["python", "-m", "hosting.engine_worker_ipc"],
    )
    svc.register_spawned(
        engine_id="worker2",
        pid=os.getpid(),
        command=["python", "-m", "hosting.engine_worker_ipc"],
    )
    svc.set_control_config(
        traffic_policy={
            "allowed_methods": ["GET"],
            "allowed_path_prefixes": ["/health"],
        },
        engine_traffic_policies={
            "worker2": {
                "allowed_methods": ["GET"],
                "allowed_path_prefixes": ["/other"],
            }
        },
    )

    with pytest.raises(PermissionError):
        svc.proxy_request(engine_id="worker1", method="GET", path="/other")

    # worker2 override allows /other path; backend returns 404 but should pass policy.
    out = svc.proxy_request(engine_id="worker2", method="GET", path="/other")
    assert int(out["status_code"]) == 404


def test_proxy_rpc_reports_clear_error_when_engine_not_registered(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)

    with pytest.raises(ValueError, match="engine 'missing_worker' is not registered"):
        svc.proxy_rpc_call(engine_id="missing_worker", method="run-inference", params={})


def test_proxy_rpc_reports_clear_error_when_worker_ipc_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = _make_service(tmp_path)
    svc.register_spawned(
        engine_id="worker_missing_ipc",
        pid=os.getpid(),
        command=["python", "-m", "hosting.engine_worker_ipc"],
        worker_ipc_family="AF_PIPE" if os.name == "nt" else "AF_UNIX",
        worker_ipc_address="\\\\.\\pipe\\mp13-missing-ipc" if os.name == "nt" else str(tmp_path / "missing.sock"),
    )

    class _MissingPipeClient:
        def __init__(self, *args, **kwargs) -> None:
            raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.setattr("hosting.service.proxy.MPClient", _MissingPipeClient)

    with pytest.raises(RuntimeError, match="worker IPC endpoint is unavailable for engine 'worker_missing_ipc'"):
        svc.proxy_rpc_call(engine_id="worker_missing_ipc", method="run-inference", params={})


def test_auth_audit_sessions_and_tokens_redact_secrets(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(key_id="mgmt1", key_secret="secret1", role="admin")
    svc.set_control_config(require_auth=True)
    issued = svc.auth_issue_session(
        key_id="mgmt1",
        key_secret="secret1",
        scope="control",
        ttl_seconds=300,
    )
    session_token = str(issued["token"])
    token_row = svc.issue_token("worker1", backend_id="backend:abc")
    issued_token = str(token_row["token"])

    sessions = svc.auth_list_sessions()
    assert int(sessions.get("sessions_count") or 0) >= 1
    session_rows = list(sessions.get("sessions") or [])
    assert any(str(r.get("key_id") or "") == "mgmt1" for r in session_rows)
    for r in session_rows:
        preview = str(r.get("token_preview") or "")
        assert preview
        assert preview != session_token

    tokens = svc.auth_list_issued_tokens()
    assert int(tokens.get("engine_tokens_count") or 0) >= 1
    engine_rows = list(tokens.get("engine_tokens") or [])
    assert any(str(r.get("engine_id") or "") == "worker1" for r in engine_rows)
    for r in engine_rows:
        preview = str(r.get("token_preview") or "")
        assert preview
        assert preview != issued_token


def test_auth_audit_commands_require_control_scope_when_auth_enabled(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(key_id="mgmt1", key_secret="secret1", role="admin")
    svc.set_control_config(require_auth=True)

    with pytest.raises(PermissionError):
        svc.authorize_command("auth-list-sessions", {})
    with pytest.raises(PermissionError):
        svc.authorize_command("auth-list-issued-tokens", {})

    issued = svc.auth_issue_session(
        key_id="mgmt1",
        key_secret="secret1",
        scope="control",
        ttl_seconds=300,
    )
    token = str(issued["token"])
    svc.authorize_command("auth-list-sessions", {"session_token": token})
    svc.authorize_command("auth-list-issued-tokens", {"session_token": token})


def test_ssh_session_binding_enforced(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(key_id="mgmt1", key_secret="secret1", role="admin")
    svc.set_control_config(require_auth=True)
    issued = svc.auth_issue_session(
        key_id="mgmt1",
        key_secret="secret1",
        scope="control",
        ttl_seconds=300,
        ssh_binding={"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
    )
    token = str(issued["token"])

    # Missing binding is denied for bound sessions.
    with pytest.raises(PermissionError):
        svc.authorize_command("discover-running", {"session_token": token})

    # Target mismatch is denied.
    with pytest.raises(PermissionError):
        svc.authorize_command(
            "discover-running",
            {
                "session_token": token,
                "_ssh_session_binding": {"target": "user@other-host", "key_fingerprint": "SHA256:abc"},
            },
        )

    # Exact binding is accepted.
    svc.authorize_command(
        "discover-running",
        {
            "session_token": token,
            "_ssh_session_binding": {"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
        },
    )


def test_auth_list_sessions_filter_and_pagination(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(key_id="mgmt1", key_secret="secret1", role="admin")
    svc.auth_upsert_key(key_id="traffic1", key_secret="secret2", role="model_user", allowed_engines=["worker1"])
    svc.set_control_config(require_auth=True)
    _ = svc.auth_issue_session(key_id="mgmt1", key_secret="secret1", scope="control", ttl_seconds=300)
    _ = svc.auth_issue_session(key_id="traffic1", key_secret="secret2", scope="traffic", engine_ids=["worker1"], ttl_seconds=300)

    filtered = svc.auth_list_sessions(scope="traffic", limit=10, offset=0)
    assert int(filtered.get("sessions_count") or 0) >= 1
    rows = list(filtered.get("sessions") or [])
    assert rows
    assert all(str(r.get("scope") or "").lower() == "traffic" for r in rows)

    paged1 = svc.auth_list_sessions(limit=1, offset=0)
    paged2 = svc.auth_list_sessions(limit=1, offset=1)
    assert int(paged1.get("count") or 0) == 1
    assert int(paged2.get("count") or 0) in {0, 1}
    assert int(paged1.get("offset") or 0) == 0
    assert int(paged1.get("limit") or 0) == 1


def test_auth_list_issued_tokens_filter_and_pagination(tmp_path: Path) -> None:
    svc = _make_service(tmp_path)
    _ = svc.issue_token("worker1", backend_id="backend:a")
    _ = svc.issue_token("worker2", backend_id="backend:b")
    _ = svc.issue_resource_token("dataset", "data1", backend_id="backend:a")

    filtered_engine = svc.auth_list_issued_tokens(engine_id="worker1", limit=10, offset=0)
    engine_rows = list(filtered_engine.get("engine_tokens") or [])
    assert engine_rows
    assert all(str(r.get("engine_id") or "") == "worker1" for r in engine_rows)

    filtered_resource = svc.auth_list_issued_tokens(resource_kind="dataset", resource_id="data1", limit=10, offset=0)
    resource_rows = list(filtered_resource.get("resource_tokens") or [])
    assert resource_rows
    assert all(str(r.get("resource_kind") or "") == "dataset" for r in resource_rows)
    assert all(str(r.get("resource_id") or "") == "data1" for r in resource_rows)

    page1 = svc.auth_list_issued_tokens(limit=1, offset=0)
    page2 = svc.auth_list_issued_tokens(limit=1, offset=1)
    assert int(page1.get("count") or 0) == 1
    assert int(page2.get("count") or 0) in {0, 1}
    assert int(page1.get("offset") or 0) == 0
    assert int(page1.get("limit") or 0) == 1


def test_public_key_challenge_flow_issues_session(tmp_path: Path, monkeypatch) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(
        key_id="admin-pub",
        role="admin",
        auth_method="public_key",
        public_key="ssh-ed25519 AAAATESTKEY comment",
    )
    svc.set_control_config(require_auth=True)

    # Shared-secret issuance must be blocked for public_key auth_method.
    with pytest.raises(PermissionError):
        svc.auth_issue_session(
            key_id="admin-pub",
            key_secret="unused",
            scope="control",
            ttl_seconds=300,
        )

    begin = svc.auth_begin_challenge(key_id="admin-pub", scope="control", ttl_seconds=120)
    challenge_id = str(begin.get("challenge_id") or "")
    assert challenge_id

    monkeypatch.setattr(
        EngineHostService,
        "_verify_ssh_signature",
        staticmethod(lambda **_kwargs: True),
    )
    out = svc.auth_complete_challenge(
        challenge_id=challenge_id,
        signature_ssh="-----BEGIN SSH SIGNATURE-----\nFAKE\n-----END SSH SIGNATURE-----",
    )
    assert out["status"] == "ok"
    assert str(out.get("token") or "")


def test_public_key_challenge_invalid_signature_denied(tmp_path: Path, monkeypatch) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(
        key_id="traffic-pub",
        role="model_user",
        auth_method="public_key",
        public_key="ssh-ed25519 AAAATESTKEY comment",
        allowed_engines=["worker1"],
    )
    svc.set_control_config(require_auth=True)
    begin = svc.auth_begin_challenge(
        key_id="traffic-pub",
        scope="traffic",
        ttl_seconds=120,
        engine_ids=["worker1"],
    )
    challenge_id = str(begin.get("challenge_id") or "")
    monkeypatch.setattr(
        EngineHostService,
        "_verify_ssh_signature",
        staticmethod(lambda **_kwargs: False),
    )
    with pytest.raises(PermissionError):
        svc.auth_complete_challenge(
            challenge_id=challenge_id,
            signature_ssh="-----BEGIN SSH SIGNATURE-----\nBAD\n-----END SSH SIGNATURE-----",
        )


def test_challenge_telemetry_tracks_success_and_replay_suspected(tmp_path: Path, monkeypatch) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(
        key_id="admin-pub",
        role="admin",
        auth_method="public_key",
        public_key="ssh-ed25519 AAAATESTKEY comment",
    )
    svc.set_control_config(require_auth=True)
    begin = svc.auth_begin_challenge(key_id="admin-pub", scope="control", ttl_seconds=120)
    cid = str(begin.get("challenge_id") or "")
    monkeypatch.setattr(
        EngineHostService,
        "_verify_ssh_signature",
        staticmethod(lambda **_kwargs: True),
    )
    _ = svc.auth_complete_challenge(
        challenge_id=cid,
        signature_ssh="-----BEGIN SSH SIGNATURE-----\nGOOD\n-----END SSH SIGNATURE-----",
    )
    # Re-using the same challenge_id should be treated as replay-suspected.
    with pytest.raises(PermissionError):
        svc.auth_complete_challenge(
            challenge_id=cid,
            signature_ssh="-----BEGIN SSH SIGNATURE-----\nGOOD\n-----END SSH SIGNATURE-----",
        )

    metrics = svc.get_host_metrics()
    auth = dict(metrics.get("auth") or {})
    assert int(auth.get("challenge_begin_total") or 0) >= 1
    assert int(auth.get("challenge_complete_ok") or 0) >= 1
    assert int(auth.get("challenge_replay_suspected") or 0) >= 1
    recent = list(auth.get("challenge_recent_events") or [])
    assert recent
    assert any(str(ev.get("event") or "") == "complete_ok" for ev in recent)
    assert any(bool(ev.get("replay_suspected")) for ev in recent)


def test_challenge_completion_enforces_ssh_binding(tmp_path: Path, monkeypatch) -> None:
    svc = _make_service(tmp_path)
    svc.auth_upsert_key(
        key_id="admin-pub",
        role="admin",
        auth_method="public_key",
        public_key="ssh-ed25519 AAAATESTKEY comment",
    )
    svc.set_control_config(require_auth=True)
    begin = svc.auth_begin_challenge(
        key_id="admin-pub",
        scope="control",
        ttl_seconds=120,
        ssh_binding={"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
    )
    cid = str(begin.get("challenge_id") or "")
    challenge_txt = str(begin.get("challenge") or "")
    assert "\"ssh_binding_target\":\"user@example-host\"" in challenge_txt
    assert "\"ssh_binding_key_fingerprint\":\"SHA256:abc\"" in challenge_txt

    monkeypatch.setattr(
        EngineHostService,
        "_verify_ssh_signature",
        staticmethod(lambda **_kwargs: True),
    )
    with pytest.raises(PermissionError):
        svc.auth_complete_challenge(
            challenge_id=cid,
            signature_ssh="-----BEGIN SSH SIGNATURE-----\nGOOD\n-----END SSH SIGNATURE-----",
            presented_ssh_binding={"target": "user@other-host", "key_fingerprint": "SHA256:abc"},
        )

    ok = svc.auth_complete_challenge(
        challenge_id=cid,
        signature_ssh="-----BEGIN SSH SIGNATURE-----\nGOOD\n-----END SSH SIGNATURE-----",
        presented_ssh_binding={"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
    )
    assert ok["status"] == "ok"
