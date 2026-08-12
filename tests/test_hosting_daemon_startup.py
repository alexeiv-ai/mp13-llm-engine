from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from hosting.daemon import EngineHostDaemon, run_daemon_foreground
from tests.hosting_v3_fixtures import write_hosting_configuration


class _FakePidFile:
    def __init__(self, events: list[str]):
        self._events = events

    def write(self, *, pid: int, port: int, shutdown_token: str) -> None:
        self._events.append(f"pid_write:{pid}:{port}:{bool(shutdown_token)}")

    def remove(self) -> None:
        self._events.append("pid_remove")


def test_daemon_run_writes_pid_after_local_listener_ready(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        mp13_config_file=write_hosting_configuration(tmp_path),
    )
    daemon.pid_file = _FakePidFile(events)  # type: ignore[assignment]

    def _fake_start_listener() -> None:
        events.append("listener_start")
        assert daemon._stop_event is not None
        daemon._stop_event.set()
        return None

    monkeypatch.setattr(daemon, "_start_local_control_listener", _fake_start_listener)
    monkeypatch.setattr(daemon, "_stop_local_control_listener", lambda: events.append("listener_stop"))

    asyncio.run(daemon.run())

    assert events[0] == "listener_start"
    assert any(x.startswith("pid_write:") for x in events)
    assert events.index("listener_start") < next(i for i, x in enumerate(events) if x.startswith("pid_write:"))


def test_daemon_tcp_control_listener_is_not_supported_even_with_remote_roles(tmp_path) -> None:
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        mp13_config_file=write_hosting_configuration(
            tmp_path,
            require_auth=True,
            connectivity_mode="ssh_tunnel_only",
            lifecycle={"profile": "detached_user_process"},
        ),
    )
    daemon.svc.auth_upsert_key(
        key_id="admin",
        role="admin",
        auth_method="public_key",
        public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAadmin admin@example",
    )
    daemon.svc.auth_upsert_key(
        key_id="transport",
        role="transport",
        auth_method="public_key",
        public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAtransport transport@example",
    )
    assert daemon._should_enable_tcp() is False  # noqa: SLF001


def test_daemon_status_control_command_reports_start_time_and_uptime(monkeypatch, tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        mp13_config_file=write_hosting_configuration(tmp_path),
    )
    daemon._started_at = 100.0  # noqa: SLF001
    daemon._started_monotonic = 10.0  # noqa: SLF001
    monkeypatch.setattr("hosting.daemon.local_ipc.time.time", lambda: 125.0)
    monkeypatch.setattr("hosting.daemon.local_ipc.time.monotonic", lambda: 42.5)

    response = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            '{"seq": 7, "cmd": "daemon-status", "payload": {}}',
            peer_host="127.0.0.1",
        )
    )

    assert response["ok"] is True
    assert response["result"]["started_at"] == 100.0
    assert response["result"]["uptime_seconds"] == 32.5
    assert response["result"]["pid"] == os.getpid()

    ping_response = asyncio.run(
        daemon._dispatch(  # noqa: SLF001
            '{"seq": 8, "cmd": "__ping__", "payload": {}}',
            peer_host="127.0.0.1",
        )
    )
    assert ping_response["result"] == "pong"
    assert ping_response["started_at"] == 100.0
    assert ping_response["uptime_seconds"] == 32.5


def test_daemon_startup_recovery_stops_foreign_owner_registrations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        mp13_config_file=write_hosting_configuration(tmp_path),
    )
    daemon.svc.register_spawned(
        engine_id="foreign-worker",
        pid=12345,
        command=["python", "-m", "hosting.engine_worker_ipc"],
    )
    rows = daemon.svc._read_engines()  # noqa: SLF001
    rows[0]["owner_host_pid"] = os.getpid() + 1000
    daemon.svc._write_engines(rows)  # noqa: SLF001
    calls: list[tuple[str, float]] = []

    def _fake_shutdown(engine_id: str, *, timeout_seconds: float = 8.0) -> dict[str, object]:
        calls.append((engine_id, timeout_seconds))
        return {"status": "stopped", "engine_id": engine_id, "alive": False}

    monkeypatch.setattr(daemon.svc, "shutdown", _fake_shutdown)

    report = daemon._execute_startup_worker_recovery()  # noqa: SLF001

    assert report["foreign_attempted"] == 1
    assert report["foreign_stopped"] == 1
    assert calls == [("foreign-worker", 3.0)]


def test_normal_daemon_wires_generic_package_sources_and_policy(tmp_path: Path) -> None:
    mp13_config = write_hosting_configuration(
        tmp_path,
        package_sources={
            "release": {
                "kind": "airgap_store",
                "locator": "@packages/artifacts",
                "enabled": True,
                "priority": 1,
            }
        },
        dependency_policy={
            "policy_id": "release-only",
            "revision": 3,
            "allowed_source_ids": ["release"],
            "allowed_platforms": ["*"],
            "allowed_runtimes": ["python"],
            "max_artifact_bytes": 4096,
            "require_sha256": True,
            "optional_verifier": None,
        },
    )
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        mp13_config_file=mp13_config,
    )

    package = dict(daemon.svc.hosting_configuration.package_management)
    assert dict(package["sources"])["release"]["kind"] == "airgap_store"
    assert dict(package["dependency_policy"])["policy_id"] == "release-only"
    assert daemon.svc._hermetic_toolbox_environment_builder is not None  # noqa: SLF001
    summary = daemon.svc.hosting_setup_summary()
    assert summary["configuration"]["source_availability"] == {
        "release": {"configured": True, "enabled": True}
    }
    assert "toolbox_host_project" not in summary


def test_foreground_production_launcher_forwards_only_mp13_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeDaemon:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)

        async def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr("hosting.daemon.foreground.EngineHostDaemon", _FakeDaemon)
    monkeypatch.setattr(
        "hosting.daemon.foreground._apply_foreground_terminal_disconnect_policy",
        lambda _daemon: None,
    )

    mp13_config = write_hosting_configuration(tmp_path)
    run_daemon_foreground(port=0, mp13_config_file=mp13_config)

    assert captured["mp13_config_file"] == mp13_config
    assert not any(key.startswith("toolbox_") for key in captured)
    assert captured["ran"] is True


def test_foreground_production_launcher_uses_configuration_path_without_loading_secrets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    config_file = write_hosting_configuration(
        tmp_path,
        package_credentials={"private": "SENTINEL_SECRET"},
    )

    class _FakeDaemon:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)

        async def run(self) -> None:
            return

    monkeypatch.setattr("hosting.daemon.foreground.EngineHostDaemon", _FakeDaemon)
    monkeypatch.setattr(
        "hosting.daemon.foreground._apply_foreground_terminal_disconnect_policy",
        lambda _daemon: None,
    )

    run_daemon_foreground(mp13_config_file=config_file)

    assert captured["mp13_config_file"] == config_file
    assert "SENTINEL_SECRET" not in str(captured)


def test_missing_package_source_and_environment_template_are_bounded(tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        mp13_config_file=write_hosting_configuration(tmp_path),
    )
    readiness = {row["subsystem"]: row for row in daemon.svc.hosting_setup_summary()["readiness"]}
    assert readiness["control"]["code"] == "ready"
    assert readiness["package"]["code"] == "package_source_unavailable"
    assert readiness["environment"]["code"] == "environment_template_unavailable"
