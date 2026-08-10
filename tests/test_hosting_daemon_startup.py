from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from hosting.daemon import EngineHostDaemon
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.target import detect_current_toolbox_target


class _FakePidFile:
    def __init__(self, events: list[str]):
        self._events = events

    def write(self, *, pid: int, port: int, shutdown_token: str) -> None:
        self._events.append(f"pid_write:{pid}:{port}:{bool(shutdown_token)}")

    def remove(self) -> None:
        self._events.append("pid_remove")


def test_daemon_run_writes_pid_after_local_listener_ready(monkeypatch) -> None:
    events: list[str] = []
    daemon = EngineHostDaemon(
        port=0,
        pid_file=Path("X:/tmp/daemon.pid"),
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
        control_state_file=tmp_path / "control.json",
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
    daemon.svc.set_control_config(
        require_auth=True,
        access_profile={"connectivity_mode": "ssh_tunnel_only"},
        lifecycle_profile="detached_user_process",
    )

    assert daemon._should_enable_tcp() is False  # noqa: SLF001


def test_daemon_status_control_command_reports_start_time_and_uptime(monkeypatch, tmp_path: Path) -> None:
    daemon = EngineHostDaemon(pid_file=tmp_path / "daemon.pid")
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
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "control.json",
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


def test_normal_daemon_wires_strict_toolbox_configuration_sources_and_policy(
    tmp_path: Path,
) -> None:
    target = detect_current_toolbox_target()
    source_root = tmp_path / "airgap"
    source_root.mkdir()
    configuration = {
        "builtins": [
            {
                "template_id": template_id,
                "imports": ["hosting"],
                "package_requirements": [],
                "sandbox_policy": "compute-only",
                "required": True,
                "prewarm": False,
                "provenance": "parent-release",
            }
            for template_id in ("core", "py-compute")
        ],
        "sources": [
            {
                "source_id": "parent-release-resources",
                "kind": "airgap_store",
                "origin": "airgap://parent-release-resources",
                "credential_ref": None,
                "allowed_package_namespaces": ["*"],
                "priority": 100,
                "trust_key_ids": ["parent-release-toolbox-v1"],
                "maximum_download_bytes": 536_870_912,
            }
        ],
        "resolution": {
            "mode": "air_gapped",
            "timeout_seconds": 300,
            "maximum_bytes": 536_870_912,
            "maximum_artifacts": 256,
            "allowed_redirect_origins": [],
            "wheel_only": True,
        },
        "retention": {
            "artifact_cache_grace_seconds": 604_800,
            "maximum_cache_bytes": 10_737_418_240,
            "maximum_cache_artifacts": 4096,
            "protected_digests": [],
            "remove_unreferenced_custom_revisions_on_apply": False,
        },
    }
    policy_body = {
        "allowed_template_ids": ["core", "py-compute"],
        "allowed_targets": [target.name],
        "package_allowlist": [],
        "package_denylist": [],
        "allow_custom": False,
        "custom_requires_approval": True,
        "online_resolution_allowed": False,
        "allowed_index_origins": [],
    }
    dependency_policy = {
        "revision": identity_digest("hosting.toolbox.test.policy.v1", policy_body),
        **policy_body,
    }

    daemon = EngineHostDaemon(
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "control.json",
        toolbox_host_project_configuration=configuration,
        toolbox_artifact_sources={"parent-release-resources": source_root},
        toolbox_dependency_policy=dependency_policy,
    )

    assert daemon.svc._toolbox_target == target  # noqa: SLF001
    assert daemon.svc._hermetic_toolbox_environment_builder is not None  # noqa: SLF001
    assert daemon.svc._configured_toolbox_dependency_policy.to_dict() == dependency_policy  # noqa: SLF001
    assert daemon.svc._toolbox_startup["status"] == "configured"  # noqa: SLF001
    summary = daemon.svc.hosting_setup_summary()
    assert summary["toolbox_readiness"]["status"] == "degraded"
    assert summary["toolbox_host_project"]["target"]["platform"] == target.platform
    assert "credential_ref" not in str(summary["toolbox_host_project"])
