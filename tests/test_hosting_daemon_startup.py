from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hosting.daemon import EngineHostDaemon


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
