from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from hosting.client_realm import FileSecretStore, write_client_profile
from hosting.engine_host_channel import EngineHostControlChannel


class _FakeConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        p = dict(payload or {})
        self.calls.append((cmd, p))
        if cmd == "auth-issue-session":
            return {"token": "tok-123"}
        if cmd == "discover-running":
            return []
        return {}

    def is_alive(self) -> bool:
        return True

    def close(self) -> None:
        return


def test_auto_session_from_key_credentials() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel(
        {
            "engine_host_key_id": "mgmt-key",
            "engine_host_key_secret": "secret-1",
            "engine_host_session_scope": "control",
            "engine_host_daemon_auto_bootstrap": False,
        }
    )
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    rows = ch.discover_running()
    assert rows == []
    assert len(fake.calls) == 2
    assert fake.calls[0][0] == "auth-issue-session"
    assert fake.calls[0][1]["key_id"] == "mgmt-key"
    assert fake.calls[1][0] == "discover-running"
    assert fake.calls[1][1]["session_token"] == "tok-123"
    assert ch.get_session_token() == "tok-123"


def test_daemon_status_includes_auth_snapshot(monkeypatch) -> None:
    class _FakePidFile:
        def __init__(self, _path: Optional[str] = None):
            self.path = "X:/tmp/daemon.pid"

        def read(self) -> Dict[str, Any]:
            return {"pid": 9999, "port": 19876, "started_at": 123.0}

        def is_alive(self) -> bool:
            return True

    class _FakeSocket:
        def __init__(self, **_kwargs: Any):
            return

        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            if cmd == "__ping__":
                assert payload == {}
                return "pong"
            assert cmd == "auth-status"
            assert payload == {}
            return {
                "daemon_version": "2.1.0",
                "capabilities": {
                    "claim_acl_v2": True,
                    "structured_denials_v1": True,
                    "force_override_confirmation_v1": True,
                    "structured_progress_events_v1": True,
                    "reachability_status_v1": True,
                    "non_blocking_ops_v1": True,
                },
                "require_auth": False,
                "keys_count": 0,
                "sessions_count": 0,
                "roles": [],
            }

        def close(self) -> None:
            return

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeSocket)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_read_local_control_snapshot",
        lambda self: {
            "require_auth": False,
            "keys_count": 0,
            "endpoint_mode_default": "exclusive",
        },
    )

    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    status = ch.get_daemon_status()
    assert status["alive"] is True
    assert status["pid_alive"] is True
    assert status["reachable"] is True
    assert status["reachability_error"] is None
    assert status["pid"] == 9999
    assert status["port"] == 19876
    assert status["auth_status"] == {
        "daemon_version": "2.1.0",
        "capabilities": {
            "claim_acl_v2": True,
            "structured_denials_v1": True,
            "force_override_confirmation_v1": True,
            "structured_progress_events_v1": True,
            "reachability_status_v1": True,
            "non_blocking_ops_v1": True,
        },
        "require_auth": False,
        "keys_count": 0,
        "sessions_count": 0,
        "roles": [],
    }
    assert status["auth_status_error"] is None
    assert status["require_auth"] is False
    assert status["keys_count"] == 0
    assert status["endpoint_mode_default"] == "exclusive"
    assert len(list(status["warnings"] or [])) == 1
    assert "Configure hosting_access as soon as possible" in str(status["warnings"][0] or "")
    assert status["status_event"] is None


def test_ssh_mode_injects_binding_without_auto_shared_secret_bootstrap() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel(
        {
            "engine_host_key_id": "mgmt-key",
            "engine_host_key_secret": "secret-1",
            "engine_host_session_scope": "control",
            "engine_host_ssh_target": "user@example-host",
            "control_ssh_key": "C:/keys/id_ed25519",
            "control_ssh_fingerprint": "SHA256:abc",
            "engine_host_daemon_auto_bootstrap": False,
        }
    )
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    _ = ch.discover_running()
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "discover-running"
    assert fake.calls[0][1]["_ssh_session_binding"] == {
        "target": "user@example-host",
        "key_fingerprint": "SHA256:abc",
    }
    assert "session_token" not in fake.calls[0][1]


def test_channel_init_resolves_client_profile_into_ssh_settings(tmp_path: Path) -> None:
    realm_root = tmp_path / "client-realm"
    store = FileSecretStore(realm_root, realm="default")
    store.put_secret(
        tag="transport_private_key",
        payload="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----\n",
        secret_id="transport-key",
    )
    known_hosts_file = realm_root / "known_hosts" / "demo.known_hosts"
    known_hosts_file.parent.mkdir(parents=True, exist_ok=True)
    known_hosts_file.write_text("example ssh-ed25519 AAAATEST\n", encoding="utf-8")
    write_client_profile(
        realm_root,
        "demo",
        {
            "engine_host_ssh_target": "user@example-host",
            "control_ssh_key_secret_id": "transport-key",
            "ssh_known_hosts_file": str(known_hosts_file),
            "control_ssh_fingerprint": "SHA256:abc",
        },
    )

    ch = EngineHostControlChannel(
        {
            "engine_host_client_realm_root": str(realm_root),
            "engine_host_client_profile": "demo",
            "engine_host_daemon_auto_bootstrap": False,
        }
    )

    assert ch.control_settings["engine_host_ssh_target"] == "user@example-host"
    assert ch.control_settings["ssh_known_hosts_line"] == "example ssh-ed25519 AAAATEST"
    assert ch.control_settings["control_ssh_fingerprint"] == "SHA256:abc"
    assert Path(str(ch.control_settings["control_ssh_key"])).exists()


def test_raw_auth_begin_challenge_includes_ssh_binding_for_remote_target() -> None:
    class FakeConn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def invoke(self, command: str, payload: dict) -> dict:
            self.calls.append((command, dict(payload)))
            return {"challenge_id": "chal-1", "challenge": "payload"}

    fake = FakeConn()
    ch = EngineHostControlChannel(
        {
            "engine_host_ssh_target": "user@example-host",
            "control_ssh_key": "C:/keys/transport_ed25519",
            "control_ssh_fingerprint": "SHA256:abc",
            "ssh_known_hosts_line": "example-host ssh-ed25519 AAAATEST",
        }
    )
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    out = ch.invoke_control_command("auth-begin-challenge", {"key_id": "admin-main", "scope": "control"})

    assert out["challenge_id"] == "chal-1"
    assert fake.calls == [
        (
            "auth-begin-challenge",
            {
                "key_id": "admin-main",
                "scope": "control",
                "ssh_binding": {
                    "target": "user@example-host",
                    "key_fingerprint": "SHA256:abc",
                },
            },
        )
    ]
    target = ch.get_target()
    assert target["mode"] == "ssh"
    assert target["target"] == "user@example-host"


def test_set_control_config_forwards_lifecycle_fields() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]
    ch.set_session_token("tok-123")

    out = ch.set_control_config(
        lifecycle_profile="service_managed",
        lifecycle_policy={
            "on_terminal_disconnect": "keep_daemon_running",
            "terminal_control_enabled": False,
            "owner_disconnect_shutdown": False,
        },
    )
    assert out == {}
    assert fake.calls
    cmd, payload = fake.calls[-1]
    assert cmd == "set-control-config"
    assert str(payload.get("lifecycle_profile") or "") == "service_managed"
    assert dict(payload.get("lifecycle_policy") or {}) == {
        "on_terminal_disconnect": "keep_daemon_running",
        "terminal_control_enabled": False,
        "owner_disconnect_shutdown": False,
    }
    assert str(payload.get("session_token") or "") == "tok-123"


def test_sandbox_fs_channel_methods_forward_expected_payloads() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]
    ch.set_session_token("tok-123")

    callback_context = {"tool_name": "peek", "tool_call_id": "call-1"}
    ch.sandbox_fs_list(engine_id="worker1", root_id="rw", relative_path="nested", callback_context=callback_context)
    ch.sandbox_fs_read_text(engine_id="worker1", root_id="rw", relative_path="nested/a.txt", callback_context=callback_context)
    ch.sandbox_fs_write_text(engine_id="worker1", root_id="rw", relative_path="nested/a.txt", text="hello")
    ch.sandbox_fs_mkdir(engine_id="worker1", root_id="rw", relative_path="nested")
    ch.sandbox_fs_stat(engine_id="worker1", root_id="rw", relative_path="nested/a.txt")

    assert [cmd for cmd, _ in fake.calls] == [
        "sandbox-fs-list",
        "sandbox-fs-read-text",
        "sandbox-fs-write-text",
        "sandbox-fs-mkdir",
        "sandbox-fs-stat",
    ]
    assert fake.calls[0][1] == {
        "engine_id": "worker1",
        "root_id": "rw",
        "relative_path": "nested",
        "callback_context": {"tool_name": "peek", "tool_call_id": "call-1"},
        "session_token": "tok-123",
    }
    assert fake.calls[2][1]["text"] == "hello"
    assert fake.calls[2][1]["session_token"] == "tok-123"


def test_sandbox_http_fetch_channel_method_forwards_expected_payload() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]
    ch.set_session_token("tok-123")

    ch.sandbox_http_fetch(
        engine_id="worker1",
        url="https://example.com/api/test",
        method="POST",
        headers={"Content-Type": "application/json"},
        body_b64="e30=",
        timeout_seconds=5.0,
        max_response_bytes=512,
        callback_context={"tool_name": "http_tool", "tool_call_id": "call-http-1"},
    )

    assert fake.calls == [
        (
            "sandbox-http-fetch",
            {
                "engine_id": "worker1",
                "url": "https://example.com/api/test",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body_b64": "e30=",
                "timeout_seconds": 5.0,
                "max_response_bytes": 512,
                "callback_context": {"tool_name": "http_tool", "tool_call_id": "call-http-1"},
                "session_token": "tok-123",
            },
        )
    ]


def test_toolbox_lifecycle_channel_methods_forward_expected_payloads() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]
    ch.set_session_token("tok-123")

    ch.toolbox_register_auto(
        toolbox_id="toolbox-demo",
        requests=[{"module_name": "demo_mod", "callable_name": "demo_tool"}],
        python_executable="python-demo",
    )
    ch.toolbox_unregister_auto(
        toolbox_id="toolbox-demo",
        tool_keys=["demo_mod:demo_tool"],
        python_executable="python-demo",
    )
    ch.toolbox_register_intrinsics(
        toolbox_id="toolbox-demo",
        intrinsic_tool_names=["symbolic_algebra"],
        include_guides=True,
        python_executable="python-demo",
    )
    ch.toolbox_unregister_intrinsics(
        toolbox_id="toolbox-demo",
        intrinsic_tool_names=["symbolic_algebra"],
        include_guides=True,
        python_executable="python-demo",
    )
    ch.toolbox_register_manual(
        toolbox_id="toolbox-demo",
        requests=[{"module_name": "manual_mod", "callable_name": "manual_tool", "tool_definition": {"function": {"name": "manual_tool"}}}],
        python_executable="python-demo",
    )
    ch.toolbox_unregister_manual(
        toolbox_id="toolbox-demo",
        tool_keys=["manual:manual_mod:manual_tool"],
        python_executable="python-demo",
    )
    ch.toolbox_environment_description_upsert(
        name="math-env",
        base_env_name="base",
        extra_packages=["numpy", "sympy"],
        allow_online_install=False,
    )
    ch.toolbox_environment_description_clone(
        source_name="math-env",
        target_name="math-env-v2",
        extra_packages=["numpy", "sympy", "pandas"],
        allow_online_install=False,
    )
    ch.toolbox_environment_description_list()
    ch.toolbox_environment_resolve_requirements(
        toolbox_id="toolbox-demo",
        environment_name="math-env",
        tool_keys=["demo_mod:demo_tool"],
    )
    ch.toolbox_environment_apply(
        environment_name="math-env",
        toolbox_ids=["toolbox-demo"],
    )
    ch.toolbox_environment_realize(
        toolbox_id="toolbox-demo",
        environment_name="math-env",
        tool_keys=["demo_mod:demo_tool"],
    )
    ch.toolbox_environment_sync_description(
        toolbox_id="toolbox-demo",
        source_environment_name="math-env",
        target_environment_name="math-env-v2",
        tool_keys=["demo_mod:demo_tool"],
        apply=True,
        realize=True,
    )
    ch.toolbox_environment_prepare_install(
        toolbox_id="toolbox-demo",
        environment_name="math-env",
        tool_keys=["demo_mod:demo_tool"],
    )
    ch.toolbox_environment_lock_install(
        toolbox_id="toolbox-demo",
        environment_name="math-env",
        tool_keys=["demo_mod:demo_tool"],
    )
    ch.toolbox_environment_resolve_install_lock(
        toolbox_id="toolbox-demo",
        environment_name="math-env",
        tool_keys=["demo_mod:demo_tool"],
        allow_resolution=True,
    )
    ch.toolbox_environment_verify_install_lock(
        toolbox_id="toolbox-demo",
        environment_name="math-env",
        tool_keys=["demo_mod:demo_tool"],
    )
    ch.toolbox_environment_verify_install_receipt(
        toolbox_id="toolbox-demo",
        environment_name="math-env",
        tool_keys=["demo_mod:demo_tool"],
    )
    ch.toolbox_environment_execute_install(
        toolbox_id="toolbox-demo",
        environment_name="math-env",
        tool_keys=["demo_mod:demo_tool"],
        allow_execution=True,
    )
    ch.toolbox_gate(
        toolbox_id="toolbox-demo",
        tool_name="demo_tool",
    )
    ch.toolbox_cancel(
        toolbox_id="toolbox-demo",
        tool_name="demo_tool",
        tool_call_id="call-demo-1",
        timeout_seconds=3.0,
    )
    ch.toolbox_gc()
    ch.toolbox_references()
    ch.toolbox_consistency()
    ch.toolbox_review_snapshot(toolbox_ids=["toolbox-demo"])
    ch.toolbox_repair(toolbox_ids=["toolbox-demo"], only_inconsistent=False)
    ch.toolbox_reconcile(toolbox_ids=["toolbox-demo"], only_inconsistent=False)

    assert fake.calls == [
        (
            "toolbox-register-auto",
            {
                "toolbox_id": "toolbox-demo",
                "requests": [{"module_name": "demo_mod", "callable_name": "demo_tool"}],
                "python_executable": "python-demo",
                "worker_profile_class": "generic",
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-unregister-auto",
            {
                "toolbox_id": "toolbox-demo",
                "tool_keys": ["demo_mod:demo_tool"],
                "python_executable": "python-demo",
                "worker_profile_class": "generic",
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-register-intrinsics",
            {
                "toolbox_id": "toolbox-demo",
                "intrinsic_tool_names": ["symbolic_algebra"],
                "include_guides": True,
                "sandbox_profile": None,
                "python_executable": "python-demo",
                "worker_profile_class": "generic",
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-unregister-intrinsics",
            {
                "toolbox_id": "toolbox-demo",
                "intrinsic_tool_names": ["symbolic_algebra"],
                "include_guides": True,
                "python_executable": "python-demo",
                "worker_profile_class": "generic",
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-register-manual",
            {
                "toolbox_id": "toolbox-demo",
                "requests": [{"module_name": "manual_mod", "callable_name": "manual_tool", "tool_definition": {"function": {"name": "manual_tool"}}}],
                "python_executable": "python-demo",
                "worker_profile_class": "generic",
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-unregister-manual",
            {
                "toolbox_id": "toolbox-demo",
                "tool_keys": ["manual:manual_mod:manual_tool"],
                "python_executable": "python-demo",
                "worker_profile_class": "generic",
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-upsert",
            {
                "name": "math-env",
                "base_env_name": "base",
                "extra_packages": ["numpy", "sympy"],
                "allow_online_install": False,
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-clone",
            {
                "source_name": "math-env",
                "target_name": "math-env-v2",
                "extra_packages": ["numpy", "sympy", "pandas"],
                "allow_online_install": False,
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-list",
            {
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-resolve",
            {
                "toolbox_id": "toolbox-demo",
                "environment_name": "math-env",
                "tool_keys": ["demo_mod:demo_tool"],
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-apply",
            {
                "environment_name": "math-env",
                "toolbox_ids": ["toolbox-demo"],
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-realize",
            {
                "toolbox_id": "toolbox-demo",
                "environment_name": "math-env",
                "tool_keys": ["demo_mod:demo_tool"],
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-sync",
            {
                "toolbox_id": "toolbox-demo",
                "source_environment_name": "math-env",
                "target_environment_name": "math-env-v2",
                "tool_keys": ["demo_mod:demo_tool"],
                "apply": True,
                "realize": True,
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-prepare-install",
            {
                "toolbox_id": "toolbox-demo",
                "environment_name": "math-env",
                "tool_keys": ["demo_mod:demo_tool"],
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-lock-install",
            {
                "toolbox_id": "toolbox-demo",
                "environment_name": "math-env",
                "tool_keys": ["demo_mod:demo_tool"],
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-resolve-install-lock",
            {
                "toolbox_id": "toolbox-demo",
                "environment_name": "math-env",
                "tool_keys": ["demo_mod:demo_tool"],
                "allow_resolution": True,
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-verify-install-lock",
            {
                "toolbox_id": "toolbox-demo",
                "environment_name": "math-env",
                "tool_keys": ["demo_mod:demo_tool"],
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-verify-install-receipt",
            {
                "toolbox_id": "toolbox-demo",
                "environment_name": "math-env",
                "tool_keys": ["demo_mod:demo_tool"],
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-environment-execute-install",
            {
                "toolbox_id": "toolbox-demo",
                "environment_name": "math-env",
                "tool_keys": ["demo_mod:demo_tool"],
                "allow_execution": True,
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-gate",
            {
                "engine_id": "",
                "toolbox_id": "toolbox-demo",
                "tool_name": "demo_tool",
                "tools_view": None,
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-cancel",
            {
                "engine_id": "",
                "toolbox_id": "toolbox-demo",
                "tool_name": "demo_tool",
                "tool_call_id": "call-demo-1",
                "timeout_seconds": 3.0,
                "respawn": True,
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-gc",
            {
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-references",
            {
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-consistency",
            {
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-review-snapshot",
            {
                "toolbox_ids": ["toolbox-demo"],
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-repair",
            {
                "toolbox_ids": ["toolbox-demo"],
                "only_inconsistent": False,
                "details": False,
                "session_token": "tok-123",
            },
        ),
        (
            "toolbox-reconcile",
            {
                "toolbox_ids": ["toolbox-demo"],
                "only_inconsistent": False,
                "details": False,
                "session_token": "tok-123",
            },
        ),
    ]


def test_bootstrap_daemon_forwards_custom_pid_file(monkeypatch) -> None:
    custom_pid_file = Path("X:/tmp/custom_host.pid")
    captured: Dict[str, Any] = {}

    class _FakePidFile:
        def __init__(self, _path: Optional[str] = None):
            return

        def is_alive(self) -> bool:
            return False

    def _fake_start_daemon_background(
        *,
        port: int,
        pid_file: Optional[Path] = None,
        wait_ready_seconds: float = 8.0,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        captured["port"] = port
        captured["pid_file"] = pid_file
        captured["wait_ready_seconds"] = wait_ready_seconds
        return {"pid": 12345, "port": 19876}

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr(
        "hosting.daemon.start_daemon_background",
        _fake_start_daemon_background,
    )
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_prepare_local_unconfigured_bootstrap",
        lambda self: {
            "require_auth": False,
            "endpoint_mode_default": "exclusive",
            "keys_count": 0,
        },
    )

    ch = EngineHostControlChannel(
        {
            "engine_host_daemon_pid_file": str(custom_pid_file),
            "engine_host_daemon_auto_bootstrap": True,
        }
    )
    ch.get_daemon_status = lambda: {"alive": True, "port": 19876}  # type: ignore[method-assign]

    result = ch.bootstrap_daemon(wait_ready_seconds=1.5)

    assert captured["pid_file"] == custom_pid_file
    assert captured["wait_ready_seconds"] == 1.5
    assert result["alive"] is True
    assert result["bootstrap_control_config"] == {
        "require_auth": False,
        "endpoint_mode_default": "exclusive",
        "keys_count": 0,
    }


def test_get_connection_auto_bootstrap_forwards_custom_pid_file(monkeypatch) -> None:
    custom_pid_file = Path("X:/tmp/custom_host.pid")
    captured: Dict[str, Any] = {}

    class _FakePidFile:
        def __init__(self, _path: Optional[str] = None):
            return

        def get_port(self) -> int:
            return 0

        def is_alive(self) -> bool:
            return False

    class _FakeSocket:
        def __init__(self, *, port: int, timeout: float, **_kwargs: Any):
            self.port = port
            self.timeout = timeout

        def is_alive(self) -> bool:
            return True

        def close(self) -> None:
            return

    def _fake_start_daemon_background(
        *,
        port: int,
        pid_file: Optional[Path] = None,
        wait_ready_seconds: float = 8.0,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        captured["port"] = port
        captured["pid_file"] = pid_file
        captured["wait_ready_seconds"] = wait_ready_seconds
        return {"pid": 12345, "port": 24444}

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr(
        "hosting.daemon.start_daemon_background",
        _fake_start_daemon_background,
    )
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeSocket)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_prepare_local_unconfigured_bootstrap",
        lambda self: {
            "require_auth": False,
            "endpoint_mode_default": "exclusive",
            "keys_count": 0,
        },
    )

    ch = EngineHostControlChannel(
        {
            "engine_host_daemon_pid_file": str(custom_pid_file),
            "engine_host_daemon_auto_bootstrap": True,
        }
    )
    conn = ch._get_connection()

    assert captured["pid_file"] == custom_pid_file
    assert captured["wait_ready_seconds"] == 8.0
    assert conn is not None
    assert getattr(conn, "port", None) == 24444


def test_auto_bootstrap_forwards_daemon_log_file(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    class _FakePidFile:
        def __init__(self, _path: Optional[str] = None):
            return

        def get_port(self) -> int:
            return 0

        def is_alive(self) -> bool:
            return False

    class _FakeSocket:
        def __init__(self, *, port: int, timeout: float, **_kwargs: Any):
            self.port = port
            self.timeout = timeout

        def is_alive(self) -> bool:
            return True

        def close(self) -> None:
            return

    def _fake_start_daemon_background(
        *,
        port: int,
        pid_file: Optional[Path] = None,
        log_file: Optional[Path] = None,
        wait_ready_seconds: float = 8.0,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        captured["port"] = port
        captured["pid_file"] = pid_file
        captured["log_file"] = log_file
        captured["wait_ready_seconds"] = wait_ready_seconds
        return {"pid": 12345, "port": 24444, "log_file": str(log_file) if log_file else None}

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr(
        "hosting.daemon.start_daemon_background",
        _fake_start_daemon_background,
    )
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeSocket)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_prepare_local_unconfigured_bootstrap",
        lambda self: {
            "require_auth": False,
            "endpoint_mode_default": "exclusive",
            "keys_count": 0,
        },
    )

    ch = EngineHostControlChannel(
        {
            "engine_host_daemon_auto_bootstrap": True,
            "engine_host_daemon_log_file": "X:/tmp/daemon.log",
        }
    )
    conn = ch._get_connection()

    assert captured["log_file"] == Path("X:/tmp/daemon.log")
    assert conn is not None
    assert getattr(conn, "port", None) == 24444


def test_daemon_status_marks_unreachable_when_ping_fails(monkeypatch) -> None:
    class _FakePidFile:
        def __init__(self, _path: Optional[str] = None):
            self.path = "X:/tmp/daemon.pid"

        def read(self) -> Dict[str, Any]:
            return {"pid": 9999, "port": 19876, "started_at": 123.0}

        def is_alive(self) -> bool:
            return True

    class _FakeSocket:
        def __init__(self, **_kwargs: Any):
            return

        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
            if cmd == "__ping__":
                raise RuntimeError("connect refused")
            raise AssertionError(f"unexpected command {cmd}")

        def close(self) -> None:
            return

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeSocket)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_read_local_control_snapshot",
        lambda self: {
            "require_auth": True,
            "keys_count": 1,
            "endpoint_mode_default": "shared",
        },
    )

    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    status = ch.get_daemon_status()
    assert status["pid_alive"] is True
    assert status["reachable"] is False
    assert status["alive"] is False
    assert "connect refused" in str(status["reachability_error"] or "")
    assert status["require_auth"] is True
    assert status["keys_count"] == 1
    assert status["endpoint_mode_default"] == "shared"
    assert status["status_event"] is None


def test_prepare_local_unconfigured_bootstrap_forces_no_auth_exclusive(monkeypatch) -> None:
    captured: Dict[str, Any] = {}
    control_state_path = Path("X:/tmp/access_control.json")

    class _FakeSvc:
        def __init__(self, *, control_state_file: Optional[Path] = None, **_kwargs: Any):
            captured["control_state_file"] = control_state_file

        def get_control_config(self) -> Dict[str, Any]:
            return {
                "require_auth": True,
                "keys_count": 0,
                "endpoint_mode_default": "shared",
            }

        def set_control_config(self, **kwargs: Any) -> Dict[str, Any]:
            captured["set_control_config"] = dict(kwargs)
            return {
                "require_auth": False,
                "keys_count": 0,
                "endpoint_mode_default": "exclusive",
                "access_profile": {"connectivity_mode": "local_only"},
            }

    monkeypatch.setattr("hosting.service.host_service.EngineHostService", _FakeSvc)

    ch = EngineHostControlChannel(
        {
            "engine_host_control_state_file": str(control_state_path),
            "engine_host_daemon_auto_bootstrap": False,
        }
    )
    out = ch._prepare_local_unconfigured_bootstrap()

    assert captured["control_state_file"] == control_state_path
    assert captured["set_control_config"] == {
        "require_auth": False,
        "access_profile": {"connectivity_mode": "local_only"},
        "endpoint_mode_default": "exclusive",
    }
    assert out == {
        "require_auth": False,
        "keys_count": 0,
        "endpoint_mode_default": "exclusive",
        "access_profile": {"connectivity_mode": "local_only"},
    }


def test_daemon_status_emits_event_when_pid_file_disappears(monkeypatch) -> None:
    state = {"present": True}

    class _FakePidFile:
        def __init__(self, _path: Optional[str] = None):
            self.path = "X:/tmp/daemon.pid"

        def read(self) -> Dict[str, Any]:
            if state["present"]:
                return {"pid": 9999, "port": 19876, "started_at": 123.0}
            return {}

        def is_alive(self) -> bool:
            return bool(state["present"])

    class _FakeSocket:
        def __init__(self, **_kwargs: Any):
            return

        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
            if cmd == "__ping__":
                return "pong"
            if cmd == "auth-status":
                return {"require_auth": False, "keys_count": 0, "sessions_count": 0}
            raise AssertionError(f"unexpected command {cmd}")

        def close(self) -> None:
            return

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeSocket)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_read_local_control_snapshot",
        lambda self: {
            "require_auth": False,
            "keys_count": 0,
            "endpoint_mode_default": "exclusive",
        },
    )

    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    first = ch.get_daemon_status()
    assert first["status_event"] is None

    state["present"] = False
    second = ch.get_daemon_status()
    assert dict(second["status_event"] or {})["event"] == "daemon_status_changed"
    assert dict(second["status_event"] or {})["reason"] == "pid_file_removed"


def test_reset_hosting_access_is_local_helper_only(monkeypatch, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}

    class _FakePidFile:
        def __init__(self, _path: Optional[str] = None):
            self.path = "X:/tmp/daemon.pid"

        def read(self) -> Dict[str, Any]:
            return {}

        def is_alive(self) -> bool:
            return False

        def remove(self) -> None:
            captured["pid_removed"] = True

    class _FakeSvc:
        def __init__(self, *, control_state_file: Optional[Path] = None, **_kwargs: Any):
            captured["control_state_file"] = control_state_file

        def reset_hosting_access(self) -> Dict[str, Any]:
            return {"status": "ok", "cleared_keys": 1, "cleared_sessions": 0, "cleared_challenges": 0}

        def get_control_config(self) -> Dict[str, Any]:
            return {"require_auth": True, "keys_count": 0, "endpoint_mode_default": "shared"}

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.service.host_service.EngineHostService", _FakeSvc)

    ch = EngineHostControlChannel(
        {
            "engine_host_control_state_file": str(tmp_path / "access_control.json"),
            "engine_host_daemon_auto_bootstrap": False,
        }
    )
    out = ch.reset_hosting_access()

    assert out["status"] == "ok"
    assert out["local_helper_only"] is True
    assert out["rpc_accessible"] is False
    assert out["daemon_stop"]["status"] == "not_running"
