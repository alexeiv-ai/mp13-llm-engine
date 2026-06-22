from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import hosting.engine_host_channel as channel_module
from hosting.client_realm import FileSecretStore, write_client_profile
from hosting.engine_host_connection import CommandError
from hosting.engine_host_channel import EngineHostControlChannel


@pytest.fixture(autouse=True)
def _clear_auto_session_cache() -> None:
    channel_module._clear_auto_session_cache_for_tests()
    yield
    channel_module._clear_auto_session_cache_for_tests()


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


def test_auto_session_is_reused_across_channels_in_process() -> None:
    settings = {
        "engine_host_key_id": "mgmt-key",
        "engine_host_key_secret": "secret-1",
        "engine_host_session_scope": "control",
        "engine_host_daemon_auto_bootstrap": False,
        "engine_host_control_state_file": "C:/state/access_control.json",
    }
    first = _FakeConn()
    ch1 = EngineHostControlChannel(settings)
    ch1._get_connection = lambda: first  # type: ignore[method-assign]

    second = _FakeConn()
    ch2 = EngineHostControlChannel(settings)
    ch2._get_connection = lambda: second  # type: ignore[method-assign]

    assert ch1.discover_running() == []
    assert ch2.discover_running() == []

    assert [cmd for cmd, _payload in first.calls] == ["auth-issue-session", "discover-running"]
    assert [cmd for cmd, _payload in second.calls] == ["discover-running"]
    assert second.calls[0][1]["session_token"] == "tok-123"


def test_public_key_session_reuses_non_control_token_on_same_channel() -> None:
    class _PublicKeyConn(_FakeConn):
        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
            self.calls.append((cmd, dict(payload or {})))
            if cmd == "auth-begin-challenge":
                return {"challenge_id": "chal-1", "challenge": "sign-me"}
            if cmd == "auth-complete-challenge":
                return {
                    "status": "ok",
                    "token": "pk-token",
                    "key_id": "admin-pub",
                    "auth_method": "public_key",
                    "scope": "traffic",
                    "expires_at": 9999999999.0,
                }
            if cmd == "auth-validate-session":
                if payload and payload.get("token") == "pk-token":
                    return {
                        "valid": True,
                        "reason": "ok",
                        "key_id": "admin-pub",
                        "auth_method": "public_key",
                        "role": "admin",
                        "scope": "traffic",
                        "expires_at": 9999999999.0,
                        "ssh_bound": False,
                        "ssh_binding": {},
                    }
                return {"valid": False, "reason": "missing_or_invalid_session_token"}
            if cmd == "auth-status":
                raise CommandError("auth_failed", code="insufficient_scope")
            return {}

    fake = _PublicKeyConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    signer_calls: list[dict] = []

    def signer(challenge: dict) -> dict:
        signer_calls.append(dict(challenge))
        return {"signature_ssh": "SIG"}

    first = ch.ensure_public_key_session(key_id="admin-pub", scope="traffic", signer=signer)
    second = ch.ensure_public_key_session(key_id="admin-pub", scope="traffic", signer=signer)

    assert first == "pk-token"
    assert second == "pk-token"
    assert [cmd for cmd, _payload in fake.calls] == [
        "auth-begin-challenge",
        "auth-complete-challenge",
        "auth-validate-session",
    ]
    assert len(signer_calls) == 1


def test_public_key_session_cache_reuses_non_control_token_across_channels() -> None:
    settings = {
        "engine_host_daemon_auto_bootstrap": False,
        "engine_host_control_state_file": "C:/state/access_control.json",
    }

    class _PublicKeyConn(_FakeConn):
        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
            self.calls.append((cmd, dict(payload or {})))
            if cmd == "auth-begin-challenge":
                return {"challenge_id": "chal-1", "challenge": "sign-me"}
            if cmd == "auth-complete-challenge":
                return {
                    "status": "ok",
                    "token": "pk-token",
                    "key_id": "admin-pub",
                    "auth_method": "public_key",
                    "scope": "traffic",
                    "expires_at": 9999999999.0,
                }
            if cmd == "auth-validate-session":
                if payload and payload.get("token") == "pk-token":
                    return {
                        "valid": True,
                        "reason": "ok",
                        "key_id": "admin-pub",
                        "auth_method": "public_key",
                        "role": "admin",
                        "scope": "traffic",
                        "expires_at": 9999999999.0,
                        "ssh_bound": False,
                        "ssh_binding": {},
                    }
                return {"valid": False, "reason": "missing_or_invalid_session_token"}
            if cmd == "auth-status":
                raise CommandError("auth_failed", code="insufficient_scope")
            return {}

    first = _PublicKeyConn()
    ch1 = EngineHostControlChannel(settings)
    ch1._get_connection = lambda: first  # type: ignore[method-assign]

    second = _PublicKeyConn()
    ch2 = EngineHostControlChannel(settings)
    ch2._get_connection = lambda: second  # type: ignore[method-assign]

    signer_calls: list[dict] = []

    def signer(challenge: dict) -> str:
        signer_calls.append(dict(challenge))
        return "SIG"

    assert ch1.ensure_public_key_session(key_id="admin-pub", scope="traffic", signer=signer) == "pk-token"
    assert ch2.ensure_public_key_session(key_id="admin-pub", scope="traffic", signer=signer) == "pk-token"

    assert [cmd for cmd, _payload in first.calls] == ["auth-begin-challenge", "auth-complete-challenge"]
    assert [cmd for cmd, _payload in second.calls] == ["auth-validate-session"]
    assert len(signer_calls) == 1


def test_discover_running_auth_failure_does_not_use_subprocess_fallback() -> None:
    class _AuthFailConn(_FakeConn):
        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
            self.calls.append((cmd, dict(payload or {})))
            raise CommandError("auth_failed", code="missing_or_invalid_session_token")

    fake = _AuthFailConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch.set_session_token("expired-token")
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    def fail_subprocess(_command: str, _payload: Dict[str, Any]) -> Any:
        raise AssertionError("subprocess fallback should not be used for daemon auth failures")

    ch._invoke_subprocess = fail_subprocess  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="auth_failed"):
        ch.invoke_control_command("discover-running", {})

    assert ch.get_session_token() is None
    assert len(fake.calls) == 2


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


def test_daemon_status_keeps_reachable_when_auth_status_requires_session(monkeypatch) -> None:
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
                return "pong"
            if cmd == "auth-status":
                raise PermissionError("session_token_required")
            raise AssertionError(cmd)

        def close(self) -> None:
            return

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeSocket)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_read_local_control_snapshot",
        lambda self: {"require_auth": True, "keys_count": 1, "endpoint_mode_default": "exclusive"},
    )

    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    status = ch.get_daemon_status()

    assert status["alive"] is True
    assert status["reachable"] is True
    assert status["auth_status"] is None
    assert status["auth_status_error"] == "session_token_required"
    assert status["require_auth"] is True
    assert status["keys_count"] == 1


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
    _ = ch.workflow_python_event_subscribe(stream_id="stream-1")
    _ = ch.workflow_js_event_subscribe(stream_id="stream-js")
    assert len(fake.calls) == 3
    assert fake.calls[0][0] == "discover-running"
    assert fake.calls[0][1]["_ssh_session_binding"] == {
        "target": "user@example-host",
        "key_fingerprint": "SHA256:abc",
    }
    assert "session_token" not in fake.calls[0][1]
    assert fake.calls[1][0] == "workflow-python-event-subscribe"
    assert fake.calls[1][1]["_ssh_session_binding"] == {
        "target": "user@example-host",
        "key_fingerprint": "SHA256:abc",
    }
    assert fake.calls[2][0] == "workflow-js-event-subscribe"
    assert fake.calls[2][1]["_ssh_session_binding"] == {
        "target": "user@example-host",
        "key_fingerprint": "SHA256:abc",
    }


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


def test_host_capability_register_known_methods_helper_forwards_descriptors() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    out = ch.host_capability_session_register_known_methods(
        session_id="known-host-api",
        scope={"workflow_id": "wf-1"},
        binding={"transport": "local_ipc", "address": "client-callback"},
        allow_override=True,
    )

    assert out == {}
    assert fake.calls[0][0] == "host-capability-session-register"
    payload = fake.calls[0][1]
    method_names = [row["name"] for row in payload["methods"]]
    assert payload["session_id"] == "known-host-api"
    assert payload["scope"] == {"workflow_id": "wf-1"}
    assert payload["allow_override"] is True
    assert "fs.read_text" in method_names
    assert "fs.write_text" in method_names
    assert "http.fetch" in method_names
    assert all("provider" not in row for row in payload["methods"])


def test_host_capability_session_filtered_helpers_use_public_session_shapes() -> None:
    class FakeSessionConn(_FakeConn):
        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
            p = dict(payload or {})
            self.calls.append((cmd, p))
            if cmd == "host-capability-session-list":
                return {
                    "status": "ok",
                    "sessions": [
                        {
                            "session_id": "crm-provider",
                            "owner": "client-a",
                            "scope": {"workflow_id": "wf-1", "request_id": "req-1"},
                            "provider": {"kind": "client_session", "visibility": "workflow"},
                            "methods": [{"name": "crm.customer.lookup"}],
                        },
                        {
                            "session_id": "erp-provider",
                            "owner": "client-a",
                            "scope": {"workflow_id": "wf-2"},
                            "provider": {"kind": "client_session", "visibility": "workflow"},
                            "methods": [{"name": "erp.customer.lookup"}],
                        },
                    ],
                }
            if cmd == "host-capability-session-close":
                return {"status": "closed", "session_id": p["session_id"], "closed": True}
            return {}

    fake = FakeSessionConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    listing = ch.host_capability_session_list_filtered(workflow_id="wf-1", method="crm.customer.lookup")
    closed = ch.host_capability_session_close_filtered(workflow_id="wf-1", method="crm.customer.lookup")

    assert [row["session_id"] for row in listing["sessions"]] == ["crm-provider"]
    assert closed["count"] == 1
    assert fake.calls[-1] == ("host-capability-session-close", {"session_id": "crm-provider", "force": False})


def test_host_capability_session_upsert_closes_matching_session_before_register() -> None:
    class FakeUpsertConn(_FakeConn):
        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
            p = dict(payload or {})
            self.calls.append((cmd, p))
            if cmd == "host-capability-session-list":
                return {
                    "status": "ok",
                    "sessions": [
                        {
                            "session_id": "crm-provider",
                            "owner": "client-a",
                            "scope": {"workflow_id": "wf-1"},
                            "provider": {"kind": "client_session", "visibility": "workflow"},
                            "methods": [{"name": "crm.customer.lookup"}],
                        }
                    ],
                }
            if cmd == "host-capability-session-close":
                return {"status": "closed", "session_id": p["session_id"], "closed": True}
            if cmd == "host-capability-session-register":
                return {"status": "ok", "session": {"session_id": p["session_id"]}}
            return {}

    fake = FakeUpsertConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    out = ch.host_capability_session_upsert(
        session_id="crm-provider",
        scope={"workflow_id": "wf-1"},
        methods=[{"name": "crm.customer.lookup", "namespace": "crm", "group_path": ["CRM"]}],
        binding={"transport": "local_ipc"},
    )

    assert out["registered"]["session"]["session_id"] == "crm-provider"
    assert [cmd for cmd, _payload in fake.calls] == [
        "host-capability-session-list",
        "host-capability-session-close",
        "host-capability-session-register",
    ]


def test_host_capability_audit_list_helper_forwards_filters() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    ch.host_capability_audit_list(
        workflow_id="wf-1",
        instance_id="inst-1",
        request_id="req-1",
        provider_id="provider-1",
        method="crm.customer.lookup",
        approval_id="approval-1",
        since=1.0,
        until=2.0,
        limit=50,
        offset=5,
    )

    assert fake.calls[0] == (
        "host-capability-audit-list",
        {
            "workflow_id": "wf-1",
            "instance_id": "inst-1",
            "request_id": "req-1",
            "provider_id": "provider-1",
            "method": "crm.customer.lookup",
            "approval_id": "approval-1",
            "since": 1.0,
            "until": 2.0,
            "limit": 50,
            "offset": 5,
        },
    )


def test_host_capability_register_toolbox_helper_adds_toolbox_binding() -> None:
    class FakeToolboxConn(_FakeConn):
        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> Any:
            p = dict(payload or {})
            self.calls.append((cmd, p))
            if cmd == "toolbox-describe":
                return {
                    "status": "ok",
                    "toolbox_id": "tb-1",
                    "all_registered_tool_names": ["hello_tool"],
                    "allowed_tool_names": ["hello_tool"],
                    "advertised_tool_names": ["hello_tool"],
                    "tool_metadata": {"hello_tool": {"description": "Say hello."}},
                }
            if cmd == "host-capability-session-list":
                return {"status": "ok", "sessions": []}
            if cmd == "host-capability-session-register":
                return {"status": "ok", "session": {"session_id": p["session_id"]}}
            return {}

    fake = FakeToolboxConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]

    ch.host_capability_session_register_toolbox(
        toolbox_id="tb-1",
        session_id="tb-session",
        scope={"workflow_id": "wf-1"},
        tools_view={"allowed_tools": ["hello_tool"]},
        namespace="tools",
    )

    register_payload = fake.calls[-1][1]
    assert [cmd for cmd, _payload in fake.calls] == [
        "toolbox-describe",
        "host-capability-session-list",
        "host-capability-session-register",
    ]
    assert register_payload["provider_kind"] == "toolbox_session"
    assert register_payload["binding"] == {
        "transport": "toolbox_harness",
        "engine_id": "",
        "toolbox_id": "tb-1",
        "tools_view": {"allowed_tools": ["hello_tool"]},
    }
    assert register_payload["methods"][0]["name"] == "tools.hello_tool"


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


def test_workflow_python_channel_facade_forwards_expected_payloads() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]
    ch.set_session_token("tok-123")

    ch.workflow_python_environment_spec(
        profile="helper",
        python={"import_allowlist": ["json"]},
        sandbox_policy={"sandbox": {"enabled": True}},
    )
    ch.workflow_python_prepare_environment(
        environment_name="workflow-python-helper",
        python={"package_pins": {"requests": "2.32.3"}},
        package_id="pkg",
        workflow_id="wf",
    )
    ch.workflow_python_lock_environment(environment={"environment_key": "env-key"})
    ch.workflow_python_verify_environment(environment={"environment_key": "env-key"})
    ch.workflow_python_install_environment(environment={"environment_key": "env-key"}, allow_execution=True)
    ch.workflow_python_verify_install_receipt(environment={"environment_key": "env-key"})
    ch.ensure_workflow_python(
        profile="helper",
        environment_key="env-key",
        python={"import_allowlist": ["json"]},
        engine_id="wf-py",
        capacity=4,
    )
    ch.execute_workflow_python(
        profile="helper",
        environment_key="env-key",
        engine_id="wf-py",
        request={"request_id": "req-1"},
    )
    ch.execute_workflow_python(
        profile="node",
        environment_name="workflow-python-node",
        environment_key="env-node",
        engine_id="wf-node",
        request={"request_id": "req-node", "operation": "run"},
        capacity=2,
    )
    ch.workflow_python_resources(profile="helper", environment_key="env-key", engine_id="wf-py")
    ch.set_workflow_python_capacity(profile="helper", environment_key="env-key", engine_id="wf-py", capacity=6)
    ch.cancel_workflow_python_request(profile="helper", environment_key="env-key", engine_id="wf-py", request_id="req-1")
    ch.workflow_python_request_status(profile="helper", environment_key="env-key", engine_id="wf-py", request_id="req-1")
    ch.workflow_python_stream_open(
        profile="node",
        environment_key="env-key",
        engine_id="wf-py",
        request={"request_id": "req-1"},
        capacity=3,
    )
    ch.workflow_python_event_subscribe(stream_id="stream-1", max_items=5)
    ch.workflow_python_stream_send(stream_id="stream-1", message={"action": "cancel"})
    ch.workflow_python_stream_close(stream_id="stream-1")

    assert fake.calls == [
        (
            "workflow-python-environment-spec",
            {
                "profile": "helper",
                "environment_name": "workflow-python-helper",
                "python": {"import_allowlist": ["json"]},
                "sandbox_policy": {"sandbox": {"enabled": True}},
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-prepare-environment",
            {
                "environment_name": "workflow-python-helper",
                "python": {"package_pins": {"requests": "2.32.3"}},
                "package_id": "pkg",
                "workflow_id": "wf",
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-lock-environment",
            {
                "environment": {"environment_key": "env-key"},
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-verify-environment",
            {
                "environment": {"environment_key": "env-key"},
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-install-environment",
            {
                "environment": {"environment_key": "env-key"},
                "allow_execution": True,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-verify-install-receipt",
            {
                "environment": {"environment_key": "env-key"},
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-ensure",
            {
                "profile": "helper",
                "environment_name": "workflow-python-helper",
                "environment_key": "env-key",
                "python": {"import_allowlist": ["json"]},
                "python_executable": None,
                "capacity": 4,
                "sandbox_policy": None,
                "engine_id": "wf-py",
                "worker_profile_class": "generic",
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-execute",
            {
                "profile": "helper",
                "environment_name": "workflow-python-helper",
                "environment_key": "env-key",
                "engine_id": "wf-py",
                "request": {"request_id": "req-1"},
                "capacity": 1,
                "sandbox_policy": None,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-execute",
            {
                "profile": "node",
                "environment_name": "workflow-python-node",
                "environment_key": "env-node",
                "engine_id": "wf-node",
                "request": {"request_id": "req-node", "operation": "run"},
                "capacity": 2,
                "sandbox_policy": None,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-resources",
            {
                "profile": "helper",
                "environment_name": "workflow-python-helper",
                "environment_key": "env-key",
                "engine_id": "wf-py",
                "python": {},
                "sandbox_policy": None,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-set-capacity",
            {
                "profile": "helper",
                "environment_key": "env-key",
                "engine_id": "wf-py",
                "capacity": 6,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-cancel-request",
            {
                "profile": "helper",
                "environment_key": "env-key",
                "engine_id": "wf-py",
                "request_id": "req-1",
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-request-status",
            {
                "profile": "helper",
                "environment_key": "env-key",
                "engine_id": "wf-py",
                "request_id": "req-1",
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-stream-open",
            {
                "profile": "node",
                "environment_name": "workflow-python-node",
                "environment_key": "env-key",
                "engine_id": "wf-py",
                "request": {"request_id": "req-1"},
                "python": {},
                "sandbox_policy": None,
                "capacity": 3,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-event-subscribe",
            {
                "stream_id": "stream-1",
                "max_items": 5,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-stream-send",
            {
                "stream_id": "stream-1",
                "message": {"action": "cancel"},
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-python-stream-close",
            {
                "stream_id": "stream-1",
                "session_token": "tok-123",
            },
        ),
    ]


def test_workflow_js_channel_facade_forwards_expected_payloads() -> None:
    fake = _FakeConn()
    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    ch._get_connection = lambda: fake  # type: ignore[method-assign]
    ch.set_session_token("tok-123")

    ch.workflow_js_environment_spec(
        profile="node",
        node={"runtime_hash": "quickjs-demo"},
        javascript={"host_api": {"enabled": True}},
        sandbox_policy={"sandbox": {"enabled": True}},
    )
    ch.ensure_workflow_js(
        profile="node",
        environment_key="env-js",
        node={"runtime_hash": "quickjs-demo"},
        javascript={"host_api": {"enabled": True}},
        engine_id="wf-js",
        capacity=4,
    )
    ch.workflow_js_resources(profile="node", environment_key="env-js", engine_id="wf-js", javascript={"host_api": {"enabled": True}})
    ch.execute_workflow_js(
        profile="node",
        environment_key="env-js",
        engine_id="wf-js",
        javascript={"host_api": {"enabled": True}},
        request={
            "request_id": "req-1",
            "module_source": "exports.run = function(input, api) { return { output: { accepted: true } }; };",
            "module_sha256": "sha",
            "export_name": "run",
            "payload": {},
        },
    )
    ch.set_workflow_js_capacity(profile="node", environment_key="env-js", engine_id="wf-js", capacity=6)
    ch.cancel_workflow_js_request(profile="node", environment_key="env-js", engine_id="wf-js", request_id="req-1")
    ch.workflow_js_request_status(profile="node", environment_key="env-js", engine_id="wf-js", request_id="req-1")
    ch.workflow_js_stream_open(
        profile="node",
        environment_key="env-js",
        engine_id="wf-js",
        request={"request_id": "req-js-stream"},
        node={"runtime_hash": "quickjs-demo"},
        javascript={"host_api": {"enabled": True}},
        capacity=3,
    )
    ch.workflow_js_event_subscribe(stream_id="js-stream-1", max_items=5)
    ch.workflow_js_stream_send(stream_id="js-stream-1", message={"action": "cancel"})
    ch.workflow_js_stream_close(stream_id="js-stream-1")

    assert fake.calls == [
        (
            "workflow-js-environment-spec",
            {
                "profile": "node",
                "environment_name": "workflow-js-node",
                "node": {"runtime_hash": "quickjs-demo"},
                "javascript": {"host_api": {"enabled": True}},
                "sandbox_policy": {"sandbox": {"enabled": True}},
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-ensure",
            {
                "profile": "node",
                "environment_name": "workflow-js-node",
                "environment_key": "env-js",
                "node": {"runtime_hash": "quickjs-demo"},
                "javascript": {"host_api": {"enabled": True}},
                "capacity": 4,
                "sandbox_policy": None,
                "engine_id": "wf-js",
                "worker_profile_class": "generic",
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-resources",
            {
                "profile": "node",
                "environment_name": "workflow-js-node",
                "environment_key": "env-js",
                "engine_id": "wf-js",
                "node": {},
                "javascript": {"host_api": {"enabled": True}},
                "sandbox_policy": None,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-execute",
            {
                "profile": "node",
                "environment_name": "workflow-js-node",
                "environment_key": "env-js",
                "engine_id": "wf-js",
                "request": {
                    "request_id": "req-1",
                    "module_source": "exports.run = function(input, api) { return { output: { accepted: true } }; };",
                    "module_sha256": "sha",
                    "export_name": "run",
                    "payload": {},
                },
                "node": {},
                "javascript": {"host_api": {"enabled": True}},
                "capacity": 1,
                "sandbox_policy": None,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-set-capacity",
            {
                "profile": "node",
                "environment_key": "env-js",
                "engine_id": "wf-js",
                "capacity": 6,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-cancel-request",
            {
                "profile": "node",
                "environment_key": "env-js",
                "engine_id": "wf-js",
                "request_id": "req-1",
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-request-status",
            {
                "profile": "node",
                "environment_key": "env-js",
                "engine_id": "wf-js",
                "request_id": "req-1",
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-stream-open",
            {
                "profile": "node",
                "environment_name": "workflow-js-node",
                "environment_key": "env-js",
                "engine_id": "wf-js",
                "request": {"request_id": "req-js-stream"},
                "node": {"runtime_hash": "quickjs-demo"},
                "javascript": {"host_api": {"enabled": True}},
                "sandbox_policy": None,
                "capacity": 3,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-event-subscribe",
            {
                "stream_id": "js-stream-1",
                "max_items": 5,
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-stream-send",
            {
                "stream_id": "js-stream-1",
                "message": {"action": "cancel"},
                "session_token": "tok-123",
            },
        ),
        (
            "workflow-js-stream-close",
            {
                "stream_id": "js-stream-1",
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


def test_get_connection_uses_default_pid_file_for_local_ipc(monkeypatch) -> None:
    default_pid_file = Path("X:/tmp/default_host.pid")
    captured: Dict[str, Any] = {}

    class _FakePidFile:
        path = default_pid_file

        def __init__(self, _path: Optional[str] = None):
            return

        def get_port(self) -> int:
            return 19876

        def is_alive(self) -> bool:
            return True

    class _FakeSocket:
        def __init__(self, *, port: int, pid_file: Optional[Path] = None, timeout: float, **_kwargs: Any):
            captured["port"] = port
            captured["pid_file"] = pid_file
            captured["timeout"] = timeout

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeSocket)

    ch = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    conn = ch._get_connection()

    assert conn is not None
    assert captured["pid_file"] == default_pid_file
    assert captured["port"] == 19876


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


def test_prepare_local_unconfigured_bootstrap_rejects_legacy_backend_root_when_hosting_access_exists(tmp_path: Path) -> None:
    from hosting.service.host_service import EngineHostService

    configured = tmp_path / "hosting" / "access_control.json"
    svc = EngineHostService(control_state_file=configured)
    svc.auth_upsert_key(key_id="admin-main", key_secret="secret", role="admin")
    svc.set_control_config(
        require_auth=True,
        access_profile={"connectivity_mode": "local_only"},
        endpoint_mode_default="shared",
    )

    ch = EngineHostControlChannel(
        {
            "engine_host_control_state_file": str(tmp_path / "backend" / "engine_host_control.json"),
            "engine_host_daemon_auto_bootstrap": False,
        }
    )

    with pytest.raises(RuntimeError, match="Refusing temporary no-auth local daemon bootstrap"):
        ch._prepare_local_unconfigured_bootstrap()


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


def test_force_stop_daemon_stops_registered_workers_before_daemon_pid(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    pid_file = tmp_path / "daemon.pid"
    pid_alive = {"value": True}

    class _FakePidFile:
        path = pid_file

        def __init__(self, _path: Optional[str] = None):
            return

        def read(self) -> Dict[str, Any]:
            return {"pid": 4444, "port": 19876, "shutdown_token": "tok", "started_at": 1.0}

        def is_alive(self) -> bool:
            return bool(pid_alive["value"])

        def remove(self) -> None:
            calls.append(("remove_pid_file", None))

    class _FakeService:
        def __init__(self, **_kwargs: Any):
            return

        def discover_running(self, **_kwargs: Any) -> list[Dict[str, Any]]:
            return [{"engine_id": "worker-a"}]

        def shutdown(self, engine_id: str, *, timeout_seconds: float = 2.0) -> Dict[str, Any]:
            calls.append(("shutdown_worker", engine_id))
            return {"status": "stopped", "engine_id": engine_id, "alive": False}

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.service.host_service.EngineHostService", _FakeService)
    monkeypatch.setattr(EngineHostControlChannel, "stop_daemon", lambda self: {"status": "error", "error": "unreachable"})
    monkeypatch.setattr(EngineHostControlChannel, "_list_local_engine_worker_processes", lambda self: [])

    def _fake_kill(pid: int, sig: int) -> None:
        calls.append(("kill", (pid, sig)))
        pid_alive["value"] = False

    monkeypatch.setattr("hosting.engine_host_channel.os.kill", _fake_kill)
    monkeypatch.setattr("hosting.engine_host_channel.time.sleep", lambda _sec: None)

    ch = EngineHostControlChannel({"engine_host_daemon_pid_file": str(pid_file)})
    out = ch.force_stop_daemon(stop_workers=True)

    assert out["status"] == "ok"
    assert out["worker_shutdown"]["attempted"] == 1
    assert calls[0] == ("shutdown_worker", "worker-a")
    assert calls[1][0] == "kill"
    assert ("remove_pid_file", None) in calls


def test_stop_daemon_sends_shutdown_reason(monkeypatch, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}
    pid_file = tmp_path / "daemon.pid"

    class _FakePidFile:
        path = pid_file

        def __init__(self, _path: Optional[str] = None):
            return

        def read(self) -> Dict[str, Any]:
            return {"pid": 4444, "port": 19876, "shutdown_token": "tok", "started_at": 1.0}

    class _FakeSocket:
        def __init__(self, *, port: int, pid_file: Path, timeout: float, max_reconnect_attempts: int):
            captured["port"] = port
            captured["pid_file"] = pid_file

        def invoke(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> str:
            captured["cmd"] = cmd
            captured["payload"] = dict(payload or {})
            return "shutting_down"

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.engine_host_connection.LocalSocketConnection", _FakeSocket)

    ch = EngineHostControlChannel({"engine_host_daemon_pid_file": str(pid_file)})
    out = ch.stop_daemon(reason="operator_test", requested_by="unit-test")

    assert out == {"status": "shutdown_sent"}
    assert captured["cmd"] == "__shutdown__"
    assert captured["payload"]["shutdown_token"] == "tok"
    assert captured["payload"]["shutdown_reason"] == "operator_test"
    assert captured["payload"]["requested_by"] == "unit-test"


def test_force_stop_daemon_kills_orphan_engine_worker_processes(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    alive = {7777: True}

    class _FakePidFile:
        path = tmp_path / "daemon.pid"

        def __init__(self, _path: Optional[str] = None):
            return

        def read(self) -> Dict[str, Any]:
            return {}

        def is_alive(self) -> bool:
            return False

        def remove(self) -> None:
            calls.append(("remove_pid_file", None))

    class _FakeService:
        def __init__(self, **_kwargs: Any):
            return

        def discover_running(self, **_kwargs: Any) -> list[Dict[str, Any]]:
            return []

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.service.host_service.EngineHostService", _FakeService)
    monkeypatch.setattr(EngineHostControlChannel, "stop_daemon", lambda self: {"status": "not_running"})
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_list_local_engine_worker_processes",
        lambda self: [{"pid": 7777, "parent_pid": 1, "command": "python -m hosting.engine_worker_ipc"}],
    )
    monkeypatch.setattr("hosting.engine_host_channel.pid_alive", lambda pid: bool(alive.get(pid, False)))

    def _fake_kill(pid: int, sig: int) -> None:
        calls.append(("kill", (pid, sig)))
        alive[pid] = False

    monkeypatch.setattr("hosting.engine_host_channel.os.kill", _fake_kill)
    monkeypatch.setattr("hosting.engine_host_channel.time.sleep", lambda _sec: None)

    ch = EngineHostControlChannel({"engine_host_daemon_pid_file": str(tmp_path / "daemon.pid")})
    out = ch.force_stop_daemon(stop_workers=True)

    assert out["worker_shutdown"]["orphan_attempted"] == 1
    assert out["worker_shutdown"]["orphan_stopped"] == 1
    assert calls[0][0] == "kill"


def test_bootstrap_blocks_unreachable_exclusive_daemon_by_default(monkeypatch, tmp_path: Path) -> None:
    state = {"alive": True}
    captured: Dict[str, Any] = {}

    class _FakePidFile:
        path = tmp_path / "daemon.pid"

        def __init__(self, _path: Optional[str] = None):
            return

        def is_alive(self) -> bool:
            return bool(state["alive"])

        def read(self) -> Dict[str, Any]:
            return {"pid": 4444, "port": 19876, "started_at": 1.0, "shutdown_token": "tok"}

    def _fake_start_daemon_background(**kwargs: Any) -> Dict[str, Any]:
        captured["started"] = True
        captured["pid_file"] = kwargs.get("pid_file")
        return {"pid": 5555, "port": 19876}

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.daemon.start_daemon_background", _fake_start_daemon_background)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "get_daemon_status",
        lambda self: {"pid_alive": bool(state["alive"]), "reachable": False, "alive": False},
    )
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_read_local_control_snapshot",
        lambda self: {"endpoint_mode_default": "exclusive", "lifecycle_profile": "detached_user_process"},
    )

    def _fake_force_stop(self, **_kwargs: Any) -> Dict[str, Any]:
        state["alive"] = False
        return {"status": "ok"}

    monkeypatch.setattr(EngineHostControlChannel, "force_stop_daemon", _fake_force_stop)

    ch = EngineHostControlChannel({"engine_host_daemon_pid_file": str(tmp_path / "daemon.pid")})
    out = ch.bootstrap_daemon(wait_ready_seconds=1.0)

    assert out["blocked_by_unreachable_pid"] is True
    assert out["auto_recovery_attempted"] is False
    assert out["auto_recovery_allowed"] is True
    assert out["auto_recovery_requires_explicit_request"] is True
    assert captured.get("started") is None


def test_bootstrap_recovers_unreachable_exclusive_daemon_when_explicit(monkeypatch, tmp_path: Path) -> None:
    state = {"alive": True}
    captured: Dict[str, Any] = {}

    class _FakePidFile:
        path = tmp_path / "daemon.pid"

        def __init__(self, _path: Optional[str] = None):
            return

        def is_alive(self) -> bool:
            return bool(state["alive"])

        def read(self) -> Dict[str, Any]:
            return {"pid": 4444, "port": 19876, "started_at": 1.0, "shutdown_token": "tok"}

    def _fake_start_daemon_background(**kwargs: Any) -> Dict[str, Any]:
        captured["started"] = True
        captured["pid_file"] = kwargs.get("pid_file")
        return {"pid": 5555, "port": 19876}

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.daemon.start_daemon_background", _fake_start_daemon_background)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "get_daemon_status",
        lambda self: {"pid_alive": bool(state["alive"]), "reachable": False, "alive": False},
    )
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_read_local_control_snapshot",
        lambda self: {"endpoint_mode_default": "exclusive", "lifecycle_profile": "detached_user_process"},
    )

    def _fake_force_stop(self, **_kwargs: Any) -> Dict[str, Any]:
        state["alive"] = False
        return {"status": "ok"}

    monkeypatch.setattr(EngineHostControlChannel, "force_stop_daemon", _fake_force_stop)

    ch = EngineHostControlChannel({"engine_host_daemon_pid_file": str(tmp_path / "daemon.pid")})
    out = ch.bootstrap_daemon(wait_ready_seconds=1.0, recover_unreachable=True)

    assert out["auto_recovery_attempted"] is True
    assert captured["started"] is True
    assert out["pid"] == 5555


def test_bootstrap_blocks_unreachable_shared_detached_daemon(monkeypatch, tmp_path: Path) -> None:
    class _FakePidFile:
        path = tmp_path / "daemon.pid"

        def __init__(self, _path: Optional[str] = None):
            return

        def is_alive(self) -> bool:
            return True

        def read(self) -> Dict[str, Any]:
            return {"pid": 4444, "port": 19876, "started_at": 1.0, "shutdown_token": "tok"}

    def _fake_start_daemon_background(**_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("shared detached daemon should require explicit force")

    monkeypatch.setattr("hosting.daemon.DaemonPidFile", _FakePidFile)
    monkeypatch.setattr("hosting.daemon.start_daemon_background", _fake_start_daemon_background)
    monkeypatch.setattr(
        EngineHostControlChannel,
        "get_daemon_status",
        lambda self: {"pid_alive": True, "reachable": False, "alive": False},
    )
    monkeypatch.setattr(
        EngineHostControlChannel,
        "_read_local_control_snapshot",
        lambda self: {"endpoint_mode_default": "shared", "lifecycle_profile": "detached_user_process"},
    )

    ch = EngineHostControlChannel({"engine_host_daemon_pid_file": str(tmp_path / "daemon.pid")})
    out = ch.bootstrap_daemon(wait_ready_seconds=1.0)

    assert out["blocked_by_unreachable_pid"] is True
    assert out["auto_recovery_attempted"] is False
    assert out["auto_recovery_allowed"] is False
    assert out["auto_recovery_requires_explicit_request"] is False
    assert out["auto_recovery_policy"]["reason"] == "shared_or_detached_daemon_requires_explicit_force"
