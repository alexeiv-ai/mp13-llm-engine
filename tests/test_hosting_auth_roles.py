from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from hosting.client_realm import FileSecretStore
from hosting.service.host_service import EngineHostService
from hosting.daemon import EngineHostDaemon


def _svc(tmpdir: str) -> EngineHostService:
    root = Path(tmpdir)
    return EngineHostService(
        engines_state_file=root / "engines.json",
        control_state_file=root / "control.json",
    )


@contextmanager
def _workspace_tmpdir():
    root_base = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT", str(Path.cwd().parent / ".mp13_pytest"))).resolve()
    root = (root_base / "test_hosting_auth_roles" / str(uuid.uuid4())).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield str(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_diagnostic_user_denied_spawn_with_insufficient_role() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        svc.auth_upsert_key(
            key_id="diag",
            key_secret="diag-secret",
            role="diagnostic_user",
            auth_method="shared_secret",
        )
        session = svc.auth_issue_session(
            key_id="diag",
            key_secret="diag-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token
        svc.authorize_command("host-metrics", {"session_token": token})
        with pytest.raises(PermissionError, match="insufficient_role"):
            svc.authorize_command("spawn", {"session_token": token})


def test_diagnostic_user_toolbox_authority_is_observe_only() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="diag",
            key_secret="diag-secret",
            role="diagnostic_user",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="diag",
            key_secret="diag-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token

        for cmd in [
            "toolbox-describe",
            "toolbox-gate",
            "toolbox-references",
            "toolbox-consistency",
            "toolbox-review-snapshot",
            "toolbox-environment-list",
            "workflow-python-environment-spec",
            "workflow-python-verify-environment",
            "workflow-python-verify-install-receipt",
            "workflow-python-resources",
            "workflow-js-helper-resources",
            "sandbox-fs-list",
            "sandbox-fs-read-text",
            "sandbox-fs-stat",
        ]:
            svc.authorize_command(cmd, {"session_token": token})

        for cmd in [
            "toolbox-execute",
            "toolbox-cancel",
            "toolbox-gc",
            "toolbox-repair",
            "toolbox-reconcile",
            "toolbox-register-auto",
            "toolbox-unregister-auto",
            "toolbox-register-intrinsics",
            "toolbox-unregister-intrinsics",
            "toolbox-register-manual",
            "toolbox-unregister-manual",
            "toolbox-environment-upsert",
            "toolbox-environment-clone",
            "toolbox-environment-resolve",
            "toolbox-environment-apply",
            "toolbox-environment-realize",
            "toolbox-environment-sync",
            "toolbox-environment-prepare-install",
            "toolbox-environment-lock-install",
            "toolbox-environment-resolve-install-lock",
            "toolbox-environment-verify-install-lock",
            "toolbox-environment-verify-install-receipt",
            "toolbox-environment-execute-install",
            "sandbox-fs-write-text",
            "sandbox-fs-mkdir",
            "sandbox-http-fetch",
            "workflow-python-prepare-environment",
            "workflow-python-lock-environment",
            "workflow-python-install-environment",
            "workflow-python-ensure",
            "workflow-python-execute",
            "workflow-python-set-capacity",
            "workflow-python-cancel-request",
            "workflow-js-helper-set-capacity",
            "workflow-js-helper-cancel-request",
        ]:
            with pytest.raises(PermissionError, match="insufficient_role"):
                svc.authorize_command(cmd, {"session_token": token})


def test_worker_user_can_manage_toolbox_sandbox_authority() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="worker",
            key_secret="worker-secret",
            role="worker_user",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="worker",
            key_secret="worker-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token

        for cmd in [
            "toolbox-register-auto",
            "toolbox-register-manual",
            "toolbox-repair",
            "toolbox-reconcile",
            "toolbox-environment-upsert",
            "toolbox-environment-resolve",
            "toolbox-environment-apply",
            "toolbox-environment-realize",
            "toolbox-environment-prepare-install",
            "toolbox-environment-lock-install",
            "toolbox-environment-execute-install",
            "workflow-python-environment-spec",
            "workflow-python-prepare-environment",
            "workflow-python-lock-environment",
            "workflow-python-verify-environment",
            "workflow-python-install-environment",
            "workflow-python-verify-install-receipt",
            "workflow-python-ensure",
            "workflow-python-execute",
            "workflow-python-resources",
            "workflow-python-set-capacity",
            "workflow-python-cancel-request",
            "workflow-js-helper-set-capacity",
            "workflow-js-helper-cancel-request",
        ]:
            svc.authorize_command(cmd, {"session_token": token})


def test_worker_user_denied_raw_spawn_but_allowed_workflow_js_helper_spawn() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="worker",
            key_secret="worker-secret",
            role="worker_user",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="worker",
            key_secret="worker-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token
        with pytest.raises(PermissionError, match="insufficient_role"):
            svc.authorize_command("spawn", {"session_token": token})
        svc.authorize_command("spawn-workflow-js-helper", {"session_token": token})


def test_config_editor_allowed_raw_spawn() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="editor",
            key_secret="editor-secret",
            role="config_editor",
            auth_method="shared_secret",
            allowed_engines=["*"],
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="editor",
            key_secret="editor-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token
        svc.authorize_command("spawn", {"session_token": token})


def test_require_auth_false_rejected_for_non_local_profile() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        with pytest.raises(
            PermissionError,
            match="require_auth_false_only_supported_for_local_only_connectivity",
        ):
            svc.set_control_config(
                require_auth=False,
                access_profile={"connectivity_mode": "truly_remote"},
            )


def test_require_auth_false_rejected_when_profile_drifts_without_require_auth_field() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.set_control_config(
            require_auth=False,
            access_profile={"connectivity_mode": "local_only"},
        )
        with pytest.raises(
            PermissionError,
            match="require_auth_false_only_supported_for_local_only_connectivity",
        ):
            svc.set_control_config(
                access_profile={"connectivity_mode": "ssh_tunnel_only"},
            )


def test_auth_status_reports_local_private_key_custody_metadata_only() -> None:
    with _workspace_tmpdir() as td:
        root = Path(td)
        svc = EngineHostService(
            engines_state_file=root / "engines.json",
            control_state_file=root / "hosting" / "access_control.json",
        )
        svc.auth_upsert_key(
            key_id="admin-main",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "ssh_tunnel_only"},
        )
        store = FileSecretStore(root / "hosting_client" / "default", realm="default")
        store.put_secret(
            tag="rbac_private_key",
            payload="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----\n",
            secret_id="rbac-admin-main-private",
            metadata={"private_key_protection": "openssh_passphrase"},
        )
        keys_file = root / "hosting" / "keyring" / "keys.json"
        keys_file.parent.mkdir(parents=True, exist_ok=True)
        keys_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "keys": {
                        "admin-main": {
                            "key_id": "admin-main",
                            "role": "admin",
                            "auth_method": "public_key",
                            "public_key": "ssh-ed25519 AAAATEST admin-main",
                            "key_origin": "generated",
                            "public_key_source": "generated",
                            "private_key_storage": "client_realm_secret",
                            "private_key_secret_id": "rbac-admin-main-private",
                            "private_key_secret_realm": "default",
                            "private_key_protection": "openssh_passphrase",
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        status = svc.auth_status()
        custody = list(status.get("local_private_key_custody") or [])
        assert len(custody) == 1
        row = dict(custody[0] or {})
        assert row["key_id"] == "admin-main"
        assert row["private_key_storage"] == "client_realm_secret"
        assert row["private_key_secret_id"] == "rbac-admin-main-private"
        assert row["private_key_secret_exists"] is True
        assert row["private_key_protection"] == "openssh_passphrase"
        assert "private_key" not in row


def test_authorize_command_rejects_unsafe_no_auth_runtime_config() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        # Simulate unsafe manual control-state edit: auth disabled but remote profile.
        control = svc._read_control()  # noqa: SLF001
        cfg = dict(control.get("control_config") or {})
        cfg["require_auth"] = False
        cfg["access_profile"] = {"connectivity_mode": "truly_remote"}
        control["control_config"] = cfg
        svc._write_control(control)  # noqa: SLF001

        with pytest.raises(
            PermissionError,
            match="require_auth_false_only_supported_for_local_only_connectivity",
        ):
            svc.authorize_command("discover-running", {})


def test_require_auth_false_rejects_session_and_challenge_issue_paths() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.set_control_config(
            require_auth=False,
            access_profile={"connectivity_mode": "local_only"},
        )
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        with pytest.raises(PermissionError, match="require_auth_disabled_disallows_session_commands"):
            svc.auth_issue_session(
                key_id="admin",
                key_secret="admin-secret",
                scope="control",
            )
        svc.auth_revoke_key("admin")
        svc.auth_upsert_key(
            key_id="admin-pub",
            role="admin",
            auth_method="public_key",
            public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeFakeFakeFakeFakeFakeFakeFakeFake admin-pub",
        )
        with pytest.raises(PermissionError, match="require_auth_disabled_disallows_session_commands"):
            svc.auth_begin_challenge(
                key_id="admin-pub",
                scope="control",
                ttl_seconds=120,
            )


def test_zero_key_bootstrap_allowed_for_local_only_require_auth_true() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        svc.authorize_command("auth-upsert-key", {})
        svc.authorize_command("auth-status", {})


def test_zero_key_bootstrap_rejected_for_remote_capable_require_auth_true() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "ssh_tunnel_only"},
        )
        with pytest.raises(PermissionError, match="zero_key_bootstrap_local_only"):
            svc.authorize_command("auth-upsert-key", {})
        with pytest.raises(PermissionError, match="zero_key_bootstrap_local_only"):
            svc.authorize_command("auth-status", {})


def test_require_auth_false_forces_exclusive_endpoint_mode_default() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        out = svc.set_control_config(
            require_auth=False,
            access_profile={"connectivity_mode": "local_only"},
            endpoint_mode_default="shared",
        )
        assert bool(out.get("require_auth")) is False
        assert str(out.get("endpoint_mode_default") or "") == "exclusive"
        read_back = svc.get_control_config()
        assert str(read_back.get("endpoint_mode_default") or "") == "exclusive"


def test_runtime_policy_assertion_rejects_no_auth_shared_endpoint_mode() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.set_control_config(
            require_auth=False,
            access_profile={"connectivity_mode": "local_only"},
            endpoint_mode_default="exclusive",
        )
        control = svc._read_control()  # noqa: SLF001
        cfg = dict(control.get("control_config") or {})
        cfg["endpoint_mode_default"] = "shared"
        control["control_config"] = cfg
        svc._write_control(control)  # noqa: SLF001
        with pytest.raises(
            PermissionError,
            match="require_auth_false_requires_exclusive_endpoint_mode",
        ):
            svc.assert_runtime_policy_safe()



def test_legacy_role_name_is_rejected_on_key_upsert() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        with pytest.raises(ValueError, match="role must be one of"):
            svc.auth_upsert_key(
                key_id="legacy",
                key_secret="legacy-secret",
                role="management",
                auth_method="shared_secret",
            )


def test_runtime_policy_assertion_rejects_unsafe_unauth_profile() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        control = svc._read_control()  # noqa: SLF001
        cfg = dict(control.get("control_config") or {})
        cfg["require_auth"] = False
        cfg["access_profile"] = {"connectivity_mode": "truly_remote"}
        control["control_config"] = cfg
        svc._write_control(control)  # noqa: SLF001
        with pytest.raises(
            PermissionError,
            match="require_auth_false_only_supported_for_local_only_connectivity",
        ):
            svc.assert_runtime_policy_safe()


def test_claim_engine_uses_endpoint_mode_default_when_exclusive_omitted() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.set_control_config(endpoint_mode_default="exclusive")
        out = svc.claim_engine("worker1", backend_id="backend:a", exclusive=None)
        assert str(out.get("mode") or "") == "exclusive"
        assert str(out.get("exclusive_owner") or "") == "backend:a"


def test_lifecycle_profile_defaults_for_service_managed() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        out = svc.set_control_config(lifecycle_profile="service_managed")
        assert str(out.get("lifecycle_profile") or "") == "service_managed"
        policy = dict(out.get("lifecycle_policy") or {})
        assert str(policy.get("on_terminal_disconnect") or "") == "keep_daemon_running"
        assert bool(policy.get("terminal_control_enabled")) is False
        assert bool(policy.get("owner_disconnect_shutdown")) is False
        effective = svc.get_lifecycle_policy_effective()
        assert str(effective.get("profile") or "") == "service_managed"
        eff = dict(effective.get("effective") or {})
        assert bool(eff.get("daemon_survives_terminal_disconnect")) is True


def test_invalid_lifecycle_profile_is_rejected() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        with pytest.raises(ValueError, match="lifecycle_profile must be one of"):
            svc.set_control_config(lifecycle_profile="unknown_profile")


def test_lifecycle_policy_override_is_persisted() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        out = svc.set_control_config(
            lifecycle_profile="detached_user_process",
            lifecycle_policy={
                "on_terminal_disconnect": "stop_daemon",
                "terminal_control_enabled": False,
                "owner_disconnect_shutdown": True,
            },
        )
        policy = dict(out.get("lifecycle_policy") or {})
        assert str(policy.get("on_terminal_disconnect") or "") == "stop_daemon"
        assert bool(policy.get("terminal_control_enabled")) is False
        assert bool(policy.get("owner_disconnect_shutdown")) is True


def test_daemon_runtime_endpoint_override_applies_to_claims() -> None:
    with _workspace_tmpdir() as td:
        root = Path(td)
        daemon = EngineHostDaemon(
            port=0,
            engines_state_file=root / "engines.json",
            control_state_file=root / "control.json",
        )
        daemon.svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        daemon.svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
            endpoint_mode_default="shared",
        )
        session = daemon.svc.auth_issue_session(
            key_id="admin",
            key_secret="admin-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token

        set_req = json.dumps(
            {
                "seq": 1,
                "cmd": "set-endpoint-mode-override",
                "payload": {"mode": "exclusive", "session_token": token},
            }
        )
        set_res = asyncio.run(daemon._dispatch(set_req, peer_host="127.0.0.1"))  # noqa: SLF001
        assert bool(set_res.get("ok")) is True
        set_out = dict(set_res.get("result") or {})
        assert str(set_out.get("runtime_override") or "") == "exclusive"
        assert str(set_out.get("effective") or "") == "exclusive"

        claim_req = json.dumps(
            {
                "seq": 2,
                "cmd": "claim-endpoint",
                "payload": {
                    "backend_id": "backend:a",
                    "session_token": token,
                },
            }
        )
        claim_res = asyncio.run(daemon._dispatch(claim_req, peer_host="127.0.0.1"))  # noqa: SLF001
        assert bool(claim_res.get("ok")) is True
        claim_out = dict(claim_res.get("result") or {})
        assert str(claim_out.get("mode") or "") == "exclusive"


def test_model_user_cannot_override_model_in_connect_from_config() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="model",
            key_secret="model-secret",
            role="model_user",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="model",
            key_secret="model-secret",
            scope="traffic",
            engine_ids=["*"],
        )
        token = str(session.get("token") or "")
        assert token
        with pytest.raises(PermissionError, match="insufficient_role"):
            svc.authorize_command(
                "connect-from-config",
                {
                    "session_token": token,
                    "config_path": "default",
                    "model_path": "/tmp/override-model.gguf",
                },
            )


def test_model_user_with_model_control_can_override_model_in_connect_from_config() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="model-control",
            key_secret="model-control-secret",
            role="model_user_with_model_control",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="model-control",
            key_secret="model-control-secret",
            scope="traffic",
            engine_ids=["*"],
        )
        token = str(session.get("token") or "")
        assert token
        svc.authorize_command(
            "connect-from-config",
            {
                "session_token": token,
                "config_path": "default",
                "model_path": "/tmp/override-model.gguf",
            },
        )


def test_model_user_cannot_connect_generic_worker_profile() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        original_merge = svc._merge_default_and_selected_config  # noqa: SLF001
        svc._merge_default_and_selected_config = lambda _config_path: {  # type: ignore[method-assign]  # noqa: SLF001
            "worker_kind": "generic",
            "worker_command": ["python", "-c", "print('generic')"],
        }
        svc.auth_upsert_key(
            key_id="model",
            key_secret="model-secret",
            role="model_user",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="model",
            key_secret="model-secret",
            scope="traffic",
            engine_ids=["*"],
        )
        token = str(session.get("token") or "")
        assert token
        try:
            with pytest.raises(PermissionError, match="insufficient_role"):
                svc.authorize_command(
                    "connect-from-config",
                    {
                        "session_token": token,
                        "config_path": "generic_worker",
                    },
                )
        finally:
            svc._merge_default_and_selected_config = original_merge  # type: ignore[method-assign]  # noqa: SLF001


def test_worker_user_cannot_connect_generic_worker_profile() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        original_merge = svc._merge_default_and_selected_config  # noqa: SLF001
        svc._merge_default_and_selected_config = lambda _config_path: {  # type: ignore[method-assign]  # noqa: SLF001
            "worker_kind": "generic",
            "worker_command": ["python", "-c", "print('generic')"],
        }
        svc.auth_upsert_key(
            key_id="worker",
            key_secret="worker-secret",
            role="worker_user",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="worker",
            key_secret="worker-secret",
            scope="traffic",
            engine_ids=["*"],
        )
        token = str(session.get("token") or "")
        assert token
        try:
            with pytest.raises(PermissionError, match="insufficient_role"):
                svc.authorize_command(
                    "connect-from-config",
                    {
                        "session_token": token,
                        "config_path": "generic_worker",
                    },
                )
        finally:
            svc._merge_default_and_selected_config = original_merge  # type: ignore[method-assign]  # noqa: SLF001


def test_config_editor_can_connect_generic_worker_profile() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        original_merge = svc._merge_default_and_selected_config  # noqa: SLF001
        svc._merge_default_and_selected_config = lambda _config_path: {  # type: ignore[method-assign]  # noqa: SLF001
            "worker_kind": "generic",
            "worker_command": ["python", "-c", "print('generic')"],
        }
        svc.auth_upsert_key(
            key_id="editor",
            key_secret="editor-secret",
            role="config_editor",
            auth_method="shared_secret",
            allowed_engines=["*"],
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="editor",
            key_secret="editor-secret",
            scope="traffic",
        )
        token = str(session.get("token") or "")
        assert token
        try:
            svc.authorize_command(
                "connect-from-config",
                {
                    "session_token": token,
                    "config_path": "generic_worker",
                },
            )
        finally:
            svc._merge_default_and_selected_config = original_merge  # type: ignore[method-assign]  # noqa: SLF001


def test_model_user_denied_proxy_to_generic_registered_engine() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.register_spawned(
            engine_id="generic1",
            pid=12345,
            command=["python", "-m", "hosting.engine_worker_ipc"],
            worker_profile_class="generic",
        )
        svc.auth_upsert_key(
            key_id="model",
            key_secret="model-secret",
            role="model_user",
            auth_method="shared_secret",
            allowed_engines=["generic1"],
        )
        svc.set_control_config(require_auth=True, access_profile={"connectivity_mode": "local_only"})
        session = svc.auth_issue_session(
            key_id="model",
            key_secret="model-secret",
            scope="traffic",
            engine_ids=["generic1"],
        )
        token = str(session.get("token") or "")
        assert token
        with pytest.raises(PermissionError, match="insufficient_role"):
            svc.authorize_command(
                "proxy-request",
                {"session_token": token, "engine_id": "generic1"},
            )


def test_model_user_allowed_proxy_to_workflow_js_helper_engine() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.register_spawned(
            engine_id="workflow-js-helper",
            pid=12345,
            command=["python", "-m", "hosting.workflow_js_helper_ipc"],
            worker_profile_class="generic",
            executor_kind="workflow_js_helper",
        )
        svc.auth_upsert_key(
            key_id="model",
            key_secret="model-secret",
            role="model_user",
            auth_method="shared_secret",
            allowed_engines=["workflow-js-helper"],
        )
        svc.set_control_config(require_auth=True, access_profile={"connectivity_mode": "local_only"})
        session = svc.auth_issue_session(
            key_id="model",
            key_secret="model-secret",
            scope="traffic",
            engine_ids=["workflow-js-helper"],
        )
        token = str(session.get("token") or "")
        assert token
        svc.authorize_command(
            "proxy-rpc-call",
            {"session_token": token, "engine_id": "workflow-js-helper"},
        )


def test_worker_user_allowed_proxy_to_generic_registered_engine() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.register_spawned(
            engine_id="generic1",
            pid=12345,
            command=["python", "-m", "hosting.engine_worker_ipc"],
            worker_profile_class="generic",
        )
        svc.auth_upsert_key(
            key_id="worker",
            key_secret="worker-secret",
            role="worker_user",
            auth_method="shared_secret",
            allowed_engines=["generic1"],
        )
        svc.set_control_config(require_auth=True, access_profile={"connectivity_mode": "local_only"})
        session = svc.auth_issue_session(
            key_id="worker",
            key_secret="worker-secret",
            scope="traffic",
            engine_ids=["generic1"],
        )
        token = str(session.get("token") or "")
        assert token
        svc.authorize_command(
            "proxy-request",
            {"session_token": token, "engine_id": "generic1"},
        )


def test_transport_role_rejects_shared_secret_key_upsert() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        with pytest.raises(ValueError, match="transport role requires public_key auth_method"):
            svc.auth_upsert_key(
                key_id="transport1",
                key_secret="secret",
                role="transport",
                auth_method="shared_secret",
            )


def test_transport_role_public_key_cannot_issue_session_or_challenge() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        svc.auth_upsert_key(
            key_id="transport1",
            role="transport",
            auth_method="public_key",
            public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeFakeFakeFakeFakeFakeFakeFakeFake transport1",
        )
        with pytest.raises(PermissionError, match="auth_method_requires_challenge_flow"):
            svc.auth_issue_session(
                key_id="transport1",
                key_secret="unused",
                scope="control",
            )
        with pytest.raises(PermissionError, match="transport_role_cannot_issue_session"):
            svc.auth_begin_challenge(
                key_id="transport1",
                scope="traffic",
                ttl_seconds=120,
            )


def test_remote_connectivity_disallows_shared_secret_session_issue() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "truly_remote"},
        )
        with pytest.raises(PermissionError, match="shared_secret_bootstrap_not_supported_for_remote_connectivity"):
            svc.auth_issue_session(
                key_id="admin",
                key_secret="admin-secret",
                scope="control",
            )
        with pytest.raises(PermissionError, match="shared_secret_bootstrap_not_supported_for_remote_connectivity"):
            svc.auth_issue_session(
                key_id="admin",
                key_secret="admin-secret",
                scope="control",
                ssh_binding={"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
            )


def test_remote_connectivity_requires_ssh_binding_for_public_key_challenge_begin() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin-pub",
            role="admin",
            auth_method="public_key",
            public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeFakeFakeFakeFakeFakeFakeFakeFake admin-pub",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "ssh_tunnel_only"},
        )
        with pytest.raises(PermissionError, match="ssh_binding_required_for_remote_connectivity"):
            svc.auth_begin_challenge(
                key_id="admin-pub",
                scope="control",
                ttl_seconds=120,
            )
        out = svc.auth_begin_challenge(
            key_id="admin-pub",
            scope="control",
            ttl_seconds=120,
            ssh_binding={"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
        )
        assert str(out.get("challenge_id") or "")


def test_remote_connectivity_command_denied_when_ssh_binding_not_presented() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="admin",
            key_secret="admin-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "ssh_tunnel_only"},
        )
        with pytest.raises(PermissionError, match="ssh_binding_required_for_remote_connectivity"):
            svc.authorize_command("discover-running", {"session_token": token})


def test_remote_connectivity_denies_legacy_unbound_session() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="admin",
            key_secret="admin-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token
        # Simulate profile flip after legacy session issuance.
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "truly_remote"},
        )
        with pytest.raises(PermissionError, match="ssh_binding_required_for_remote_connectivity"):
            svc.authorize_command(
                "discover-running",
                {
                    "session_token": token,
                    "_ssh_session_binding": {"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
                },
            )


def test_auth_validate_session_reports_binding_and_identity() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="admin",
            key_secret="admin-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        out = svc.auth_validate_session(token=token, scope="control", expected_key_id="admin")
        assert out["valid"] is True
        assert out["reason"] == "ok"
        assert out["key_id"] == "admin"
        assert out["actor_key_id"] == "admin"
        assert out["role"] == "admin"
        assert out["scope"] == "control"
        assert out["auth_method"] == "shared_secret"
        assert out["ssh_bound"] is False

        mismatch = svc.auth_validate_session(token=token, scope="control", expected_key_id="other")
        assert mismatch["valid"] is False
        assert mismatch["reason"] == "key_id_mismatch"


def test_auth_renew_session_extends_valid_session() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="admin",
            key_secret="admin-secret",
            scope="control",
            ttl_seconds=60,
        )
        token = str(session.get("token") or "")
        assert token

        before = svc.auth_validate_session(token=token, scope="control")
        renewed = svc.auth_renew_session(token=token, scope="control", ttl_seconds=3600)
        after = svc.auth_validate_session(token=token, scope="control")

        assert renewed["status"] == "ok"
        assert renewed["ttl_seconds"] == 3600
        assert after["valid"] is True
        assert float(after["expires_at"]) > float(before["expires_at"])
        svc.authorize_command("auth-renew-session", {"session_token": token})


def test_auth_validate_session_checks_remote_ssh_binding() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="admin",
            key_secret="admin-secret",
            scope="control",
            ssh_binding={"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
        )
        token = str(session.get("token") or "")
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "ssh_tunnel_only"},
        )

        denied = svc.auth_validate_session(token=token, scope="control", presented_ssh_binding={})
        assert denied["valid"] is False
        assert denied["reason"] in {"ssh_binding_required", "ssh_binding_required_for_remote_connectivity"}

        unchecked = svc.auth_validate_session(token=token, scope="control", check_ssh_binding=False)
        assert unchecked["valid"] is True
        assert unchecked["ssh_bound"] is True

        accepted = svc.auth_validate_session(
            token=token,
            scope="control",
            presented_ssh_binding={"target": "user@example-host", "key_fingerprint": "SHA256:abc"},
        )
        assert accepted["valid"] is True
        assert accepted["ssh_bound"] is True


def test_auth_capabilities_advertise_session_and_audit_contracts() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        caps = dict(svc.get_control_config().get("capabilities") or {})
        assert caps["auth_session_validate"] is True
        assert caps["auth_session_adopt"] is True
        assert caps["auth_session_list"] is True
        assert caps["auth_audit_list"] is True
        status_caps = dict(svc.auth_status().get("capabilities") or {})
        assert status_caps["auth_session_validate"] is True
        assert status_caps["auth_audit_list"] is True


def test_config_editor_cannot_authorize_admin_key_revocation_commands() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="editor",
            key_secret="editor-secret",
            role="config_editor",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="editor",
            key_secret="editor-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token
        with pytest.raises(PermissionError, match="insufficient_role"):
            svc.authorize_command("auth-revoke-key", {"session_token": token, "key_id": "victim"})
        with pytest.raises(PermissionError, match="insufficient_role"):
            svc.authorize_command("auth-revoke-session", {"session_token": token, "token": "abc"})


def test_admin_can_authorize_key_revocation_commands() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="admin",
            key_secret="admin-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token
        svc.authorize_command("auth-revoke-key", {"session_token": token, "key_id": "victim"})
        svc.authorize_command("auth-revoke-session", {"session_token": token, "token": "abc"})


def test_config_editor_cannot_authorize_auth_audit_list() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="editor",
            key_secret="editor-secret",
            role="config_editor",
            auth_method="shared_secret",
        )
        svc.set_control_config(
            require_auth=True,
            access_profile={"connectivity_mode": "local_only"},
        )
        session = svc.auth_issue_session(
            key_id="editor",
            key_secret="editor-secret",
            scope="control",
        )
        token = str(session.get("token") or "")
        assert token
        with pytest.raises(PermissionError, match="insufficient_role"):
            svc.authorize_command("auth-audit-list", {"session_token": token})


def test_admin_can_list_auth_audit_events_with_filters() -> None:
    with _workspace_tmpdir() as td:
        svc = _svc(td)
        svc.auth_upsert_key(
            key_id="admin",
            key_secret="admin-secret",
            role="admin",
            auth_method="shared_secret",
        )
        svc.auth_upsert_key(
            key_id="worker",
            key_secret="worker-secret",
            role="worker_user",
            auth_method="shared_secret",
        )
        svc.auth_revoke_key("worker")
        out_all = svc.auth_list_audit_events(limit=50, offset=0)
        rows_all = list(out_all.get("events") or [])
        assert rows_all
        assert any(str(r.get("event_type") or "") == "auth_revoke_key" for r in rows_all)

        out_filtered = svc.auth_list_audit_events(event_type="auth_revoke_key", limit=50, offset=0)
        rows = list(out_filtered.get("events") or [])
        assert rows
        assert all(str(r.get("event_type") or "") == "auth_revoke_key" for r in rows)
