from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from hosting.hosting_config_cli import (
    UserCancelled,
    _bool_prompt,
    _suggest_auto_configuration,
    run_client_keys,
    run_doctor,
    run_setup,
    run_transport_admin_setup,
    run_transport_bootstrap,
)


TEST_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeFakeFakeFakeFakeFakeFakeFakeFake admin-main"


def _args(
    *,
    default_config_dir: Path,
    control_state_file: Path,
    mode: str = "local_only",
    endpoint_mode: str = "exclusive",
    lifecycle_profile: str = "detached_user_process",
    require_auth: bool | None = True,
    key_source: str = "import",
    admin_key_id: str = "admin-main",
    admin_public_key: str = TEST_PUBLIC_KEY,
    doctor: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        default_config_dir=str(default_config_dir),
        control_state_file=str(control_state_file),
        doctor=doctor,
        interactive=False,
        mode=mode,
        endpoint_mode=endpoint_mode,
        lifecycle_profile=lifecycle_profile,
        require_auth=require_auth,
        key_source=key_source,
        admin_key_id=admin_key_id,
        admin_public_key_file="",
        admin_public_key=admin_public_key,
        generated_key_passphrase="",
        export_private_key=False,
        export_private_key_path="",
        client_realm="default",
        client_realm_root="",
        client_list_keys=False,
        client_generate_key=False,
        client_import_key=False,
        client_export_key=False,
        client_key_id="",
        client_key_tag="rbac_private_key",
        client_private_key_file="",
        client_public_key_file="",
        client_public_key_inline="",
        client_export_key_path="",
        transport_harden_ssh=False,
        transport_export_bootstrap=False,
        transport_import_bootstrap=False,
        transport_validate_profile=False,
        transport_provision_ssh_artifacts=False,
        transport_install_authorized_key=False,
        bootstrap_bundle_file="",
        transport_target="",
        transport_key_id="",
        transport_public_key_file="",
        transport_public_key_inline="",
        transport_private_key_file="",
        transport_private_key_inline="",
        ssh_known_hosts_file="",
        ssh_known_hosts_line="",
        transport_profile_name="",
        control_ssh_fingerprint="",
        overwrite_profile=False,
        bootstrap_password="",
        client_secret_password="",
        validation_no_ssh_run=False,
        validation_ssh_bin="ssh",
        validation_remote_command="exit 0",
        validation_timeout_seconds=15.0,
        ssh_config_alias="",
        overwrite_ssh_config=False,
        ssh_authorized_keys_file="",
        ssh_authorized_key_command="python -m hosting.engine_host_cli --relay-wrapper",
        ssh_authorized_key_unrestricted=False,
        admin_capability="no_admin_available",
        transport_admin_setup=False,
        admin_setup_execute=False,
        admin_setup_enable_ssh_service=True,
        admin_setup_enable_firewall=False,
        admin_setup_enable_user_linger=False,
        admin_setup_target_user="",
    )


@contextmanager
def _workspace_tmpdir():
    root_base = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT", str(Path.cwd().parent / ".mp13_pytest"))).resolve()
    root = (root_base / "test_hosting_config" / str(uuid.uuid4())).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_relay_wrapper_returns_structured_policy_error() -> None:
    with _workspace_tmpdir() as root:
        control_state = root / "control.json"
        env = dict(os.environ)
        src_path = str(Path.cwd() / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hosting.engine_host_cli",
                "--relay-wrapper",
                "--control-state-file",
                str(control_state),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            assert proc.stdin is not None
            assert proc.stdout is not None
            proc.stdin.write(b'{"seq": 7, "cmd": "__ping__", "payload": {}}\n')
            proc.stdin.flush()
            line = proc.stdout.readline().decode("utf-8")
            resp = json.loads(line)
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        assert resp["seq"] == 7
        assert resp["ok"] is False
        assert resp["error_code"] == "relay_autostart_requires_remote_connectivity"
        assert resp["error_details"]["connectivity_mode"] == "local_only"


def test_setup_import_writes_expected_hosting_files() -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        args = _args(default_config_dir=root, control_state_file=control)
        out = run_setup(args)

        assert str(out.get("status") or "") == "ok"
        assert str(out.get("connectivity_mode") or "") == "local_only"
        assert str(out.get("endpoint_mode_default") or "") == "exclusive"
        assert str(out.get("lifecycle_profile") or "") == "detached_user_process"
        assert bool(out.get("require_auth")) is True

        hosting_root = root / "hosting"
        assert (hosting_root / "access_control.json").exists()
        assert (hosting_root / "keyring" / "keys.json").exists()
        assert (hosting_root / "bootstrap" / "client_key_map.json").exists()
        assert (hosting_root / "bootstrap" / "bootstrap_state.json").exists()
        assert control.exists()

        keyring = json.loads((hosting_root / "keyring" / "keys.json").read_text(encoding="utf-8"))
        keys = dict(keyring.get("keys") or {})
        admin = dict(keys.get("admin-main") or {})
        assert str(admin.get("role") or "") == "admin"
        access = json.loads((hosting_root / "access_control.json").read_text(encoding="utf-8"))
        assert str(access.get("control_config", {}).get("lifecycle_profile") or "") == "detached_user_process"
        control_payload = json.loads(control.read_text(encoding="utf-8"))
        cfg = dict(control_payload.get("control_config") or {})
        assert str(cfg.get("lifecycle_profile") or "") == "detached_user_process"


def test_prompt_q_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "q")
    with pytest.raises(UserCancelled):
        _bool_prompt("Continue?", True)


def test_prompt_ctrl_c_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_keyboard_interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_keyboard_interrupt)
    with pytest.raises(UserCancelled):
        _bool_prompt("Continue?", True)


def test_setup_persists_requested_lifecycle_profile() -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        args = _args(
            default_config_dir=root,
            control_state_file=control,
            lifecycle_profile="service_managed",
        )
        out = run_setup(args)
        assert str(out.get("status") or "") == "ok"
        assert str(out.get("lifecycle_profile") or "") == "service_managed"
        control_payload = json.loads(control.read_text(encoding="utf-8"))
        cfg = dict(control_payload.get("control_config") or {})
        assert str(cfg.get("lifecycle_profile") or "") == "service_managed"


def test_setup_generate_consolidates_importable_key_material(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"

        monkeypatch.setattr(
            "hosting.hosting_config_cli._generate_keypair",
            lambda **_kwargs: (
                "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
                "ssh-ed25519 AAAATESTGENERATED admin-main",
            ),
        )

        args = _args(
            default_config_dir=root,
            control_state_file=control,
            key_source="generate",
        )
        out = run_setup(args)
        assert str(out.get("status") or "") == "ok"

        keyring = json.loads((root / "hosting" / "keyring" / "keys.json").read_text(encoding="utf-8"))
        admin = dict(dict(keyring.get("keys") or {}).get("admin-main") or {})
        assert str(admin.get("public_key") or "") == "ssh-ed25519 AAAATESTGENERATED admin-main"
        assert "private_key_openssh" not in admin
        assert str(admin.get("private_key_storage") or "") == "client_realm_secret"
        assert str(admin.get("private_key_secret_id") or "") == "rbac-admin-main-private"
        assert "created_at" not in admin
        assert "updated_at" not in admin
        secret_file = root / "hosting_client" / "default" / "secrets" / "rbac-admin-main-private.json"
        assert secret_file.exists()
        secret_payload = json.loads(secret_file.read_text(encoding="utf-8"))
        assert str(secret_payload.get("tag") or "") == "rbac_private_key"
        assert "BEGIN OPENSSH PRIVATE KEY" in str(secret_payload.get("payload") or "")


def test_setup_generate_can_store_encrypted_client_realm_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"

        monkeypatch.setattr(
            "hosting.hosting_config_cli._generate_keypair",
            lambda **_kwargs: (
                "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
                "ssh-ed25519 AAAATESTGENERATED admin-main",
            ),
        )

        args = _args(
            default_config_dir=root,
            control_state_file=control,
            key_source="generate",
        )
        args.client_secret_password = "secret-pw"
        out = run_setup(args)
        assert str(out.get("status") or "") == "ok"
        assert str(out.get("admin_private_key_secret_encryption") or "") == "password_v1"

        secret_file = root / "hosting_client" / "default" / "secrets" / "rbac-admin-main-private.json"
        secret_payload = json.loads(secret_file.read_text(encoding="utf-8"))
        assert str(secret_payload.get("encryption") or "") == "password_v1"


def test_setup_generate_with_export_keeps_private_key_out_of_keyring_and_client_realm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        export_path = root / "exported" / "admin-main"

        monkeypatch.setattr(
            "hosting.hosting_config_cli._generate_keypair",
            lambda **_kwargs: (
                "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
                "ssh-ed25519 AAAATESTGENERATED admin-main",
            ),
        )

        args = _args(
            default_config_dir=root,
            control_state_file=control,
            key_source="generate",
        )
        args.export_private_key = True
        args.export_private_key_path = str(export_path)
        out = run_setup(args)
        assert str(out.get("status") or "") == "ok"

        keyring = json.loads((root / "hosting" / "keyring" / "keys.json").read_text(encoding="utf-8"))
        admin = dict(dict(keyring.get("keys") or {}).get("admin-main") or {})
        assert "private_key_openssh" not in admin
        assert str(admin.get("private_key_storage") or "") == "exported_file"
        assert not str(admin.get("private_key_secret_id") or "").strip()
        assert export_path.exists()
        assert not (root / "hosting_client" / "default" / "secrets" / "rbac-admin-main-private.json").exists()


def test_run_client_keys_generate_list_and_export(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_tmpdir() as root:
        args = _args(default_config_dir=root, control_state_file=root / "hosting" / "access_control.json")
        monkeypatch.setattr(
            "hosting.hosting_config_cli._generate_keypair",
            lambda **_kwargs: (
                "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKECLIENT\n-----END OPENSSH PRIVATE KEY-----",
                "ssh-ed25519 AAAACLIENTPUB client-admin",
            ),
        )
        args.client_generate_key = True
        args.client_key_id = "client-admin"
        args.client_secret_password = "secret-pw"
        generated = run_client_keys(args)
        assert generated["status"] == "ok"
        assert generated["secret_encryption"] == "password_v1"

        args.client_generate_key = False
        args.client_list_keys = True
        listed = run_client_keys(args)
        assert "client-admin" in dict(listed.get("keys") or {})

        args.client_list_keys = False
        args.client_export_key = True
        args.client_export_key_path = str(root / "exported" / "client-admin")
        exported = run_client_keys(args)
        assert exported["status"] == "ok"
        assert Path(exported["export_path"]).exists()
        assert "FAKECLIENT" in Path(exported["export_path"]).read_text(encoding="utf-8")


def test_run_client_keys_import_can_derive_public_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_tmpdir() as root:
        private_key = root / "client.key"
        private_key.write_text(
            "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKEIMPORT\n-----END OPENSSH PRIVATE KEY-----\n",
            encoding="utf-8",
        )
        args = _args(default_config_dir=root, control_state_file=root / "hosting" / "access_control.json")
        args.client_import_key = True
        args.client_key_id = "imported-client"
        args.client_private_key_file = str(private_key)
        monkeypatch.setattr(
            "hosting.hosting_config_cli._derive_public_key_from_private",
            lambda _text: "ssh-ed25519 AAAADERIVED imported-client",
        )
        imported = run_client_keys(args)
        assert imported["status"] == "ok"
        assert imported["public_key"] == "ssh-ed25519 AAAADERIVED imported-client"
        keys_payload = json.loads((root / "hosting_client" / "default" / "keyring" / "keys.json").read_text(encoding="utf-8"))
        assert "imported-client" in dict(keys_payload.get("keys") or {})


def test_run_transport_bootstrap_export_and_import() -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        bundle_path = root / "transport_bundle.json"
        args.transport_export_bootstrap = True
        args.bootstrap_bundle_file = str(bundle_path)
        args.transport_target = "user@example"
        args.transport_key_id = "transport-key"
        args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport"
        args.transport_private_key_inline = "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----"
        args.ssh_known_hosts_line = "example ssh-ed25519 AAAATESTHOSTKEY"
        args.transport_profile_name = "demo"
        args.control_ssh_fingerprint = "SHA256:abc"

        exported = run_transport_bootstrap(args)
        assert exported["status"] == "ok"
        assert bundle_path.exists()

        args.transport_export_bootstrap = False
        args.transport_import_bootstrap = True
        imported = run_transport_bootstrap(args)
        assert imported["status"] == "ok"
        assert imported["profile_name"] == "demo"
        assert (root / "hosting_client" / "default" / "profiles" / "demo.json").exists()
        assert (root / "hosting_client" / "default" / "known_hosts" / "demo.known_hosts").exists()


def test_run_transport_bootstrap_validate_profile_without_ssh_probe() -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        bundle_path = root / "transport_bundle.json"
        args.transport_export_bootstrap = True
        args.bootstrap_bundle_file = str(bundle_path)
        args.transport_target = "user@example"
        args.transport_key_id = "transport-key"
        args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport"
        args.transport_private_key_inline = "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----"
        args.ssh_known_hosts_line = "example ssh-ed25519 AAAATESTHOSTKEY"
        args.transport_profile_name = "demo"
        run_transport_bootstrap(args)

        args.transport_export_bootstrap = False
        args.transport_import_bootstrap = True
        run_transport_bootstrap(args)

        args.transport_import_bootstrap = False
        args.transport_validate_profile = True
        args.validation_no_ssh_run = True
        validated = run_transport_bootstrap(args)
        assert validated["status"] == "ok"
        assert validated["action"] == "transport_validate_profile"
        assert validated["profile_name"] == "demo"
        assert validated["ssh_probe_ran"] is False


def test_run_transport_bootstrap_provisions_ssh_artifacts() -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        bundle_path = root / "transport_bundle.json"
        args.transport_export_bootstrap = True
        args.bootstrap_bundle_file = str(bundle_path)
        args.transport_target = "user@example"
        args.transport_key_id = "transport-key"
        args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport"
        args.transport_private_key_inline = "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----"
        args.ssh_known_hosts_line = "example ssh-ed25519 AAAATESTHOSTKEY"
        args.transport_profile_name = "demo"
        run_transport_bootstrap(args)

        args.transport_export_bootstrap = False
        args.transport_import_bootstrap = True
        run_transport_bootstrap(args)

        args.transport_import_bootstrap = False
        args.transport_provision_ssh_artifacts = True
        args.ssh_config_alias = "demo-host"
        provisioned = run_transport_bootstrap(args)
        assert provisioned["status"] == "ok"
        assert provisioned["action"] == "transport_provision_ssh_artifacts"
        config_path = Path(str(provisioned["ssh_config_file"]))
        identity_path = Path(str(provisioned["identity_file"]))
        assert config_path.exists()
        assert identity_path.exists()
        config_text = config_path.read_text(encoding="utf-8")
        assert "Host demo-host" in config_text
        assert "StrictHostKeyChecking yes" in config_text
        assert str(provisioned["ssh_command"]).endswith(" demo-host")


def test_run_transport_bootstrap_installs_authorized_key() -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        auth_file = root / ".ssh" / "authorized_keys"
        args.transport_install_authorized_key = True
        args.transport_key_id = "transport-key"
        args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport-key"
        args.ssh_authorized_keys_file = str(auth_file)
        result = run_transport_bootstrap(args)
        assert result["status"] == "ok"
        assert result["action"] == "transport_install_authorized_key"
        text = auth_file.read_text(encoding="utf-8")
        assert "ssh-ed25519 AAAATESTPUB transport-key" in text
        assert 'command="python -m hosting.engine_host_cli --relay-wrapper"' in text
        assert "no-pty,no-agent-forwarding,no-X11-forwarding,no-port-forwarding" in text
        assert "BEGIN mp13-hosting-transport transport-key" in text
        keys = json.loads((root / "hosting" / "keyring" / "keys.json").read_text(encoding="utf-8"))
        transport_key = keys["keys"]["transport-key"]
        assert transport_key["role"] == "transport"
        assert transport_key["public_key"] == "ssh-ed25519 AAAATESTPUB transport-key"


def test_run_transport_harden_ssh_composes_strict_profile_and_server_key() -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        auth_file = root / ".ssh" / "authorized_keys"
        args.transport_harden_ssh = True
        args.transport_target = "user@example"
        args.transport_key_id = "transport-key"
        args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport-key"
        args.transport_private_key_inline = "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----"
        args.ssh_known_hosts_line = "example ssh-ed25519 AAAATESTHOSTKEY"
        args.transport_profile_name = "demo"
        args.ssh_authorized_keys_file = str(auth_file)
        args.validation_no_ssh_run = True

        result = run_transport_bootstrap(args)
        rerun = run_transport_bootstrap(args)

        assert result["status"] == "ok"
        assert result["action"] == "transport_harden_ssh"
        assert rerun["status"] == "ok"
        assert result["validation_status"] == "ok"
        assert result["ssh_probe_ran"] is False
        auth_text = auth_file.read_text(encoding="utf-8")
        assert 'command="python -m hosting.engine_host_cli --relay-wrapper"' in auth_text
        assert "no-pty,no-agent-forwarding,no-X11-forwarding,no-port-forwarding" in auth_text
        ssh_config = Path(str(result["ssh_config_file"])).read_text(encoding="utf-8")
        assert "StrictHostKeyChecking yes" in ssh_config
        assert "IdentitiesOnly yes" in ssh_config
        keys = json.loads((root / "hosting" / "keyring" / "keys.json").read_text(encoding="utf-8"))
        assert keys["keys"]["transport-key"]["role"] == "transport"

        doctor_args = _args(default_config_dir=root, control_state_file=root / "hosting" / "access_control.json", doctor=True)
        doctor_args.transport_key_id = "transport-key"
        doctor_args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport-key"
        doctor_args.ssh_authorized_keys_file = str(auth_file)
        doctor = run_doctor(doctor_args)
        checks = {str(row["check"]): row for row in doctor["checks"]}
        assert checks["transport_authorized_key_present"]["ok"] is True
        assert checks["transport_authorized_key_hardened"]["ok"] is True
        assert checks["transport_rbac_registered"]["ok"] is True
        assert checks["transport_rbac_matches_ssh"]["ok"] is True


def test_remote_auto_configuration_includes_admin_capability_followup() -> None:
    suggestion = _suggest_auto_configuration(
        {
            "consumer": "ssh_relay",
            "lifecycle": "reconnect_shared",
            "access": "single_admin",
            "credentials": "ssh_keys",
            "admin_capability": "admin_available_interactive",
        }
    )

    assert suggestion["mode"] == "ssh_tunnel_only"
    assert suggestion["admin_capability"] == "admin_available_interactive"
    followups = "\n".join(str(item) for item in suggestion["followups"])
    assert "Run SSH transport hardening" in followups
    assert "explicit elevated steps" in followups


def test_transport_admin_setup_dry_run_generates_platform_script() -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        args.transport_admin_setup = True
        args.admin_setup_enable_firewall = True
        args.admin_setup_enable_user_linger = True
        args.admin_setup_target_user = "demo-user"

        result = run_transport_admin_setup(args)

        assert result["status"] == "dry_run"
        assert result["action"] == "transport_admin_setup"
        assert result["execute"] is False
        script = str(result["script"])
        if result["platform"] == "windows":
            assert "Start-Service sshd" in script
            assert "New-NetFirewallRule" in script
        else:
            assert "systemctl enable --now" in script
            assert "demo-user" in script


def test_interactive_admin_setup_followup_runs_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from hosting.hosting_config_cli import _interactive_admin_setup_followup

    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        monkeypatch.setattr("hosting.hosting_config_cli._prompt_menu", lambda *a, **k: "generate")
        result = _interactive_admin_setup_followup(
            args,
            {
                "mode": "ssh_tunnel_only",
                "admin_capability": "admin_available_interactive",
            },
        )

        assert result is not None
        assert result["status"] == "dry_run"
        assert result["action"] == "transport_admin_setup"
        assert result["user_linger"] is True


def test_run_transport_bootstrap_validate_profile_runs_strict_ssh_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        bundle_path = root / "transport_bundle.json"
        args.transport_export_bootstrap = True
        args.bootstrap_bundle_file = str(bundle_path)
        args.transport_target = "user@example"
        args.transport_key_id = "transport-key"
        args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport"
        args.transport_private_key_inline = "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----"
        args.ssh_known_hosts_line = "example ssh-ed25519 AAAATESTHOSTKEY"
        args.transport_profile_name = "demo"
        run_transport_bootstrap(args)

        args.transport_export_bootstrap = False
        args.transport_import_bootstrap = True
        run_transport_bootstrap(args)

        captured: dict[str, object] = {}

        def _fake_run(cmd: list[str], **kwargs: object):
            captured["cmd"] = list(cmd)
            captured["kwargs"] = dict(kwargs)

            class _Completed:
                returncode = 0
                stdout = "ok\n"
                stderr = ""

            return _Completed()

        monkeypatch.setattr("hosting.transport_bootstrap.subprocess.run", _fake_run)

        args.transport_import_bootstrap = False
        args.transport_validate_profile = True
        args.validation_no_ssh_run = False
        args.validation_remote_command = "exit 0"
        validated = run_transport_bootstrap(args)
        assert validated["status"] == "ok"
        assert validated["ssh_probe_ran"] is True
        assert validated["ssh_probe_returncode"] == 0
        cmd = list(captured["cmd"] or [])
        assert cmd[:2] == ["ssh", "-i"]
        assert "StrictHostKeyChecking=yes" in cmd
        assert any(str(part).startswith("UserKnownHostsFile=") for part in cmd)
        assert "user@example" in cmd


def test_run_transport_bootstrap_import_encrypted_bundle_to_encrypted_client_secret() -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        bundle_path = root / "transport_bundle.json"
        args.transport_export_bootstrap = True
        args.bootstrap_bundle_file = str(bundle_path)
        args.transport_target = "user@example"
        args.transport_key_id = "transport-key"
        args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport"
        args.transport_private_key_inline = "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----"
        args.ssh_known_hosts_line = "example ssh-ed25519 AAAATESTHOSTKEY"
        args.transport_profile_name = "demo"
        args.bootstrap_password = "bundle-pw"
        exported = run_transport_bootstrap(args)
        assert exported["status"] == "ok"

        args.transport_export_bootstrap = False
        args.transport_import_bootstrap = True
        args.client_secret_password = "secret-pw"
        imported = run_transport_bootstrap(args)
        assert imported["status"] == "ok"
        assert imported["secret_encryption"] == "password_v1"

        args.transport_import_bootstrap = False
        args.transport_validate_profile = True
        args.validation_no_ssh_run = True
        validated = run_transport_bootstrap(args)
        assert validated["status"] == "ok"


def test_setup_rejects_unsafe_no_require_auth_profile() -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        args = _args(
            default_config_dir=root,
            control_state_file=control,
            mode="ssh_tunnel_only",
            endpoint_mode="shared",
            require_auth=False,
        )
        with pytest.raises(
            ValueError,
            match="require_auth=false is only allowed for local_only connectivity with exclusive endpoint mode",
        ):
            run_setup(args)


def test_setup_migrates_legacy_key_file_to_migrated_suffix() -> None:
    with _workspace_tmpdir() as root:
        legacy = root / "backend" / "host_auth_keys.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text('{"legacy":"keys"}', encoding="utf-8")

        control = root / "hosting" / "access_control.json"
        args = _args(default_config_dir=root, control_state_file=control)
        out = run_setup(args)
        assert str(out.get("status") or "") == "ok"
        migration = dict(out.get("legacy_migration") or {})
        assert int(migration.get("migrated_count") or 0) >= 1

        moved = Path(str(legacy) + ".migrated")
        assert moved.exists()
        assert not legacy.exists()

        migrations_file = root / "hosting" / "keyring" / "migrations.json"
        payload = json.loads(migrations_file.read_text(encoding="utf-8"))
        rows = list(payload.get("migrations") or [])
        assert any(str(row.get("source") or "") == str(legacy.resolve()) for row in rows)


def test_doctor_reports_ok_after_valid_setup() -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        setup_args = _args(default_config_dir=root, control_state_file=control)
        run_setup(setup_args)

        doctor_args = _args(default_config_dir=root, control_state_file=control, doctor=True)
        out = run_doctor(doctor_args)
        assert str(out.get("status") or "") == "ok"
        assert int(out.get("issues_count") or 0) == 0
        checks = list(out.get("checks") or [])
        keygen_probe = [c for c in checks if str(c.get("check") or "") == "ssh_keygen_host_path_probe"]
        assert keygen_probe


def test_doctor_flags_plaintext_admin_client_secret_non_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        monkeypatch.setattr(
            "hosting.hosting_config_cli._generate_keypair",
            lambda **_kwargs: (
                "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
                "ssh-ed25519 AAAATESTGENERATED admin-main",
            ),
        )
        setup_args = _args(default_config_dir=root, control_state_file=control, key_source="generate")
        run_setup(setup_args)

        doctor_args = _args(default_config_dir=root, control_state_file=control, doctor=True)
        out = run_doctor(doctor_args)
        checks = list(out.get("checks") or [])
        encrypted = [c for c in checks if str(c.get("check") or "") == "admin_client_secret_encrypted"]
        assert encrypted
        assert bool(encrypted[0].get("ok")) is False
        assert str(encrypted[0].get("details", {}).get("encryption") or "") == "none"


def test_doctor_accepts_encrypted_admin_client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        monkeypatch.setattr(
            "hosting.hosting_config_cli._generate_keypair",
            lambda **_kwargs: (
                "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
                "ssh-ed25519 AAAATESTGENERATED admin-main",
            ),
        )
        setup_args = _args(default_config_dir=root, control_state_file=control, key_source="generate")
        setup_args.client_secret_password = "secret-pw"
        run_setup(setup_args)

        doctor_args = _args(default_config_dir=root, control_state_file=control, doctor=True)
        out = run_doctor(doctor_args)
        checks = list(out.get("checks") or [])
        encrypted = [c for c in checks if str(c.get("check") or "") == "admin_client_secret_encrypted"]
        assert encrypted
        assert bool(encrypted[0].get("ok")) is True
        assert str(encrypted[0].get("details", {}).get("encryption") or "") == "password_v1"


def test_doctor_flags_broken_client_transport_profile_integrity() -> None:
    with _workspace_tmpdir() as root:
        args = _args(
            default_config_dir=root,
            control_state_file=root / "hosting" / "access_control.json",
        )
        bundle_path = root / "transport_bundle.json"
        args.transport_export_bootstrap = True
        args.bootstrap_bundle_file = str(bundle_path)
        args.transport_target = "user@example"
        args.transport_key_id = "transport-key"
        args.transport_public_key_inline = "ssh-ed25519 AAAATESTPUB transport"
        args.transport_private_key_inline = "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----"
        args.ssh_known_hosts_line = "example ssh-ed25519 AAAATESTHOSTKEY"
        args.transport_profile_name = "demo"
        run_transport_bootstrap(args)

        args.transport_export_bootstrap = False
        args.transport_import_bootstrap = True
        imported = run_transport_bootstrap(args)
        Path(str(imported["known_hosts_file"])).unlink()

        doctor_args = _args(default_config_dir=root, control_state_file=root / "hosting" / "access_control.json", doctor=True)
        out = run_doctor(doctor_args)
        assert str(out.get("status") or "") == "issues_found"
        checks = list(out.get("checks") or [])
        integrity = [c for c in checks if str(c.get("check") or "") == "client_transport_profiles_integrity"]
        assert integrity
        assert bool(integrity[0].get("ok")) is False
        assert "root_cause" in dict(integrity[0].get("details") or {})
        assert "recommendation" in dict(integrity[0].get("details") or {})


def test_doctor_flags_unsafe_runtime_policy() -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        setup_args = _args(default_config_dir=root, control_state_file=control)
        run_setup(setup_args)

        payload = json.loads(control.read_text(encoding="utf-8"))
        cfg = dict(payload.get("control_config") or {})
        cfg["require_auth"] = False
        cfg["access_profile"] = {"connectivity_mode": "truly_remote"}
        payload["control_config"] = cfg
        control.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        doctor_args = _args(default_config_dir=root, control_state_file=control, doctor=True)
        out = run_doctor(doctor_args)
        assert str(out.get("status") or "") == "issues_found"
        assert int(out.get("issues_count") or 0) >= 1
        checks = list(out.get("checks") or [])
        runtime = [c for c in checks if str(c.get("check") or "") == "runtime_policy_safe"]
        assert runtime
        assert bool(runtime[0].get("ok")) is False
        assert "root_cause" in dict(runtime[0].get("details") or {})
        assert "recommendation" in dict(runtime[0].get("details") or {})


def test_doctor_flags_remote_require_auth_with_zero_keys() -> None:
    with _workspace_tmpdir() as root:
        control = root / "hosting" / "access_control.json"
        setup_args = _args(default_config_dir=root, control_state_file=control)
        run_setup(setup_args)

        keys_file = root / "hosting" / "keyring" / "keys.json"
        keys_payload = json.loads(keys_file.read_text(encoding="utf-8"))
        keys_payload["keys"] = {}
        keys_file.write_text(json.dumps(keys_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        payload = json.loads(control.read_text(encoding="utf-8"))
        cfg = dict(payload.get("control_config") or {})
        cfg["require_auth"] = True
        cfg["access_profile"] = {"connectivity_mode": "ssh_tunnel_only"}
        payload["control_config"] = cfg
        control.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        doctor_args = _args(default_config_dir=root, control_state_file=control, doctor=True)
        out = run_doctor(doctor_args)
        assert str(out.get("status") or "") == "issues_found"
        checks = list(out.get("checks") or [])
        bootstrap = [c for c in checks if str(c.get("check") or "") == "zero_key_remote_bootstrap_policy"]
        assert bootstrap
        assert bool(bootstrap[0].get("ok")) is False
