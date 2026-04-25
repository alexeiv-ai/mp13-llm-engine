from __future__ import annotations

import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from hosting.client_realm_api import delete_client_realm_key, export_client_realm_key, import_client_realm_key
from hosting.client_realm import (
    CLIENT_REALM_ROOT_SUBDIR,
    FileSecretStore,
    append_client_audit_event,
    client_realm_layout,
    ensure_client_realm_dirs,
    get_default_client_realm_root,
    list_client_profiles,
    list_client_audit_events,
    managed_key_path,
    materialize_secret_file,
    read_client_key_metadata,
    read_client_profile,
    read_client_access,
    resolve_client_profile_control_settings,
    write_client_profile,
    write_client_access,
)
from hosting.transport_bootstrap import (
    TRANSPORT_BOOTSTRAP_KIND,
    import_transport_bootstrap_bundle,
    install_transport_authorized_key,
    make_transport_bootstrap_bundle,
    read_transport_bootstrap_bundle,
    write_transport_bootstrap_bundle,
)


@contextmanager
def _workspace_tmpdir():
    root_base = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT", str(Path.cwd().parent / ".mp13_pytest"))).resolve()
    root = (root_base / "test_hosting_client_realm" / str(uuid.uuid4())).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_default_client_realm_root_uses_dedicated_subdir() -> None:
    with _workspace_tmpdir() as root:
        path = get_default_client_realm_root(default_config_dir=root, realm="demo")
        assert path == (root / CLIENT_REALM_ROOT_SUBDIR / "demo").resolve()


def test_ensure_client_realm_dirs_creates_expected_layout() -> None:
    with _workspace_tmpdir() as root:
        realm_root = (root / "client-realm").resolve()
        layout = ensure_client_realm_dirs(realm_root)
        expected = client_realm_layout(realm_root)
        assert layout == expected
        for key in ("root", "keyring", "secrets", "managed_keys", "known_hosts", "audit", "profiles"):
            assert layout[key].exists()
            assert layout[key].is_dir()


def test_file_secret_store_round_trip_and_listing() -> None:
    with _workspace_tmpdir() as root:
        store = FileSecretStore(root / "realm", realm="client-a")
        first = store.put_secret(
            tag="transport_private_key",
            payload="secret-1",
            secret_id="transport-1",
            metadata={"target": "user@example"},
        )
        second = store.put_secret(
            tag="session_token",
            payload="secret-2",
            secret_id="session-1",
        )

        assert first.encryption == "none"
        assert store.get_secret_payload("transport-1") == "secret-1"
        listed = store.list_records()
        assert [row.secret_id for row in listed] == ["session-1", "transport-1"]
        tagged = store.list_records(tag="transport_private_key")
        assert [row.secret_id for row in tagged] == ["transport-1"]

        raw = json.loads((store.layout["secrets"] / "transport-1.json").read_text(encoding="utf-8"))
        assert raw["metadata"]["target"] == "user@example"
        assert store.delete_secret(second.secret_id) is True
        assert store.delete_secret(second.secret_id) is False


def test_client_realm_api_exports_to_result_or_known_realm_folder() -> None:
    with _workspace_tmpdir() as root:
        realm_root = root / "client-realm"
        imported = import_client_realm_key(
            {
                "client_realm_root": realm_root,
                "key_id": "client-admin",
                "private_key_text": "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKECLIENT\n-----END OPENSSH PRIVATE KEY-----",
                "public_key_text": "ssh-ed25519 AAAACLIENT client-admin",
            }
        )
        assert imported["key_id"] == "client-admin"

        result_export = export_client_realm_key({"client_realm_root": realm_root, "key_id": "client-admin"})
        assert result_export["returned_in_result"] is True
        assert "FAKECLIENT" in str(result_export["private_key"])

        file_export = export_client_realm_key(
            {
                "client_realm_root": realm_root,
                "key_id": "client-admin",
                "export_path": realm_root / "managed_keys" / "client-admin.key",
            }
        )
        assert Path(str(file_export["export_path"])).exists()

        with pytest.raises(ValueError, match="client realm"):
            export_client_realm_key(
                {
                    "client_realm_root": realm_root,
                    "key_id": "client-admin",
                    "export_path": root / "exported" / "client-admin.key",
                }
            )


def test_client_realm_api_delete_key_removes_secret_and_metadata() -> None:
    with _workspace_tmpdir() as root:
        realm_root = root / "client-realm"
        imported = import_client_realm_key(
            {
                "client_realm_root": realm_root,
                "key_id": "delete-me",
                "private_key_text": "-----BEGIN OPENSSH PRIVATE KEY-----\nFAKEDELETE\n-----END OPENSSH PRIVATE KEY-----",
                "public_key_text": "ssh-ed25519 AAAADELETE delete-me",
            }
        )
        secret_path = Path(str(imported["secret_path"]))
        assert secret_path.exists()

        deleted = delete_client_realm_key({"client_realm_root": realm_root, "key_id": "delete-me"})
        assert deleted["status"] == "ok"
        assert deleted["deleted_secret"] is True
        assert not secret_path.exists()
        assert "delete-me" not in dict(read_client_key_metadata(realm_root).get("keys") or {})
        assert Path(str(deleted["audit_path"])).exists()


def test_file_secret_store_rejects_custom_password_encryption() -> None:
    with _workspace_tmpdir() as root:
        store = FileSecretStore(root / "realm")
        with pytest.raises(ValueError, match="encryption must be one of"):
            store.put_secret(
                tag="transport_private_key",
                payload="secret",
                secret_id="transport-secure",
                encryption="password_v1",
                password="pw1",
            )


def test_write_and_read_client_access_round_trip() -> None:
    with _workspace_tmpdir() as root:
        realm_root = (root / "client-realm").resolve()
        path = write_client_access(
            realm_root,
            {
                "profiles": {
                    "demo": {
                        "engine_host_ssh_target": "user@example",
                        "secret_ref": "transport-1",
                    }
                }
            },
            realm="client-a",
        )
        assert path.exists()
        payload = read_client_access(realm_root)
        assert payload["realm"] == "client-a"
        profiles = dict(payload["client_access"].get("profiles") or {})
        assert profiles["demo"]["secret_ref"] == "transport-1"


def test_client_audit_event_round_trip_and_filtering() -> None:
    with _workspace_tmpdir() as root:
        realm_root = root / "client-realm"
        first = append_client_audit_event(
            realm_root,
            event_type="transport_bootstrap_import",
            realm="client-a",
            payload={"profile_name": "demo"},
        )
        second = append_client_audit_event(
            realm_root,
            event_type="profile_materialize",
            realm="client-a",
            payload={"profile_name": "demo"},
        )
        assert first.exists()
        assert second.exists()
        rows = list_client_audit_events(realm_root)
        assert len(rows) == 2
        assert rows[0]["realm"] == "client-a"
        filtered = list_client_audit_events(realm_root, event_type="transport_bootstrap_import")
        assert len(filtered) == 1
        assert filtered[0]["payload"]["profile_name"] == "demo"


def test_client_profile_round_trip_and_listing() -> None:
    with _workspace_tmpdir() as root:
        realm_root = (root / "client-realm").resolve()
        path = write_client_profile(
            realm_root,
            "demo",
            {
                "engine_host_ssh_target": "user@example",
                "control_ssh_key_secret_id": "transport-key",
            },
            realm="client-a",
        )
        assert path.exists()
        payload = read_client_profile(realm_root, "demo")
        assert payload["realm"] == "client-a"
        assert payload["profile"]["control_ssh_key_secret_id"] == "transport-key"
        assert list_client_profiles(realm_root) == ["demo"]


def test_transport_bootstrap_bundle_requires_known_hosts_line() -> None:
    with pytest.raises(ValueError, match="ssh_known_hosts_line is required"):
        make_transport_bootstrap_bundle(
            target="user@example",
            ssh_known_hosts_line="",
            transport_key_id="transport-key",
            transport_public_key="ssh-ed25519 AAAA transport",
            transport_private_key_openssh="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
        )


def test_transport_bootstrap_bundle_write_read_and_import() -> None:
    with _workspace_tmpdir() as root:
        bundle = make_transport_bootstrap_bundle(
            target="user@example",
            ssh_known_hosts_line="example ssh-ed25519 AAAATESTHOSTKEY",
            transport_key_id="transport-key",
            transport_public_key="ssh-ed25519 AAAATESTPUB transport",
            transport_private_key_openssh="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
            control_ssh_fingerprint="SHA256:abc",
            profile_name="demo",
            notes=["bootstrap import"],
        )
        assert bundle["kind"] == TRANSPORT_BOOTSTRAP_KIND
        bundle_path = root / "bundle.json"
        write_transport_bootstrap_bundle(bundle, bundle_path)
        loaded = read_transport_bootstrap_bundle(bundle_path)
        result = import_transport_bootstrap_bundle(
            bundle=loaded,
            client_realm_root=root / "client-realm",
            realm="client-a",
        )
        assert result["status"] == "ok"
        secret_file = root / "client-realm" / "secrets" / "transport-transport-key-private.json"
        known_hosts_file = root / "client-realm" / "known_hosts" / "demo.known_hosts"
        profile_file = root / "client-realm" / "profiles" / "demo.json"
        assert secret_file.exists()
        assert known_hosts_file.exists()
        assert profile_file.exists()
        assert Path(result["audit_path"]).exists()
        profile = read_client_profile(root / "client-realm", "demo")
        assert profile["profile"]["control_ssh_key_secret_id"] == "transport-transport-key-private"
        assert profile["profile"]["control_ssh_fingerprint"] == "SHA256:abc"
        access = read_client_access(root / "client-realm")
        assert access["client_access"]["profiles"]["demo"]["transport_key_id"] == "transport-key"
        assert access["client_access"]["profiles"]["demo"]["control_ssh_key_secret_id"] == "transport-transport-key-private"
        audit_rows = list_client_audit_events(root / "client-realm", event_type="transport_bootstrap_import")
        assert len(audit_rows) == 1
        assert audit_rows[0]["payload"]["secret_id"] == "transport-transport-key-private"


def test_transport_bootstrap_provisions_realm_local_ssh_artifacts() -> None:
    with _workspace_tmpdir() as root:
        realm_root = root / "client-realm"
        bundle = make_transport_bootstrap_bundle(
            target="user@example",
            ssh_known_hosts_line="example ssh-ed25519 AAAATESTHOSTKEY",
            transport_key_id="transport-key",
            transport_public_key="ssh-ed25519 AAAATESTPUB transport",
            transport_private_key_openssh="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
            profile_name="demo",
        )
        imported = import_transport_bootstrap_bundle(
            bundle=bundle,
            client_realm_root=realm_root,
            realm="client-a",
        )
        assert imported["status"] == "ok"

        from hosting.transport_bootstrap import provision_client_ssh_artifacts

        provisioned = provision_client_ssh_artifacts(
            client_realm_root=realm_root,
            profile_name="demo",
            realm="client-a",
            ssh_alias="demo-host",
        )
        config_path = Path(str(provisioned["ssh_config_file"]))
        identity_path = Path(str(provisioned["identity_file"]))
        known_hosts_path = Path(str(provisioned["known_hosts_file"]))
        assert config_path.exists()
        assert identity_path.exists()
        assert known_hosts_path.exists()
        text = config_path.read_text(encoding="utf-8")
        assert "Host demo-host" in text
        assert "HostName example" in text
        assert "User user" in text
        assert f"IdentityFile {identity_path}" in text
        assert f"UserKnownHostsFile {known_hosts_path}" in text
        assert "StrictHostKeyChecking yes" in text
        assert str(provisioned["ssh_command"]).endswith(" demo-host")


def test_install_transport_authorized_key_updates_managed_block() -> None:
    with _workspace_tmpdir() as root:
        auth_file = root / ".ssh" / "authorized_keys"
        public_key = "ssh-ed25519 AAAATESTPUB transport-key"
        first = install_transport_authorized_key(
            transport_public_key=public_key,
            authorized_keys_file=auth_file,
            transport_key_id="transport-key",
        )
        second = install_transport_authorized_key(
            transport_public_key=public_key,
            authorized_keys_file=auth_file,
            transport_key_id="transport-key",
        )
        text = auth_file.read_text(encoding="utf-8")
        assert first["status"] == "ok"
        assert second["replaced"] is True
        assert text.count(public_key) == 1
        assert 'command="python -m hosting.engine_host_cli --relay-wrapper"' in text
        assert "no-pty,no-agent-forwarding,no-X11-forwarding,no-port-forwarding" in text
        assert "# BEGIN mp13-hosting-transport transport-key" in text
        assert "# END mp13-hosting-transport transport-key" in text


def test_transport_bootstrap_bundle_password_protects_openssh_private_key_and_imports_secret(monkeypatch) -> None:
    with _workspace_tmpdir() as root:
        monkeypatch.setattr(
            "hosting.transport_bootstrap._protect_openssh_private_key",
            lambda private_key_text, **_kwargs: str(private_key_text) + "\nPROTECTED",
        )
        bundle = make_transport_bootstrap_bundle(
            target="user@example",
            ssh_known_hosts_line="example ssh-ed25519 AAAATESTHOSTKEY",
            transport_key_id="transport-key",
            transport_public_key="ssh-ed25519 AAAATESTPUB transport",
            transport_private_key_openssh="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
            bundle_password="bundle-pw",
            profile_name="demo",
        )
        assert bundle["transport_private_key_format"] == "openssh"
        assert bundle["transport_private_key_protection"] == "openssh_passphrase"
        assert "PROTECTED" in bundle["transport_private_key_openssh"]
        result = import_transport_bootstrap_bundle(
            bundle=bundle,
            client_realm_root=root / "client-realm",
            realm="client-a",
            bundle_password="bundle-pw",
            secret_password="secret-pw",
        )
        assert result["secret_encryption"] == "none"
        assert result["private_key_protection"] == "openssh_passphrase"
        store = FileSecretStore(root / "client-realm", realm="client-a")
        payload = store.get_secret_payload("transport-transport-key-private", password="secret-pw")
        assert "BEGIN OPENSSH PRIVATE KEY" in str(payload or "")


def test_transport_bootstrap_import_rejects_conflicting_existing_host_pin() -> None:
    with _workspace_tmpdir() as root:
        realm_root = root / "client-realm"
        write_client_profile(
            realm_root,
            "demo",
            {
                "ssh_known_hosts_line": "old ssh-ed25519 AAAAOLD",
            },
            realm="client-a",
        )
        bundle = make_transport_bootstrap_bundle(
            target="user@example",
            ssh_known_hosts_line="new ssh-ed25519 AAAANEW",
            transport_key_id="transport-key",
            transport_public_key="ssh-ed25519 AAAATESTPUB transport",
            transport_private_key_openssh="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----",
            profile_name="demo",
        )
        with pytest.raises(ValueError, match="conflicting pinned SSH host key"):
            import_transport_bootstrap_bundle(
                bundle=bundle,
                client_realm_root=realm_root,
                realm="client-a",
            )


def test_materialize_secret_file_and_resolve_client_profile_control_settings() -> None:
    with _workspace_tmpdir() as root:
        realm_root = root / "client-realm"
        store = FileSecretStore(realm_root, realm="client-a")
        store.put_secret(
            tag="transport_private_key",
            payload="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----\n",
            secret_id="transport-key",
        )
        known_hosts_file = realm_root / "known_hosts" / "demo.known_hosts"
        ensure_client_realm_dirs(realm_root)
        known_hosts_file.write_text("example ssh-ed25519 AAAATEST\n", encoding="utf-8")
        write_client_profile(
            realm_root,
            "demo",
            {
                "engine_host_ssh_target": "user@example",
                "control_ssh_key_secret_id": "transport-key",
                "ssh_known_hosts_file": str(known_hosts_file),
                "control_ssh_fingerprint": "SHA256:demo",
            },
            realm="client-a",
        )

        out = resolve_client_profile_control_settings(
            {
                "engine_host_client_realm_root": str(realm_root),
                "engine_host_client_realm": "client-a",
                "engine_host_client_profile": "demo",
            }
        )

        assert out["engine_host_ssh_target"] == "user@example"
        assert out["ssh_known_hosts_line"] == "example ssh-ed25519 AAAATEST"
        assert out["control_ssh_fingerprint"] == "SHA256:demo"
        assert Path(out["control_ssh_key"]).exists()
        assert Path(out["control_ssh_key"]) == managed_key_path(realm_root, "demo-transport-key")
        assert "FAKE" in Path(out["control_ssh_key"]).read_text(encoding="utf-8")


def test_resolve_client_profile_control_settings_materializes_openssh_secret() -> None:
    with _workspace_tmpdir() as root:
        realm_root = root / "client-realm"
        store = FileSecretStore(realm_root, realm="client-a")
        store.put_secret(
            tag="transport_private_key",
            payload="-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n-----END OPENSSH PRIVATE KEY-----\n",
            secret_id="transport-key",
            metadata={"private_key_format": "openssh", "private_key_protection": "none"},
        )
        write_client_profile(
            realm_root,
            "demo",
            {
                "engine_host_ssh_target": "user@example",
                "control_ssh_key_secret_id": "transport-key",
                "ssh_known_hosts_line": "example ssh-ed25519 AAAATEST",
            },
            realm="client-a",
        )
        out = resolve_client_profile_control_settings(
            {
                "engine_host_client_realm_root": str(realm_root),
                "engine_host_client_realm": "client-a",
                "engine_host_client_profile": "demo",
            }
        )
        assert Path(out["control_ssh_key"]).exists()


def test_resolve_client_profile_control_settings_preserves_explicit_overrides() -> None:
    with _workspace_tmpdir() as root:
        realm_root = root / "client-realm"
        store = FileSecretStore(realm_root, realm="default")
        store.put_secret(tag="transport_private_key", payload="secret", secret_id="transport-key")
        write_client_profile(
            realm_root,
            "demo",
            {
                "engine_host_ssh_target": "user@example",
                "control_ssh_key_secret_id": "transport-key",
                "ssh_known_hosts_line": "example ssh-ed25519 AAAATEST",
            },
        )

        out = resolve_client_profile_control_settings(
            {
                "engine_host_client_realm_root": str(realm_root),
                "engine_host_client_profile": "demo",
                "control_ssh_key": "X:/preconfigured/key",
                "engine_host_ssh_target": "override@example",
            }
        )

        assert out["control_ssh_key"] == "X:/preconfigured/key"
        assert out["engine_host_ssh_target"] == "override@example"
        assert not managed_key_path(realm_root, "demo-transport-key").exists()


def test_materialize_secret_file_rejects_missing_secret() -> None:
    with _workspace_tmpdir() as root:
        with pytest.raises(ValueError, match="is not present"):
            materialize_secret_file(root / "client-realm", secret_id="missing")
