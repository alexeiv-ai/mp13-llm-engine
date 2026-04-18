"""
Transport bootstrap bundle helpers.

These helpers model the out-of-band artifact used to provision transport keys
and pinned SSH host-key material to a client realm.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .client_realm import (
    FileSecretStore,
    _password_v1_decrypt,
    _password_v1_encrypt,
    append_client_audit_event,
    ensure_client_realm_dirs,
    profile_path,
    read_client_access,
    read_client_profile,
    resolve_client_profile_control_settings,
    secret_record_path,
    write_client_access,
    write_client_profile,
)


TRANSPORT_BOOTSTRAP_KIND = "hosting_transport_bootstrap"


def make_transport_bootstrap_bundle(
    *,
    target: str,
    ssh_known_hosts_line: str,
    transport_key_id: str,
    transport_public_key: str,
    transport_private_key_openssh: str = "",
    bundle_password: str = "",
    control_ssh_fingerprint: str = "",
    profile_name: str = "",
    notes: Optional[list[str]] = None,
) -> Dict[str, Any]:
    target_norm = str(target or "").strip()
    known_hosts_norm = str(ssh_known_hosts_line or "").strip()
    key_id_norm = str(transport_key_id or "").strip()
    public_key_norm = str(transport_public_key or "").strip()
    private_key_norm = str(transport_private_key_openssh or "").strip()
    if not target_norm:
        raise ValueError("target is required")
    if not known_hosts_norm:
        raise ValueError("ssh_known_hosts_line is required")
    if not key_id_norm:
        raise ValueError("transport_key_id is required")
    if not public_key_norm:
        raise ValueError("transport_public_key is required")
    if not private_key_norm:
        raise ValueError("transport_private_key_openssh is required")
    bundle_password_norm = str(bundle_password or "")
    bundle: Dict[str, Any] = {
        "bundle_version": 1,
        "kind": TRANSPORT_BOOTSTRAP_KIND,
        "created_at": time.time(),
        "target": target_norm,
        "ssh_known_hosts_line": known_hosts_norm,
        "transport_key_id": key_id_norm,
        "transport_public_key": public_key_norm,
        "transport_private_key_encryption": "none",
    }
    if bundle_password_norm:
        bundle["transport_private_key_encryption"] = "password_v1"
        bundle["transport_private_key_password_v1"] = json.loads(
            _password_v1_encrypt(private_key_norm, bundle_password_norm)
        )
    else:
        bundle["transport_private_key_openssh"] = private_key_norm
    fingerprint_norm = str(control_ssh_fingerprint or "").strip()
    if fingerprint_norm:
        bundle["control_ssh_fingerprint"] = fingerprint_norm
    profile_norm = str(profile_name or "").strip()
    if profile_norm:
        bundle["profile_name"] = profile_norm
    note_rows = [str(item).strip() for item in list(notes or []) if str(item).strip()]
    if note_rows:
        bundle["notes"] = note_rows
    return bundle


def validate_transport_bootstrap_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(bundle or {})
    kind = str(payload.get("kind") or "").strip()
    if kind != TRANSPORT_BOOTSTRAP_KIND:
        raise ValueError(f"bundle kind must be {TRANSPORT_BOOTSTRAP_KIND!r}")
    version = max(1, int(payload.get("bundle_version") or 1))
    target = str(payload.get("target") or "").strip()
    known_hosts_line = str(payload.get("ssh_known_hosts_line") or "").strip()
    key_id = str(payload.get("transport_key_id") or "").strip()
    public_key = str(payload.get("transport_public_key") or "").strip()
    private_key_encryption = str(payload.get("transport_private_key_encryption") or "none").strip() or "none"
    private_key = str(payload.get("transport_private_key_openssh") or "").strip()
    encrypted_private_key = payload.get("transport_private_key_password_v1")
    if not target:
        raise ValueError("bundle target is required")
    if not known_hosts_line:
        raise ValueError("bundle ssh_known_hosts_line is required")
    if not key_id:
        raise ValueError("bundle transport_key_id is required")
    if not public_key:
        raise ValueError("bundle transport_public_key is required")
    if private_key_encryption == "none":
        if not private_key:
            raise ValueError("bundle transport_private_key_openssh is required")
    elif private_key_encryption == "password_v1":
        if not isinstance(encrypted_private_key, dict):
            raise ValueError("bundle transport_private_key_password_v1 is required")
    else:
        raise ValueError("bundle transport_private_key_encryption is invalid")
    out = {
        "bundle_version": version,
        "kind": TRANSPORT_BOOTSTRAP_KIND,
        "created_at": float(payload.get("created_at") or 0.0),
        "target": target,
        "ssh_known_hosts_line": known_hosts_line,
        "transport_key_id": key_id,
        "transport_public_key": public_key,
        "transport_private_key_encryption": private_key_encryption,
        "transport_private_key_openssh": private_key if private_key_encryption == "none" else None,
        "transport_private_key_password_v1": encrypted_private_key if private_key_encryption == "password_v1" else None,
        "control_ssh_fingerprint": str(payload.get("control_ssh_fingerprint") or "").strip() or None,
        "profile_name": str(payload.get("profile_name") or "").strip() or None,
        "notes": [str(item).strip() for item in list(payload.get("notes") or []) if str(item).strip()],
    }
    return out


def write_transport_bootstrap_bundle(bundle: Dict[str, Any], path: Path) -> Path:
    validated = validate_transport_bootstrap_bundle(bundle)
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        out_path.chmod(0o600)
    except Exception:
        pass
    return out_path


def read_transport_bootstrap_bundle(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return validate_transport_bootstrap_bundle(payload)


def import_transport_bootstrap_bundle(
    *,
    bundle: Dict[str, Any],
    client_realm_root: Path,
    realm: str = "default",
    profile_name: Optional[str] = None,
    overwrite_profile: bool = False,
    bundle_password: str = "",
    secret_password: str = "",
) -> Dict[str, Any]:
    validated = validate_transport_bootstrap_bundle(bundle)
    realm_norm = str(realm or "default").strip() or "default"
    layout = ensure_client_realm_dirs(client_realm_root)
    name = str(profile_name or validated.get("profile_name") or "default").strip() or "default"
    existing = read_client_profile(client_realm_root, name)
    existing_profile = dict(existing.get("profile") or {})
    if existing_profile and not overwrite_profile:
        existing_known_hosts = str(existing_profile.get("ssh_known_hosts_line") or "").strip()
        incoming_known_hosts = str(validated.get("ssh_known_hosts_line") or "").strip()
        if existing_known_hosts and existing_known_hosts != incoming_known_hosts:
            raise ValueError("profile already exists with conflicting pinned SSH host key")
    secret_store = FileSecretStore(client_realm_root, realm=realm_norm)
    secret_id = f"transport-{validated['transport_key_id']}-private"
    private_key_text = str(validated.get("transport_private_key_openssh") or "")
    if str(validated.get("transport_private_key_encryption") or "") == "password_v1":
        private_key_text = _password_v1_decrypt(
            json.dumps(dict(validated.get("transport_private_key_password_v1") or {}), ensure_ascii=False, sort_keys=True),
            str(bundle_password or ""),
        )
    secret = secret_store.put_secret(
        tag="transport_private_key",
        payload=private_key_text,
        secret_id=secret_id,
        metadata={
            "target": str(validated["target"]),
            "transport_key_id": str(validated["transport_key_id"]),
            "source": "transport_bootstrap_import",
        },
        encryption="password_v1" if str(secret_password or "") else "none",
        password=str(secret_password or ""),
    )
    known_hosts_path = (layout["known_hosts"] / f"{name}.known_hosts").resolve()
    known_hosts_path.write_text(str(validated["ssh_known_hosts_line"]) + "\n", encoding="utf-8")
    try:
        known_hosts_path.chmod(0o600)
    except Exception:
        pass
    profile_payload: Dict[str, Any] = {
        "engine_host_ssh_target": str(validated["target"]),
        "control_ssh_key_secret_id": secret.secret_id,
        "control_ssh_key_secret_path": str(secret_record_path(client_realm_root, secret.secret_id)),
        "ssh_known_hosts_line": str(validated["ssh_known_hosts_line"]),
        "ssh_known_hosts_file": str(known_hosts_path),
        "transport_key_id": str(validated["transport_key_id"]),
        "transport_public_key": str(validated["transport_public_key"]),
    }
    fingerprint = str(validated.get("control_ssh_fingerprint") or "").strip()
    if fingerprint:
        profile_payload["control_ssh_fingerprint"] = fingerprint
    if list(validated.get("notes") or []):
        profile_payload["notes"] = list(validated.get("notes") or [])
    write_client_profile(client_realm_root, name, profile_payload, realm=realm_norm)
    client_access = read_client_access(client_realm_root)
    profiles = dict(client_access.get("client_access", {}).get("profiles") or {})
    profiles[name] = {
        "engine_host_ssh_target": str(validated["target"]),
        "transport_key_id": str(validated["transport_key_id"]),
        "control_ssh_key_secret_id": secret.secret_id,
        "ssh_known_hosts_file": str(known_hosts_path),
        "profile_path": str(profile_path(client_realm_root, name)),
    }
    fingerprint = str(validated.get("control_ssh_fingerprint") or "").strip()
    if fingerprint:
        profiles[name]["control_ssh_fingerprint"] = fingerprint
    write_client_access(
        client_realm_root,
        {"profiles": profiles},
        realm=realm_norm,
    )
    audit_path = append_client_audit_event(
        client_realm_root,
        event_type="transport_bootstrap_import",
        realm=realm_norm,
        payload={
            "profile_name": name,
            "target": str(validated["target"]),
            "transport_key_id": str(validated["transport_key_id"]),
            "secret_id": secret.secret_id,
            "secret_encryption": str(secret.encryption),
            "known_hosts_file": str(known_hosts_path),
        },
    )
    return {
        "status": "ok",
        "realm": realm_norm,
        "profile_name": name,
        "secret_id": secret.secret_id,
        "secret_path": str(secret_record_path(client_realm_root, secret.secret_id)),
        "secret_encryption": str(secret.encryption),
        "known_hosts_file": str(known_hosts_path),
        "profile_path": str(profile_path(client_realm_root, name)),
        "audit_path": str(audit_path),
    }


def validate_client_transport_profile(
    *,
    client_realm_root: Path,
    profile_name: str,
    realm: str = "default",
    run_ssh: bool = True,
    ssh_bin: str = "ssh",
    remote_command: str = "exit 0",
    timeout_seconds: float = 15.0,
    secret_password: str = "",
) -> Dict[str, Any]:
    resolved = resolve_client_profile_control_settings(
        {
            "engine_host_client_realm_root": str(Path(client_realm_root).expanduser().resolve()),
            "engine_host_client_realm": str(realm or "default").strip() or "default",
            "engine_host_client_profile": str(profile_name or "").strip(),
            "engine_host_client_secret_password": str(secret_password or ""),
        }
    )
    target = str(resolved.get("engine_host_ssh_target") or "").strip()
    ssh_key = str(resolved.get("control_ssh_key") or "").strip()
    known_hosts_file = str(resolved.get("ssh_known_hosts_file") or "").strip()
    known_hosts_line = str(resolved.get("ssh_known_hosts_line") or "").strip()
    if not target:
        raise ValueError("client profile is missing engine_host_ssh_target")
    if not ssh_key:
        raise ValueError("client profile is missing control_ssh_key")
    if not known_hosts_line:
        raise ValueError("client profile is missing ssh_known_hosts_line")
    if not known_hosts_file:
        raise ValueError("client profile is missing ssh_known_hosts_file")
    key_path = Path(ssh_key).expanduser().resolve()
    hosts_path = Path(known_hosts_file).expanduser().resolve()
    if not key_path.exists():
        raise ValueError(f"control_ssh_key does not exist: {key_path}")
    if not hosts_path.exists():
        raise ValueError(f"ssh_known_hosts_file does not exist: {hosts_path}")
    result: Dict[str, Any] = {
        "status": "ok",
        "profile_name": str(profile_name or "").strip(),
        "realm": str(realm or "default").strip() or "default",
        "target": target,
        "control_ssh_key": str(key_path),
        "ssh_known_hosts_file": str(hosts_path),
        "ssh_known_hosts_line": known_hosts_line,
        "control_ssh_fingerprint": str(resolved.get("control_ssh_fingerprint") or "").strip() or None,
        "ssh_probe_ran": False,
    }
    if not run_ssh:
        return result
    cmd = [
        str(ssh_bin),
        "-i",
        str(key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={hosts_path}",
        target,
        str(remote_command),
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1.0, float(timeout_seconds or 15.0)),
    )
    result["ssh_probe_ran"] = True
    result["ssh_probe_command"] = cmd
    result["ssh_probe_returncode"] = int(completed.returncode)
    result["ssh_probe_stdout"] = str(completed.stdout or "")
    result["ssh_probe_stderr"] = str(completed.stderr or "")
    result["status"] = "ok" if completed.returncode == 0 else "ssh_probe_failed"
    return result
