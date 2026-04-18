"""
Client-side hosting realm helpers and file-backed secret records.

This module intentionally starts with a simple file backend so later work can
swap in OS-specific secret storage without changing higher-level workflows.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mp13_engine.mp13_config_paths import get_default_config_dir


CLIENT_REALM_ROOT_SUBDIR = "hosting_client"
VALID_SECRET_RECORD_ENCRYPTION = {"none", "password_v1"}
_PASSWORD_V1_SCRYPT_N = 1 << 14
_PASSWORD_V1_SCRYPT_R = 8
_PASSWORD_V1_SCRYPT_P = 1
_PASSWORD_V1_KEY_LEN = 64
_PASSWORD_V1_NONCE_LEN = 16


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(raw: str) -> bytes:
    return base64.b64decode(str(raw or "").encode("ascii"))


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))


def _password_v1_keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _password_v1_encrypt(plaintext: str, password: str) -> str:
    pwd = str(password or "")
    if not pwd:
        raise ValueError("password is required for password_v1 encryption")
    plaintext_bytes = str(plaintext or "").encode("utf-8")
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(_PASSWORD_V1_NONCE_LEN)
    key_material = hashlib.scrypt(
        pwd.encode("utf-8"),
        salt=salt,
        n=_PASSWORD_V1_SCRYPT_N,
        r=_PASSWORD_V1_SCRYPT_R,
        p=_PASSWORD_V1_SCRYPT_P,
        dklen=_PASSWORD_V1_KEY_LEN,
    )
    enc_key = key_material[:32]
    mac_key = key_material[32:]
    ciphertext = _xor_bytes(plaintext_bytes, _password_v1_keystream(enc_key, nonce, len(plaintext_bytes)))
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    envelope = {
        "scheme": "password_v1",
        "kdf": {
            "name": "scrypt",
            "salt_b64": _b64e(salt),
            "n": _PASSWORD_V1_SCRYPT_N,
            "r": _PASSWORD_V1_SCRYPT_R,
            "p": _PASSWORD_V1_SCRYPT_P,
            "dklen": _PASSWORD_V1_KEY_LEN,
        },
        "cipher": {
            "name": "hmac_sha256_xor_stream",
            "nonce_b64": _b64e(nonce),
            "ciphertext_b64": _b64e(ciphertext),
            "tag_b64": _b64e(tag),
        },
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True)


def _password_v1_decrypt(payload: str, password: str) -> str:
    pwd = str(password or "")
    if not pwd:
        raise ValueError("password is required for password_v1 decryption")
    envelope = dict(json.loads(str(payload or "")))
    if str(envelope.get("scheme") or "") != "password_v1":
        raise ValueError("password_v1 payload is missing scheme marker")
    kdf = dict(envelope.get("kdf") or {})
    cipher = dict(envelope.get("cipher") or {})
    salt = _b64d(str(kdf.get("salt_b64") or ""))
    nonce = _b64d(str(cipher.get("nonce_b64") or ""))
    ciphertext = _b64d(str(cipher.get("ciphertext_b64") or ""))
    tag = _b64d(str(cipher.get("tag_b64") or ""))
    key_material = hashlib.scrypt(
        pwd.encode("utf-8"),
        salt=salt,
        n=max(2, int(kdf.get("n") or _PASSWORD_V1_SCRYPT_N)),
        r=max(1, int(kdf.get("r") or _PASSWORD_V1_SCRYPT_R)),
        p=max(1, int(kdf.get("p") or _PASSWORD_V1_SCRYPT_P)),
        dklen=max(32, int(kdf.get("dklen") or _PASSWORD_V1_KEY_LEN)),
    )
    enc_key = key_material[:32]
    mac_key = key_material[32:64]
    expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("password_v1 authentication failed")
    plaintext = _xor_bytes(ciphertext, _password_v1_keystream(enc_key, nonce, len(ciphertext)))
    return plaintext.decode("utf-8")


def get_default_client_realm_root(*, default_config_dir: Optional[Path] = None, realm: str = "default") -> Path:
    cfg_root = Path(default_config_dir or get_default_config_dir()).expanduser().resolve()
    realm_name = str(realm or "default").strip() or "default"
    return (cfg_root / CLIENT_REALM_ROOT_SUBDIR / realm_name).resolve()


def client_realm_layout(root: Path) -> Dict[str, Path]:
    realm_root = Path(root).expanduser().resolve()
    return {
        "root": realm_root,
        "client_access": realm_root / "client_access.json",
        "keyring": realm_root / "keyring",
        "keys": realm_root / "keyring" / "keys.json",
        "secrets": realm_root / "secrets",
        "managed_keys": realm_root / "managed_keys",
        "known_hosts": realm_root / "known_hosts",
        "audit": realm_root / "audit",
        "profiles": realm_root / "profiles",
    }


def ensure_client_realm_dirs(root: Path) -> Dict[str, Path]:
    layout = client_realm_layout(root)
    for key in ("root", "keyring", "secrets", "managed_keys", "known_hosts", "audit", "profiles"):
        layout[key].mkdir(parents=True, exist_ok=True)
    return layout


def secret_record_path(root: Path, secret_id: str) -> Path:
    sid = str(secret_id or "").strip()
    if not sid:
        raise ValueError("secret_id is required")
    return (client_realm_layout(root)["secrets"] / f"{sid}.json").resolve()


def profile_path(root: Path, profile_name: str) -> Path:
    name = str(profile_name or "").strip()
    if not name:
        raise ValueError("profile_name is required")
    return (client_realm_layout(root)["profiles"] / f"{name}.json").resolve()


def managed_key_path(root: Path, name: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip()).strip("._")
    if not stem:
        raise ValueError("managed key name is required")
    return (client_realm_layout(root)["managed_keys"] / f"{stem}.key").resolve()


@dataclass(frozen=True)
class SecretRecord:
    version: int
    secret_id: str
    tag: str
    realm: str
    created_at: float
    updated_at: float
    encryption: str
    payload: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "secret_id": str(self.secret_id),
            "tag": str(self.tag),
            "realm": str(self.realm),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "encryption": str(self.encryption),
            "payload": str(self.payload),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SecretRecord":
        data = dict(payload or {})
        metadata = dict(data.get("metadata") or {})
        return cls(
            version=max(1, int(data.get("version") or 1)),
            secret_id=str(data.get("secret_id") or "").strip(),
            tag=str(data.get("tag") or "").strip(),
            realm=str(data.get("realm") or "default").strip() or "default",
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
            encryption=str(data.get("encryption") or "none").strip() or "none",
            payload=str(data.get("payload") or ""),
            metadata=metadata,
        )


class FileSecretStore:
    """
    File-backed tagged secret records for the client realm.

    The first implementation supports plaintext record persistence and reserves
    the record shape needed for future password-encrypted backends.
    """

    def __init__(self, root: Path, *, realm: str = "default") -> None:
        self.root = Path(root).expanduser().resolve()
        self.realm = str(realm or "default").strip() or "default"
        self.layout = ensure_client_realm_dirs(self.root)

    def _record_path(self, secret_id: str) -> Path:
        return secret_record_path(self.root, secret_id)

    def put_secret(
        self,
        *,
        tag: str,
        payload: str,
        secret_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        encryption: str = "none",
        password: Optional[str] = None,
    ) -> SecretRecord:
        tag_norm = str(tag or "").strip()
        if not tag_norm:
            raise ValueError("tag is required")
        enc = str(encryption or "none").strip().lower() or "none"
        if enc not in VALID_SECRET_RECORD_ENCRYPTION:
            raise ValueError(
                "encryption must be one of: "
                + ", ".join(sorted(VALID_SECRET_RECORD_ENCRYPTION))
            )
        payload_text = str(payload or "")
        if enc == "password_v1":
            payload_text = _password_v1_encrypt(payload_text, str(password or ""))
        sid = str(secret_id or "").strip() or secrets.token_urlsafe(12)
        now = time.time()
        existing = self.get_secret_record(sid)
        created_at = float(existing.created_at) if existing else now
        record = SecretRecord(
            version=1,
            secret_id=sid,
            tag=tag_norm,
            realm=self.realm,
            created_at=created_at,
            updated_at=now,
            encryption=enc,
            payload=payload_text,
            metadata=dict(metadata or {}),
        )
        path = self._record_path(sid)
        path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            path.chmod(0o600)
        except Exception:
            pass
        return record

    def get_secret_record(self, secret_id: str) -> Optional[SecretRecord]:
        path = self._record_path(secret_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = SecretRecord.from_dict(payload)
        if not record.secret_id:
            raise ValueError(f"secret record {path} is missing secret_id")
        if record.realm and record.realm != self.realm:
            raise ValueError(f"secret record {path} belongs to realm {record.realm}, expected {self.realm}")
        return record

    def get_secret_payload(self, secret_id: str, *, password: Optional[str] = None) -> Optional[str]:
        record = self.get_secret_record(secret_id)
        if record is None:
            return None
        if record.encryption == "none":
            return str(record.payload)
        if record.encryption == "password_v1":
            return _password_v1_decrypt(str(record.payload), str(password or ""))
        raise ValueError(f"Unsupported secret-record encryption: {record.encryption}")

    def delete_secret(self, secret_id: str) -> bool:
        path = self._record_path(secret_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_records(self, *, tag: Optional[str] = None) -> list[SecretRecord]:
        records: list[SecretRecord] = []
        want_tag = str(tag or "").strip()
        for path in sorted(self.layout["secrets"].glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = SecretRecord.from_dict(payload)
            if record.realm and record.realm != self.realm:
                continue
            if want_tag and record.tag != want_tag:
                continue
            records.append(record)
        return records

    def reencrypt_secret(
        self,
        secret_id: str,
        *,
        encryption: str,
        password: Optional[str] = None,
        current_password: Optional[str] = None,
    ) -> SecretRecord:
        record = self.get_secret_record(secret_id)
        if record is None:
            raise ValueError("secret_id is not present")
        enc = str(encryption or "").strip().lower()
        if not enc:
            raise ValueError("encryption is required")
        if enc == record.encryption:
            return record
        plaintext = self.get_secret_payload(secret_id, password=current_password)
        return self.put_secret(
            tag=record.tag,
            payload=str(plaintext or ""),
            secret_id=record.secret_id,
            metadata=dict(record.metadata or {}),
            encryption=enc,
            password=password,
        )


def write_client_access(
    root: Path,
    payload: Dict[str, Any],
    *,
    realm: str = "default",
) -> Path:
    layout = ensure_client_realm_dirs(root)
    out = {
        "version": 1,
        "realm": str(realm or "default").strip() or "default",
        "updated_at": time.time(),
        "client_access": dict(payload or {}),
    }
    path = layout["client_access"]
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except Exception:
        pass
    return path


def read_client_access(root: Path) -> Dict[str, Any]:
    path = client_realm_layout(root)["client_access"]
    if not path.exists():
        return {"version": 1, "realm": "default", "client_access": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": max(1, int(payload.get("version") or 1)),
        "realm": str(payload.get("realm") or "default").strip() or "default",
        "updated_at": float(payload.get("updated_at") or 0.0),
        "client_access": dict(payload.get("client_access") or {}),
    }


def write_client_profile(
    root: Path,
    profile_name: str,
    payload: Dict[str, Any],
    *,
    realm: str = "default",
) -> Path:
    layout = ensure_client_realm_dirs(root)
    out = {
        "version": 1,
        "realm": str(realm or "default").strip() or "default",
        "profile_name": str(profile_name or "").strip(),
        "updated_at": time.time(),
        "profile": dict(payload or {}),
    }
    path = profile_path(root, profile_name)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except Exception:
        pass
    return path


def read_client_profile(root: Path, profile_name: str) -> Dict[str, Any]:
    path = profile_path(root, profile_name)
    if not path.exists():
        return {
            "version": 1,
            "realm": "default",
            "profile_name": str(profile_name or "").strip(),
            "profile": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": max(1, int(payload.get("version") or 1)),
        "realm": str(payload.get("realm") or "default").strip() or "default",
        "profile_name": str(payload.get("profile_name") or profile_name).strip(),
        "updated_at": float(payload.get("updated_at") or 0.0),
        "profile": dict(payload.get("profile") or {}),
    }


def list_client_profiles(root: Path) -> list[str]:
    layout = client_realm_layout(root)
    if not layout["profiles"].exists():
        return []
    return sorted(path.stem for path in layout["profiles"].glob("*.json") if path.is_file())


def iter_secret_ids(root: Path, *, realm: str = "default", tag: Optional[str] = None) -> Iterable[str]:
    store = FileSecretStore(root, realm=realm)
    for record in store.list_records(tag=tag):
        yield record.secret_id


def append_client_audit_event(
    root: Path,
    *,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
    realm: str = "default",
) -> Path:
    layout = ensure_client_realm_dirs(root)
    event_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(event_type or "").strip()).strip("._")
    if not event_name:
        raise ValueError("event_type is required")
    event_path = layout["audit"] / f"{int(time.time() * 1000)}-{event_name}.json"
    out = {
        "version": 1,
        "realm": str(realm or "default").strip() or "default",
        "event_type": event_name,
        "created_at": time.time(),
        "payload": dict(payload or {}),
    }
    event_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        event_path.chmod(0o600)
    except Exception:
        pass
    return event_path


def list_client_audit_events(root: Path, *, event_type: str = "") -> list[Dict[str, Any]]:
    layout = client_realm_layout(root)
    audit_dir = layout["audit"]
    if not audit_dir.exists():
        return []
    want_type = str(event_type or "").strip()
    rows: list[Dict[str, Any]] = []
    for path in sorted(audit_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "version": max(1, int(payload.get("version") or 1)),
            "realm": str(payload.get("realm") or "default").strip() or "default",
            "event_type": str(payload.get("event_type") or "").strip(),
            "created_at": float(payload.get("created_at") or 0.0),
            "payload": dict(payload.get("payload") or {}),
            "path": str(path.resolve()),
        }
        if want_type and row["event_type"] != want_type:
            continue
        rows.append(row)
    return rows


def materialize_secret_file(
    root: Path,
    *,
    secret_id: str,
    realm: str = "default",
    name: Optional[str] = None,
    password: Optional[str] = None,
) -> Path:
    store = FileSecretStore(root, realm=realm)
    record = store.get_secret_record(secret_id)
    if record is None:
        raise ValueError(f"secret_id {secret_id!r} is not present")
    out_path = managed_key_path(root, name or secret_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(str(store.get_secret_payload(secret_id, password=password) or ""), encoding="utf-8")
    try:
        out_path.chmod(0o600)
    except Exception:
        pass
    return out_path


def resolve_client_profile_control_settings(
    control_settings: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    raw = dict(control_settings or {})
    profile_name = str(
        raw.get("engine_host_client_profile")
        or raw.get("client_profile_name")
        or ""
    ).strip()
    if not profile_name:
        return raw
    realm = str(
        raw.get("engine_host_client_realm")
        or raw.get("client_realm")
        or "default"
    ).strip() or "default"
    root_raw = str(
        raw.get("engine_host_client_realm_root")
        or raw.get("client_realm_root")
        or ""
    ).strip()
    if root_raw:
        root = Path(root_raw).expanduser().resolve()
    else:
        default_config_dir = raw.get("engine_host_default_config_dir") or raw.get("default_config_dir")
        cfg_root = None if default_config_dir in (None, "") else Path(str(default_config_dir)).expanduser().resolve()
        root = get_default_client_realm_root(default_config_dir=cfg_root, realm=realm)
    profile_payload = read_client_profile(root, profile_name)
    profile = dict(profile_payload.get("profile") or {})
    if not profile:
        raise ValueError(f"client profile {profile_name!r} was not found in realm {realm!r}")
    resolved = dict(profile)
    resolved["engine_host_client_profile"] = profile_name
    resolved["engine_host_client_realm"] = realm
    resolved["engine_host_client_realm_root"] = str(root)
    secret_id = str(
        profile.get("control_ssh_key_secret_id")
        or profile.get("ssh_key_secret_id")
        or ""
    ).strip()
    if secret_id and not str(raw.get("control_ssh_key") or "").strip():
        managed_name = f"{profile_name}-{secret_id}"
        resolved["control_ssh_key"] = str(
            materialize_secret_file(
                root,
                secret_id=secret_id,
                realm=realm,
                name=managed_name,
                password=raw.get("engine_host_client_secret_password") or raw.get("client_secret_password"),
            )
        )
    if not str(resolved.get("ssh_known_hosts_line") or "").strip():
        known_hosts_file = str(profile.get("ssh_known_hosts_file") or "").strip()
        if known_hosts_file:
            line = Path(known_hosts_file).expanduser().resolve().read_text(encoding="utf-8").strip()
            if line:
                resolved["ssh_known_hosts_line"] = line.splitlines()[0].strip()
    resolved.update(raw)
    return resolved
