"""
Client-side hosting realm helpers and file-backed secret records.

This module intentionally starts with a simple file backend so later work can
swap in OS-specific secret storage without changing higher-level workflows.
"""
from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mp13_engine.mp13_config_paths import get_default_config_dir


CLIENT_REALM_ROOT_SUBDIR = "hosting_client"
VALID_SECRET_RECORD_ENCRYPTION = {"none"}


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
        "ssh_config": realm_root / "ssh_config",
        "audit": realm_root / "audit",
        "profiles": realm_root / "profiles",
    }


def ensure_client_realm_dirs(root: Path) -> Dict[str, Path]:
    layout = client_realm_layout(root)
    for key in ("root", "keyring", "secrets", "managed_keys", "known_hosts", "ssh_config", "audit", "profiles"):
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


def require_client_realm_private_key_path(root: Path, path: Path) -> Path:
    target = Path(path).expanduser().resolve()
    layout = ensure_client_realm_dirs(root)
    allowed_roots = (
        layout["managed_keys"].resolve(),
        layout["secrets"].resolve(),
    )
    if not any(target == base or base in target.parents for base in allowed_roots):
        raise ValueError(
            "private-key files managed by hosting must stay under the client realm "
            "managed_keys/ or secrets/ directories"
        )
    return target


def normalize_pasted_private_key(value: str) -> str:
    """
    Normalize private-key text accepted through a CLI argument or paste field.

    This accepts common copy/paste forms such as quoted text and literal ``\n``
    escapes without trying to validate the key format.
    """

    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def read_client_key_metadata(root: Path) -> Dict[str, Any]:
    path = client_realm_layout(root)["keys"]
    if not path.exists():
        return {"version": 1, "keys": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {"version": 1, "keys": {}}


def _read_keyring_file(keys_file: Path) -> Dict[str, Any]:
    path = Path(keys_file).expanduser().resolve()
    if not path.exists():
        return {"version": 1, "keys": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {"version": 1, "keys": {}}


def _write_keyring_file(keys_file: Path, payload: Dict[str, Any]) -> None:
    path = Path(keys_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload or {})
    out["version"] = max(1, int(out.get("version") or 1))
    out["updated_at"] = time.time()
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except Exception:
        pass


def discover_exported_private_keys(*, keys_file: Path) -> list[Dict[str, Any]]:
    payload = _read_keyring_file(keys_file)
    rows: list[Dict[str, Any]] = []
    for key_id, row_value in sorted(dict(payload.get("keys") or {}).items()):
        row = dict(row_value or {})
        export_path = str(row.get("private_key_export_path") or "").strip()
        storage = str(row.get("private_key_storage") or "").strip()
        if storage != "exported_file" and not export_path:
            continue
        path = Path(export_path).expanduser().resolve() if export_path else None
        rows.append(
            {
                "key_id": str(key_id),
                "role": str(row.get("role") or "").strip(),
                "auth_method": str(row.get("auth_method") or "").strip(),
                "public_key": str(row.get("public_key") or "").strip(),
                "private_key_storage": storage or "exported_file",
                "private_key_export_path": str(path) if path else "",
                "private_key_export_exists": bool(path and path.exists()),
                "private_key_handoff_recorded": bool(row.get("private_key_handoff_recorded")),
            }
        )
    return rows


def handoff_exported_private_key_to_realm(
    *,
    keys_file: Path,
    target_root: Path,
    key_id: str,
    realm: str = "default",
    tag: str = "rbac_private_key",
    delete_source_file: bool = False,
) -> Dict[str, Any]:
    key_id_norm = str(key_id or "").strip()
    if not key_id_norm:
        raise ValueError("key_id is required")
    payload = _read_keyring_file(keys_file)
    keys = dict(payload.get("keys") or {})
    row = dict(keys.get(key_id_norm) or {})
    export_raw = str(row.get("private_key_export_path") or "").strip()
    if not export_raw:
        raise ValueError(f"key {key_id_norm!r} has no exported private-key path")
    export_path = Path(export_raw).expanduser().resolve()
    if not export_path.exists():
        raise ValueError(f"exported private-key file does not exist: {export_path}")
    stored = store_private_key_in_realm(
        Path(target_root).expanduser().resolve(),
        realm=realm,
        key_id=key_id_norm,
        tag=tag,
        private_key_text=export_path.read_text(encoding="utf-8"),
        public_key=str(row.get("public_key") or ""),
        role=str(row.get("role") or "admin"),
        auth_method=str(row.get("auth_method") or "public_key"),
        key_origin=str(row.get("key_origin") or row.get("key_source") or "imported"),
        source="exported_private_key_handoff",
        private_key_protection=str(row.get("private_key_protection") or "unknown").strip() or "unknown",
    )
    audit_path = append_client_audit_event(
        Path(target_root).expanduser().resolve(),
        event_type="client_exported_key_handoff_imported",
        realm=realm,
        payload={
            "key_id": key_id_norm,
            "tag": tag,
            "source_keys_file": str(Path(keys_file).expanduser().resolve()),
            "source_export_path": str(export_path),
            "secret_id": stored.get("secret_id"),
        },
    )
    deleted = False
    if delete_source_file and export_path.exists():
        export_path.unlink()
        deleted = True
    row["private_key_storage"] = "exported_file"
    row["private_key_export_path"] = str(export_path)
    row["private_key_handoff_recorded"] = True
    row["private_key_export_exists"] = bool(export_path.exists())
    keys[key_id_norm] = row
    payload["keys"] = keys
    _write_keyring_file(keys_file, payload)
    return {
        **stored,
        "source_keys_file": str(Path(keys_file).expanduser().resolve()),
        "source_export_path": str(export_path),
        "deleted_source_file": deleted,
        "audit_path": str(audit_path),
    }


def purge_exported_private_key(*, keys_file: Path, key_id: str) -> Dict[str, Any]:
    key_id_norm = str(key_id or "").strip()
    if not key_id_norm:
        raise ValueError("key_id is required")
    payload = _read_keyring_file(keys_file)
    keys = dict(payload.get("keys") or {})
    row = dict(keys.get(key_id_norm) or {})
    export_raw = str(row.get("private_key_export_path") or "").strip()
    if not export_raw:
        raise ValueError(f"key {key_id_norm!r} has no exported private-key path")
    export_path = Path(export_raw).expanduser().resolve()
    deleted = False
    if export_path.exists():
        export_path.unlink()
        deleted = True
    row["private_key_storage"] = "exported_file"
    row["private_key_export_path"] = str(export_path)
    row["private_key_export_exists"] = False
    keys[key_id_norm] = row
    payload["keys"] = keys
    _write_keyring_file(keys_file, payload)
    warning = ""
    if not bool(row.get("private_key_handoff_recorded")):
        warning = "Purged exported private key without recording client-realm hand-off."
    return {
        "status": "ok",
        "action": "client_purge_exported_key",
        "key_id": key_id_norm,
        "source_keys_file": str(Path(keys_file).expanduser().resolve()),
        "source_export_path": str(export_path),
        "deleted_source_file": deleted,
        "warning": warning,
    }


def store_private_key_in_realm(
    root: Path,
    *,
    realm: str = "default",
    key_id: str,
    tag: str,
    private_key_text: str,
    public_key: str,
    role: str = "admin",
    auth_method: str = "public_key",
    key_origin: str = "imported",
    source: str = "client_import",
    private_key_protection: str = "none",
) -> Dict[str, Any]:
    key_id_norm = str(key_id or "").strip()
    if not key_id_norm:
        raise ValueError("key_id is required")
    tag_norm = str(tag or "").strip() or "rbac_private_key"
    payload = normalize_pasted_private_key(private_key_text)
    if not payload:
        raise ValueError("private key is required")
    realm_norm = str(realm or "default").strip() or "default"
    layout = ensure_client_realm_dirs(root)
    store = FileSecretStore(root, realm=realm_norm)
    secret_record = store.put_secret(
        tag=tag_norm,
        payload=payload,
        secret_id=f"{tag_norm.replace('_private_key', '')}-{key_id_norm}-private",
        metadata={
            "key_id": key_id_norm,
            "tag": tag_norm,
            "source": source,
            "private_key_format": "openssh",
            "private_key_protection": str(private_key_protection or "none").strip() or "none",
        },
        encryption="none",
    )
    keys_path = layout["keys"]
    keys_payload = read_client_key_metadata(root)
    keys = dict(keys_payload.get("keys") or {})
    now = time.time()
    keys[key_id_norm] = {
        **dict(keys.get(key_id_norm) or {}),
        "key_id": key_id_norm,
        "tag": tag_norm,
        "role": str(role or "admin").strip() or "admin",
        "auth_method": str(auth_method or "public_key").strip() or "public_key",
        "public_key": str(public_key or "").strip(),
        "key_origin": str(key_origin or "imported").strip() or "imported",
        "key_source": "import",
        "public_key_source": "metadata_or_derived",
        "private_key_storage": "client_realm_secret",
        "private_key_secret_id": secret_record.secret_id,
        "private_key_secret_realm": realm_norm,
        "private_key_protection": str(private_key_protection or "none").strip() or "none",
        "updated_at": now,
    }
    keys_payload["version"] = max(1, int(keys_payload.get("version") or 1))
    keys_payload["updated_at"] = now
    keys_payload["keys"] = keys
    keys_path.write_text(json.dumps(keys_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        keys_path.chmod(0o600)
    except Exception:
        pass
    return {
        "key_id": key_id_norm,
        "secret_id": secret_record.secret_id,
        "secret_path": str(secret_record_path(root, secret_record.secret_id)),
        "keys_file": str(keys_path),
    }


def delete_client_key_from_realm(
    root: Path,
    *,
    key_id: str,
    realm: str = "default",
) -> Dict[str, Any]:
    key_id_norm = str(key_id or "").strip()
    if not key_id_norm:
        raise ValueError("key_id is required")
    root_path = Path(root).expanduser().resolve()
    keys_path = client_realm_layout(root_path)["keys"]
    keys_payload = read_client_key_metadata(root_path)
    keys = dict(keys_payload.get("keys") or {})
    row = dict(keys.get(key_id_norm) or {})
    if not row:
        raise ValueError(f"client key {key_id_norm!r} was not found")
    secret_id = str(row.get("private_key_secret_id") or "").strip()
    deleted_secret = False
    deleted_export_file = False
    deleted_export_path: Optional[str] = None
    if secret_id:
        store = FileSecretStore(root_path, realm=str(realm or "default").strip() or "default")
        deleted_secret = bool(store.delete_secret(secret_id))
    export_path_raw = str(row.get("private_key_export_path") or "").strip()
    if export_path_raw:
        export_path = require_client_realm_private_key_path(root_path, Path(export_path_raw))
        deleted_export_path = str(export_path)
        if export_path.exists():
            export_path.unlink()
            deleted_export_file = True
    keys.pop(key_id_norm, None)
    keys_payload["version"] = max(1, int(keys_payload.get("version") or 1))
    keys_payload["updated_at"] = time.time()
    keys_payload["keys"] = keys
    keys_path.write_text(json.dumps(keys_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        keys_path.chmod(0o600)
    except Exception:
        pass
    return {
        "status": "ok",
        "key_id": key_id_norm,
        "deleted_secret": deleted_secret,
        "secret_id": secret_id or None,
        "deleted_export_file": deleted_export_file,
        "deleted_export_path": deleted_export_path,
        "keys_file": str(keys_path),
    }


HANDOFF_PAYLOAD_KIND = "mp13.hosting.client_private_key_handoff"


def create_private_key_handoff_text(
    root: Path,
    *,
    key_id: str,
    realm: str = "default",
    password: Optional[str] = None,
) -> Dict[str, Any]:
    key_id_norm = str(key_id or "").strip()
    if not key_id_norm:
        raise ValueError("key_id is required")
    realm_norm = str(realm or "default").strip() or "default"
    keys_payload = read_client_key_metadata(root)
    row = dict(dict(keys_payload.get("keys") or {}).get(key_id_norm) or {})
    secret_id = str(row.get("private_key_secret_id") or "").strip()
    if not secret_id:
        raise ValueError(f"client key {key_id_norm!r} does not reference a client-realm secret")
    store = FileSecretStore(root, realm=realm_norm)
    private_key_text = str(store.get_secret_payload(secret_id, password=password) or "")
    if not private_key_text:
        raise ValueError(f"client key {key_id_norm!r} has an empty private-key secret")
    payload = {
        "version": 1,
        "kind": HANDOFF_PAYLOAD_KIND,
        "created_at": time.time(),
        "realm": realm_norm,
        "key_id": key_id_norm,
        "tag": str(row.get("tag") or "rbac_private_key").strip() or "rbac_private_key",
        "role": str(row.get("role") or "admin").strip() or "admin",
        "auth_method": str(row.get("auth_method") or "public_key").strip() or "public_key",
        "public_key": str(row.get("public_key") or "").strip(),
        "key_origin": str(row.get("key_origin") or row.get("key_source") or "imported").strip() or "imported",
        "private_key_protection": str(row.get("private_key_protection") or "unknown").strip() or "unknown",
        "private_key": normalize_pasted_private_key(private_key_text),
    }
    audit_path = append_client_audit_event(
        root,
        event_type="client_key_handoff_text_created",
        realm=realm_norm,
        payload={
            "key_id": key_id_norm,
            "tag": payload["tag"],
            "role": payload["role"],
            "auth_method": payload["auth_method"],
            "source_secret_id": secret_id,
        },
    )
    return {
        "key_id": key_id_norm,
        "handoff": payload,
        "handoff_text": json.dumps(payload, ensure_ascii=False, indent=2),
        "audit_path": str(audit_path),
    }


def store_private_key_handoff_in_realm(
    root: Path,
    *,
    handoff_text: str | Dict[str, Any],
    realm: str = "default",
    tag: str = "",
) -> Dict[str, Any]:
    if isinstance(handoff_text, dict):
        payload = dict(handoff_text)
    else:
        payload = json.loads(str(handoff_text or "").strip())
    if str(payload.get("kind") or "") != HANDOFF_PAYLOAD_KIND:
        raise ValueError("handoff payload kind is not supported")
    key_id = str(payload.get("key_id") or "").strip()
    if not key_id:
        raise ValueError("handoff payload key_id is required")
    realm_norm = str(realm or payload.get("realm") or "default").strip() or "default"
    tag_norm = str(tag or payload.get("tag") or "rbac_private_key").strip() or "rbac_private_key"
    stored = store_private_key_in_realm(
        root,
        realm=realm_norm,
        key_id=key_id,
        tag=tag_norm,
        private_key_text=str(payload.get("private_key") or ""),
        public_key=str(payload.get("public_key") or ""),
        role=str(payload.get("role") or "admin"),
        auth_method=str(payload.get("auth_method") or "public_key"),
        key_origin=str(payload.get("key_origin") or "imported"),
        source="client_handoff_text",
        private_key_protection=str(payload.get("private_key_protection") or "unknown"),
    )
    audit_path = append_client_audit_event(
        root,
        event_type="client_key_handoff_text_imported",
        realm=realm_norm,
        payload={
            "key_id": key_id,
            "tag": tag_norm,
            "role": str(payload.get("role") or "admin"),
            "auth_method": str(payload.get("auth_method") or "public_key"),
            "secret_id": stored.get("secret_id"),
        },
    )
    return {**stored, "audit_path": str(audit_path)}


def migrate_private_key_between_realms(
    *,
    source_root: Path,
    target_root: Path,
    key_id: str,
    source_realm: str = "default",
    target_realm: str = "default",
    target_tag: str = "rbac_private_key",
    delete_source_secret: bool = False,
) -> Dict[str, Any]:
    key_id_norm = str(key_id or "").strip()
    if not key_id_norm:
        raise ValueError("key_id is required")
    source_payload = read_client_key_metadata(source_root)
    source_row = dict(dict(source_payload.get("keys") or {}).get(key_id_norm) or {})
    secret_id = str(source_row.get("private_key_secret_id") or "").strip()
    if not secret_id:
        raise ValueError(f"client key {key_id_norm!r} does not reference a client-realm secret")
    source_store = FileSecretStore(source_root, realm=source_realm)
    private_key_text = str(source_store.get_secret_payload(secret_id) or "")
    if not private_key_text:
        raise ValueError(f"client key {key_id_norm!r} has an empty private-key secret")
    stored = store_private_key_in_realm(
        target_root,
        realm=target_realm,
        key_id=key_id_norm,
        tag=target_tag,
        private_key_text=private_key_text,
        public_key=str(source_row.get("public_key") or ""),
        role=str(source_row.get("role") or ("transport" if target_tag == "transport_private_key" else "admin")),
        auth_method=str(source_row.get("auth_method") or "public_key"),
        key_origin=str(source_row.get("key_origin") or source_row.get("key_source") or "imported"),
        source="client_realm_migration",
        private_key_protection=str(source_row.get("private_key_protection") or "unknown").strip() or "unknown",
    )
    deleted = False
    if delete_source_secret:
        deleted = source_store.delete_secret(secret_id)
    return {
        **stored,
        "source_root": str(Path(source_root).expanduser().resolve()),
        "source_realm": str(source_realm or "default").strip() or "default",
        "target_root": str(Path(target_root).expanduser().resolve()),
        "target_realm": str(target_realm or "default").strip() or "default",
        "source_secret_id": secret_id,
        "deleted_source_secret": deleted,
    }


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

    Private-key passphrase protection is handled by OpenSSH key formatting, not
    by an app-specific encryption envelope.
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
