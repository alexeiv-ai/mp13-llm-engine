"""
Client-side hosting realm helpers and file-backed secret records.

This module intentionally starts with a simple file backend so later work can
swap in OS-specific secret storage without changing higher-level workflows.
"""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mp13_engine.mp13_config_paths import get_default_config_dir


CLIENT_REALM_ROOT_SUBDIR = "hosting_client"
VALID_SECRET_RECORD_ENCRYPTION = {"none", "password_v1"}


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
        "known_hosts": realm_root / "known_hosts",
        "audit": realm_root / "audit",
        "profiles": realm_root / "profiles",
    }


def ensure_client_realm_dirs(root: Path) -> Dict[str, Path]:
    layout = client_realm_layout(root)
    for key in ("root", "keyring", "secrets", "known_hosts", "audit", "profiles"):
        layout[key].mkdir(parents=True, exist_ok=True)
    return layout


def secret_record_path(root: Path, secret_id: str) -> Path:
    sid = str(secret_id or "").strip()
    if not sid:
        raise ValueError("secret_id is required")
    return (client_realm_layout(root)["secrets"] / f"{sid}.json").resolve()


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
        if enc != "none":
            raise NotImplementedError("password-encrypted secret records are not implemented yet")
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
            payload=str(payload or ""),
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

    def get_secret_payload(self, secret_id: str) -> Optional[str]:
        record = self.get_secret_record(secret_id)
        return None if record is None else str(record.payload)

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

    def reencrypt_secret(self, secret_id: str, *, encryption: str) -> SecretRecord:
        record = self.get_secret_record(secret_id)
        if record is None:
            raise ValueError("secret_id is not present")
        enc = str(encryption or "").strip().lower()
        if not enc:
            raise ValueError("encryption is required")
        if enc == record.encryption:
            return record
        if enc != "none":
            raise NotImplementedError("password-encrypted secret records are not implemented yet")
        return self.put_secret(
            tag=record.tag,
            payload=record.payload,
            secret_id=record.secret_id,
            metadata=dict(record.metadata or {}),
            encryption=enc,
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


def iter_secret_ids(root: Path, *, realm: str = "default", tag: Optional[str] = None) -> Iterable[str]:
    store = FileSecretStore(root, realm=realm)
    for record in store.list_records(tag=tag):
        yield record.secret_id
