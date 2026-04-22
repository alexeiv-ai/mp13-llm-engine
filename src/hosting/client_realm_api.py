"""Stable client-realm key custody API.

This module is the integration surface for backend/client code. It deliberately
returns structured dictionaries and does not depend on CLI output.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .client_realm import (
    append_client_audit_event,
    create_private_key_handoff_text,
    normalize_pasted_private_key,
    read_client_key_metadata,
    store_private_key_handoff_in_realm,
    store_private_key_in_realm,
)
from .transport_bootstrap import _protect_openssh_private_key


@dataclass(frozen=True)
class ClientRealmKeyRequest:
    client_realm_root: Path
    key_id: str
    realm: str = "default"
    tag: str = "rbac_private_key"
    private_key_text: str = ""
    public_key_text: str = ""
    private_key_file: Optional[Path] = None
    public_key_file: Optional[Path] = None
    passphrase: str = ""
    role: str = ""


def _generate_keypair(*, key_id: str, passphrase: str = "") -> tuple[str, str]:
    tmpdir = Path(tempfile.mkdtemp(prefix="hosting_keygen_")).resolve()
    try:
        private_path = (tmpdir / f"{key_id}_ed25519").resolve()
        proc = subprocess.run(  # noqa: S603
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-C",
                str(key_id or "client-key"),
                "-f",
                str(private_path),
                "-N",
                str(passphrase or ""),
            ],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if int(proc.returncode) != 0:
            raise RuntimeError(str(proc.stderr or "").strip() or "ssh-keygen failed")
        public_path = Path(str(private_path) + ".pub")
        return private_path.read_text(encoding="utf-8").strip(), public_path.read_text(encoding="utf-8").strip()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _derive_public_key(private_key_text: str) -> str:
    tmpdir = Path(tempfile.mkdtemp(prefix="hosting_pubderive_")).resolve()
    try:
        private_path = (tmpdir / "private_key").resolve()
        private_path.write_text(str(private_key_text or "").strip() + "\n", encoding="utf-8")
        try:
            private_path.chmod(0o600)
        except Exception:
            pass
        proc = subprocess.run(  # noqa: S603
            ["ssh-keygen", "-y", "-f", str(private_path)],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if int(proc.returncode) != 0:
            raise RuntimeError(str(proc.stderr or "").strip() or "ssh-keygen -y failed")
        public_key = str(proc.stdout or "").strip()
        if not public_key:
            raise RuntimeError("ssh-keygen -y returned empty public key")
        return public_key
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _request_dict(request: ClientRealmKeyRequest | Dict[str, Any]) -> Dict[str, Any]:
    return asdict(request) if isinstance(request, ClientRealmKeyRequest) else dict(request or {})


def list_client_realm_keys(request: Dict[str, Any] | None = None, *, client_realm_root: Optional[Path] = None) -> Dict[str, Any]:
    data = dict(request or {})
    root_value = client_realm_root or data.get("client_realm_root")
    if not root_value:
        raise ValueError("client_realm_root is required")
    root = Path(root_value).expanduser().resolve()
    payload = read_client_key_metadata(root)
    return {"status": "ok", "client_realm_root": str(root), "keys": dict(payload.get("keys") or {})}


def generate_client_realm_key(request: ClientRealmKeyRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _request_dict(request)
    root = Path(data.get("client_realm_root") or "").expanduser().resolve()
    key_id = str(data.get("key_id") or "").strip()
    if not key_id:
        raise ValueError("key_id is required")
    tag = str(data.get("tag") or "rbac_private_key").strip() or "rbac_private_key"
    passphrase = str(data.get("passphrase") or "")
    private_key, public_key = _generate_keypair(key_id=key_id, passphrase=passphrase)
    role = str(data.get("role") or ("transport" if tag == "transport_private_key" else "admin")).strip()
    stored = store_private_key_in_realm(
        root,
        realm=str(data.get("realm") or "default").strip() or "default",
        key_id=key_id,
        tag=tag,
        private_key_text=private_key,
        public_key=public_key,
        role=role,
        auth_method="public_key",
        key_origin="generated",
        source="client_realm_api_generate",
        private_key_protection="openssh_passphrase" if passphrase else "none",
    )
    audit_path = append_client_audit_event(
        root,
        event_type="client_key_generate",
        realm=str(data.get("realm") or "default").strip() or "default",
        payload={"key_id": key_id, "tag": tag, "secret_id": stored.get("secret_id")},
    )
    return {**stored, "audit_path": str(audit_path)}


def import_client_realm_key(request: ClientRealmKeyRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _request_dict(request)
    root = Path(data.get("client_realm_root") or "").expanduser().resolve()
    key_id = str(data.get("key_id") or "").strip()
    if not key_id:
        raise ValueError("key_id is required")
    private_key_text = normalize_pasted_private_key(str(data.get("private_key_text") or ""))
    private_key_file = data.get("private_key_file")
    if not private_key_text and private_key_file:
        private_key_text = normalize_pasted_private_key(Path(private_key_file).expanduser().resolve().read_text(encoding="utf-8"))
    if not private_key_text:
        raise ValueError("private_key_text or private_key_file is required")
    public_key_text = str(data.get("public_key_text") or "").strip()
    public_key_file = data.get("public_key_file")
    if not public_key_text and public_key_file:
        public_key_text = Path(public_key_file).expanduser().resolve().read_text(encoding="utf-8").strip()
    if not public_key_text:
        public_key_text = _derive_public_key(private_key_text)
    passphrase = str(data.get("passphrase") or "")
    if passphrase:
        private_key_text = _protect_openssh_private_key(private_key_text, new_passphrase=passphrase)
    tag = str(data.get("tag") or "rbac_private_key").strip() or "rbac_private_key"
    role = str(data.get("role") or ("transport" if tag == "transport_private_key" else "admin")).strip()
    stored = store_private_key_in_realm(
        root,
        realm=str(data.get("realm") or "default").strip() or "default",
        key_id=key_id,
        tag=tag,
        private_key_text=private_key_text,
        public_key=public_key_text,
        role=role,
        auth_method="public_key",
        key_origin="imported",
        source="client_realm_api_import",
        private_key_protection="openssh_passphrase" if passphrase else "none",
    )
    audit_path = append_client_audit_event(
        root,
        event_type="client_key_import",
        realm=str(data.get("realm") or "default").strip() or "default",
        payload={"key_id": key_id, "tag": tag, "secret_id": stored.get("secret_id")},
    )
    return {**stored, "audit_path": str(audit_path)}


def create_client_realm_key_handoff(request: ClientRealmKeyRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _request_dict(request)
    return create_private_key_handoff_text(
        Path(data.get("client_realm_root") or "").expanduser().resolve(),
        key_id=str(data.get("key_id") or "").strip(),
        realm=str(data.get("realm") or "default").strip() or "default",
        password=str(data.get("passphrase") or "") or None,
    )


def import_client_realm_key_handoff(request: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(request or {})
    return store_private_key_handoff_in_realm(
        Path(data.get("client_realm_root") or "").expanduser().resolve(),
        handoff_text=data.get("handoff_text") or data.get("handoff") or "",
        realm=str(data.get("realm") or "default").strip() or "default",
        tag=str(data.get("tag") or "").strip(),
    )


__all__ = [
    "ClientRealmKeyRequest",
    "list_client_realm_keys",
    "generate_client_realm_key",
    "import_client_realm_key",
    "create_client_realm_key_handoff",
    "import_client_realm_key_handoff",
]
