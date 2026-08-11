"""Stable client-realm key custody API.

This module is the integration surface for backend/client code. It deliberately
returns structured dictionaries and does not depend on CLI output.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ._process_utils import hidden_subprocess_kwargs
from .client_realm import (
    FileSecretStore,
    append_client_audit_event,
    create_private_key_handoff_text,
    delete_client_key_from_realm,
    normalize_pasted_private_key,
    read_client_key_metadata,
    require_client_realm_private_key_path,
    store_private_key_handoff_in_realm,
    store_private_key_in_realm,
)
from .transport_bootstrap import _protect_openssh_private_key, _protect_windows_private_key_path


def _coerce_signature_ssh(result: Any, *, expected_challenge_id: str = "") -> str:
    if isinstance(result, dict):
        signer_challenge_id = str(result.get("challenge_id") or "").strip()
        expected = str(expected_challenge_id or "").strip()
        if expected and signer_challenge_id and signer_challenge_id != expected:
            raise ValueError("signer returned signature for a different challenge_id")
        return str(result.get("signature_ssh") or result.get("signature") or "").strip()
    return str(result or "").strip()


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
    export_path: Optional[Path] = None


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
            **hidden_subprocess_kwargs(),
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
        _protect_windows_private_key_path(tmpdir)
        private_path = (tmpdir / "private_key").resolve()
        private_path.write_text(str(private_key_text or "").strip() + "\n", encoding="utf-8")
        _protect_windows_private_key_path(private_path)
        proc = subprocess.run(  # noqa: S603
            ["ssh-keygen", "-y", "-f", str(private_path)],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
            **hidden_subprocess_kwargs(),
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


def export_client_realm_key(request: ClientRealmKeyRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _request_dict(request)
    root = Path(data.get("client_realm_root") or "").expanduser().resolve()
    key_id = str(data.get("key_id") or "").strip()
    if not key_id:
        raise ValueError("key_id is required")
    realm = str(data.get("realm") or "default").strip() or "default"
    payload = read_client_key_metadata(root)
    row = dict(dict(payload.get("keys") or {}).get(key_id) or {})
    secret_id = str(row.get("private_key_secret_id") or "").strip()
    if not secret_id:
        raise ValueError(f"client key {key_id!r} does not reference a client-realm secret")
    store = FileSecretStore(root, realm=realm)
    private_key_text = str(store.get_secret_payload(secret_id, password=str(data.get("passphrase") or "") or None) or "")
    if not private_key_text:
        raise ValueError(f"client key {key_id!r} has an empty private-key secret")
    export_path_raw = str(data.get("export_path") or "").strip()
    if export_path_raw:
        export_path = require_client_realm_private_key_path(root, Path(export_path_raw))
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(private_key_text, encoding="utf-8")
        try:
            export_path.chmod(0o600)
        except Exception:
            pass
        private_key_output = None
    else:
        export_path = None
        private_key_output = private_key_text
    audit_path = append_client_audit_event(
        root,
        event_type="client_key_export",
        realm=realm,
        payload={
            "key_id": key_id,
            "secret_id": secret_id,
            "export_path": str(export_path) if export_path else None,
            "returned_in_result": export_path is None,
        },
    )
    result: Dict[str, Any] = {
        "status": "ok",
        "client_realm_root": str(root),
        "realm": realm,
        "key_id": key_id,
        "secret_id": secret_id,
        "export_path": str(export_path) if export_path else None,
        "returned_in_result": export_path is None,
        "audit_path": str(audit_path),
    }
    if private_key_output is not None:
        result["private_key"] = private_key_output
    return result


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


def delete_client_realm_key(request: ClientRealmKeyRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _request_dict(request)
    root = Path(data.get("client_realm_root") or "").expanduser().resolve()
    key_id = str(data.get("key_id") or "").strip()
    if not key_id:
        raise ValueError("key_id is required")
    realm = str(data.get("realm") or "default").strip() or "default"
    deleted = delete_client_key_from_realm(root, key_id=key_id, realm=realm)
    audit_path = append_client_audit_event(
        root,
        event_type="client_key_deleted",
        realm=realm,
        payload={
            "key_id": key_id,
            "secret_id": deleted.get("secret_id"),
            "deleted_secret": bool(deleted.get("deleted_secret")),
            "deleted_export_file": bool(deleted.get("deleted_export_file")),
            "deleted_export_path": deleted.get("deleted_export_path"),
        },
    )
    return {**deleted, "audit_path": str(audit_path)}


def begin_client_key_authentication(
    client: Any,
    *,
    key_id: str,
    scope: str = "control",
    ttl_seconds: int = 120,
    config_paths: Optional[list[str]] = None,
    engine_ids: Optional[list[str]] = None,
    bind_to_ssh: bool = True,
) -> Dict[str, Any]:
    """Request a daemon auth challenge without performing any UI or signing work."""
    key_id_norm = str(key_id or "").strip()
    if not key_id_norm:
        raise ValueError("key_id is required")
    challenge = client.auth_begin_challenge(
        key_id=key_id_norm,
        scope=str(scope or "control").strip() or "control",
        ttl_seconds=int(ttl_seconds or 120),
        config_paths=list(config_paths or []),
        engine_ids=list(engine_ids or []),
        bind_to_ssh=bool(bind_to_ssh),
    )
    out = dict(challenge or {})
    challenge_id = str(out.get("challenge_id") or "").strip()
    challenge_text = str(out.get("challenge") or out.get("challenge_text") or "")
    if not challenge_id or not challenge_text:
        raise RuntimeError("daemon did not return a usable auth challenge")
    out["challenge_id"] = challenge_id
    out["challenge"] = challenge_text
    out["challenge_text"] = challenge_text
    out.setdefault("key_id", key_id_norm)
    out.setdefault("scope", str(scope or "control").strip() or "control")
    return out


def sign_client_auth_challenge_with_private_key(
    *,
    private_key_text: str,
    challenge_text: str,
    namespace: str = "engine-host-auth",
    timeout_seconds: float = 30.0,
) -> str:
    """
    Sign a daemon challenge with an unencrypted OpenSSH private key.

    This helper is deliberately non-interactive: it does not call input(), does
    not attach stdin, and disables SSH_ASKPASS. GUI clients that need passphrase
    prompts should use begin_client_key_authentication(), sign in their UI/key
    layer, then call complete_client_key_authentication().
    """
    private_key = str(private_key_text or "").strip()
    challenge = str(challenge_text or "")
    if not private_key:
        raise ValueError("private_key_text is required")
    if not challenge:
        raise ValueError("challenge_text is required")
    ns = str(namespace or "").strip()
    if not ns:
        raise ValueError("namespace is required")
    tmpdir = Path(tempfile.mkdtemp(prefix="host_auth_")).resolve()
    try:
        _protect_windows_private_key_path(tmpdir)

        pk_file = tmpdir / "private_key"
        pk_file.write_text(private_key + "\n", encoding="utf-8")
        _protect_windows_private_key_path(pk_file)

        chal_file = tmpdir / "challenge.txt"
        chal_file.write_text(challenge, encoding="utf-8")
        _protect_windows_private_key_path(chal_file)

        proc = subprocess.run(  # noqa: S603
            ["ssh-keygen", "-Y", "sign", "-f", str(pk_file), "-n", ns, str(chal_file)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=float(timeout_seconds or 30.0),
            check=False,
            env={**os.environ, "SSH_ASKPASS_REQUIRE": "never"},
            **hidden_subprocess_kwargs(),
        )
        if int(proc.returncode) != 0:
            err = str(proc.stderr or "").strip()
            raise RuntimeError(f"ssh-keygen failed to sign challenge: {err or f'exit {proc.returncode}'}")

        sig_file = tmpdir / "challenge.txt.sig"
        if not sig_file.exists():
            raise RuntimeError("ssh-keygen did not create a signature file")

        signature = sig_file.read_text(encoding="utf-8").strip()
        if not signature:
            raise RuntimeError("ssh-keygen returned an empty signature")
        return signature
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def complete_client_key_authentication(
    client: Any,
    *,
    challenge_id: str,
    signature_ssh: str,
    adopt: bool = True,
) -> Dict[str, Any]:
    """Complete daemon challenge authentication with a caller-provided signature."""
    cid = str(challenge_id or "").strip()
    sig = str(signature_ssh or "").strip()
    if not cid:
        raise ValueError("challenge_id is required")
    if not sig:
        raise ValueError("signature_ssh is required")
    try:
        result = client.auth_complete_challenge(challenge_id=cid, signature_ssh=sig, adopt=bool(adopt))
    except TypeError:
        result = client.auth_complete_challenge(challenge_id=cid, signature_ssh=sig)
    return dict(result or {})


def authenticate_client_with_key(
    client: Any,
    key_id: str,
    private_key_text: str = "",
    scope: str = "control",
    *,
    signer: Optional[Callable[[Dict[str, Any]], str]] = None,
    signature_ssh: str = "",
    ttl_seconds: int = 120,
    config_paths: Optional[list[str]] = None,
    engine_ids: Optional[list[str]] = None,
    bind_to_ssh: bool = True,
    adopt: bool = True,
    namespace: str = "engine-host-auth",
    sign_timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """
    Orchestrate daemon public-key authentication and return the complete session
    result, including token, identity, role, authentication method, and scope.

    GUI clients should usually pass a signer callback. The callback receives the
    challenge dictionary from begin_client_key_authentication() and returns an
    OpenSSH armored signature. Headless callers may pass unencrypted
    private_key_text to use the non-interactive ssh-keygen signer.
    """
    ensure_session = getattr(client, "ensure_public_key_session", None)
    if callable(ensure_session):
        ensured = ensure_session(
            key_id=key_id,
            scope=scope,
            signer=signer,
            private_key_text=private_key_text,
            signature_ssh=signature_ssh,
            ttl_seconds=ttl_seconds,
            config_paths=config_paths,
            engine_ids=engine_ids,
            bind_to_ssh=bind_to_ssh,
            adopt=adopt,
            namespace=namespace,
            sign_timeout_seconds=sign_timeout_seconds,
        )
        if isinstance(ensured, dict):
            result = dict(ensured)
        else:
            # Compatibility with custom/older channel implementations.
            result = {
                "status": "ok",
                "token": str(ensured or "").strip(),
                "key_id": str(key_id or "").strip(),
                "auth_method": "public_key",
                "scope": str(scope or "control").strip().lower() or "control",
            }
        if not str(result.get("token") or "").strip():
            raise RuntimeError("authentication failed: no token returned")
        return result
    challenge = begin_client_key_authentication(
        client,
        key_id=key_id,
        scope=scope,
        ttl_seconds=ttl_seconds,
        config_paths=config_paths,
        engine_ids=engine_ids,
        bind_to_ssh=bind_to_ssh,
    )
    signature = str(signature_ssh or "").strip()
    if not signature and signer is not None:
        signature = _coerce_signature_ssh(
            signer(dict(challenge)),
            expected_challenge_id=str(challenge.get("challenge_id") or ""),
        )
    if not signature and private_key_text:
        signature = sign_client_auth_challenge_with_private_key(
            private_key_text=private_key_text,
            challenge_text=str(challenge["challenge_text"]),
            namespace=namespace,
            timeout_seconds=sign_timeout_seconds,
        )
    if not signature:
        raise ValueError("signature_ssh, signer, or private_key_text is required")

    result = complete_client_key_authentication(
        client,
        challenge_id=str(challenge["challenge_id"]),
        signature_ssh=signature,
        adopt=adopt,
    )
    token = str(result.get("token") or "").strip()
    if not token:
        raise RuntimeError("authentication failed: no token returned")
    return dict(result)


__all__ = [
    "ClientRealmKeyRequest",
    "list_client_realm_keys",
    "generate_client_realm_key",
    "import_client_realm_key",
    "export_client_realm_key",
    "create_client_realm_key_handoff",
    "import_client_realm_key_handoff",
    "delete_client_realm_key",
    "begin_client_key_authentication",
    "sign_client_auth_challenge_with_private_key",
    "complete_client_key_authentication",
    "authenticate_client_with_key",
]
