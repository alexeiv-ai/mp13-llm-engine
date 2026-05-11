"""Versioned secure-state JSON helpers.

The helpers in this module are intentionally importable Python APIs. They do
not depend on the hosting CLI, and they fail closed when encrypted state cannot
be decrypted.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional


SECURE_STATE_KIND = "mp13.secure_state.json"
SECURE_STATE_VERSION = 1
SECURE_STATE_ALGORITHM = "pbkdf2-sha256+hmac-sha256-stream"
SECURE_STATE_ENV_NAMES = ("MP13_SECURE_STATE_KEY", "MP13_HOSTING_SECURE_STATE_KEY")
DEFAULT_KDF_ITERATIONS = 390_000


class SecureStateError(RuntimeError):
    """Base class for secure-state failures."""


class SecureStateLockedError(SecureStateError):
    """Raised when encrypted state exists but no usable key is available."""


class SecureStateFormatError(SecureStateError):
    """Raised when state is malformed or unsupported."""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(value: Any) -> bytes:
    text = str(value or "").strip()
    if not text:
        return b""
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + pad).encode("ascii"))


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            tmp_path.chmod(0o600)
        except Exception:
            pass
        tmp_path.replace(target)
        try:
            target.chmod(0o600)
        except Exception:
            pass
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def resolve_secure_state_key(key: Optional[str | bytes] = None) -> str:
    """Resolve an explicit key or the documented secure-state environment key."""

    if isinstance(key, bytes):
        text = key.decode("utf-8", errors="strict")
    else:
        text = str(key or "")
    if text:
        return text
    for name in SECURE_STATE_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return str(value)
    return ""


def secure_state_key_available(key: Optional[str | bytes] = None) -> bool:
    return bool(resolve_secure_state_key(key))


def is_secure_state_envelope(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        str(payload.get("kind") or "") == SECURE_STATE_KIND
        and int(payload.get("version") or 0) == SECURE_STATE_VERSION
        and isinstance(payload.get("ciphertext"), str)
    )


def _derive_keys(key: str, salt: bytes, *, iterations: int) -> tuple[bytes, bytes]:
    if not key:
        raise SecureStateLockedError("secure_state_key_required")
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        key.encode("utf-8"),
        salt,
        max(100_000, int(iterations or DEFAULT_KDF_ITERATIONS)),
        dklen=64,
    )
    return derived[:32], derived[32:]


def _keystream(key: bytes, nonce: bytes, size: int) -> bytes:
    chunks: list[bytes] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < size:
        chunks.append(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return b"".join(chunks)[:size]


def encrypt_json_payload(
    payload: Dict[str, Any],
    *,
    key: Optional[str | bytes] = None,
    metadata: Optional[Dict[str, Any]] = None,
    kdf_iterations: int = DEFAULT_KDF_ITERATIONS,
) -> Dict[str, Any]:
    """Return an encrypted secure-state envelope for a JSON object."""

    state_key = resolve_secure_state_key(key)
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    enc_key, mac_key = _derive_keys(state_key, salt, iterations=kdf_iterations)
    plaintext = _canonical_json_bytes(dict(payload or {}))
    stream = _keystream(enc_key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    envelope = {
        "kind": SECURE_STATE_KIND,
        "version": SECURE_STATE_VERSION,
        "algorithm": SECURE_STATE_ALGORITHM,
        "created_at": time.time(),
        "kdf": {
            "name": "pbkdf2_hmac_sha256",
            "iterations": max(100_000, int(kdf_iterations or DEFAULT_KDF_ITERATIONS)),
            "salt": _b64e(salt),
        },
        "nonce": _b64e(nonce),
        "ciphertext": _b64e(ciphertext),
        "metadata": dict(metadata or {}),
    }
    mac_payload = {k: v for k, v in envelope.items() if k != "mac"}
    envelope["mac"] = _b64e(hmac.new(mac_key, _canonical_json_bytes(mac_payload), hashlib.sha256).digest())
    return envelope


def decrypt_json_payload(envelope: Dict[str, Any], *, key: Optional[str | bytes] = None) -> Dict[str, Any]:
    """Decrypt a secure-state envelope into a JSON object."""

    if not is_secure_state_envelope(envelope):
        raise SecureStateFormatError("secure_state_envelope_required")
    if str(envelope.get("algorithm") or "") != SECURE_STATE_ALGORITHM:
        raise SecureStateFormatError("unsupported_secure_state_algorithm")
    kdf = dict(envelope.get("kdf") or {})
    salt = _b64d(kdf.get("salt"))
    nonce = _b64d(envelope.get("nonce"))
    ciphertext = _b64d(envelope.get("ciphertext"))
    if not salt or not nonce or not ciphertext:
        raise SecureStateFormatError("malformed_secure_state_envelope")
    enc_key, mac_key = _derive_keys(resolve_secure_state_key(key), salt, iterations=int(kdf.get("iterations") or 0))
    mac_payload = {k: v for k, v in dict(envelope).items() if k != "mac"}
    expected_mac = hmac.new(mac_key, _canonical_json_bytes(mac_payload), hashlib.sha256).digest()
    actual_mac = _b64d(envelope.get("mac"))
    if not actual_mac or not hmac.compare_digest(expected_mac, actual_mac):
        raise SecureStateLockedError("secure_state_authentication_failed")
    stream = _keystream(enc_key, nonce, len(ciphertext))
    plaintext = bytes(a ^ b for a, b in zip(ciphertext, stream))
    try:
        decoded = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise SecureStateFormatError("secure_state_plaintext_decode_failed") from exc
    if not isinstance(decoded, dict):
        raise SecureStateFormatError("secure_state_json_object_required")
    return dict(decoded)


def read_secure_json(
    path: Path,
    *,
    key: Optional[str | bytes] = None,
    allow_plaintext: bool = True,
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return dict(default or {})
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SecureStateFormatError(f"secure_state_json_unreadable:{target}") from exc
    if is_secure_state_envelope(raw):
        return decrypt_json_payload(dict(raw), key=key)
    if not allow_plaintext:
        raise SecureStateFormatError("plaintext_secure_state_disallowed")
    if not isinstance(raw, dict):
        raise SecureStateFormatError("secure_state_json_object_required")
    return dict(raw)


def write_secure_json(
    path: Path,
    payload: Dict[str, Any],
    *,
    encrypt: bool = False,
    key: Optional[str | bytes] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target = Path(path).expanduser().resolve()
    out = encrypt_json_payload(dict(payload or {}), key=key, metadata=metadata) if encrypt else dict(payload or {})
    _atomic_write_json(target, out)
    return secure_state_status(target, key=key)


def secure_state_status(path: Path, *, key: Optional[str | bytes] = None) -> Dict[str, Any]:
    target = Path(path).expanduser().resolve()
    out: Dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
        "state": "missing",
        "encrypted": False,
        "locked": False,
        "key_available": secure_state_key_available(key),
    }
    if not target.exists():
        return out
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        out.update({"state": "unreadable", "error": str(exc)})
        return out
    if is_secure_state_envelope(payload):
        kdf = dict(payload.get("kdf") or {})
        out.update(
            {
                "state": "encrypted",
                "encrypted": True,
                "locked": not secure_state_key_available(key),
                "kind": str(payload.get("kind") or ""),
                "version": int(payload.get("version") or 0),
                "algorithm": str(payload.get("algorithm") or ""),
                "kdf": str(kdf.get("name") or ""),
                "kdf_iterations": int(kdf.get("iterations") or 0),
                "metadata": dict(payload.get("metadata") or {}),
            }
        )
        return out
    out.update({"state": "plaintext", "encrypted": False, "locked": False})
    return out


def encrypt_secure_json_file(
    path: Path,
    *,
    key: Optional[str | bytes] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = read_secure_json(path, key=key, allow_plaintext=True)
    return write_secure_json(path, payload, encrypt=True, key=key, metadata=metadata)


def decrypt_secure_json_file(path: Path, *, key: Optional[str | bytes] = None) -> Dict[str, Any]:
    payload = read_secure_json(path, key=key, allow_plaintext=False)
    return write_secure_json(path, payload, encrypt=False)


def rotate_secure_json_file(
    path: Path,
    *,
    old_key: Optional[str | bytes] = None,
    new_key: Optional[str | bytes] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = read_secure_json(path, key=old_key, allow_plaintext=False)
    return write_secure_json(path, payload, encrypt=True, key=new_key, metadata=metadata)


__all__ = [
    "SECURE_STATE_KIND",
    "SECURE_STATE_VERSION",
    "SECURE_STATE_ALGORITHM",
    "SECURE_STATE_ENV_NAMES",
    "SecureStateError",
    "SecureStateLockedError",
    "SecureStateFormatError",
    "resolve_secure_state_key",
    "secure_state_key_available",
    "is_secure_state_envelope",
    "encrypt_json_payload",
    "decrypt_json_payload",
    "read_secure_json",
    "write_secure_json",
    "secure_state_status",
    "encrypt_secure_json_file",
    "decrypt_secure_json_file",
    "rotate_secure_json_file",
]
