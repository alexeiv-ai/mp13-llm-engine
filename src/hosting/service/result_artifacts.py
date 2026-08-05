"""Authorized, bounded storage for hosted-operation terminal results."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from ..operation_contract import HostedResultRef, canonical_json_bytes


MAX_TERMINAL_RESULT_ARTIFACT_BYTES = 16 * 1024 * 1024
_ARTIFACT_ID = re.compile(r"^result_[A-Za-z0-9_-]{16,128}$")


class ResultArtifactError(RuntimeError):
    pass


class TerminalResultArtifactStore:
    """Stores opaque JSON result artifacts without exposing host paths."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = MAX_TERMINAL_RESULT_ARTIFACT_BYTES,
        ttl_seconds: float = 7 * 24 * 3600,
        clock: Any = time.time,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_bytes = max(1, min(int(max_bytes), MAX_TERMINAL_RESULT_ARTIFACT_BYTES))
        self.ttl_ms = max(1, int(float(ttl_seconds) * 1000))
        self._clock = clock
        self._lock = threading.RLock()

    def _now_ms(self) -> int:
        return max(0, int(float(self._clock()) * 1000))

    def _paths(self, artifact_id: str) -> tuple[Path, Path]:
        if _ARTIFACT_ID.fullmatch(artifact_id) is None:
            raise ResultArtifactError("result_artifact_id_invalid")
        return self.root / f"{artifact_id}.json", self.root / f"{artifact_id}.meta.json"

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def put(self, *, owner_actor_id: str, operation_id: str, content: bytes) -> HostedResultRef:
        payload = bytes(content)
        if len(payload) > self.max_bytes:
            raise ResultArtifactError("result_artifact_too_large")
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        artifact_id = f"result_{secrets.token_urlsafe(24)}"
        expires_at_ms = self._now_ms() + self.ttl_ms
        data_path, meta_path = self._paths(artifact_id)
        metadata = {
            "artifact_id": artifact_id,
            "owner_actor_id": str(owner_actor_id),
            "operation_id": str(operation_id),
            "digest": digest,
            "size_bytes": len(payload),
            "media_type": "application/json",
            "expires_at_ms": expires_at_ms,
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._atomic_write(data_path, payload)
            try:
                self._atomic_write(meta_path, canonical_json_bytes(metadata))
            except Exception:
                data_path.unlink(missing_ok=True)
                raise
        return HostedResultRef(
            artifact_id=artifact_id,
            digest=digest,
            size_bytes=len(payload),
            media_type="application/json",
            expires_at_ms=expires_at_ms,
        )

    def read(
        self,
        *,
        ref: HostedResultRef | Mapping[str, Any],
        owner_actor_id: str,
        operation_id: str,
    ) -> bytes:
        result_ref = ref if isinstance(ref, HostedResultRef) else HostedResultRef.from_dict(ref)
        data_path, meta_path = self._paths(result_ref.artifact_id)
        with self._lock:
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ResultArtifactError("result_artifact_missing") from exc
            except Exception as exc:
                raise ResultArtifactError("result_artifact_metadata_invalid") from exc
            expected = result_ref.to_dict()
            expected.pop("contract", None)
            if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
                raise ResultArtifactError("result_artifact_metadata_mismatch")
            if metadata.get("owner_actor_id") != str(owner_actor_id) or metadata.get("operation_id") != str(operation_id):
                raise ResultArtifactError("result_artifact_unauthorized")
            if self._now_ms() >= int(metadata.get("expires_at_ms") or 0):
                self.delete(result_ref.artifact_id)
                raise ResultArtifactError("result_artifact_expired")
            try:
                with data_path.open("rb") as handle:
                    payload = handle.read(self.max_bytes + 1)
            except FileNotFoundError as exc:
                raise ResultArtifactError("result_artifact_missing") from exc
            if len(payload) > self.max_bytes or len(payload) != result_ref.size_bytes:
                raise ResultArtifactError("result_artifact_size_mismatch")
            digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
            if digest != result_ref.digest:
                raise ResultArtifactError("result_artifact_digest_mismatch")
            return payload

    def delete(self, artifact_id: str) -> None:
        data_path, meta_path = self._paths(artifact_id)
        with self._lock:
            data_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    def prune(self, *, live_artifact_ids: Iterable[str]) -> None:
        live = set(live_artifact_ids)
        with self._lock:
            if not self.root.exists():
                return
            for meta_path in self.root.glob("result_*.meta.json"):
                artifact_id = meta_path.name[: -len(".meta.json")]
                expired = False
                try:
                    metadata: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
                    expired = self._now_ms() >= int(metadata.get("expires_at_ms") or 0)
                except Exception:
                    expired = True
                if artifact_id not in live or expired:
                    self.delete(artifact_id)


__all__ = [
    "MAX_TERMINAL_RESULT_ARTIFACT_BYTES",
    "ResultArtifactError",
    "TerminalResultArtifactStore",
]
