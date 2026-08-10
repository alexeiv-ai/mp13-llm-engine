from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from ..toolbox.host_project_config import ToolboxHostProjectConfiguration
from ..toolbox.identity import require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


UPLOAD_STATE_CONTRACT = "hosting.toolbox.artifact_uploads.v1"
MAX_UPLOADS = 64
MAX_CHUNK_BYTES = 1024 * 1024
UPLOAD_TTL_SECONDS = 15 * 60
_ID_RE = re.compile(r"[A-Za-z0-9_-]{16,128}")


class ToolboxArtifactUploadError(RuntimeError):
    _SUMMARIES = {
        "artifact_upload_invalid": "The artifact upload request is invalid.",
        "artifact_upload_conflict": "The artifact upload request conflicts with existing state.",
        "artifact_upload_not_found": "The artifact upload does not exist.",
        "artifact_upload_expired": "The artifact upload has expired.",
        "artifact_upload_not_open": "The artifact upload is not open.",
        "artifact_upload_chunk_invalid": "The artifact upload chunk is invalid.",
        "artifact_upload_bounds_exceeded": "The artifact upload exceeds configured bounds.",
        "artifact_upload_state_invalid": "The artifact upload state is invalid.",
    }

    def __init__(self, code: str):
        if code not in self._SUMMARIES:
            raise ValueError("artifact_upload_error_code_invalid")
        self.code = code
        self.summary = self._SUMMARIES[code]
        super().__init__(code)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _identity(value: Any, *, label: str, maximum: int = 256) -> str:
    text = str(value or "").strip()
    if not text or len(text.encode("utf-8")) > maximum or any(ord(item) < 32 for item in text):
        raise ToolboxArtifactUploadError("artifact_upload_invalid")
    return text


def _decode_chunk(value: str) -> bytes:
    text = str(value or "").strip()
    if (
        not text
        or len(text) > (MAX_CHUNK_BYTES * 4 + 2) // 3
        or "=" in text
        or not re.fullmatch(r"[A-Za-z0-9_-]+", text)
    ):
        raise ToolboxArtifactUploadError("artifact_upload_chunk_invalid")
    try:
        result = base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))
    except (TypeError, ValueError) as exc:
        raise ToolboxArtifactUploadError("artifact_upload_chunk_invalid") from exc
    if not result or len(result) > MAX_CHUNK_BYTES:
        raise ToolboxArtifactUploadError("artifact_upload_chunk_invalid")
    return result


class AtomicToolboxArtifactUploadRepository:
    """Process-safe untrusted upload staging; it never mutates the verified CAS."""

    def __init__(
        self,
        root: Path,
        *,
        configuration: ToolboxHostProjectConfiguration,
        clock=time.time,
    ) -> None:
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        self.root = Path(root).expanduser().resolve()
        self.state_path = self.root / "uploads.json"
        self.lock_path = self.root / ".uploads.lock"
        self.staging_root = self.root / "staged"
        self.configuration = configuration
        self.clock = clock

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": UPLOAD_STATE_CONTRACT, "uploads": {}}

    def _validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {"contract", "uploads"} or row.get("contract") != UPLOAD_STATE_CONTRACT:
            raise ToolboxArtifactUploadError("artifact_upload_state_invalid")
        if not isinstance(row["uploads"], dict) or len(row["uploads"]) > MAX_UPLOADS:
            raise ToolboxArtifactUploadError("artifact_upload_state_invalid")
        uploads: dict[str, dict[str, Any]] = {}
        for upload_id, raw in row["uploads"].items():
            item = dict(raw or {})
            if not _ID_RE.fullmatch(str(upload_id)) or set(item) != {
                "owner_actor_id", "request_id", "fingerprint", "source_id",
                "config_revision", "source_set_revision", "target", "archive_sha256",
                "total_size", "received_size", "chunks", "state", "created_at_ms",
                "updated_at_ms", "expires_at_ms",
            }:
                raise ToolboxArtifactUploadError("artifact_upload_state_invalid")
            require_digest(item["fingerprint"], label="artifact_upload_fingerprint")
            require_digest(item["archive_sha256"], label="artifact_upload_archive_digest")
            if (
                item["state"] not in {"open", "canceled", "expired", "committing", "committed"}
                or isinstance(item["total_size"], bool)
                or not isinstance(item["total_size"], int)
                or isinstance(item["received_size"], bool)
                or not isinstance(item["received_size"], int)
                or not 0 <= item["received_size"] <= item["total_size"]
                or not isinstance(item["chunks"], list)
            ):
                raise ToolboxArtifactUploadError("artifact_upload_state_invalid")
            expected_offset = 0
            for index, chunk in enumerate(item["chunks"]):
                if not isinstance(chunk, dict) or set(chunk) != {"index", "offset", "size", "sha256"}:
                    raise ToolboxArtifactUploadError("artifact_upload_state_invalid")
                if chunk["index"] != index or chunk["offset"] != expected_offset or chunk["size"] < 1:
                    raise ToolboxArtifactUploadError("artifact_upload_state_invalid")
                require_digest(chunk["sha256"], label="artifact_upload_chunk_digest")
                expected_offset += chunk["size"]
            if expected_offset != item["received_size"]:
                raise ToolboxArtifactUploadError("artifact_upload_state_invalid")
            uploads[str(upload_id)] = item
        return {"contract": UPLOAD_STATE_CONTRACT, "uploads": uploads}

    def _read(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ToolboxArtifactUploadError("artifact_upload_state_invalid") from exc
        return self._validate(payload)

    def _write(self, payload: Mapping[str, Any]) -> None:
        value = self._validate(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, raw = tempfile.mkstemp(prefix=".uploads.", suffix=".tmp", dir=self.root)
        temporary = Path(raw)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_canonical(value))
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_bounded_retries(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _stage_path(self, upload_id: str) -> Path:
        if not _ID_RE.fullmatch(upload_id):
            raise ToolboxArtifactUploadError("artifact_upload_invalid")
        return (self.staging_root / f"{upload_id}.zip.part").resolve()

    @staticmethod
    def _public(upload_id: str, item: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(item)
        row.pop("owner_actor_id", None)
        row.pop("fingerprint", None)
        row.pop("chunks", None)
        return {"upload_id": upload_id, **row}

    def _expire(self, upload_id: str, item: dict[str, Any], *, now_ms: int) -> bool:
        if item["state"] == "open" and now_ms >= item["expires_at_ms"]:
            item["state"] = "expired"
            item["updated_at_ms"] = now_ms
            self._stage_path(upload_id).unlink(missing_ok=True)
            return True
        return False

    def begin(
        self,
        *,
        owner_actor_id: str,
        request_id: str,
        source_id: str,
        total_size: int,
        archive_sha256: str,
    ) -> dict[str, Any]:
        owner = _identity(owner_actor_id, label="owner_actor_id")
        request = _identity(request_id, label="request_id")
        source = _identity(source_id, label="source_id", maximum=128)
        source_config = next(
            (
                item for item in self.configuration.sources
                if item.source_id == source and item.kind == "airgap_store"
            ),
            None,
        )
        if source_config is None:
            raise ToolboxArtifactUploadError("artifact_upload_invalid")
        digest = require_digest(archive_sha256, label="artifact_upload_archive_digest")
        maximum = min(
            source_config.maximum_download_bytes,
            self.configuration.resolution.maximum_bytes,
        )
        if isinstance(total_size, bool) or not isinstance(total_size, int) or not 1 <= total_size <= maximum:
            raise ToolboxArtifactUploadError("artifact_upload_bounds_exceeded")
        fingerprint = "sha256:" + hashlib.sha256(
            _canonical(
                {
                    "owner_actor_id": owner,
                    "request_id": request,
                    "source_id": source,
                    "config_revision": self.configuration.config_revision,
                    "source_set_revision": self.configuration.source_set_revision,
                    "target": self.configuration.target.name,
                    "total_size": total_size,
                    "archive_sha256": digest,
                }
            )
        ).hexdigest()
        now_ms = int(self.clock() * 1000)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            for existing_id, existing in state["uploads"].items():
                self._expire(existing_id, existing, now_ms=now_ms)
                if existing["owner_actor_id"] == owner and existing["request_id"] == request:
                    if existing["fingerprint"] != fingerprint:
                        self._write(state)
                        raise ToolboxArtifactUploadError("artifact_upload_conflict")
                    self._write(state)
                    return self._public(existing_id, existing)
            active = [item for item in state["uploads"].values() if item["state"] == "open"]
            if len(active) >= MAX_UPLOADS:
                raise ToolboxArtifactUploadError("artifact_upload_bounds_exceeded")
            if len(state["uploads"]) >= MAX_UPLOADS:
                removable = sorted(
                    (
                        (item["updated_at_ms"], upload_id)
                        for upload_id, item in state["uploads"].items()
                        if item["state"] != "open"
                    )
                )
                if not removable:
                    raise ToolboxArtifactUploadError("artifact_upload_bounds_exceeded")
                state["uploads"].pop(removable[0][1], None)
            upload_id = f"upload_{secrets.token_urlsafe(24)}"
            item = {
                "owner_actor_id": owner,
                "request_id": request,
                "fingerprint": fingerprint,
                "source_id": source,
                "config_revision": self.configuration.config_revision,
                "source_set_revision": self.configuration.source_set_revision,
                "target": self.configuration.target.name,
                "archive_sha256": digest,
                "total_size": total_size,
                "received_size": 0,
                "chunks": [],
                "state": "open",
                "created_at_ms": now_ms,
                "updated_at_ms": now_ms,
                "expires_at_ms": now_ms + UPLOAD_TTL_SECONDS * 1000,
            }
            self.staging_root.mkdir(parents=True, exist_ok=True)
            self._stage_path(upload_id).touch(exist_ok=False)
            state["uploads"][upload_id] = item
            self._write(state)
            return self._public(upload_id, item)

    def append_chunk(
        self,
        *,
        owner_actor_id: str,
        upload_id: str,
        chunk_index: int,
        offset: int,
        chunk_base64url: str,
    ) -> dict[str, Any]:
        owner = _identity(owner_actor_id, label="owner_actor_id")
        upload = _identity(upload_id, label="upload_id", maximum=128)
        content = _decode_chunk(chunk_base64url)
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        now_ms = int(self.clock() * 1000)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            item = state["uploads"].get(upload)
            if item is None or item["owner_actor_id"] != owner:
                raise ToolboxArtifactUploadError("artifact_upload_not_found")
            if self._expire(upload, item, now_ms=now_ms):
                self._write(state)
                raise ToolboxArtifactUploadError("artifact_upload_expired")
            if item["state"] != "open":
                raise ToolboxArtifactUploadError("artifact_upload_not_open")
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
                raise ToolboxArtifactUploadError("artifact_upload_chunk_invalid")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ToolboxArtifactUploadError("artifact_upload_chunk_invalid")
            if chunk_index < len(item["chunks"]):
                previous = item["chunks"][chunk_index]
                if previous == {
                    "index": chunk_index,
                    "offset": offset,
                    "size": len(content),
                    "sha256": digest,
                }:
                    return self._public(upload, item)
                raise ToolboxArtifactUploadError("artifact_upload_conflict")
            if (
                chunk_index != len(item["chunks"])
                or offset != item["received_size"]
                or offset + len(content) > item["total_size"]
            ):
                raise ToolboxArtifactUploadError("artifact_upload_chunk_invalid")
            path = self._stage_path(upload)
            try:
                actual_size = path.stat().st_size
                if actual_size < item["received_size"]:
                    raise ToolboxArtifactUploadError("artifact_upload_state_invalid")
                if actual_size > item["received_size"]:
                    with path.open("r+b") as handle:
                        handle.truncate(item["received_size"])
                with path.open("ab") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise ToolboxArtifactUploadError("artifact_upload_state_invalid") from exc
            item["chunks"].append(
                {"index": chunk_index, "offset": offset, "size": len(content), "sha256": digest}
            )
            item["received_size"] += len(content)
            item["updated_at_ms"] = now_ms
            self._write(state)
            return self._public(upload, item)

    def cancel(self, *, owner_actor_id: str, upload_id: str) -> dict[str, Any]:
        owner = _identity(owner_actor_id, label="owner_actor_id")
        upload = _identity(upload_id, label="upload_id", maximum=128)
        now_ms = int(self.clock() * 1000)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            item = state["uploads"].get(upload)
            if item is None or item["owner_actor_id"] != owner:
                raise ToolboxArtifactUploadError("artifact_upload_not_found")
            if item["state"] == "open":
                item["state"] = "canceled"
                item["updated_at_ms"] = now_ms
                self._stage_path(upload).unlink(missing_ok=True)
                self._write(state)
            return self._public(upload, item)

    def status(self, *, owner_actor_id: str, upload_id: str) -> dict[str, Any]:
        owner = _identity(owner_actor_id, label="owner_actor_id")
        upload = _identity(upload_id, label="upload_id", maximum=128)
        now_ms = int(self.clock() * 1000)
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read()
            item = state["uploads"].get(upload)
            if item is None or item["owner_actor_id"] != owner:
                raise ToolboxArtifactUploadError("artifact_upload_not_found")
            if self._expire(upload, item, now_ms=now_ms):
                self._write(state)
            return self._public(upload, item)


__all__ = [
    "MAX_CHUNK_BYTES",
    "UPLOAD_STATE_CONTRACT",
    "AtomicToolboxArtifactUploadRepository",
    "ToolboxArtifactUploadError",
]
