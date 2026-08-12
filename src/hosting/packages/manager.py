"""Authorized content-addressed package ingress and lock persistence."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .contracts import PackageLock, PackagePolicy, PackageSource, PackageVerifier, require_digest
from ..service.operation_repository import _exclusive_process_file_lock


_UPLOAD_ID = re.compile(r"upload_[A-Za-z0-9_-]{16,128}")
_REQUEST_ID = re.compile(r"[\x21-\x7e]{1,256}")
MAX_CHUNK_BYTES = 1024 * 1024
MAX_ACTIVE_UPLOADS = 64
UPLOAD_TTL_MS = 15 * 60 * 1000


class PackageError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


class PackageArtifactManager:
    def __init__(
        self,
        *,
        artifact_root: Path,
        lock_root: Path,
        scratch_root: Path,
        sources: Mapping[str, PackageSource],
        credentials: Mapping[str, Any],
        policy: PackagePolicy,
        configuration_revision: str,
        verifier: Optional[PackageVerifier] = None,
        clock=time.time,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.lock_root = Path(lock_root).resolve()
        self.scratch_root = (Path(scratch_root).resolve() / "package-uploads")
        self.sources = dict(sources)
        self._credentials = dict(credentials)
        self.policy = policy
        self.configuration_revision = require_digest(configuration_revision, "configuration_revision")
        self.verifier = verifier
        self.clock = clock
        self._lock = threading.RLock()
        self._state_path = self.scratch_root / "uploads.json"
        self._process_lock_path = self.scratch_root / ".uploads.lock"
        self._audit_path = Path(scratch_root).resolve().parent / "audit" / "package_events.jsonl"

    @contextmanager
    def _locked(self):
        with self._lock:
            with _exclusive_process_file_lock(self._process_lock_path):
                yield

    def _read(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {"contract": "hosting.package_upload_state.v1", "uploads": {}}
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PackageError("package_upload_state_invalid") from exc
        if not isinstance(value, dict) or value.get("contract") != "hosting.package_upload_state.v1" or not isinstance(value.get("uploads"), dict):
            raise PackageError("package_upload_state_invalid")
        return value

    def _write(self, value: Mapping[str, Any]) -> None:
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        temp = self._state_path.with_name(
            f".{self._state_path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        )
        descriptor = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(dict(value), handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self._state_path)
        finally:
            temp.unlink(missing_ok=True)

    def _part(self, upload_id: str) -> Path:
        if not _UPLOAD_ID.fullmatch(str(upload_id)):
            raise PackageError("package_upload_invalid")
        return self.scratch_root / f"{upload_id}.part"

    def _audit(self, *, event: str, actor_id: str, request_id: str, result: str, artifact_id: Optional[str] = None) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "contract": "hosting.package_audit_event.v1",
            "event_id": secrets.token_urlsafe(18),
            "event": event,
            "actor_id": str(actor_id),
            "request_id": str(request_id),
            "result": result,
            "artifact_id": artifact_id,
            "configuration_revision": self.configuration_revision,
            "timestamp_ms": int(self.clock() * 1000),
        }
        with self._audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _public(upload_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        result = {
            "upload_id": upload_id,
            "state": row["state"],
            "received_bytes": row["received_bytes"],
            "next_chunk_index": len(row["chunks"]),
            "expires_at_ms": row["expires_at_ms"],
            "configuration_revision": row["configuration_revision"],
        }
        if row.get("computed_digest"):
            result["computed_digest"] = row["computed_digest"]
        if row.get("result"):
            result["result"] = dict(row["result"])
        return result

    def _owned(self, state: dict[str, Any], upload_id: str, actor_id: str) -> dict[str, Any]:
        row = state["uploads"].get(upload_id)
        if not isinstance(row, dict) or row.get("actor_id") != actor_id:
            raise PackageError("package_upload_not_found")
        if row["state"] == "open" and int(self.clock() * 1000) >= row["expires_at_ms"]:
            row["state"] = "expired"
            self._part(upload_id).unlink(missing_ok=True)
            self._write(state)
            raise PackageError("package_upload_expired")
        return row

    def begin(
        self,
        *,
        actor_id: str,
        source_id: str,
        total_size: int,
        expected_digest: Optional[str],
        request_id: str,
    ) -> dict[str, Any]:
        if not _REQUEST_ID.fullmatch(str(request_id or "")):
            raise PackageError("package_request_invalid")
        source = self.sources.get(str(source_id))
        if source is None or not source.enabled or source.source_id not in self.policy.allowed_source_ids:
            raise PackageError("package_source_unavailable")
        if source.credential_ref is not None and source.credential_ref not in self._credentials:
            raise PackageError("package_credential_unavailable")
        if isinstance(total_size, bool) or not isinstance(total_size, int) or not 1 <= total_size <= self.policy.max_artifact_bytes:
            raise PackageError("package_upload_bounds_exceeded")
        digest = require_digest(expected_digest, "expected_digest") if expected_digest else None
        with self._locked():
            state = self._read()
            for upload_id, row in state["uploads"].items():
                if row.get("actor_id") == actor_id and row.get("request_id") == request_id:
                    if (row["source_id"], row["total_size"], row.get("expected_digest")) != (source.source_id, total_size, digest):
                        raise PackageError("package_upload_conflict")
                    return {**self._public(upload_id, row), "chunk_size": MAX_CHUNK_BYTES}
            active = sum(1 for row in state["uploads"].values() if row.get("state") == "open")
            if active >= MAX_ACTIVE_UPLOADS:
                raise PackageError("package_upload_capacity")
            upload_id = "upload_" + secrets.token_urlsafe(24)
            now = int(self.clock() * 1000)
            row = {
                "actor_id": str(actor_id),
                "request_id": request_id,
                "source_id": source.source_id,
                "total_size": total_size,
                "expected_digest": digest,
                "received_bytes": 0,
                "chunks": [],
                "state": "open",
                "expires_at_ms": now + UPLOAD_TTL_MS,
                "configuration_revision": self.configuration_revision,
                "computed_digest": None,
                "commit_request_id": None,
                "result": None,
            }
            self.scratch_root.mkdir(parents=True, exist_ok=True)
            self._part(upload_id).touch(exist_ok=False)
            state["uploads"][upload_id] = row
            self._write(state)
            return {**self._public(upload_id, row), "chunk_size": MAX_CHUNK_BYTES}

    def chunk(self, *, actor_id: str, upload_id: str, chunk_index: int, offset: int, chunk_base64url: str) -> dict[str, Any]:
        try:
            raw = str(chunk_base64url or "")
            if not raw or "=" in raw or not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
                raise ValueError
            content = base64.urlsafe_b64decode(raw + "=" * ((4 - len(raw) % 4) % 4))
        except (TypeError, ValueError) as exc:
            raise PackageError("package_upload_chunk_invalid") from exc
        if not content or len(content) > MAX_CHUNK_BYTES:
            raise PackageError("package_upload_chunk_invalid")
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        with self._locked():
            state = self._read()
            row = self._owned(state, upload_id, actor_id)
            if row["state"] != "open":
                raise PackageError("package_upload_not_open")
            if chunk_index < len(row["chunks"]):
                if row["chunks"][chunk_index] == {"index": chunk_index, "offset": offset, "size": len(content), "digest": digest}:
                    return self._public(upload_id, row)
                raise PackageError("package_upload_conflict")
            if chunk_index != len(row["chunks"]) or offset != row["received_bytes"] or offset + len(content) > row["total_size"]:
                raise PackageError("package_upload_chunk_invalid")
            with self._part(upload_id).open("ab") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            row["chunks"].append({"index": chunk_index, "offset": offset, "size": len(content), "digest": digest})
            row["received_bytes"] += len(content)
            self._write(state)
            return self._public(upload_id, row)

    def status(self, *, actor_id: str, upload_id: str) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            return self._public(upload_id, self._owned(state, upload_id, actor_id))

    def cancel(self, *, actor_id: str, upload_id: str, request_id: str) -> dict[str, Any]:
        if not _REQUEST_ID.fullmatch(str(request_id or "")):
            raise PackageError("package_request_invalid")
        with self._locked():
            state = self._read()
            row = self._owned(state, upload_id, actor_id)
            if row["state"] == "committed":
                raise PackageError("package_upload_conflict")
            row["state"] = "cancelled"
            self._part(upload_id).unlink(missing_ok=True)
            self._write(state)
            self._audit(event="package_upload_cancelled", actor_id=actor_id, request_id=request_id, result="cancelled")
            return {"upload_id": upload_id, "state": "cancelled"}

    def commit(self, *, actor_id: str, upload_id: str, request_id: str) -> dict[str, Any]:
        if not _REQUEST_ID.fullmatch(str(request_id or "")):
            raise PackageError("package_request_invalid")
        with self._locked():
            state = self._read()
            row = self._owned(state, upload_id, actor_id)
            if row["state"] == "committed":
                if row["commit_request_id"] != request_id:
                    raise PackageError("package_upload_conflict")
                return dict(row["result"])
            if row["state"] != "open" or row["received_bytes"] != row["total_size"]:
                raise PackageError("package_upload_incomplete")
            path = self._part(upload_id)
            hasher = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                while content := handle.read(MAX_CHUNK_BYTES):
                    hasher.update(content)
                    size += len(content)
            digest = "sha256:" + hasher.hexdigest()
            if size != row["total_size"] or (row.get("expected_digest") and row["expected_digest"] != digest):
                row["state"] = "quarantined"
                row["computed_digest"] = digest
                path.unlink(missing_ok=True)
                self._write(state)
                self._audit(event="package_upload_rejected", actor_id=actor_id, request_id=request_id, result="hash_mismatch")
                raise PackageError("package_artifact_hash_mismatch")
            source = self.sources[row["source_id"]]
            verification = dict(self.verifier.verify(str(path), source=source)) if self.verifier is not None else None
            target = self.artifact_root / "sha256" / digest.split(":", 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                path.unlink(missing_ok=True)
            else:
                os.replace(path, target)
            receipt = {
                "contract": "hosting.package_artifact_receipt.v1",
                "artifact_id": digest,
                "source_id": source.source_id,
                "configuration_revision": self.configuration_revision,
                "verification": verification,
            }
            result = {"artifact_id": digest, "digest": digest, "size_bytes": size, "receipt": receipt}
            row.update(state="committed", computed_digest=digest, commit_request_id=request_id, result=result)
            self._write(state)
            self._audit(event="package_artifact_committed", actor_id=actor_id, request_id=request_id, result="committed", artifact_id=digest)
            return result

    def create_lock(
        self,
        *,
        lock_id: str,
        revision: int,
        runtime_kind: str,
        platform: str,
        artifacts: Sequence[Mapping[str, Any]],
        dependencies: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if (
            runtime_kind not in self.policy.allowed_runtimes
            or ("*" not in self.policy.allowed_platforms and platform not in self.policy.allowed_platforms)
        ):
            raise PackageError("package_policy_rejected")
        for artifact in artifacts:
            if artifact.get("source_id") not in self.policy.allowed_source_ids:
                raise PackageError("package_policy_rejected")
            digest = require_digest(artifact.get("artifact_id"), "package_artifact_id")
            if not (self.artifact_root / "sha256" / digest.split(":", 1)[1]).is_file():
                raise PackageError("package_artifact_unavailable")
        lock = PackageLock.build(
            lock_id=lock_id,
            revision=revision,
            policy=self.policy,
            artifacts=artifacts,
            dependencies=dependencies,
        )
        self.lock_root.mkdir(parents=True, exist_ok=True)
        target = self.lock_root / f"{lock.lock_digest.split(':', 1)[1]}.json"
        encoded = json.dumps(lock.to_dict(), sort_keys=True, separators=(",", ":"))
        if target.exists() and target.read_text(encoding="utf-8") != encoded:
            raise PackageError("package_lock_conflict")
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, target)
        return lock.to_dict()

    def artifact_path(self, artifact_id: str) -> Path:
        """Return the immutable generic-CAS path for an artifact identity."""
        digest = require_digest(artifact_id, "package_artifact_id")
        return self.artifact_root / "sha256" / digest.split(":", 1)[1]

    def source_artifacts(self, source_id: str) -> dict[str, Path]:
        """Return daemon-indexed local artifacts for bounded offline resolution."""
        source = self.sources.get(str(source_id))
        if source is None or not source.enabled or source.source_id not in self.policy.allowed_source_ids:
            raise PackageError("package_source_unavailable")
        root = self.artifact_root / "by-source" / source.source_id
        if not root.is_dir():
            return {}
        return {
            path.name: path.resolve()
            for path in sorted(root.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.parent == root
        }

    def source_artifact_path(
        self, *, source_id: str, filename: str, artifact_id: str
    ) -> Path:
        """Resolve one indexed wheel and revalidate its generic-CAS identity."""
        logical_filename = str(filename or "")
        if (
            Path(logical_filename).name != logical_filename
            or not logical_filename.lower().endswith(".whl")
        ):
            raise PackageError("package_artifact_filename_invalid")
        path = self.source_artifacts(source_id).get(logical_filename)
        if path is None or not path.is_file():
            raise PackageError("package_artifact_unavailable")
        expected = require_digest(artifact_id, "package_artifact_id")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                hasher.update(block)
        if "sha256:" + hasher.hexdigest() != expected or not self.artifact_path(expected).is_file():
            raise PackageError("package_artifact_hash_mismatch")
        return path

    def import_verified_file(
        self, *, source_id: str, path: Path, expected_digest: str, actor_id: str, request_id: str
    ) -> dict[str, Any]:
        """Rehash daemon-local resolved bytes into the generic CAS."""
        source = self.sources.get(str(source_id))
        if source is None or not source.enabled or source.source_id not in self.policy.allowed_source_ids:
            raise PackageError("package_source_unavailable")
        if source.credential_ref is not None and source.credential_ref not in self._credentials:
            raise PackageError("package_credential_unavailable")
        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise PackageError("package_artifact_unavailable")
        filename = source_path.name
        if (
            filename in {"", ".", ".."}
            or Path(filename).name != filename
            or not filename.lower().endswith(".whl")
        ):
            raise PackageError("package_artifact_filename_invalid")
        size = source_path.stat().st_size
        if size < 1 or size > self.policy.max_artifact_bytes:
            raise PackageError("package_upload_bounds_exceeded")
        hasher = hashlib.sha256()
        with source_path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                hasher.update(block)
        digest = "sha256:" + hasher.hexdigest()
        if digest != require_digest(expected_digest, "package_expected_digest"):
            raise PackageError("package_artifact_hash_mismatch")
        target = self.artifact_path(digest)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.scratch_root / f"import-{secrets.token_hex(12)}.part"
            temporary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, target)
        source_root = self.artifact_root / "by-source" / source.source_id
        source_root.mkdir(parents=True, exist_ok=True)
        alias = source_root / filename
        if alias.exists():
            alias_hasher = hashlib.sha256(alias.read_bytes()).hexdigest()
            if alias_hasher != hasher.hexdigest():
                raise PackageError("package_artifact_filename_conflict")
        else:
            temporary_alias = source_root / f".{filename}.{secrets.token_hex(8)}.tmp"
            try:
                os.link(target, temporary_alias)
            except OSError:
                shutil.copyfile(target, temporary_alias)
            os.replace(temporary_alias, alias)
        self._audit(event="package_artifact_import", actor_id=actor_id, request_id=request_id, result="committed", artifact_id=digest)
        return {"artifact_id": digest, "size_bytes": size, "source_id": source.source_id}
