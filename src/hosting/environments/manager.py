"""Content-addressed environment lifecycle shared by all worker kinds."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..service.operation_repository import _exclusive_process_file_lock
from .contracts import (
    EnvironmentLock,
    EnvironmentReceipt,
    EnvironmentReference,
    EnvironmentRequest,
    EnvironmentTemplate,
)


class EnvironmentBuilder(Protocol):
    builder_id: str
    runtime_kind: str

    def build(self, *, request: EnvironmentRequest, destination: Path, package_lock: Mapping[str, Any]) -> Mapping[str, Any]: ...


class EnvironmentError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


class EnvironmentManager:
    MAX_TEMPLATES = 4096
    MAX_REFERENCES = 100_000
    MAX_PAGE_SIZE = 500
    def __init__(
        self,
        *,
        environment_root: Path,
        scratch_root: Path,
        package_lock_root: Path,
        configuration_revision: str,
        builders: Mapping[str, EnvironmentBuilder],
        retention_seconds: int = 0,
        clock=time.time,
    ) -> None:
        self.root = Path(environment_root).resolve()
        self.scratch_root = Path(scratch_root).resolve() / "environment-builds"
        self.package_lock_root = Path(package_lock_root).resolve()
        self.configuration_revision = str(configuration_revision)
        self.builders = dict(builders)
        self.retention_seconds = max(0, int(retention_seconds))
        self.clock = clock
        self._thread_lock = threading.RLock()
        self._build_locks_guard = threading.Lock()
        self._build_locks: dict[str, threading.Lock] = {}
        self._state_path = self.root / "state.json"
        self._lock_path = self.root / ".environment.lock"

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": "hosting.environment_state.v2", "templates": {}, "references": {}, "busy": {}, "active": {}}

    def _read(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return self._empty()
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EnvironmentError("environment_state_invalid") from exc
        if not isinstance(value, dict) or value.get("contract") != "hosting.environment_state.v2" or set(value) != {"contract", "templates", "references", "busy", "active"}:
            raise EnvironmentError("environment_state_invalid")
        if any(not isinstance(value[key], dict) for key in ("templates", "references", "busy", "active")):
            raise EnvironmentError("environment_state_invalid")
        if len(value["templates"]) > self.MAX_TEMPLATES or len(value["references"]) > self.MAX_REFERENCES:
            raise EnvironmentError("environment_state_limit_exceeded")
        try:
            value["templates"] = {
                key: EnvironmentTemplate.from_dict(row).to_dict()
                for key, row in value["templates"].items()
            }
            value["references"] = {
                key: EnvironmentReference.from_dict(row).to_dict()
                for key, row in value["references"].items()
            }
        except (TypeError, ValueError) as exc:
            raise EnvironmentError("environment_state_invalid") from exc
        return value

    def receipt(self, *, environment_id: str) -> dict[str, Any]:
        digest = environment_id.split(":", 1)[1] if environment_id.startswith("sha256:") else ""
        if not digest:
            raise EnvironmentError("environment_id_invalid")
        path = self.root / "receipts" / f"{digest}.json"
        try:
            receipt = EnvironmentReceipt.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise EnvironmentError("environment_receipt_invalid") from exc
        if receipt.environment_id != environment_id or receipt.configuration_revision != self.configuration_revision:
            raise EnvironmentError("environment_receipt_stale")
        return receipt.to_dict()

    def _write(self, value: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_name(f".{self._state_path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(dict(value), handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _locked(self):
        class _Combined:
            def __init__(inner, manager: "EnvironmentManager") -> None:
                inner.manager = manager
                inner.process = None

            def __enter__(inner):
                inner.manager._thread_lock.acquire()
                inner.process = _exclusive_process_file_lock(inner.manager._lock_path)
                inner.process.__enter__()
                return inner

            def __exit__(inner, exc_type, exc, tb):
                try:
                    assert inner.process is not None
                    return inner.process.__exit__(exc_type, exc, tb)
                finally:
                    inner.manager._thread_lock.release()

        return _Combined(self)

    @staticmethod
    def _template_key(template_id: str, revision: int) -> str:
        return f"{template_id}@{revision}"

    def _build_lock(self, content_key: str):
        digest = content_key.split(":", 1)[1]
        with self._build_locks_guard:
            lock = self._build_locks.setdefault(digest, threading.Lock())

        class _BuildLock:
            def __init__(inner) -> None:
                inner.process = None

            def __enter__(inner):
                lock.acquire()
                inner.process = _exclusive_process_file_lock(self.root / "build-locks" / f"{digest}.lock")
                inner.process.__enter__()
                return inner

            def __exit__(inner, exc_type, exc, tb):
                try:
                    assert inner.process is not None
                    return inner.process.__exit__(exc_type, exc, tb)
                finally:
                    lock.release()

        return _BuildLock()

    def put_template(self, template: EnvironmentTemplate) -> dict[str, Any]:
        if not isinstance(template, EnvironmentTemplate):
            raise TypeError("environment_template_required")
        with self._locked():
            state = self._read()
            key = self._template_key(template.template_id, template.revision)
            existing = state["templates"].get(key)
            value = template.to_dict()
            if existing is not None and existing != value:
                raise EnvironmentError("environment_template_conflict")
            if existing is None and len(state["templates"]) >= self.MAX_TEMPLATES:
                raise EnvironmentError("environment_template_limit_exceeded")
            state["templates"][key] = value
            self._write(state)
            return value

    def set_template_state(self, *, template_id: str, revision: int, state_value: str) -> dict[str, Any]:
        if state_value not in {"active", "deprecated", "revoked"}:
            raise EnvironmentError("environment_template_state_invalid")
        with self._locked():
            state = self._read()
            key = self._template_key(template_id, revision)
            row = state["templates"].get(key)
            if not isinstance(row, dict):
                raise EnvironmentError("environment_template_unavailable")
            if row.get("state") == "revoked" and state_value != "revoked":
                raise EnvironmentError("environment_template_revoked")
            if state_value == "active":
                for other in state["templates"].values():
                    if other.get("template_id") == template_id and other.get("state") == "active":
                        other["state"] = "deprecated"
            row["state"] = state_value
            self._write(state)
            return dict(row)

    def list_templates(self, *, include_revoked: bool = False) -> dict[str, Any]:
        with self._locked():
            rows = [dict(row) for row in self._read()["templates"].values() if include_revoked or row.get("state") != "revoked"]
        return {"templates": sorted(rows, key=lambda row: (row["template_id"], row["revision"])), "configuration_revision": self.configuration_revision}

    def describe_template(self, *, template_id: str, revision: int | None = None) -> dict[str, Any]:
        with self._locked():
            rows = [dict(row) for row in self._read()["templates"].values() if row.get("template_id") == template_id and (revision is None or row.get("revision") == revision)]
        if not rows:
            raise EnvironmentError("environment_template_unavailable")
        return max(rows, key=lambda row: row["revision"])

    def _package_lock(self, digest: str, expected_id: str) -> dict[str, Any]:
        path = self.package_lock_root / f"{digest.split(':', 1)[1]}.json"
        if not path.is_file():
            raise EnvironmentError("package_lock_unavailable")
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise EnvironmentError("package_lock_invalid") from exc
        if row.get("contract") != "hosting.package_lock.v1" or row.get("lock_digest") != digest or row.get("lock_id") != expected_id:
            raise EnvironmentError("package_lock_invalid")
        return dict(row)

    @staticmethod
    def _content_key(request: EnvironmentRequest, template: EnvironmentTemplate) -> str:
        value = {
            "runtime_kind": request.runtime_kind,
            "platform": request.platform,
            "template_id": template.template_id,
            "template_revision": template.revision,
            "builder_id": template.builder_id,
            "package_lock_digest": request.package_lock_digest,
            "configuration_revision": request.configuration_revision,
        }
        return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def ensure(self, request: EnvironmentRequest) -> dict[str, Any]:
        if not isinstance(request, EnvironmentRequest):
            raise TypeError("environment_request_required")
        if request.configuration_revision != self.configuration_revision:
            raise EnvironmentError("environment_configuration_revision_stale")
        with self._locked():
            state = self._read()
            raw_template = state["templates"].get(self._template_key(request.template_id, request.template_revision))
            if not isinstance(raw_template, dict) or raw_template.get("state") != "active":
                raise EnvironmentError("environment_template_unavailable")
            template = EnvironmentTemplate.from_dict(raw_template)
            if request.runtime_kind != template.runtime_kind or request.platform not in template.platforms:
                raise EnvironmentError("environment_template_incompatible")
            builder = self.builders.get(template.builder_id)
            if builder is None or builder.runtime_kind != request.runtime_kind:
                raise EnvironmentError("environment_builder_unavailable")
            package_lock = self._package_lock(request.package_lock_digest, template.package_lock_id)
            content_key = self._content_key(request, template)
        environment_id = content_key
        digest = content_key.split(":", 1)[1]
        with self._build_lock(content_key):
            content_dir = self.root / "content" / digest
            receipt_path = self.root / "receipts" / f"{digest}.json"
            with self._locked():
                state = self._read()
                if receipt_path.is_file() and content_dir.is_dir():
                    receipt = self.receipt(environment_id=environment_id)
                    reference = self._acquire_locked(state, environment_id, request)
                    self._write(state)
                    return {"receipt": receipt, "reference": reference, "reused": True}
                state["busy"][environment_id] = {"request_id": request.request_id, "started_at_ms": int(self.clock() * 1000)}
                self._write(state)
            staging = self.scratch_root / f"{digest}.{secrets.token_hex(8)}.part"
            try:
                staging.mkdir(parents=True, exist_ok=False)
                builder_result = dict(builder.build(request=request, destination=staging, package_lock=package_lock))
                if content_dir.exists():
                    shutil.rmtree(staging)
                else:
                    content_dir.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging, content_dir)
                lock = EnvironmentLock(environment_id, content_key, request.runtime_kind, request.platform, template.template_id, template.revision, request.package_lock_digest, request.configuration_revision)
                receipt = EnvironmentReceipt(environment_id, content_key, 1, f"@environments/content/{digest}", request.runtime_kind, request.platform, template.template_id, template.revision, request.package_lock_digest, request.configuration_revision, builder_result).to_dict()
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                lock_path = self.root / "locks" / f"{digest}.json"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                if not receipt_path.exists():
                    temporary = receipt_path.with_suffix(".tmp")
                    temporary.write_text(encoded, encoding="utf-8")
                    os.replace(temporary, receipt_path)
                if not lock_path.exists():
                    temporary_lock = lock_path.with_suffix(".tmp")
                    temporary_lock.write_text(
                        json.dumps(lock.to_dict(), sort_keys=True, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    os.replace(temporary_lock, lock_path)
                with self._locked():
                    state = self._read()
                    state["busy"].pop(environment_id, None)
                    reference = self._acquire_locked(state, environment_id, request)
                    self._write(state)
                return {"receipt": receipt, "reference": reference, "reused": False}
            except Exception as exc:
                shutil.rmtree(staging, ignore_errors=True)
                with self._locked():
                    state = self._read()
                    state["busy"].pop(environment_id, None)
                    self._write(state)
                if isinstance(exc, EnvironmentError):
                    raise
                raise EnvironmentError("environment_build_failed") from exc

    def adopt_published(
        self,
        *,
        environment_id: str,
        consumer_kind: str,
        consumer_id: str,
        revision: int,
        template_id: str,
        template_revision: int,
        package_lock_digest: str,
        runtime_kind: str,
        platform: str,
        builder_id: str,
    ) -> dict[str, Any]:
        """Attach neutral receipt/reference authority to validated published bytes."""
        request = EnvironmentRequest.from_dict({
            "contract": EnvironmentRequest.CONTRACT,
            "request_id": "adopt-" + hashlib.sha256(f"{consumer_kind}|{consumer_id}|{revision}".encode("utf-8")).hexdigest(),
            "consumer_kind": consumer_kind,
            "consumer_id": consumer_id,
            "revision": revision,
            "template_id": template_id,
            "template_revision": template_revision,
            "package_lock_digest": package_lock_digest,
            "runtime_kind": runtime_kind,
            "platform": platform,
            "configuration_revision": self.configuration_revision,
        })
        if not environment_id.startswith("sha256:") or len(environment_id) != 71 or any(character not in "0123456789abcdef" for character in environment_id[7:]):
            raise EnvironmentError("environment_id_invalid")
        digest = environment_id.split(":", 1)[1]
        content_dir = self.root / "content" / digest
        if not content_dir.is_dir():
            raise EnvironmentError("environment_unavailable")
        receipt_path = self.root / "receipts" / f"{digest}.json"
        lock_path = self.root / "locks" / f"{digest}.json"
        lock = EnvironmentLock(
            environment_id, environment_id, runtime_kind, platform, template_id,
            template_revision, package_lock_digest, self.configuration_revision,
        ).to_dict()
        receipt = EnvironmentReceipt(
            environment_id, environment_id, 1, f"@environments/content/{digest}",
            runtime_kind, platform, template_id, template_revision,
            package_lock_digest, self.configuration_revision,
            {"builder_id": builder_id, "adopted": True},
        ).to_dict()
        existed = receipt_path.exists()
        with self._locked():
            state = self._read()
            if receipt_path.exists():
                existing = self.receipt(environment_id=environment_id)
                if existing != receipt:
                    raise EnvironmentError("environment_receipt_conflict")
            else:
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = receipt_path.with_name(f".{receipt_path.name}.{secrets.token_hex(8)}.tmp")
                temporary.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                os.replace(temporary, receipt_path)
            if lock_path.exists():
                try:
                    existing_lock = EnvironmentLock.from_dict(
                        json.loads(lock_path.read_text(encoding="utf-8"))
                    ).to_dict()
                except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise EnvironmentError("environment_lock_invalid") from exc
                if existing_lock != lock:
                    raise EnvironmentError("environment_lock_conflict")
            else:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_lock = lock_path.with_name(f".{lock_path.name}.{secrets.token_hex(8)}.tmp")
                temporary_lock.write_text(json.dumps(lock, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                os.replace(temporary_lock, lock_path)
            reference = self._acquire_locked(state, environment_id, request)
            self._write(state)
        return {"receipt": receipt, "reference": reference, "reused": existed}

    def _acquire_locked(self, state: dict[str, Any], environment_id: str, request: EnvironmentRequest) -> dict[str, Any]:
        seed = f"{environment_id}|{request.consumer_kind}|{request.consumer_id}|{request.revision}"
        reference_id = "ref-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]
        existing = state["references"].get(reference_id)
        if isinstance(existing, dict) and existing.get("released_at_ms") is None:
            return dict(existing)
        if existing is None and len(state["references"]) >= self.MAX_REFERENCES:
            raise EnvironmentError("environment_reference_limit_exceeded")
        reference = EnvironmentReference(reference_id, environment_id, request.consumer_kind, request.consumer_id, request.revision, int(self.clock() * 1000), None).to_dict()
        state["references"][reference_id] = reference
        return reference

    def release(self, *, reference_id: str) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            row = state["references"].get(reference_id)
            if not isinstance(row, dict):
                raise EnvironmentError("environment_reference_unavailable")
            if row.get("released_at_ms") is None:
                row["released_at_ms"] = int(self.clock() * 1000)
                self._write(state)
            return dict(row)

    def list_references(self, *, cursor: str = "", limit: int = 100) -> dict[str, Any]:
        page_size = max(1, min(int(limit), self.MAX_PAGE_SIZE))
        with self._locked():
            rows = sorted((dict(row) for row in self._read()["references"].values()), key=lambda row: row["reference_id"])
        if cursor:
            rows = [row for row in rows if row["reference_id"] > cursor]
        page = rows[:page_size]
        return {"references": page, "next_cursor": page[-1]["reference_id"] if len(rows) > page_size else None}

    def execution_begin(self, *, environment_id: str, execution_id: str) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            digest = environment_id.split(":", 1)[1] if environment_id.startswith("sha256:") else ""
            if not digest or not (self.root / "receipts" / f"{digest}.json").is_file():
                raise EnvironmentError("environment_unavailable")
            existing = state["active"].get(execution_id)
            row = {"environment_id": environment_id, "execution_id": execution_id, "started_at_ms": int(self.clock() * 1000)}
            if existing is not None and existing != row:
                raise EnvironmentError("environment_execution_conflict")
            state["active"][execution_id] = row
            self._write(state)
            return dict(row)

    def execution_end(self, *, execution_id: str) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            row = state["active"].pop(execution_id, None)
            if row is not None:
                self._write(state)
            return {"execution_id": execution_id, "state": "ended"}

    def remove(self, *, environment_id: str, force_retention: bool = False) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            if environment_id in state["busy"]:
                raise EnvironmentError("environment_busy")
            if any(row.get("environment_id") == environment_id for row in state["active"].values()):
                raise EnvironmentError("environment_active")
            if any(row.get("environment_id") == environment_id and row.get("released_at_ms") is None for row in state["references"].values()):
                raise EnvironmentError("environment_referenced")
            released = [int(row.get("released_at_ms") or 0) for row in state["references"].values() if row.get("environment_id") == environment_id]
            if released and not force_retention and int(self.clock() * 1000) < max(released) + self.retention_seconds * 1000:
                raise EnvironmentError("environment_retained")
            digest = environment_id.split(":", 1)[1] if environment_id.startswith("sha256:") else ""
            if not digest:
                raise EnvironmentError("environment_id_invalid")
            content_path = self.root / "content" / digest
            receipt_path = self.root / "receipts" / f"{digest}.json"
            lock_path = self.root / "locks" / f"{digest}.json"
            if not content_path.exists() and not receipt_path.exists() and not lock_path.exists():
                return {"environment_id": environment_id, "state": "already_absent"}
            shutil.rmtree(content_path, ignore_errors=True)
            receipt_path.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)
            return {"environment_id": environment_id, "state": "removed"}

    def gc(self) -> dict[str, Any]:
        removed = []
        receipt_root = self.root / "receipts"
        for receipt_path in list(receipt_root.glob("*.json")) if receipt_root.exists() else []:
            try:
                environment_id = str(json.loads(receipt_path.read_text(encoding="utf-8")).get("environment_id") or "")
                self.remove(environment_id=environment_id)
                removed.append(environment_id)
            except EnvironmentError as exc:
                if exc.code not in {"environment_referenced", "environment_busy", "environment_active", "environment_retained"}:
                    raise
        return {"removed_environment_ids": sorted(removed), "removed_count": len(removed)}
