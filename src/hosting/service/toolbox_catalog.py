"""Immutable toolbox template catalog repository and service API."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..toolbox.catalog import ToolboxEnvironmentTemplateSpec
from ..toolbox.identity import identity_digest, require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


CATALOG_STATE_CONTRACT = "hosting.toolbox.template_catalog_state.v1"
CATALOG_REVISION_DOMAIN = "hosting.toolbox.template_catalog_revision.v1"
TEMPLATE_REVISION_DOMAIN = "hosting.toolbox.template_revision.v1"
MAX_CATALOG_REVISIONS = 512
MAX_CATALOG_AUDIT_EVENTS = 4096
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{43,1024}")


def _strict_fields(row: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    unknown = sorted(set(row) - fields)
    missing = sorted(fields - set(row))
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")
    if missing:
        raise ValueError(f"{label}_missing_fields:{','.join(missing)}")


def _bounded_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}_must_be_string")
    text = value.strip()
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"{label}_invalid")
    return text


@dataclass(frozen=True, order=True)
class ToolboxTemplateArtifactReference:
    source_id: str
    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _bounded_id(self.source_id, label="artifact_source_id"))
        if not isinstance(self.filename, str):
            raise ValueError("artifact_filename_must_be_string")
        filename = self.filename.strip()
        if (
            not filename
            or len(filename.encode("utf-8")) > 256
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or any(ord(char) < 32 for char in filename)
        ):
            raise ValueError("artifact_filename_invalid")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "sha256", require_digest(self.sha256, label="artifact_sha256"))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("artifact_size_bytes_must_be_integer")
        if self.size_bytes <= 0 or self.size_bytes > 16 * 1024 * 1024 * 1024:
            raise ValueError("artifact_size_bytes_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxTemplateArtifactReference":
        row = dict(payload or {})
        fields = {"source_id", "filename", "sha256", "size_bytes"}
        _strict_fields(row, fields, label="template_artifact_reference")
        return cls(**row)


def _template_digest(
    template: ToolboxEnvironmentTemplateSpec,
    artifacts: Sequence[ToolboxTemplateArtifactReference],
    manifest_signature: str,
) -> str:
    return identity_digest(
        TEMPLATE_REVISION_DOMAIN,
        {
            "template": template.to_dict(),
            "artifacts": [item.to_dict() for item in sorted(artifacts)],
            "manifest_signature": manifest_signature,
        },
    )


class AtomicJsonToolboxTemplateCatalog:
    """Strict process-locked catalog with atomic state and bounded audit."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.clock = clock

    @staticmethod
    def _catalog_revision(entries: Sequence[Mapping[str, Any]], active: Mapping[str, Any]) -> str:
        return identity_digest(
            CATALOG_REVISION_DOMAIN,
            {
                "entries": sorted(
                    [dict(item) for item in entries],
                    key=lambda item: (str(item["template_id"]), str(item["template_digest"])),
                ),
                "active": {key: active[key] for key in sorted(active)},
            },
        )

    @classmethod
    def _empty_state(cls) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        active: dict[str, str] = {}
        return {
            "contract": CATALOG_STATE_CONTRACT,
            "catalog_revision": cls._catalog_revision(entries, active),
            "entries": entries,
            "active": active,
            "audit": [],
        }

    @staticmethod
    def _parse_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        fields = {
            "template_id",
            "template_digest",
            "identity_key",
            "template",
            "artifacts",
            "manifest_signature",
            "lifecycle",
            "published_at_ms",
            "published_by",
        }
        _strict_fields(row, fields, label="template_catalog_entry")
        template = ToolboxEnvironmentTemplateSpec.from_dict(row["template"])
        artifacts_raw = row["artifacts"]
        if not isinstance(artifacts_raw, list) or not artifacts_raw or len(artifacts_raw) > 2048:
            raise ValueError("template_artifacts_invalid")
        artifacts = tuple(ToolboxTemplateArtifactReference.from_dict(item) for item in artifacts_raw)
        if (
            len({item.sha256 for item in artifacts}) != len(artifacts)
            or len({(item.source_id, item.filename) for item in artifacts}) != len(artifacts)
        ):
            raise ValueError("template_artifact_duplicate")
        signature = str(row["manifest_signature"] or "")
        if not _SIGNATURE_RE.fullmatch(signature):
            raise ValueError("template_manifest_signature_invalid")
        digest = require_digest(row["template_digest"], label="template_digest")
        if digest != _template_digest(template, artifacts, signature):
            raise ValueError("template_digest_mismatch")
        if row["template_id"] != template.template_id:
            raise ValueError("template_id_mismatch")
        identity_key = str(row["identity_key"] or "")
        expected_identity = (
            f"{template.template_id}|{template.provenance.manifest_digest}|{template.lock_digest}"
        )
        if identity_key != expected_identity:
            raise ValueError("template_identity_key_mismatch")
        lifecycle = str(row["lifecycle"] or "")
        if lifecycle not in {"active", "deprecated", "revoked"}:
            raise ValueError("template_lifecycle_invalid")
        published_at_ms = row["published_at_ms"]
        if isinstance(published_at_ms, bool) or not isinstance(published_at_ms, int) or published_at_ms < 0:
            raise ValueError("template_published_at_ms_invalid")
        published_by = _bounded_id(row["published_by"], label="template_published_by")
        return {
            **row,
            "template": template.to_dict(),
            "artifacts": [item.to_dict() for item in sorted(artifacts)],
            "manifest_signature": signature,
            "published_by": published_by,
        }

    @classmethod
    def _validate_state(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        fields = {"contract", "catalog_revision", "entries", "active", "audit"}
        _strict_fields(row, fields, label="template_catalog_state")
        if row["contract"] != CATALOG_STATE_CONTRACT:
            raise ValueError("template_catalog_state_contract_invalid")
        if not isinstance(row["entries"], list) or len(row["entries"]) > MAX_CATALOG_REVISIONS:
            raise ValueError("template_catalog_entries_invalid")
        entries = [cls._parse_entry(item) for item in row["entries"]]
        digests = [item["template_digest"] for item in entries]
        identities = [item["identity_key"] for item in entries]
        if len(set(digests)) != len(digests) or len(set(identities)) != len(identities):
            raise ValueError("template_catalog_duplicate_revision")
        if not isinstance(row["active"], dict):
            raise ValueError("template_catalog_active_invalid")
        active = {
            _bounded_id(key, label="active_template_id"): require_digest(
                value, label="active_template_digest"
            )
            for key, value in row["active"].items()
        }
        by_digest = {item["template_digest"]: item for item in entries}
        for template_id, digest in active.items():
            entry = by_digest.get(digest)
            if entry is None or entry["template_id"] != template_id or entry["lifecycle"] != "active":
                raise ValueError("template_catalog_active_reference_invalid")
        if not isinstance(row["audit"], list) or len(row["audit"]) > MAX_CATALOG_AUDIT_EVENTS:
            raise ValueError("template_catalog_audit_invalid")
        for event in row["audit"]:
            if not isinstance(event, dict):
                raise ValueError("template_catalog_audit_event_invalid")
            if set(event) != {"at_ms", "actor_id", "action", "template_id", "template_digest", "outcome"}:
                raise ValueError("template_catalog_audit_event_invalid")
            if (
                isinstance(event["at_ms"], bool)
                or not isinstance(event["at_ms"], int)
                or event["at_ms"] < 0
            ):
                raise ValueError("template_catalog_audit_event_invalid")
            _bounded_id(event["actor_id"], label="template_audit_actor_id")
            if event["action"] not in {"publish", "deprecated", "revoked"}:
                raise ValueError("template_catalog_audit_event_invalid")
            _bounded_id(event["template_id"], label="template_audit_template_id")
            require_digest(event["template_digest"], label="template_audit_digest")
            _bounded_id(event["outcome"], label="template_audit_outcome")
        revision = require_digest(row["catalog_revision"], label="catalog_revision")
        if revision != cls._catalog_revision(entries, active):
            raise ValueError("template_catalog_revision_mismatch")
        return {
            "contract": CATALOG_STATE_CONTRACT,
            "catalog_revision": revision,
            "entries": entries,
            "active": active,
            "audit": list(row["audit"]),
        }

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("template_catalog_state_corrupt") from exc
        if not isinstance(payload, dict):
            raise ValueError("template_catalog_state_corrupt")
        return self._validate_state(payload)

    def _write_unlocked(self, state: Mapping[str, Any]) -> None:
        validated = self._validate_state(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_temp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(raw_temp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(validated, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_bounded_retries(temporary, self.path)
            if os.name != "nt":
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def read(self) -> dict[str, Any]:
        with _exclusive_process_file_lock(self.lock_path):
            return self._read_unlocked()

    def _audit(
        self,
        state: dict[str, Any],
        *,
        actor_id: str,
        action: str,
        template_id: str,
        template_digest: str,
        outcome: str,
    ) -> None:
        events = list(state["audit"])
        events.append(
            {
                "at_ms": int(self.clock() * 1000),
                "actor_id": _bounded_id(actor_id, label="template_actor_id"),
                "action": action,
                "template_id": template_id,
                "template_digest": template_digest,
                "outcome": outcome,
            }
        )
        state["audit"] = events[-MAX_CATALOG_AUDIT_EVENTS:]

    def publish(
        self,
        *,
        template: ToolboxEnvironmentTemplateSpec,
        artifacts: Sequence[ToolboxTemplateArtifactReference],
        manifest_signature: str,
        activate: bool,
        actor_id: str,
    ) -> dict[str, Any]:
        artifact_tuple = tuple(artifacts)
        if not artifact_tuple or len(artifact_tuple) > 2048:
            raise ValueError("template_artifacts_invalid")
        if any(not isinstance(item, ToolboxTemplateArtifactReference) for item in artifact_tuple):
            raise ValueError("template_artifact_type_invalid")
        if (
            len({item.sha256 for item in artifact_tuple}) != len(artifact_tuple)
            or len({(item.source_id, item.filename) for item in artifact_tuple})
            != len(artifact_tuple)
        ):
            raise ValueError("template_artifact_duplicate")
        if not isinstance(manifest_signature, str) or not _SIGNATURE_RE.fullmatch(manifest_signature):
            raise ValueError("template_manifest_signature_invalid")
        if not isinstance(activate, bool):
            raise ValueError("template_activate_must_be_boolean")
        digest = _template_digest(template, artifact_tuple, manifest_signature)
        identity_key = f"{template.template_id}|{template.provenance.manifest_digest}|{template.lock_digest}"
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            collision = next(
                (item for item in state["entries"] if item["identity_key"] == identity_key),
                None,
            )
            if collision is not None and collision["template_digest"] != digest:
                raise ValueError("template_immutable_publish_conflict")
            entry = next(
                (item for item in state["entries"] if item["template_digest"] == digest),
                None,
            )
            outcome = "idempotent"
            if entry is None:
                if len(state["entries"]) >= MAX_CATALOG_REVISIONS:
                    raise ValueError("template_catalog_capacity")
                entry = {
                    "template_id": template.template_id,
                    "template_digest": digest,
                    "identity_key": identity_key,
                    "template": template.to_dict(),
                    "artifacts": [item.to_dict() for item in sorted(artifact_tuple)],
                    "manifest_signature": manifest_signature,
                    "lifecycle": "active",
                    "published_at_ms": int(self.clock() * 1000),
                    "published_by": _bounded_id(actor_id, label="template_actor_id"),
                }
                state["entries"].append(entry)
                outcome = "published"
            if activate:
                if entry["lifecycle"] != "active":
                    raise ValueError("template_activate_lifecycle_invalid")
                state["active"][template.template_id] = digest
                outcome = "activated" if outcome == "idempotent" else "published_and_activated"
            state["catalog_revision"] = self._catalog_revision(state["entries"], state["active"])
            self._audit(
                state,
                actor_id=actor_id,
                action="publish",
                template_id=template.template_id,
                template_digest=digest,
                outcome=outcome,
            )
            self._write_unlocked(state)
            return {
                "entry": dict(entry),
                "template_id": template.template_id,
                "template_digest": digest,
                "catalog_revision": state["catalog_revision"],
                "outcome": outcome,
                "active_revision": state["active"].get(template.template_id) == digest,
            }

    def set_lifecycle(
        self,
        *,
        template_id: str,
        template_digest: str,
        lifecycle: str,
        actor_id: str,
    ) -> dict[str, Any]:
        target_id = _bounded_id(template_id, label="template_id")
        target_digest = require_digest(template_digest, label="template_digest")
        if lifecycle not in {"deprecated", "revoked"}:
            raise ValueError("template_lifecycle_transition_invalid")
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            entry = next(
                (item for item in state["entries"] if item["template_digest"] == target_digest),
                None,
            )
            if entry is None or entry["template_id"] != target_id:
                raise ValueError("template_revision_not_found")
            current = entry["lifecycle"]
            if current == "revoked" and lifecycle != "revoked":
                raise ValueError("template_lifecycle_transition_invalid")
            outcome = "idempotent" if current == lifecycle else lifecycle
            entry["lifecycle"] = lifecycle
            if state["active"].get(target_id) == target_digest:
                state["active"].pop(target_id, None)
            state["catalog_revision"] = self._catalog_revision(state["entries"], state["active"])
            self._audit(
                state,
                actor_id=actor_id,
                action=lifecycle,
                template_id=target_id,
                template_digest=target_digest,
                outcome=outcome,
            )
            self._write_unlocked(state)
            return {"entry": dict(entry), "catalog_revision": state["catalog_revision"], "outcome": outcome}


def _consumer_descriptor(entry: Mapping[str, Any], *, catalog_revision: str, active: Mapping[str, str]) -> dict[str, Any]:
    template = ToolboxEnvironmentTemplateSpec.from_dict(entry["template"])
    lifecycle = str(entry["lifecycle"])
    is_active = active.get(template.template_id) == entry["template_digest"]
    return {
        "contract": "hosting.toolbox.environment_template",
        "template_id": template.template_id,
        "template_digest": entry["template_digest"],
        "lock_digest": template.lock_digest,
        "catalog_revision": catalog_revision,
        "python_abis": list(template.python_abis),
        "platforms": list(template.platforms),
        "import_roots": list(template.exposed_import_roots),
        "lifecycle": lifecycle,
        "active_revision": is_active,
        "materialization": "not_materialized",
        "user_projection": {
            "state": "setup_needed",
            "code": "template_not_materialized",
            "summary": "The template revision is published but has not been verified on this runtime.",
        },
    }


class ToolboxTemplateCatalogMixin:
    @property
    def _toolbox_template_catalog(self) -> AtomicJsonToolboxTemplateCatalog:
        return AtomicJsonToolboxTemplateCatalog(
            (self.hosting_root / "state" / "toolbox_template_catalog.json").resolve()
        )

    def toolbox_template_list(self) -> dict[str, Any]:
        state = self._toolbox_template_catalog.read()
        descriptors = [
            _consumer_descriptor(
                item,
                catalog_revision=state["catalog_revision"],
                active=state["active"],
            )
            for item in state["entries"]
        ]
        return {
            "catalog_revision": state["catalog_revision"],
            "templates": sorted(
                descriptors,
                key=lambda item: (item["template_id"], item["template_digest"]),
            ),
        }

    def toolbox_template_describe(
        self, *, template_id: str, template_digest: str | None = None
    ) -> dict[str, Any]:
        state = self._toolbox_template_catalog.read()
        target_id = _bounded_id(template_id, label="template_id")
        target_digest = (
            require_digest(template_digest, label="template_digest")
            if template_digest is not None
            else state["active"].get(target_id)
        )
        if target_digest is None:
            raise ValueError("template_active_revision_not_found")
        entry = next(
            (
                item
                for item in state["entries"]
                if item["template_id"] == target_id
                and item["template_digest"] == target_digest
            ),
            None,
        )
        if entry is None:
            raise ValueError("template_revision_not_found")
        return _consumer_descriptor(
            entry,
            catalog_revision=state["catalog_revision"],
            active=state["active"],
        )

    def toolbox_template_publish(
        self,
        *,
        template: Mapping[str, Any],
        artifact_references: Sequence[Mapping[str, Any]],
        manifest_signature: str,
        activate: bool = False,
        actor_id: str = "service:local",
    ) -> dict[str, Any]:
        result = self._toolbox_template_catalog.publish(
            template=ToolboxEnvironmentTemplateSpec.from_dict(template),
            artifacts=tuple(
                ToolboxTemplateArtifactReference.from_dict(item)
                for item in artifact_references
            ),
            manifest_signature=manifest_signature,
            activate=activate,
            actor_id=actor_id,
        )
        entry = result.pop("entry")
        return {
            **result,
            "template_id": entry["template_id"],
            "template_digest": entry["template_digest"],
            "lifecycle": entry["lifecycle"],
            "active_revision": result["active_revision"],
        }

    def toolbox_template_deprecate(
        self,
        *,
        template_id: str,
        template_digest: str,
        actor_id: str = "service:local",
    ) -> dict[str, Any]:
        result = self._toolbox_template_catalog.set_lifecycle(
            template_id=template_id,
            template_digest=template_digest,
            lifecycle="deprecated",
            actor_id=actor_id,
        )
        return {
            "template_id": template_id,
            "template_digest": template_digest,
            "lifecycle": result["entry"]["lifecycle"],
            "catalog_revision": result["catalog_revision"],
            "outcome": result["outcome"],
        }

    def toolbox_template_revoke(
        self,
        *,
        template_id: str,
        template_digest: str,
        actor_id: str = "service:local",
    ) -> dict[str, Any]:
        result = self._toolbox_template_catalog.set_lifecycle(
            template_id=template_id,
            template_digest=template_digest,
            lifecycle="revoked",
            actor_id=actor_id,
        )
        return {
            "template_id": template_id,
            "template_digest": template_digest,
            "lifecycle": result["entry"]["lifecycle"],
            "catalog_revision": result["catalog_revision"],
            "outcome": result["outcome"],
        }


__all__ = [
    "AtomicJsonToolboxTemplateCatalog",
    "ToolboxTemplateArtifactReference",
    "ToolboxTemplateCatalogMixin",
]
