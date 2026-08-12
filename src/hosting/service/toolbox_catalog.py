"""Immutable toolbox template catalog repository and service API."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..toolbox.catalog import ToolboxEnvironmentTemplateSpec
from ..toolbox.identity import identity_digest, require_digest
from ..toolbox.host_project_config import ToolboxHostProjectConfiguration
from ..toolbox.host_project_config import ToolboxBuiltinIntent
from ..toolbox.builtin_resolver import AirgapBuiltinWheelResolver
from ..toolbox.builtin_templates import resolved_builtin_template_candidate
from ..toolbox.template_resolver import (
    VerifiedTemplateCandidate,
    resolve_verified_template_environment,
)
from ..operation_contract import (
    HostedExecutionKind,
    HostedOperationLifecycle,
    HostedOperationProgress,
    HostedOperationSelector,
    hosted_execution_fingerprint,
)
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries
from .toolbox_artifact_store import ToolboxArtifactBundleError
from .toolbox_https_acquisition import (
    ToolboxHttpsAcquisitionError,
    ToolboxHttpsArtifactAcquirer,
)
from .toolbox_materialization import (
    AtomicJsonToolboxMaterializationReceipts,
    HermeticToolboxTemplateMaterializer,
    ToolboxTemplateMaterializationError,
    ToolboxTemplateMaterializationReceipt,
    derived_environment_digest,
    materialization_target,
)


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
        if lifecycle not in {"inactive", "active", "deprecated", "revoked"}:
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
            if event["action"] not in {
                "construct", "activate", "replace", "publish", "deprecated", "revoked"
            }:
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

    def publish_inactive(
        self,
        *,
        template: ToolboxEnvironmentTemplateSpec,
        artifacts: Sequence[ToolboxTemplateArtifactReference],
        manifest_signature: str,
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
                    "lifecycle": "inactive",
                    "published_at_ms": int(self.clock() * 1000),
                    "published_by": _bounded_id(actor_id, label="template_actor_id"),
                }
                state["entries"].append(entry)
                outcome = "constructed"
            state["catalog_revision"] = self._catalog_revision(state["entries"], state["active"])
            self._audit(
                state,
                actor_id=actor_id,
                action="construct",
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

    def activate(
        self,
        *,
        template_id: str,
        template_digest: str,
        actor_id: str,
    ) -> dict[str, Any]:
        target_id = _bounded_id(template_id, label="template_id")
        target_digest = require_digest(template_digest, label="template_digest")
        actor = _bounded_id(actor_id, label="template_actor_id")
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            entry = next(
                (item for item in state["entries"] if item["template_digest"] == target_digest),
                None,
            )
            if entry is None or entry["template_id"] != target_id:
                raise ValueError("template_revision_not_found")
            current = state["active"].get(target_id)
            if current is not None and current != target_digest:
                raise ValueError("template_active_revision_exists")
            if entry["lifecycle"] in {"deprecated", "revoked"}:
                raise ValueError("template_activate_lifecycle_invalid")
            outcome = "idempotent" if current == target_digest else "activated"
            entry["lifecycle"] = "active"
            state["active"][target_id] = target_digest
            state["catalog_revision"] = self._catalog_revision(state["entries"], state["active"])
            self._audit(
                state,
                actor_id=actor,
                action="activate",
                template_id=target_id,
                template_digest=target_digest,
                outcome=outcome,
            )
            self._write_unlocked(state)
            return {"entry": dict(entry), "catalog_revision": state["catalog_revision"], "outcome": outcome}

    def replace(
        self,
        *,
        template_id: str,
        expected_active_digest: str,
        replacement_digest: str,
        actor_id: str,
    ) -> dict[str, Any]:
        target_id = _bounded_id(template_id, label="template_id")
        expected = require_digest(expected_active_digest, label="expected_active_digest")
        replacement = require_digest(replacement_digest, label="replacement_digest")
        actor = _bounded_id(actor_id, label="template_actor_id")
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            if state["active"].get(target_id) != expected:
                raise ValueError("template_active_revision_conflict")
            old_entry = next(
                (item for item in state["entries"] if item["template_digest"] == expected),
                None,
            )
            new_entry = next(
                (item for item in state["entries"] if item["template_digest"] == replacement),
                None,
            )
            if (
                old_entry is None
                or new_entry is None
                or old_entry["template_id"] != target_id
                or new_entry["template_id"] != target_id
            ):
                raise ValueError("template_revision_not_found")
            if new_entry["lifecycle"] in {"deprecated", "revoked"}:
                raise ValueError("template_replace_lifecycle_invalid")
            outcome = "idempotent" if replacement == expected else "replaced"
            if replacement != expected:
                old_entry["lifecycle"] = "deprecated"
                new_entry["lifecycle"] = "active"
                state["active"][target_id] = replacement
            state["catalog_revision"] = self._catalog_revision(state["entries"], state["active"])
            self._audit(
                state,
                actor_id=actor,
                action="replace",
                template_id=target_id,
                template_digest=replacement,
                outcome=outcome,
            )
            self._write_unlocked(state)
            return {
                "entry": dict(new_entry),
                "replaced_template_digest": expected,
                "catalog_revision": state["catalog_revision"],
                "outcome": outcome,
            }

    def publish_batch(
        self,
        *,
        releases: Sequence[Mapping[str, Any]],
        actor_id: str,
    ) -> tuple[dict[str, Any], ...]:
        batch: list[
            tuple[
                ToolboxEnvironmentTemplateSpec,
                tuple[ToolboxTemplateArtifactReference, ...],
                str,
                str,
            ]
        ] = []
        for raw in releases:
            row = dict(raw or {})
            if set(row) != {
                "template", "template_digest", "artifact_references", "manifest_signature"
            }:
                raise ValueError("template_publish_batch_fields_invalid")
            template = ToolboxEnvironmentTemplateSpec.from_dict(row["template"])
            artifacts = tuple(
                ToolboxTemplateArtifactReference.from_dict(item)
                for item in row["artifact_references"]
            )
            signature = str(row["manifest_signature"] or "")
            if (
                not artifacts
                or len({item.sha256 for item in artifacts}) != len(artifacts)
                or len({(item.source_id, item.filename) for item in artifacts})
                != len(artifacts)
                or not _SIGNATURE_RE.fullmatch(signature)
            ):
                raise ValueError("template_publish_batch_invalid")
            digest = _template_digest(template, artifacts, signature)
            if require_digest(row["template_digest"], label="template_digest") != digest:
                raise ValueError("template_publish_batch_digest_mismatch")
            batch.append((template, artifacts, signature, digest))
        identities = [
            f"{item.template_id}|{item.provenance.manifest_digest}|{item.lock_digest}"
            for item, _artifacts, _signature, _digest in batch
        ]
        if (
            not batch
            or len({item.template_id for item, *_rest in batch}) != len(batch)
            or len(set(identities)) != len(identities)
        ):
            raise ValueError("template_publish_batch_duplicate")
        actor = _bounded_id(actor_id, label="template_actor_id")
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            results: list[dict[str, Any]] = []
            now_ms = int(self.clock() * 1000)
            for (template, artifacts, signature, digest), identity_key in zip(
                batch, identities
            ):
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
                    entry = {
                        "template_id": template.template_id,
                        "template_digest": digest,
                        "identity_key": identity_key,
                        "template": template.to_dict(),
                        "artifacts": [item.to_dict() for item in sorted(artifacts)],
                        "manifest_signature": signature,
                        "lifecycle": "active",
                        "published_at_ms": now_ms,
                        "published_by": actor,
                    }
                    state["entries"].append(entry)
                    outcome = "published_and_activated"
                elif entry["lifecycle"] in {"deprecated", "revoked"}:
                    raise ValueError("template_activate_lifecycle_invalid")
                elif state["active"].get(template.template_id) != digest:
                    outcome = "activated"
                entry["lifecycle"] = "active"
                state["active"][template.template_id] = digest
                self._audit(
                    state,
                    actor_id=actor,
                    action="publish",
                    template_id=template.template_id,
                    template_digest=digest,
                    outcome=outcome,
                )
                results.append(
                    {
                        "template_id": template.template_id,
                        "template_digest": digest,
                        "outcome": outcome,
                        "active_revision": True,
                    }
                )
            if len(state["entries"]) > MAX_CATALOG_REVISIONS:
                raise ValueError("template_catalog_capacity")
            state["catalog_revision"] = self._catalog_revision(
                state["entries"], state["active"]
            )
            self._write_unlocked(state)
            catalog_revision = state["catalog_revision"]
        return tuple({**item, "catalog_revision": catalog_revision} for item in results)

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


def _consumer_descriptor(
    entry: Mapping[str, Any],
    *,
    catalog_revision: str,
    active: Mapping[str, str],
    receipts: Sequence[ToolboxTemplateMaterializationReceipt] = (),
) -> dict[str, Any]:
    template = ToolboxEnvironmentTemplateSpec.from_dict(entry["template"])
    lifecycle = str(entry["lifecycle"])
    is_active = active.get(template.template_id) == entry["template_digest"]
    verified = bool(receipts) and lifecycle != "revoked"
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
        "materialization": "ready" if verified else "not_materialized",
        "user_projection": {
            "state": "ready" if verified else "setup_needed",
            "code": "template_ready" if verified else "template_not_materialized",
            "summary": (
                "The template is ready on this tool runtime."
                if verified
                else "The template revision is published but has not been verified on this runtime."
            ),
        },
    }


class ToolboxTemplateCatalogMixin:
    @property
    def _toolbox_template_catalog(self) -> AtomicJsonToolboxTemplateCatalog:
        return AtomicJsonToolboxTemplateCatalog(
            (self.hosting_root / "state" / "toolbox_template_catalog.json").resolve()
        )

    @property
    def _toolbox_materialization_receipts(self) -> AtomicJsonToolboxMaterializationReceipts:
        return AtomicJsonToolboxMaterializationReceipts(
            (self.hosting_root / "state" / "toolbox_template_materializations.json").resolve()
        )

    def _template_receipts(self, template_digest: str) -> tuple[ToolboxTemplateMaterializationReceipt, ...]:
        return self._toolbox_materialization_receipts.list_for_template(
            template_digest=template_digest
        )

    def toolbox_template_list(self) -> dict[str, Any]:
        state = self._toolbox_template_catalog.read()
        descriptors = [
            _consumer_descriptor(
                item,
                catalog_revision=state["catalog_revision"],
                active=state["active"],
                receipts=self._template_receipts(item["template_digest"]),
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

    def toolbox_required_template_status(
        self, *, python_abi: str, platform: str
    ) -> dict[str, Any]:
        target = materialization_target(python_abi=python_abi, platform=platform)
        configured = getattr(self, "_toolbox_host_project_config", None)
        if not isinstance(configured, ToolboxHostProjectConfiguration):
            return {
                "status": "unavailable",
                "code": "toolbox_configuration_missing",
                "config_revision": None,
                "catalog_revision": self._toolbox_template_catalog.read()["catalog_revision"],
                "target": target,
                "templates": [],
                "diagnostics": [
                    {
                        "code": "toolbox_configuration_missing",
                        "summary": "Toolbox hosting configuration is not available.",
                    }
                ],
            }
        required_ids = tuple(item.template_id for item in configured.builtins if item.required)
        config_revision = (
            configured.config_revision
            if isinstance(configured, ToolboxHostProjectConfiguration)
            else None
        )
        configured_trust_keys = (
            {key_id for source in configured.sources for key_id in source.trust_key_ids}
            if isinstance(configured, ToolboxHostProjectConfiguration)
            else set()
        )
        state = self._toolbox_template_catalog.read()
        diagnostics: list[dict[str, str]] = []
        templates: list[dict[str, Any]] = []
        for template_id in required_ids:
            entry = next(
                (
                    item for item in state["entries"]
                    if item["template_id"] == template_id
                    and state["active"].get(template_id) == item["template_digest"]
                ),
                None,
            )
            if entry is None:
                code = "required_template_missing"
                ready = False
                template_digest = None
                manifest_digest = None
                lock_digest = None
            elif entry["lifecycle"] == "revoked":
                code = "required_template_lock_invalid"
                ready = False
                template_digest = entry["template_digest"]
                manifest_digest = entry["template"]["provenance"]["manifest_digest"]
                lock_digest = entry["template"]["lock_digest"]
            elif isinstance(configured, ToolboxHostProjectConfiguration) and not (
                entry["template"]["provenance"]["signing_key_id"]
                in configured_trust_keys
                or (
                    entry["template"]["provenance"]["signing_key_id"].startswith(
                        "ed25519-set:"
                    )
                    and set(
                        entry["template"]["provenance"]["signing_key_id"]
                        .removeprefix("ed25519-set:")
                        .split("+")
                    ).issubset(configured_trust_keys)
                )
            ):
                code = "required_template_signature_invalid"
                ready = False
                template_digest = entry["template_digest"]
                manifest_digest = entry["template"]["provenance"]["manifest_digest"]
                lock_digest = entry["template"]["lock_digest"]
            else:
                template_digest = entry["template_digest"]
                manifest_digest = entry["template"]["provenance"]["manifest_digest"]
                lock_digest = entry["template"]["lock_digest"]
                receipt = self._toolbox_materialization_receipts.get(
                    template_digest=template_digest,
                    python_abi=python_abi,
                    platform=platform,
                )
                ready = receipt is not None
                code = "required_template_ready" if ready else "required_template_materialization_failed"
            templates.append(
                {
                    "template_id": template_id,
                    "template_digest": template_digest,
                    "manifest_digest": manifest_digest,
                    "lock_digest": lock_digest,
                    "target": target,
                    "ready": ready,
                    "code": code,
                }
            )
            if not ready:
                diagnostics.append(
                    {
                        "code": code,
                        "summary": f"Required template {template_id} is not ready on target {target}.",
                    }
                )
        ready = all(item["ready"] for item in templates) and tuple(
            item["template_id"] for item in templates
        ) == required_ids
        return {
            "status": "ready" if ready else "degraded",
            "code": "required_templates_ready" if ready else diagnostics[0]["code"],
            "config_revision": config_revision,
            "catalog_revision": state["catalog_revision"],
            "target": target,
            "templates": templates,
            "diagnostics": diagnostics,
        }

    def resolve_hosted_template_environment(
        self,
        *,
        consumer_kind: str,
        files: Sequence[Mapping[str, Any]],
        python_abi: str,
        platform: str,
        declared_imports: Sequence[str] = (),
        package_requirements: Sequence[str] = (),
        intrinsic_names: Sequence[str] = (),
        allowed_template_ids: Sequence[str] | None = None,
        sandbox_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve one consumer against active exact receipts without starting work."""

        materialization_target(python_abi=python_abi, platform=platform)
        state = self._toolbox_template_catalog.read()
        candidates: list[VerifiedTemplateCandidate] = []
        for template_id, template_digest in sorted(state["active"].items()):
            entry = next(
                item for item in state["entries"]
                if item["template_id"] == template_id
                and item["template_digest"] == template_digest
            )
            receipt = self._toolbox_materialization_receipts.get(
                template_digest=template_digest,
                python_abi=python_abi,
                platform=platform,
            )
            if receipt is None:
                continue
            candidates.append(
                VerifiedTemplateCandidate(
                    template=ToolboxEnvironmentTemplateSpec.from_dict(entry["template"]),
                    template_digest=template_digest,
                    environment_digest=receipt.environment_digest,
                    python_abi=receipt.python_abi,
                    platform=receipt.platform,
                )
            )
        resolution = resolve_verified_template_environment(
            consumer_kind=consumer_kind,
            files=files,
            candidates=candidates,
            python_abi=python_abi,
            platform=platform,
            declared_imports=declared_imports,
            package_requirements=package_requirements,
            intrinsic_names=intrinsic_names,
            allowed_template_ids=allowed_template_ids,
            sandbox_policy=sandbox_policy,
        )
        return resolution.to_dict()

    def materialize_toolbox_environment_for_bundle(
        self,
        *,
        files: Sequence[Mapping[str, Any]],
        python_abi: str,
        platform: str,
        declared_imports: Sequence[str] = (),
        package_requirements: Sequence[str] = (),
        intrinsic_names: Sequence[str] = (),
        allowed_template_ids: Sequence[str] | None = None,
        sandbox_policy: Mapping[str, Any] | None = None,
        reference_id: str,
        resolved_environment: Mapping[str, Any] | None = None,
    ):
        """Resolve and acquire a receipt-verified physical toolbox environment."""

        builder = getattr(self, "_hermetic_toolbox_environment_builder", None)
        if builder is None:
            raise ToolboxTemplateMaterializationError(
                "template_materializer_unconfigured",
                "This runtime host has no configured hermetic toolbox environment builder.",
            )
        if resolved_environment:
            from ..toolbox.hermetic_environment import ResolvedToolboxEnvironmentInput

            resolved = ResolvedToolboxEnvironmentInput.from_dict(resolved_environment)
            allowed = {
                str(item or "").strip() for item in list(allowed_template_ids or [])
                if str(item or "").strip()
            }
            if allowed and resolved.template_id not in allowed:
                raise ToolboxTemplateMaterializationError(
                    "resolved_template_not_allowed",
                    "The confirmed environment does not use the assigned template.",
                )
            if resolved.python_abi != python_abi or resolved.platform != platform:
                raise ToolboxTemplateMaterializationError(
                    "resolved_target_mismatch",
                    "The confirmed environment target no longer matches this host.",
                )
            if not set(declared_imports).issubset(set(resolved.resolved_import_roots)):
                raise ToolboxTemplateMaterializationError(
                    "resolved_import_roots_incomplete",
                    "The confirmed environment does not cover all assigned imports.",
                )
            builder.extend_verified_artifact_paths({
                (item.source_id, item.filename): self._toolbox_artifact_store.object_path(item.sha256)
                for item in resolved.locked_artifacts
            })
            return builder.materialize_environment(resolved, reference_id=reference_id)
        resolution = self.resolve_hosted_template_environment(
            consumer_kind="toolbox",
            files=files,
            python_abi=python_abi,
            platform=platform,
            declared_imports=declared_imports,
            package_requirements=package_requirements,
            intrinsic_names=intrinsic_names,
            allowed_template_ids=allowed_template_ids,
            sandbox_policy=sandbox_policy,
        )
        template_digest = resolution["binding"]["template_digest"]
        state = self._toolbox_template_catalog.read()
        entry = next(
            item for item in state["entries"]
            if item["template_digest"] == template_digest and item["lifecycle"] == "active"
        )
        resolved = HermeticToolboxTemplateMaterializer._resolved_input(
            entry, python_abi=python_abi, platform=platform
        )
        return builder.materialize_environment(resolved, reference_id=reference_id)

    def initialize_configured_toolbox_templates(
        self,
        *,
        configuration: ToolboxHostProjectConfiguration,
        request_id_prefix: str,
        actor_id: str = "service:startup",
    ) -> dict[str, Any]:
        """Resolve configured intent without publishing a partial catalog."""

        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        prefix = str(request_id_prefix or "").strip()
        if not prefix:
            raise ValueError("template_setup_request_id_prefix_required")
        python_abi = configuration.target.python_abi
        platform = configuration.target.platform
        materialization_target(python_abi=python_abi, platform=platform)
        from ..toolbox.builtin_resolver import AirgapBuiltinWheelResolver

        ingestion_diagnostic = getattr(self, "_toolbox_artifact_ingestion_diagnostic", None)
        if ingestion_diagnostic is not None:
            return {
                "status": "not_ready",
                "config_revision": configuration.config_revision,
                "source_set_revision": configuration.source_set_revision,
                "target": configuration.target.name,
                "closures": [],
                "diagnostics": [dict(ingestion_diagnostic)],
                "published": [],
                "operations": [],
            }
        resolution = AirgapBuiltinWheelResolver(
            configuration,
            artifact_sources=(
                {}
                if getattr(self, "_toolbox_trust_public_keys", None) is not None
                else getattr(self, "_toolbox_artifact_sources", {})
            ),
            verified_artifacts=getattr(self, "_toolbox_verified_artifacts", {}),
        ).resolve().to_dict()
        return {**resolution, "published": [], "operations": []}

    def _resolve_configured_toolbox_startup(
        self,
        *,
        progress: Callable[[str, str, int | None, int | None, str, bool], None],
    ) -> dict[str, Any]:
        """Ingest verified sources and resolve intent inside the setup worker."""
        configuration = getattr(self, "_toolbox_host_project_config", None)
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        self._toolbox_verified_artifacts = {}
        self._toolbox_artifact_ingestion_diagnostic = None
        progress(
            "resolution", "builtin_resolution_started", 0, 1,
            "The configured built-in closure is being resolved.", False,
        )
        if getattr(self, "_toolbox_trust_public_keys", None) is not None:
            try:
                for source in configuration.sources:
                    if source.kind != "airgap_store":
                        continue
                    source_root = self._toolbox_artifact_sources[source.source_id]
                    for bundle_path in sorted(source_root.glob("*.zip")):
                        self._toolbox_artifact_store.import_signed_bundle(
                            bundle_path,
                            configuration=configuration,
                            trust_public_keys=self._toolbox_trust_public_keys,
                        )
                    self._toolbox_verified_artifacts[source.source_id] = (
                        self._toolbox_artifact_store.source_artifacts(source.source_id)
                    )
            except ToolboxArtifactBundleError as exc:
                self._toolbox_artifact_ingestion_diagnostic = {
                    "code": exc.code,
                    "summary": exc.summary,
                }
            except (KeyError, OSError, ValueError):
                self._toolbox_artifact_ingestion_diagnostic = {
                    "code": "artifact_store_invalid",
                    "summary": "The verified toolbox artifact store is invalid.",
                }
        if (
            self._toolbox_artifact_ingestion_diagnostic is None
            and configuration.resolution.mode == "prefer_airgap"
            and self._toolbox_verified_artifacts
        ):
            airgap_result = self.initialize_configured_toolbox_templates(
                configuration=configuration,
                request_id_prefix=f"host-startup-{configuration.config_revision}",
            )
            if airgap_result.get("status") == "resolved":
                self._toolbox_startup = airgap_result
                self._configure_verified_toolbox_artifact_paths()
                progress(
                    "resolution", "builtin_resolution_checked", 1, 1,
                    "The configured built-in closure resolution was checked.", False,
                )
                return airgap_result
        if (
            self._toolbox_artifact_ingestion_diagnostic is None
            and configuration.resolution.mode in {"online", "prefer_airgap"}
        ):
            try:
                acquisition = ToolboxHttpsArtifactAcquirer(
                    configuration,
                    artifact_store=self._toolbox_artifact_store,
                    trust_public_keys=self._toolbox_trust_public_keys,
                    source_credentials=getattr(self, "_toolbox_source_credentials", {}),
                ).discover_and_acquire(
                    tuple(
                        requirement
                        for intent in configuration.builtins
                        for requirement in intent.package_requirements
                    ),
                    progress=progress,
                )
                self._toolbox_verified_artifacts.update(acquisition["verified_artifacts"])
            except ToolboxHttpsAcquisitionError as exc:
                self._toolbox_artifact_ingestion_diagnostic = {
                    "code": exc.code,
                    "summary": exc.summary,
                }
        self._configure_verified_toolbox_artifact_paths()
        startup = self.initialize_configured_toolbox_templates(
            configuration=configuration,
            request_id_prefix=f"host-startup-{configuration.config_revision}",
        )
        self._toolbox_startup = startup
        progress(
            "resolution", "builtin_resolution_checked", 1, 1,
            "The configured built-in closure resolution was checked.", False,
        )
        return startup

    def _configure_verified_toolbox_artifact_paths(self) -> None:
        exact_artifact_paths = {
            (source_id, filename): path
            for source_id, artifacts in self._toolbox_verified_artifacts.items()
            for filename, path in artifacts.items()
        }
        builder = getattr(self, "_hermetic_toolbox_environment_builder", None)
        if exact_artifact_paths and builder is not None:
            builder.configure_verified_artifact_paths(exact_artifact_paths)

    def prepare_configured_toolbox_templates(
        self,
        *,
        progress: Callable[[str, str, int | None, int | None, str, bool], None]
        | None = None,
    ) -> dict[str, Any]:
        """Build and probe all resolved built-ins without publishing state."""
        from ..toolbox.builtin_resolver import ResolvedBuiltinWheelClosure
        from ..toolbox.builtin_templates import resolved_builtin_template_candidate

        configuration = getattr(self, "_toolbox_host_project_config", None)
        startup = getattr(self, "_toolbox_startup", None)
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        if not isinstance(startup, dict) or startup.get("status") != "resolved":
            raise ValueError("required_builtin_resolution_not_ready")
        intents = {item.template_id: item for item in configuration.builtins}
        prepared: list[dict[str, Any]] = []
        references: list[tuple[str, str]] = []
        report = progress or (lambda *_args: None)
        try:
            total = len(startup["closures"])
            for index, raw_closure in enumerate(startup["closures"]):
                closure = ResolvedBuiltinWheelClosure.from_dict(raw_closure)
                report(
                    "artifact_verification",
                    "builtin_artifacts_verifying",
                    index,
                    total,
                    "The exact signed built-in artifact closure is being verified.",
                    False,
                )
                evidence = self._toolbox_artifact_store.verified_evidence_for_artifacts(
                    {item.sha256 for item in closure.locked_artifacts},
                    source_ids={item.source_id for item in closure.locked_artifacts},
                )
                candidate = resolved_builtin_template_candidate(
                    intent=intents[closure.template_id],
                    closure=closure,
                    target=configuration.target,
                    evidence=evidence,
                )
                artifacts = tuple(
                    ToolboxTemplateArtifactReference.from_dict(item)
                    for item in candidate.artifact_references
                )
                template_digest = _template_digest(
                    candidate.template, artifacts, candidate.manifest_signature
                )
                entry = {
                    "template_id": candidate.template.template_id,
                    "template_digest": template_digest,
                    "template": candidate.template.to_dict(),
                    "artifacts": [item.to_dict() for item in artifacts],
                    "manifest_signature": candidate.manifest_signature,
                }
                resolved = HermeticToolboxTemplateMaterializer._resolved_input(
                    entry,
                    python_abi=configuration.target.python_abi,
                    platform=configuration.target.platform,
                )
                reference_id = f"template:{template_digest.removeprefix('sha256:')}"
                references.append((resolved.environment_key, reference_id))
                receipt = self._toolbox_template_materializer.materialize(
                    catalog_entry=entry,
                    python_abi=configuration.target.python_abi,
                    platform=configuration.target.platform,
                    progress=report,
                )
                expected_artifacts = tuple(sorted(item.sha256 for item in artifacts))
                expected_roots = tuple(sorted(candidate.template.exposed_import_roots))
                if (
                    not isinstance(receipt, ToolboxTemplateMaterializationReceipt)
                    or receipt.template_id != candidate.template.template_id
                    or receipt.template_digest != template_digest
                    or receipt.python_abi != configuration.target.python_abi
                    or receipt.platform != configuration.target.platform
                    or tuple(sorted(receipt.artifact_digests)) != expected_artifacts
                    or tuple(sorted(receipt.verified_import_roots)) != expected_roots
                    or receipt.environment_digest
                    != derived_environment_digest(
                        template_digest=template_digest,
                        python_abi=configuration.target.python_abi,
                        platform=configuration.target.platform,
                        artifact_digests=expected_artifacts,
                    )
                ):
                    raise ToolboxTemplateMaterializationError(
                        "template_materialization_receipt_mismatch",
                        "The built-in verification receipt does not cover its exact candidate.",
                    )
                prepared.append(
                    {
                        "template": candidate.template.to_dict(),
                        "template_digest": template_digest,
                        "artifact_references": [item.to_dict() for item in artifacts],
                        "manifest_signature": candidate.manifest_signature,
                        "source_bundle_id": candidate.source_bundle_id,
                        "environment_key": resolved.environment_key,
                        "reference_id": reference_id,
                        "receipt": receipt.to_dict(),
                    }
                )
                report(
                    "import_probe",
                    "builtin_imports_verified",
                    index + 1,
                    total,
                    "The built-in candidate passed its complete import probes.",
                    False,
                )
        except Exception:
            builder = getattr(self, "_hermetic_toolbox_environment_builder", None)
            if builder is not None:
                for environment_key, reference_id in references:
                    builder.release_reference(
                        environment_key=environment_key, reference_id=reference_id
                    )
            raise
        return {
            "status": "prepared",
            "config_revision": configuration.config_revision,
            "source_set_revision": configuration.source_set_revision,
            "target": configuration.target.name,
            "candidates": prepared,
        }

    def publish_prepared_configured_toolbox_templates(
        self,
        prepared: Mapping[str, Any],
        *,
        actor_id: str = "service:setup",
    ) -> dict[str, Any]:
        """Commit a complete prepared receipt/catalog batch without partial activation."""
        configuration = getattr(self, "_toolbox_host_project_config", None)
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        row = dict(prepared or {})
        if set(row) != {
            "status", "config_revision", "source_set_revision", "target", "candidates"
        } or row.get("status") != "prepared":
            raise ValueError("prepared_builtin_batch_invalid")
        if (
            row["config_revision"] != configuration.config_revision
            or row["source_set_revision"] != configuration.source_set_revision
            or row["target"] != configuration.target.name
            or not isinstance(row["candidates"], list)
        ):
            raise ValueError("prepared_builtin_batch_stale")
        candidates = [dict(item or {}) for item in row["candidates"]]
        expected_ids = {item.template_id for item in configuration.builtins}
        actual_ids = {dict(item.get("template") or {}).get("template_id") for item in candidates}
        if not candidates or actual_ids != expected_ids or len(actual_ids) != len(candidates):
            raise ValueError("prepared_builtin_batch_incomplete")
        receipts = tuple(
            ToolboxTemplateMaterializationReceipt.from_dict(item["receipt"])
            for item in candidates
        )
        catalog_before = self._toolbox_template_catalog.read()
        active_before = set(catalog_before["active"].values())
        inserted: tuple[ToolboxTemplateMaterializationReceipt, ...] = ()
        try:
            _all_receipts, inserted = self._toolbox_materialization_receipts.put_many(receipts)
            published = self._toolbox_template_catalog.publish_batch(
                releases=[
                    {
                        "template": item["template"],
                        "template_digest": item["template_digest"],
                        "artifact_references": item["artifact_references"],
                        "manifest_signature": item["manifest_signature"],
                    }
                    for item in candidates
                ],
                actor_id=actor_id,
            )
        except Exception:
            if inserted:
                self._toolbox_materialization_receipts.remove_exact(inserted)
            builder = getattr(self, "_hermetic_toolbox_environment_builder", None)
            if builder is not None:
                for item in candidates:
                    if item["template_digest"] not in active_before:
                        builder.release_reference(
                            environment_key=item["environment_key"],
                            reference_id=item["reference_id"],
                        )
            raise
        result = {
            "status": "published",
            "config_revision": configuration.config_revision,
            "source_set_revision": configuration.source_set_revision,
            "target": configuration.target.name,
            "catalog_revision": published[0]["catalog_revision"],
            "templates": [dict(item) for item in published],
        }
        self._toolbox_startup = {**result, "closures": [], "diagnostics": []}
        return result

    def toolbox_setup_start(self, *, request_id: str | None = None) -> dict[str, Any]:
        """Start the one system-owned built-in realization operation."""
        configuration = getattr(self, "_toolbox_host_project_config", None)
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        rid = str(
            request_id
            or f"toolbox-setup-{configuration.config_revision.removeprefix('sha256:')}"
        ).strip()
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.TOOLBOX_SETUP.value,
                "host_scope": "toolbox-host",
                "config_revision": configuration.config_revision,
                "source_set_revision": configuration.source_set_revision,
                "target": configuration.target.name,
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id="system:toolbox-setup",
            execution_kind=HostedExecutionKind.TOOLBOX_SETUP,
            selector=HostedOperationSelector(kind="host_scope", id="toolbox-host"),
            namespace="toolbox_setup:toolbox-host",
            request_id=rid,
            fingerprint=fingerprint,
            metadata={
                "config_revision": configuration.config_revision,
                "source_set_revision": configuration.source_set_revision,
                "target": configuration.target.name,
            },
        )
        if prepared["action"] != "dispatch":
            status = prepared.get("status")
            if status is None:
                raise RuntimeError("hosted_operation_capacity")
            current = dict(status)
            if current.get("lifecycle") == (
                HostedOperationLifecycle.INTERRUPTED_AFTER_DISPATCH_UNKNOWN.value
            ):
                current = self._reconcile_interrupted_toolbox_setup(current)
            self._toolbox_setup_operation = current
            return current
        operation_id = str(prepared["status"]["operation"]["operation_id"])
        thread = threading.Thread(
            target=self._run_toolbox_setup,
            kwargs={"operation_id": operation_id},
            name=f"toolbox-setup-{operation_id[-8:]}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._toolbox_setup_operation = self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "code": "toolbox_setup_dispatch_failed",
                    "summary": "The toolbox setup worker could not be started.",
                },
                reason="toolbox_setup_dispatch_failed",
            )
            raise
        status = dict(prepared["status"])
        self._toolbox_setup_operation = status
        return status

    def _reconcile_interrupted_toolbox_setup(
        self, status: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Fail closed unless the durable post-publication checkpoint proves commit."""
        row = dict(status or {})
        operation_id = str(dict(row.get("operation") or {}).get("operation_id") or "")
        progress = dict(row.get("progress") or {})
        publication_proven = (
            progress.get("phase") == "publication"
            and progress.get("code") == "builtin_publication_committed"
            and progress.get("completed_units") == 1
            and progress.get("total_units") == 1
        )
        readiness = self.toolbox_required_template_status(
            python_abi=self._toolbox_host_project_config.target.python_abi,
            platform=self._toolbox_host_project_config.target.platform,
        )
        if publication_proven and readiness.get("status") == "ready":
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope={
                    "status": "ok",
                    "code": "toolbox_setup_ready",
                    "catalog_revision": readiness.get("catalog_revision"),
                    "templates": readiness.get("templates", []),
                    "reconciled_after_restart": True,
                },
            )
        return self._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
            envelope={
                "status": "error",
                "code": "toolbox_setup_interrupted_after_dispatch",
                "summary": (
                    "Toolbox setup was interrupted without a durable complete-publication "
                    "checkpoint. Readiness remains false."
                ),
            },
            reason="toolbox_setup_interrupted_after_dispatch",
        )

    def _run_toolbox_setup(self, *, operation_id: str) -> None:
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)

        def checkpoint(
            phase: str,
            code: str,
            completed_units: int | None,
            total_units: int | None,
            summary: str,
            _cancellable: bool = False,
        ) -> None:
            self._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=HostedOperationProgress(
                    phase=phase,
                    code=code,
                    completed_units=completed_units,
                    total_units=total_units,
                    updated_at_ms=int(time.time() * 1000),
                    summary=summary,
                    cancellable=False,
                ),
            )

        try:
            startup = self._resolve_configured_toolbox_startup(progress=checkpoint)
            if not isinstance(startup, dict) or startup.get("status") != "resolved":
                diagnostic = dict((startup or {}).get("diagnostics", [{}])[0] or {})
                code = str(diagnostic.get("code") or "required_builtin_resolution_not_ready")
                summary = str(
                    diagnostic.get("summary")
                    or "The required built-in wheel closure is not ready."
                )
                self._toolbox_setup_operation = self._hosted_operations.finish(
                    operation_id=operation_id,
                    lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                    envelope={"status": "not_ready", "code": code, "summary": summary},
                    reason=code,
                )
                return
            artifacts = [
                artifact
                for closure in startup["closures"]
                for artifact in closure["locked_artifacts"]
            ]
            checkpoint(
                "acquisition", "builtin_artifacts_available",
                sum(item["size_bytes"] for item in artifacts),
                sum(item["size_bytes"] for item in artifacts),
                "All exact built-in wheel bytes are available in verified storage."
            )
            prepared = self.prepare_configured_toolbox_templates(progress=checkpoint)
            checkpoint(
                "prewarm", "builtin_candidates_prewarmed",
                len(prepared["candidates"]), len(prepared["candidates"]),
                "All required built-in candidates are materialized and probed."
            )
            checkpoint(
                "publication", "builtin_publication_committing", 0, 1,
                "The complete built-in receipt and catalog batch is being committed."
            )
            result = self.publish_prepared_configured_toolbox_templates(prepared)
            checkpoint(
                "publication", "builtin_publication_committed", 1, 1,
                "The complete built-in receipt and catalog batch is active."
            )
            self._toolbox_setup_operation = self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope={**result, "status": "ok", "code": "toolbox_setup_ready"},
            )
        except Exception as exc:
            candidate_code = str(getattr(exc, "code", "") or str(exc)).strip()
            code = (
                candidate_code
                if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", candidate_code)
                else "toolbox_setup_failed"
            )
            self._toolbox_setup_operation = self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "code": code,
                    "summary": "Toolbox built-in setup failed before complete publication.",
                },
                reason=code,
            )

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
            receipts=self._template_receipts(entry["template_digest"]),
        )

    def toolbox_template_prewarm(
        self,
        *,
        template_id: str,
        python_abi: str,
        platform: str,
        request_id: str,
        template_digest: str | None = None,
        owner_actor_id: str = "service:local",
    ) -> dict[str, Any]:
        """Persist and dispatch exact-revision materialization on this runtime host."""

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
                item for item in state["entries"]
                if item["template_id"] == target_id and item["template_digest"] == target_digest
            ),
            None,
        )
        if entry is None:
            raise ValueError("template_revision_not_found")
        if entry["lifecycle"] == "revoked":
            raise ValueError("template_revision_revoked")
        target = materialization_target(python_abi=python_abi, platform=platform)
        template = ToolboxEnvironmentTemplateSpec.from_dict(entry["template"])
        if python_abi not in template.python_abis or platform not in template.platforms:
            raise ValueError("template_target_unsupported")
        rid = str(request_id or "").strip()
        if not rid:
            raise ValueError("template_prewarm_request_id_required")
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.TOOLBOX_TEMPLATE_PREWARM.value,
                "template_id": target_id,
                "template_digest": target_digest,
                "target": target,
                "catalog_revision": state["catalog_revision"],
            }
        )
        owner = self._operation_owner(owner_actor_id)
        prepared = self._hosted_operations.prepare(
            owner_actor_id=owner,
            execution_kind=HostedExecutionKind.TOOLBOX_TEMPLATE_PREWARM,
            selector=HostedOperationSelector(kind="template_id", id=target_id),
            namespace=f"environment_template_prewarm:{target_id}",
            request_id=rid,
            fingerprint=fingerprint,
            metadata={"template_digest": target_digest, "target": target},
        )
        if prepared["action"] != "dispatch":
            status = prepared.get("status")
            if status is None:
                raise RuntimeError("hosted_operation_capacity")
            return dict(status)
        operation_id = str(prepared["status"]["operation"]["operation_id"])
        thread = threading.Thread(
            target=self._run_toolbox_template_prewarm,
            kwargs={
                "operation_id": operation_id,
                "entry": dict(entry),
                "python_abi": python_abi,
                "platform": platform,
            },
            name=f"environment-template-prewarm-{operation_id[-8:]}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "code": "template_prewarm_dispatch_failed",
                    "summary": "The target-host materialization task could not be started.",
                },
                reason="template_prewarm_dispatch_failed",
            )
            raise
        return dict(prepared["status"])

    def _run_toolbox_template_prewarm(
        self,
        *,
        operation_id: str,
        entry: Mapping[str, Any],
        python_abi: str,
        platform: str,
    ) -> None:
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)

        def checkpoint(
            phase: str,
            code: str,
            completed_units: int | None,
            total_units: int | None,
            summary: str,
            cancellable: bool,
        ) -> None:
            self._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=HostedOperationProgress(
                    phase=phase,
                    code=code,
                    completed_units=completed_units,
                    total_units=total_units,
                    updated_at_ms=int(time.time() * 1000),
                    summary=summary,
                    cancellable=cancellable,
                ),
            )

        try:
            checkpoint("validation", "template_revision_validated", 1, 1, "The exact template revision and target were validated.", True)
            receipt = self._toolbox_template_materializer.materialize(
                catalog_entry=dict(entry),
                python_abi=python_abi,
                platform=platform,
                progress=checkpoint,
            )
            if not isinstance(receipt, ToolboxTemplateMaterializationReceipt):
                raise ToolboxTemplateMaterializationError(
                    "template_materialization_receipt_invalid",
                    "The target-host materializer returned an invalid verification receipt.",
                )
            if (
                receipt.template_id != entry["template_id"]
                or receipt.template_digest != entry["template_digest"]
                or receipt.python_abi != python_abi
                or receipt.platform != platform
            ):
                raise ToolboxTemplateMaterializationError(
                    "template_materialization_receipt_mismatch",
                    "The verification receipt did not match the requested template revision and target.",
                )
            expected_artifacts = sorted(item["sha256"] for item in entry["artifacts"])
            expected_roots = sorted(entry["template"]["exposed_import_roots"])
            expected_environment_digest = derived_environment_digest(
                template_digest=entry["template_digest"],
                python_abi=python_abi,
                platform=platform,
                artifact_digests=expected_artifacts,
            )
            if (
                sorted(receipt.artifact_digests) != expected_artifacts
                or sorted(receipt.verified_import_roots) != expected_roots
                or receipt.environment_digest != expected_environment_digest
            ):
                raise ToolboxTemplateMaterializationError(
                    "template_materialization_verification_incomplete",
                    "The verification receipt did not cover the complete artifact lock and import probes.",
                )
            checkpoint("receipt_commit", "verification_receipt_committing", 0, 1, "The verified materialization receipt is being committed.", False)
            self._toolbox_materialization_receipts.put(receipt)
            checkpoint("receipt_commit", "verification_receipt_committed", 1, 1, "The verified materialization receipt was committed.", False)
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope={
                    "status": "ok",
                    "code": "template_materialization_verified",
                    "template_id": receipt.template_id,
                    "template_digest": receipt.template_digest,
                    "python_abi": receipt.python_abi,
                    "platform": receipt.platform,
                    "environment_digest": receipt.environment_digest,
                    "verified_at_ms": receipt.verified_at_ms,
                },
            )
        except ToolboxTemplateMaterializationError as exc:
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={"status": "error", "code": exc.code, "summary": exc.summary},
                reason=exc.code,
            )
        except Exception:
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "code": "template_materialization_failed",
                    "summary": "The target-host materialization failed verification.",
                },
                reason="template_materialization_failed",
            )

    def toolbox_template_construct(
        self,
        *,
        template_id: str,
        base_template_digest: str,
        imports: Sequence[str],
        package_requirements: Sequence[str],
        request_id: str,
        owner_actor_id: str = "service:local",
    ) -> dict[str, Any]:
        configuration = getattr(self, "_toolbox_host_project_config", None)
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        target_id = _bounded_id(template_id, label="template_id")
        base_digest = require_digest(base_template_digest, label="base_template_digest")
        intent = ToolboxBuiltinIntent(
            template_id=target_id,
            imports=tuple(imports),
            package_requirements=tuple(package_requirements),
            sandbox_policy="compute-only",
            required=False,
            prewarm=False,
            provenance=f"constructed-from:{base_digest}",
        )
        state = self._toolbox_template_catalog.read()
        base_entry = next(
            (item for item in state["entries"] if item["template_digest"] == base_digest),
            None,
        )
        if base_entry is None:
            raise ValueError("base_template_revision_not_found")
        if base_entry["lifecycle"] == "revoked":
            raise ValueError("base_template_revision_revoked")
        rid = str(request_id or "").strip()
        if not rid:
            raise ValueError("template_construct_request_id_required")
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.TOOLBOX_TEMPLATE_CONSTRUCT.value,
                "template_id": target_id,
                "base_template_digest": base_digest,
                "imports": list(intent.imports),
                "package_requirements": list(intent.package_requirements),
                "config_revision": configuration.config_revision,
                "source_set_revision": configuration.source_set_revision,
                "target": configuration.target.name,
            }
        )
        owner = self._operation_owner(owner_actor_id)
        prepared = self._hosted_operations.prepare(
            owner_actor_id=owner,
            execution_kind=HostedExecutionKind.TOOLBOX_TEMPLATE_CONSTRUCT,
            selector=HostedOperationSelector(kind="template_id", id=target_id),
            namespace=f"environment_template_construct:{target_id}",
            request_id=rid,
            fingerprint=fingerprint,
            metadata={
                "base_template_digest": base_digest,
                "config_revision": configuration.config_revision,
                "source_set_revision": configuration.source_set_revision,
                "target": configuration.target.name,
            },
        )
        if prepared["action"] != "dispatch":
            status = prepared.get("status")
            if status is None:
                raise RuntimeError("hosted_operation_capacity")
            return dict(status)
        operation_id = str(prepared["status"]["operation"]["operation_id"])
        thread = threading.Thread(
            target=self._run_toolbox_template_construct,
            kwargs={
                "operation_id": operation_id,
                "intent": intent,
                "base_entry": dict(base_entry),
                "owner_actor_id": owner,
            },
            name=f"environment-template-construct-{operation_id[-8:]}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "code": "template_construct_dispatch_failed",
                    "summary": "The template construction worker could not be started.",
                },
                reason="template_construct_dispatch_failed",
            )
            raise
        return dict(prepared["status"])

    def _resolve_constructed_template_candidate(
        self,
        *,
        intent: ToolboxBuiltinIntent,
        base_entry: Mapping[str, Any],
        progress: Callable[[str, str, int | None, int | None, str, bool], None],
    ) -> tuple[ToolboxEnvironmentTemplateSpec, tuple[ToolboxTemplateArtifactReference, ...], str]:
        configuration = self._toolbox_host_project_config
        base = ToolboxEnvironmentTemplateSpec.from_dict(base_entry["template"])
        requirements = tuple(
            sorted(
                {
                    *(f"{item.name}=={item.version}" for item in base.locked_distributions),
                    *intent.package_requirements,
                }
            )
        )
        combined_intent = ToolboxBuiltinIntent(
            template_id=intent.template_id,
            imports=tuple(sorted(set(base.exposed_import_roots) | set(intent.imports))),
            package_requirements=requirements,
            sandbox_policy="compute-only",
            required=False,
            prewarm=False,
            provenance=intent.provenance,
        )
        progress("resolution", "template_requirements_resolving", 0, 1, "The exact base and requested package closure are being resolved.", True)
        resolver = AirgapBuiltinWheelResolver(
            configuration,
            artifact_sources=(
                {} if self._toolbox_trust_public_keys is not None else self._toolbox_artifact_sources
            ),
            verified_artifacts=self._toolbox_verified_artifacts,
        )
        try:
            closure = resolver.resolve_requirements(
                template_id=intent.template_id,
                package_requirements=requirements,
            )
        except RuntimeError:
            if configuration.resolution.mode not in {"online", "prefer_airgap"}:
                raise
            acquisition = ToolboxHttpsArtifactAcquirer(
                configuration,
                artifact_store=self._toolbox_artifact_store,
                trust_public_keys=self._toolbox_trust_public_keys,
                source_credentials=self._toolbox_source_credentials,
            ).discover_and_acquire(requirements, progress=progress)
            self._toolbox_verified_artifacts.update(acquisition["verified_artifacts"])
            self._configure_verified_toolbox_artifact_paths()
            closure = AirgapBuiltinWheelResolver(
                configuration,
                artifact_sources={},
                verified_artifacts=self._toolbox_verified_artifacts,
            ).resolve_requirements(
                template_id=intent.template_id,
                package_requirements=requirements,
            )
        progress("resolution", "template_requirements_resolved", 1, 1, "The exact current-host package closure was resolved.", True)
        progress("artifact_verification", "template_artifacts_verifying", 0, len(closure.locked_artifacts), "The complete resolved artifact closure is being verified.", True)
        evidence = self._toolbox_artifact_store.verified_evidence_for_artifacts(
            {item.sha256 for item in closure.locked_artifacts},
            source_ids={item.source_id for item in closure.locked_artifacts},
        )
        candidate = resolved_builtin_template_candidate(
            intent=combined_intent,
            closure=closure,
            target=configuration.target,
            evidence=evidence,
        )
        template_row = candidate.template.to_dict()
        template_row["provenance"] = {
            **template_row["provenance"],
            "source": f"constructed:{base_entry['template_digest']}",
            "revision": identity_digest(
                "hosting.toolbox.constructed_template_revision.v1",
                {
                    "base_template_digest": base_entry["template_digest"],
                    "lock_digest": candidate.template.lock_digest,
                    "imports": list(combined_intent.imports),
                },
            ),
        }
        template = ToolboxEnvironmentTemplateSpec.from_dict(template_row)
        artifacts = tuple(
            ToolboxTemplateArtifactReference.from_dict(item)
            for item in candidate.artifact_references
        )
        progress("artifact_verification", "template_artifacts_verified", len(artifacts), len(artifacts), "The complete resolved artifact closure was verified.", True)
        return template, artifacts, candidate.manifest_signature

    def _run_toolbox_template_construct(
        self,
        *,
        operation_id: str,
        intent: ToolboxBuiltinIntent,
        base_entry: Mapping[str, Any],
        owner_actor_id: str,
    ) -> None:
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)

        def checkpoint(phase, code, completed_units, total_units, summary, cancellable=True):
            self._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=HostedOperationProgress(
                    phase=phase,
                    code=code,
                    completed_units=completed_units,
                    total_units=total_units,
                    updated_at_ms=int(time.time() * 1000),
                    summary=summary,
                    cancellable=cancellable,
                ),
            )

        reference: tuple[str, str] | None = None
        receipt: ToolboxTemplateMaterializationReceipt | None = None
        try:
            checkpoint("validation", "template_construct_validated", 1, 1, "The exact base revision and construction request were validated.", True)
            template, artifacts, signature = self._resolve_constructed_template_candidate(
                intent=intent, base_entry=base_entry, progress=checkpoint
            )
            template_digest = _template_digest(template, artifacts, signature)
            entry = {
                "template_id": template.template_id,
                "template_digest": template_digest,
                "template": template.to_dict(),
                "artifacts": [item.to_dict() for item in artifacts],
                "manifest_signature": signature,
            }
            resolved = HermeticToolboxTemplateMaterializer._resolved_input(
                entry,
                python_abi=self._toolbox_host_project_config.target.python_abi,
                platform=self._toolbox_host_project_config.target.platform,
            )
            reference = (
                resolved.environment_key,
                f"template:{template_digest.removeprefix('sha256:')}",
            )
            receipt = self._toolbox_template_materializer.materialize(
                catalog_entry=entry,
                python_abi=self._toolbox_host_project_config.target.python_abi,
                platform=self._toolbox_host_project_config.target.platform,
                progress=checkpoint,
            )
            if (
                not isinstance(receipt, ToolboxTemplateMaterializationReceipt)
                or receipt.template_digest != template_digest
                or tuple(sorted(receipt.artifact_digests))
                != tuple(sorted(item.sha256 for item in artifacts))
                or tuple(sorted(receipt.verified_import_roots))
                != tuple(sorted(template.exposed_import_roots))
            ):
                raise ToolboxTemplateMaterializationError(
                    "template_materialization_receipt_mismatch",
                    "The constructed template receipt did not cover its complete exact candidate.",
                )
            checkpoint("receipt_commit", "template_receipt_committing", 0, 1, "The verified construction receipt is being committed.", False)
            self._toolbox_materialization_receipts.put(receipt)
            checkpoint("receipt_commit", "template_receipt_committed", 1, 1, "The verified construction receipt was committed.", False)
            checkpoint("publication", "inactive_template_publishing", 0, 1, "The immutable inactive template revision is being published.", False)
            published = self._toolbox_template_catalog.publish_inactive(
                template=template,
                artifacts=artifacts,
                manifest_signature=signature,
                actor_id=owner_actor_id,
            )
            checkpoint("publication", "inactive_template_published", 1, 1, "The immutable template revision was published inactive.", False)
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope={
                    "contract": "hosting.toolbox.template_construct_result.v1",
                    "status": "ok",
                    "code": "template_constructed_inactive",
                    "template_id": template.template_id,
                    "template_digest": template_digest,
                    "base_template_digest": base_entry["template_digest"],
                    "environment_digest": receipt.environment_digest,
                    "lifecycle": published["entry"]["lifecycle"],
                    "active_revision": False,
                    "catalog_revision": published["catalog_revision"],
                },
            )
        except Exception as exc:
            if receipt is not None:
                self._toolbox_materialization_receipts.remove_exact((receipt,))
            builder = getattr(self, "_hermetic_toolbox_environment_builder", None)
            if reference is not None and builder is not None:
                builder.release_reference(environment_key=reference[0], reference_id=reference[1])
            code = str(getattr(exc, "code", "") or str(exc)).strip()
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code):
                code = "template_construct_failed"
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "code": code,
                    "summary": "Template construction failed before inactive publication.",
                },
                reason=code,
            )

    def toolbox_template_activate(
        self,
        *,
        template_id: str,
        template_digest: str,
        actor_id: str = "service:local",
    ) -> dict[str, Any]:
        result = self._toolbox_template_catalog.activate(
            template_id=template_id,
            template_digest=template_digest,
            actor_id=actor_id,
        )
        return {
            "template_id": template_id,
            "template_digest": template_digest,
            "lifecycle": result["entry"]["lifecycle"],
            "active_revision": True,
            "catalog_revision": result["catalog_revision"],
            "outcome": result["outcome"],
        }

    def toolbox_template_replace(
        self,
        *,
        template_id: str,
        expected_active_digest: str,
        replacement_digest: str,
        actor_id: str = "service:local",
    ) -> dict[str, Any]:
        result = self._toolbox_template_catalog.replace(
            template_id=template_id,
            expected_active_digest=expected_active_digest,
            replacement_digest=replacement_digest,
            actor_id=actor_id,
        )
        return {
            "template_id": template_id,
            "template_digest": replacement_digest,
            "replaced_template_digest": result["replaced_template_digest"],
            "lifecycle": result["entry"]["lifecycle"],
            "active_revision": True,
            "catalog_revision": result["catalog_revision"],
            "outcome": result["outcome"],
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
