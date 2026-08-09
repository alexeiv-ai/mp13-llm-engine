"""Durable target-host verification receipts for toolbox template revisions."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..toolbox.identity import identity_digest, require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


MATERIALIZATION_STATE_CONTRACT = "hosting.toolbox.materialization_state.v1"
MATERIALIZATION_RECEIPT_CONTRACT = "hosting.toolbox.materialization_receipt.v1"
MATERIALIZATION_ENVIRONMENT_DOMAIN = "hosting.toolbox.materialized_environment.v1"
MAX_MATERIALIZATION_RECEIPTS = 1024
_TARGET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_VERIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class ToolboxTemplateMaterializationError(RuntimeError):
    """A bounded, user-safe materialization failure."""

    def __init__(self, code: str, summary: str):
        if not _VERIFIER_RE.fullmatch(str(code or "")):
            raise ValueError("materialization_error_code_invalid")
        text = str(summary or "").strip()
        if not text or len(text.encode("utf-8")) > 512:
            raise ValueError("materialization_error_summary_invalid")
        self.code = str(code)
        self.summary = text
        super().__init__(self.code)


def materialization_target(*, python_abi: str, platform: str) -> str:
    abi = str(python_abi or "").strip()
    platform_name = str(platform or "").strip()
    if not _TARGET_RE.fullmatch(abi):
        raise ValueError("template_python_abi_invalid")
    if not _TARGET_RE.fullmatch(platform_name):
        raise ValueError("template_platform_invalid")
    return f"{abi}@{platform_name}"


@dataclass(frozen=True)
class ToolboxTemplateMaterializationReceipt:
    template_id: str
    template_digest: str
    python_abi: str
    platform: str
    environment_digest: str
    artifact_digests: tuple[str, ...]
    verified_import_roots: tuple[str, ...]
    verified_at_ms: int
    verifier: str
    contract: str = MATERIALIZATION_RECEIPT_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != MATERIALIZATION_RECEIPT_CONTRACT:
            raise ValueError("materialization_receipt_contract_invalid")
        if not _VERIFIER_RE.fullmatch(str(self.template_id or "")):
            raise ValueError("materialization_template_id_invalid")
        require_digest(self.template_digest, label="materialization_template_digest")
        materialization_target(python_abi=self.python_abi, platform=self.platform)
        require_digest(self.environment_digest, label="materialization_environment_digest")
        artifacts = tuple(require_digest(item, label="materialization_artifact_digest") for item in self.artifact_digests)
        if not artifacts or len(artifacts) > 2048 or tuple(sorted(set(artifacts))) != tuple(sorted(artifacts)):
            raise ValueError("materialization_artifact_digests_invalid")
        roots = tuple(str(item or "").strip() for item in self.verified_import_roots)
        if (
            not roots
            or len(roots) > 2048
            or tuple(sorted(set(roots))) != tuple(sorted(roots))
            or any(not _VERIFIER_RE.fullmatch(item) for item in roots)
        ):
            raise ValueError("materialization_verified_import_roots_invalid")
        if isinstance(self.verified_at_ms, bool) or not isinstance(self.verified_at_ms, int) or self.verified_at_ms < 0:
            raise ValueError("materialization_verified_at_ms_invalid")
        if not _VERIFIER_RE.fullmatch(str(self.verifier or "")):
            raise ValueError("materialization_verifier_invalid")

    @property
    def target(self) -> str:
        return materialization_target(python_abi=self.python_abi, platform=self.platform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "template_id": self.template_id,
            "template_digest": self.template_digest,
            "python_abi": self.python_abi,
            "platform": self.platform,
            "environment_digest": self.environment_digest,
            "artifact_digests": list(self.artifact_digests),
            "verified_import_roots": list(self.verified_import_roots),
            "verified_at_ms": self.verified_at_ms,
            "verifier": self.verifier,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxTemplateMaterializationReceipt":
        row = dict(payload or {})
        fields = {
            "contract", "template_id", "template_digest", "python_abi", "platform",
            "environment_digest", "artifact_digests", "verified_import_roots",
            "verified_at_ms", "verifier",
        }
        if set(row) != fields:
            raise ValueError("materialization_receipt_fields_invalid")
        return cls(
            contract=row["contract"],
            template_id=row["template_id"],
            template_digest=row["template_digest"],
            python_abi=row["python_abi"],
            platform=row["platform"],
            environment_digest=row["environment_digest"],
            artifact_digests=tuple(row["artifact_digests"]),
            verified_import_roots=tuple(row["verified_import_roots"]),
            verified_at_ms=row["verified_at_ms"],
            verifier=row["verifier"],
        )


MaterializationProgress = Callable[[str, str, int | None, int | None, str, bool], None]


class ToolboxTemplateMaterializer(Protocol):
    """Target-host builder boundary shared by prewarm and lazy apply."""

    def materialize(
        self,
        *,
        catalog_entry: Mapping[str, Any],
        python_abi: str,
        platform: str,
        progress: MaterializationProgress,
    ) -> ToolboxTemplateMaterializationReceipt: ...


class UnconfiguredToolboxTemplateMaterializer:
    """Fail closed until host setup supplies the physical builder (P1-11/P2)."""

    def materialize(
        self,
        *,
        catalog_entry: Mapping[str, Any],
        python_abi: str,
        platform: str,
        progress: MaterializationProgress,
    ) -> ToolboxTemplateMaterializationReceipt:
        del catalog_entry, python_abi, platform, progress
        raise ToolboxTemplateMaterializationError(
            "template_materializer_unconfigured",
            "This runtime host has no configured toolbox template materializer.",
        )


class AtomicJsonToolboxMaterializationReceipts:
    """Process-safe exact-revision receipt store; failed builds never become ready."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": MATERIALIZATION_STATE_CONTRACT, "receipts": {}}

    @classmethod
    def _validate(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {"contract", "receipts"} or row.get("contract") != MATERIALIZATION_STATE_CONTRACT:
            raise ValueError("materialization_state_contract_invalid")
        if not isinstance(row.get("receipts"), dict) or len(row["receipts"]) > MAX_MATERIALIZATION_RECEIPTS:
            raise ValueError("materialization_state_receipts_invalid")
        receipts: dict[str, dict[str, Any]] = {}
        for key, value in row["receipts"].items():
            receipt = ToolboxTemplateMaterializationReceipt.from_dict(value)
            expected = f"{receipt.template_digest}|{receipt.target}"
            if key != expected:
                raise ValueError("materialization_receipt_key_invalid")
            receipts[key] = receipt.to_dict()
        return {"contract": MATERIALIZATION_STATE_CONTRACT, "receipts": receipts}

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("materialization_state_corrupt") from exc
        if not isinstance(value, dict):
            raise ValueError("materialization_state_corrupt")
        return self._validate(value)

    def _write_unlocked(self, state: Mapping[str, Any]) -> None:
        value = self._validate(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_temp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(raw_temp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
            temporary.unlink(missing_ok=True)

    def get(self, *, template_digest: str, python_abi: str, platform: str) -> ToolboxTemplateMaterializationReceipt | None:
        key = f"{require_digest(template_digest, label='materialization_template_digest')}|{materialization_target(python_abi=python_abi, platform=platform)}"
        with _exclusive_process_file_lock(self.lock_path):
            row = self._read_unlocked()["receipts"].get(key)
            return ToolboxTemplateMaterializationReceipt.from_dict(row) if row is not None else None

    def list_for_template(self, *, template_digest: str) -> tuple[ToolboxTemplateMaterializationReceipt, ...]:
        digest = require_digest(template_digest, label="materialization_template_digest")
        with _exclusive_process_file_lock(self.lock_path):
            receipts = [
                ToolboxTemplateMaterializationReceipt.from_dict(row)
                for row in self._read_unlocked()["receipts"].values()
                if row.get("template_digest") == digest
            ]
        return tuple(sorted(receipts, key=lambda item: (item.target, item.verified_at_ms)))

    def put(self, receipt: ToolboxTemplateMaterializationReceipt) -> ToolboxTemplateMaterializationReceipt:
        if not isinstance(receipt, ToolboxTemplateMaterializationReceipt):
            raise ValueError("materialization_receipt_type_invalid")
        key = f"{receipt.template_digest}|{receipt.target}"
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            existing = state["receipts"].get(key)
            if existing is not None and existing != receipt.to_dict():
                raise ValueError("materialization_receipt_conflict")
            if existing is None and len(state["receipts"]) >= MAX_MATERIALIZATION_RECEIPTS:
                raise ValueError("materialization_receipt_capacity")
            state["receipts"][key] = receipt.to_dict()
            self._write_unlocked(state)
        return receipt


def derived_environment_digest(
    *, template_digest: str, python_abi: str, platform: str, artifact_digests: Sequence[str]
) -> str:
    return identity_digest(
        MATERIALIZATION_ENVIRONMENT_DOMAIN,
        {
            "template_digest": require_digest(template_digest, label="materialization_template_digest"),
            "python_abi": python_abi,
            "platform": platform,
            "artifact_digests": sorted(require_digest(item, label="materialization_artifact_digest") for item in artifact_digests),
        },
    )


__all__ = [
    "AtomicJsonToolboxMaterializationReceipts",
    "ToolboxTemplateMaterializationError",
    "ToolboxTemplateMaterializationReceipt",
    "ToolboxTemplateMaterializer",
    "UnconfiguredToolboxTemplateMaterializer",
    "derived_environment_digest",
    "materialization_target",
]
