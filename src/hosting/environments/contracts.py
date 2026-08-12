"""Strict worker-neutral environment records."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional


_ID = re.compile(r"[\x21-\x7e]{1,128}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _id(value: Any, label: str) -> str:
    text = str(value or "")
    if not _ID.fullmatch(text):
        raise ValueError(f"{label}_invalid")
    return text


def _digest(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if not _DIGEST.fullmatch(text):
        raise ValueError(f"{label}_invalid")
    return text


def _strict(payload: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    row = dict(payload or {})
    if set(row) != fields:
        raise ValueError(f"{label}_fields_invalid")
    return row


@dataclass(frozen=True)
class EnvironmentTemplate:
    template_id: str
    revision: int
    runtime_kind: str
    builder_id: str
    package_lock_id: str
    platforms: tuple[str, ...]
    state: str

    CONTRACT = "hosting.environment_template.v1"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnvironmentTemplate":
        row = _strict(payload, {"contract", "template_id", "revision", "runtime_kind", "builder_id", "package_lock_id", "platforms", "state"}, "environment_template")
        if row["contract"] != cls.CONTRACT:
            raise ValueError("environment_template_contract_unsupported")
        if isinstance(row["revision"], bool) or not isinstance(row["revision"], int) or row["revision"] < 1:
            raise ValueError("environment_template_revision_invalid")
        if not isinstance(row["platforms"], list) or not row["platforms"] or any(not isinstance(item, str) for item in row["platforms"]):
            raise ValueError("environment_template_platforms_invalid")
        state = str(row["state"])
        if state not in {"draft", "active", "deprecated", "revoked"}:
            raise ValueError("environment_template_state_invalid")
        return cls(
            template_id=_id(row["template_id"], "environment_template_id"),
            revision=row["revision"],
            runtime_kind=_id(row["runtime_kind"], "environment_runtime_kind"),
            builder_id=_id(row["builder_id"], "environment_builder_id"),
            package_lock_id=_id(row["package_lock_id"], "package_lock_id"),
            platforms=tuple(_id(item, "environment_platform") for item in row["platforms"]),
            state=state,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.CONTRACT, "template_id": self.template_id, "revision": self.revision, "runtime_kind": self.runtime_kind, "builder_id": self.builder_id, "package_lock_id": self.package_lock_id, "platforms": list(self.platforms), "state": self.state}


@dataclass(frozen=True)
class EnvironmentRequest:
    request_id: str
    consumer_kind: str
    consumer_id: str
    revision: int
    template_id: str
    template_revision: int
    package_lock_digest: str
    runtime_kind: str
    platform: str
    configuration_revision: str

    CONTRACT = "hosting.environment_request.v1"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnvironmentRequest":
        fields = {"contract", "request_id", "consumer_kind", "consumer_id", "revision", "template_id", "template_revision", "package_lock_digest", "runtime_kind", "platform", "configuration_revision"}
        row = _strict(payload, fields, "environment_request")
        if row["contract"] != cls.CONTRACT:
            raise ValueError("environment_request_contract_unsupported")
        for key in ("revision", "template_revision"):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 1:
                raise ValueError("environment_request_revision_invalid")
        return cls(
            request_id=_id(row["request_id"], "environment_request_id"),
            consumer_kind=_id(row["consumer_kind"], "environment_consumer_kind"),
            consumer_id=_id(row["consumer_id"], "environment_consumer_id"),
            revision=row["revision"],
            template_id=_id(row["template_id"], "environment_template_id"),
            template_revision=row["template_revision"],
            package_lock_digest=_digest(row["package_lock_digest"], "package_lock_digest"),
            runtime_kind=_id(row["runtime_kind"], "environment_runtime_kind"),
            platform=_id(row["platform"], "environment_platform"),
            configuration_revision=_digest(row["configuration_revision"], "configuration_revision"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.CONTRACT, **self.__dict__}


@dataclass(frozen=True)
class EnvironmentLock:
    environment_id: str
    content_key: str
    runtime_kind: str
    platform: str
    template_id: str
    template_revision: int
    package_lock_digest: str
    configuration_revision: str

    CONTRACT = "hosting.environment_lock.v1"

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.CONTRACT, **self.__dict__}


@dataclass(frozen=True)
class EnvironmentReceipt:
    environment_id: str
    content_key: str
    receipt_revision: int
    logical_root: str
    runtime_kind: str
    platform: str
    template_id: str
    template_revision: int
    package_lock_digest: str
    configuration_revision: str
    builder_result: Mapping[str, Any]

    CONTRACT = "hosting.environment_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.CONTRACT, **self.__dict__, "builder_result": dict(self.builder_result)}


@dataclass(frozen=True)
class EnvironmentReference:
    reference_id: str
    environment_id: str
    consumer_kind: str
    consumer_id: str
    revision: int
    acquired_at_ms: int
    released_at_ms: Optional[int]

    CONTRACT = "hosting.environment_reference.v1"

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.CONTRACT, **self.__dict__}
