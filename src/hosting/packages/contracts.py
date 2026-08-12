"""Strict worker-neutral package contracts."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence


_ID = re.compile(r"[\x21-\x7e]{1,128}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _strict(row: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    value = dict(row or {})
    if set(value) != fields:
        raise ValueError(f"{label}_fields_invalid")
    return value


def _id(value: Any, label: str) -> str:
    text = str(value or "")
    if not _ID.fullmatch(text):
        raise ValueError(f"{label}_invalid")
    return text


def require_digest(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if not _DIGEST.fullmatch(text):
        raise ValueError(f"{label}_invalid")
    return text


class PackageVerifier(Protocol):
    """Optional external publisher verifier; absent in the baseline."""

    def verify(self, path: str, *, source: "PackageSource") -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PackageSource:
    source_id: str
    kind: str
    locator: str
    credential_ref: Optional[str]
    enabled: bool
    priority: int

    CONTRACT = "hosting.package_source.v1"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PackageSource":
        row = _strict(
            payload,
            {"contract", "source_id", "kind", "locator", "credential_ref", "enabled", "priority"},
            "package_source",
        )
        if row["contract"] != cls.CONTRACT:
            raise ValueError("package_source_contract_unsupported")
        if not isinstance(row["enabled"], bool) or isinstance(row["priority"], bool) or not isinstance(row["priority"], int):
            raise ValueError("package_source_type_invalid")
        credential = row["credential_ref"]
        if credential is not None:
            credential = _id(credential, "package_source_credential_ref")
        locator = str(row["locator"] or "").strip()
        if not locator:
            raise ValueError("package_source_locator_invalid")
        return cls(
            source_id=_id(row["source_id"], "package_source_id"),
            kind=_id(row["kind"], "package_source_kind"),
            locator=locator,
            credential_ref=credential,
            enabled=row["enabled"],
            priority=row["priority"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.CONTRACT, **self.__dict__}


@dataclass(frozen=True)
class PackagePolicy:
    policy_id: str
    revision: int
    allowed_source_ids: tuple[str, ...]
    allowed_platforms: tuple[str, ...]
    allowed_runtimes: tuple[str, ...]
    max_artifact_bytes: int
    require_sha256: bool
    optional_verifier: Optional[str]

    CONTRACT = "hosting.package_policy.v1"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PackagePolicy":
        row = _strict(
            payload,
            {"contract", "policy_id", "revision", "allowed_source_ids", "allowed_platforms", "allowed_runtimes", "max_artifact_bytes", "require_sha256", "optional_verifier"},
            "package_policy",
        )
        if row["contract"] != cls.CONTRACT:
            raise ValueError("package_policy_contract_unsupported")
        for key in ("allowed_source_ids", "allowed_platforms", "allowed_runtimes"):
            if not isinstance(row[key], list) or any(not isinstance(item, str) for item in row[key]):
                raise ValueError("package_policy_type_invalid")
        if (
            isinstance(row["revision"], bool) or not isinstance(row["revision"], int) or row["revision"] < 1
            or isinstance(row["max_artifact_bytes"], bool) or not isinstance(row["max_artifact_bytes"], int) or row["max_artifact_bytes"] < 1
            or not isinstance(row["require_sha256"], bool)
            or (row["optional_verifier"] is not None and not isinstance(row["optional_verifier"], str))
        ):
            raise ValueError("package_policy_type_invalid")
        return cls(
            policy_id=_id(row["policy_id"], "package_policy_id"),
            revision=row["revision"],
            allowed_source_ids=tuple(_id(item, "package_source_id") for item in row["allowed_source_ids"]),
            allowed_platforms=tuple(_id(item, "package_platform") for item in row["allowed_platforms"]),
            allowed_runtimes=tuple(_id(item, "package_runtime") for item in row["allowed_runtimes"]),
            max_artifact_bytes=row["max_artifact_bytes"],
            require_sha256=row["require_sha256"],
            optional_verifier=row["optional_verifier"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "policy_id": self.policy_id,
            "revision": self.revision,
            "allowed_source_ids": list(self.allowed_source_ids),
            "allowed_platforms": list(self.allowed_platforms),
            "allowed_runtimes": list(self.allowed_runtimes),
            "max_artifact_bytes": self.max_artifact_bytes,
            "require_sha256": self.require_sha256,
            "optional_verifier": self.optional_verifier,
        }


@dataclass(frozen=True)
class PackageLock:
    lock_id: str
    revision: int
    policy_id: str
    policy_revision: int
    artifacts: tuple[Mapping[str, Any], ...]
    dependencies: tuple[Mapping[str, Any], ...]
    lock_digest: str

    CONTRACT = "hosting.package_lock.v1"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PackageLock":
        row = _strict(
            payload,
            {
                "contract", "lock_id", "revision", "policy_id", "policy_revision",
                "artifacts", "dependencies", "lock_digest",
            },
            "package_lock",
        )
        if row["contract"] != cls.CONTRACT:
            raise ValueError("package_lock_contract_unsupported")
        if (
            isinstance(row["revision"], bool)
            or not isinstance(row["revision"], int)
            or row["revision"] < 1
            or isinstance(row["policy_revision"], bool)
            or not isinstance(row["policy_revision"], int)
            or row["policy_revision"] < 1
            or not isinstance(row["artifacts"], list)
            or not isinstance(row["dependencies"], list)
            or any(not isinstance(item, Mapping) for item in row["artifacts"])
            or any(not isinstance(item, Mapping) for item in row["dependencies"])
        ):
            raise ValueError("package_lock_type_invalid")
        policy = PackagePolicy(
            policy_id=_id(row["policy_id"], "package_policy_id"),
            revision=row["policy_revision"],
            allowed_source_ids=tuple(
                sorted({_id(item.get("source_id"), "package_source_id") for item in row["artifacts"]})
            ),
            allowed_platforms=(),
            allowed_runtimes=(),
            max_artifact_bytes=1,
            require_sha256=True,
            optional_verifier=None,
        )
        rebuilt = cls.build(
            lock_id=row["lock_id"],
            revision=row["revision"],
            policy=policy,
            artifacts=row["artifacts"],
            dependencies=row["dependencies"],
        )
        expected = require_digest(row["lock_digest"], "package_lock_digest")
        if rebuilt.lock_digest != expected:
            raise ValueError("package_lock_digest_mismatch")
        return rebuilt

    @staticmethod
    def build(
        *,
        lock_id: str,
        revision: int,
        policy: PackagePolicy,
        artifacts: Sequence[Mapping[str, Any]],
        dependencies: Sequence[Mapping[str, Any]],
    ) -> "PackageLock":
        normalized_artifacts = []
        for raw in artifacts:
            row = _strict(raw, {"artifact_id", "size_bytes", "source_id"}, "package_lock_artifact")
            size = row["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError("package_artifact_size_invalid")
            normalized_artifacts.append(
                {
                    "artifact_id": require_digest(row["artifact_id"], "package_artifact_id"),
                    "size_bytes": size,
                    "source_id": _id(row["source_id"], "package_source_id"),
                }
            )
        normalized_dependencies = []
        available = {item["artifact_id"] for item in normalized_artifacts}
        for raw in dependencies:
            row = _strict(raw, {"name", "version", "artifact_id"}, "package_lock_dependency")
            artifact_id = require_digest(row["artifact_id"], "package_artifact_id")
            if artifact_id not in available:
                raise ValueError("package_dependency_artifact_missing")
            normalized_dependencies.append(
                {
                    "name": _id(row["name"], "package_name"),
                    "version": _id(row["version"], "package_version"),
                    "artifact_id": artifact_id,
                }
            )
        normalized_lock_id = _id(lock_id, "package_lock_id")
        normalized_revision = int(revision)
        if normalized_revision < 1:
            raise ValueError("package_lock_revision_invalid")
        sorted_artifacts = sorted(
            normalized_artifacts, key=lambda item: (item["source_id"], item["artifact_id"])
        )
        sorted_dependencies = sorted(
            normalized_dependencies,
            key=lambda item: (item["name"], item["version"], item["artifact_id"]),
        )
        unsigned: dict[str, Any] = {
            "contract": PackageLock.CONTRACT,
            "lock_id": normalized_lock_id,
            "revision": normalized_revision,
            "policy_id": policy.policy_id,
            "policy_revision": policy.revision,
            "artifacts": sorted_artifacts,
            "dependencies": sorted_dependencies,
        }
        digest = "sha256:" + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return PackageLock(
            lock_id=normalized_lock_id,
            revision=normalized_revision,
            policy_id=policy.policy_id,
            policy_revision=policy.revision,
            artifacts=tuple(sorted_artifacts),
            dependencies=tuple(sorted_dependencies),
            lock_digest=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "lock_id": self.lock_id,
            "revision": self.revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "artifacts": [dict(item) for item in self.artifacts],
            "dependencies": [dict(item) for item in self.dependencies],
            "lock_digest": self.lock_digest,
        }
