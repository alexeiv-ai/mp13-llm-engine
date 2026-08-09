"""Toolbox-only immutable environment identity and hermetic build inputs."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import ToolboxLockedDistributionSpec, normalize_import_root
from .identity import ENVIRONMENT_IDENTITY_DOMAIN, environment_identity, require_digest


TOOLBOX_ENVIRONMENT_KEY_DOMAIN = ENVIRONMENT_IDENTITY_DOMAIN
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def _strict_fields(row: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    unknown = sorted(set(row) - fields)
    missing = sorted(fields - set(row))
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")
    if missing:
        raise ValueError(f"{label}_missing_fields:{','.join(missing)}")


def _id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value.strip()):
        raise ValueError(f"{label}_invalid")
    return value.strip()


@dataclass(frozen=True)
class ResolvedToolboxEnvironmentInput:
    """Complete host-derived input; never accepts a venv or environment name."""

    template_id: str
    template_digest: str
    runtime_version: str
    runtime_artifact_digest: str
    python_abi: str
    platform: str
    complete_lock_digest: str
    complete_lock: tuple[ToolboxLockedDistributionSpec, ...]
    custom_resolved_lock_digest: str | None
    isolation_policy_version: str
    resolved_import_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "template_id", _id(self.template_id, label="resolved_template_id"))
        object.__setattr__(self, "template_digest", require_digest(self.template_digest, label="resolved_template_digest"))
        object.__setattr__(self, "runtime_version", _id(self.runtime_version, label="resolved_runtime_version"))
        object.__setattr__(self, "runtime_artifact_digest", require_digest(self.runtime_artifact_digest, label="resolved_runtime_artifact_digest"))
        if not re.fullmatch(r"cp[0-9]{3,4}", str(self.python_abi or "")):
            raise ValueError("resolved_python_abi_invalid")
        if self.platform not in {"win_amd64", "manylinux_2_28_x86_64"}:
            raise ValueError("resolved_platform_invalid")
        object.__setattr__(self, "complete_lock_digest", require_digest(self.complete_lock_digest, label="resolved_complete_lock_digest"))
        lock = tuple(self.complete_lock)
        if any(not isinstance(item, ToolboxLockedDistributionSpec) for item in lock):
            raise ValueError("resolved_complete_lock_item_invalid")
        if not lock or tuple(sorted(lock)) != lock or len({item.name for item in lock}) != len(lock):
            raise ValueError("resolved_complete_lock_invalid")
        object.__setattr__(self, "complete_lock", lock)
        if self.custom_resolved_lock_digest is not None:
            object.__setattr__(
                self,
                "custom_resolved_lock_digest",
                require_digest(self.custom_resolved_lock_digest, label="resolved_custom_lock_digest"),
            )
        object.__setattr__(
            self,
            "isolation_policy_version",
            _id(self.isolation_policy_version, label="resolved_isolation_policy_version"),
        )
        roots = tuple(sorted(normalize_import_root(item) for item in self.resolved_import_roots))
        if len(set(roots)) != len(roots):
            raise ValueError("resolved_import_roots_duplicate")
        object.__setattr__(self, "resolved_import_roots", roots)

    @property
    def environment_key(self) -> str:
        return environment_identity(
            runtime_identity={
                "runtime_version": self.runtime_version,
                "runtime_artifact_digest": self.runtime_artifact_digest,
                "python_abi": self.python_abi,
                "platform": self.platform,
            },
            template_lock_digest=self.complete_lock_digest,
            custom_lock_digest=self.custom_resolved_lock_digest,
            isolation_policy={"version": self.isolation_policy_version},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "template_digest": self.template_digest,
            "runtime_version": self.runtime_version,
            "runtime_artifact_digest": self.runtime_artifact_digest,
            "python_abi": self.python_abi,
            "platform": self.platform,
            "complete_lock_digest": self.complete_lock_digest,
            "complete_lock": [item.to_dict() for item in self.complete_lock],
            "custom_resolved_lock_digest": self.custom_resolved_lock_digest,
            "isolation_policy_version": self.isolation_policy_version,
            "resolved_import_roots": list(self.resolved_import_roots),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedToolboxEnvironmentInput":
        row = dict(payload or {})
        fields = {
            "template_id", "template_digest", "runtime_version",
            "runtime_artifact_digest", "python_abi", "platform",
            "complete_lock_digest", "complete_lock", "custom_resolved_lock_digest",
            "isolation_policy_version", "resolved_import_roots",
        }
        _strict_fields(row, fields, label="resolved_toolbox_environment")
        if not isinstance(row["complete_lock"], Sequence) or isinstance(row["complete_lock"], (str, bytes, bytearray)):
            raise ValueError("resolved_complete_lock_invalid")
        if not isinstance(row["resolved_import_roots"], Sequence) or isinstance(row["resolved_import_roots"], (str, bytes, bytearray)):
            raise ValueError("resolved_import_roots_invalid")
        return cls(
            **{
                **row,
                "complete_lock": tuple(ToolboxLockedDistributionSpec.from_dict(item) for item in row["complete_lock"]),
                "resolved_import_roots": tuple(row["resolved_import_roots"]),
            }
        )


@dataclass(frozen=True)
class HermeticToolboxEnvironmentSpec:
    resolved: ResolvedToolboxEnvironmentInput
    environment_key: str
    environment_root: str
    python_executable: str

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedToolboxEnvironmentInput):
            raise ValueError("hermetic_environment_resolved_invalid")
        if self.environment_key != self.resolved.environment_key:
            raise ValueError("hermetic_environment_key_mismatch")
        root = Path(self.environment_root).expanduser().resolve()
        python = Path(self.python_executable).expanduser().resolve()
        expected = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if python != expected:
            raise ValueError("hermetic_environment_python_path_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved.to_dict(),
            "environment_key": self.environment_key,
            "environment_root": self.environment_root,
            "python_executable": self.python_executable,
        }


class HermeticToolboxEnvironmentResolver:
    """Pure toolbox resolver; physical building is supplied by the next slice."""

    def __init__(self, hosting_root: Path):
        self.hosting_root = Path(hosting_root).expanduser().resolve()
        self.environments_root = (self.hosting_root / "toolbox_environment_cache").resolve()

    def environment_spec(
        self, resolved: ResolvedToolboxEnvironmentInput | Mapping[str, Any]
    ) -> HermeticToolboxEnvironmentSpec:
        model = (
            resolved
            if isinstance(resolved, ResolvedToolboxEnvironmentInput)
            else ResolvedToolboxEnvironmentInput.from_dict(resolved)
        )
        root = (self.environments_root / model.environment_key.removeprefix("sha256:")).resolve()
        python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return HermeticToolboxEnvironmentSpec(
            resolved=model,
            environment_key=model.environment_key,
            environment_root=str(root),
            python_executable=str(python),
        )


__all__ = [
    "HermeticToolboxEnvironmentResolver",
    "HermeticToolboxEnvironmentSpec",
    "ResolvedToolboxEnvironmentInput",
    "TOOLBOX_ENVIRONMENT_KEY_DOMAIN",
]
