"""Strict host-owned toolbox catalog and sandbox project configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .shipped_templates import (
    SHIPPED_CATALOG_RESOURCE,
    SHIPPED_TEMPLATE_IDS,
    compute_only_sandbox_policy,
)


_FIELDS = {
    "resource",
    "trusted_signing_key_ids",
    "required_template_ids",
    "required_target",
    "prewarm_required",
    "artifact_source_ids",
    "offline_preseed_source_id",
    "cache_grace_seconds",
    "build_timeout_seconds",
}
_SUPPORTED_TARGETS = {
    "cp312-win_amd64": ("cp312", "win_amd64"),
    "cp312-manylinux_2_28_x86_64": ("cp312", "manylinux_2_28_x86_64"),
}


def _strict_id_list(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}_invalid")
    rows = tuple(str(item or "").strip() for item in value)
    if any(not item or len(item.encode("utf-8")) > 128 for item in rows):
        raise ValueError(f"{label}_invalid")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{label}_duplicate")
    return rows


@dataclass(frozen=True)
class ToolboxHostProjectConfiguration:
    resource: str
    trusted_signing_key_ids: tuple[str, ...]
    required_template_ids: tuple[str, ...]
    required_target: str
    prewarm_required: bool
    artifact_source_ids: tuple[str, ...]
    offline_preseed_source_id: str | None
    cache_grace_seconds: int = 604_800
    build_timeout_seconds: int = 1_800

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolboxHostProjectConfiguration":
        if not isinstance(value, Mapping):
            raise ValueError("toolbox_environment_catalog_must_be_object")
        row = dict(value)
        unknown = sorted(set(row) - _FIELDS)
        missing = sorted(_FIELDS - set(row))
        if unknown:
            raise ValueError(f"toolbox_environment_catalog_unknown_fields:{','.join(unknown)}")
        if missing:
            raise ValueError(f"toolbox_environment_catalog_missing_fields:{','.join(missing)}")
        resource = str(row["resource"] or "").strip()
        if resource != SHIPPED_CATALOG_RESOURCE:
            raise ValueError("toolbox_environment_catalog_resource_invalid")
        trusted = _strict_id_list(row["trusted_signing_key_ids"], label="trusted_signing_key_ids")
        required = _strict_id_list(row["required_template_ids"], label="required_template_ids")
        if required != SHIPPED_TEMPLATE_IDS:
            raise ValueError("required_template_ids_invalid")
        target = str(row["required_target"] or "").strip().lower()
        if target not in _SUPPORTED_TARGETS:
            raise ValueError("required_target_invalid")
        if not isinstance(row["prewarm_required"], bool):
            raise ValueError("prewarm_required_must_be_boolean")
        sources = _strict_id_list(row["artifact_source_ids"], label="artifact_source_ids")
        offline_raw = row["offline_preseed_source_id"]
        offline = None if offline_raw is None else str(offline_raw or "").strip()
        if offline is not None and offline not in sources:
            raise ValueError("offline_preseed_source_id_invalid")
        grace = row["cache_grace_seconds"]
        timeout = row["build_timeout_seconds"]
        if isinstance(grace, bool) or not isinstance(grace, int) or not 86_400 <= grace <= 7_776_000:
            raise ValueError("cache_grace_seconds_invalid")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 60 <= timeout <= 1_800:
            raise ValueError("build_timeout_seconds_invalid")
        return cls(
            resource=resource,
            trusted_signing_key_ids=trusted,
            required_template_ids=required,
            required_target=target,
            prewarm_required=row["prewarm_required"],
            artifact_source_ids=sources,
            offline_preseed_source_id=offline,
            cache_grace_seconds=grace,
            build_timeout_seconds=timeout,
        )

    @property
    def target(self) -> tuple[str, str]:
        return _SUPPORTED_TARGETS[self.required_target]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "trusted_signing_key_ids": list(self.trusted_signing_key_ids),
            "required_template_ids": list(self.required_template_ids),
            "required_target": self.required_target,
            "prewarm_required": self.prewarm_required,
            "artifact_source_ids": list(self.artifact_source_ids),
            "offline_preseed_source_id": self.offline_preseed_source_id,
            "cache_grace_seconds": self.cache_grace_seconds,
            "build_timeout_seconds": self.build_timeout_seconds,
        }


def validate_toolbox_sandbox_policies(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"compute_only"}:
        raise ValueError("toolbox_sandbox_policies_invalid")
    configured = dict(value.get("compute_only") or {})
    expected = compute_only_sandbox_policy()
    if configured != expected:
        raise ValueError("compute_only_policy_invalid")
    return {"compute_only": expected}


def standard_toolbox_host_project_configuration(*, target: str) -> dict[str, Any]:
    return ToolboxHostProjectConfiguration.from_dict(
        {
            "resource": SHIPPED_CATALOG_RESOURCE,
            "trusted_signing_key_ids": ["parent-release-toolbox-v1"],
            "required_template_ids": list(SHIPPED_TEMPLATE_IDS),
            "required_target": target,
            "prewarm_required": True,
            "artifact_source_ids": ["parent-release-resources"],
            "offline_preseed_source_id": None,
            "cache_grace_seconds": 604_800,
            "build_timeout_seconds": 1_800,
        }
    ).to_dict()


__all__ = [
    "ToolboxHostProjectConfiguration",
    "standard_toolbox_host_project_configuration",
    "validate_toolbox_sandbox_policies",
]
