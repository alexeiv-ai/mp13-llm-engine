"""Canonical identities for toolbox runtime bindings and worker registrations."""
from __future__ import annotations

from typing import Any, Mapping

from ..toolbox.identity import identity_digest


def normalize_manifest_digest(value: Any) -> str:
    """Return the canonical ``sha256:`` spelling used by persisted state."""

    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        return raw
    if len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw):
        return f"sha256:{raw}"
    return raw


def runtime_binding_digest(
    *,
    toolbox_id: str,
    profile_id: str,
    manifest_hash: str,
    environment_reference: str,
    engine_id: str,
    definition_revision: str = "",
) -> str:
    """Bind a persisted semantic profile to one concrete runtime registration."""

    return identity_digest(
        "hosting.toolbox.runtime-binding.v1",
        {
            "toolbox_id": str(toolbox_id or "").strip(),
            "profile_id": str(profile_id or "").strip(),
            "manifest_hash": normalize_manifest_digest(manifest_hash),
            "environment_reference": str(environment_reference or "").strip(),
            "engine_id": str(engine_id or "").strip(),
            "definition_revision": str(definition_revision or "").strip(),
        },
    )


def registration_binding_digest(registration: Mapping[str, Any]) -> str:
    """Derive the binding digest carried by an engine registration."""

    row = dict(registration or {})
    bundle = dict(row.get("bundle") or {})
    environment = dict(row.get("environment") or {})
    return runtime_binding_digest(
        toolbox_id=str(bundle.get("toolbox_id") or row.get("toolbox_id") or ""),
        profile_id=str(
            bundle.get("resolved_profile_id")
            or bundle.get("sandbox_profile_id")
            or row.get("sandbox_profile_id")
            or ""
        ),
        manifest_hash=str(bundle.get("manifest_hash") or row.get("manifest_hash") or ""),
        environment_reference=str(
            row.get("environment_reference")
            or environment.get("environment_reference")
            or environment.get("reference_id")
            or ""
        ),
        engine_id=str(row.get("engine_id") or ""),
        definition_revision=str(bundle.get("definition_revision") or ""),
    )


__all__ = [
    "normalize_manifest_digest",
    "registration_binding_digest",
    "runtime_binding_digest",
]
