"""Canonical, domain-separated identities for hosted toolbox version 2."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Dict


DEFINITION_REVISION_DOMAIN = "hosting.toolbox.definition.v2"
RESOLVED_PROFILE_DOMAIN = "hosting.toolbox.resolved_profile.v2"
ENVIRONMENT_IDENTITY_DOMAIN = "hosting.toolbox.environment.v2"
BUNDLE_MANIFEST_DOMAIN = "hosting.toolbox.bundle_manifest.v2"
TEMPLATE_LOCK_DOMAIN = "hosting.toolbox.template_lock.v1"
CUSTOM_LOCK_DOMAIN = "hosting.toolbox.custom_lock.v1"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _nfc(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity_number_must_be_finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError("identity_object_key_must_be_string")
            key = _nfc(raw_key)
            if key in out:
                raise ValueError("identity_object_key_normalization_conflict")
            out[key] = _canonical_value(child)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(child) for child in value]
    raise ValueError(f"identity_value_not_json:{type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the frozen UTF-8 canonical JSON representation."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def identity_digest(domain: str, value: Any) -> str:
    normalized_domain = _nfc(str(domain or "").strip())
    if not normalized_domain:
        raise ValueError("identity_domain_required")
    envelope = {"domain": normalized_domain, "value": value}
    return f"sha256:{hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()}"


def require_digest(value: Any, *, label: str) -> str:
    digest = str(value or "").strip()
    if not _DIGEST_RE.fullmatch(digest):
        raise ValueError(f"{label}_must_be_canonical_sha256")
    return digest


def _normalized_path(value: Any) -> str:
    raw = _nfc(str(value or "")).replace("\\", "/")
    if not raw or raw.startswith("/") or raw.endswith("/"):
        raise ValueError("identity_file_path_invalid")
    segments = raw.split("/")
    if any(not segment or segment in {".", ".."} for segment in segments):
        raise ValueError("identity_file_path_invalid")
    return "/".join(segments)


def _sorted_unique_strings(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("identity_string_list_required")
    return sorted({_nfc(str(item or "").strip()) for item in values if str(item or "").strip()})


def _sorted_records(values: Any) -> list[Any]:
    if values is None:
        return []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("identity_record_list_required")
    normalized = [_canonical_value(item) for item in values]
    return sorted(normalized, key=canonical_json_bytes)


def _canonical_files(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("identity_files_required")
    by_path: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ValueError("identity_file_object_required")
        row = dict(raw)
        path = _normalized_path(row.get("relative_path"))
        row["relative_path"] = path
        normalized = _canonical_value(row)
        existing = by_path.get(path)
        if existing is not None and canonical_json_bytes(existing) != canonical_json_bytes(normalized):
            raise ValueError("identity_file_path_conflict")
        by_path[path] = normalized
    return [by_path[path] for path in sorted(by_path)]


def _canonical_dependency(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("identity_dependency_object_required")
    row = dict(raw)
    row["declared_imports"] = _sorted_unique_strings(row.get("declared_imports"))
    row["package_requirements"] = _sorted_unique_strings(row.get("package_requirements"))
    return _canonical_value(row)


def _canonical_request(raw: Any, *, manual: bool) -> tuple[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise ValueError("identity_request_object_required")
    row = dict(raw)
    module_name = _nfc(str(row.get("module_name") or "").strip())
    callable_name = _nfc(str(row.get("callable_name") or "").strip())
    if not module_name or not callable_name:
        raise ValueError("identity_request_key_required")
    row["module_name"] = module_name
    row["callable_name"] = callable_name
    row["files"] = _canonical_files(row.get("files"))
    row["dependency"] = _canonical_dependency(row.get("dependency"))
    key = f"manual:{module_name}:{callable_name}" if manual else f"{module_name}:{callable_name}"
    return key, _canonical_value(row)


def canonical_definition_payload(definition: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(definition or {})
    row.pop("expected_revision", None)
    for field, manual in (("auto_requests", False), ("manual_requests", True)):
        by_key: dict[str, dict[str, Any]] = {}
        raw_requests = row.get(field)
        if not isinstance(raw_requests, Sequence) or isinstance(raw_requests, (str, bytes, bytearray)):
            raise ValueError(f"identity_{field}_required")
        for raw_request in raw_requests:
            key, request = _canonical_request(raw_request, manual=manual)
            if key in by_key:
                raise ValueError("identity_duplicate_stable_key")
            by_key[key] = request
        row[field] = [by_key[key] for key in sorted(by_key)]
    raw_intrinsics = row.get("intrinsics")
    if not isinstance(raw_intrinsics, Mapping):
        raise ValueError("identity_intrinsics_object_required")
    intrinsics = dict(raw_intrinsics)
    intrinsics["names"] = _sorted_unique_strings(intrinsics.get("names"))
    row["intrinsics"] = _canonical_value(intrinsics)
    return _canonical_value(row)


def definition_revision(definition: Mapping[str, Any]) -> str:
    return identity_digest(DEFINITION_REVISION_DOMAIN, canonical_definition_payload(definition))


def resolved_profile_identity(*, environment_identity: str, sandbox_policy: Mapping[str, Any]) -> str:
    return identity_digest(
        RESOLVED_PROFILE_DOMAIN,
        {
            "environment_identity": require_digest(environment_identity, label="environment_identity"),
            "sandbox_policy": dict(sandbox_policy or {}),
        },
    )


def environment_identity(
    *,
    runtime_identity: Mapping[str, Any],
    template_lock_digest: str,
    custom_lock_digest: str | None,
    isolation_policy: Mapping[str, Any],
) -> str:
    return identity_digest(
        ENVIRONMENT_IDENTITY_DOMAIN,
        {
            "runtime_identity": dict(runtime_identity or {}),
            "template_lock_digest": require_digest(template_lock_digest, label="template_lock_digest"),
            "custom_lock_digest": (
                require_digest(custom_lock_digest, label="custom_lock_digest") if custom_lock_digest else None
            ),
            "isolation_policy": dict(isolation_policy or {}),
        },
    )


def bundle_manifest_digest(manifest: Mapping[str, Any]) -> str:
    row = dict(manifest or {})
    row.pop("manifest_hash", None)
    row.pop("bundle_revision", None)
    if "files" in row:
        row["files"] = _canonical_files(row.get("files"))
    if "tools" in row:
        row["tools"] = _sorted_records(row.get("tools"))
    if "auto_tools" in row:
        row["auto_tools"] = _sorted_records(row.get("auto_tools"))
    for field in (
        "intrinsic_tool_names",
        "active_intrinsic_tool_names",
        "hidden_intrinsic_tool_names",
        "hidden_tool_names",
    ):
        if field in row:
            row[field] = _sorted_unique_strings(row.get(field))
    return identity_digest(BUNDLE_MANIFEST_DOMAIN, row)


def template_lock_digest(lock: Mapping[str, Any]) -> str:
    row = dict(lock or {})
    for field in ("distributions", "artifacts"):
        if field in row:
            row[field] = _sorted_records(row.get(field))
    if "import_roots" in row:
        row["import_roots"] = _sorted_unique_strings(row.get("import_roots"))
    return identity_digest(TEMPLATE_LOCK_DOMAIN, row)


def custom_lock_digest(lock: Mapping[str, Any]) -> str:
    row = dict(lock or {})
    if "base_template_lock_digest" in row:
        row["base_template_lock_digest"] = require_digest(
            row.get("base_template_lock_digest"), label="base_template_lock_digest"
        )
    for field in ("distributions", "artifacts"):
        if field in row:
            row[field] = _sorted_records(row.get(field))
    if "import_roots" in row:
        row["import_roots"] = _sorted_unique_strings(row.get("import_roots"))
    return identity_digest(CUSTOM_LOCK_DOMAIN, row)


__all__ = [
    "BUNDLE_MANIFEST_DOMAIN",
    "CUSTOM_LOCK_DOMAIN",
    "DEFINITION_REVISION_DOMAIN",
    "ENVIRONMENT_IDENTITY_DOMAIN",
    "RESOLVED_PROFILE_DOMAIN",
    "TEMPLATE_LOCK_DOMAIN",
    "bundle_manifest_digest",
    "canonical_definition_payload",
    "canonical_json_bytes",
    "custom_lock_digest",
    "definition_revision",
    "environment_identity",
    "identity_digest",
    "resolved_profile_identity",
    "template_lock_digest",
]
