"""Strict revisioned host-owned toolbox project configuration."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from packaging.requirements import InvalidRequirement, Requirement

from .identity import identity_digest, require_digest
from .target import ToolboxTargetIdentity, detect_current_toolbox_target


_CONFIG_FIELDS = {"builtins", "sources", "resolution", "retention"}
_BUILTIN_FIELDS = {
    "template_id",
    "imports",
    "package_requirements",
    "sandbox_policy",
    "required",
    "prewarm",
    "provenance",
}
_SOURCE_FIELDS = {
    "source_id",
    "kind",
    "origin",
    "credential_ref",
    "allowed_package_namespaces",
    "priority",
    "trust_key_ids",
    "maximum_download_bytes",
}
_RESOLUTION_FIELDS = {
    "mode",
    "timeout_seconds",
    "maximum_bytes",
    "maximum_artifacts",
    "allowed_redirect_origins",
    "wheel_only",
}
_RETENTION_FIELDS = {
    "artifact_cache_grace_seconds",
    "maximum_cache_bytes",
    "maximum_cache_artifacts",
    "protected_digests",
    "remove_unreferenced_custom_revisions_on_apply",
}
_ID_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_IMPORT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NAMESPACE_RE = re.compile(r"(?:\*|[a-z0-9]+(?:-[a-z0-9]+)*(?:\.\*)?)")
_SOURCE_KINDS = {"https_index", "https_artifact", "airgap_store"}
_RESOLUTION_MODES = {"online", "prefer_airgap", "air_gapped"}


def _strict_fields(row: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    unknown = sorted(set(row) - fields)
    missing = sorted(fields - set(row))
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")
    if missing:
        raise ValueError(f"{label}_missing_fields:{','.join(missing)}")


def _sequence(value: Any, *, label: str, maximum: int, allow_empty: bool = False) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label}_must_be_array")
    result = tuple(value)
    if (not allow_empty and not result) or len(result) > maximum:
        raise ValueError(f"{label}_invalid")
    return result


def _text(value: Any, *, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}_must_be_string")
    result = value.strip()
    if not result or len(result.encode("utf-8")) > maximum or any(ord(item) < 32 for item in result):
        raise ValueError(f"{label}_invalid")
    return result


def _identifier(value: Any, *, label: str) -> str:
    result = _text(value, label=label, maximum=128).lower()
    if not _ID_RE.fullmatch(result):
        raise ValueError(f"{label}_invalid")
    return result


def _bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label}_must_be_boolean")
    return value


def _bounded_int(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label}_invalid")
    return value


def _unique_strings(
    value: Any,
    *,
    label: str,
    maximum: int,
    normalizer,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    result = tuple(normalizer(item) for item in _sequence(value, label=label, maximum=maximum, allow_empty=allow_empty))
    if len(set(result)) != len(result):
        raise ValueError(f"{label}_duplicate")
    return result


def _https_origin(value: Any, *, label: str, allow_path: bool) -> str:
    raw = _text(value, label=label, maximum=2048)
    parsed = urlsplit(raw)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (not allow_path and parsed.path not in {"", "/"})
    ):
        raise ValueError(f"{label}_invalid")
    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{label}_invalid") from exc
    authority = hostname if port in {None, 443} else f"{hostname}:{port}"
    path = parsed.path or ("/" if allow_path else "")
    return urlunsplit(("https", authority, path, "", ""))


@dataclass(frozen=True)
class ToolboxBuiltinIntent:
    template_id: str
    imports: tuple[str, ...]
    package_requirements: tuple[str, ...]
    sandbox_policy: str
    required: bool
    prewarm: bool
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "template_id", _identifier(self.template_id, label="builtin_template_id"))
        imports = _unique_strings(
            self.imports,
            label="builtin_imports",
            maximum=512,
            normalizer=lambda item: _text(item, label="builtin_import", maximum=128),
        )
        if any(not _IMPORT_RE.fullmatch(item) for item in imports):
            raise ValueError("builtin_import_invalid")
        object.__setattr__(self, "imports", tuple(sorted(imports)))
        requirements: list[str] = []
        for item in _sequence(
            self.package_requirements,
            label="builtin_package_requirements",
            maximum=512,
            allow_empty=True,
        ):
            raw = _text(item, label="builtin_package_requirement", maximum=512)
            try:
                parsed = Requirement(raw)
            except InvalidRequirement as exc:
                raise ValueError("builtin_package_requirement_invalid") from exc
            if parsed.url is not None:
                raise ValueError("builtin_package_requirement_url_forbidden")
            requirements.append(str(parsed))
        if len(set(requirements)) != len(requirements):
            raise ValueError("builtin_package_requirements_duplicate")
        object.__setattr__(self, "package_requirements", tuple(sorted(requirements)))
        object.__setattr__(self, "sandbox_policy", _identifier(self.sandbox_policy, label="builtin_sandbox_policy"))
        _bool(self.required, label="builtin_required")
        _bool(self.prewarm, label="builtin_prewarm")
        if self.prewarm and not self.required:
            raise ValueError("builtin_optional_prewarm_invalid")
        object.__setattr__(self, "provenance", _text(self.provenance, label="builtin_provenance", maximum=512))

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "imports": list(self.imports),
            "package_requirements": list(self.package_requirements),
            "sandbox_policy": self.sandbox_policy,
            "required": self.required,
            "prewarm": self.prewarm,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxBuiltinIntent":
        row = dict(payload or {})
        _strict_fields(row, _BUILTIN_FIELDS, label="toolbox_builtin_intent")
        return cls(
            **{
                **row,
                "imports": tuple(row["imports"]) if isinstance(row["imports"], list) else row["imports"],
                "package_requirements": (
                    tuple(row["package_requirements"])
                    if isinstance(row["package_requirements"], list)
                    else row["package_requirements"]
                ),
            }
        )


@dataclass(frozen=True)
class ToolboxPackageSource:
    source_id: str
    kind: str
    origin: str
    credential_ref: str | None
    allowed_package_namespaces: tuple[str, ...]
    priority: int
    trust_key_ids: tuple[str, ...]
    maximum_download_bytes: int

    def __post_init__(self) -> None:
        source_id = _identifier(self.source_id, label="package_source_id")
        object.__setattr__(self, "source_id", source_id)
        kind = _text(self.kind, label="package_source_kind", maximum=32).lower()
        if kind not in _SOURCE_KINDS:
            raise ValueError("package_source_kind_invalid")
        object.__setattr__(self, "kind", kind)
        if kind == "airgap_store":
            if self.origin != f"airgap://{source_id}":
                raise ValueError("package_source_origin_invalid")
            if self.credential_ref is not None:
                raise ValueError("package_source_credential_ref_forbidden")
        else:
            object.__setattr__(self, "origin", _https_origin(self.origin, label="package_source_origin", allow_path=True))
            if self.credential_ref is not None:
                object.__setattr__(
                    self,
                    "credential_ref",
                    _text(self.credential_ref, label="package_source_credential_ref", maximum=512),
                )
        namespaces = _unique_strings(
            self.allowed_package_namespaces,
            label="package_source_allowed_namespaces",
            maximum=512,
            normalizer=lambda item: _text(item, label="package_source_allowed_namespace", maximum=128).lower(),
        )
        if any(not _NAMESPACE_RE.fullmatch(item) for item in namespaces):
            raise ValueError("package_source_allowed_namespace_invalid")
        object.__setattr__(self, "allowed_package_namespaces", namespaces)
        _bounded_int(self.priority, label="package_source_priority", minimum=0, maximum=10_000)
        trust = _unique_strings(
            self.trust_key_ids,
            label="package_source_trust_key_ids",
            maximum=64,
            normalizer=lambda item: _identifier(item, label="package_source_trust_key_id"),
        )
        object.__setattr__(self, "trust_key_ids", trust)
        _bounded_int(
            self.maximum_download_bytes,
            label="package_source_maximum_download_bytes",
            minimum=1,
            maximum=16 * 1024 * 1024 * 1024,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "origin": self.origin,
            "credential_ref": self.credential_ref,
            "allowed_package_namespaces": list(self.allowed_package_namespaces),
            "priority": self.priority,
            "trust_key_ids": list(self.trust_key_ids),
            "maximum_download_bytes": self.maximum_download_bytes,
        }

    def public_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.to_dict().items() if key != "credential_ref"}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxPackageSource":
        row = dict(payload or {})
        _strict_fields(row, _SOURCE_FIELDS, label="toolbox_package_source")
        return cls(
            **{
                **row,
                "allowed_package_namespaces": (
                    tuple(row["allowed_package_namespaces"])
                    if isinstance(row["allowed_package_namespaces"], list)
                    else row["allowed_package_namespaces"]
                ),
                "trust_key_ids": (
                    tuple(row["trust_key_ids"])
                    if isinstance(row["trust_key_ids"], list)
                    else row["trust_key_ids"]
                ),
            }
        )


@dataclass(frozen=True)
class ToolboxResolutionPolicy:
    mode: str
    timeout_seconds: int
    maximum_bytes: int
    maximum_artifacts: int
    allowed_redirect_origins: tuple[str, ...]
    wheel_only: bool

    def __post_init__(self) -> None:
        mode = _text(self.mode, label="toolbox_resolution_mode", maximum=32).lower()
        if mode not in _RESOLUTION_MODES:
            raise ValueError("toolbox_resolution_mode_invalid")
        object.__setattr__(self, "mode", mode)
        _bounded_int(self.timeout_seconds, label="toolbox_resolution_timeout_seconds", minimum=1, maximum=600)
        _bounded_int(self.maximum_bytes, label="toolbox_resolution_maximum_bytes", minimum=1, maximum=16 * 1024 * 1024 * 1024)
        _bounded_int(self.maximum_artifacts, label="toolbox_resolution_maximum_artifacts", minimum=1, maximum=4096)
        origins = _unique_strings(
            self.allowed_redirect_origins,
            label="toolbox_resolution_allowed_redirect_origins",
            maximum=64,
            normalizer=lambda item: _https_origin(
                item, label="toolbox_resolution_allowed_redirect_origin", allow_path=False
            ),
            allow_empty=True,
        )
        object.__setattr__(self, "allowed_redirect_origins", origins)
        if not _bool(self.wheel_only, label="toolbox_resolution_wheel_only"):
            raise ValueError("toolbox_resolution_wheel_only_required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "timeout_seconds": self.timeout_seconds,
            "maximum_bytes": self.maximum_bytes,
            "maximum_artifacts": self.maximum_artifacts,
            "allowed_redirect_origins": list(self.allowed_redirect_origins),
            "wheel_only": self.wheel_only,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxResolutionPolicy":
        row = dict(payload or {})
        _strict_fields(row, _RESOLUTION_FIELDS, label="toolbox_resolution")
        return cls(
            **{
                **row,
                "allowed_redirect_origins": (
                    tuple(row["allowed_redirect_origins"])
                    if isinstance(row["allowed_redirect_origins"], list)
                    else row["allowed_redirect_origins"]
                ),
            }
        )


@dataclass(frozen=True)
class ToolboxRetentionPolicy:
    artifact_cache_grace_seconds: int
    maximum_cache_bytes: int
    maximum_cache_artifacts: int
    protected_digests: tuple[str, ...]
    remove_unreferenced_custom_revisions_on_apply: bool

    def __post_init__(self) -> None:
        _bounded_int(
            self.artifact_cache_grace_seconds,
            label="toolbox_retention_artifact_cache_grace_seconds",
            minimum=0,
            maximum=7_776_000,
        )
        _bounded_int(self.maximum_cache_bytes, label="toolbox_retention_maximum_cache_bytes", minimum=1, maximum=1 << 50)
        _bounded_int(self.maximum_cache_artifacts, label="toolbox_retention_maximum_cache_artifacts", minimum=1, maximum=1_000_000)
        digests = _unique_strings(
            self.protected_digests,
            label="toolbox_retention_protected_digests",
            maximum=4096,
            normalizer=lambda item: require_digest(item, label="toolbox_retention_protected_digest"),
            allow_empty=True,
        )
        object.__setattr__(self, "protected_digests", tuple(sorted(digests)))
        _bool(
            self.remove_unreferenced_custom_revisions_on_apply,
            label="toolbox_retention_remove_unreferenced_custom_revisions_on_apply",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_cache_grace_seconds": self.artifact_cache_grace_seconds,
            "maximum_cache_bytes": self.maximum_cache_bytes,
            "maximum_cache_artifacts": self.maximum_cache_artifacts,
            "protected_digests": list(self.protected_digests),
            "remove_unreferenced_custom_revisions_on_apply": self.remove_unreferenced_custom_revisions_on_apply,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxRetentionPolicy":
        row = dict(payload or {})
        _strict_fields(row, _RETENTION_FIELDS, label="toolbox_retention")
        return cls(
            **{
                **row,
                "protected_digests": (
                    tuple(row["protected_digests"])
                    if isinstance(row["protected_digests"], list)
                    else row["protected_digests"]
                ),
            }
        )


@dataclass(frozen=True)
class ToolboxHostProjectConfiguration:
    builtins: tuple[ToolboxBuiltinIntent, ...]
    sources: tuple[ToolboxPackageSource, ...]
    resolution: ToolboxResolutionPolicy
    retention: ToolboxRetentionPolicy

    def __post_init__(self) -> None:
        builtins = tuple(self.builtins)
        if not builtins or any(not isinstance(item, ToolboxBuiltinIntent) for item in builtins):
            raise ValueError("toolbox_host_builtins_invalid")
        if len({item.template_id for item in builtins}) != len(builtins):
            raise ValueError("toolbox_host_builtin_duplicate")
        if not any(item.required for item in builtins):
            raise ValueError("toolbox_host_required_builtin_missing")
        object.__setattr__(self, "builtins", builtins)
        sources = tuple(self.sources)
        if not sources or any(not isinstance(item, ToolboxPackageSource) for item in sources):
            raise ValueError("toolbox_host_sources_invalid")
        if len({item.source_id for item in sources}) != len(sources):
            raise ValueError("toolbox_host_source_duplicate")
        priorities = [item.priority for item in sources]
        if priorities != sorted(priorities, reverse=True):
            raise ValueError("toolbox_host_source_priority_order_invalid")
        object.__setattr__(self, "sources", sources)
        if not isinstance(self.resolution, ToolboxResolutionPolicy):
            raise ValueError("toolbox_host_resolution_invalid")
        if not isinstance(self.retention, ToolboxRetentionPolicy):
            raise ValueError("toolbox_host_retention_invalid")
        if self.resolution.mode == "online" and not any(item.kind.startswith("https_") for item in sources):
            raise ValueError("toolbox_host_online_source_required")
        if self.resolution.mode == "air_gapped" and any(item.kind.startswith("https_") for item in sources):
            raise ValueError("toolbox_host_air_gapped_https_source_forbidden")
        if self.resolution.mode in {"air_gapped", "prefer_airgap"} and not any(
            item.kind == "airgap_store" for item in sources
        ):
            raise ValueError("toolbox_host_airgap_source_required")

    @property
    def config_revision(self) -> str:
        return identity_digest("hosting.toolbox.host_config.v2", self.to_dict())

    @property
    def source_set_revision(self) -> str:
        return identity_digest(
            "hosting.toolbox.source_set.v1",
            {
                "sources": [item.to_dict() for item in self.sources],
                "resolution": self.resolution.to_dict(),
            },
        )

    @property
    def target(self) -> ToolboxTargetIdentity:
        return detect_current_toolbox_target()

    def to_dict(self) -> dict[str, Any]:
        return {
            "builtins": [item.to_dict() for item in self.builtins],
            "sources": [item.to_dict() for item in self.sources],
            "resolution": self.resolution.to_dict(),
            "retention": self.retention.to_dict(),
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "builtins": [item.to_dict() for item in self.builtins],
            "sources": [item.public_dict() for item in self.sources],
            "resolution": self.resolution.to_dict(),
            "retention": self.retention.to_dict(),
            "config_revision": self.config_revision,
            "source_set_revision": self.source_set_revision,
            "target": self.target.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolboxHostProjectConfiguration":
        if not isinstance(value, Mapping):
            raise ValueError("toolbox_host_project_configuration_must_be_object")
        row = dict(value)
        _strict_fields(row, _CONFIG_FIELDS, label="toolbox_host_project_configuration")
        builtins = _sequence(row["builtins"], label="toolbox_host_builtins", maximum=256)
        sources = _sequence(row["sources"], label="toolbox_host_sources", maximum=64)
        if not isinstance(row["resolution"], Mapping):
            raise ValueError("toolbox_host_resolution_must_be_object")
        if not isinstance(row["retention"], Mapping):
            raise ValueError("toolbox_host_retention_must_be_object")
        return cls(
            builtins=tuple(ToolboxBuiltinIntent.from_dict(item) for item in builtins),
            sources=tuple(ToolboxPackageSource.from_dict(item) for item in sources),
            resolution=ToolboxResolutionPolicy.from_dict(row["resolution"]),
            retention=ToolboxRetentionPolicy.from_dict(row["retention"]),
        )


__all__ = [
    "ToolboxBuiltinIntent",
    "ToolboxHostProjectConfiguration",
    "ToolboxPackageSource",
    "ToolboxResolutionPolicy",
    "ToolboxRetentionPolicy",
]
