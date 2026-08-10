"""Fail-closed policy checks for resolved toolbox dependencies."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from mp13_engine.mp13_intrinsics_metadata import intrinsic_dependency_metadata

from .catalog import normalize_distribution_name
from .dependency_analysis import ToolboxResolvedDependencies, ToolboxTemplateSelection
from .identity import require_digest
from .target import validate_target_name


_FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "allow_execution",
        "allow_resolution",
        "approval",
        "approval_ref",
        "approved",
        "capabilities",
        "dependency_approval_ref",
        "filesystem",
        "host_api_permissions",
        "network",
        "model_environment",
        "model_python_executable",
        "model_runtime",
        "model_runtime_id",
        "model_runtime_ref",
        "sandbox",
        "sandbox_policy",
        "subprocess",
    }
)


def _strict_fields(row: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    unknown = sorted(set(row) - fields)
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")
    missing = sorted(fields - set(row))
    if missing:
        raise ValueError(f"{label}_missing_fields:{','.join(missing)}")


def _strings(value: Any, *, label: str, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label}_must_be_array")
    if len(value) > maximum:
        raise ValueError(f"{label}_too_many")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label}_item_must_be_string")
        text = item.strip()
        if not text or len(text.encode("utf-8")) > 512 or any(ord(char) < 32 for char in text):
            raise ValueError(f"{label}_item_invalid")
        result.append(text)
    if len(set(result)) != len(result):
        raise ValueError(f"{label}_duplicate")
    return tuple(sorted(result))


def normalize_https_origin(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("package_index_origin_must_be_string")
    raw = value.strip()
    parsed = urlsplit(raw)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("package_index_origin_invalid")
    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("package_index_origin_invalid") from exc
    if port in {None, 443}:
        return f"https://{hostname}"
    return f"https://{hostname}:{port}"


@dataclass(frozen=True)
class ToolboxDependencyPolicy:
    revision: str
    allowed_template_ids: tuple[str, ...]
    allowed_targets: tuple[str, ...]
    package_allowlist: tuple[str, ...]
    package_denylist: tuple[str, ...]
    allow_custom: bool
    custom_requires_approval: bool
    online_resolution_allowed: bool
    allowed_index_origins: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", require_digest(self.revision, label="package_policy_revision"))
        template_ids = _strings(
            self.allowed_template_ids, label="policy_allowed_template_ids", maximum=256
        )
        if not template_ids:
            raise ValueError("policy_allowed_template_ids_required")
        if any(not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item) for item in template_ids):
            raise ValueError("policy_allowed_template_id_invalid")
        object.__setattr__(self, "allowed_template_ids", template_ids)
        targets = _strings(self.allowed_targets, label="policy_allowed_targets", maximum=32)
        if not targets:
            raise ValueError("policy_allowed_targets_required")
        try:
            targets = tuple(validate_target_name(item, label="policy_allowed_target") for item in targets)
        except ValueError as exc:
            raise ValueError("policy_allowed_target_invalid") from exc
        object.__setattr__(self, "allowed_targets", targets)
        allowlist = tuple(
            sorted(
                normalize_distribution_name(item)
                for item in _strings(
                    self.package_allowlist,
                    label="policy_package_allowlist",
                    maximum=2048,
                )
            )
        )
        denylist = tuple(
            sorted(
                normalize_distribution_name(item)
                for item in _strings(
                    self.package_denylist,
                    label="policy_package_denylist",
                    maximum=2048,
                )
            )
        )
        if len(set(allowlist)) != len(allowlist):
            raise ValueError("policy_package_allowlist_duplicate")
        if len(set(denylist)) != len(denylist):
            raise ValueError("policy_package_denylist_duplicate")
        object.__setattr__(self, "package_allowlist", allowlist)
        object.__setattr__(self, "package_denylist", denylist)
        for field_name in (
            "allow_custom",
            "custom_requires_approval",
            "online_resolution_allowed",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"policy_{field_name}_must_be_boolean")
        origins = tuple(
            sorted(
                normalize_https_origin(item)
                for item in _strings(
                    self.allowed_index_origins,
                    label="policy_allowed_index_origins",
                    maximum=64,
                )
            )
        )
        if len(set(origins)) != len(origins):
            raise ValueError("policy_allowed_index_origins_duplicate")
        object.__setattr__(self, "allowed_index_origins", origins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "allowed_template_ids": list(self.allowed_template_ids),
            "allowed_targets": list(self.allowed_targets),
            "package_allowlist": list(self.package_allowlist),
            "package_denylist": list(self.package_denylist),
            "allow_custom": self.allow_custom,
            "custom_requires_approval": self.custom_requires_approval,
            "online_resolution_allowed": self.online_resolution_allowed,
            "allowed_index_origins": list(self.allowed_index_origins),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxDependencyPolicy":
        row = dict(payload or {})
        fields = {
            "revision",
            "allowed_template_ids",
            "allowed_targets",
            "package_allowlist",
            "package_denylist",
            "allow_custom",
            "custom_requires_approval",
            "online_resolution_allowed",
            "allowed_index_origins",
        }
        _strict_fields(row, fields, label="toolbox_dependency_policy")
        return cls(
            revision=row["revision"],
            allowed_template_ids=row["allowed_template_ids"],
            allowed_targets=row["allowed_targets"],
            package_allowlist=row["package_allowlist"],
            package_denylist=row["package_denylist"],
            allow_custom=row["allow_custom"],
            custom_requires_approval=row["custom_requires_approval"],
            online_resolution_allowed=row["online_resolution_allowed"],
            allowed_index_origins=row["allowed_index_origins"],
        )


class ToolboxDependencyPolicyError(ValueError):
    def __init__(self, code: str, summary: str):
        self.code = code
        self.summary = summary
        super().__init__(code)


@dataclass(frozen=True)
class ToolboxDependencyPolicyDecision:
    policy_revision: str
    template_id: str
    target: str
    approval_required: bool
    package_distributions: tuple[str, ...]
    index_origins: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_revision": self.policy_revision,
            "template_id": self.template_id,
            "target": self.target,
            "approval_required": self.approval_required,
            "package_distributions": list(self.package_distributions),
            "index_origins": list(self.index_origins),
        }


def _reject_payload_authority(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(_FORBIDDEN_AUTHORITY_KEYS & {str(key) for key in value})
        if forbidden:
            raise ToolboxDependencyPolicyError(
                "dependency_payload_authority_forbidden",
                f"Dependency metadata cannot assert '{forbidden[0]}'.",
            )
        for nested in value.values():
            _reject_payload_authority(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_payload_authority(nested)


def validate_toolbox_dependency_policy(
    selection: ToolboxTemplateSelection,
    dependencies: ToolboxResolvedDependencies,
    policy: ToolboxDependencyPolicy,
    *,
    python_abi: str,
    platform: str,
    requested_template_id: str | None = None,
    requested_index_origins: Sequence[str] = (),
    intrinsic_names: Sequence[str] = (),
    dependency_payload: Mapping[str, Any] | None = None,
) -> ToolboxDependencyPolicyDecision:
    if dependency_payload is not None and not isinstance(dependency_payload, Mapping):
        raise ValueError("dependency_payload_must_be_object")
    if not isinstance(requested_index_origins, Sequence) or isinstance(
        requested_index_origins, (str, bytes, bytearray)
    ):
        raise ValueError("requested_index_origins_must_be_array")
    if not isinstance(intrinsic_names, Sequence) or isinstance(
        intrinsic_names, (str, bytes, bytearray)
    ):
        raise ValueError("intrinsic_names_must_be_array")
    _reject_payload_authority(dict(dependency_payload or {}))
    template = selection.template
    target = f"{python_abi}-{platform}"
    if target not in policy.allowed_targets:
        raise ToolboxDependencyPolicyError(
            "dependency_target_denied", "The Python ABI/platform target is not allowed."
        )
    if python_abi not in template.python_abis or platform not in template.platforms:
        raise ToolboxDependencyPolicyError(
            "dependency_template_target_mismatch", "The template does not support the requested target."
        )
    if template.template_id not in policy.allowed_template_ids:
        raise ToolboxDependencyPolicyError(
            "dependency_template_denied", "The selected template is not allowed by package policy."
        )
    if requested_template_id is not None:
        if not isinstance(requested_template_id, str):
            raise ValueError("requested_template_id_must_be_string")
        requested = requested_template_id.strip()
        if requested != template.template_id or requested not in policy.allowed_template_ids:
            raise ToolboxDependencyPolicyError(
                "dependency_template_denied", "The requested template is not the allowed selected template."
            )
    if selection.mode == "custom" and not policy.allow_custom:
        raise ToolboxDependencyPolicyError(
            "dependency_custom_denied", "Custom dependency deltas are disabled."
        )

    distributions = tuple(sorted(item.distribution for item in dependencies.requirements))
    denied = sorted(set(distributions) & set(policy.package_denylist))
    if denied:
        raise ToolboxDependencyPolicyError(
            "dependency_package_denied", f"Distribution '{denied[0]}' is denied by package policy."
        )
    if policy.package_allowlist:
        outside = sorted(set(distributions) - set(policy.package_allowlist))
        if outside:
            raise ToolboxDependencyPolicyError(
                "dependency_package_not_allowed",
                f"Distribution '{outside[0]}' is not in the package allowlist.",
            )

    origins = tuple(sorted({normalize_https_origin(item) for item in requested_index_origins}))
    if origins and not policy.online_resolution_allowed:
        raise ToolboxDependencyPolicyError(
            "dependency_online_resolution_denied", "Online package resolution is disabled."
        )
    outside_origins = sorted(set(origins) - set(policy.allowed_index_origins))
    if outside_origins:
        raise ToolboxDependencyPolicyError(
            "dependency_index_denied", "A requested package index origin is not allowed."
        )

    intrinsic_dependencies = intrinsic_dependency_metadata(intrinsic_names)
    by_distribution = {item.distribution: item for item in dependencies.requirements}
    missing_roots = set(intrinsic_dependencies["import_roots"])
    for raw_requirement in intrinsic_dependencies["package_requirements"]:
        requirement = Requirement(raw_requirement)
        distribution = normalize_distribution_name(requirement.name)
        resolved = by_distribution.get(distribution)
        if resolved is None:
            raise ToolboxDependencyPolicyError(
                "dependency_intrinsic_requirement_missing",
                f"Intrinsic distribution '{distribution}' is missing.",
            )
        pins = [
            Version(specifier.version)
            for specifier in requirement.specifier
            if specifier.operator == "==" and "*" not in specifier.version
        ]
        if pins and not all(pin in SpecifierSet(resolved.constraint) for pin in pins):
            raise ToolboxDependencyPolicyError(
                "dependency_intrinsic_requirement_conflict",
                f"Intrinsic distribution '{distribution}' has an incompatible constraint.",
            )
        missing_roots -= set(resolved.import_roots)
    if missing_roots:
        raise ToolboxDependencyPolicyError(
            "dependency_intrinsic_import_missing",
            f"Intrinsic import root '{sorted(missing_roots)[0]}' is missing.",
        )

    return ToolboxDependencyPolicyDecision(
        policy_revision=policy.revision,
        template_id=template.template_id,
        target=target,
        approval_required=bool(selection.mode == "custom" and policy.custom_requires_approval),
        package_distributions=distributions,
        index_origins=origins,
    )


__all__ = [
    "ToolboxDependencyPolicy",
    "ToolboxDependencyPolicyDecision",
    "ToolboxDependencyPolicyError",
    "normalize_https_origin",
    "validate_toolbox_dependency_policy",
]
