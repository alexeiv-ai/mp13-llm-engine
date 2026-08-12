"""Strict immutable toolbox template and reviewed dependency catalog models."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .target import SUPPORTED_PYTHON_ABI, validate_target_platform


_CANONICAL_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_DISTRIBUTION_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_EXTRA_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_IMPORT_ROOT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TEMPLATE_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}")
_CONSTRAINT_PART_RE = re.compile(r"(?:===|==|!=|~=|>=|<=|>|<)[A-Za-z0-9][A-Za-z0-9.!+*_-]{0,127}")

MAX_TEMPLATE_DISTRIBUTIONS = 512
MAX_TEMPLATE_IMPORT_ROOTS = 512
MAX_CATALOG_RULES = 1024
MAX_ALIASES_PER_RULE = 32


def _required_text(value: Any, *, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}_must_be_string")
    text = value.strip()
    if not text or len(text.encode("utf-8")) > maximum or any(ord(char) < 32 for char in text):
        raise ValueError(f"{label}_invalid")
    return text


def normalize_distribution_name(value: Any) -> str:
    """Return the PEP 503-style normalized distribution name."""

    name = re.sub(
        r"[-_.]+",
        "-",
        _required_text(value, label="distribution_name").lower(),
    )
    if not _DISTRIBUTION_RE.fullmatch(name):
        raise ValueError("distribution_name_invalid")
    return name


def normalize_import_root(value: Any) -> str:
    root = _required_text(value, label="import_root")
    if not _IMPORT_ROOT_RE.fullmatch(root):
        raise ValueError("import_root_invalid")
    return root


def normalize_version_constraint(value: Any) -> str:
    raw = _required_text(value, label="version_constraint").replace(" ", "")
    parts = raw.split(",")
    if len(parts) > 16 or any(not _CONSTRAINT_PART_RE.fullmatch(part) for part in parts):
        raise ValueError("version_constraint_invalid")
    return ",".join(sorted(set(parts)))


def _strict_fields(row: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")


def _sequence(value: Any, *, label: str, maximum: int) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label}_must_be_array")
    items = list(value)
    if len(items) > maximum:
        raise ValueError(f"{label}_too_many")
    return items


def _unique_sorted_strings(
    value: Any,
    *,
    label: str,
    maximum: int,
    normalizer,
) -> tuple[str, ...]:
    items = _sequence(value, label=label, maximum=maximum)
    normalized = [normalizer(item) for item in items]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label}_duplicate")
    return tuple(sorted(normalized))


def _canonical_digest(value: Any, *, label: str) -> str:
    digest = _required_text(value, label=label)
    if not _CANONICAL_DIGEST_RE.fullmatch(digest):
        raise ValueError(f"{label}_invalid")
    return digest


@dataclass(frozen=True, order=True)
class ToolboxLockedDistributionSpec:
    """One exact distribution entry in a complete template lock."""

    name: str
    version: str
    extras: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_distribution_name(self.name))
        version = _required_text(self.version, label="locked_distribution_version", maximum=128)
        if not _VERSION_RE.fullmatch(version):
            raise ValueError("locked_distribution_version_invalid")
        object.__setattr__(self, "version", version)
        extras = _unique_sorted_strings(
            self.extras,
            label="locked_distribution_extras",
            maximum=32,
            normalizer=normalize_distribution_name,
        )
        object.__setattr__(self, "extras", extras)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "extras": list(self.extras)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxLockedDistributionSpec":
        row = dict(payload or {})
        _strict_fields(row, {"name", "version", "extras"}, label="locked_distribution")
        return cls(
            name=str(row.get("name") or ""),
            version=str(row.get("version") or ""),
            extras=tuple(
                _sequence(row.get("extras", []), label="locked_distribution_extras", maximum=32)
            ),
        )


@dataclass(frozen=True)
class ToolboxTemplateProvenance:
    source: str
    revision: str
    evidence_digest: str
    verifier_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("source", "revision"):
            value = _required_text(
                getattr(self, field_name), label=f"template_provenance_{field_name}"
            )
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "evidence_digest",
            _canonical_digest(self.evidence_digest, label="template_evidence_digest"),
        )
        verifier = str(self.verifier_id or "").strip() or None
        object.__setattr__(self, "verifier_id", verifier)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "source": self.source,
            "revision": self.revision,
            "evidence_digest": self.evidence_digest,
            "verifier_id": self.verifier_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxTemplateProvenance":
        row = dict(payload or {})
        _strict_fields(
            row,
            {"source", "revision", "evidence_digest", "verifier_id"},
            label="template_provenance",
        )
        return cls(
            source=str(row.get("source") or ""),
            revision=str(row.get("revision") or ""),
            evidence_digest=str(row.get("evidence_digest") or ""),
            verifier_id=row.get("verifier_id"),
        )


@dataclass(frozen=True)
class ToolboxEnvironmentTemplateSpec:
    """An immutable, complete parent-published toolbox environment template."""

    template_id: str
    python_requires: str
    python_abis: tuple[str, ...]
    runtime_kind: str
    worker_protocol_version: str
    platforms: tuple[str, ...]
    locked_distributions: tuple[ToolboxLockedDistributionSpec, ...]
    exposed_import_roots: tuple[str, ...]
    lock_digest: str
    parent_worker_artifact_digest: str
    isolation_policy_version: str
    provenance: ToolboxTemplateProvenance

    def __post_init__(self) -> None:
        template_id = _required_text(self.template_id, label="template_id")
        if not _TEMPLATE_ID_RE.fullmatch(template_id):
            raise ValueError("template_id_invalid")
        object.__setattr__(self, "template_id", template_id)
        object.__setattr__(self, "python_requires", normalize_version_constraint(self.python_requires))
        object.__setattr__(
            self,
            "python_abis",
            _unique_sorted_strings(
                self.python_abis,
                label="template_python_abis",
                maximum=16,
                normalizer=lambda item: _required_text(item, label="template_python_abi").lower(),
            ),
        )
        if not self.python_abis or any(item != SUPPORTED_PYTHON_ABI for item in self.python_abis):
            raise ValueError("template_python_abis_invalid")
        runtime_kind = _required_text(self.runtime_kind, label="template_runtime_kind")
        if runtime_kind != "toolbox_python":
            raise ValueError("template_runtime_kind_invalid")
        object.__setattr__(self, "runtime_kind", runtime_kind)
        protocol = _required_text(
            self.worker_protocol_version,
            label="template_worker_protocol_version",
            maximum=128,
        )
        if not _VERSION_RE.fullmatch(protocol):
            raise ValueError("template_worker_protocol_version_invalid")
        object.__setattr__(self, "worker_protocol_version", protocol)
        object.__setattr__(
            self,
            "platforms",
            _unique_sorted_strings(
                self.platforms,
                label="template_platforms",
                maximum=16,
                normalizer=lambda item: _required_text(item, label="template_platform").lower(),
            ),
        )
        if not self.platforms:
            raise ValueError("template_platforms_invalid")
        try:
            for item in self.platforms:
                validate_target_platform(item, label="template_platform")
        except ValueError as exc:
            raise ValueError("template_platforms_invalid") from exc
        distributions = tuple(self.locked_distributions)
        if not distributions or len(distributions) > MAX_TEMPLATE_DISTRIBUTIONS:
            raise ValueError("template_locked_distributions_invalid")
        if any(not isinstance(item, ToolboxLockedDistributionSpec) for item in distributions):
            raise ValueError("template_locked_distribution_type_invalid")
        if len({item.name for item in distributions}) != len(distributions):
            raise ValueError("template_locked_distribution_duplicate")
        object.__setattr__(self, "locked_distributions", tuple(sorted(distributions)))
        roots = _unique_sorted_strings(
            self.exposed_import_roots,
            label="template_exposed_import_roots",
            maximum=MAX_TEMPLATE_IMPORT_ROOTS,
            normalizer=normalize_import_root,
        )
        object.__setattr__(self, "exposed_import_roots", roots)
        object.__setattr__(self, "lock_digest", _canonical_digest(self.lock_digest, label="template_lock_digest"))
        object.__setattr__(
            self,
            "parent_worker_artifact_digest",
            _canonical_digest(self.parent_worker_artifact_digest, label="parent_worker_artifact_digest"),
        )
        isolation = _required_text(
            self.isolation_policy_version,
            label="template_isolation_policy_version",
            maximum=128,
        )
        if not _VERSION_RE.fullmatch(isolation):
            raise ValueError("template_isolation_policy_version_invalid")
        object.__setattr__(self, "isolation_policy_version", isolation)
        if not isinstance(self.provenance, ToolboxTemplateProvenance):
            raise ValueError("template_provenance_type_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "python_requires": self.python_requires,
            "python_abis": list(self.python_abis),
            "runtime_kind": self.runtime_kind,
            "worker_protocol_version": self.worker_protocol_version,
            "platforms": list(self.platforms),
            "locked_distributions": [item.to_dict() for item in self.locked_distributions],
            "exposed_import_roots": list(self.exposed_import_roots),
            "lock_digest": self.lock_digest,
            "parent_worker_artifact_digest": self.parent_worker_artifact_digest,
            "isolation_policy_version": self.isolation_policy_version,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxEnvironmentTemplateSpec":
        row = dict(payload or {})
        fields = {
            "template_id",
            "python_requires",
            "python_abis",
            "runtime_kind",
            "worker_protocol_version",
            "platforms",
            "locked_distributions",
            "exposed_import_roots",
            "lock_digest",
            "parent_worker_artifact_digest",
            "isolation_policy_version",
            "provenance",
        }
        _strict_fields(row, fields, label="toolbox_environment_template")
        if set(row) != fields:
            missing = ",".join(sorted(fields - set(row)))
            raise ValueError(f"toolbox_environment_template_missing_fields:{missing}")
        return cls(
            template_id=row["template_id"],
            python_requires=row["python_requires"],
            python_abis=tuple(_sequence(row["python_abis"], label="template_python_abis", maximum=16)),
            runtime_kind=row["runtime_kind"],
            worker_protocol_version=row["worker_protocol_version"],
            platforms=tuple(_sequence(row["platforms"], label="template_platforms", maximum=16)),
            locked_distributions=tuple(
                ToolboxLockedDistributionSpec.from_dict(item)
                for item in _sequence(
                    row["locked_distributions"],
                    label="template_locked_distributions",
                    maximum=MAX_TEMPLATE_DISTRIBUTIONS,
                )
            ),
            exposed_import_roots=tuple(
                _sequence(
                    row["exposed_import_roots"],
                    label="template_exposed_import_roots",
                    maximum=MAX_TEMPLATE_IMPORT_ROOTS,
                )
            ),
            lock_digest=row["lock_digest"],
            parent_worker_artifact_digest=row["parent_worker_artifact_digest"],
            isolation_policy_version=row["isolation_policy_version"],
            provenance=ToolboxTemplateProvenance.from_dict(row["provenance"]),
        )


@dataclass(frozen=True)
class ReviewedImportDistributionRule:
    """Reviewed mapping from one or more import roots to one distribution."""

    distribution: str
    import_roots: tuple[str, ...]
    package_aliases: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()
    version_constraint: str = ""
    provenance: str = "phase0-inventory"

    def __post_init__(self) -> None:
        object.__setattr__(self, "distribution", normalize_distribution_name(self.distribution))
        roots = _unique_sorted_strings(
            self.import_roots,
            label="catalog_import_roots",
            maximum=MAX_ALIASES_PER_RULE,
            normalizer=normalize_import_root,
        )
        if not roots:
            raise ValueError("catalog_import_roots_required")
        object.__setattr__(self, "import_roots", roots)
        aliases = _unique_sorted_strings(
            self.package_aliases,
            label="catalog_package_aliases",
            maximum=MAX_ALIASES_PER_RULE,
            normalizer=normalize_distribution_name,
        )
        if self.distribution in aliases:
            raise ValueError("catalog_package_alias_redundant")
        object.__setattr__(self, "package_aliases", aliases)
        extras = _unique_sorted_strings(
            self.extras,
            label="catalog_extras",
            maximum=32,
            normalizer=normalize_distribution_name,
        )
        object.__setattr__(self, "extras", extras)
        object.__setattr__(self, "version_constraint", normalize_version_constraint(self.version_constraint))
        provenance = _required_text(self.provenance, label="catalog_provenance")
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "import_roots": list(self.import_roots),
            "package_aliases": list(self.package_aliases),
            "extras": list(self.extras),
            "version_constraint": self.version_constraint,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewedImportDistributionRule":
        row = dict(payload or {})
        fields = {
            "distribution",
            "import_roots",
            "package_aliases",
            "extras",
            "version_constraint",
            "provenance",
        }
        _strict_fields(row, fields, label="catalog_rule")
        if set(row) != fields:
            missing = ",".join(sorted(fields - set(row)))
            raise ValueError(f"catalog_rule_missing_fields:{missing}")
        return cls(
            distribution=row["distribution"],
            import_roots=tuple(
                _sequence(row["import_roots"], label="catalog_import_roots", maximum=MAX_ALIASES_PER_RULE)
            ),
            package_aliases=tuple(
                _sequence(row["package_aliases"], label="catalog_package_aliases", maximum=MAX_ALIASES_PER_RULE)
            ),
            extras=tuple(_sequence(row["extras"], label="catalog_extras", maximum=32)),
            version_constraint=row["version_constraint"],
            provenance=row["provenance"],
        )


@dataclass(frozen=True)
class ReviewedImportDistributionCatalog:
    rules: tuple[ReviewedImportDistributionRule, ...]
    _by_import: Mapping[str, ReviewedImportDistributionRule] = field(init=False, repr=False, compare=False)
    _by_package: Mapping[str, ReviewedImportDistributionRule] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if len(rules) > MAX_CATALOG_RULES:
            raise ValueError("catalog_too_many_rules")
        if any(not isinstance(item, ReviewedImportDistributionRule) for item in rules):
            raise ValueError("catalog_rule_type_invalid")
        by_import: dict[str, ReviewedImportDistributionRule] = {}
        by_package: dict[str, ReviewedImportDistributionRule] = {}
        for rule in rules:
            for root in rule.import_roots:
                if root in by_import:
                    raise ValueError(f"catalog_import_root_ambiguous:{root}")
                by_import[root] = rule
            for package in (rule.distribution, *rule.package_aliases):
                if package in by_package:
                    raise ValueError(f"catalog_package_alias_ambiguous:{package}")
                by_package[package] = rule
        ordered = tuple(sorted(rules, key=lambda item: (item.distribution, item.import_roots)))
        object.__setattr__(self, "rules", ordered)
        object.__setattr__(self, "_by_import", by_import)
        object.__setattr__(self, "_by_package", by_package)

    def for_import(self, import_name: Any) -> ReviewedImportDistributionRule | None:
        name = _required_text(import_name, label="import_name")
        root = normalize_import_root(name.split(".", 1)[0])
        return self._by_import.get(root)

    def for_distribution(self, package_name: Any) -> ReviewedImportDistributionRule | None:
        return self._by_package.get(normalize_distribution_name(package_name))

    def to_dict(self) -> dict[str, Any]:
        return {"contract": "hosting.toolbox.import_distribution_catalog.v1", "rules": [r.to_dict() for r in self.rules]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewedImportDistributionCatalog":
        row = dict(payload or {})
        _strict_fields(row, {"contract", "rules"}, label="import_distribution_catalog")
        if row.get("contract") != "hosting.toolbox.import_distribution_catalog.v1":
            raise ValueError("import_distribution_catalog_contract_invalid")
        return cls(
            rules=tuple(
                ReviewedImportDistributionRule.from_dict(item)
                for item in _sequence(row.get("rules"), label="catalog_rules", maximum=MAX_CATALOG_RULES)
            )
        )


PHASE0_REVIEWED_IMPORT_RULES: tuple[ReviewedImportDistributionRule, ...] = (
    ReviewedImportDistributionRule(
        distribution="numpy",
        import_roots=("numpy",),
        version_constraint=">=1.26.0",
    ),
    ReviewedImportDistributionRule(
        distribution="sympy",
        import_roots=("sympy",),
        version_constraint="==1.14.0",
    ),
    ReviewedImportDistributionRule(
        distribution="numexpr",
        import_roots=("numexpr",),
        version_constraint=">=2.11.0",
    ),
    ReviewedImportDistributionRule(
        distribution="requests",
        import_roots=("requests",),
        version_constraint=">=2.26.0",
    ),
    ReviewedImportDistributionRule(
        distribution="matplotlib",
        import_roots=("matplotlib",),
        version_constraint=">=3.10.1,<4",
    ),
)

PHASE0_REVIEWED_IMPORT_CATALOG = ReviewedImportDistributionCatalog(PHASE0_REVIEWED_IMPORT_RULES)


__all__ = [
    "MAX_CATALOG_RULES",
    "MAX_TEMPLATE_DISTRIBUTIONS",
    "MAX_TEMPLATE_IMPORT_ROOTS",
    "PHASE0_REVIEWED_IMPORT_CATALOG",
    "PHASE0_REVIEWED_IMPORT_RULES",
    "ReviewedImportDistributionCatalog",
    "ReviewedImportDistributionRule",
    "ToolboxEnvironmentTemplateSpec",
    "ToolboxLockedDistributionSpec",
    "ToolboxTemplateProvenance",
    "normalize_distribution_name",
    "normalize_import_root",
    "normalize_version_constraint",
]
