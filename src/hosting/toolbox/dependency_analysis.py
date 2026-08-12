"""Side-effect-free toolbox source analysis and template selection."""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .bundle_models import ToolboxBundleFile
from .catalog import (
    PHASE0_REVIEWED_IMPORT_CATALOG,
    ReviewedImportDistributionCatalog,
    ToolboxEnvironmentTemplateSpec,
    normalize_distribution_name,
    normalize_import_root,
)


MAX_ANALYZED_IMPORTS = 512
MAX_IMPORT_EVIDENCE = 16
PARENT_RUNTIME_IMPORT_ROOTS = frozenset({"hosting", "mp13_engine"})


@dataclass(frozen=True, order=True)
class ToolboxImportEvidence:
    relative_path: str
    line: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "line": self.line,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxImportEvidence":
        row = dict(payload or {})
        if set(row) != {"relative_path", "line", "kind"}:
            raise ValueError("toolbox_import_evidence_fields_invalid")
        if (
            not isinstance(row["relative_path"], str)
            or not row["relative_path"]
            or isinstance(row["line"], bool)
            or not isinstance(row["line"], int)
            or row["line"] < 0
            or not isinstance(row["kind"], str)
            or not row["kind"]
        ):
            raise ValueError("toolbox_import_evidence_invalid")
        return cls(**row)


@dataclass(frozen=True)
class ToolboxDependencyDiagnostic:
    code: str
    summary: str
    relative_path: str
    line: int
    import_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "summary": self.summary,
            "relative_path": self.relative_path,
            "line": self.line,
            "import_root": self.import_root,
        }


class ToolboxDependencyAnalysisError(ValueError):
    def __init__(self, diagnostics: Sequence[ToolboxDependencyDiagnostic]):
        self.diagnostics = tuple(diagnostics)
        code = self.diagnostics[0].code if self.diagnostics else "dependency_analysis_failed"
        super().__init__(code)


@dataclass(frozen=True)
class ToolboxAnalyzedImport:
    import_root: str
    classification: str
    distribution: str | None
    evidence: tuple[ToolboxImportEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_root": self.import_root,
            "classification": self.classification,
            "distribution": self.distribution,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxAnalyzedImport":
        row = dict(payload or {})
        if set(row) != {"import_root", "classification", "distribution", "evidence"}:
            raise ValueError("toolbox_analyzed_import_fields_invalid")
        if row["classification"] not in {
            "standard_library", "local_staged", "parent_runtime",
            "known_third_party", "declared_dynamic", "unresolved",
        } or not isinstance(row["evidence"], list) or len(row["evidence"]) > MAX_IMPORT_EVIDENCE:
            raise ValueError("toolbox_analyzed_import_invalid")
        root = normalize_import_root(row["import_root"])
        distribution = row["distribution"]
        if distribution is not None:
            distribution = normalize_distribution_name(distribution)
        evidence = tuple(ToolboxImportEvidence.from_dict(item) for item in row["evidence"])
        if len(set(evidence)) != len(evidence):
            raise ValueError("toolbox_import_evidence_duplicate")
        return cls(root, row["classification"], distribution, evidence)


@dataclass(frozen=True)
class ToolboxSourceAnalysis:
    imports: tuple[ToolboxAnalyzedImport, ...]
    diagnostics: tuple[ToolboxDependencyDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "imports": [item.to_dict() for item in self.imports],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class ToolboxResolvedRequirement:
    distribution: str
    extras: tuple[str, ...]
    constraint: str
    import_roots: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "extras": list(self.extras),
            "constraint": self.constraint,
            "import_roots": list(self.import_roots),
        }


@dataclass(frozen=True)
class ToolboxResolvedDependencies:
    requirements: tuple[ToolboxResolvedRequirement, ...]
    analysis: ToolboxSourceAnalysis

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": [item.to_dict() for item in self.requirements],
            "analysis": self.analysis.to_dict(),
        }


@dataclass(frozen=True)
class ToolboxTemplateSelection:
    mode: str
    template: ToolboxEnvironmentTemplateSpec
    custom_delta: tuple[ToolboxResolvedRequirement, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"template", "custom"}:
            raise ValueError("template_selection_mode_invalid")
        if self.mode == "template" and self.custom_delta:
            raise ValueError("template_selection_delta_unexpected")
        if self.mode == "custom" and not self.custom_delta:
            raise ValueError("template_selection_delta_required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "template_id": self.template.template_id,
            "template_lock_digest": self.template.lock_digest,
            "custom_delta": [item.to_dict() for item in self.custom_delta],
        }


@dataclass(frozen=True)
class _ImportOccurrence:
    root: str
    path: str
    line: int
    kind: str
    requires_declaration: bool = False
    relative: bool = False


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str, local_root: str):
        self.relative_path = relative_path
        self.local_root = local_root
        self.optional_depth = 0
        self.type_checking_depth = 0
        self.occurrences: list[_ImportOccurrence] = []
        self.diagnostics: list[ToolboxDependencyDiagnostic] = []

    def _kind(self, default: str) -> tuple[str, bool]:
        if self.type_checking_depth:
            return "type_checking_import", True
        if self.optional_depth:
            return "optional_import", True
        return default, False

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        kind, declaration = self._kind("import")
        for alias in node.names:
            self.occurrences.append(
                _ImportOccurrence(
                    root=alias.name.split(".", 1)[0],
                    path=self.relative_path,
                    line=node.lineno,
                    kind=kind,
                    requires_declaration=declaration,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        kind, declaration = self._kind("from_import")
        relative = node.level > 0
        if relative:
            kind = "relative_import"
            root = self.local_root
        else:
            root = str(node.module or "").split(".", 1)[0]
        if root:
            self.occurrences.append(
                _ImportOccurrence(
                    root=root,
                    path=self.relative_path,
                    line=node.lineno,
                    kind=kind,
                    requires_declaration=declaration and not relative,
                    relative=relative,
                )
            )

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        optional = any(
            (
                isinstance(handler.type, ast.Name)
                and handler.type.id in {"ImportError", "ModuleNotFoundError"}
            )
            or (
                isinstance(handler.type, ast.Tuple)
                and any(
                    isinstance(item, ast.Name)
                    and item.id in {"ImportError", "ModuleNotFoundError"}
                    for item in handler.type.elts
                )
            )
            for handler in node.handlers
        )
        if optional:
            self.optional_depth += 1
        for item in node.body:
            self.visit(item)
        if optional:
            self.optional_depth -= 1
        for handler in node.handlers:
            for item in handler.body:
                self.visit(item)
        for item in (*node.orelse, *node.finalbody):
            self.visit(item)

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        is_type_checking = (
            isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ) or (
            isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "typing"
            and node.test.attr == "TYPE_CHECKING"
        )
        if is_type_checking:
            self.type_checking_depth += 1
        for item in node.body:
            self.visit(item)
        if is_type_checking:
            self.type_checking_depth -= 1
        for item in node.orelse:
            self.visit(item)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        is_dynamic = (
            isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
        ) or (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
        )
        if is_dynamic:
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                value = node.args[0].value.strip()
                relative = value.startswith(".")
                raw = value.lstrip(".")
                if raw:
                    self.occurrences.append(
                        _ImportOccurrence(
                            root=self.local_root if relative else raw.split(".", 1)[0],
                            path=self.relative_path,
                            line=node.lineno,
                            kind="relative_import" if relative else "dynamic_import",
                            requires_declaration=not relative,
                            relative=relative,
                        )
                    )
            else:
                self.diagnostics.append(
                    ToolboxDependencyDiagnostic(
                        code="dynamic_import_unresolved",
                        summary="A dynamic import target must have a literal root and an explicit declaration.",
                        relative_path=self.relative_path,
                        line=node.lineno,
                    )
                )
        self.generic_visit(node)


def _normalized_files(
    files: Sequence[ToolboxBundleFile | Mapping[str, Any]],
) -> tuple[ToolboxBundleFile, ...]:
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)):
        raise ValueError("bundle_files_must_be_array")
    normalized: list[ToolboxBundleFile] = []
    seen: set[str] = set()
    for raw in files:
        item = raw if isinstance(raw, ToolboxBundleFile) else ToolboxBundleFile.from_runtime_dict(dict(raw))
        path = item.normalized_path()
        key = path.casefold()
        if key in seen:
            raise ToolboxDependencyAnalysisError(
                (
                    ToolboxDependencyDiagnostic(
                        code="duplicate_staged_path",
                        summary="Staged Python paths must be unique after normalization.",
                        relative_path=path,
                        line=0,
                    ),
                )
            )
        seen.add(key)
        normalized.append(ToolboxBundleFile(relative_path=path, content=item.content))
    return tuple(sorted(normalized, key=lambda item: item.relative_path.casefold()))


def _local_roots(files: Iterable[ToolboxBundleFile]) -> frozenset[str]:
    roots: set[str] = set()
    for item in files:
        path = PurePosixPath(item.relative_path)
        if path.suffix != ".py":
            continue
        first = path.parts[0]
        if first == "__init__.py":
            continue
        roots.add(first[:-3] if first.endswith(".py") else first)
    return frozenset(roots)


def analyze_toolbox_bundle_imports(
    files: Sequence[ToolboxBundleFile | Mapping[str, Any]],
    *,
    declared_imports: Sequence[str] = (),
    catalog: ReviewedImportDistributionCatalog = PHASE0_REVIEWED_IMPORT_CATALOG,
) -> ToolboxSourceAnalysis:
    normalized_files = _normalized_files(files)
    if not isinstance(declared_imports, Sequence) or isinstance(
        declared_imports, (str, bytes, bytearray)
    ):
        raise ValueError("declared_imports_must_be_array")
    if any(not isinstance(item, str) for item in declared_imports):
        raise ValueError("declared_import_must_be_string")
    declared = {normalize_import_root(item.split(".", 1)[0]) for item in declared_imports}
    local_roots = _local_roots(normalized_files)
    occurrences: list[_ImportOccurrence] = []
    diagnostics: list[ToolboxDependencyDiagnostic] = []
    for item in normalized_files:
        if not item.relative_path.endswith(".py"):
            continue
        try:
            tree = ast.parse(item.content, filename=item.relative_path)
        except SyntaxError as exc:
            raise ToolboxDependencyAnalysisError(
                (
                    ToolboxDependencyDiagnostic(
                        code="source_syntax_error",
                        summary=str(exc.msg or "Python source is invalid."),
                        relative_path=item.relative_path,
                        line=int(exc.lineno or 0),
                    ),
                )
            ) from exc
        path = PurePosixPath(item.relative_path)
        local_root = path.parts[0]
        local_root = local_root[:-3] if local_root.endswith(".py") else local_root
        visitor = _ImportVisitor(relative_path=item.relative_path, local_root=local_root)
        visitor.visit(tree)
        occurrences.extend(visitor.occurrences)
        diagnostics.extend(visitor.diagnostics)

    by_root: dict[str, list[_ImportOccurrence]] = {}
    for occurrence in occurrences:
        root = normalize_import_root(occurrence.root)
        by_root.setdefault(root, []).append(occurrence)
    for root in sorted(declared - set(by_root)):
        by_root[root] = [
            _ImportOccurrence(
                root=root,
                path="<definition>",
                line=0,
                kind="declaration",
                requires_declaration=True,
            )
        ]
    if len(by_root) > MAX_ANALYZED_IMPORTS:
        raise ValueError("analyzed_imports_too_many")

    imports: list[ToolboxAnalyzedImport] = []
    classification_rank = {
        "standard_library": 0,
        "local_staged": 1,
        "parent_runtime": 2,
        "known_third_party": 3,
        "declared_dynamic": 4,
        "unresolved": 5,
    }
    for root, root_occurrences in sorted(by_root.items()):
        classifications: list[str] = []
        for occurrence in root_occurrences:
            if occurrence.relative or root in local_roots:
                classification = "local_staged"
            elif root in sys.stdlib_module_names:
                classification = "standard_library"
            elif root in PARENT_RUNTIME_IMPORT_ROOTS:
                classification = "parent_runtime"
            elif occurrence.requires_declaration:
                classification = "declared_dynamic" if root in declared else "unresolved"
            elif catalog.for_import(root) is not None:
                classification = "known_third_party"
            else:
                classification = "unresolved"
            classifications.append(classification)
        classification = max(classifications, key=classification_rank.__getitem__)
        rule = catalog.for_import(root)
        evidence = tuple(
            sorted(
                {
                    ToolboxImportEvidence(item.path, item.line, item.kind)
                    for item in root_occurrences
                }
            )[:MAX_IMPORT_EVIDENCE]
        )
        imports.append(
            ToolboxAnalyzedImport(
                import_root=root,
                classification=classification,
                distribution=rule.distribution if rule is not None else None,
                evidence=evidence,
            )
        )
    return ToolboxSourceAnalysis(imports=tuple(imports), diagnostics=tuple(diagnostics))


def _parse_requirement(value: Any) -> Requirement:
    if not isinstance(value, str):
        raise ValueError("package_requirement_must_be_string")
    try:
        requirement = Requirement(value.strip())
    except InvalidRequirement as exc:
        raise ValueError("package_requirement_invalid") from exc
    if requirement.marker is not None or requirement.url is not None or not str(requirement.specifier):
        raise ValueError("package_requirement_invalid")
    return requirement


def _next_prefix_version(version: Version, *, compatible: bool = False) -> Version:
    release = list(version.release)
    if compatible and len(release) > 1:
        release = release[:-1]
    release[-1] += 1
    return Version(".".join(str(item) for item in release))


def _range_conflicts(specifiers: Sequence[SpecifierSet]) -> bool:
    lower: tuple[Version, bool] | None = None
    upper: tuple[Version, bool] | None = None
    for specifier_set in specifiers:
        for specifier in specifier_set:
            operator = specifier.operator
            raw = specifier.version
            if operator == "==" and raw.endswith(".*"):
                version = Version(raw[:-2])
                candidate_lower = (version, True)
                candidate_upper = (_next_prefix_version(version), False)
            elif operator == "~=":
                version = Version(raw)
                candidate_lower = (version, True)
                candidate_upper = (_next_prefix_version(version, compatible=True), False)
            elif operator in {">", ">="}:
                candidate_lower = (Version(raw), operator == ">=")
                candidate_upper = None
            elif operator in {"<", "<="}:
                candidate_lower = None
                candidate_upper = (Version(raw), operator == "<=")
            else:
                continue
            if candidate_lower is not None and (
                lower is None
                or candidate_lower[0] > lower[0]
                or (
                    candidate_lower[0] == lower[0]
                    and not candidate_lower[1]
                    and lower[1]
                )
            ):
                lower = candidate_lower
            if candidate_upper is not None and (
                upper is None
                or candidate_upper[0] < upper[0]
                or (
                    candidate_upper[0] == upper[0]
                    and not candidate_upper[1]
                    and upper[1]
                )
            ):
                upper = candidate_upper
    if lower is None or upper is None:
        return False
    return lower[0] > upper[0] or (
        lower[0] == upper[0] and not (lower[1] and upper[1])
    )


def _merge_specifiers(constraints: Iterable[str]) -> str:
    try:
        specifiers = [SpecifierSet(item) for item in constraints if item]
    except InvalidSpecifier as exc:
        raise ValueError("package_requirement_invalid") from exc
    merged = SpecifierSet(",".join(str(item) for item in specifiers))
    try:
        exact = {
            Version(specifier.version)
            for specifier in merged
            if specifier.operator in {"==", "==="} and "*" not in specifier.version
        }
    except InvalidVersion as exc:
        raise ValueError("package_requirement_invalid") from exc
    if len(exact) > 1 or (exact and not all(next(iter(exact)) in item for item in specifiers)):
        raise ValueError("dependency_requirement_conflict")
    if not exact and _range_conflicts(specifiers):
        raise ValueError("dependency_requirement_conflict")
    return str(merged)


def resolve_toolbox_dependencies(
    analysis: ToolboxSourceAnalysis,
    *,
    package_requirements: Sequence[str] = (),
    catalog: ReviewedImportDistributionCatalog = PHASE0_REVIEWED_IMPORT_CATALOG,
) -> ToolboxResolvedDependencies:
    diagnostics = list(analysis.diagnostics)
    constraints: dict[str, list[str]] = {}
    extras: dict[str, set[str]] = {}
    roots: dict[str, set[str]] = {}
    for item in analysis.imports:
        if item.classification in {"standard_library", "local_staged", "parent_runtime"}:
            continue
        if item.classification == "unresolved" or item.distribution is None:
            evidence = item.evidence[0]
            diagnostics.append(
                ToolboxDependencyDiagnostic(
                    code="dependency_unresolved",
                    summary=f"Import root '{item.import_root}' has no reviewed dependency mapping.",
                    relative_path=evidence.relative_path,
                    line=evidence.line,
                    import_root=item.import_root,
                )
            )
            continue
        rule = catalog.for_import(item.import_root)
        if rule is None:
            continue
        constraints.setdefault(rule.distribution, []).append(rule.version_constraint)
        extras.setdefault(rule.distribution, set()).update(rule.extras)
        roots.setdefault(rule.distribution, set()).add(item.import_root)

    for raw in package_requirements:
        requirement = _parse_requirement(raw)
        distribution = normalize_distribution_name(requirement.name)
        rule = catalog.for_distribution(distribution)
        if rule is None:
            diagnostics.append(
                ToolboxDependencyDiagnostic(
                    code="dependency_package_unreviewed",
                    summary=f"Distribution '{distribution}' is not in the reviewed catalog.",
                    relative_path="<definition>",
                    line=0,
                )
            )
            continue
        requested_extras = {normalize_distribution_name(item) for item in requirement.extras}
        if not requested_extras.issubset(set(rule.extras)):
            diagnostics.append(
                ToolboxDependencyDiagnostic(
                    code="dependency_extra_unreviewed",
                    summary=f"Distribution '{distribution}' requests an unreviewed extra.",
                    relative_path="<definition>",
                    line=0,
                )
            )
            continue
        canonical = rule.distribution
        constraints.setdefault(canonical, []).extend(
            [rule.version_constraint, str(requirement.specifier)]
        )
        extras.setdefault(canonical, set()).update(rule.extras)
        roots.setdefault(canonical, set()).update(rule.import_roots)
    if diagnostics:
        raise ToolboxDependencyAnalysisError(tuple(diagnostics))

    resolved: list[ToolboxResolvedRequirement] = []
    for distribution in sorted(constraints):
        try:
            constraint = _merge_specifiers(constraints[distribution])
        except ValueError as exc:
            if str(exc) != "dependency_requirement_conflict":
                raise
            raise ToolboxDependencyAnalysisError(
                (
                    ToolboxDependencyDiagnostic(
                        code="dependency_requirement_conflict",
                        summary=f"Distribution '{distribution}' has incompatible constraints.",
                        relative_path="<definition>",
                        line=0,
                    ),
                )
            ) from exc
        resolved.append(
            ToolboxResolvedRequirement(
                distribution=distribution,
                extras=tuple(sorted(extras.get(distribution, set()))),
                constraint=constraint,
                import_roots=tuple(sorted(roots.get(distribution, set()))),
            )
        )
    return ToolboxResolvedDependencies(requirements=tuple(resolved), analysis=analysis)


def _target_templates(
    templates: Sequence[ToolboxEnvironmentTemplateSpec],
    *,
    python_abi: str,
    platform: str,
    allowed_template_ids: Sequence[str] | None,
) -> list[ToolboxEnvironmentTemplateSpec]:
    allowed = set(allowed_template_ids) if allowed_template_ids is not None else None
    return [
        template
        for template in templates
        if python_abi in template.python_abis
        and platform in template.platforms
        and (allowed is None or template.template_id in allowed)
    ]


def _template_delta(
    template: ToolboxEnvironmentTemplateSpec,
    requirements: Sequence[ToolboxResolvedRequirement],
) -> tuple[ToolboxResolvedRequirement, ...] | None:
    locked = {item.name: item for item in template.locked_distributions}
    delta: list[ToolboxResolvedRequirement] = []
    for requirement in requirements:
        installed = locked.get(requirement.distribution)
        if installed is None:
            delta.append(requirement)
            continue
        try:
            covered = Version(installed.version) in SpecifierSet(requirement.constraint)
        except (InvalidVersion, InvalidSpecifier) as exc:
            raise ValueError("template_lock_version_invalid") from exc
        if not covered:
            return None
        if not set(requirement.import_roots).issubset(set(template.exposed_import_roots)):
            delta.append(requirement)
    return tuple(delta)


def select_toolbox_environment_template(
    dependencies: ToolboxResolvedDependencies,
    templates: Sequence[ToolboxEnvironmentTemplateSpec],
    *,
    python_abi: str,
    platform: str,
    allowed_template_ids: Sequence[str] | None = None,
) -> ToolboxTemplateSelection:
    candidates = _target_templates(
        templates,
        python_abi=python_abi,
        platform=platform,
        allowed_template_ids=allowed_template_ids,
    )
    if not candidates:
        raise ValueError("template_target_unavailable")
    evaluated = [
        (template, _template_delta(template, dependencies.requirements))
        for template in candidates
    ]
    exact = [
        template
        for template, delta in evaluated
        if delta is not None and not delta
    ]
    if exact:
        selected = min(
            exact,
            key=lambda item: (len(item.locked_distributions), item.template_id, item.lock_digest),
        )
        return ToolboxTemplateSelection(mode="template", template=selected, custom_delta=())
    custom = [
        (template, delta)
        for template, delta in evaluated
        if delta is not None
    ]
    if not custom:
        raise ValueError("template_requirement_conflict")
    selected, delta = min(
        custom,
        key=lambda item: (
            len(item[1] or ()),
            len(item[0].locked_distributions),
            item[0].template_id,
            item[0].lock_digest,
        ),
    )
    return ToolboxTemplateSelection(mode="custom", template=selected, custom_delta=tuple(delta or ()))


__all__ = [
    "MAX_ANALYZED_IMPORTS",
    "MAX_IMPORT_EVIDENCE",
    "PARENT_RUNTIME_IMPORT_ROOTS",
    "ToolboxAnalyzedImport",
    "ToolboxDependencyAnalysisError",
    "ToolboxDependencyDiagnostic",
    "ToolboxImportEvidence",
    "ToolboxResolvedDependencies",
    "ToolboxResolvedRequirement",
    "ToolboxSourceAnalysis",
    "ToolboxTemplateSelection",
    "analyze_toolbox_bundle_imports",
    "resolve_toolbox_dependencies",
    "select_toolbox_environment_template",
]
