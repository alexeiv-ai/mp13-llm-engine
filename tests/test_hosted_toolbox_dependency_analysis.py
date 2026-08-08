from __future__ import annotations

import copy

import pytest

from hosting.toolbox.bundle_models import ToolboxBundleFile
from hosting.toolbox.catalog import (
    ReviewedImportDistributionCatalog,
    ReviewedImportDistributionRule,
    ToolboxEnvironmentTemplateSpec,
)
from hosting.toolbox.dependency_analysis import (
    ToolboxDependencyAnalysisError,
    analyze_toolbox_bundle_imports,
    resolve_toolbox_dependencies,
    select_toolbox_environment_template,
)


def _digest(char: str) -> str:
    return f"sha256:{char * 64}"


def _template(
    template_id: str,
    distributions: list[tuple[str, str]],
    import_roots: list[str],
    *,
    platforms: list[str] | None = None,
) -> ToolboxEnvironmentTemplateSpec:
    return ToolboxEnvironmentTemplateSpec.from_dict(
        {
            "template_id": template_id,
            "python_requires": ">=3.12,<3.13",
            "python_abis": ["cp312"],
            "runtime_kind": "toolbox_python",
            "worker_protocol_version": "1.0",
            "platforms": platforms or ["win_amd64"],
            "locked_distributions": [
                {"name": name, "version": version, "extras": []}
                for name, version in distributions
            ],
            "exposed_import_roots": import_roots,
            "lock_digest": _digest("a" if template_id == "core" else "b"),
            "parent_worker_artifact_digest": _digest("c"),
            "isolation_policy_version": "1.0",
            "provenance": {
                "source": "test",
                "revision": "1",
                "manifest_digest": _digest("d"),
                "signing_key_id": "test-key",
            },
        }
    )


def _templates() -> tuple[ToolboxEnvironmentTemplateSpec, ...]:
    core = _template(
        "core",
        [("hosting-runtime", "1.0")],
        ["hosting", "mp13_engine"],
    )
    compute = _template(
        "py-compute",
        [
            ("hosting-runtime", "1.0"),
            ("numpy", "2.4.3"),
            ("sympy", "1.14.0"),
            ("numexpr", "2.14.1"),
        ],
        ["hosting", "mp13_engine", "numpy", "sympy", "numexpr"],
    )
    return core, compute


def _analyze(source: str, *, declared: tuple[str, ...] = ()):
    return analyze_toolbox_bundle_imports(
        [ToolboxBundleFile(relative_path="pkg/main.py", content=source)],
        declared_imports=declared,
    )


def test_ast_analysis_classifies_standard_local_parent_and_known_imports() -> None:
    files = [
        ToolboxBundleFile(relative_path="pkg/__init__.py", content=""),
        ToolboxBundleFile(relative_path="pkg/local.py", content="VALUE = 1\n"),
        ToolboxBundleFile(
            relative_path="pkg/main.py",
            content=(
                "import json\n"
                "from . import local\n"
                "from pkg.local import VALUE\n"
                "from hosting.toolbox import ToolboxBundleFile\n"
                "from mp13_engine.mp13_config import ToolCall\n"
                "import numpy as np\n"
            ),
        ),
    ]
    analysis = analyze_toolbox_bundle_imports(files)
    by_root = {item.import_root: item for item in analysis.imports}
    assert {root: item.classification for root, item in by_root.items()} == {
        "hosting": "parent_runtime",
        "json": "standard_library",
        "mp13_engine": "parent_runtime",
        "numpy": "known_third_party",
        "pkg": "local_staged",
    }
    assert by_root["numpy"].distribution == "numpy"
    assert [item.line for item in by_root["pkg"].evidence] == [2, 3]


def test_optional_dynamic_and_type_checking_imports_require_declarations() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "import importlib\n"
        "try:\n"
        "    import requests\n"
        "except ImportError:\n"
        "    requests = None\n"
        "chart = importlib.import_module('matplotlib.pyplot')\n"
        "if TYPE_CHECKING:\n"
        "    import sympy\n"
    )
    undeclared = _analyze(source)
    unresolved = {
        item.import_root
        for item in undeclared.imports
        if item.classification == "unresolved"
    }
    assert unresolved == {"matplotlib", "requests", "sympy"}
    declared = _analyze(source, declared=("sympy", "requests", "matplotlib.pyplot"))
    by_root = {item.import_root: item for item in declared.imports}
    assert by_root["requests"].classification == "declared_dynamic"
    assert by_root["matplotlib"].classification == "declared_dynamic"
    assert by_root["sympy"].classification == "declared_dynamic"
    assert by_root["requests"].evidence[0].kind == "optional_import"
    assert by_root["matplotlib"].evidence[0].kind == "dynamic_import"
    assert by_root["sympy"].evidence[0].kind == "type_checking_import"


def test_declared_import_without_static_evidence_is_resolved() -> None:
    analysis = _analyze("def load_plugin():\n    return None\n", declared=("requests",))
    assert analysis.imports[0].classification == "declared_dynamic"
    assert analysis.imports[0].evidence[0].relative_path == "<definition>"
    resolved = resolve_toolbox_dependencies(analysis)
    assert resolved.requirements[0].distribution == "requests"


def test_literal_relative_dynamic_import_is_local_without_declaration() -> None:
    analysis = _analyze("import importlib\nplugin = importlib.import_module('.plugin', __package__)\n")
    by_root = {item.import_root: item for item in analysis.imports}
    assert by_root["pkg"].classification == "local_staged"
    assert by_root["pkg"].evidence[0].kind == "relative_import"


def test_dynamic_expression_is_a_bounded_file_line_error() -> None:
    analysis = _analyze("import importlib\nname = 'numpy'\nimportlib.import_module(name)\n")
    with pytest.raises(ToolboxDependencyAnalysisError) as caught:
        resolve_toolbox_dependencies(analysis)
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "dynamic_import_unresolved"
    assert diagnostic.relative_path == "pkg/main.py"
    assert diagnostic.line == 3


def test_unresolved_import_reports_exact_file_and_line() -> None:
    analysis = _analyze("x = 1\nimport unknown_package\n")
    with pytest.raises(ToolboxDependencyAnalysisError) as caught:
        resolve_toolbox_dependencies(analysis)
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "dependency_unresolved"
    assert diagnostic.import_root == "unknown_package"
    assert diagnostic.relative_path == "pkg/main.py"
    assert diagnostic.line == 2


def test_syntax_and_duplicate_staged_paths_fail_before_resolution() -> None:
    with pytest.raises(ToolboxDependencyAnalysisError) as syntax:
        _analyze("def broken(:\n    pass\n")
    assert syntax.value.diagnostics[0].code == "source_syntax_error"
    assert syntax.value.diagnostics[0].line == 1
    with pytest.raises(ToolboxDependencyAnalysisError) as duplicate:
        analyze_toolbox_bundle_imports(
            [
                ToolboxBundleFile(relative_path="pkg/main.py", content=""),
                ToolboxBundleFile(relative_path="PKG\\MAIN.py", content=""),
            ]
        )
    assert duplicate.value.diagnostics[0].code == "duplicate_staged_path"


def test_requirement_resolution_merges_reviewed_and_explicit_constraints() -> None:
    analysis = _analyze("import numpy\n")
    resolved = resolve_toolbox_dependencies(
        analysis,
        package_requirements=("NumPy>=2,<3",),
    )
    requirement = resolved.requirements[0]
    assert requirement.distribution == "numpy"
    assert requirement.import_roots == ("numpy",)
    assert VersionForTest("2.4.3", requirement.constraint)


def VersionForTest(version: str, constraint: str) -> bool:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    return Version(version) in SpecifierSet(constraint)


def test_incompatible_or_unreviewed_explicit_requirements_fail() -> None:
    analysis = _analyze("import numpy\n")
    with pytest.raises(ToolboxDependencyAnalysisError) as conflict:
        resolve_toolbox_dependencies(analysis, package_requirements=("numpy==1.0",))
    assert conflict.value.diagnostics[0].code == "dependency_requirement_conflict"
    with pytest.raises(ToolboxDependencyAnalysisError) as unreviewed:
        resolve_toolbox_dependencies(analysis, package_requirements=("pandas==2.3.0",))
    assert unreviewed.value.diagnostics[0].code == "dependency_package_unreviewed"


def test_empty_range_constraints_conflict_without_rejecting_narrow_valid_ranges() -> None:
    catalog = ReviewedImportDistributionCatalog(
        (
            ReviewedImportDistributionRule(
                distribution="demo",
                import_roots=("demo",),
                version_constraint=">1,<1.0.1",
            ),
        )
    )
    analysis = analyze_toolbox_bundle_imports(
        [ToolboxBundleFile(relative_path="demo_tool.py", content="import demo\n")],
        catalog=catalog,
    )
    # PEP 440 admits versions such as 1.0.0.1 in this narrow range.
    assert resolve_toolbox_dependencies(analysis, catalog=catalog).requirements
    conflict_catalog = ReviewedImportDistributionCatalog(
        (
            ReviewedImportDistributionRule(
                distribution="demo",
                import_roots=("demo",),
                version_constraint=">=2,<2",
            ),
        )
    )
    conflict_analysis = analyze_toolbox_bundle_imports(
        [ToolboxBundleFile(relative_path="demo_tool.py", content="import demo\n")],
        catalog=conflict_catalog,
    )
    with pytest.raises(ToolboxDependencyAnalysisError) as conflict:
        resolve_toolbox_dependencies(conflict_analysis, catalog=conflict_catalog)
    assert conflict.value.diagnostics[0].code == "dependency_requirement_conflict"


def test_catalog_import_distribution_alias_and_reviewed_extra_resolve() -> None:
    catalog = ReviewedImportDistributionCatalog(
        (
            ReviewedImportDistributionRule(
                distribution="Pillow",
                import_roots=("PIL",),
                package_aliases=("pillow-simd",),
                extras=("image",),
                version_constraint=">=10,<12",
            ),
        )
    )
    analysis = analyze_toolbox_bundle_imports(
        [ToolboxBundleFile(relative_path="image_tool.py", content="from PIL import Image\n")],
        catalog=catalog,
    )
    resolved = resolve_toolbox_dependencies(
        analysis,
        package_requirements=("pillow_simd[image]>=10.2",),
        catalog=catalog,
    )
    assert resolved.requirements[0].distribution == "pillow"
    assert resolved.requirements[0].extras == ("image",)
    assert resolved.requirements[0].import_roots == ("PIL",)


def test_smallest_template_is_selected_for_empty_and_compute_requirements() -> None:
    templates = _templates()
    standard = resolve_toolbox_dependencies(_analyze("import json\n"))
    selected = select_toolbox_environment_template(
        standard, templates, python_abi="cp312", platform="win_amd64"
    )
    assert selected.mode == "template"
    assert selected.template.template_id == "core"
    compute = resolve_toolbox_dependencies(_analyze("import numpy\nimport sympy\n"))
    selected = select_toolbox_environment_template(
        compute, reversed(templates), python_abi="cp312", platform="win_amd64"
    )
    assert selected.mode == "template"
    assert selected.template.template_id == "py-compute"


def test_custom_selection_minimizes_exact_delta_deterministically() -> None:
    dependencies = resolve_toolbox_dependencies(
        _analyze("import numpy\nimport matplotlib.pyplot as plt\n")
    )
    left = select_toolbox_environment_template(
        dependencies, _templates(), python_abi="cp312", platform="win_amd64"
    )
    right = select_toolbox_environment_template(
        dependencies, reversed(_templates()), python_abi="cp312", platform="win_amd64"
    )
    assert left == right
    assert left.mode == "custom"
    assert left.template.template_id == "py-compute"
    assert [item.distribution for item in left.custom_delta] == ["matplotlib"]


def test_selection_enforces_target_and_allowed_template_ids() -> None:
    dependencies = resolve_toolbox_dependencies(_analyze("import numpy\n"))
    with pytest.raises(ValueError, match="template_target_unavailable"):
        select_toolbox_environment_template(
            dependencies,
            _templates(),
            python_abi="cp313",
            platform="win_amd64",
        )
    selected = select_toolbox_environment_template(
        dependencies,
        _templates(),
        python_abi="cp312",
        platform="win_amd64",
        allowed_template_ids=("core",),
    )
    assert selected.mode == "custom"
    assert selected.template.template_id == "core"
    assert [item.distribution for item in selected.custom_delta] == ["numpy"]


def test_analysis_and_resolution_are_order_deterministic() -> None:
    files = [
        ToolboxBundleFile(relative_path="b.py", content="import sympy\n"),
        ToolboxBundleFile(relative_path="a.py", content="import numpy\n"),
    ]
    left = analyze_toolbox_bundle_imports(files, declared_imports=("requests",))
    right = analyze_toolbox_bundle_imports(
        list(reversed(copy.deepcopy(files))), declared_imports=("requests",)
    )
    assert left == right
    assert resolve_toolbox_dependencies(left) == resolve_toolbox_dependencies(right)
