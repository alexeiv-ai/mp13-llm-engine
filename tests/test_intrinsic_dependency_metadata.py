from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mp13_engine.mp13_intrinsics_metadata import (
    INTRINSICS_METADATA_REGISTRY,
    intrinsic_dependency_metadata,
    intrinsic_dependency_profile_id,
    intrinsic_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
METADATA_SOURCE = ROOT / "src" / "mp13_engine" / "mp13_intrinsics_metadata.py"
BUNDLE_MODELS_SOURCE = ROOT / "src" / "hosting" / "toolbox" / "bundle_models.py"


def test_registry_has_exact_intrinsic_dependencies_including_shared_module_imports() -> None:
    assert set(INTRINSICS_METADATA_REGISTRY) == {
        "scriptable_calculator",
        "symbolic_algebra",
    }
    calculator = INTRINSICS_METADATA_REGISTRY["scriptable_calculator"]
    assert calculator.import_roots == ("numexpr", "numpy", "sympy")
    assert calculator.package_requirements == (
        "numexpr==2.14.1",
        "numpy==2.4.3",
        "sympy==1.14.0",
    )
    symbolic = INTRINSICS_METADATA_REGISTRY["symbolic_algebra"]
    assert symbolic.import_roots == ("numpy", "sympy")
    assert symbolic.package_requirements == ("numpy==2.4.3", "sympy==1.14.0")


def test_guides_resolve_to_parent_and_aggregation_is_deterministic() -> None:
    assert intrinsic_metadata("symbolic_algebra_guide").name == "symbolic_algebra"
    left = intrinsic_dependency_metadata(
        ["symbolic_algebra_guide", "scriptable_calculator", "symbolic_algebra"]
    )
    right = intrinsic_dependency_metadata(
        ["scriptable_calculator_guide", "symbolic_algebra"]
    )
    assert left == right
    assert left == {
        "intrinsics": ["scriptable_calculator", "symbolic_algebra"],
        "import_roots": ["numexpr", "numpy", "sympy"],
        "package_requirements": [
            "numexpr==2.14.1",
            "numpy==2.4.3",
            "sympy==1.14.0",
        ],
    }
    assert intrinsic_dependency_profile_id(left["intrinsics"]).startswith("intrinsics-")
    assert intrinsic_dependency_profile_id(left["intrinsics"]) == intrinsic_dependency_profile_id(
        reversed(left["intrinsics"])
    )
    assert intrinsic_dependency_profile_id([]) == "none"


def test_unknown_intrinsic_is_rejected() -> None:
    with pytest.raises(ValueError, match="intrinsic_unknown:not_registered"):
        intrinsic_dependency_metadata(["not_registered"])


def test_metadata_module_has_only_standard_library_imports() -> None:
    tree = ast.parse(METADATA_SOURCE.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert imported <= {"__future__", "dataclasses", "hashlib", "json", "typing"}


def test_intrinsic_discovery_does_not_load_implementation_or_math_packages() -> None:
    code = """
import json
import sys
sys.path.insert(0, 'src')
import mp13_engine.mp13_toolbox as toolbox
toolbox._get_tools_builtin_module = lambda: (_ for _ in ()).throw(AssertionError('implementation_loaded'))
items = toolbox.Toolbox().available_intrinsics(include_guides=True)
print(json.dumps({'names': [item['name'] for item in items], 'loaded': [name for name in ('mp13_engine.mp13_tools_builtin', 'numpy', 'sympy', 'numexpr') if name in sys.modules]}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["names"] == [
        "scriptable_calculator",
        "scriptable_calculator_guide",
        "symbolic_algebra",
        "symbolic_algebra_guide",
    ]
    assert result["loaded"] == []


def test_sandbox_profile_no_longer_owns_intrinsic_dependency_branching() -> None:
    source = BUNDLE_MODELS_SOURCE.read_text(encoding="utf-8")
    assert "def intrinsics_profile_id" not in source
    assert "scriptable_calculator" not in source
    assert "symbolic_algebra" not in source


def test_sympy_is_an_exact_direct_project_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "poetry.lock").read_text(encoding="utf-8")
    assert 'sympy = "==1.14.0"' in pyproject
    assert 'name = "sympy"' in lock
    assert 'version = "1.14.0"' in lock
