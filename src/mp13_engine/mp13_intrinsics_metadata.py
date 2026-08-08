"""Import-safe intrinsic discovery and dependency metadata.

This module intentionally imports only the Python standard library. Registry
inspection must not import intrinsic implementations or their optional packages.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class IntrinsicRegistryMetadata:
    name: str
    guide_name: str
    description: str
    guide_description: str
    import_roots: tuple[str, ...]
    package_requirements: tuple[str, ...]
    implementation_module: str = "mp13_engine.mp13_tools_builtin"

    def to_dependency_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "import_roots": list(self.import_roots),
            "package_requirements": list(self.package_requirements),
            "implementation_module": self.implementation_module,
        }


INTRINSICS_METADATA_REGISTRY: dict[str, IntrinsicRegistryMetadata] = {
    "scriptable_calculator": IntrinsicRegistryMetadata(
        name="scriptable_calculator",
        guide_name="scriptable_calculator_guide",
        description=(
            "A scriptable calculator that evaluates mathematical expressions and "
            "assignments. It uses the NumExpr library for safe, fast numerical "
            "computation. Ideal for multi-step calculations where intermediate "
            "results are stored in variables."
        ),
        guide_description=(
            "Provides detailed guidance on using the scriptable_calculator tool. "
            "Use topic='help' to see all topics."
        ),
        # The implementation currently shares a module with symbolic_algebra;
        # all unguarded module imports therefore belong to this load boundary.
        import_roots=("numexpr", "numpy", "sympy"),
        package_requirements=("numexpr==2.14.1", "numpy==2.4.3", "sympy==1.14.0"),
    ),
    "symbolic_algebra": IntrinsicRegistryMetadata(
        name="symbolic_algebra",
        guide_name="symbolic_algebra_guide",
        description=(
            "Performs symbolic algebraic manipulations on mathematical expressions, "
            "such as simplifying, expanding, factoring, solving equations, and "
            "calculus operations (differentiation, integration). This tool works "
            "with symbols, not numerical values."
        ),
        guide_description=(
            "Provides detailed guidance on using the symbolic_algebra tool. Use "
            "topic='help' to see all topics."
        ),
        import_roots=("numpy", "sympy"),
        package_requirements=("numpy==2.4.3", "sympy==1.14.0"),
    ),
}


def intrinsic_metadata(name: Any) -> IntrinsicRegistryMetadata:
    requested = str(name or "").strip()
    for metadata in INTRINSICS_METADATA_REGISTRY.values():
        if requested in {metadata.name, metadata.guide_name}:
            return metadata
    raise ValueError(f"intrinsic_unknown:{requested}")


def intrinsic_dependency_metadata(names: Iterable[Any]) -> dict[str, Any]:
    selected: dict[str, IntrinsicRegistryMetadata] = {}
    for raw_name in names:
        metadata = intrinsic_metadata(raw_name)
        selected[metadata.name] = metadata
    roots = sorted({root for metadata in selected.values() for root in metadata.import_roots})
    requirements = sorted(
        {
            requirement
            for metadata in selected.values()
            for requirement in metadata.package_requirements
        }
    )
    return {
        "intrinsics": sorted(selected),
        "import_roots": roots,
        "package_requirements": requirements,
    }


def intrinsic_dependency_profile_id(names: Iterable[Any]) -> str:
    dependencies = intrinsic_dependency_metadata(names)
    if not dependencies["intrinsics"]:
        return "none"
    encoded = json.dumps(
        dependencies,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"intrinsics-{hashlib.sha256(encoded).hexdigest()[:16]}"


__all__ = [
    "INTRINSICS_METADATA_REGISTRY",
    "IntrinsicRegistryMetadata",
    "intrinsic_dependency_metadata",
    "intrinsic_dependency_profile_id",
    "intrinsic_metadata",
]
