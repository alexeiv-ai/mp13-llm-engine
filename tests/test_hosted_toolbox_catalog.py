from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from hosting.toolbox.catalog import (
    PHASE0_REVIEWED_IMPORT_CATALOG,
    ReviewedImportDistributionCatalog,
    ReviewedImportDistributionRule,
    ToolboxEnvironmentTemplateSpec,
    ToolboxLockedDistributionSpec,
    ToolboxTemplateProvenance,
    normalize_distribution_name,
    normalize_version_constraint,
)


def _digest(char: str) -> str:
    return f"sha256:{char * 64}"


def _template_payload() -> dict:
    return {
        "template_id": "py-compute",
        "python_requires": ">=3.12,<3.13",
        "python_abis": ["cp312"],
        "runtime_kind": "toolbox_python",
        "worker_protocol_version": "1.0",
        "platforms": ["win_amd64", "manylinux_2_28_x86_64"],
        "locked_distributions": [
            {"name": "SymPy", "version": "1.14.0", "extras": []},
            {"name": "numpy", "version": "2.4.3", "extras": []},
            {"name": "numexpr", "version": "2.14.1", "extras": []},
        ],
        "exposed_import_roots": ["sympy", "numpy", "numexpr"],
        "lock_digest": _digest("a"),
        "parent_worker_artifact_digest": _digest("b"),
        "isolation_policy_version": "1.0",
        "provenance": {
            "source": "shipped-catalog",
            "revision": "2026.08.08",
            "manifest_digest": _digest("c"),
            "signing_key_id": "release-key-1",
        },
    }


def test_template_is_strict_immutable_and_canonical() -> None:
    template = ToolboxEnvironmentTemplateSpec.from_dict(_template_payload())
    assert [item["name"] for item in template.to_dict()["locked_distributions"]] == [
        "numexpr",
        "numpy",
        "sympy",
    ]
    assert template.to_dict()["platforms"] == ["manylinux_2_28_x86_64", "win_amd64"]
    assert template.python_requires == "<3.13,>=3.12"
    assert ToolboxEnvironmentTemplateSpec.from_dict(template.to_dict()) == template
    with pytest.raises(FrozenInstanceError):
        template.template_id = "core"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.update({"unknown": True}), "unknown_fields"),
        (lambda row: row.pop("lock_digest"), "missing_fields"),
        (lambda row: row.update({"template_id": "py_compute"}), "template_id_invalid"),
        (lambda row: row.update({"runtime_kind": "model"}), "runtime_kind_invalid"),
        (lambda row: row.update({"platforms": ["macos_arm64"]}), "platforms_invalid"),
        (lambda row: row.update({"python_abis": "cp312"}), "must_be_array"),
        (lambda row: row.update({"worker_protocol_version": 1}), "must_be_string"),
        (lambda row: row.update({"lock_digest": "abc"}), "lock_digest_invalid"),
    ],
)
def test_template_rejects_invalid_shapes(mutation, match: str) -> None:
    payload = _template_payload()
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        ToolboxEnvironmentTemplateSpec.from_dict(payload)


def test_template_rejects_duplicate_distributions_and_roots() -> None:
    payload = _template_payload()
    payload["locked_distributions"].append(
        {"name": "numpy", "version": "2.4.3", "extras": []}
    )
    with pytest.raises(ValueError, match="locked_distribution_duplicate"):
        ToolboxEnvironmentTemplateSpec.from_dict(payload)
    payload = _template_payload()
    payload["exposed_import_roots"].append("numpy")
    with pytest.raises(ValueError, match="import_roots_duplicate"):
        ToolboxEnvironmentTemplateSpec.from_dict(payload)


def test_distribution_and_constraint_normalization() -> None:
    assert normalize_distribution_name("zope.interface") == "zope-interface"
    assert normalize_distribution_name("Pillow_SIMD") == "pillow-simd"
    assert normalize_version_constraint(">=10, <12,>=10") == "<12,>=10"
    with pytest.raises(ValueError, match="version_constraint_invalid"):
        normalize_version_constraint("latest")
    with pytest.raises(ValueError, match="distribution_name_invalid"):
        normalize_distribution_name("https://example.invalid/pkg.whl")
    with pytest.raises(ValueError, match="must_be_string"):
        normalize_distribution_name(123)


def test_reviewed_catalog_supports_import_and_package_aliases_extras_and_constraints() -> None:
    rule = ReviewedImportDistributionRule(
        distribution="Pillow",
        import_roots=("PIL",),
        package_aliases=("Pillow_SIMD",),
        extras=("image",),
        version_constraint=">=10,<12",
        provenance="review:images-1",
    )
    catalog = ReviewedImportDistributionCatalog((rule,))
    assert catalog.for_import("PIL.Image") is rule
    assert catalog.for_distribution("pillow") is rule
    assert catalog.for_distribution("pillow_simd") is rule
    assert rule.extras == ("image",)
    assert rule.version_constraint == "<12,>=10"
    assert ReviewedImportDistributionCatalog.from_dict(catalog.to_dict()) == catalog


def test_reviewed_catalog_rejects_ambiguous_roots_and_package_aliases() -> None:
    left = ReviewedImportDistributionRule(
        distribution="left",
        import_roots=("shared",),
        package_aliases=("common-package",),
        version_constraint="==1",
    )
    right_root = ReviewedImportDistributionRule(
        distribution="right",
        import_roots=("shared",),
        version_constraint="==1",
    )
    with pytest.raises(ValueError, match="import_root_ambiguous:shared"):
        ReviewedImportDistributionCatalog((left, right_root))
    right_alias = ReviewedImportDistributionRule(
        distribution="right",
        import_roots=("other",),
        package_aliases=("common_package",),
        version_constraint="==1",
    )
    with pytest.raises(ValueError, match="package_alias_ambiguous:common-package"):
        ReviewedImportDistributionCatalog((left, right_alias))


def test_phase0_seed_contains_only_observed_third_party_imports() -> None:
    expected = {
        "numpy": ("numpy", ">=1.26.0"),
        "sympy": ("sympy", "==1.14.0"),
        "numexpr": ("numexpr", ">=2.11.0"),
        "requests": ("requests", ">=2.26.0"),
        "matplotlib": ("matplotlib", "<4,>=3.10.1"),
    }
    assert {
        rule.distribution: (rule.import_roots[0], rule.version_constraint)
        for rule in PHASE0_REVIEWED_IMPORT_CATALOG.rules
    } == expected
    assert PHASE0_REVIEWED_IMPORT_CATALOG.for_import("json") is None
    assert PHASE0_REVIEWED_IMPORT_CATALOG.for_import("torch") is None


def test_catalog_and_nested_rows_reject_unknown_or_missing_fields() -> None:
    payload = PHASE0_REVIEWED_IMPORT_CATALOG.to_dict()
    payload["extra"] = True
    with pytest.raises(ValueError, match="unknown_fields"):
        ReviewedImportDistributionCatalog.from_dict(payload)
    payload = PHASE0_REVIEWED_IMPORT_CATALOG.to_dict()
    payload["rules"][0].pop("extras")
    with pytest.raises(ValueError, match="missing_fields"):
        ReviewedImportDistributionCatalog.from_dict(payload)


def test_locked_distribution_rejects_non_exact_versions_and_duplicate_extras() -> None:
    with pytest.raises(ValueError, match="version_invalid"):
        ToolboxLockedDistributionSpec(name="numpy", version=">=2")
    with pytest.raises(ValueError, match="extras_duplicate"):
        ToolboxLockedDistributionSpec(name="demo", version="1.0", extras=("x", "X"))
