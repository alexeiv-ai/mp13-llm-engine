from __future__ import annotations

import pytest

from hosting.toolbox.bundle_models import ToolboxBundleFile
from hosting.toolbox.catalog import ToolboxEnvironmentTemplateSpec
from hosting.toolbox.dependency_analysis import (
    ToolboxResolvedDependencies,
    ToolboxResolvedRequirement,
    analyze_toolbox_bundle_imports,
    resolve_toolbox_dependencies,
    select_toolbox_environment_template,
)
from hosting.toolbox.dependency_policy import (
    ToolboxDependencyPolicy,
    ToolboxDependencyPolicyError,
    normalize_https_origin,
    validate_toolbox_dependency_policy,
)


def _digest(char: str) -> str:
    return f"sha256:{char * 64}"


def _template(
    template_id: str,
    distributions: list[tuple[str, str]],
    roots: list[str],
) -> ToolboxEnvironmentTemplateSpec:
    return ToolboxEnvironmentTemplateSpec.from_dict(
        {
            "template_id": template_id,
            "python_requires": ">=3.12,<3.13",
            "python_abis": ["cp312"],
            "runtime_kind": "toolbox_python",
            "worker_protocol_version": "1.0",
            "platforms": ["win_amd64"],
            "locked_distributions": [
                {"name": name, "version": version, "extras": []}
                for name, version in distributions
            ],
            "exposed_import_roots": roots,
            "lock_digest": _digest("a" if template_id == "core" else "b"),
            "parent_worker_artifact_digest": _digest("c"),
            "isolation_policy_version": "1.0",
            "provenance": {
                "source": "test",
                "revision": "1",
                "evidence_digest": _digest("d"),
                "verifier_id": "key-1",
            },
        }
    )


def _templates() -> tuple[ToolboxEnvironmentTemplateSpec, ...]:
    return (
        _template("core", [("hosting-runtime", "1.0")], ["hosting", "mp13_engine"]),
        _template(
            "py-compute",
            [
                ("hosting-runtime", "1.0"),
                ("numpy", "2.4.3"),
                ("sympy", "1.14.0"),
                ("numexpr", "2.14.1"),
            ],
            ["hosting", "mp13_engine", "numpy", "sympy", "numexpr"],
        ),
    )


def _dependencies(source: str) -> ToolboxResolvedDependencies:
    analysis = analyze_toolbox_bundle_imports(
        [ToolboxBundleFile(relative_path="tool.py", content=source)]
    )
    return resolve_toolbox_dependencies(analysis)


def _selection(source: str):
    dependencies = _dependencies(source)
    selection = select_toolbox_environment_template(
        dependencies,
        _templates(),
        python_abi="cp312",
        platform="win_amd64",
    )
    return dependencies, selection


def _policy_payload() -> dict:
    return {
        "revision": _digest("e"),
        "allowed_template_ids": ["core", "py-compute"],
        "allowed_targets": ["cp312-win_amd64"],
        "package_allowlist": ["numpy", "sympy", "numexpr", "matplotlib", "requests"],
        "package_denylist": [],
        "allow_custom": True,
        "custom_requires_approval": True,
        "online_resolution_allowed": False,
        "allowed_index_origins": ["https://packages.example.test"],
    }


def _policy(**changes) -> ToolboxDependencyPolicy:
    payload = _policy_payload()
    payload.update(changes)
    return ToolboxDependencyPolicy.from_dict(payload)


def test_policy_is_strict_immutable_and_canonical() -> None:
    policy = _policy(
        package_allowlist=["SymPy", "NumPy"],
        package_denylist=["Requests"],
        allowed_index_origins=["https://PACKAGES.example.test:443/"],
    )
    assert policy.package_allowlist == ("numpy", "sympy")
    assert policy.package_denylist == ("requests",)
    assert policy.allowed_index_origins == ("https://packages.example.test",)
    assert ToolboxDependencyPolicy.from_dict(policy.to_dict()) == policy
    payload = _policy_payload()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown_fields"):
        ToolboxDependencyPolicy.from_dict(payload)
    payload = _policy_payload()
    payload["allowed_targets"] = "cp312-win_amd64"
    with pytest.raises(ValueError, match="must_be_array"):
        ToolboxDependencyPolicy.from_dict(payload)
    payload = _policy_payload()
    payload["allow_custom"] = 1
    with pytest.raises(ValueError, match="must_be_boolean"):
        ToolboxDependencyPolicy.from_dict(payload)


def test_template_and_target_are_both_authoritative() -> None:
    dependencies, selection = _selection("import numpy\n")
    with pytest.raises(ToolboxDependencyPolicyError) as target:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(),
            python_abi="cp312",
            platform="manylinux_2_28_x86_64",
        )
    assert target.value.code == "dependency_target_denied"
    with pytest.raises(ToolboxDependencyPolicyError) as template:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(allowed_template_ids=["core"]),
            python_abi="cp312",
            platform="win_amd64",
        )
    assert template.value.code == "dependency_template_denied"
    with pytest.raises(ToolboxDependencyPolicyError) as requested:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(),
            python_abi="cp312",
            platform="win_amd64",
            requested_template_id="core",
        )
    assert requested.value.code == "dependency_template_denied"


def test_package_deny_precedes_allow_and_allowlist_is_enforced() -> None:
    dependencies, selection = _selection("import numpy\n")
    with pytest.raises(ToolboxDependencyPolicyError) as denied:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(package_allowlist=["numpy"], package_denylist=["NumPy"]),
            python_abi="cp312",
            platform="win_amd64",
        )
    assert denied.value.code == "dependency_package_denied"
    with pytest.raises(ToolboxDependencyPolicyError) as absent:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(package_allowlist=["sympy"]),
            python_abi="cp312",
            platform="win_amd64",
        )
    assert absent.value.code == "dependency_package_not_allowed"


def test_custom_delta_requires_policy_and_returns_approval_decision() -> None:
    dependencies, selection = _selection("import numpy\nimport matplotlib\n")
    assert selection.mode == "custom"
    with pytest.raises(ToolboxDependencyPolicyError) as denied:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(allow_custom=False),
            python_abi="cp312",
            platform="win_amd64",
        )
    assert denied.value.code == "dependency_custom_denied"
    decision = validate_toolbox_dependency_policy(
        selection,
        dependencies,
        _policy(),
        python_abi="cp312",
        platform="win_amd64",
    )
    assert decision.approval_required is True
    assert decision.package_distributions == ("matplotlib", "numpy")


def test_https_index_origins_are_normalized_and_online_policy_is_enforced() -> None:
    assert normalize_https_origin("https://PYPI.Example.test:443/") == "https://pypi.example.test"
    for invalid in [
        "http://pypi.example.test",
        "https://user:p@example.test",
        "https://pypi.example.test/simple",
        "https://pypi.example.test?token=x",
    ]:
        with pytest.raises(ValueError, match="package_index_origin_invalid"):
            normalize_https_origin(invalid)
    dependencies, selection = _selection("import numpy\nimport matplotlib\n")
    with pytest.raises(ToolboxDependencyPolicyError) as offline:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(),
            python_abi="cp312",
            platform="win_amd64",
            requested_index_origins=("https://packages.example.test",),
        )
    assert offline.value.code == "dependency_online_resolution_denied"
    decision = validate_toolbox_dependency_policy(
        selection,
        dependencies,
        _policy(online_resolution_allowed=True),
        python_abi="cp312",
        platform="win_amd64",
        requested_index_origins=("https://PACKAGES.example.test:443/",),
    )
    assert decision.index_origins == ("https://packages.example.test",)
    with pytest.raises(ToolboxDependencyPolicyError) as origin:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(online_resolution_allowed=True),
            python_abi="cp312",
            platform="win_amd64",
            requested_index_origins=("https://other.example.test",),
        )
    assert origin.value.code == "dependency_index_denied"


def test_intrinsic_requirements_must_be_complete_and_pin_compatible() -> None:
    dependencies, selection = _selection("import numpy\nimport sympy\n")
    decision = validate_toolbox_dependency_policy(
        selection,
        dependencies,
        _policy(),
        python_abi="cp312",
        platform="win_amd64",
        intrinsic_names=("symbolic_algebra_guide",),
    )
    assert decision.approval_required is False
    empty_dependencies, empty_selection = _selection("import json\n")
    with pytest.raises(ToolboxDependencyPolicyError) as missing:
        validate_toolbox_dependency_policy(
            empty_selection,
            empty_dependencies,
            _policy(),
            python_abi="cp312",
            platform="win_amd64",
            intrinsic_names=("symbolic_algebra",),
        )
    assert missing.value.code == "dependency_intrinsic_requirement_missing"

    incompatible = ToolboxResolvedDependencies(
        requirements=(
            ToolboxResolvedRequirement(
                distribution="numpy",
                extras=(),
                constraint="==2.4.3",
                import_roots=("numpy",),
            ),
            ToolboxResolvedRequirement(
                distribution="sympy",
                extras=(),
                constraint="<1.14",
                import_roots=("sympy",),
            ),
        ),
        analysis=dependencies.analysis,
    )
    with pytest.raises(ToolboxDependencyPolicyError) as conflict:
        validate_toolbox_dependency_policy(
            selection,
            incompatible,
            _policy(),
            python_abi="cp312",
            platform="win_amd64",
            intrinsic_names=("symbolic_algebra",),
        )
    assert conflict.value.code == "dependency_intrinsic_requirement_conflict"

    missing_root = ToolboxResolvedDependencies(
        requirements=(
            ToolboxResolvedRequirement(
                distribution="numpy",
                extras=(),
                constraint="==2.4.3",
                import_roots=("numpy",),
            ),
            ToolboxResolvedRequirement(
                distribution="sympy",
                extras=(),
                constraint="==1.14.0",
                import_roots=(),
            ),
        ),
        analysis=dependencies.analysis,
    )
    with pytest.raises(ToolboxDependencyPolicyError) as root:
        validate_toolbox_dependency_policy(
            selection,
            missing_root,
            _policy(),
            python_abi="cp312",
            platform="win_amd64",
            intrinsic_names=("symbolic_algebra",),
        )
    assert root.value.code == "dependency_intrinsic_import_missing"


@pytest.mark.parametrize(
    "payload",
    [
        {"allow_resolution": True},
        {"metadata": {"approved": True}},
        {"package": {"sandbox": {"network": True}}},
        {"capabilities": ["filesystem"]},
        {"dependency_approval_ref": "fabricated"},
    ],
)
def test_dependency_payload_cannot_assert_approval_or_sandbox_capability(payload: dict) -> None:
    dependencies, selection = _selection("import numpy\n")
    with pytest.raises(ToolboxDependencyPolicyError) as caught:
        validate_toolbox_dependency_policy(
            selection,
            dependencies,
            _policy(),
            python_abi="cp312",
            platform="win_amd64",
            dependency_payload=payload,
        )
    assert caught.value.code == "dependency_payload_authority_forbidden"
