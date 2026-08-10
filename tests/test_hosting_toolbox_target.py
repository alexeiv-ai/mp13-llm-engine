from __future__ import annotations

import platform
import sys

import pytest
from packaging.tags import Tag

from hosting.toolbox.target import (
    SUPPORTED_TARGET_PLATFORMS,
    detect_current_toolbox_target,
    toolbox_target_identity,
    validate_target_name,
    wheel_is_compatible,
)


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Windows", "AMD64", "win_amd64"),
        ("Windows", "ARM64", "win_arm64"),
        ("Linux", "x86_64", "manylinux_2_28_x86_64"),
        ("Linux", "aarch64", "manylinux_2_28_aarch64"),
        ("Darwin", "arm64", "macosx_11_0_arm64"),
    ],
)
def test_all_declared_target_families_are_canonical(
    system: str, machine: str, expected: str
) -> None:
    identity = toolbox_target_identity(
        python_abi="cp312",
        system=system,
        machine=machine,
        compatible_tags=(Tag("cp312", "cp312", expected), Tag("py3", "none", "any")),
    )
    assert identity.platform == expected
    assert identity.name == f"cp312-{expected}"
    assert validate_target_name(identity.name) == identity.name
    assert wheel_is_compatible("native-1.0-cp312-cp312-" + expected + ".whl", identity)
    assert wheel_is_compatible("portable-1.0-py3-none-any.whl", identity)


def test_unsupported_cross_family_and_python_are_rejected() -> None:
    with pytest.raises(ValueError, match="family_unsupported"):
        toolbox_target_identity(
            python_abi="cp312",
            system="Darwin",
            machine="x86_64",
            compatible_tags=("py3-none-any",),
        )
    with pytest.raises(ValueError, match="python_abi_unsupported"):
        toolbox_target_identity(
            python_abi="cp313",
            system="Windows",
            machine="AMD64",
            compatible_tags=("py3-none-any",),
        )


def test_current_native_detector_matches_running_machine_and_imports_native_module() -> None:
    target = detect_current_toolbox_target()
    assert target.python_abi == f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert target.platform in SUPPORTED_TARGET_PLATFORMS
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        assert target.architecture == "x64"
    else:
        assert machine in {"arm64", "aarch64"}
        assert target.architecture == "arm64"
    import _ssl

    assert _ssl.OPENSSL_VERSION


def test_wheel_compatibility_uses_ordered_sys_tag_set() -> None:
    target = detect_current_toolbox_target()
    first = target.compatible_tags[0].split("-", 2)
    assert wheel_is_compatible(f"native-1.0-{first[0]}-{first[1]}-{first[2]}.whl", target)
    incompatible = "win_arm64" if target.platform != "win_arm64" else "win_amd64"
    assert not wheel_is_compatible(f"foreign-1.0-cp312-cp312-{incompatible}.whl", target)
