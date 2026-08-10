"""Canonical current-host target identity for hosted toolbox environments."""
from __future__ import annotations

import platform as platform_module
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename


SUPPORTED_PYTHON_ABI = "cp312"
SUPPORTED_TARGET_PLATFORMS = frozenset(
    {
        "win_amd64",
        "win_arm64",
        "manylinux_2_28_x86_64",
        "manylinux_2_28_aarch64",
        "macosx_11_0_arm64",
    }
)

_TARGET_FAMILIES = {
    ("windows", "x64"): "win_amd64",
    ("windows", "arm64"): "win_arm64",
    ("linux", "x64"): "manylinux_2_28_x86_64",
    ("linux", "arm64"): "manylinux_2_28_aarch64",
    ("macos", "arm64"): "macosx_11_0_arm64",
}
_SYSTEM_NAMES = {"windows": "windows", "linux": "linux", "darwin": "macos", "macos": "macos"}
_MACHINE_NAMES = {
    "amd64": "x64",
    "x86_64": "x64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


@dataclass(frozen=True)
class ToolboxTargetIdentity:
    python_abi: str
    operating_system: str
    architecture: str
    platform: str
    platform_baseline: str
    compatible_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.python_abi != SUPPORTED_PYTHON_ABI:
            raise ValueError("toolbox_target_python_abi_unsupported")
        expected = _TARGET_FAMILIES.get((self.operating_system, self.architecture))
        if expected is None or self.platform != expected or self.platform_baseline != expected:
            raise ValueError("toolbox_target_family_unsupported")
        if not self.compatible_tags or len(set(self.compatible_tags)) != len(self.compatible_tags):
            raise ValueError("toolbox_target_compatible_tags_invalid")

    @property
    def name(self) -> str:
        return f"{self.python_abi}-{self.platform}"

    @property
    def tag_set(self) -> frozenset[Tag]:
        parsed: set[Tag] = set()
        for value in self.compatible_tags:
            try:
                interpreter, abi, platform_name = value.split("-", 2)
            except ValueError as exc:
                raise ValueError("toolbox_target_compatible_tag_invalid") from exc
            parsed.add(Tag(interpreter, abi, platform_name))
        return frozenset(parsed)

    def to_dict(self) -> dict[str, object]:
        return {
            "python_abi": self.python_abi,
            "operating_system": self.operating_system,
            "architecture": self.architecture,
            "platform": self.platform,
            "platform_baseline": self.platform_baseline,
            "compatible_tags": list(self.compatible_tags),
        }


def _normalize_system(value: str) -> str:
    normalized = _SYSTEM_NAMES.get(str(value or "").strip().lower())
    if normalized is None:
        raise ValueError("toolbox_target_operating_system_unsupported")
    return normalized


def _normalize_machine(value: str) -> str:
    normalized = _MACHINE_NAMES.get(str(value or "").strip().lower())
    if normalized is None:
        raise ValueError("toolbox_target_architecture_unsupported")
    return normalized


def toolbox_target_identity(
    *,
    python_abi: str,
    system: str,
    machine: str,
    compatible_tags: Iterable[Tag | str],
) -> ToolboxTargetIdentity:
    operating_system = _normalize_system(system)
    architecture = _normalize_machine(machine)
    platform_name = _TARGET_FAMILIES.get((operating_system, architecture))
    if platform_name is None:
        raise ValueError("toolbox_target_family_unsupported")
    ordered_tags = tuple(dict.fromkeys(str(item) for item in compatible_tags))
    return ToolboxTargetIdentity(
        python_abi=str(python_abi or "").strip().lower(),
        operating_system=operating_system,
        architecture=architecture,
        platform=platform_name,
        platform_baseline=platform_name,
        compatible_tags=ordered_tags,
    )


@lru_cache(maxsize=1)
def detect_current_toolbox_target() -> ToolboxTargetIdentity:
    if sys.implementation.name != "cpython":
        raise ValueError("toolbox_target_interpreter_unsupported")
    python_abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return toolbox_target_identity(
        python_abi=python_abi,
        system=platform_module.system(),
        machine=platform_module.machine(),
        compatible_tags=sys_tags(),
    )


def validate_target_platform(value: str, *, label: str = "toolbox_platform") -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in SUPPORTED_TARGET_PLATFORMS:
        raise ValueError(f"{label}_invalid")
    return normalized


def validate_target_name(value: str, *, label: str = "toolbox_target") -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {f"{SUPPORTED_PYTHON_ABI}-{item}" for item in SUPPORTED_TARGET_PLATFORMS}:
        raise ValueError(f"{label}_invalid")
    return normalized


def wheel_is_compatible(filename: str, target: ToolboxTargetIdentity | None = None) -> bool:
    try:
        _, _, _, wheel_tags = parse_wheel_filename(str(filename or "").strip())
    except InvalidWheelFilename:
        return False
    current = target or detect_current_toolbox_target()
    return bool(set(wheel_tags) & set(current.tag_set))


__all__ = [
    "SUPPORTED_PYTHON_ABI",
    "SUPPORTED_TARGET_PLATFORMS",
    "ToolboxTargetIdentity",
    "detect_current_toolbox_target",
    "toolbox_target_identity",
    "validate_target_name",
    "validate_target_platform",
    "wheel_is_compatible",
]
