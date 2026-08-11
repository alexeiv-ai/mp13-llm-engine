"""Toolbox configuration transport for production daemon launchers."""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .paths import _default_state_dir
from .security import _atomic_write_secure_json


TOOLBOX_LAUNCH_CONFIGURATION_FIELDS = (
    "toolbox_host_project_configuration",
    "toolbox_artifact_sources",
    "toolbox_trust_public_keys",
    "toolbox_source_credentials",
    "toolbox_dependency_policy",
)


def toolbox_launch_configuration(
    *,
    toolbox_host_project_configuration: Optional[Mapping[str, Any]] = None,
    toolbox_artifact_sources: Optional[Mapping[str, Path]] = None,
    toolbox_trust_public_keys: Optional[Mapping[str, str]] = None,
    toolbox_source_credentials: Optional[Mapping[str, str]] = None,
    toolbox_dependency_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return only explicitly supplied EngineHostDaemon toolbox inputs."""
    values = {
        "toolbox_host_project_configuration": toolbox_host_project_configuration,
        "toolbox_artifact_sources": toolbox_artifact_sources,
        "toolbox_trust_public_keys": toolbox_trust_public_keys,
        "toolbox_source_credentials": toolbox_source_credentials,
        "toolbox_dependency_policy": toolbox_dependency_policy,
    }
    return {
        key: dict(value)
        for key, value in values.items()
        if value is not None
    }


def load_toolbox_launch_configuration(path: Path) -> Dict[str, Any]:
    """Load the strict launcher file without projecting secret values."""
    import json

    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("toolbox_launch_configuration_unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("toolbox_launch_configuration_must_be_object")
    unknown = sorted(set(payload) - set(TOOLBOX_LAUNCH_CONFIGURATION_FIELDS))
    if unknown:
        raise ValueError("toolbox_launch_configuration_unknown_fields")
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"{key}_must_be_object")
        if key == "toolbox_artifact_sources":
            out[key] = {str(source_id): Path(source_path) for source_id, source_path in value.items()}
        else:
            out[key] = dict(value)
    return out


def resolve_toolbox_launch_configuration(
    *,
    toolbox_config_file: Optional[Path] = None,
    toolbox_host_project_configuration: Optional[Mapping[str, Any]] = None,
    toolbox_artifact_sources: Optional[Mapping[str, Path]] = None,
    toolbox_trust_public_keys: Optional[Mapping[str, str]] = None,
    toolbox_source_credentials: Optional[Mapping[str, str]] = None,
    toolbox_dependency_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve either a launcher file or direct inputs, never both."""
    direct = toolbox_launch_configuration(
        toolbox_host_project_configuration=toolbox_host_project_configuration,
        toolbox_artifact_sources=toolbox_artifact_sources,
        toolbox_trust_public_keys=toolbox_trust_public_keys,
        toolbox_source_credentials=toolbox_source_credentials,
        toolbox_dependency_policy=toolbox_dependency_policy,
    )
    if toolbox_config_file is not None and direct:
        raise ValueError("toolbox_launch_configuration_inputs_conflict")
    if toolbox_config_file is not None:
        return load_toolbox_launch_configuration(toolbox_config_file)
    return direct


def write_ephemeral_toolbox_launch_configuration(configuration: Mapping[str, Any]) -> Path:
    """Write a short-lived, ACL-hardened file for a detached child process."""
    launch_root = _default_state_dir() / "launch"
    path = launch_root / f"toolbox-{os.getpid()}-{secrets.token_hex(12)}.json"
    payload: Dict[str, Any] = {}
    for key, value in dict(configuration).items():
        if key not in TOOLBOX_LAUNCH_CONFIGURATION_FIELDS or not isinstance(value, Mapping):
            raise ValueError("toolbox_launch_configuration_invalid")
        payload[key] = (
            {
                str(source_id): str(Path(source_path).expanduser().resolve())
                for source_id, source_path in value.items()
            }
            if key == "toolbox_artifact_sources"
            else dict(value)
        )
    _atomic_write_secure_json(path, payload)
    return path


__all__ = [
    "TOOLBOX_LAUNCH_CONFIGURATION_FIELDS",
    "load_toolbox_launch_configuration",
    "resolve_toolbox_launch_configuration",
    "toolbox_launch_configuration",
    "write_ephemeral_toolbox_launch_configuration",
]
