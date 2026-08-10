"""Bounded exclusive model-runtime status and generic-selection guard."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .toolbox.identity import require_digest
from .toolbox.target import SUPPORTED_PYTHON_ABI, validate_target_platform


MODEL_RUNTIME_STATUS_FIELDS = frozenset(
    {
        "state",
        "code",
        "summary",
        "python_abi",
        "platform",
        "engine_artifact_digest",
        "complete_lock_digest",
        "optional_package_set",
        "materialization_revision",
        "updated_at_ms",
    }
)
_MODEL_RUNTIME_ALIASES = frozenset(
    {"model", "model-env", "model-environment", "model-runtime", "model_runtime", "local-model"}
)
_DIRECT_SELECTOR_KEYS = frozenset(
    {
        "model_environment",
        "model_environment_id",
        "model_python_executable",
        "model_runtime",
        "model_runtime_id",
        "model_runtime_ref",
    }
)
_VALUE_SELECTOR_KEYS = frozenset(
    {
        "base_environment",
        "base_template",
        "environment_name",
        "profile",
        "python_executable",
        "runtime",
        "runtime_kind",
        "template_id",
        "worker_profile_class",
    }
)


def _text(value: Any, *, label: str, maximum: int = 512, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label}_must_be_string")
    text = value.strip()
    if (not text and not optional) or len(text.encode("utf-8")) > maximum or any(ord(char) < 32 for char in text):
        raise ValueError(f"{label}_invalid")
    return text or None


@dataclass(frozen=True)
class ModelRuntimeIdentity:
    python_abi: str
    platform: str
    engine_artifact_digest: str
    complete_lock_digest: str
    optional_package_set: str | None
    materialization_revision: str
    verified: bool
    updated_at_ms: int

    def __post_init__(self) -> None:
        if self.python_abi != SUPPORTED_PYTHON_ABI:
            raise ValueError("model_runtime_python_abi_invalid")
        validate_target_platform(self.platform, label="model_runtime_platform")
        require_digest(self.engine_artifact_digest, label="model_runtime_engine_artifact_digest")
        require_digest(self.complete_lock_digest, label="model_runtime_complete_lock_digest")
        _text(self.optional_package_set, label="model_runtime_optional_package_set", optional=True)
        _text(self.materialization_revision, label="model_runtime_materialization_revision")
        if not isinstance(self.verified, bool):
            raise ValueError("model_runtime_verified_invalid")
        if isinstance(self.updated_at_ms, bool) or not isinstance(self.updated_at_ms, int) or self.updated_at_ms < 0:
            raise ValueError("model_runtime_updated_at_ms_invalid")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelRuntimeIdentity":
        row = dict(payload or {})
        fields = {
            "python_abi", "platform", "engine_artifact_digest", "complete_lock_digest",
            "optional_package_set", "materialization_revision", "verified", "updated_at_ms",
        }
        if set(row) != fields:
            raise ValueError("model_runtime_identity_fields_invalid")
        return cls(**row)


@dataclass(frozen=True)
class ModelRuntimeStatus:
    state: str
    code: str
    summary: str
    python_abi: str | None
    platform: str | None
    engine_artifact_digest: str | None
    complete_lock_digest: str | None
    optional_package_set: str | None
    materialization_revision: str | None
    updated_at_ms: int

    def __post_init__(self) -> None:
        if self.state not in {"ready", "degraded", "unavailable"}:
            raise ValueError("model_runtime_status_state_invalid")
        _text(self.code, label="model_runtime_status_code", maximum=128)
        _text(self.summary, label="model_runtime_status_summary")
        if self.python_abi is not None and self.python_abi != SUPPORTED_PYTHON_ABI:
            raise ValueError("model_runtime_status_python_abi_invalid")
        if self.platform is not None:
            validate_target_platform(self.platform, label="model_runtime_status_platform")
        for name, value in (
            ("engine_artifact_digest", self.engine_artifact_digest),
            ("complete_lock_digest", self.complete_lock_digest),
        ):
            if value is not None:
                require_digest(value, label=f"model_runtime_status_{name}")
        _text(self.optional_package_set, label="model_runtime_status_optional_package_set", optional=True)
        _text(self.materialization_revision, label="model_runtime_status_materialization_revision", optional=True)
        if isinstance(self.updated_at_ms, bool) or not isinstance(self.updated_at_ms, int) or self.updated_at_ms < 0:
            raise ValueError("model_runtime_status_updated_at_ms_invalid")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "state": self.state,
            "code": self.code,
            "summary": self.summary,
            "python_abi": self.python_abi,
            "platform": self.platform,
            "engine_artifact_digest": self.engine_artifact_digest,
            "complete_lock_digest": self.complete_lock_digest,
            "optional_package_set": self.optional_package_set,
            "materialization_revision": self.materialization_revision,
            "updated_at_ms": self.updated_at_ms,
        }
        if set(value) != MODEL_RUNTIME_STATUS_FIELDS:
            raise AssertionError("model_runtime_status_fields_invalid")
        return value


def _looks_like_model_runtime(value: str) -> bool:
    normalized = value.strip().lower().replace("\\", "/")
    if normalized in _MODEL_RUNTIME_ALIASES:
        return True
    segments = [item for item in re.split(r"[/|:]", normalized) if item]
    return any(item in _MODEL_RUNTIME_ALIASES for item in segments)


def reject_model_runtime_selection(value: Any, *, path: str = "payload") -> None:
    """Reject model-runtime selectors in a generic environment input tree."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key or "").strip().lower()
            if key in _DIRECT_SELECTOR_KEYS:
                raise ValueError(f"model_runtime_selection_denied:{path}.{key}")
            if key in _VALUE_SELECTOR_KEYS and isinstance(nested, str) and _looks_like_model_runtime(nested):
                raise ValueError(f"model_runtime_selection_denied:{path}.{key}")
            reject_model_runtime_selection(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            reject_model_runtime_selection(nested, path=f"{path}[{index}]")


__all__ = [
    "MODEL_RUNTIME_STATUS_FIELDS",
    "ModelRuntimeIdentity",
    "ModelRuntimeStatus",
    "reject_model_runtime_selection",
]
