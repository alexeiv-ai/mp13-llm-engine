"""Atomic history of strict toolbox host-configuration revisions."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ..toolbox.host_project_config import ToolboxHostProjectConfiguration
from ..toolbox.identity import require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


HOST_CONFIG_STATE_CONTRACT = "hosting.toolbox.host_configuration_state.v1"
MAX_HOST_CONFIG_REVISIONS = 64


class AtomicJsonToolboxHostConfigurationRepository:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.clock = clock

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"contract": HOST_CONFIG_STATE_CONTRACT, "current_revision": None, "revisions": {}}

    @classmethod
    def _validate(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {"contract", "current_revision", "revisions"}:
            raise ValueError("toolbox_host_config_state_fields_invalid")
        if row.get("contract") != HOST_CONFIG_STATE_CONTRACT:
            raise ValueError("toolbox_host_config_state_contract_invalid")
        current = row.get("current_revision")
        if current is not None:
            current = require_digest(current, label="toolbox_host_current_config_revision")
        revisions = row.get("revisions")
        if not isinstance(revisions, dict) or len(revisions) > MAX_HOST_CONFIG_REVISIONS:
            raise ValueError("toolbox_host_config_state_capacity_invalid")
        validated: dict[str, dict[str, Any]] = {}
        for key, value in revisions.items():
            revision = require_digest(key, label="toolbox_host_config_revision")
            item = dict(value or {})
            if set(item) != {"config", "source_set_revision", "recorded_at_ms"}:
                raise ValueError("toolbox_host_config_revision_fields_invalid")
            config = ToolboxHostProjectConfiguration.from_dict(item["config"])
            if revision != config.config_revision:
                raise ValueError("toolbox_host_config_revision_mismatch")
            if require_digest(
                item["source_set_revision"], label="toolbox_host_source_set_revision"
            ) != config.source_set_revision:
                raise ValueError("toolbox_host_source_set_revision_mismatch")
            recorded = item["recorded_at_ms"]
            if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded < 0:
                raise ValueError("toolbox_host_config_recorded_at_invalid")
            validated[revision] = {
                "config": config.to_dict(),
                "source_set_revision": config.source_set_revision,
                "recorded_at_ms": recorded,
            }
        if current is not None and current not in validated:
            raise ValueError("toolbox_host_current_config_revision_missing")
        return {
            "contract": HOST_CONFIG_STATE_CONTRACT,
            "current_revision": current,
            "revisions": validated,
        }

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("toolbox_host_config_state_corrupt") from exc
        if not isinstance(payload, dict):
            raise ValueError("toolbox_host_config_state_corrupt")
        return self._validate(payload)

    def _write_unlocked(self, payload: Mapping[str, Any]) -> None:
        value = self._validate(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(raw)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_bounded_retries(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self) -> dict[str, Any]:
        with _exclusive_process_file_lock(self.lock_path):
            return self._read_unlocked()

    def apply(self, configuration: ToolboxHostProjectConfiguration) -> dict[str, Any]:
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        revision = configuration.config_revision
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            previous = state["current_revision"]
            changed = previous != revision
            if revision not in state["revisions"]:
                if len(state["revisions"]) >= MAX_HOST_CONFIG_REVISIONS:
                    removable = sorted(
                        (
                            (item["recorded_at_ms"], key)
                            for key, item in state["revisions"].items()
                            if key != previous
                        )
                    )
                    if not removable:
                        raise ValueError("toolbox_host_config_state_capacity")
                    state["revisions"].pop(removable[0][1])
                state["revisions"][revision] = {
                    "config": configuration.to_dict(),
                    "source_set_revision": configuration.source_set_revision,
                    "recorded_at_ms": int(self.clock() * 1000),
                }
            state["current_revision"] = revision
            self._write_unlocked(state)
        return {
            "changed": changed,
            "previous_revision": previous,
            "config_revision": revision,
            "source_set_revision": configuration.source_set_revision,
        }


__all__ = [
    "AtomicJsonToolboxHostConfigurationRepository",
    "HOST_CONFIG_STATE_CONTRACT",
    "MAX_HOST_CONFIG_REVISIONS",
]
