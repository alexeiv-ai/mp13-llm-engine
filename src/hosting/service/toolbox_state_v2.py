"""Strict process-safe version-2 toolbox definition and active-route state."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..toolbox.bundle_models import ResolvedToolboxProfileSpec, ToolboxDefinitionSpec
from ..toolbox.identity import identity_digest, require_digest
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries
from .toolbox_runtime_identity import runtime_binding_digest


TOOLBOX_STATE_V2_CONTRACT = "hosting.toolbox.state.v2"
TOOLBOX_STATE_V2_DIGEST_DOMAIN = "hosting.toolbox.state.v2.digest"
MAX_ROLLOUT_HISTORY = 32


class ToolboxRevisionConflictError(RuntimeError):
    pass


class LegacyToolboxStateError(RuntimeError):
    pass


class AtomicJsonToolboxStateV2Repository:
    def __init__(self, path: Path, *, legacy_path: Path | None = None):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.legacy_path = Path(legacy_path).expanduser().resolve() if legacy_path is not None else None

    @staticmethod
    def _payload(toolboxes: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "contract": TOOLBOX_STATE_V2_CONTRACT,
            "version": 2,
            "toolboxes": copy.deepcopy(dict(toolboxes)),
        }
        return {
            **body,
            "state_digest": identity_digest(TOOLBOX_STATE_V2_DIGEST_DOMAIN, body),
        }

    @classmethod
    def _validate_snapshot(cls, toolbox_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        fields = {
            "toolbox_id", "active_revision", "definition", "profiles", "tool_routes",
            "environment_references", "rollout_history", "published_at_ms",
        }
        if set(row) != fields or row.get("toolbox_id") != toolbox_id:
            raise ValueError("toolbox_state_v2_snapshot_fields_invalid")
        revision = require_digest(row.get("active_revision"), label="toolbox_active_revision")
        definition = ToolboxDefinitionSpec.from_dict(row.get("definition"))
        if definition.toolbox_id != toolbox_id or definition.revision != revision:
            raise ValueError("toolbox_state_v2_definition_mismatch")
        if not isinstance(row.get("profiles"), dict) or not isinstance(row.get("tool_routes"), dict):
            raise ValueError("toolbox_state_v2_routes_invalid")
        profiles: dict[str, dict[str, Any]] = {}
        for profile_id, value in row["profiles"].items():
            profile_row = dict(value or {})
            allowed_profile_fields = {
                "profile", "manifest_hash", "engine_id", "tool_names", "environment_reference",
                "resolved_environment", "runtime_binding_digest",
            }
            if set(profile_row) - allowed_profile_fields or not {
                "profile", "manifest_hash", "engine_id", "tool_names", "environment_reference",
                "resolved_environment",
            }.issubset(profile_row):
                raise ValueError("toolbox_state_v2_profile_fields_invalid")
            profile = ResolvedToolboxProfileSpec.from_dict(profile_row["profile"])
            if profile_id != profile.profile_id:
                raise ValueError("toolbox_state_v2_profile_key_mismatch")
            manifest_hash = str(profile_row["manifest_hash"] or "").strip()
            if not (manifest_hash.startswith("sha256:") and len(manifest_hash) == 71):
                raise ValueError("toolbox_state_v2_manifest_hash_invalid")
            engine_id = str(profile_row["engine_id"] or "").strip()
            tool_names = sorted(str(item or "").strip() for item in profile_row["tool_names"])
            if not engine_id or any(not item for item in tool_names) or len(set(tool_names)) != len(tool_names):
                raise ValueError("toolbox_state_v2_profile_runtime_invalid")
            environment_reference = str(profile_row["environment_reference"] or "").strip()
            if not environment_reference:
                raise ValueError("toolbox_state_v2_environment_reference_required")
            binding_digest = str(profile_row.get("runtime_binding_digest") or "").strip()
            if not binding_digest:
                binding_digest = runtime_binding_digest(
                    toolbox_id=toolbox_id,
                    profile_id=profile.profile_id,
                    manifest_hash=manifest_hash,
                    environment_reference=environment_reference,
                    engine_id=engine_id,
                    definition_revision=revision,
                )
            binding_digest = require_digest(binding_digest, label="toolbox_state_v2_runtime_binding_digest")
            resolved_environment = dict(profile_row["resolved_environment"] or {})
            if profile.custom_resolved_lock_digest is not None:
                from ..toolbox.hermetic_environment import ResolvedToolboxEnvironmentInput

                resolved = ResolvedToolboxEnvironmentInput.from_dict(resolved_environment)
                if resolved.environment_key != profile.environment_key:
                    raise ValueError("toolbox_state_v2_resolved_environment_mismatch")
            profiles[profile_id] = {
                "profile": profile.to_dict(),
                "manifest_hash": manifest_hash,
                "engine_id": engine_id,
                "tool_names": tool_names,
                "environment_reference": environment_reference,
                "resolved_environment": resolved_environment,
                "runtime_binding_digest": binding_digest,
            }
        routes: dict[str, dict[str, Any]] = {}
        for tool_name, value in row["tool_routes"].items():
            name = str(tool_name or "").strip()
            route = dict(value or {})
            if not name or set(route) != {"profile_id", "engine_id", "non_restartable"}:
                raise ValueError("toolbox_state_v2_route_fields_invalid")
            profile_id = str(route["profile_id"] or "").strip()
            engine_id = str(route["engine_id"] or "").strip()
            profile_row = profiles.get(profile_id)
            if profile_row is None or engine_id != profile_row["engine_id"] or name not in profile_row["tool_names"]:
                raise ValueError("toolbox_state_v2_route_target_invalid")
            if not isinstance(route["non_restartable"], bool):
                raise ValueError("toolbox_state_v2_route_policy_invalid")
            routes[name] = {
                "profile_id": profile_id,
                "engine_id": engine_id,
                "non_restartable": route["non_restartable"],
            }
        references = sorted(str(item or "").strip() for item in row["environment_references"])
        if any(not item for item in references) or len(set(references)) != len(references):
            raise ValueError("toolbox_state_v2_environment_references_invalid")
        if set(references) != {
            profile["environment_reference"] for profile in profiles.values()
        }:
            raise ValueError("toolbox_state_v2_environment_references_incomplete")
        history = [dict(item or {}) for item in row["rollout_history"]]
        if len(history) > MAX_ROLLOUT_HISTORY:
            raise ValueError("toolbox_state_v2_history_unbounded")
        for item in history:
            if set(item) != {"revision", "published_at_ms", "profile_count", "tool_count"}:
                raise ValueError("toolbox_state_v2_history_fields_invalid")
            require_digest(item["revision"], label="toolbox_history_revision")
            if any(isinstance(item[key], bool) or not isinstance(item[key], int) or item[key] < 0 for key in ("published_at_ms", "profile_count", "tool_count")):
                raise ValueError("toolbox_state_v2_history_value_invalid")
        published_at_ms = row["published_at_ms"]
        if isinstance(published_at_ms, bool) or not isinstance(published_at_ms, int) or published_at_ms < 0:
            raise ValueError("toolbox_state_v2_published_at_invalid")
        return {
            "toolbox_id": toolbox_id,
            "active_revision": revision,
            "definition": definition.to_dict(),
            "profiles": profiles,
            "tool_routes": routes,
            "environment_references": references,
            "rollout_history": history,
            "published_at_ms": published_at_ms,
        }

    @classmethod
    def _validate_state(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {"contract", "version", "toolboxes", "state_digest"}:
            raise ValueError("toolbox_state_v2_fields_invalid")
        if row.get("contract") != TOOLBOX_STATE_V2_CONTRACT or row.get("version") != 2:
            raise ValueError("toolbox_state_v2_contract_invalid")
        if not isinstance(row.get("toolboxes"), dict):
            raise ValueError("toolbox_state_v2_toolboxes_invalid")
        toolboxes = {
            str(toolbox_id): cls._validate_snapshot(str(toolbox_id), value)
            for toolbox_id, value in row["toolboxes"].items()
        }
        expected = cls._payload(toolboxes)
        if row.get("state_digest") != expected["state_digest"]:
            raise ValueError("toolbox_state_v2_digest_mismatch")
        return expected

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            if self.legacy_path is not None and self.legacy_path.exists():
                raise LegacyToolboxStateError(
                    "toolbox_state_v1_unsupported: run toolbox-state-archive-v1 before using definition APIs"
                )
            return self._payload({})
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("toolbox_state_v2_corrupt") from exc
        if not isinstance(payload, dict):
            raise ValueError("toolbox_state_v2_corrupt")
        return self._validate_state(payload)

    def _write_unlocked(self, payload: Mapping[str, Any]) -> None:
        state = self._validate_state(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        temporary = Path(raw)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_bounded_retries(temporary, self.path)
            self._fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def initialize_empty(self) -> dict[str, Any]:
        with _exclusive_process_file_lock(self.lock_path):
            if self.path.exists():
                raise FileExistsError("toolbox_state_v2_already_initialized")
            if self.legacy_path is not None and self.legacy_path.exists():
                raise LegacyToolboxStateError("toolbox_state_v1_still_present")
            state = self._payload({})
            self._write_unlocked(state)
            return copy.deepcopy(state)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            if self.legacy_path is not None and self.legacy_path.exists():
                raise LegacyToolboxStateError(
                    "toolbox_state_v1_unsupported: run toolbox-state-archive-v1 before using definition APIs"
                )
            return copy.deepcopy(self._payload({}))
        with _exclusive_process_file_lock(self.lock_path):
            return copy.deepcopy(self._read_unlocked())

    def get(self, toolbox_id: str) -> dict[str, Any] | None:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id_required")
        return copy.deepcopy(self.read()["toolboxes"].get(tid))

    def publish(
        self,
        *,
        toolbox_id: str,
        expected_revision: str | None,
        definition: Mapping[str, Any],
        profiles: Mapping[str, Any],
        tool_routes: Mapping[str, Any],
        environment_references: Sequence[str],
        published_at_ms: int,
    ) -> dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        model = ToolboxDefinitionSpec.from_dict(definition)
        if model.toolbox_id != tid or model.expected_revision != expected_revision:
            raise ValueError("toolbox_state_v2_publish_definition_mismatch")
        with _exclusive_process_file_lock(self.lock_path):
            state = self._read_unlocked()
            active = dict(state["toolboxes"].get(tid) or {})
            current_revision = active.get("active_revision")
            if current_revision != expected_revision:
                raise ToolboxRevisionConflictError("toolbox_revision_conflict")
            history = list(active.get("rollout_history") or [])
            history.append(
                {
                    "revision": model.revision,
                    "published_at_ms": int(published_at_ms),
                    "profile_count": len(profiles),
                    "tool_count": len(tool_routes),
                }
            )
            snapshot = self._validate_snapshot(
                tid,
                {
                    "toolbox_id": tid,
                    "active_revision": model.revision,
                    "definition": model.to_dict(),
                    "profiles": dict(profiles),
                    "tool_routes": dict(tool_routes),
                    "environment_references": list(environment_references),
                    "rollout_history": history[-MAX_ROLLOUT_HISTORY:],
                    "published_at_ms": int(published_at_ms),
                },
            )
            state["toolboxes"][tid] = snapshot
            updated = self._payload(state["toolboxes"])
            self._write_unlocked(updated)
            return copy.deepcopy(snapshot)


__all__ = [
    "AtomicJsonToolboxStateV2Repository",
    "MAX_ROLLOUT_HISTORY",
    "TOOLBOX_STATE_V2_CONTRACT",
    "LegacyToolboxStateError",
    "ToolboxRevisionConflictError",
]
