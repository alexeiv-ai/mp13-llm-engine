"""Strict atomic toolbox tool-change merge and deterministic change identities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .bundle_models import (
    ToolboxAutoAssignmentRequestV2,
    ToolboxDefinitionSpec,
    ToolboxManualAssignmentRequestV2,
    ToolboxEnvironmentMutationSpec,
    ToolboxPackageMutationSpec,
)
from .dependency_analysis import ToolboxAnalyzedImport, analyze_toolbox_bundle_imports
from .identity import identity_digest, require_digest


_PRINTABLE_ID = re.compile(r"[\x21-\x7e]{1,128}")
_TOOL_KEY = re.compile(r"[\x21-\x7e]{1,512}")


def _strict(payload: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    row = dict(payload or {})
    if set(row) != fields:
        raise ValueError(f"{label}_fields_invalid")
    return row


def _tool_key(value: Any, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    text = str(value or "")
    if not _TOOL_KEY.fullmatch(text):
        raise ValueError("tool_change_target_invalid")
    return text


@dataclass(frozen=True)
class ToolboxToolChange:
    change_id: str
    kind: str
    target_tool_key: str | None
    request_kind: str | None
    request: ToolboxAutoAssignmentRequestV2 | ToolboxManualAssignmentRequestV2 | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "kind": self.kind,
            "target_tool_key": self.target_tool_key,
            "request_kind": self.request_kind,
            "request": None if self.request is None else self.request.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxToolChange":
        row = _strict(
            payload,
            {"change_id", "kind", "target_tool_key", "request_kind", "request"},
            "tool_change",
        )
        change_id = str(row["change_id"] or "")
        if not _PRINTABLE_ID.fullmatch(change_id):
            raise ValueError("tool_change_id_invalid")
        kind = str(row["kind"] or "")
        if kind not in {"add", "update", "rename", "remove"}:
            raise ValueError("tool_change_kind_invalid")
        if kind == "add":
            target = _tool_key(row["target_tool_key"], nullable=True)
            if target is not None:
                raise ValueError("tool_change_add_target_invalid")
        else:
            target = _tool_key(row["target_tool_key"])
        if kind == "remove":
            if row["request_kind"] is not None or row["request"] is not None:
                raise ValueError("tool_change_remove_request_invalid")
            request_kind = None
            request = None
        else:
            request_kind = str(row["request_kind"] or "")
            if request_kind == "auto":
                request = ToolboxAutoAssignmentRequestV2.from_dict(row["request"])
            elif request_kind == "manual":
                request = ToolboxManualAssignmentRequestV2.from_dict(row["request"])
            else:
                raise ValueError("tool_change_request_kind_invalid")
        return cls(change_id, kind, target, request_kind, request)


@dataclass(frozen=True)
class NormalizedToolboxToolChange:
    change_id: str
    kind: str
    prior_tool_key: str | None
    tool_key: str | None
    request_kind: str | None

    def __post_init__(self) -> None:
        if not _PRINTABLE_ID.fullmatch(str(self.change_id or "")):
            raise ValueError("tool_change_id_invalid")
        if self.kind not in {"add", "update", "rename", "remove"}:
            raise ValueError("tool_change_kind_invalid")
        prior = _tool_key(self.prior_tool_key, nullable=True)
        resulting = _tool_key(self.tool_key, nullable=True)
        if (
            (self.kind == "add" and (prior is not None or resulting is None))
            or (self.kind == "remove" and (prior is None or resulting is not None))
            or (self.kind == "update" and (prior is None or prior != resulting))
            or (self.kind == "rename" and (prior is None or resulting is None or prior == resulting))
        ):
            raise ValueError("tool_change_normalized_keys_invalid")
        if self.request_kind not in {"auto", "manual", None}:
            raise ValueError("tool_change_request_kind_invalid")
        keys = {item for item in (prior, resulting) if item is not None}
        if self.request_kind is None and any(
            not item.startswith("intrinsic:") for item in keys
        ):
            raise ValueError("tool_change_request_kind_invalid")
        if self.request_kind is not None and any(
            item.startswith("intrinsic:") for item in keys
        ):
            raise ValueError("tool_change_request_kind_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "kind": self.kind,
            "prior_tool_key": self.prior_tool_key,
            "tool_key": self.tool_key,
            "request_kind": self.request_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizedToolboxToolChange":
        return cls(**_strict(
            payload,
            {"change_id", "kind", "prior_tool_key", "tool_key", "request_kind"},
            "tool_change_normalized",
        ))


@dataclass(frozen=True)
class ToolboxToolAnalysis:
    change_id: str
    tool_key: str | None
    prior_tool_key: str | None
    change: str
    imports: tuple[ToolboxAnalyzedImport, ...]
    environment_id: str
    package_mutations: tuple[ToolboxPackageMutationSpec, ...]
    approval_required: bool

    def __post_init__(self) -> None:
        normalized = NormalizedToolboxToolChange(
            self.change_id,
            self.change,
            self.prior_tool_key,
            self.tool_key,
            (
                None
                if all(
                    key is None or key.startswith("intrinsic:")
                    for key in (self.prior_tool_key, self.tool_key)
                )
                else "auto"
            ),
        )
        imports = tuple(sorted(self.imports, key=lambda item: item.import_root))
        if len(imports) > 512 or len({item.import_root for item in imports}) != len(imports):
            raise ValueError("tool_analysis_imports_invalid")
        mutations = tuple(sorted(
            self.package_mutations, key=lambda item: (item.distribution, item.mutation)
        ))
        if len({item.distribution for item in mutations}) != len(mutations):
            raise ValueError("tool_analysis_package_mutations_invalid")
        if not isinstance(self.approval_required, bool):
            raise ValueError("tool_analysis_approval_invalid")
        object.__setattr__(self, "change_id", normalized.change_id)
        object.__setattr__(self, "tool_key", normalized.tool_key)
        object.__setattr__(self, "prior_tool_key", normalized.prior_tool_key)
        object.__setattr__(self, "change", normalized.kind)
        object.__setattr__(
            self, "environment_id", require_digest(
                self.environment_id, label="tool_analysis_environment_id"
            )
        )
        object.__setattr__(self, "imports", imports)
        object.__setattr__(self, "package_mutations", mutations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "tool_key": self.tool_key,
            "prior_tool_key": self.prior_tool_key,
            "change": self.change,
            "imports": [item.to_dict() for item in self.imports],
            "environment_id": self.environment_id,
            "package_mutations": [item.to_dict() for item in self.package_mutations],
            "approval_required": self.approval_required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxToolAnalysis":
        row = _strict(
            payload,
            {
                "change_id", "tool_key", "prior_tool_key", "change", "imports",
                "environment_id", "package_mutations", "approval_required",
            },
            "tool_analysis",
        )
        if not isinstance(row["imports"], list) or not isinstance(row["package_mutations"], list):
            raise ValueError("tool_analysis_lists_invalid")
        return cls(
            **{
                **row,
                "imports": tuple(ToolboxAnalyzedImport.from_dict(item) for item in row["imports"]),
                "package_mutations": tuple(
                    ToolboxPackageMutationSpec.from_dict(item)
                    for item in row["package_mutations"]
                ),
            }
        )


def _definition_requests(
    definition: ToolboxDefinitionSpec,
) -> dict[str, tuple[str, ToolboxAutoAssignmentRequestV2 | ToolboxManualAssignmentRequestV2]]:
    return {
        **{item.stable_key: ("auto", item) for item in definition.auto_requests},
        **{item.stable_key: ("manual", item) for item in definition.manual_requests},
    }


def merge_toolbox_tool_changes(
    *,
    toolbox_id: str,
    expected_revision: str | None,
    active_revision: str | None,
    active_definition: ToolboxDefinitionSpec | Mapping[str, Any],
    changes: Sequence[ToolboxToolChange | Mapping[str, Any]],
) -> tuple[ToolboxDefinitionSpec, tuple[NormalizedToolboxToolChange, ...]]:
    """Merge one validated batch without mutating authoritative state."""

    active = (
        active_definition
        if isinstance(active_definition, ToolboxDefinitionSpec)
        else ToolboxDefinitionSpec.from_dict(active_definition)
    )
    if active.toolbox_id != str(toolbox_id or "").strip():
        raise ValueError("tool_change_toolbox_mismatch")
    expected = (
        None
        if expected_revision is None
        else require_digest(expected_revision, label="tool_change_expected_revision")
    )
    authoritative = (
        None
        if active_revision is None
        else require_digest(active_revision, label="tool_change_active_revision")
    )
    if expected != authoritative:
        raise ValueError("tool_change_revision_conflict")
    parsed = tuple(
        item if isinstance(item, ToolboxToolChange) else ToolboxToolChange.from_dict(item)
        for item in changes
    )
    if not 1 <= len(parsed) <= 512:
        raise ValueError("tool_change_count_invalid")
    if len({item.change_id for item in parsed}) != len(parsed):
        raise ValueError("tool_change_id_duplicate")
    targets = [item.target_tool_key for item in parsed if item.target_tool_key is not None]
    if len(set(targets)) != len(targets):
        raise ValueError("tool_change_target_duplicate")

    active_requests = _definition_requests(active)
    normalized: list[NormalizedToolboxToolChange] = []
    for change in parsed:
        prior = active_requests.get(change.target_tool_key or "")
        if change.kind != "add" and prior is None:
            raise ValueError("tool_change_target_not_found")
        if change.kind in {"update", "rename"}:
            if prior is None or change.request is None:
                raise ValueError("tool_change_request_invalid")
            if change.request_kind != prior[0]:
                raise ValueError("tool_change_request_kind_conflict")
            resulting_key = change.request.stable_key
            if change.kind == "update" and resulting_key != change.target_tool_key:
                raise ValueError("tool_change_update_key_changed")
            if change.kind == "update" and change.request.to_dict() == prior[1].to_dict():
                raise ValueError("tool_change_no_effect")
            if change.kind == "rename" and resulting_key == change.target_tool_key:
                raise ValueError("tool_change_rename_key_unchanged")
        elif change.kind == "add":
            if change.request is None:
                raise ValueError("tool_change_request_invalid")
            resulting_key = change.request.stable_key
        else:
            resulting_key = None
        normalized.append(NormalizedToolboxToolChange(
            change.change_id,
            change.kind,
            change.target_tool_key,
            resulting_key,
            prior[0] if change.kind == "remove" and prior is not None else change.request_kind,
        ))

    merged = {
        key: value for key, value in active_requests.items() if key not in set(targets)
    }
    for change in parsed:
        if change.request is None:
            continue
        key = change.request.stable_key
        if key in merged:
            raise ValueError("tool_change_result_conflict")
        merged[key] = (str(change.request_kind), change.request)

    proposed = ToolboxDefinitionSpec(
        toolbox_id=active.toolbox_id,
        expected_revision=authoritative,
        auto_requests=tuple(
            request for kind, request in merged.values() if kind == "auto"
        ),
        manual_requests=tuple(
            request for kind, request in merged.values() if kind == "manual"
        ),
        intrinsics=active.intrinsics,
    )
    return proposed, tuple(sorted(normalized, key=lambda item: item.change_id))


def deterministic_definition_changes(
    active_definition: ToolboxDefinitionSpec,
    proposed_definition: ToolboxDefinitionSpec,
) -> tuple[NormalizedToolboxToolChange, ...]:
    """Assign stable host IDs to request and intrinsic changes in a full definition."""

    active = _definition_requests(active_definition)
    proposed = _definition_requests(proposed_definition)
    rows: list[NormalizedToolboxToolChange] = []
    for key in sorted(set(active) | set(proposed)):
        before = active.get(key)
        after = proposed.get(key)
        if before is not None and after is not None and before[1].to_dict() == after[1].to_dict():
            continue
        kind = "add" if before is None else "remove" if after is None else "update"
        request_kind = after[0] if after is not None else before[0] if before is not None else None
        prior_key = key if before is not None else None
        tool_key = key if after is not None else None
        change_id = "host:" + identity_digest(
            "hosting.toolbox.host_change_id.v1",
            {"kind": kind, "prior_tool_key": prior_key, "tool_key": tool_key},
        )
        rows.append(NormalizedToolboxToolChange(
            change_id, kind, prior_key, tool_key, request_kind
        ))
    active_intrinsics = set(active_definition.intrinsics.names)
    proposed_intrinsics = set(proposed_definition.intrinsics.names)
    intrinsic_policy_changed = (
        active_definition.intrinsics.include_guides
        != proposed_definition.intrinsics.include_guides
        or dict(active_definition.intrinsics.sandbox_policy)
        != dict(proposed_definition.intrinsics.sandbox_policy)
    )
    changed_intrinsics = active_intrinsics ^ proposed_intrinsics
    if intrinsic_policy_changed:
        changed_intrinsics |= active_intrinsics & proposed_intrinsics
    for name in sorted(changed_intrinsics):
        in_active = name in active_intrinsics
        in_proposed = name in proposed_intrinsics
        kind = "update" if in_active and in_proposed else "remove" if in_active else "add"
        prior_key = f"intrinsic:{name}" if in_active else None
        tool_key = f"intrinsic:{name}" if in_proposed else None
        change_id = "host:" + identity_digest(
            "hosting.toolbox.host_change_id.v1",
            {"kind": kind, "prior_tool_key": prior_key, "tool_key": tool_key},
        )
        rows.append(NormalizedToolboxToolChange(
            change_id, kind, prior_key, tool_key, None
        ))
    return tuple(sorted(rows, key=lambda item: item.change_id))


def build_toolbox_tool_analysis(
    *,
    active_definition: ToolboxDefinitionSpec,
    proposed_definition: ToolboxDefinitionSpec,
    changes: Sequence[NormalizedToolboxToolChange],
    environment_mutations: Sequence[ToolboxEnvironmentMutationSpec],
) -> tuple[ToolboxToolAnalysis, ...]:
    """Bind each normalized change to bounded source and resolved offer evidence."""

    from mp13_engine.mp13_intrinsics_metadata import intrinsic_dependency_metadata

    active = _definition_requests(active_definition)
    proposed = _definition_requests(proposed_definition)
    offers_by_tool: dict[str, ToolboxEnvironmentMutationSpec] = {}
    for environment in environment_mutations:
        for tool in environment.tool_mutations:
            if tool.tool_key in offers_by_tool:
                raise ValueError("tool_analysis_environment_duplicate")
            offers_by_tool[tool.tool_key] = environment
    rows = []
    for change in changes:
        selected_key = change.tool_key or change.prior_tool_key
        if selected_key is None:
            raise ValueError("tool_analysis_tool_key_invalid")
        offer = offers_by_tool.get(selected_key)
        if offer is None:
            raise ValueError("tool_analysis_environment_missing")
        request_row = (
            proposed.get(change.tool_key or "")
            or active.get(change.prior_tool_key or "")
        )
        if request_row is not None:
            request = request_row[1]
            source_analysis = analyze_toolbox_bundle_imports(
                request.files,
                declared_imports=request.dependency.declared_imports,
            )
        else:
            intrinsic_key = change.tool_key or change.prior_tool_key or ""
            if not intrinsic_key.startswith("intrinsic:"):
                raise ValueError("tool_analysis_request_missing")
            metadata = intrinsic_dependency_metadata((intrinsic_key.split(":", 1)[1],))
            source_analysis = analyze_toolbox_bundle_imports(
                (), declared_imports=metadata["import_roots"]
            )
        alternative = next(
            item for item in offer.alternatives
            if item.alternative_id == offer.preferred_alternative_id
        )
        rows.append(ToolboxToolAnalysis(
            change_id=change.change_id,
            tool_key=change.tool_key,
            prior_tool_key=change.prior_tool_key,
            change=change.kind,
            imports=source_analysis.imports,
            environment_id=offer.environment_id,
            package_mutations=alternative.package_mutations,
            approval_required=offer.dependency_approval_required,
        ))
    return tuple(sorted(rows, key=lambda item: item.change_id))


__all__ = [
    "NormalizedToolboxToolChange",
    "ToolboxToolChange",
    "ToolboxToolAnalysis",
    "build_toolbox_tool_analysis",
    "deterministic_definition_changes",
    "merge_toolbox_tool_changes",
]
