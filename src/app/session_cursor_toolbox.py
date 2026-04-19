from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .context_cursor import ChatCursor
from mp13_engine.mp13_toolbox import Toolbox, ToolsScope


def _normalized_name_set(items: Optional[Iterable[Any]]) -> set[str]:
    return {
        str(item or "").strip()
        for item in list(items or [])
        if str(item or "").strip()
    }


def all_tool_names(toolbox: Toolbox) -> List[str]:
    try:
        return [entry[0] for entry in toolbox.list_tools()]
    except Exception:
        return []


def tool_wildcard_groups(toolbox: Toolbox) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    if not toolbox:
        return groups
    intrinsic_names = sorted(toolbox.intrinsic_tools.keys()) if getattr(toolbox, "intrinsic_tools", None) else []
    if intrinsic_names:
        groups["*i"] = intrinsic_names
    callable_names = sorted(
        name for name, definition in getattr(toolbox, "tools", {}).items()
        if definition.get("_type") == "callable"
    )
    external_names = sorted(
        name for name, definition in getattr(toolbox, "tools", {}).items()
        if definition.get("_type") == "external"
    )
    if callable_names:
        groups["*c"] = callable_names
    if external_names:
        groups["*e"] = external_names
    return groups


def normalize_scope_tool_names(scope: ToolsScope, toolbox: Toolbox) -> Tuple[ToolsScope, List[str]]:
    if not toolbox:
        return scope, []

    known_names = sorted(set(toolbox.tools.keys()) | set(toolbox.intrinsic_tools.keys()))
    lower_map = {name.lower(): name for name in known_names}

    def resolve_name(raw: str) -> Optional[str]:
        if raw == "*":
            return "*"
        key = str(raw or "").lower()
        if key in lower_map:
            return lower_map[key]
        prefix_matches = [name for name in known_names if name.lower().startswith(key)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        non_guides = [name for name in prefix_matches if not name.lower().endswith("_guide")]
        if len(non_guides) == 1:
            return non_guides[0]
        return None

    warnings: List[str] = []

    def normalize_set(names: Set[str]) -> Set[str]:
        normalized: Set[str] = set()
        for raw in names:
            resolved = resolve_name(raw)
            if resolved:
                normalized.add(resolved)
            else:
                warnings.append(f"Tool '{raw}' not recognized for scope.")
        return normalized

    return (
        ToolsScope(
            mode=scope.mode,
            advertise_tools=normalize_set(scope.advertise_tools),
            silent_tools=normalize_set(scope.silent_tools),
            disabled_tools=normalize_set(scope.disabled_tools),
            gated_tools=normalize_set(scope.gated_tools),
            tool_constraints=dict(scope.tool_constraints or {}),
            label=scope.label,
        ).clean(),
        warnings,
    )


def collect_tools_scope_entries(cursor: ChatCursor) -> List[Tuple[Optional[str], ToolsScope]]:
    if not cursor or not cursor.current_turn:
        return []
    return cursor.session.get_effective_tools_scope_entries(cursor.current_turn)
