from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mp13_engine.mp13_toolbox import ToolsView


def _normalized_name_set(items: Optional[Iterable[Any]]) -> set[str]:
    return {
        str(item or "").strip()
        for item in list(items or [])
        if str(item or "").strip()
    }


def summarize_effective_tool_view(
    tools_view: ToolsView,
    *,
    hosted_advertised_tool_names: Optional[Sequence[str]] = None,
    hosted_hidden_allowed_tool_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    local_advertised = _normalized_name_set(tools_view.advertised_tools)
    local_hidden = _normalized_name_set(tools_view.hidden_allowed_tools)
    local_disabled = _normalized_name_set(tools_view.disabled_tools)
    hosted_advertised = _normalized_name_set(hosted_advertised_tool_names)
    hosted_hidden = _normalized_name_set(hosted_hidden_allowed_tool_names)
    hosted_allowed = hosted_advertised | hosted_hidden
    if hosted_allowed:
        effective_advertised = local_advertised & hosted_advertised
        effective_hidden = (local_hidden & hosted_allowed) | ((local_advertised & hosted_hidden) - hosted_advertised)
        local_known = local_advertised | local_hidden | local_disabled
        hosted_gated = ((local_known & hosted_allowed) - effective_advertised - effective_hidden) | (
            (local_advertised | local_hidden) - hosted_allowed
        )
    else:
        effective_advertised = set(local_advertised)
        effective_hidden = set(local_hidden)
        hosted_gated = set()
    return {
        "mode": str(tools_view.mode or "").strip(),
        "local_advertised_tools": sorted(local_advertised),
        "effective_advertised_tools": sorted(effective_advertised),
        "effective_hidden_allowed_tools": sorted(effective_hidden),
        "disabled_tools": sorted(local_disabled),
        "hosted_gated_tools": sorted(hosted_gated),
        "hosted_visible_tools": sorted(effective_advertised),
        "hosted_hidden_allowed_tools": sorted(effective_hidden),
        "hosted_execution": bool(hosted_allowed),
    }


def annotate_tool_listing(
    listed_tools: Sequence[Tuple[str, str, str, bool, bool, bool, bool]],
    *,
    tools_view: Optional[ToolsView] = None,
    hosted_advertised_tool_names: Optional[Sequence[str]] = None,
    hosted_hidden_allowed_tool_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    hosted_advertised = _normalized_name_set(hosted_advertised_tool_names)
    hosted_hidden = _normalized_name_set(hosted_hidden_allowed_tool_names)
    hosted = hosted_advertised | hosted_hidden
    local_allowed = _normalized_name_set(tools_view.allowed_tools) if tools_view else set()
    local_advertised = _normalized_name_set(tools_view.advertised_tools) if tools_view else set()
    rows: List[Dict[str, Any]] = []
    for item in list(listed_tools or []):
        name, description, tool_type, is_active, is_hidden, is_guide, is_modified = item
        tool_name = str(name or "").strip()
        local_visible = tool_name in local_advertised if tools_view else bool(is_active and not is_hidden)
        scope_allowed = tool_name in local_allowed if tools_view else bool(is_active)
        if hosted:
            if tool_name in hosted and scope_allowed:
                availability = "Yes"
                via = "hosted-hidden" if tool_name in hosted_hidden else "hosted"
            elif local_visible or scope_allowed:
                availability = "No"
                via = "gated"
            else:
                availability = "No"
                via = "hidden"
        else:
            availability = "Yes" if scope_allowed else "No"
            via = "native" if scope_allowed else "hidden"
        rows.append(
            {
                "name": tool_name,
                "description": str(description or ""),
                "tool_type": str(tool_type or ""),
                "is_active": bool(is_active),
                "is_hidden": bool(is_hidden),
                "is_guide": bool(is_guide),
                "is_modified": bool(is_modified),
                "availability": availability,
                "via": via,
            }
        )
    return rows
