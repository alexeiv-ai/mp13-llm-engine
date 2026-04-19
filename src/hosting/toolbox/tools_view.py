"""ToolsView serialization, approval, and scope helpers."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mp13_engine.mp13_toolbox import ToolBoxRef, ToolsScope, ToolsView

_HOSTED_TOOL_APPROVAL_CALLBACK = "tool_requires_confirmation"
_HOSTED_TOOL_APPROVAL_DECISIONS = ("deny", "allow_once", "add_to_scope")


def serialize_tools_view(tools_view: Optional[ToolsView]) -> Optional[Dict[str, Any]]:
    if tools_view is None:
        return None
    payload = {
        "view_id": str(tools_view.view_id or "").strip(),
        "mode": str(tools_view.mode or "").strip(),
        "allowed_tools": sorted(str(item or "").strip() for item in list(tools_view.allowed_tools or []) if str(item or "").strip()),
        "advertised_tools": sorted(str(item or "").strip() for item in list(tools_view.advertised_tools or []) if str(item or "").strip()),
        "hidden_allowed_tools": sorted(str(item or "").strip() for item in list(tools_view.hidden_allowed_tools or []) if str(item or "").strip()),
        "disabled_tools": sorted(str(item or "").strip() for item in list(tools_view.disabled_tools or []) if str(item or "").strip()),
        "gated_tools": sorted(str(item or "").strip() for item in list(tools_view.gated_tools or []) if str(item or "").strip()),
    }
    if dict(tools_view.tool_constraints or {}):
        payload["tool_constraints"] = json.loads(json.dumps(dict(tools_view.tool_constraints or {})))
    return payload

def _clone_tools_view(tools_view: Optional[ToolsView]) -> Optional[ToolsView]:
    if tools_view is None:
        return None
    return ToolsView(
        view_id=str(tools_view.view_id or "").strip(),
        mode=str(tools_view.mode or "").strip() or "advertised",
        allowed_tools=set(tools_view.allowed_tools or set()),
        advertised_tools=set(tools_view.advertised_tools or set()),
        hidden_allowed_tools=set(tools_view.hidden_allowed_tools or set()),
        disabled_tools=set(tools_view.disabled_tools or set()),
        gated_tools=set(tools_view.gated_tools or set()),
        tool_constraints=json.loads(json.dumps(dict(tools_view.tool_constraints or {}))),
    )


def _approve_tool_in_view(tools_view: Optional[ToolsView], tool_name: str, *, mutate: bool) -> Optional[ToolsView]:
    view = tools_view if mutate else _clone_tools_view(tools_view)
    if view is None:
        return None
    name = str(tool_name or "").strip()
    if not name:
        return view
    view.gated_tools.discard(name)
    if name not in view.disabled_tools:
        view.allowed_tools.add(name)
    return view


def _extract_scope_constraints(result: Any, tool_name: str) -> Optional[Dict[str, Any]]:
    payload = dict(result or {}) if isinstance(result, dict) else {}
    tool_key = str(tool_name or "").strip()
    scoped = payload.get("scope_constraints")
    if isinstance(scoped, dict):
        if tool_key and tool_key in scoped:
            if scoped.get(tool_key) is None:
                return None
            if isinstance(scoped.get(tool_key), dict):
                return json.loads(json.dumps(dict(scoped.get(tool_key) or {})))
            return {}
        if any(isinstance(v, dict) or v is None for v in scoped.values()):
            return {}
        return json.loads(json.dumps(scoped))
    return {}


def _merge_scope_ref_into_callback_context(callback_context: Any, scope_ref: Optional[ToolBoxRef]) -> Any:
    if scope_ref is None:
        return callback_context
    if isinstance(callback_context, dict):
        merged = dict(callback_context)
        merged.setdefault("toolbox_ref", scope_ref)
        return merged
    if callback_context is None:
        return {"toolbox_ref": scope_ref}
    return {"toolbox_ref": scope_ref, "user_context": callback_context}


def _apply_tool_constraints_in_view(
    tools_view: Optional[ToolsView],
    tool_name: str,
    constraints: Optional[Dict[str, Any]],
    *,
    mutate: bool,
) -> Optional[ToolsView]:
    view = tools_view if mutate else _clone_tools_view(tools_view)
    if view is None:
        return None
    name = str(tool_name or "").strip()
    if not name:
        return view
    payload = dict(constraints or {})
    if payload:
        view.tool_constraints[name] = json.loads(json.dumps(payload))
    else:
        view.tool_constraints.pop(name, None)
    return view


def _resolve_scope_ref_from_callback_context(callback_context: Any) -> Optional[ToolBoxRef]:
    candidate = None
    cursor = None
    if isinstance(callback_context, ToolBoxRef):
        return callback_context
    if isinstance(callback_context, dict):
        candidate = callback_context.get("toolbox_ref") or callback_context.get("scope_ref")
        cursor = callback_context.get("cursor")
    else:
        candidate = getattr(callback_context, "toolbox_ref", None) or getattr(callback_context, "scope_ref", None)
        cursor = getattr(callback_context, "cursor", None)
    if isinstance(candidate, ToolBoxRef):
        return candidate
    context = getattr(cursor, "context", None) if cursor is not None else None
    ref = getattr(context, "toolbox_ref", None) if context is not None else None
    return ref if isinstance(ref, ToolBoxRef) else None


def _persist_approved_tool(scope_ref: Optional[ToolBoxRef], tool_name: str) -> bool:
    if scope_ref is None or not callable(getattr(scope_ref, "mutate_scope", None)):
        return False
    name = str(tool_name or "").strip()
    if not name:
        return False

    def _update(scope: ToolsScope) -> ToolsScope:
        scope = scope or ToolsScope()
        scope.gated_tools.discard(name)
        return scope

    scope_ref.mutate_scope(_update)
    return True


def _persist_scope_constraints(scope_ref: Optional[ToolBoxRef], tool_name: str, constraints: Optional[Dict[str, Any]]) -> bool:
    if scope_ref is None or not callable(getattr(scope_ref, "mutate_scope", None)):
        return False
    name = str(tool_name or "").strip()
    if not name:
        return False
    payload = dict(constraints or {})

    def _update(scope: ToolsScope) -> ToolsScope:
        scope = scope or ToolsScope()
        if payload:
            scope.tool_constraints[name] = json.loads(json.dumps(payload))
        else:
            scope.tool_constraints.pop(name, None)
        return scope

    scope_ref.mutate_scope(_update)
    return True


def _coerce_approval_decision(result: Any) -> str:
    if isinstance(result, str):
        decision = str(result or "").strip().lower()
    elif isinstance(result, dict):
        decision = str(result.get("decision") or "").strip().lower()
    else:
        decision = ""
    return decision if decision in _HOSTED_TOOL_APPROVAL_DECISIONS else "deny"


def _approval_timeout_seconds(callback_context: Any, default_seconds: float = 15.0) -> float:
    raw: Any = None
    if isinstance(callback_context, dict):
        raw = callback_context.get("approval_timeout_seconds")
    else:
        raw = getattr(callback_context, "approval_timeout_seconds", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default_seconds or 15.0)
    if value <= 0:
        value = float(default_seconds or 15.0)
    return value
