"""Reusable callable-surface adapters for Host Capability and Toolbox metadata."""
from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from .sandbox.host_capabilities import (
    HOST_CAPABILITY_APPROVAL_CONTRACT,
    HOST_CAPABILITY_CALL_CONTRACT,
    HostCapabilityApproval,
    HostCapabilityDescriptor,
    HostCapabilityProviderRef,
    default_group_path,
    validate_provider_response,
)

HOST_CALLABLE_SCHEMA_CONTRACT = "hosting.sandbox.callable_schema.v1"
HOST_CAPABILITY_PROVIDER_RESPONSE_CONTRACT = "hosting.sandbox.host_capability_provider_response.v1"
HOST_CAPABILITY_APPROVAL_DECISION_CONTRACT = "hosting.sandbox.host_capability_approval_decision.v1"

SAFE_CORRELATION_FIELDS = (
    "workflow_id",
    "instance_id",
    "node_id",
    "request_id",
    "cursor_id",
    "context_id",
    "branch_id",
    "session_tree_id",
    "session_id",
    "actor",
    "provider_id",
    "method",
    "approval_id",
    "host_call_id",
    "provider_call_id",
)

_IDENT_RE = re.compile(r"[^a-z0-9_]+")
_APPROVAL_DECISIONS = {"deny", "allow_once", "add_to_scope"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    out: Dict[str, Any] = {}
    for name in (
        "view_id",
        "mode",
        "allowed_tools",
        "advertised_tools",
        "hidden_allowed_tools",
        "disabled_tools",
        "gated_tools",
        "tool_constraints",
    ):
        if hasattr(value, name):
            out[name] = getattr(value, name)
    return out


def _string_set(values: Any) -> set[str]:
    return {_clean(item) for item in list(values or []) if _clean(item)}


def _json_schema(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {"type": "object"}


def _method_segment(value: str) -> str:
    raw = _IDENT_RE.sub("_", _clean(value).lower()).strip("_")
    return raw or "method"


def _method_name(namespace: str, tool_name: str) -> str:
    raw = _clean(tool_name)
    if "." in raw:
        return ".".join(_method_segment(part) for part in raw.split(".") if _method_segment(part))
    return f"{_method_segment(namespace)}.{_method_segment(raw)}"


def _metadata_for_tool(tool_metadata: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    for key in (tool_name, _clean(tool_name).split(".", 1)[-1]):
        value = tool_metadata.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _toolbox_method_names(toolbox_description: Dict[str, Any], tools_view: Dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for key in (
        "all_registered_tool_names",
        "allowed_tool_names",
        "advertised_tool_names",
        "hidden_allowed_tool_names",
        "disabled_tool_names",
        "gated_tool_names",
        "allowed_tools",
        "advertised_tools",
        "hidden_allowed_tools",
        "disabled_tools",
        "gated_tools",
    ):
        names.update(_string_set(toolbox_description.get(key)))
        names.update(_string_set(tools_view.get(key)))
    metadata = toolbox_description.get("tool_metadata")
    if isinstance(metadata, dict):
        names.update(_string_set(metadata.keys()))
    return sorted(names)


def toolbox_to_host_capability_descriptors(
    toolbox_description: Dict[str, Any],
    *,
    tools_view: Optional[Any] = None,
    provider_id: str = "",
    owner: str = "client",
    visibility: str = "workflow",
    namespace: str = "toolbox",
) -> list[HostCapabilityDescriptor]:
    """Convert toolbox describe output and a ToolsView into host capability descriptors."""
    toolbox = dict(toolbox_description or {})
    view = _as_dict(tools_view if tools_view is not None else toolbox.get("tools_view"))
    provider_name = _clean(provider_id) or _clean(toolbox.get("toolbox_id")) or "toolbox"
    ns = _method_segment(namespace or provider_name)
    allowed = _string_set(toolbox.get("allowed_tool_names") or toolbox.get("allowed_tools")) | _string_set(view.get("allowed_tools"))
    advertised = _string_set(toolbox.get("advertised_tool_names") or toolbox.get("advertised_tools")) | _string_set(view.get("advertised_tools"))
    hidden_allowed = _string_set(toolbox.get("hidden_allowed_tool_names") or toolbox.get("hidden_allowed_tools")) | _string_set(view.get("hidden_allowed_tools"))
    disabled = _string_set(toolbox.get("disabled_tool_names") or toolbox.get("disabled_tools")) | _string_set(view.get("disabled_tools"))
    gated = _string_set(toolbox.get("gated_tool_names") or toolbox.get("gated_tools")) | _string_set(view.get("gated_tools"))
    constraints = dict(toolbox.get("tool_constraints") or view.get("tool_constraints") or {})
    tool_metadata = dict(toolbox.get("tool_metadata") or {})

    descriptors: list[HostCapabilityDescriptor] = []
    for tool_name in _toolbox_method_names(toolbox, view):
        meta = _metadata_for_tool(tool_metadata, tool_name)
        method = _method_name(ns, tool_name)
        method_ns = method.split(".", 1)[0]
        permissions = list(meta.get("permissions") or meta.get("scopes") or [])
        scope_requirements = list(meta.get("scope_requirements") or [])
        descriptors.append(
            HostCapabilityDescriptor(
                name=method,
                namespace=method_ns,
                group_path=list(meta.get("group_path") or default_group_path(method)),
                description=_clean(meta.get("description") or meta.get("summary") or f"Invoke toolbox tool {tool_name}."),
                args_schema=_json_schema(meta.get("args_schema") or meta.get("parameters") or meta.get("input_schema")),
                result_schema=_json_schema(meta.get("result_schema") or meta.get("output_schema")),
                permissions=[_clean(item) for item in permissions if _clean(item)],
                scope_requirements=[dict(item or {}) for item in scope_requirements if isinstance(item, dict)],
                approval=HostCapabilityApproval(mode="always" if tool_name in gated else "none", ttl_seconds=0),
                provider=HostCapabilityProviderRef(
                    provider_id=provider_name,
                    kind="toolbox_session",
                    owner=_clean(owner) or "client",
                    visibility=_clean(visibility) or "workflow",
                ),
                metadata={
                    "toolbox": {
                        "tool_name": tool_name,
                        "toolbox_id": toolbox.get("toolbox_id") or provider_name,
                        "view_id": view.get("view_id"),
                        "mode": view.get("mode"),
                        "allowed": tool_name in allowed,
                        "advertised": tool_name in advertised,
                        "hidden_allowed": tool_name in hidden_allowed,
                        "disabled": tool_name in disabled,
                        "gated": tool_name in gated,
                        "constraints": dict(constraints.get(tool_name) or {}),
                    }
                },
            )
        )
    return descriptors


def host_capability_descriptors_to_callable_schemas(
    descriptors: Iterable[HostCapabilityDescriptor | Dict[str, Any]],
    *,
    include_hidden: bool = False,
    include_disabled: bool = False,
) -> list[Dict[str, Any]]:
    """Convert host capability descriptors to sandbox/model-facing callable schemas."""
    out: list[Dict[str, Any]] = []
    for item in descriptors:
        descriptor = item if isinstance(item, HostCapabilityDescriptor) else HostCapabilityDescriptor.from_dict(dict(item or {}))
        row = descriptor.to_dict()
        toolbox = dict(dict(row.get("metadata") or {}).get("toolbox") or {})
        if bool(toolbox.get("hidden_allowed", False)) and not include_hidden:
            continue
        if bool(toolbox.get("disabled", False)) and not include_disabled:
            continue
        out.append(
            {
                "contract": HOST_CALLABLE_SCHEMA_CONTRACT,
                "name": row["name"],
                "namespace": row["namespace"],
                "description": row.get("description", ""),
                "arguments_schema": dict(row.get("args_schema") or {}),
                "result_schema": dict(row.get("result_schema") or {}),
                "permissions": list(row.get("permissions") or []),
                "scope_requirements": list(row.get("scope_requirements") or []),
                "approval": dict(row.get("approval") or {}),
                "provider": dict(row.get("provider") or {}),
                "metadata": dict(row.get("metadata") or {}),
            }
        )
    return out


def extract_safe_correlation_metadata(*payloads: Any) -> Dict[str, Any]:
    """Return only safe correlation fields from one or more payload dictionaries."""
    out: Dict[str, Any] = {}
    for payload in payloads:
        row = dict(payload or {}) if isinstance(payload, dict) else {}
        for key in SAFE_CORRELATION_FIELDS:
            value = row.get(key)
            if value is not None and _clean(value):
                out[key] = value
        for nested_key in ("context", "correlation", "provider", "approval"):
            nested = row.get(nested_key)
            if isinstance(nested, dict):
                for key in SAFE_CORRELATION_FIELDS:
                    value = nested.get(key)
                    if value is not None and _clean(value):
                        out[key] = value
    return out


def host_capability_provider_success(provider_call_id: str, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "contract": HOST_CAPABILITY_PROVIDER_RESPONSE_CONTRACT,
        "status": "ok",
        "provider_call_id": _clean(provider_call_id),
        "result": dict(result or {}),
    }


def host_capability_provider_error(
    provider_call_id: str,
    *,
    reason: str,
    message: str = "",
    detail: Optional[Dict[str, Any]] = None,
    status: str = "error",
) -> Dict[str, Any]:
    return {
        "contract": HOST_CAPABILITY_PROVIDER_RESPONSE_CONTRACT,
        "status": _clean(status) or "error",
        "provider_call_id": _clean(provider_call_id),
        "reason": _clean(reason) or "host_capability_provider_error",
        "message": _clean(message),
        "detail": dict(detail or {}),
    }


def normalize_host_capability_provider_response(response: Dict[str, Any], *, provider_call_id: str) -> Dict[str, Any]:
    return validate_provider_response(dict(response or {}), provider_call_id=provider_call_id)


def bind_host_capability_provider_callback(callback: Callable[..., Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Wrap a client callback so it consumes and returns normalized provider envelopes."""

    def _invoke(envelope: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(envelope or {})
        provider_call_id = _clean(row.get("provider_call_id"))
        if not provider_call_id:
            return host_capability_provider_error("", reason="host_capability_provider_call_id_required")
        try:
            if len(inspect.signature(callback).parameters) <= 1:
                result = callback(row)
            else:
                result = callback(
                    row.get("method"),
                    dict(row.get("arguments") or {}),
                    dict(row.get("context") or {}),
                )
            if isinstance(result, dict) and _clean(result.get("provider_call_id")):
                normalize_host_capability_provider_response(result, provider_call_id=provider_call_id)
                return dict(result)
            return host_capability_provider_success(provider_call_id, dict(result or {}))
        except TimeoutError as exc:
            return host_capability_provider_error(provider_call_id, reason="host_call_timeout", message=str(exc))
        except KeyboardInterrupt:
            return host_capability_provider_error(provider_call_id, reason="host_call_canceled", message="provider callback canceled")
        except Exception as exc:
            return host_capability_provider_error(
                provider_call_id,
                reason="host_capability_provider_error",
                message=str(exc),
                detail={"error_type": type(exc).__name__},
            )

    return _invoke


def host_capability_approval_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(payload or {})
    arguments = dict(row.get("arguments") or {})
    context = dict(row.get("context") or {})
    return {
        "contract": HOST_CAPABILITY_APPROVAL_CONTRACT,
        "approval_id": _clean(row.get("approval_id")),
        "provider_call_id": _clean(row.get("provider_call_id")),
        "host_call_id": _clean(row.get("host_call_id")),
        "method": _clean(row.get("method")),
        "provider": dict(row.get("provider") or {}),
        "approval": dict(row.get("approval") or {}),
        "context": {
            key: context.get(key)
            for key in ("workflow_id", "instance_id", "request_id", "package_id", "actor", "node_id", "cursor_id", "context_id")
            if context.get(key) is not None
        },
        "argument_keys": sorted(_clean(key) for key in arguments.keys() if _clean(key)),
        "correlation": extract_safe_correlation_metadata(row, context),
    }


def host_capability_approval_decision(
    decision: str,
    *,
    scope_constraints: Optional[Dict[str, Any]] = None,
    reason: str = "",
    message: str = "",
    approval_id: str = "",
) -> Dict[str, Any]:
    value = _clean(decision).lower()
    if value not in _APPROVAL_DECISIONS:
        value = "deny"
    approved = value in {"allow_once", "add_to_scope"}
    return {
        "contract": HOST_CAPABILITY_APPROVAL_DECISION_CONTRACT,
        "status": "approved" if approved else "denied",
        "approved": approved,
        "decision": value,
        "approval_id": _clean(approval_id) or None,
        "reason": _clean(reason) or None,
        "message": _clean(message) or None,
        "scope_constraints": dict(scope_constraints or {}),
    }


__all__ = [
    "HOST_CALLABLE_SCHEMA_CONTRACT",
    "HOST_CAPABILITY_APPROVAL_DECISION_CONTRACT",
    "HOST_CAPABILITY_PROVIDER_RESPONSE_CONTRACT",
    "SAFE_CORRELATION_FIELDS",
    "bind_host_capability_provider_callback",
    "extract_safe_correlation_metadata",
    "host_capability_approval_decision",
    "host_capability_approval_request",
    "host_capability_descriptors_to_callable_schemas",
    "host_capability_provider_error",
    "host_capability_provider_success",
    "normalize_host_capability_provider_response",
    "toolbox_to_host_capability_descriptors",
]
