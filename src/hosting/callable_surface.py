"""Reusable callable-surface adapters for Host Capability and Toolbox metadata."""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from .sandbox.host_capabilities import (
    HOST_CAPABILITY_APPROVAL_CONTRACT,
    HOST_CAPABILITY_CALL_CONTRACT,
    HostCapabilityApproval,
    HostCapabilityDescriptor,
    HostCapabilityProviderRef,
    build_argument_preview,
    default_group_path,
    validate_provider_response,
)

HOST_CALLABLE_SCHEMA_CONTRACT = "hosting.sandbox.callable_schema.v1"
HOST_CAPABILITY_PROVIDER_RESPONSE_CONTRACT = "hosting.sandbox.host_capability_provider_response.v1"
HOST_CAPABILITY_APPROVAL_DECISION_CONTRACT = "hosting.sandbox.host_capability_approval_decision.v1"
HOST_CAPABILITY_PROVIDER_CALLBACK_NAME = "host_capability.call"
HOST_CAPABILITY_DISPATCH_CALLBACK_NAME = "host_capability.dispatch"
HOST_CAPABILITY_APPROVAL_CALLBACK_NAME = "host_capability.approval"
HOST_CAPABILITY_BRIDGE_POLICY_CONTRACT = "hosting.sandbox.host_capability_bridge_policy.v1"
TOOLBOX_BROKERED_IO_CALL_SURFACE_CONTRACT = "hosting.toolbox.brokered_io.call_surface.v1"

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
    "toolbox_id",
    "actor",
    "provider_kind",
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
    conflict_policy: str = "error",
    session_id: str = "",
) -> list[Dict[str, Any]]:
    """Convert host capability descriptors to sandbox/model-facing callable schemas."""
    out: list[Dict[str, Any]] = []
    seen_names: set[str] = set()
    policy = _clean(conflict_policy).lower() or "error"
    if policy not in {"error", "keep_first"}:
        raise ValueError(f"callable_surface_invalid_conflict_policy:{policy}")
    for item in descriptors:
        descriptor = item if isinstance(item, HostCapabilityDescriptor) else HostCapabilityDescriptor.from_dict(dict(item or {}))
        row = descriptor.to_dict()
        toolbox = dict(dict(row.get("metadata") or {}).get("toolbox") or {})
        if bool(toolbox.get("hidden_allowed", False)) and not include_hidden:
            continue
        if bool(toolbox.get("disabled", False)) and not include_disabled:
            continue
        if row["name"] in seen_names:
            if policy == "keep_first":
                continue
            raise ValueError(f"callable_surface_duplicate_name:{row['name']}")
        seen_names.add(row["name"])
        identity = callable_surface_identity(descriptor, session_id=session_id)
        digests = callable_surface_digests(descriptor)
        out.append(
            {
                "contract": HOST_CALLABLE_SCHEMA_CONTRACT,
                "name": row["name"],
                "namespace": row["namespace"],
                "group_path": list(row.get("group_path") or []),
                "description": row.get("description", ""),
                "arguments_schema": dict(row.get("args_schema") or {}),
                "result_schema": dict(row.get("result_schema") or {}),
                "permissions": list(row.get("permissions") or []),
                "scope_requirements": list(row.get("scope_requirements") or []),
                "approval": dict(row.get("approval") or {}),
                "provider": dict(row.get("provider") or {}),
                "identity": identity,
                "schema_digest": digests["schema_digest"],
                "method_digest": digests["method_digest"],
                "policy_digest": digests["policy_digest"],
                "metadata": dict(row.get("metadata") or {}),
            }
        )
    return out


def toolbox_to_callable_schemas(
    toolbox_description: Dict[str, Any],
    *,
    tools_view: Optional[Any] = None,
    provider_id: str = "",
    owner: str = "client",
    visibility: str = "workflow",
    namespace: str = "toolbox",
    session_id: str = "",
    include_hidden: bool = False,
    include_disabled: bool = False,
    conflict_policy: str = "error",
) -> list[Dict[str, Any]]:
    """Convert toolbox metadata directly to sandbox/model-facing callable schemas."""
    descriptors = toolbox_to_host_capability_descriptors(
        toolbox_description,
        tools_view=tools_view,
        provider_id=provider_id,
        owner=owner,
        visibility=visibility,
        namespace=namespace,
    )
    return host_capability_descriptors_to_callable_schemas(
        descriptors,
        include_hidden=include_hidden,
        include_disabled=include_disabled,
        conflict_policy=conflict_policy,
        session_id=session_id,
    )


def callable_surface_identity(
    descriptor: HostCapabilityDescriptor | Dict[str, Any],
    *,
    session_id: str = "",
    provider_id: str = "",
    toolbox_id: str = "",
    provider_kind: str = "",
) -> Dict[str, Any]:
    """Return the stable identity tuple for one advertised callable method."""
    row = descriptor.to_dict() if isinstance(descriptor, HostCapabilityDescriptor) else HostCapabilityDescriptor.from_dict(dict(descriptor or {})).to_dict()
    provider = dict(row.get("provider") or {})
    toolbox = dict(dict(row.get("metadata") or {}).get("toolbox") or {})
    sid = _clean(session_id) or _clean(row.get("session_id")) or _clean(toolbox.get("session_id"))
    pid = _clean(provider_id) or _clean(provider.get("provider_id"))
    tbid = _clean(toolbox_id) or _clean(toolbox.get("toolbox_id")) or pid
    return {
        "provider_kind": _clean(provider_kind) or _clean(provider.get("kind")) or "client_session",
        "provider_id": pid,
        "toolbox_id": tbid or None,
        "session_id": sid or None,
        "method": _clean(row.get("name")),
    }


def callable_surface_digests(descriptor: HostCapabilityDescriptor | Dict[str, Any]) -> Dict[str, str]:
    """Return stable schema/method/policy digests for approval and conflict decisions."""
    row = descriptor.to_dict() if isinstance(descriptor, HostCapabilityDescriptor) else HostCapabilityDescriptor.from_dict(dict(descriptor or {})).to_dict()
    schema_payload = {
        "arguments_schema": dict(row.get("args_schema") or {}),
        "result_schema": dict(row.get("result_schema") or {}),
    }
    policy_payload = {
        "permissions": list(row.get("permissions") or []),
        "scope_requirements": list(row.get("scope_requirements") or []),
        "approval": dict(row.get("approval") or {}),
        "constraints": dict(dict(row.get("metadata") or {}).get("constraints") or {}),
        "toolbox_constraints": dict(dict(dict(row.get("metadata") or {}).get("toolbox") or {}).get("constraints") or {}),
    }
    method_payload = {
        "name": _clean(row.get("name")),
        "namespace": _clean(row.get("namespace")),
        "schema": schema_payload,
        "policy": policy_payload,
    }
    return {
        "schema_digest": _digest(schema_payload),
        "method_digest": _digest(method_payload),
        "policy_digest": _digest(policy_payload),
    }


def toolbox_brokered_io_call_surface(
    method: str,
    *,
    arguments: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    toolbox_policy: Optional[Dict[str, Any]] = None,
    host_capability_policy: Optional[Dict[str, Any]] = None,
    bridge_policy: Optional[Dict[str, Any]] = None,
    provider_id: str = "",
    toolbox_id: str = "",
    session_id: str = "",
) -> Dict[str, Any]:
    """Describe one toolbox brokered-IO call using shared callable-surface metadata.

    Toolbox execution remains toolbox-native, while brokered IO routes through
    the shared service-broker Host Capability path. This surface shares the same
    identity, digest, correlation, and bridge-policy vocabulary used by Host
    Capability provider calls.
    """
    meth = _clean(method)
    if not meth:
        meth = "callback.invoke"
    try:
        from .sandbox.host_api import known_host_capability_method_descriptors

        known = {str(row.get("name") or ""): dict(row or {}) for row in known_host_capability_method_descriptors()}
    except Exception:
        known = {}
    descriptor_row = dict(known.get(meth) or {})
    if descriptor_row:
        descriptor_row["provider"] = HostCapabilityProviderRef(
            provider_id=_clean(provider_id) or _clean(toolbox_id) or "toolbox",
            kind="toolbox_session",
            owner="client",
            visibility="workflow",
        ).to_dict()
        metadata = dict(descriptor_row.get("metadata") or {})
        metadata["toolbox"] = {
            **dict(metadata.get("toolbox") or {}),
            "toolbox_id": _clean(toolbox_id) or _clean(provider_id) or "toolbox",
            "session_id": _clean(session_id) or None,
            "brokered_io": True,
        }
        descriptor_row["metadata"] = metadata
        descriptor = HostCapabilityDescriptor.from_dict(descriptor_row)
    else:
        if "." in meth:
            normalized = ".".join(_method_segment(part) for part in meth.split(".") if _method_segment(part))
        else:
            normalized = f"callback.{_method_segment(meth)}"
        namespace = normalized.split(".", 1)[0]
        descriptor = HostCapabilityDescriptor(
            name=normalized,
            namespace=namespace,
            group_path=default_group_path(normalized),
            description=f"Invoke toolbox brokered host method {meth}.",
            args_schema={"type": "object"},
            result_schema={"type": "object"},
            provider=HostCapabilityProviderRef(
                provider_id=_clean(provider_id) or _clean(toolbox_id) or "toolbox",
                kind="toolbox_session",
                owner="client",
                visibility="workflow",
            ),
            metadata={
                "toolbox": {
                    "toolbox_id": _clean(toolbox_id) or _clean(provider_id) or "toolbox",
                    "session_id": _clean(session_id) or None,
                    "brokered_io": True,
                }
            },
        )
    toolbox = dict(toolbox_policy or {})
    host_api = dict(host_capability_policy if host_capability_policy is not None else toolbox)
    bridge = dict(bridge_policy if bridge_policy is not None else toolbox)
    row = descriptor.to_dict()
    return {
        "contract": TOOLBOX_BROKERED_IO_CALL_SURFACE_CONTRACT,
        "method": meth,
        "namespace": row["namespace"],
        "argument_keys": sorted(str(key) for key in dict(arguments or {}).keys()),
        "identity": callable_surface_identity(descriptor, session_id=session_id, provider_id=provider_id, toolbox_id=toolbox_id, provider_kind="toolbox_session"),
        "digests": callable_surface_digests(descriptor),
        "bridge_policy": host_capability_bridge_policy(
            toolbox_policy=toolbox,
            host_capability_policy=host_api,
            bridge_policy=bridge,
        ),
        "correlation": extract_safe_correlation_metadata(context or {}, {"method": meth, "provider_id": provider_id, "toolbox_id": toolbox_id, "session_id": session_id}),
    }


def host_capability_bridge_policy(
    *,
    toolbox_policy: Optional[Dict[str, Any]] = None,
    host_capability_policy: Optional[Dict[str, Any]] = None,
    bridge_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the explicit sandbox-to-provider bridge policy intersection.

    A namespace is effectively allowed only when the explicit bridge policy and
    both endpoint policies allow it. Missing bridge entries are denied.
    """
    toolbox = _policy_payload(toolbox_policy)
    host_api = _policy_payload(host_capability_policy)
    bridge = _policy_payload(bridge_policy)
    namespace_names = sorted(
        {
            *_policy_namespace_names(toolbox),
            *_policy_namespace_names(host_api),
            *_policy_namespace_names(bridge),
        }
    )
    effective = {
        name: bool(_policy_namespace_allowed(bridge, name))
        and bool(_policy_namespace_allowed(toolbox, name))
        and bool(_policy_namespace_allowed(host_api, name))
        for name in namespace_names
    }
    return {
        "contract": HOST_CAPABILITY_BRIDGE_POLICY_CONTRACT,
        "mode": "explicit_intersection",
        "namespaces": effective,
        "inputs": {
            "toolbox": {"namespaces": {name: bool(_policy_namespace_allowed(toolbox, name)) for name in namespace_names}},
            "host_capability": {"namespaces": {name: bool(_policy_namespace_allowed(host_api, name)) for name in namespace_names}},
            "bridge": {"namespaces": {name: bool(_policy_namespace_allowed(bridge, name)) for name in namespace_names}},
        },
    }


def _policy_namespace_names(policy: Dict[str, Any]) -> set[str]:
    policy = _policy_payload(policy)
    names = {str(key or "").strip() for key in dict(policy.get("namespaces") or {}).keys()}
    names.update(str(key or "").strip() for key in ("fs", "http", "state", "subprocess") if key in policy)
    brokered = dict(policy.get("brokered_io") or {})
    names.update(str(key or "").strip() for key in ("http", "subprocess") if key in brokered)
    if "filesystem" in brokered:
        names.add("fs")
    return {name for name in names if name}


def _policy_namespace_allowed(policy: Dict[str, Any], namespace: str) -> bool:
    policy = _policy_payload(policy)
    ns = _clean(namespace)
    names = dict(policy.get("namespaces") or {})
    if ns in names:
        return bool(names.get(ns))
    if ns in policy:
        return bool(policy.get(ns))
    brokered = dict(policy.get("brokered_io") or {})
    if ns == "fs" and "filesystem" in brokered:
        return bool(brokered.get("filesystem"))
    if ns in brokered:
        return bool(brokered.get(ns))
    return False


def _policy_payload(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = dict(policy or {})
    return dict(row.get("sandbox") or row)


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


class HostCapabilityProviderCallbackRelay:
    """Local provider callback relay for client-owned Host Capability sessions."""

    def __init__(self) -> None:
        from .toolbox.callbacks import _HostedToolCallbackRelay

        self._relay = _HostedToolCallbackRelay()
        self._session_tokens: set[str] = set()

    def bind_callback(
        self,
        callback: Callable[..., Any],
        *,
        provider_id: str = "",
        method: str = "",
        user_context: Any = None,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        wrapped = bind_host_capability_provider_callback(callback)

        def _processor(*, callback_name: str, payload: Any, context: Any) -> Dict[str, Any]:
            name = _clean(callback_name)
            row = dict(payload or {}) if isinstance(payload, dict) else {}
            if name not in {HOST_CAPABILITY_PROVIDER_CALLBACK_NAME, HOST_CAPABILITY_CALL_CONTRACT}:
                return host_capability_provider_error(
                    _clean(row.get("provider_call_id")),
                    reason="host_capability_callback_unsupported",
                    message=f"unsupported callback {name}",
                )
            return wrapped(row)

        callback_binding = self._relay.bind_session(
            processor=_processor,
            toolbox_id=_clean(provider_id) or "host_capability",
            tool_name=_clean(method) or HOST_CAPABILITY_PROVIDER_CALLBACK_NAME,
            tool_call_id="",
            tool_arguments={},
            callback_signature=dict(callback_signature or {"contract": HOST_CAPABILITY_CALL_CONTRACT}),
            user_context=user_context,
        )
        token = _clean(callback_binding.get("session_token"))
        if token:
            self._session_tokens.add(token)
        return {
            "transport": "local_ipc",
            "callback_binding": callback_binding,
        }

    def release(self, binding: Dict[str, Any]) -> None:
        row = dict(binding or {})
        callback_binding = dict(row.get("callback_binding") or row)
        token = _clean(callback_binding.get("session_token"))
        self._relay.release_session(token)
        self._session_tokens.discard(token)

    def __enter__(self) -> "HostCapabilityProviderCallbackRelay":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for token in list(self._session_tokens):
            self._relay.release_session(token)
            self._session_tokens.discard(token)
        return None


def bind_host_capability_approval_callback(callback: Callable[..., Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Wrap a client approval callback so it consumes normalized approval requests."""

    def _invoke(envelope: Dict[str, Any]) -> Dict[str, Any]:
        request = host_capability_approval_request(dict(envelope or {}))
        try:
            if len(inspect.signature(callback).parameters) <= 1:
                result = callback(request)
            else:
                result = callback(request.get("method"), request)
            row = dict(result or {})
            if not _clean(row.get("decision")):
                row["decision"] = "allow_once" if bool(row.get("approved")) else "deny"
            decision = _clean(row.get("decision")).lower()
            if decision not in _APPROVAL_DECISIONS:
                decision = "deny"
            approved = decision in {"allow_once", "add_to_scope"}
            row["contract"] = _clean(row.get("contract")) or HOST_CAPABILITY_APPROVAL_DECISION_CONTRACT
            row["decision"] = decision
            row["approved"] = bool(row.get("approved", approved))
            row["status"] = _clean(row.get("status")) or ("approved" if row["approved"] else "denied")
            row["approval_id"] = _clean(row.get("approval_id")) or request.get("approval_id") or None
            row["scope_constraints"] = dict(row.get("scope_constraints") or row.get("constraints") or {})
            return row
        except TimeoutError as exc:
            return host_capability_approval_decision("deny", approval_id=str(request.get("approval_id") or ""), reason="approval_timeout", message=str(exc))
        except KeyboardInterrupt:
            return host_capability_approval_decision("deny", approval_id=str(request.get("approval_id") or ""), reason="approval_canceled", message="approval callback canceled")
        except Exception as exc:
            return host_capability_approval_decision(
                "deny",
                approval_id=str(request.get("approval_id") or ""),
                reason="approval_callback_error",
                message=str(exc),
            )

    return _invoke


class HostCapabilityApprovalCallbackRelay:
    """Local approval callback relay for daemon/control-channel workflow execution."""

    def __init__(self) -> None:
        from .toolbox.callbacks import _HostedToolCallbackRelay

        self._relay = _HostedToolCallbackRelay()
        self._session_tokens: set[str] = set()

    def bind_callback(
        self,
        callback: Callable[..., Any],
        *,
        provider_id: str = "",
        method: str = "",
        user_context: Any = None,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        wrapped = bind_host_capability_approval_callback(callback)

        def _processor(*, callback_name: str, payload: Any, context: Any) -> Dict[str, Any]:
            name = _clean(callback_name)
            row = dict(payload or {}) if isinstance(payload, dict) else {}
            if name not in {HOST_CAPABILITY_APPROVAL_CALLBACK_NAME, HOST_CAPABILITY_APPROVAL_CONTRACT}:
                return host_capability_approval_decision(
                    "deny",
                    approval_id=str(row.get("approval_id") or ""),
                    reason="host_capability_approval_callback_unsupported",
                    message=f"unsupported callback {name}",
                )
            return wrapped(row)

        callback_binding = self._relay.bind_session(
            processor=_processor,
            toolbox_id=_clean(provider_id) or "host_capability",
            tool_name=_clean(method) or HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
            tool_call_id="",
            tool_arguments={},
            callback_signature=dict(callback_signature or {"contract": HOST_CAPABILITY_APPROVAL_CONTRACT}),
            user_context=user_context,
        )
        token = _clean(callback_binding.get("session_token"))
        if token:
            self._session_tokens.add(token)
        return {
            "transport": "local_ipc",
            "callback_binding": callback_binding,
        }

    def release(self, binding: Dict[str, Any]) -> None:
        row = dict(binding or {})
        callback_binding = dict(row.get("callback_binding") or row)
        token = _clean(callback_binding.get("session_token"))
        self._relay.release_session(token)
        self._session_tokens.discard(token)

    def __enter__(self) -> "HostCapabilityApprovalCallbackRelay":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for token in list(self._session_tokens):
            self._relay.release_session(token)
            self._session_tokens.discard(token)
        return None


def host_capability_approval_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(payload or {})
    arguments = dict(row.get("arguments") or {})
    argument_preview = (
        dict(row.get("argument_preview") or {})
        if isinstance(row.get("argument_preview"), dict)
        else build_argument_preview(arguments)
    )
    argument_keys = (
        sorted(_clean(key) for key in arguments.keys() if _clean(key))
        if arguments
        else sorted(_clean(key) for key in list(row.get("argument_keys") or []) if _clean(key))
    )
    context = dict(row.get("context") or {})
    descriptor = row.get("descriptor")
    identity = dict(row.get("identity") or {})
    digests = dict(row.get("digests") or {})
    if descriptor and isinstance(descriptor, (HostCapabilityDescriptor, dict)):
        identity = {**callable_surface_identity(descriptor, session_id=_clean(row.get("session_id"))), **identity}
        digests = {**callable_surface_digests(descriptor), **digests}
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
            for key in (
                "workflow_id",
                "instance_id",
                "request_id",
                "package_id",
                "actor",
                "node_id",
                "cursor_id",
                "context_id",
                "branch_id",
                "session_tree_id",
                "session_id",
                "toolbox_id",
            )
            if context.get(key) is not None
        },
        "identity": identity,
        "digests": digests,
        "argument_keys": argument_keys,
        "argument_preview": argument_preview,
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
    "HOST_CAPABILITY_APPROVAL_CALLBACK_NAME",
    "HOST_CAPABILITY_APPROVAL_DECISION_CONTRACT",
    "HOST_CAPABILITY_BRIDGE_POLICY_CONTRACT",
    "HOST_CAPABILITY_PROVIDER_CALLBACK_NAME",
    "HOST_CAPABILITY_DISPATCH_CALLBACK_NAME",
    "HOST_CAPABILITY_PROVIDER_RESPONSE_CONTRACT",
    "TOOLBOX_BROKERED_IO_CALL_SURFACE_CONTRACT",
    "HostCapabilityApprovalCallbackRelay",
    "HostCapabilityProviderCallbackRelay",
    "SAFE_CORRELATION_FIELDS",
    "bind_host_capability_approval_callback",
    "bind_host_capability_provider_callback",
    "callable_surface_digests",
    "callable_surface_identity",
    "extract_safe_correlation_metadata",
    "host_capability_approval_decision",
    "host_capability_approval_request",
    "host_capability_bridge_policy",
    "host_capability_descriptors_to_callable_schemas",
    "host_capability_provider_error",
    "host_capability_provider_success",
    "normalize_host_capability_provider_response",
    "toolbox_brokered_io_call_surface",
    "toolbox_to_callable_schemas",
    "toolbox_to_host_capability_descriptors",
]
