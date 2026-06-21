"""Shared host capability descriptor and broker primitives."""
from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional


HOST_CAPABILITY_CONTRACT = "hosting.sandbox.host_capability.v1"
HOST_CAPABILITY_SESSION_CONTRACT = "hosting.sandbox.host_capability_session.v1"
HOST_CAPABILITY_DISCOVERY_CONTRACT = "hosting.sandbox.discovery.v1"

_METHOD_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_PROVIDER_KINDS = {"builtin", "client_session", "toolbox_session"}
_ALLOWED_VISIBILITY = {"request", "workflow", "instance", "consumer"}
_MAX_SCHEMA_CHARS = 65536
_MAX_DESCRIPTION_CHARS = 4096

CapabilityHandler = Callable[[Dict[str, Any]], Dict[str, Any]]
AsyncCapabilityHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _jsonish_size(value: Any) -> int:
    try:
        import json

        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    except Exception:
        return len(str(value))


def _string_list(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = _clean(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def default_group_path(name: str) -> list[str]:
    parts = [_clean(part) for part in _clean(name).split(".") if _clean(part)]
    if len(parts) <= 1:
        return [parts[0].upper()] if parts else ["Host"]
    return [part.replace("_", " ").title() for part in parts[:-1]]


@dataclass(frozen=True)
class HostCapabilityApproval:
    mode: str = "none"
    cache_key: str = "method+scope+actor"
    ttl_seconds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": _clean(self.mode) or "none",
            "cache_key": _clean(self.cache_key) or "method+scope+actor",
            "ttl_seconds": max(0, int(self.ttl_seconds or 0)),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "HostCapabilityApproval":
        row = dict(payload or {})
        return cls(
            mode=_clean(row.get("mode")) or "none",
            cache_key=_clean(row.get("cache_key")) or "method+scope+actor",
            ttl_seconds=max(0, int(row.get("ttl_seconds") or 0)),
        )


@dataclass(frozen=True)
class HostCapabilityProviderRef:
    provider_id: str
    kind: str = "builtin"
    owner: str = "service"
    visibility: str = "request"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": _clean(self.provider_id),
            "kind": _clean(self.kind) or "builtin",
            "owner": _clean(self.owner) or "service",
            "visibility": _clean(self.visibility) or "request",
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "HostCapabilityProviderRef":
        row = dict(payload or {})
        return cls(
            provider_id=_clean(row.get("provider_id")),
            kind=_clean(row.get("kind")) or "builtin",
            owner=_clean(row.get("owner")) or "service",
            visibility=_clean(row.get("visibility")) or "request",
        )


@dataclass(frozen=True)
class HostCapabilityDescriptor:
    name: str
    namespace: str
    group_path: list[str]
    description: str = ""
    args_schema: Dict[str, Any] = field(default_factory=dict)
    result_schema: Dict[str, Any] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)
    scope_requirements: list[Dict[str, Any]] = field(default_factory=list)
    approval: HostCapabilityApproval = field(default_factory=HostCapabilityApproval)
    provider: HostCapabilityProviderRef = field(default_factory=lambda: HostCapabilityProviderRef(provider_id="builtin"))
    contract: str = HOST_CAPABILITY_CONTRACT

    def validate(self) -> None:
        name = _clean(self.name)
        namespace = _clean(self.namespace)
        if not name:
            raise ValueError("host_capability_name_required")
        if not _METHOD_RE.match(name):
            raise ValueError(f"host_capability_invalid_name:{name}")
        if not namespace or not _NAMESPACE_RE.match(namespace):
            raise ValueError(f"host_capability_invalid_namespace:{namespace}")
        if namespace != name.split(".", 1)[0]:
            raise ValueError("host_capability_namespace_mismatch")
        groups = [_clean(item) for item in list(self.group_path or [])]
        if not groups or any(not item for item in groups):
            raise ValueError("host_capability_invalid_group_path")
        if len(_clean(self.description)) > _MAX_DESCRIPTION_CHARS:
            raise ValueError("host_capability_description_too_large")
        if _jsonish_size(self.args_schema) > _MAX_SCHEMA_CHARS or _jsonish_size(self.result_schema) > _MAX_SCHEMA_CHARS:
            raise ValueError("host_capability_schema_too_large")
        provider = self.provider.to_dict()
        if provider["kind"] not in _ALLOWED_PROVIDER_KINDS:
            raise ValueError(f"host_capability_invalid_provider_kind:{provider['kind']}")
        if not provider["provider_id"]:
            raise ValueError("host_capability_provider_id_required")
        if provider["visibility"] not in _ALLOWED_VISIBILITY:
            raise ValueError(f"host_capability_invalid_visibility:{provider['visibility']}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "contract": self.contract,
            "name": _clean(self.name),
            "namespace": _clean(self.namespace),
            "group_path": [_clean(item) for item in self.group_path],
            "description": _clean(self.description),
            "args_schema": dict(self.args_schema or {}),
            "result_schema": dict(self.result_schema or {}),
            "permissions": _string_list(self.permissions),
            "scope_requirements": [dict(item or {}) for item in list(self.scope_requirements or [])],
            "approval": self.approval.to_dict(),
            "provider": self.provider.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HostCapabilityDescriptor":
        row = dict(payload or {})
        descriptor = cls(
            contract=_clean(row.get("contract")) or HOST_CAPABILITY_CONTRACT,
            name=_clean(row.get("name")),
            namespace=_clean(row.get("namespace")),
            group_path=[_clean(item) for item in list(row.get("group_path") or [])],
            description=_clean(row.get("description")),
            args_schema=dict(row.get("args_schema") or {}),
            result_schema=dict(row.get("result_schema") or {}),
            permissions=_string_list(row.get("permissions") or []),
            scope_requirements=[dict(item or {}) for item in list(row.get("scope_requirements") or [])],
            approval=HostCapabilityApproval.from_dict(row.get("approval")),
            provider=HostCapabilityProviderRef.from_dict(row.get("provider")),
        )
        descriptor.validate()
        return descriptor


@dataclass
class HostCapabilityMethod:
    descriptor: HostCapabilityDescriptor
    handler: Optional[CapabilityHandler] = None
    async_handler: Optional[AsyncCapabilityHandler] = None

    async def dispatch_async(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.async_handler is not None:
            return dict(await self.async_handler(dict(arguments or {})) or {})
        if self.handler is None:
            raise RuntimeError(f"host_capability_handler_missing:{self.descriptor.name}")
        result = self.handler(dict(arguments or {}))
        if inspect.isawaitable(result):
            return dict(await result or {})
        return dict(result or {})


@dataclass
class HostCapabilitySession:
    session_id: str
    owner: str
    provider_kind: str = "builtin"
    visibility: str = "request"
    scope: Dict[str, Any] = field(default_factory=dict)
    methods: Dict[str, HostCapabilityMethod] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    expires_at_ms: Optional[int] = None
    close_on_client_disconnect: bool = True

    def to_private_dict(self) -> Dict[str, Any]:
        return {
            "contract": HOST_CAPABILITY_SESSION_CONTRACT,
            "session_id": _clean(self.session_id),
            "owner": _clean(self.owner),
            "scope": dict(self.scope or {}),
            "methods": [method.descriptor.to_dict() for method in self.methods.values()],
            "provider": {
                "kind": _clean(self.provider_kind) or "builtin",
                "visibility": _clean(self.visibility) or "request",
            },
            "lifetime": {
                "created_at_ms": int(self.created_at_ms or 0),
                "expires_at_ms": self.expires_at_ms,
                "close_on_client_disconnect": bool(self.close_on_client_disconnect),
            },
        }


class HostCapabilityBroker:
    """Request-scoped capability broker.

    This broker is intentionally small for the first slice: it wraps built-in
    providers and provides the stable describe/dispatch surface that later
    client-owned provider sessions will plug into.
    """

    def __init__(
        self,
        *,
        request_id: str = "",
        workflow_id: str = "",
        package_id: str = "",
        runtime_kind: str = "",
        policy: Optional[Dict[str, Any]] = None,
        roots: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.request_id = _clean(request_id)
        self.workflow_id = _clean(workflow_id)
        self.package_id = _clean(package_id)
        self.runtime_kind = _clean(runtime_kind)
        self.policy = dict(policy or {})
        self.roots = dict(roots or {})
        self._sessions: Dict[str, HostCapabilitySession] = {}

    def register_session(self, session: HostCapabilitySession) -> None:
        sid = _clean(session.session_id)
        if not sid:
            raise ValueError("host_capability_session_id_required")
        for name, method in dict(session.methods or {}).items():
            if name != method.descriptor.name:
                raise ValueError("host_capability_method_name_mismatch")
            method.descriptor.validate()
        self._sessions[sid] = session

    def register_builtin_provider(
        self,
        *,
        provider_id: str = "builtin.host_api",
        owner: str = "service",
        methods: Iterable[HostCapabilityMethod],
    ) -> HostCapabilitySession:
        session = HostCapabilitySession(
            session_id=_clean(provider_id) or f"builtin.{uuid.uuid4().hex}",
            owner=_clean(owner) or "service",
            provider_kind="builtin",
            visibility="request",
            scope={"request_id": self.request_id or None, "workflow_id": self.workflow_id or None, "package_id": self.package_id or None},
            methods={method.descriptor.name: method for method in methods},
        )
        self.register_session(session)
        return session

    def descriptors(self) -> list[HostCapabilityDescriptor]:
        rows: list[HostCapabilityDescriptor] = []
        for session in self._sessions.values():
            rows.extend(method.descriptor for method in session.methods.values())
        return sorted(rows, key=lambda item: item.name)

    def method_names(self) -> list[str]:
        return [item.name for item in self.descriptors()]

    def groups(self) -> list[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for descriptor in self.descriptors():
            path = list(descriptor.group_path or [])
            key = "/".join(path)
            groups.setdefault(key, {"path": path, "methods": []})
            groups[key]["methods"].append(descriptor.name)
        return [groups[key] for key in sorted(groups.keys())]

    def providers_for_discovery(self) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        for session in sorted(self._sessions.values(), key=lambda item: item.session_id):
            out.append(
                {
                    "provider_id": session.session_id,
                    "kind": session.provider_kind,
                    "owner": session.owner,
                    "visibility": session.visibility,
                    "method_count": len(session.methods),
                }
            )
        return out

    def describe_host_capabilities(self) -> Dict[str, Any]:
        return {
            "methods": [descriptor.to_dict() for descriptor in self.descriptors()],
            "groups": self.groups(),
            "providers": self.providers_for_discovery(),
            "transport": {
                "framed": True,
                "host_call_id": True,
                "async_capable": True,
                "out_of_order_responses": True,
            },
        }

    def describe(self) -> Dict[str, Any]:
        host_capabilities = self.describe_host_capabilities()
        method_descriptions = [
            {
                "name": descriptor["name"],
                "namespace": descriptor["namespace"],
                "description": descriptor.get("description", ""),
                "args_schema": dict(descriptor.get("args_schema") or {}),
                "result_schema": dict(descriptor.get("result_schema") or {}),
                "permissions": list(descriptor.get("permissions") or []),
                "async": False,
                "group_path": list(descriptor.get("group_path") or []),
                "provider": dict(descriptor.get("provider") or {}),
            }
            for descriptor in list(host_capabilities.get("methods") or [])
        ]
        return {
            "status": "ok",
            "contract": HOST_CAPABILITY_DISCOVERY_CONTRACT,
            "request_id": self.request_id,
            "methods": self.method_names(),
            "method_descriptions": method_descriptions,
            "transport": dict(host_capabilities.get("transport") or {}),
            "runtime": {
                "runtime_kind": self.runtime_kind or None,
                "worker_contract": None,
            },
            "harness": {
                "host_api_entrypoints": ["host.call", "host.describe", "sandbox.describe"],
            },
            "events": {
                "worker_live": ["progress"],
                "host_generated": ["started", "heartbeat", "stdout", "stderr", "log", "artifact", "result", "error", "canceled", "done"],
                "observations": ["host_call", "host_response"],
                "reserved": ["approval", "state_notice", "action_notice"],
            },
            "host_capabilities": host_capabilities,
            "state": {"available": False, "scopes": []},
            "actions": {"available": False, "entries": []},
            "policy": dict(self.policy or {}),
            "roots": dict(self.roots or {}),
        }

    async def dispatch_async(self, call: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(call or {})
        method_name = _clean(row.get("method"))
        if method_name in {"sandbox.describe", "host.describe"}:
            return self.describe()
        for session in self._sessions.values():
            method = session.methods.get(method_name)
            if method is not None:
                return await method.dispatch_async(dict(row.get("arguments") or {}))
        raise RuntimeError(f"unsupported_host_method:{method_name}")

    def dispatch(self, call: Dict[str, Any]) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.dispatch_async(call))
        method_name = _clean(dict(call or {}).get("method"))
        if method_name in {"sandbox.describe", "host.describe"}:
            return self.describe()
        for session in self._sessions.values():
            method = session.methods.get(method_name)
            if method is not None and method.async_handler is None and method.handler is not None:
                return dict(method.handler(dict(dict(call or {}).get("arguments") or {})) or {})
        raise RuntimeError("async_host_capability_dispatch_requires_await")


__all__ = [
    "HOST_CAPABILITY_CONTRACT",
    "HOST_CAPABILITY_DISCOVERY_CONTRACT",
    "HOST_CAPABILITY_SESSION_CONTRACT",
    "AsyncCapabilityHandler",
    "CapabilityHandler",
    "HostCapabilityApproval",
    "HostCapabilityBroker",
    "HostCapabilityDescriptor",
    "HostCapabilityMethod",
    "HostCapabilityProviderRef",
    "HostCapabilitySession",
    "default_group_path",
]
