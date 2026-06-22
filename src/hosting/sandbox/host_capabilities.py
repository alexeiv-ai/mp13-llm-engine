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
HOST_CAPABILITY_CALL_CONTRACT = "hosting.sandbox.host_capability_call.v1"
HOST_CAPABILITY_APPROVAL_CONTRACT = "hosting.sandbox.host_capability_approval.v1"

_METHOD_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_PROVIDER_KINDS = {"builtin", "client_session", "toolbox_session"}
_ALLOWED_VISIBILITY = {"request", "workflow", "instance", "consumer"}
_MAX_SCHEMA_CHARS = 65536
_MAX_DESCRIPTION_CHARS = 4096

CapabilityHandler = Callable[[Dict[str, Any]], Dict[str, Any]]
AsyncCapabilityHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
ProviderInvoker = Callable[["HostCapabilitySession", "HostCapabilityProviderCall"], Awaitable[Dict[str, Any]] | Dict[str, Any]]
CancelChecker = Callable[[], bool]
ApprovalRequester = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]] | Dict[str, Any]]
EventEmitter = Callable[[str, Dict[str, Any]], None]
AuditEmitter = Callable[[Dict[str, Any]], None]


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
    metadata: Dict[str, Any] = field(default_factory=dict)
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
        if _jsonish_size(self.metadata) > _MAX_SCHEMA_CHARS:
            raise ValueError("host_capability_metadata_too_large")
        provider = self.provider.to_dict()
        if provider["kind"] not in _ALLOWED_PROVIDER_KINDS:
            raise ValueError(f"host_capability_invalid_provider_kind:{provider['kind']}")
        if not provider["provider_id"]:
            raise ValueError("host_capability_provider_id_required")
        if provider["visibility"] not in _ALLOWED_VISIBILITY:
            raise ValueError(f"host_capability_invalid_visibility:{provider['visibility']}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        out = {
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
        if dict(self.metadata or {}):
            out["metadata"] = dict(self.metadata or {})
        return out

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
            metadata=dict(row.get("metadata") or {}),
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


class HostCapabilityProviderError(RuntimeError):
    def __init__(self, reason: str, message: str = "", detail: Optional[Dict[str, Any]] = None) -> None:
        self.reason = _clean(reason) or "host_capability_provider_error"
        self.message = _clean(message)
        self.detail = dict(detail or {})
        super().__init__(self.reason if not self.message else f"{self.reason}:{self.message}")


class HostCapabilityProviderUnavailable(HostCapabilityProviderError):
    def __init__(self, message: str = "host capability provider is unavailable", detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("host_capability_provider_unavailable", message, detail)


class HostCapabilityTimeout(HostCapabilityProviderError):
    def __init__(self, message: str = "host capability provider call timed out", detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("host_call_timeout", message, detail)


class HostCapabilityCanceled(HostCapabilityProviderError):
    def __init__(self, message: str = "host capability provider call canceled", detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("host_call_canceled", message, detail)


class HostCapabilityApprovalDenied(HostCapabilityProviderError):
    def __init__(self, message: str = "host capability call approval denied", detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("host_call_approval_denied", message, detail)


@dataclass(frozen=True)
class HostCapabilityCallContext:
    request_id: str = ""
    instance_id: Optional[str] = None
    workflow_id: str = ""
    package_id: str = ""
    actor: str = ""
    deadline_ms: Optional[int] = None
    permissions: list[str] = field(default_factory=list)
    approved_scopes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": _clean(self.request_id) or None,
            "instance_id": _clean(self.instance_id) or None,
            "workflow_id": _clean(self.workflow_id) or None,
            "package_id": _clean(self.package_id) or None,
            "actor": _clean(self.actor) or None,
            "deadline_ms": int(self.deadline_ms) if self.deadline_ms is not None else None,
            "permissions": _string_list(self.permissions),
            "approved_scopes": _string_list(self.approved_scopes),
        }


@dataclass(frozen=True)
class HostCapabilityProviderCall:
    provider_call_id: str
    method: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    context: HostCapabilityCallContext = field(default_factory=HostCapabilityCallContext)
    contract: str = HOST_CAPABILITY_CALL_CONTRACT

    def to_dict(self) -> Dict[str, Any]:
        provider_call_id = _clean(self.provider_call_id)
        method = _clean(self.method)
        if not provider_call_id:
            raise ValueError("host_capability_provider_call_id_required")
        if not method:
            raise ValueError("host_capability_provider_call_method_required")
        return {
            "contract": self.contract,
            "provider_call_id": provider_call_id,
            "method": method,
            "arguments": dict(self.arguments or {}),
            "context": self.context.to_dict(),
        }


def validate_provider_response(payload: Dict[str, Any], *, provider_call_id: str) -> Dict[str, Any]:
    row = dict(payload or {})
    expected = _clean(provider_call_id)
    got = _clean(row.get("provider_call_id"))
    if not expected:
        raise ValueError("host_capability_provider_call_id_required")
    if got != expected:
        raise ValueError("host_capability_provider_call_id_mismatch")
    status = _clean(row.get("status")).lower()
    if status == "ok":
        return dict(row.get("result") or {})
    if status == "error":
        raise HostCapabilityProviderError(
            reason=_clean(row.get("reason")) or "host_capability_provider_error",
            message=_clean(row.get("message")),
            detail=dict(row.get("detail") or {}),
        )
    raise ValueError(f"host_capability_invalid_provider_response_status:{status}")


@dataclass
class HostCapabilitySession:
    session_id: str
    owner: str
    provider_kind: str = "builtin"
    visibility: str = "request"
    scope: Dict[str, Any] = field(default_factory=dict)
    methods: Dict[str, HostCapabilityMethod] = field(default_factory=dict)
    binding: Dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    expires_at_ms: Optional[int] = None
    close_on_client_disconnect: bool = True
    allow_override: bool = False

    def to_public_dict(self) -> Dict[str, Any]:
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
            "override": {"allow": bool(self.allow_override)},
        }

    def to_private_dict(self) -> Dict[str, Any]:
        out = self.to_public_dict()
        out["binding"] = dict(self.binding or {})
        return out


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
        provider_invoker: Optional[ProviderInvoker] = None,
        provider_timeout_seconds: float = 30.0,
        cancel_checker: Optional[CancelChecker] = None,
        instance_id: str = "",
        consumer_id: str = "",
        allowed_namespaces: Optional[Any] = None,
        approved_permissions: Optional[Iterable[str]] = None,
        approval_requester: Optional[ApprovalRequester] = None,
        event_emitter: Optional[EventEmitter] = None,
        audit_emitter: Optional[AuditEmitter] = None,
    ) -> None:
        self.request_id = _clean(request_id)
        self.workflow_id = _clean(workflow_id)
        self.package_id = _clean(package_id)
        self.instance_id = _clean(instance_id)
        self.consumer_id = _clean(consumer_id)
        self.runtime_kind = _clean(runtime_kind)
        self.policy = dict(policy or {})
        self.roots = dict(roots or {})
        self.provider_invoker = provider_invoker
        self.provider_timeout_seconds = max(0.001, float(provider_timeout_seconds or 30.0))
        self.cancel_checker = cancel_checker
        self._cancel_requested = False
        self._cancel_reason = "host_call_canceled"
        self._allowed_namespaces = self._normalize_allowed_namespaces(allowed_namespaces)
        self._approved_permissions = set(_string_list(approved_permissions or [])) if approved_permissions is not None else None
        self.approval_requester = approval_requester
        self.event_emitter = event_emitter
        self.audit_emitter = audit_emitter
        self._sessions: Dict[str, HostCapabilitySession] = {}

    def cancel(self, reason: str = "host_call_canceled") -> None:
        self._cancel_requested = True
        self._cancel_reason = _clean(reason) or "host_call_canceled"

    def _check_canceled(self) -> None:
        if self._cancel_requested or (self.cancel_checker is not None and bool(self.cancel_checker())):
            raise HostCapabilityCanceled(detail={"reason": self._cancel_reason})

    def _emit_event(self, kind: str, payload: Dict[str, Any]) -> None:
        if self.event_emitter is None:
            return
        try:
            self.event_emitter(_clean(kind), dict(payload or {}))
        except Exception:
            return

    def _emit_audit(self, payload: Dict[str, Any]) -> None:
        if self.audit_emitter is None:
            return
        self.audit_emitter(dict(payload or {}))

    def _provider_timeout_for_call(self, row: Dict[str, Any]) -> float:
        raw = row.get("provider_timeout_seconds")
        if raw is None:
            raw = row.get("timeout_seconds")
        if raw is None:
            args = dict(row.get("arguments") or {})
            raw = args.get("provider_timeout_seconds")
        if raw is None:
            return self.provider_timeout_seconds
        return max(0.001, float(raw or self.provider_timeout_seconds))

    @staticmethod
    def _normalize_allowed_namespaces(raw: Optional[Any]) -> Optional[set[str]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return {_clean(key) for key, value in raw.items() if _clean(key) and bool(value)}
        return set(_string_list(raw or []))

    @staticmethod
    def _visibility_rank(session: HostCapabilitySession) -> int:
        return {"request": 1, "instance": 2, "workflow": 3, "consumer": 4}.get(_clean(session.visibility), 9)

    def _session_visible(self, session: HostCapabilitySession) -> bool:
        scope = dict(session.scope or {})
        visibility = _clean(session.visibility)
        if visibility == "request":
            return bool(_clean(scope.get("request_id")) and _clean(scope.get("request_id")) == self.request_id)
        if visibility == "workflow":
            return bool(_clean(scope.get("workflow_id")) and _clean(scope.get("workflow_id")) == self.workflow_id)
        if visibility == "instance":
            return bool(_clean(scope.get("instance_id")) and _clean(scope.get("instance_id")) == self.instance_id)
        if visibility == "consumer":
            return bool(_clean(scope.get("consumer_id")) and _clean(scope.get("consumer_id")) == self.consumer_id)
        return False

    def _method_allowed(self, method: HostCapabilityMethod) -> bool:
        descriptor = method.descriptor
        if self._allowed_namespaces is not None and _clean(descriptor.namespace) not in self._allowed_namespaces:
            return False
        if self._approved_permissions is not None:
            required = set(_string_list(descriptor.permissions or []))
            if not required.issubset(self._approved_permissions):
                return False
        return True

    def _resolved_methods(self) -> Dict[str, tuple[HostCapabilitySession, HostCapabilityMethod]]:
        candidates: Dict[str, list[tuple[HostCapabilitySession, HostCapabilityMethod]]] = {}
        for session in self._sessions.values():
            if not self._session_visible(session):
                continue
            for method in session.methods.values():
                if self._method_allowed(method):
                    candidates.setdefault(method.descriptor.name, []).append((session, method))
        out: Dict[str, tuple[HostCapabilitySession, HostCapabilityMethod]] = {}
        for name, rows in candidates.items():
            rows.sort(key=lambda item: (0 if bool(item[0].allow_override) else 1, self._visibility_rank(item[0]), _clean(item[0].session_id)))
            out[name] = rows[0]
        return out

    async def _await_provider_response(self, response: Awaitable[Dict[str, Any]], *, timeout_seconds: float) -> Dict[str, Any]:
        task = asyncio.ensure_future(response)
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                self._check_canceled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    task.cancel()
                    raise HostCapabilityTimeout(detail={"timeout_seconds": timeout_seconds})
                done, _pending = await asyncio.wait({task}, timeout=min(remaining, 0.05))
                if done:
                    return dict(await task or {})
        except HostCapabilityCanceled:
            task.cancel()
            raise

    @staticmethod
    def _approval_required(method: HostCapabilityMethod) -> bool:
        mode = _clean(method.descriptor.approval.mode).lower()
        return bool(mode and mode != "none")

    async def _request_approval(
        self,
        *,
        session: HostCapabilitySession,
        method: HostCapabilityMethod,
        provider_call: HostCapabilityProviderCall,
        host_call_id: str = "",
    ) -> None:
        if not self._approval_required(method):
            return
        approval_id = f"cap_approval_{uuid.uuid4().hex}"
        call_id = _clean(host_call_id) or provider_call.provider_call_id
        audit_base = {
            "event_type": "host_capability_approval",
            "approval_id": approval_id,
            "call_id": call_id,
            "host_call_id": _clean(host_call_id) or None,
            "provider_call_id": provider_call.provider_call_id,
            "method": provider_call.method,
            "argument_keys": sorted(str(key) for key in dict(provider_call.arguments or {}).keys()),
            "context": provider_call.context.to_dict(),
            "approval": method.descriptor.approval.to_dict(),
            "provider": {
                "provider_id": session.session_id,
                "kind": session.provider_kind,
                "owner": session.owner,
                "visibility": session.visibility,
            },
        }
        if self.approval_requester is None:
            self._emit_audit({**audit_base, "result": "denied", "reason": "approval_requester_unavailable"})
            raise HostCapabilityApprovalDenied(
                detail={
                    "approval_id": approval_id,
                    "provider_call_id": provider_call.provider_call_id,
                    "method": provider_call.method,
                    "reason": "approval_requester_unavailable",
                }
            )
        request = {
            "contract": HOST_CAPABILITY_APPROVAL_CONTRACT,
            "approval_id": approval_id,
            "provider_call_id": provider_call.provider_call_id,
            "method": provider_call.method,
            "arguments": dict(provider_call.arguments or {}),
            "context": provider_call.context.to_dict(),
            "approval": method.descriptor.approval.to_dict(),
            "provider": {
                "provider_id": session.session_id,
                "kind": session.provider_kind,
                "owner": session.owner,
                "visibility": session.visibility,
            },
        }
        self._emit_event("approval", {"status": "requested", "call_id": call_id, "host_call_id": _clean(host_call_id) or None, **request})
        decision = self.approval_requester(request)
        if inspect.isawaitable(decision):
            decision = await decision
        row = dict(decision or {})
        status = _clean(row.get("status")).lower()
        approved = bool(row.get("approved", status in {"ok", "approved"}))
        if not approved or status in {"denied", "rejected", "error"}:
            self._emit_audit(
                {
                    **audit_base,
                    "result": "denied",
                    "reason": _clean(row.get("reason") or row.get("message")) or "approval_denied",
                    "decision": row,
                }
            )
            self._emit_event(
                "approval",
                {
                    "status": "denied",
                    "call_id": call_id,
                    "host_call_id": _clean(host_call_id) or None,
                    "approval_id": approval_id,
                    "provider_call_id": provider_call.provider_call_id,
                    "method": provider_call.method,
                    "decision": row,
                },
            )
            raise HostCapabilityApprovalDenied(
                message=_clean(row.get("message")) or "host capability call approval denied",
                detail={
                    "approval_id": approval_id,
                    "provider_call_id": provider_call.provider_call_id,
                    "method": provider_call.method,
                    "decision": row,
                },
            )
        self._emit_audit({**audit_base, "result": "approved", "reason": None, "decision": row})
        self._emit_event(
            "approval",
            {
                "status": "approved",
                "call_id": call_id,
                "host_call_id": _clean(host_call_id) or None,
                "approval_id": approval_id,
                "provider_call_id": provider_call.provider_call_id,
                "method": provider_call.method,
                "decision": row,
            },
        )

    def register_session(self, session: HostCapabilitySession) -> None:
        sid = _clean(session.session_id)
        if not sid:
            raise ValueError("host_capability_session_id_required")
        for name, method in dict(session.methods or {}).items():
            if name != method.descriptor.name:
                raise ValueError("host_capability_method_name_mismatch")
            method.descriptor.validate()
        incoming_names = set(str(name or "").strip() for name in dict(session.methods or {}).keys() if str(name or "").strip())
        if incoming_names and not bool(session.allow_override):
            for existing in self._sessions.values():
                duplicates = sorted(incoming_names.intersection(dict(existing.methods or {}).keys()))
                if duplicates:
                    raise ValueError(f"host_capability_duplicate_method:{duplicates[0]}")
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
            allow_override=False,
        )
        self.register_session(session)
        return session

    def descriptors(self) -> list[HostCapabilityDescriptor]:
        return [method.descriptor for _session, method in sorted(self._resolved_methods().values(), key=lambda item: item[1].descriptor.name)]

    def method_names(self) -> list[str]:
        return sorted({"host.describe", "sandbox.describe", *[item.name for item in self.descriptors()]})

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
        visible_methods = self._resolved_methods()
        for session in sorted(self._sessions.values(), key=lambda item: item.session_id):
            method_count = sum(1 for visible_session, _method in visible_methods.values() if visible_session.session_id == session.session_id)
            if method_count <= 0:
                continue
            out.append(
                {
                    "provider_id": session.session_id,
                    "kind": session.provider_kind,
                    "owner": session.owner,
                    "visibility": session.visibility,
                    "method_count": method_count,
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
        resolved = self._resolved_methods().get(method_name)
        if resolved is not None:
            session, method = resolved
            if method is not None:
                if method.handler is None and method.async_handler is None:
                    self._check_canceled()
                    if self.provider_invoker is None:
                        raise HostCapabilityProviderUnavailable(detail={"provider_id": session.session_id})
                    provider_call_id = f"cap_call_{uuid.uuid4().hex}"
                    host_call_id = _clean(row.get("host_call_id") or row.get("call_id"))
                    event_call_id = host_call_id or provider_call_id
                    timeout_seconds = self._provider_timeout_for_call(row)
                    call = HostCapabilityProviderCall(
                        provider_call_id=provider_call_id,
                        method=method_name,
                        arguments=dict(row.get("arguments") or {}),
                        context=HostCapabilityCallContext(
                            request_id=self.request_id,
                            workflow_id=self.workflow_id,
                            package_id=self.package_id,
                            actor=session.owner,
                            permissions=list(method.descriptor.permissions or []),
                            approved_scopes=[
                                f"{scope.get('scope')}:{scope.get('access')}"
                                for scope in list(method.descriptor.scope_requirements or [])
                                if scope.get("scope") and scope.get("access")
                            ],
                            deadline_ms=int((time.time() + timeout_seconds) * 1000),
                        ),
                    )
                    self._emit_event(
                        "host_call",
                        {
                            "method": method_name,
                            "call_id": event_call_id,
                            "host_call_id": host_call_id or None,
                            "provider_call_id": provider_call_id,
                            "provider_id": session.session_id,
                            "provider_kind": session.provider_kind,
                            "request_id": self.request_id or None,
                            "workflow_id": self.workflow_id or None,
                            "instance_id": self.instance_id or None,
                        },
                    )
                    try:
                        await self._request_approval(session=session, method=method, provider_call=call, host_call_id=host_call_id)
                        response = self.provider_invoker(session, call)
                        if inspect.isawaitable(response):
                            response = await self._await_provider_response(response, timeout_seconds=timeout_seconds)
                        self._check_canceled()
                        result = validate_provider_response(dict(response or {}), provider_call_id=provider_call_id)
                        self._emit_event(
                            "host_response",
                            {
                                "status": "ok",
                                "method": method_name,
                                "call_id": event_call_id,
                                "host_call_id": host_call_id or None,
                                "provider_call_id": provider_call_id,
                                "provider_id": session.session_id,
                            },
                        )
                        return result
                    except HostCapabilityTimeout as exc:
                        exc.detail.setdefault("provider_call_id", provider_call_id)
                        self._emit_event("provider_failure", {"method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": exc.reason, "detail": dict(exc.detail or {})})
                        self._emit_event("host_response", {"status": "error", "method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": exc.reason})
                        raise
                    except asyncio.CancelledError as exc:
                        self._emit_event("canceled", {"method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": "host_call_canceled"})
                        self._emit_event("host_response", {"status": "error", "method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": "host_call_canceled"})
                        raise HostCapabilityCanceled(detail={"provider_call_id": provider_call_id}) from exc
                    except (BrokenPipeError, ConnectionError, EOFError) as exc:
                        self._emit_event(
                            "provider_failure",
                            {
                                "method": method_name,
                                "call_id": event_call_id,
                                "host_call_id": host_call_id or None,
                                "provider_call_id": provider_call_id,
                                "reason": "host_capability_provider_unavailable",
                                "error_type": type(exc).__name__,
                                "provider_id": session.session_id,
                            },
                        )
                        self._emit_event(
                            "host_response",
                            {
                                "status": "error",
                                "method": method_name,
                                "call_id": event_call_id,
                                "host_call_id": host_call_id or None,
                                "provider_call_id": provider_call_id,
                                "reason": "host_capability_provider_unavailable",
                            },
                        )
                        raise HostCapabilityProviderUnavailable(
                            detail={"provider_call_id": provider_call_id, "provider_id": session.session_id, "error_type": type(exc).__name__}
                        ) from exc
                    except HostCapabilityCanceled as exc:
                        self._emit_event("canceled", {"method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": exc.reason, "detail": dict(exc.detail or {})})
                        self._emit_event("host_response", {"status": "error", "method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": exc.reason})
                        raise
                    except HostCapabilityApprovalDenied as exc:
                        self._emit_event("host_response", {"status": "error", "method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": exc.reason})
                        raise
                    except HostCapabilityProviderError as exc:
                        self._emit_event("provider_failure", {"method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": exc.reason, "detail": dict(exc.detail or {})})
                        self._emit_event("host_response", {"status": "error", "method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": exc.reason})
                        raise
                    except Exception as exc:
                        self._emit_event("provider_failure", {"method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": "host_call_failed", "error_type": type(exc).__name__})
                        self._emit_event("host_response", {"status": "error", "method": method_name, "call_id": event_call_id, "host_call_id": host_call_id or None, "provider_call_id": provider_call_id, "reason": "host_call_failed"})
                        raise
                host_call_id = _clean(row.get("host_call_id") or row.get("call_id"))
                self._emit_event(
                    "host_call",
                    {
                        "method": method_name,
                        "call_id": host_call_id or None,
                        "host_call_id": host_call_id or None,
                        "provider_id": session.session_id,
                        "provider_kind": session.provider_kind,
                        "request_id": self.request_id or None,
                        "workflow_id": self.workflow_id or None,
                        "instance_id": self.instance_id or None,
                    },
                )
                try:
                    result = await method.dispatch_async(dict(row.get("arguments") or {}))
                    self._emit_event("host_response", {"status": "ok", "method": method_name, "call_id": host_call_id or None, "host_call_id": host_call_id or None, "provider_id": session.session_id})
                    return result
                except Exception as exc:
                    reason = str(getattr(exc, "reason", "") or "host_call_failed")
                    self._emit_event("host_response", {"status": "error", "method": method_name, "call_id": host_call_id or None, "host_call_id": host_call_id or None, "provider_id": session.session_id, "reason": reason})
                    raise
        raise RuntimeError(f"unsupported_host_method:{method_name}")

    def dispatch(self, call: Dict[str, Any]) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.dispatch_async(call))
        method_name = _clean(dict(call or {}).get("method"))
        if method_name in {"sandbox.describe", "host.describe"}:
            return self.describe()
        resolved = self._resolved_methods().get(method_name)
        if resolved is not None:
            _session, method = resolved
            if method is not None and method.async_handler is None and method.handler is not None:
                return dict(method.handler(dict(dict(call or {}).get("arguments") or {})) or {})
        raise RuntimeError("async_host_capability_dispatch_requires_await")


__all__ = [
    "HOST_CAPABILITY_CONTRACT",
    "HOST_CAPABILITY_CALL_CONTRACT",
    "HOST_CAPABILITY_APPROVAL_CONTRACT",
    "HOST_CAPABILITY_DISCOVERY_CONTRACT",
    "HOST_CAPABILITY_SESSION_CONTRACT",
    "AuditEmitter",
    "ApprovalRequester",
    "AsyncCapabilityHandler",
    "CancelChecker",
    "CapabilityHandler",
    "EventEmitter",
    "HostCapabilityApproval",
    "HostCapabilityApprovalDenied",
    "HostCapabilityBroker",
    "HostCapabilityCallContext",
    "HostCapabilityCanceled",
    "HostCapabilityDescriptor",
    "HostCapabilityMethod",
    "HostCapabilityProviderCall",
    "HostCapabilityProviderError",
    "HostCapabilityProviderUnavailable",
    "HostCapabilityProviderRef",
    "HostCapabilitySession",
    "HostCapabilityTimeout",
    "ProviderInvoker",
    "default_group_path",
    "validate_provider_response",
]
