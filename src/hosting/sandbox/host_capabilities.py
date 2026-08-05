"""Shared host capability descriptor and broker primitives."""
from __future__ import annotations

import asyncio
import inspect
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional


HOST_CAPABILITY_CONTRACT = "hosting.sandbox.host_capability.v1"
HOST_CAPABILITY_SESSION_CONTRACT = "hosting.sandbox.host_capability_session.v1"
HOST_CAPABILITY_DISCOVERY_CONTRACT = "hosting.sandbox.discovery.v1"
HOST_CAPABILITY_CALL_CONTRACT = "hosting.sandbox.host_capability_call.v1"
HOST_CAPABILITY_APPROVAL_CONTRACT = "hosting.sandbox.host_capability_approval.v1"

_METHOD_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_PROVIDER_KINDS = {"builtin", "client_session", "toolbox_session", "service_broker"}
_ALLOWED_VISIBILITY = {"request", "workflow", "instance", "consumer"}
_MAX_SCHEMA_CHARS = 65536
_MAX_DESCRIPTION_CHARS = 4096
_MAX_PREVIEW_STRING_CHARS = 512
_MAX_PREVIEW_ITEMS = 12
_SECRET_KEY_FRAGMENTS = ("secret", "token", "password", "passwd", "credential", "authorization", "api_key", "apikey", "private_key")

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


def _is_secret_argument_key(key: str) -> bool:
    cleaned = _clean(key).lower().replace("-", "_")
    return any(fragment in cleaned for fragment in _SECRET_KEY_FRAGMENTS)


def _argument_preview_value(key: str, value: Any) -> Any:
    if _is_secret_argument_key(key):
        return {"redacted": True, "reason": "secret_key"}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_PREVIEW_STRING_CHARS:
            return value
        return {"type": "string", "chars": len(value), "omitted": True, "reason": "value_too_large"}
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) <= _MAX_PREVIEW_ITEMS and all(item is None or isinstance(item, (bool, int, float, str)) for item in items):
            preview_items = []
            for index, item in enumerate(items):
                preview_items.append(_argument_preview_value(f"{key}[{index}]", item))
            return preview_items
        return {"type": "array", "items": len(items), "omitted": True}
    if isinstance(value, dict):
        keys = sorted(_clean(name) for name in value.keys() if _clean(name))
        return {"type": "object", "keys": keys[:_MAX_PREVIEW_ITEMS], "key_count": len(keys), "omitted": True}
    return {"type": type(value).__name__, "omitted": True}


def build_argument_preview(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return bounded approval-safe argument values for policy/UI decisions."""
    out: Dict[str, Any] = {}
    for key in sorted(str(name) for name in dict(arguments or {}).keys()):
        cleaned = _clean(key)
        if not cleaned:
            continue
        out[cleaned] = _argument_preview_value(cleaned, dict(arguments or {}).get(key))
    return out


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


class HostCapabilityQueueFull(HostCapabilityProviderError):
    def __init__(self, message: str = "host capability callback queue is full", detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("host_call_queue_full", message, detail)


class HostCapabilityQueueTimeout(HostCapabilityProviderError):
    def __init__(self, message: str = "host capability callback queue timed out", detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("host_call_queue_timeout", message, detail)


@dataclass
class _HostConcurrencyLease:
    controller: "_HostConcurrencyController"
    request_id: str
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.controller.release(self.request_id)


class _HostConcurrencyController:
    """Thread-safe admission shared by all callback sessions for one provider group."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._active: Dict[str, Dict[str, Any]] = {}
        self._queued: deque[tuple[str, Dict[str, Any]]] = deque()
        self._canceled: set[str] = set()

    @staticmethod
    def _slot(policy: Dict[str, Any]) -> str:
        mode = str(policy.get("mode") or "parallel")
        group = str(policy.get("group") or "default")
        if mode == "keyed":
            return f"keyed:{group}:{str(policy.get('resource_key') or '__missing__')}"
        if mode == "serial":
            return f"serial:{group}"
        if mode == "exclusive":
            return f"exclusive:{group}"
        return f"parallel:{group}"

    def _can_admit_locked(self, policy: Dict[str, Any], max_active: int) -> bool:
        if len(self._active) >= max(1, int(max_active or 1)):
            return False
        mode = str(policy.get("mode") or "parallel")
        if mode == "exclusive" and self._active:
            return False
        candidate_slot = self._slot(policy)
        for active in self._active.values():
            active_mode = str(active.get("mode") or "parallel")
            if active_mode == "exclusive":
                return False
            if (
                candidate_slot == self._slot(active)
                and (mode in {"serial", "keyed"} or active_mode in {"serial", "keyed"})
            ):
                return False
        return True

    def acquire(
        self,
        *,
        request_id: str,
        policy: Dict[str, Any],
        max_active: int,
        queue_policy: str,
        queue_depth: int,
        timeout_seconds: float,
        cancel_checker: Optional[CancelChecker] = None,
    ) -> _HostConcurrencyLease:
        rid = _clean(request_id) or f"host-call-{time.time_ns()}"
        bounded = str(queue_policy or "bounded").strip().lower() == "bounded"
        deadline = time.monotonic() + max(0.0, float(timeout_seconds or 0.0)) if timeout_seconds else None
        queued = False
        with self._condition:
            while True:
                if rid in self._canceled:
                    self._canceled.discard(rid)
                    raise HostCapabilityCanceled()
                if cancel_checker is not None:
                    try:
                        cancel_checker()
                    except HostCapabilityCanceled:
                        self._queued = deque(item for item in self._queued if item[0] != rid)
                        self._condition.notify_all()
                        raise
                if self._can_admit_locked(policy, max_active) and (not self._queued or self._queued[0][0] == rid):
                    self._queued = deque(item for item in self._queued if item[0] != rid)
                    self._active[rid] = dict(policy)
                    return _HostConcurrencyLease(self, rid)
                if not bounded:
                    raise HostCapabilityProviderError("host_call_capacity_exceeded", detail={"max_active": max_active})
                if not queued:
                    if len(self._queued) >= max(0, int(queue_depth or 0)):
                        raise HostCapabilityQueueFull(detail={"queue_depth": max(0, int(queue_depth or 0))})
                    self._queued.append((rid, dict(policy)))
                    queued = True
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    self._queued = deque(item for item in self._queued if item[0] != rid)
                    self._condition.notify_all()
                    raise HostCapabilityQueueTimeout(detail={"queue_timeout_seconds": float(timeout_seconds or 0.0)})
                self._condition.wait(timeout=0.05 if remaining is None else min(remaining, 0.05))

    def release(self, request_id: str) -> None:
        with self._condition:
            self._active.pop(_clean(request_id), None)
            self._condition.notify_all()

    def cancel(self, request_id: str) -> None:
        rid = _clean(request_id)
        if not rid:
            return
        with self._condition:
            self._canceled.add(rid)
            self._queued = deque(item for item in self._queued if item[0] != rid)
            self._active.pop(rid, None)
            self._condition.notify_all()

    def snapshot(self, *, max_active: int, queue_policy: str, queue_depth: int, queue_timeout_seconds: float) -> Dict[str, Any]:
        with self._condition:
            return {
                "max_active": max(1, int(max_active or 1)),
                "queue_policy": str(queue_policy or "bounded"),
                "queue_depth": max(0, int(queue_depth or 0)),
                "queue_timeout_seconds": max(0.0, float(queue_timeout_seconds or 0.0)),
                "active_calls": len(self._active),
                "queued_calls": len(self._queued),
            }


_HOST_CONCURRENCY_CONTROLLERS: Dict[str, _HostConcurrencyController] = {}
_HOST_CONCURRENCY_CONTROLLERS_LOCK = threading.RLock()


def _host_concurrency_controller(key: str) -> _HostConcurrencyController:
    normalized = _clean(key) or "default"
    with _HOST_CONCURRENCY_CONTROLLERS_LOCK:
        controller = _HOST_CONCURRENCY_CONTROLLERS.get(normalized)
        if controller is None:
            controller = _HostConcurrencyController()
            _HOST_CONCURRENCY_CONTROLLERS[normalized] = controller
        return controller


class HostCapabilityApprovalDenied(HostCapabilityProviderError):
    def __init__(self, message: str = "host capability call approval denied", detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("host_call_approval_denied", message, detail)


@dataclass(frozen=True)
class HostCapabilityCallContext:
    request_id: str = ""
    instance_id: Optional[str] = None
    engine_id: str = ""
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
            "engine_id": _clean(self.engine_id) or None,
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
    provider_id: str
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
            "provider_id": _clean(self.provider_id),
            "owner": _clean(self.owner),
            "scope": dict(self.scope or {}),
            "methods": [method.descriptor.to_dict() for method in self.methods.values()],
            "provider": {
                "provider_id": _clean(self.provider_id),
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
        engine_id: str = "",
        consumer_id: str = "",
        allowed_namespaces: Optional[Any] = None,
        disabled_namespaces: Optional[Any] = None,
        approved_permissions: Optional[Iterable[str]] = None,
        approval_requester: Optional[ApprovalRequester] = None,
        event_emitter: Optional[EventEmitter] = None,
        audit_emitter: Optional[AuditEmitter] = None,
        state_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.request_id = _clean(request_id)
        self.workflow_id = _clean(workflow_id)
        self.package_id = _clean(package_id)
        self.instance_id = _clean(instance_id)
        self.engine_id = _clean(engine_id)
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
        self._disabled_namespaces = self._normalize_allowed_namespaces(disabled_namespaces) or set()
        self._approved_permissions = set(_string_list(approved_permissions or [])) if approved_permissions is not None else None
        self.approval_requester = approval_requester
        self.event_emitter = event_emitter
        self.audit_emitter = audit_emitter
        self.state_info = dict(state_info or {})
        self._sessions: Dict[str, HostCapabilitySession] = {}
        self._approval_grants: list[Dict[str, Any]] = []

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
        try:
            self.audit_emitter(dict(payload or {}))
        except Exception:
            return

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
    def _concurrency_policy(
        session: HostCapabilitySession,
        method: HostCapabilityMethod,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = dict(method.descriptor.metadata or {})
        raw = dict(metadata.get("concurrency") or metadata.get("concurrency_policy") or {})
        mode = _clean(raw.get("mode")).lower() or "parallel"
        if mode not in {"parallel", "serial", "keyed", "exclusive"}:
            mode = "parallel"
        group = _clean(raw.get("group")) or method.descriptor.name
        if mode == "exclusive" and not _clean(raw.get("group")):
            group = "provider"
        args = dict(arguments or {})
        resource_key = _clean(raw.get("resource_key"))
        key_argument = _clean(raw.get("key_argument") or raw.get("resource_key_argument"))
        if mode == "keyed" and not resource_key:
            if key_argument:
                current: Any = args
                for part in key_argument.split("."):
                    current = current.get(part) if isinstance(current, dict) else None
                resource_key = str(current if current is not None else "__missing__")
            else:
                resource_key = str(args.get("resource_key", args.get("key", "__missing__")))
        try:
            max_raw = raw.get("max_concurrency")
            max_active = int(max_raw) if max_raw is not None else 32
        except Exception:
            max_active = 32
        if mode == "serial":
            max_active = 1
        try:
            depth_raw = raw.get("queue_depth")
            queue_depth = int(depth_raw) if depth_raw is not None else 64
        except Exception:
            queue_depth = 64
        try:
            timeout_raw = raw.get("queue_timeout_seconds")
            queue_timeout = float(timeout_raw) if timeout_raw is not None else 30.0
        except Exception:
            queue_timeout = 30.0
        queue_policy = _clean(raw.get("queue_policy")).lower() or "bounded"
        if queue_policy not in {"bounded", "fail_fast"}:
            queue_policy = "bounded"
        return {
            "mode": mode,
            "group": group,
            "resource_key": resource_key,
            "max_concurrency": max(1, min(max_active, 1024)),
            "queue_policy": queue_policy,
            "queue_depth": max(0, min(queue_depth, 4096)),
            "queue_timeout_seconds": max(0.0, min(queue_timeout, 3600.0)),
            "thread_safe_required": mode == "parallel",
            "provider_id": session.provider_id,
            "method": method.descriptor.name,
        }

    async def _acquire_concurrency(
        self,
        *,
        session: HostCapabilitySession,
        method: HostCapabilityMethod,
        request_id: str,
        arguments: Dict[str, Any],
        timeout_seconds: float,
    ) -> _HostConcurrencyLease:
        policy = self._concurrency_policy(session, method, arguments)
        controller = _host_concurrency_controller(f"{session.session_id}:{policy['group']}")
        acquire_task = asyncio.create_task(
            asyncio.to_thread(
                controller.acquire,
                request_id=request_id,
                policy=policy,
                max_active=int(policy["max_concurrency"]),
                queue_policy=str(policy["queue_policy"]),
                queue_depth=int(policy["queue_depth"]),
                timeout_seconds=min(
                    float(timeout_seconds or 30.0),
                    float(policy["queue_timeout_seconds"] or timeout_seconds or 30.0),
                ),
                cancel_checker=self._check_canceled,
            )
        )
        try:
            return await acquire_task
        except asyncio.CancelledError:
            controller.cancel(request_id)
            raise

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
        if _clean(descriptor.namespace) in self._disabled_namespaces:
            return False
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

    @staticmethod
    def _scope_requirement_keys(method: HostCapabilityMethod) -> list[str]:
        out: list[str] = []
        for item in list(method.descriptor.scope_requirements or []):
            row = dict(item or {})
            scope = _clean(row.get("scope"))
            access = _clean(row.get("access"))
            if scope and access:
                out.append(f"{scope}:{access}")
            elif scope:
                out.append(scope)
        return sorted(out)

    @staticmethod
    def _decision_scope_constraints(row: Dict[str, Any]) -> Dict[str, Any]:
        constraints = row.get("scope_constraints")
        if constraints is None:
            constraints = row.get("constraints")
        if not isinstance(constraints, dict):
            return {}
        if isinstance(constraints.get("arguments"), dict):
            return dict(constraints.get("arguments") or {})
        return dict(constraints or {})

    @staticmethod
    def _constraints_match(arguments: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        args = dict(arguments or {})
        for key, expected in dict(constraints or {}).items():
            if key not in args or args.get(key) != expected:
                return False
        return True

    def _approval_grant_matches(
        self,
        *,
        session: HostCapabilitySession,
        method: HostCapabilityMethod,
        provider_call: HostCapabilityProviderCall,
    ) -> Optional[Dict[str, Any]]:
        now_ms = int(time.time() * 1000)
        scope_keys = self._scope_requirement_keys(method)
        for grant in list(self._approval_grants):
            expires_at_ms = grant.get("expires_at_ms")
            if expires_at_ms is not None and int(expires_at_ms or 0) <= now_ms:
                continue
            if _clean(grant.get("method")) != provider_call.method:
                continue
            if _clean(grant.get("provider_id")) != _clean(session.provider_id):
                continue
            if _clean(grant.get("actor")) != _clean(session.owner):
                continue
            if list(grant.get("scope_requirements") or []) != scope_keys:
                continue
            if not self._constraints_match(provider_call.arguments, dict(grant.get("constraints") or {})):
                continue
            return dict(grant)
        return None

    def _remember_approval_grant(
        self,
        *,
        approval_id: str,
        session: HostCapabilitySession,
        method: HostCapabilityMethod,
        provider_call: HostCapabilityProviderCall,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        ttl_raw = decision.get("ttl_seconds")
        ttl_seconds = int(ttl_raw) if ttl_raw is not None else int(method.descriptor.approval.ttl_seconds or 0)
        now_ms = int(time.time() * 1000)
        grant = {
            "approval_id": _clean(approval_id),
            "method": provider_call.method,
            "provider_id": session.provider_id,
            "provider_kind": session.provider_kind,
            "actor": session.owner,
            "scope_requirements": self._scope_requirement_keys(method),
            "constraints": self._decision_scope_constraints(decision),
            "created_at_ms": now_ms,
            "expires_at_ms": now_ms + ttl_seconds * 1000 if ttl_seconds > 0 else None,
        }
        self._approval_grants = [
            row
            for row in self._approval_grants
            if not (
                _clean(row.get("method")) == grant["method"]
                and _clean(row.get("provider_id")) == grant["provider_id"]
                and _clean(row.get("actor")) == grant["actor"]
                and list(row.get("scope_requirements") or []) == grant["scope_requirements"]
                and dict(row.get("constraints") or {}) == grant["constraints"]
            )
        ]
        self._approval_grants.append(grant)
        return grant

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
        argument_keys = sorted(str(key) for key in dict(provider_call.arguments or {}).keys())
        argument_preview = build_argument_preview(dict(provider_call.arguments or {}))
        audit_base = {
            "event_type": "host_capability_approval",
            "approval_id": approval_id,
            "call_id": call_id,
            "host_call_id": _clean(host_call_id) or None,
            "provider_call_id": provider_call.provider_call_id,
            "method": provider_call.method,
            "argument_keys": argument_keys,
            "argument_preview": argument_preview,
            "context": provider_call.context.to_dict(),
            "approval": method.descriptor.approval.to_dict(),
            "provider": {
                "provider_id": session.provider_id,
                "kind": session.provider_kind,
                "owner": session.owner,
                "visibility": session.visibility,
            },
        }
        existing_grant = self._approval_grant_matches(session=session, method=method, provider_call=provider_call)
        if existing_grant is not None:
            self._emit_audit({**audit_base, "result": "reused", "reason": None, "decision": {"decision": "add_to_scope", "grant": existing_grant}})
            self._emit_event(
                "approval",
                {
                    "status": "reused",
                    "call_id": call_id,
                    "host_call_id": _clean(host_call_id) or None,
                    "approval_id": _clean(existing_grant.get("approval_id")) or approval_id,
                    "provider_call_id": provider_call.provider_call_id,
                    "method": provider_call.method,
                    "decision": {"decision": "add_to_scope", "grant": existing_grant},
                },
            )
            return
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
            "argument_keys": argument_keys,
            "argument_preview": argument_preview,
            "context": provider_call.context.to_dict(),
            "approval": method.descriptor.approval.to_dict(),
            "provider": {
                "provider_id": session.provider_id,
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
        decision_name = _clean(row.get("decision")).lower()
        approved = bool(row.get("approved", status in {"ok", "approved"} or decision_name in {"allow_once", "add_to_scope"}))
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
        if decision_name == "add_to_scope":
            grant = self._remember_approval_grant(
                approval_id=approval_id,
                session=session,
                method=method,
                provider_call=provider_call,
                decision=row,
            )
            row = {**row, "grant": grant}
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
        if not _clean(session.provider_id):
            raise ValueError("host_capability_provider_id_required")
        if sid == _clean(session.provider_id):
            raise ValueError("host_capability_provider_and_session_id_must_differ")
        if sid in self._sessions:
            raise ValueError("host_capability_session_already_exists")
        for name, method in dict(session.methods or {}).items():
            if name != method.descriptor.name:
                raise ValueError("host_capability_method_name_mismatch")
            method.descriptor.validate()
            if _clean(method.descriptor.provider.provider_id) != _clean(session.provider_id):
                raise ValueError("host_capability_method_provider_id_mismatch")
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
            session_id=f"cap_{uuid.uuid4().hex}",
            provider_id=_clean(provider_id),
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
                    "provider_id": session.provider_id,
                    "kind": session.provider_kind,
                    "owner": session.owner,
                    "visibility": session.visibility,
                    "method_count": method_count,
                }
            )
        return out

    def describe_host_capabilities(self) -> Dict[str, Any]:
        method_rows: list[Dict[str, Any]] = []
        for session, method in sorted(self._resolved_methods().values(), key=lambda item: item[1].descriptor.name):
            descriptor = method.descriptor.to_dict()
            policy = self._concurrency_policy(session, method, {})
            controller = _host_concurrency_controller(f"{session.session_id}:{policy['group']}")
            metadata = dict(descriptor.get("metadata") or {})
            metadata["concurrency"] = {
                **policy,
                "runtime": controller.snapshot(
                    max_active=int(policy["max_concurrency"]),
                    queue_policy=str(policy["queue_policy"]),
                    queue_depth=int(policy["queue_depth"]),
                    queue_timeout_seconds=float(policy["queue_timeout_seconds"]),
                ),
            }
            descriptor["metadata"] = metadata
            method_rows.append(descriptor)
        return {
            "methods": method_rows,
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
                "metadata": dict(descriptor.get("metadata") or {}),
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
            "state": dict(self.state_info or {"available": False, "scopes": []}),
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
                        raise HostCapabilityProviderUnavailable(detail={"provider_id": session.provider_id})
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
                            instance_id=self.instance_id or None,
                            engine_id=self.engine_id,
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
                            "provider_id": session.provider_id,
                            "provider_kind": session.provider_kind,
                            "request_id": self.request_id or None,
                            "workflow_id": self.workflow_id or None,
                            "instance_id": self.instance_id or None,
                        },
                    )
                    lease: Optional[_HostConcurrencyLease] = None
                    try:
                        lease = await self._acquire_concurrency(
                            session=session,
                            method=method,
                            request_id=provider_call_id,
                            arguments=dict(call.arguments or {}),
                            timeout_seconds=timeout_seconds,
                        )
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
                                "provider_id": session.provider_id,
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
                                "provider_id": session.provider_id,
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
                            detail={"provider_call_id": provider_call_id, "provider_id": session.provider_id, "error_type": type(exc).__name__}
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
                    finally:
                        if lease is not None:
                            lease.release()
                host_call_id = _clean(row.get("host_call_id") or row.get("call_id"))
                execution_request_id = f"cap_call_{uuid.uuid4().hex}"
                self._emit_event(
                    "host_call",
                    {
                        "method": method_name,
                        "call_id": host_call_id or None,
                        "host_call_id": host_call_id or None,
                        "provider_id": session.provider_id,
                        "provider_kind": session.provider_kind,
                        "request_id": self.request_id or None,
                        "workflow_id": self.workflow_id or None,
                        "instance_id": self.instance_id or None,
                    },
                )
                lease: Optional[_HostConcurrencyLease] = None
                try:
                    lease = await self._acquire_concurrency(
                        session=session,
                        method=method,
                        request_id=execution_request_id,
                        arguments=dict(row.get("arguments") or {}),
                        timeout_seconds=self._provider_timeout_for_call(row),
                    )
                    result = await method.dispatch_async(dict(row.get("arguments") or {}))
                    self._emit_event("host_response", {"status": "ok", "method": method_name, "call_id": host_call_id or None, "host_call_id": host_call_id or None, "provider_id": session.provider_id})
                    return result
                except Exception as exc:
                    reason = str(getattr(exc, "reason", "") or "host_call_failed")
                    self._emit_event("host_response", {"status": "error", "method": method_name, "call_id": host_call_id or None, "host_call_id": host_call_id or None, "provider_id": session.provider_id, "reason": reason})
                    raise
                finally:
                    if lease is not None:
                        lease.release()
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
    "HostCapabilityQueueFull",
    "HostCapabilityQueueTimeout",
    "HostCapabilitySession",
    "HostCapabilityTimeout",
    "ProviderInvoker",
    "build_argument_preview",
    "default_group_path",
    "validate_provider_response",
]
