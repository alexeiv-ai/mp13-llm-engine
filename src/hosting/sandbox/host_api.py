"""Scoped host API registry for hosted sandbox back channels."""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from .host_capabilities import (
    HostCapabilityApproval,
    HostCapabilityDescriptor,
    HostCapabilityMethod,
    HostCapabilityProviderRef,
    default_group_path,
)

HostApiHandler = Callable[[Dict[str, Any]], Dict[str, Any]]
AsyncHostApiHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _object_schema(properties: Dict[str, Any], *, required: Optional[list[str]] = None) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required or []),
        "additionalProperties": False,
    }


@dataclass
class HostApiMethod:
    name: str
    description: str
    args_schema: Dict[str, Any] = field(default_factory=dict)
    result_schema: Dict[str, Any] = field(default_factory=dict)
    namespace: str = ""
    permissions: list[str] = field(default_factory=list)
    handler: Optional[HostApiHandler] = None
    async_handler: Optional[AsyncHostApiHandler] = None

    def to_description(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace or self.name.split(".", 1)[0],
            "description": self.description,
            "args_schema": dict(self.args_schema or {}),
            "result_schema": dict(self.result_schema or {}),
            "permissions": list(self.permissions or []),
            "async": self.async_handler is not None,
        }

    def to_capability_descriptor(self, *, provider_id: str = "builtin.host_api") -> HostCapabilityDescriptor:
        namespace = self.namespace or self.name.split(".", 1)[0]
        permissions = list(self.permissions or [])
        scope_requirements = [
            {"scope": permission.rsplit(".", 1)[0], "access": permission.rsplit(".", 1)[-1]}
            for permission in permissions
            if "." in permission
        ]
        return HostCapabilityDescriptor(
            name=self.name,
            namespace=namespace,
            group_path=default_group_path(self.name),
            description=self.description,
            args_schema=dict(self.args_schema or {}),
            result_schema=dict(self.result_schema or {}),
            permissions=permissions,
            scope_requirements=scope_requirements,
            approval=HostCapabilityApproval(mode="none", ttl_seconds=0),
            provider=HostCapabilityProviderRef(
                provider_id=provider_id,
                kind="builtin",
                owner="service",
                visibility="request",
            ),
        )


class HostApiRegistry:
    """Native scoped capability registry for sandbox host calls.

    This intentionally mirrors the useful part of native toolbox execution:
    named capabilities with discovery metadata and policy-aware handlers. It
    does not include toolbox manifests, bundle staging, assignment, repair, or
    GC semantics.
    """

    def __init__(
        self,
        *,
        contract: str,
        request_id: str = "",
        policy: Optional[Dict[str, Any]] = None,
        roots: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.contract = _clean(contract) or "hosting.sandbox.host_api.v1"
        self.request_id = _clean(request_id)
        self.policy = dict(policy or {})
        self.roots = dict(roots or {})
        self._methods: Dict[str, HostApiMethod] = {}

    def register(
        self,
        name: str,
        *,
        description: str,
        args_schema: Optional[Dict[str, Any]] = None,
        result_schema: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        permissions: Optional[list[str]] = None,
        handler: Optional[HostApiHandler] = None,
        async_handler: Optional[AsyncHostApiHandler] = None,
    ) -> None:
        method_name = _clean(name)
        if not method_name:
            raise ValueError("host_api_method_name_required")
        if handler is None and async_handler is None:
            raise ValueError("host_api_handler_required")
        self._methods[method_name] = HostApiMethod(
            name=method_name,
            description=str(description or "").strip(),
            args_schema=dict(args_schema or {}),
            result_schema=dict(result_schema or {}),
            namespace=_clean(namespace) or method_name.split(".", 1)[0],
            permissions=list(permissions or []),
            handler=handler,
            async_handler=async_handler,
        )

    def method_names(self) -> list[str]:
        return sorted({"host.describe", "sandbox.describe", *self._methods.keys()})

    def describe(self) -> Dict[str, Any]:
        capability_descriptors = [self._host_describe_method().to_capability_descriptor()]
        capability_descriptors.append(self._sandbox_describe_method().to_capability_descriptor())
        capability_descriptors.extend(self._methods[name].to_capability_descriptor() for name in sorted(self._methods.keys()))
        capability_methods = [descriptor.to_dict() for descriptor in capability_descriptors]
        host_capabilities = {
            "methods": capability_methods,
            "groups": self._capability_groups(capability_descriptors),
            "providers": [
                {
                    "provider_id": "builtin.host_api",
                    "kind": "builtin",
                    "owner": "service",
                    "visibility": "request",
                    "method_count": len(capability_descriptors),
                }
            ],
            "transport": {
                "framed": True,
                "host_call_id": True,
                "async_capable": True,
                "out_of_order_responses": True,
            },
        }
        async_method_names = {name for name, method in self._methods.items() if method.async_handler is not None}
        method_descriptions = [
            {
                "name": descriptor["name"],
                "namespace": descriptor["namespace"],
                "description": descriptor.get("description", ""),
                "args_schema": dict(descriptor.get("args_schema") or {}),
                "result_schema": dict(descriptor.get("result_schema") or {}),
                "permissions": list(descriptor.get("permissions") or []),
                "async": descriptor["name"] in async_method_names,
                "group_path": list(descriptor.get("group_path") or []),
                "provider": dict(descriptor.get("provider") or {}),
            }
            for descriptor in capability_methods
        ]
        return {
            "status": "ok",
            "contract": "hosting.sandbox.discovery.v1",
            "request_id": self.request_id,
            "methods": self.method_names(),
            "method_descriptions": method_descriptions,
            "host_capabilities": host_capabilities,
            "harness": {
                "host_api_entrypoints": ["host.call", "host.describe", "sandbox.describe"],
            },
            "events": {
                "worker_live": ["progress"],
                "host_generated": ["started", "heartbeat", "stdout", "stderr", "log", "artifact", "result", "error", "canceled", "done"],
                "observations": ["host_call", "host_response"],
                "reserved": ["approval", "state_notice", "action_notice"],
            },
            "state": {"available": False, "scopes": []},
            "actions": {"available": False, "entries": []},
            "roots": dict(self.roots or {}),
            "policy": dict(self.policy or {}),
            "transport": {
                "framed": True,
                "host_call_id": True,
                "async_capable": True,
                "out_of_order_responses": True,
                "sync_handlers": True,
                "async_handlers": True,
            },
        }

    def _host_describe_method(self) -> HostApiMethod:
        describe_args, describe_result = host_describe_schema()
        return HostApiMethod(
            name="host.describe",
            namespace="host",
            description="Describe host API methods available to this sandbox request.",
            args_schema=describe_args,
            result_schema=describe_result,
            permissions=[],
            handler=lambda _args: self.describe(),
        )

    def _sandbox_describe_method(self) -> HostApiMethod:
        describe_args, describe_result = host_describe_schema()
        return HostApiMethod(
            name="sandbox.describe",
            namespace="sandbox",
            description="Describe the sandbox harness, events, roots, policy, state, actions, and host capabilities.",
            args_schema=describe_args,
            result_schema=describe_result,
            permissions=[],
            handler=lambda _args: self.describe(),
        )

    @staticmethod
    def _capability_groups(descriptors: list[HostCapabilityDescriptor]) -> list[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for descriptor in descriptors:
            path = list(descriptor.group_path or [])
            key = "/".join(path)
            groups.setdefault(key, {"path": path, "methods": []})
            groups[key]["methods"].append(descriptor.name)
        return [groups[key] for key in sorted(groups.keys())]

    def capability_methods(self, *, provider_id: str = "builtin.host_api") -> list[HostCapabilityMethod]:
        methods = [
            HostCapabilityMethod(
                descriptor=self._host_describe_method().to_capability_descriptor(provider_id=provider_id),
                handler=lambda _args: self.describe(),
            ),
            HostCapabilityMethod(
                descriptor=self._sandbox_describe_method().to_capability_descriptor(provider_id=provider_id),
                handler=lambda _args: self.describe(),
            ),
        ]
        for name in sorted(self._methods.keys()):
            method = self._methods[name]
            methods.append(
                HostCapabilityMethod(
                    descriptor=method.to_capability_descriptor(provider_id=provider_id),
                    handler=method.handler,
                    async_handler=method.async_handler,
                )
            )
        return methods

    async def dispatch_async(self, call: Dict[str, Any]) -> Dict[str, Any]:
        method_name = _clean(dict(call or {}).get("method"))
        if method_name in {"host.describe", "sandbox.describe"}:
            return self.describe()
        method = self._methods.get(method_name)
        if method is None:
            raise RuntimeError(f"unsupported_host_method:{method_name}")
        args = dict(dict(call or {}).get("arguments") or {})
        if method.async_handler is not None:
            return dict(await method.async_handler(args) or {})
        if method.handler is None:
            raise RuntimeError(f"host_method_handler_missing:{method_name}")
        result = method.handler(args)
        if inspect.isawaitable(result):
            return dict(await result or {})
        return dict(result or {})

    def dispatch(self, call: Dict[str, Any]) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.dispatch_async(call))
        method_name = _clean(dict(call or {}).get("method"))
        method = self._methods.get(method_name)
        if method_name in {"host.describe", "sandbox.describe"}:
            return self.describe()
        if method is not None and method.async_handler is None and method.handler is not None:
            return dict(method.handler(dict(dict(call or {}).get("arguments") or {})) or {})
        raise RuntimeError("async_host_api_dispatch_requires_await")


def host_describe_schema() -> tuple[Dict[str, Any], Dict[str, Any]]:
    return _object_schema({}), _object_schema(
        {
            "status": {"type": "string"},
            "contract": {"type": "string"},
            "methods": {"type": "array", "items": {"type": "string"}},
            "method_descriptions": {"type": "array", "items": {"type": "object"}},
            "roots": {"type": "object"},
            "policy": {"type": "object"},
            "transport": {"type": "object"},
        }
    )


def fs_root_args_schema(*, text: bool = False, mkdir: bool = False) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "root_id": {"type": "string", "description": "Declared artifact input or output root name."},
        "relative_path": {"type": "string", "default": "", "description": "Relative path under the selected root."},
    }
    if text:
        props["encoding"] = {"type": "string", "default": "utf-8"}
    if mkdir:
        props["parents"] = {"type": "boolean", "default": True}
        props["exist_ok"] = {"type": "boolean", "default": True}
    return _object_schema(props, required=["root_id"])


def fs_write_text_args_schema() -> Dict[str, Any]:
    schema = fs_root_args_schema(text=True)
    schema["properties"]["text"] = {"type": "string", "default": ""}
    schema["properties"]["create_parents"] = {"type": "boolean", "default": True}
    return schema


def known_host_capability_method_descriptors(
    *,
    include_fs: bool = True,
    include_http: bool = True,
) -> list[Dict[str, Any]]:
    """Return client-registerable descriptors for broker-supported host methods."""
    from .service_broker_registry import service_broker_method_descriptors

    return service_broker_method_descriptors(include_fs=include_fs, include_http=include_http)


__all__ = [
    "HostApiHandler",
    "HostApiMethod",
    "HostApiRegistry",
    "host_describe_schema",
    "fs_root_args_schema",
    "fs_write_text_args_schema",
    "known_host_capability_method_descriptors",
]
