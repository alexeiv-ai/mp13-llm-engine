from __future__ import annotations

"""Static daemon-owned host API method registry.

The registry is intentionally development-time static: clients may choose which
registered methods to expose for a sandbox scope, but they do not define new
daemon-local implementations at runtime.
"""

import inspect
import re
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Callable, Dict, Iterable, Optional, Union, get_args, get_origin

from .host_capabilities import (
    HostCapabilityDescriptor,
    HostCapabilityMethod,
    HostCapabilityProviderRef,
    HostCapabilitySession,
    default_group_path,
)

SERVICE_BROKER_PROVIDER_ID = "builtin.service_broker"
SERVICE_BROKER_PROVIDER_KIND = "service_broker"
SERVICE_BROKER_CONTRACT = "hosting.sandbox.service_broker_registry.v1"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _object_schema(properties: Dict[str, Any], *, required: Optional[list[str]] = None) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required or []),
        "additionalProperties": False,
    }


def _json_type(annotation: Any) -> Dict[str, Any]:
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Optional, Union, UnionType} and type(None) in args:
        inner = next((item for item in args if item is not type(None)), str)
        schema = _json_type(inner)
        value = schema.get("type")
        if isinstance(value, str):
            schema["type"] = [value, "null"]
        return schema
    if origin is list:
        item_type = _json_type(args[0] if args else str)
        return {"type": "array", "items": item_type}
    if origin is dict:
        return {"type": "object"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation in {dict, Dict}:
        return {"type": "object"}
    return {"type": "string"}


def _parse_doc(doc: str) -> tuple[str, Dict[str, str]]:
    text = inspect.cleandoc(str(doc or "")).strip()
    if not text:
        return "", {}
    description_lines: list[str] = []
    param_descriptions: Dict[str, str] = {}
    in_args = False
    current_param = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            current_param = ""
            continue
        if line in {"Returns:", "Raises:"}:
            in_args = False
            current_param = ""
            continue
        if not in_args:
            if line:
                description_lines.append(line)
            continue
        if not line:
            continue
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?:\s*(.*)", line)
        if match:
            current_param = match.group(1)
            param_descriptions[current_param] = match.group(2).strip()
        elif current_param:
            param_descriptions[current_param] = f"{param_descriptions[current_param]} {line}".strip()
    return " ".join(description_lines).strip(), param_descriptions


@dataclass(frozen=True)
class ServiceBrokerMethodSpec:
    name: str
    callable: Callable[..., Any]
    permissions: tuple[str, ...]
    result_schema: Dict[str, Any]
    scope_requirements: tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    policy_hint: Dict[str, Any] = field(default_factory=dict)
    namespace: str = ""
    provider_id: str = SERVICE_BROKER_PROVIDER_ID

    def args_schema(self) -> Dict[str, Any]:
        _description, param_descriptions = _parse_doc(inspect.getdoc(self.callable) or "")
        signature = inspect.signature(self.callable)
        properties: Dict[str, Any] = {}
        required: list[str] = []
        for param in signature.parameters.values():
            if param.name in {"self", "cls"}:
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            schema = _json_type(param.annotation)
            if param.name in param_descriptions:
                schema["description"] = param_descriptions[param.name]
            if param.default is not inspect.Parameter.empty:
                schema["default"] = param.default
            else:
                required.append(param.name)
            properties[param.name] = schema
        return _object_schema(properties, required=required)

    def description(self) -> str:
        description, _param_descriptions = _parse_doc(inspect.getdoc(self.callable) or "")
        return description or self.name

    def to_descriptor(self, *, approval: Optional[Dict[str, Any]] = None) -> HostCapabilityDescriptor:
        from .host_capabilities import HostCapabilityApproval

        ns = _clean(self.namespace) or self.name.split(".", 1)[0]
        if self.name in {"fs.write_text", "fs.mkdir"}:
            concurrency = {
                "mode": "keyed",
                "group": "filesystem",
                "key_argument": "relative_path",
                "max_concurrency": 1,
                "queue_policy": "bounded",
                "queue_depth": 32,
                "queue_timeout_seconds": 30.0,
                "thread_safe_required": False,
            }
        elif self.name.startswith("fs."):
            concurrency = {
                "mode": "parallel",
                "group": "filesystem",
                "max_concurrency": 32,
                "queue_policy": "bounded",
                "queue_depth": 64,
                "queue_timeout_seconds": 30.0,
                "thread_safe_required": True,
            }
        else:
            concurrency = {
                "mode": "parallel",
                "group": "http",
                "max_concurrency": 32,
                "queue_policy": "bounded",
                "queue_depth": 64,
                "queue_timeout_seconds": 30.0,
                "thread_safe_required": True,
            }
        return HostCapabilityDescriptor(
            name=self.name,
            namespace=ns,
            group_path=default_group_path(self.name),
            description=self.description(),
            args_schema=self.args_schema(),
            result_schema=dict(self.result_schema or {}),
            permissions=[str(item) for item in self.permissions],
            scope_requirements=[dict(item or {}) for item in self.scope_requirements],
            approval=HostCapabilityApproval.from_dict(approval),
            provider=HostCapabilityProviderRef(
                provider_id=self.provider_id,
                kind=SERVICE_BROKER_PROVIDER_KIND,
                owner="service",
                visibility="request",
            ),
            metadata={
                "service_broker": {
                    "contract": SERVICE_BROKER_CONTRACT,
                    "provider_id": self.provider_id,
                    "method": self.name,
                    "policy_hint": dict(self.policy_hint or {}),
                },
                "concurrency": concurrency,
            },
        )

    def contract_description(self) -> Dict[str, Any]:
        descriptor = self.to_descriptor().to_dict()
        return {
            "contract": SERVICE_BROKER_CONTRACT,
            "name": self.name,
            "namespace": descriptor["namespace"],
            "description": descriptor["description"],
            "args_schema": descriptor["args_schema"],
            "result_schema": descriptor["result_schema"],
            "permissions": descriptor["permissions"],
            "scope_requirements": descriptor["scope_requirements"],
            "policy_hint": dict(self.policy_hint or {}),
            "provider": descriptor["provider"],
            "metadata": dict(descriptor.get("metadata") or {}),
        }


class ServiceBrokerContracts:
    def fs_list(self, root_id: str, relative_path: str = "") -> Dict[str, Any]:
        """List direct children under a brokered filesystem root.

        Args:
            root_id (str): Brokered filesystem root identifier from sandbox policy.
            relative_path (str): Relative directory path under the selected root.
        """
        raise NotImplementedError

    def fs_read_text(self, root_id: str, relative_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read UTF text from a brokered filesystem root.

        Args:
            root_id (str): Brokered filesystem root identifier from sandbox policy.
            relative_path (str): Relative file path under the selected root.
            encoding (str): Text encoding used to decode the file.
        """
        raise NotImplementedError

    def fs_write_text(
        self,
        root_id: str,
        relative_path: str,
        text: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> Dict[str, Any]:
        """Write UTF text under a brokered filesystem root.

        Args:
            root_id (str): Brokered filesystem root identifier from sandbox policy.
            relative_path (str): Relative file path under the selected root.
            text (str): Text content to write.
            encoding (str): Text encoding used to write the file.
            create_parents (bool): Create missing parent directories when true.
        """
        raise NotImplementedError

    def fs_mkdir(self, root_id: str, relative_path: str, parents: bool = True, exist_ok: bool = True) -> Dict[str, Any]:
        """Create a directory under a brokered filesystem root.

        Args:
            root_id (str): Brokered filesystem root identifier from sandbox policy.
            relative_path (str): Relative directory path under the selected root.
            parents (bool): Create parent directories when true.
            exist_ok (bool): Treat an existing directory as success when true.
        """
        raise NotImplementedError

    def fs_stat(self, root_id: str, relative_path: str = "") -> Dict[str, Any]:
        """Return metadata for a path under a brokered filesystem root.

        Args:
            root_id (str): Brokered filesystem root identifier from sandbox policy.
            relative_path (str): Relative path under the selected root.
        """
        raise NotImplementedError

    def http_fetch(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        """Fetch an HTTP(S) URL through the host broker.

        Args:
            url (str): Absolute HTTP or HTTPS URL to fetch.
            method (str): HTTP method.
            headers (dict): Request headers allowed by sandbox policy.
            body_b64 (str): Base64-encoded request body.
            timeout_seconds (float): Maximum request duration.
            max_response_bytes (int): Maximum response body bytes returned.
        """
        raise NotImplementedError


_CONTRACTS = ServiceBrokerContracts()


def _object_result(properties: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": dict(properties or {})}


SERVICE_BROKER_METHOD_SPECS: Dict[str, ServiceBrokerMethodSpec] = {
    "fs.list": ServiceBrokerMethodSpec(
        name="fs.list",
        callable=_CONTRACTS.fs_list,
        permissions=("artifact.read",),
        scope_requirements=({"scope": "artifact", "access": "read"},),
        policy_hint={"kind": "filesystem", "access": "read", "allow_empty_relative_path": True},
        result_schema=_object_result(
            {
                "status": {"type": "string"},
                "root_id": {"type": "string"},
                "path": {"type": "string"},
                "entries": {"type": "array", "items": {"type": "object"}},
            }
        ),
    ),
    "fs.read_text": ServiceBrokerMethodSpec(
        name="fs.read_text",
        callable=_CONTRACTS.fs_read_text,
        permissions=("artifact.read",),
        scope_requirements=({"scope": "artifact", "access": "read"},),
        policy_hint={"kind": "filesystem", "access": "read", "allow_empty_relative_path": False},
        result_schema=_object_result(
            {
                "status": {"type": "string"},
                "root_id": {"type": "string"},
                "path": {"type": "string"},
                "text": {"type": "string"},
            }
        ),
    ),
    "fs.write_text": ServiceBrokerMethodSpec(
        name="fs.write_text",
        callable=_CONTRACTS.fs_write_text,
        permissions=("artifact.write",),
        scope_requirements=({"scope": "artifact", "access": "write"},),
        policy_hint={"kind": "filesystem", "access": "write", "allow_empty_relative_path": False},
        result_schema=_object_result(
            {
                "status": {"type": "string"},
                "root_id": {"type": "string"},
                "path": {"type": "string"},
                "bytes_written": {"type": "integer"},
            }
        ),
    ),
    "fs.mkdir": ServiceBrokerMethodSpec(
        name="fs.mkdir",
        callable=_CONTRACTS.fs_mkdir,
        permissions=("artifact.write",),
        scope_requirements=({"scope": "artifact", "access": "write"},),
        policy_hint={"kind": "filesystem", "access": "write", "allow_empty_relative_path": True},
        result_schema=_object_result(
            {
                "status": {"type": "string"},
                "root_id": {"type": "string"},
                "path": {"type": "string"},
                "created": {"type": "boolean"},
            }
        ),
    ),
    "fs.stat": ServiceBrokerMethodSpec(
        name="fs.stat",
        callable=_CONTRACTS.fs_stat,
        permissions=("artifact.read",),
        scope_requirements=({"scope": "artifact", "access": "read"},),
        policy_hint={"kind": "filesystem", "access": "read", "allow_empty_relative_path": True},
        result_schema=_object_result(
            {
                "status": {"type": "string"},
                "root_id": {"type": "string"},
                "path": {"type": "string"},
                "exists": {"type": "boolean"},
                "is_dir": {"type": "boolean"},
                "is_file": {"type": "boolean"},
                "size": {"type": "integer"},
                "mode": {"type": "string"},
            }
        ),
    ),
    "http.fetch": ServiceBrokerMethodSpec(
        name="http.fetch",
        callable=_CONTRACTS.http_fetch,
        permissions=("http.fetch",),
        scope_requirements=({"scope": "http", "access": "fetch"},),
        policy_hint={"kind": "http", "operation": "fetch"},
        result_schema=_object_result(
            {
                "status": {"type": "string"},
                "url": {"type": "string"},
                "status_code": {"type": "integer"},
                "headers": {"type": "object"},
                "body_b64": {"type": "string"},
                "body_size": {"type": "integer"},
                "truncated": {"type": "boolean"},
            }
        ),
    ),
}


def service_broker_method_policy_hint(method: str) -> Dict[str, Any]:
    spec = SERVICE_BROKER_METHOD_SPECS.get(_clean(method))
    if spec is None:
        return {}
    return dict(spec.policy_hint or {})


def service_broker_method_names(*, include_fs: bool = True, include_http: bool = True) -> list[str]:
    names = []
    for name in sorted(SERVICE_BROKER_METHOD_SPECS):
        if name.startswith("fs.") and not include_fs:
            continue
        if name.startswith("http.") and not include_http:
            continue
        names.append(name)
    return names


def service_broker_method_descriptors(
    *,
    include_fs: bool = True,
    include_http: bool = True,
    approval: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    return [
        SERVICE_BROKER_METHOD_SPECS[name].to_descriptor(approval=approval).to_dict()
        for name in service_broker_method_names(include_fs=include_fs, include_http=include_http)
    ]


def service_broker_contract_descriptions(*, include_fs: bool = True, include_http: bool = True) -> list[Dict[str, Any]]:
    return [
        SERVICE_BROKER_METHOD_SPECS[name].contract_description()
        for name in service_broker_method_names(include_fs=include_fs, include_http=include_http)
    ]


def service_broker_discover(*, include_fs: bool = True, include_http: bool = True) -> Dict[str, Any]:
    methods = service_broker_contract_descriptions(include_fs=include_fs, include_http=include_http)
    return {
        "contract": SERVICE_BROKER_CONTRACT,
        "provider": {
            "provider_id": SERVICE_BROKER_PROVIDER_ID,
            "kind": SERVICE_BROKER_PROVIDER_KIND,
            "owner": "service",
        },
        "methods": methods,
        "method_names": [row["name"] for row in methods],
    }


def service_broker_host_capability_session(
    *,
    session_id: str,
    provider_id: str,
    owner: str = "service",
    visibility: str = "request",
    scope: Optional[Dict[str, Any]] = None,
    include_fs: bool = True,
    include_http: bool = True,
    approval: Optional[Dict[str, Any]] = None,
    binding: Optional[Dict[str, Any]] = None,
    allow_override: bool = False,
) -> HostCapabilitySession:
    descriptors = [
        HostCapabilityDescriptor.from_dict(row)
        for row in service_broker_method_descriptors(include_fs=include_fs, include_http=include_http, approval=approval)
    ]
    return HostCapabilitySession(
        session_id=_clean(session_id),
        provider_id=_clean(provider_id),
        owner=_clean(owner) or "service",
        provider_kind=SERVICE_BROKER_PROVIDER_KIND,
        visibility=_clean(visibility) or "request",
        scope=dict(scope or {}),
        methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor) for descriptor in descriptors},
        binding={**dict(binding or {}), "transport": "service_broker"},
        allow_override=allow_override,
    )


def invoke_service_broker_method(
    svc: Any,
    *,
    engine_id: str,
    method: str,
    arguments: Optional[Dict[str, Any]] = None,
    callback_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    req = dict(arguments or {})
    meth = _clean(method)
    eid = _clean(engine_id)
    if not eid:
        raise ValueError("service_broker_engine_id_required")
    callback = dict(callback_context or {}) if isinstance(callback_context, dict) else None
    if meth == "fs.list":
        return svc.sandbox_fs_list(
            engine_id=eid,
            root_id=_clean(req.get("root_id")),
            relative_path=req.get("relative_path"),
            callback_context=callback,
        )
    if meth == "fs.read_text":
        return svc.sandbox_fs_read_text(
            engine_id=eid,
            root_id=_clean(req.get("root_id")),
            relative_path=_clean(req.get("relative_path")),
            encoding=_clean(req.get("encoding")) or "utf-8",
            callback_context=callback,
        )
    if meth == "fs.write_text":
        return svc.sandbox_fs_write_text(
            engine_id=eid,
            root_id=_clean(req.get("root_id")),
            relative_path=_clean(req.get("relative_path")),
            text=str(req.get("text") or ""),
            encoding=_clean(req.get("encoding")) or "utf-8",
            create_parents=bool(req.get("create_parents", True)),
            callback_context=callback,
        )
    if meth == "fs.mkdir":
        return svc.sandbox_fs_mkdir(
            engine_id=eid,
            root_id=_clean(req.get("root_id")),
            relative_path=_clean(req.get("relative_path")),
            parents=bool(req.get("parents", True)),
            exist_ok=bool(req.get("exist_ok", True)),
            callback_context=callback,
        )
    if meth == "fs.stat":
        return svc.sandbox_fs_stat(
            engine_id=eid,
            root_id=_clean(req.get("root_id")),
            relative_path=req.get("relative_path"),
            callback_context=callback,
        )
    if meth == "http.fetch":
        return svc.sandbox_http_fetch(
            engine_id=eid,
            url=_clean(req.get("url")),
            method=_clean(req.get("method")) or "GET",
            headers=dict(req.get("headers") or {}) if isinstance(req.get("headers"), dict) else None,
            body_b64=_clean(req.get("body_b64")),
            timeout_seconds=float(req.get("timeout_seconds") or 30.0),
            max_response_bytes=int(req.get("max_response_bytes") or 1024 * 1024),
            callback_context=callback,
        )
    raise RuntimeError(f"unsupported_service_broker_method:{meth}")


__all__ = [
    "SERVICE_BROKER_CONTRACT",
    "SERVICE_BROKER_METHOD_SPECS",
    "SERVICE_BROKER_PROVIDER_ID",
    "SERVICE_BROKER_PROVIDER_KIND",
    "ServiceBrokerMethodSpec",
    "invoke_service_broker_method",
    "service_broker_contract_descriptions",
    "service_broker_discover",
    "service_broker_host_capability_session",
    "service_broker_method_descriptors",
    "service_broker_method_names",
    "service_broker_method_policy_hint",
]
