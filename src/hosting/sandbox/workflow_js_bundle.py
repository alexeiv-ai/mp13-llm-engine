"""
Host-bridge import finalizer for QuickJS workflow node sources.

The JS node worker executes a single script and does not provide a module
loader. This helper is intentionally narrow: it rewrites static imports that
target known host bridge specifiers into bindings against the injected
``api``/``console`` globals, then reports every disabled or unresolved import so
callers can decide whether to execute, rebundle, or reject the source.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .policy import WorkerSandboxPolicy


_IMPORT_FROM_RE = re.compile(
    r"(?P<statement>^[ \t]*import\s+(?P<clause>[\s\S]*?)\s+from\s*(?P<quote>['\"])(?P<specifier>[^'\"]+)(?P=quote)\s*;?[ \t]*(?:\r?\n)?)",
    re.MULTILINE,
)
_SIDE_EFFECT_IMPORT_RE = re.compile(
    r"(?P<statement>^[ \t]*import\s*(?P<quote>['\"])(?P<specifier>[^'\"]+)(?P=quote)\s*;?[ \t]*(?:\r?\n)?)",
    re.MULTILINE,
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


@dataclass(frozen=True)
class WorkflowJsBridgeImport:
    """A host bridge import specifier that can be rewritten into JS globals."""

    specifier: str
    default_expression: str
    namespace_expression: str
    named_expression: str
    enabled: bool = True
    description: str = ""

    @classmethod
    def from_mapping(cls, specifier: str, value: Any) -> "WorkflowJsBridgeImport":
        if isinstance(value, WorkflowJsBridgeImport):
            return value
        if isinstance(value, str):
            return cls(
                specifier=specifier,
                default_expression=value,
                namespace_expression=value,
                named_expression=value,
            )
        if not isinstance(value, Mapping):
            raise TypeError(f"workflow_js_bridge_import_invalid:{specifier}")
        expression = str(value.get("expression") or "").strip()
        default_expression = str(value.get("default_expression") or expression).strip()
        namespace_expression = str(value.get("namespace_expression") or expression).strip()
        named_expression = str(value.get("named_expression") or expression).strip()
        if not default_expression or not namespace_expression or not named_expression:
            raise ValueError(f"workflow_js_bridge_import_expression_required:{specifier}")
        return cls(
            specifier=specifier,
            default_expression=default_expression,
            namespace_expression=namespace_expression,
            named_expression=named_expression,
            enabled=bool(value.get("enabled", True)),
            description=str(value.get("description") or ""),
        )


def _clean_set(values: Optional[Iterable[str]]) -> set[str]:
    return {str(value).strip() for value in (values or []) if str(value).strip()}


def _host_api_namespace_flags(sandbox_policy: Optional[Mapping[str, Any]]) -> tuple[bool, bool]:
    sandbox = dict(dict(sandbox_policy or {}).get("sandbox") or sandbox_policy or {})
    host_api_policy = sandbox.get("host_api") if isinstance(sandbox.get("host_api"), Mapping) else {}
    namespace_policy = dict(host_api_policy.get("namespaces") or {})
    fs_enabled = bool(host_api_policy.get("enabled", True))
    http_enabled = False
    http_namespace_enabled = bool(host_api_policy.get("enabled", True))
    for key in ("fs", "artifact_fs"):
        if key in host_api_policy:
            fs_enabled = bool(host_api_policy.get(key))
        if key in namespace_policy:
            fs_enabled = bool(namespace_policy.get(key))
    for key in ("http", "http_fetch"):
        if key in host_api_policy:
            http_namespace_enabled = bool(host_api_policy.get(key))
        if key in namespace_policy:
            http_namespace_enabled = bool(namespace_policy.get(key))
    if sandbox_policy:
        worker_policy = WorkerSandboxPolicy.from_mapping(dict(sandbox_policy or {}))
        http_enabled = (
            http_namespace_enabled
            and bool(worker_policy.enabled)
            and bool(worker_policy.brokered_io.http)
            and str(worker_policy.network.mode or "").strip().lower() == "brokered_only"
        )
    return fs_enabled, http_enabled


def workflow_js_host_bridge_imports(
    *,
    host_description: Optional[Mapping[str, Any]] = None,
    sandbox_policy: Optional[Mapping[str, Any]] = None,
    enabled_imports: Optional[Iterable[str]] = None,
    disabled_imports: Optional[Iterable[str]] = None,
) -> Dict[str, WorkflowJsBridgeImport]:
    """
    Return the default bridge import table for the JS node worker.

    ``host_description`` may be the result of ``api.describe()``/``host.describe``.
    When it contains a ``methods`` list, policy-gated bridges such as fs/http are
    enabled only when their backing methods are present. ``enabled_imports`` and
    ``disabled_imports`` are final caller overrides.
    """

    enabled_overrides = _clean_set(enabled_imports)
    disabled_overrides = _clean_set(disabled_imports)
    fs_enabled, http_enabled = _host_api_namespace_flags(sandbox_policy)
    methods = {str(method) for method in dict(host_description or {}).get("methods", [])}
    if methods:
        fs_enabled = any(method.startswith("fs.") for method in methods)
        http_enabled = "http.fetch" in methods

    def _enabled(specifier: str, default: bool = True) -> bool:
        if specifier in enabled_overrides:
            return True
        if specifier in disabled_overrides:
            return False
        return default

    return {
        "@host/api": WorkflowJsBridgeImport(
            specifier="@host/api",
            default_expression="api",
            namespace_expression="api",
            named_expression="api",
            enabled=_enabled("@host/api", True),
            description="Injected JS node host API object.",
        ),
        "@host/fs": WorkflowJsBridgeImport(
            specifier="@host/fs",
            default_expression="api.fs",
            namespace_expression="api.fs",
            named_expression="api.fs",
            enabled=_enabled("@host/fs", fs_enabled),
            description="Artifact-root filesystem bridge.",
        ),
        "@host/http": WorkflowJsBridgeImport(
            specifier="@host/http",
            default_expression="api.http",
            namespace_expression="api.http",
            named_expression="api.http",
            enabled=_enabled("@host/http", http_enabled),
            description="Brokered HTTP bridge.",
        ),
        "@host/codec": WorkflowJsBridgeImport(
            specifier="@host/codec",
            default_expression="api.codec",
            namespace_expression="api.codec",
            named_expression="api.codec",
            enabled=_enabled("@host/codec", True),
            description="Runtime-provided codec helpers.",
        ),
        "@host/crypto": WorkflowJsBridgeImport(
            specifier="@host/crypto",
            default_expression="api.crypto",
            namespace_expression="api.crypto",
            named_expression="api.crypto",
            enabled=_enabled("@host/crypto", True),
            description="Runtime-provided deterministic crypto helpers.",
        ),
        "@host/console": WorkflowJsBridgeImport(
            specifier="@host/console",
            default_expression="console",
            namespace_expression="console",
            named_expression="console",
            enabled=_enabled("@host/console", True),
            description="Bounded workflow console logger.",
        ),
        "@host/progress": WorkflowJsBridgeImport(
            specifier="@host/progress",
            default_expression="api.progress",
            namespace_expression="api.progress",
            named_expression="api",
            enabled=_enabled("@host/progress", True),
            description="Workflow progress emitter.",
        ),
        "@host/call": WorkflowJsBridgeImport(
            specifier="@host/call",
            default_expression="api.call",
            namespace_expression="api.call",
            named_expression="api",
            enabled=_enabled("@host/call", True),
            description="Low-level host method dispatcher.",
        ),
        "@host/describe": WorkflowJsBridgeImport(
            specifier="@host/describe",
            default_expression="api.describe",
            namespace_expression="api.describe",
            named_expression="api",
            enabled=_enabled("@host/describe", True),
            description="Host API discovery helper.",
        ),
    }


def _normalize_bridge_imports(bridge_imports: Optional[Mapping[str, Any]]) -> Dict[str, WorkflowJsBridgeImport]:
    if bridge_imports is None:
        return workflow_js_host_bridge_imports()
    return {str(specifier): WorkflowJsBridgeImport.from_mapping(str(specifier), value) for specifier, value in bridge_imports.items()}


def _is_identifier(value: str) -> bool:
    return bool(_IDENTIFIER_RE.match(value or ""))


def _split_import_clause(clause: str) -> tuple[Optional[str], Optional[str], List[tuple[str, str]]]:
    remaining = " ".join(str(clause or "").strip().split())
    default_binding: Optional[str] = None
    namespace_binding: Optional[str] = None
    named_bindings: List[tuple[str, str]] = []
    if not remaining:
        return default_binding, namespace_binding, named_bindings
    if remaining.startswith("* as "):
        namespace_binding = remaining[5:].strip()
        return default_binding, namespace_binding, named_bindings
    if remaining.startswith("{") and remaining.endswith("}"):
        return default_binding, namespace_binding, _parse_named_imports(remaining[1:-1])
    first, sep, rest = remaining.partition(",")
    if sep:
        default_binding = first.strip()
        rest = rest.strip()
        if rest.startswith("* as "):
            namespace_binding = rest[5:].strip()
        elif rest.startswith("{") and rest.endswith("}"):
            named_bindings = _parse_named_imports(rest[1:-1])
        else:
            named_bindings = [("", rest)]
        return default_binding, namespace_binding, named_bindings
    default_binding = remaining.strip()
    return default_binding, namespace_binding, named_bindings


def _parse_named_imports(raw: str) -> List[tuple[str, str]]:
    bindings: List[tuple[str, str]] = []
    for item in str(raw or "").split(","):
        token = " ".join(item.strip().split())
        if not token:
            continue
        if " as " in token:
            imported, local = token.split(" as ", 1)
        else:
            imported, local = token, token
        bindings.append((imported.strip(), local.strip()))
    return bindings


def _declaration_for_import(clause: str, bridge: WorkflowJsBridgeImport) -> tuple[str, List[Dict[str, str]], List[str]]:
    default_binding, namespace_binding, named_bindings = _split_import_clause(clause)
    declarations: List[str] = []
    bindings: List[Dict[str, str]] = []
    invalid: List[str] = []
    if default_binding:
        if _is_identifier(default_binding):
            declarations.append(f"const {default_binding} = {bridge.default_expression};")
            bindings.append({"kind": "default", "local": default_binding})
        else:
            invalid.append(default_binding)
    if namespace_binding:
        if _is_identifier(namespace_binding):
            declarations.append(f"const {namespace_binding} = {bridge.namespace_expression};")
            bindings.append({"kind": "namespace", "local": namespace_binding})
        else:
            invalid.append(namespace_binding)
    if named_bindings:
        parts: List[str] = []
        for imported, local in named_bindings:
            if not _is_identifier(imported) or not _is_identifier(local):
                invalid.append(local or imported)
                continue
            if imported == local:
                parts.append(imported)
            else:
                parts.append(f"{imported}: {local}")
            bindings.append({"kind": "named", "imported": imported, "local": local})
        if parts:
            declarations.append(f"const {{ {', '.join(parts)} }} = {bridge.named_expression};")
    return "\n".join(declarations), bindings, invalid


def _import_matches(source: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    occupied: List[range] = []
    for match in _IMPORT_FROM_RE.finditer(source):
        matches.append(
            {
                "start": match.start("statement"),
                "end": match.end("statement"),
                "statement": match.group("statement"),
                "specifier": match.group("specifier"),
                "clause": match.group("clause"),
                "kind": "from",
            }
        )
        occupied.append(range(match.start("statement"), match.end("statement")))
    for match in _SIDE_EFFECT_IMPORT_RE.finditer(source):
        span = range(match.start("statement"), match.end("statement"))
        if any(match.start("statement") in item or (match.end("statement") - 1) in item for item in occupied):
            continue
        matches.append(
            {
                "start": match.start("statement"),
                "end": match.end("statement"),
                "statement": match.group("statement"),
                "specifier": match.group("specifier"),
                "clause": "",
                "kind": "side_effect",
            }
        )
    return sorted(matches, key=lambda item: int(item["start"]))


def build_workflow_js_bundle(
    source: str,
    *,
    bridge_imports: Optional[Mapping[str, Any]] = None,
    host_description: Optional[Mapping[str, Any]] = None,
    sandbox_policy: Optional[Mapping[str, Any]] = None,
    enabled_imports: Optional[Iterable[str]] = None,
    disabled_imports: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Rewrite supported host bridge imports and return a single JS worker script.

    The returned ``ok`` field is true only when every static import was resolved
    to an enabled bridge. Disabled and unresolved imports are left unchanged so
    the resulting source remains diagnosable and should not be submitted unless
    ``ok`` is true.
    """

    if bridge_imports is None:
        bridges = workflow_js_host_bridge_imports(
            host_description=host_description,
            sandbox_policy=sandbox_policy,
            enabled_imports=enabled_imports,
            disabled_imports=disabled_imports,
        )
    else:
        bridges = _normalize_bridge_imports(bridge_imports)
        explicit_disabled = _clean_set(disabled_imports)
        explicit_enabled = _clean_set(enabled_imports)
        if explicit_disabled or explicit_enabled:
            updated: Dict[str, WorkflowJsBridgeImport] = {}
            for specifier, bridge in bridges.items():
                enabled = bridge.enabled
                if specifier in explicit_enabled:
                    enabled = True
                if specifier in explicit_disabled:
                    enabled = False
                updated[specifier] = WorkflowJsBridgeImport(
                    specifier=bridge.specifier,
                    default_expression=bridge.default_expression,
                    namespace_expression=bridge.namespace_expression,
                    named_expression=bridge.named_expression,
                    enabled=enabled,
                    description=bridge.description,
                )
            bridges = updated

    details: List[Dict[str, Any]] = []
    allowed: set[str] = set()
    disabled: set[str] = set()
    unresolved: set[str] = set()
    output: List[str] = []
    cursor = 0
    for item in _import_matches(str(source or "")):
        start = int(item["start"])
        end = int(item["end"])
        specifier = str(item["specifier"])
        bridge = bridges.get(specifier)
        output.append(str(source or "")[cursor:start])
        detail: Dict[str, Any] = {
            "specifier": specifier,
            "statement": str(item["statement"]).strip(),
            "status": "unresolved",
            "bindings": [],
        }
        if bridge is None:
            unresolved.add(specifier)
            output.append(str(item["statement"]))
        elif not bridge.enabled:
            disabled.add(specifier)
            detail["status"] = "disabled"
            output.append(str(item["statement"]))
        elif item["kind"] == "side_effect":
            allowed.add(specifier)
            detail["status"] = "allowed"
            output.append(f"/* workflow-js-bundle host bridge import: {specifier} */\n")
        else:
            declaration, bindings, invalid = _declaration_for_import(str(item["clause"]), bridge)
            if invalid:
                unresolved.add(specifier)
                detail["status"] = "invalid"
                detail["invalid_bindings"] = invalid
                output.append(str(item["statement"]))
            else:
                allowed.add(specifier)
                detail["status"] = "allowed"
                detail["bindings"] = bindings
                output.append(declaration + "\n")
        details.append(detail)
        cursor = end
    output.append(str(source or "")[cursor:])
    module_source = "".join(output)
    module_sha256 = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
    return {
        "ok": not disabled and not unresolved,
        "module_source": module_source,
        "module_sha256": module_sha256,
        "resolved_allowed_imports": sorted(allowed),
        "resolved_disabled_imports": sorted(disabled),
        "unresolved_imports": sorted(unresolved),
        "import_details": details,
        "bridge_imports": {
            specifier: {
                "enabled": bridge.enabled,
                "default_expression": bridge.default_expression,
                "namespace_expression": bridge.namespace_expression,
                "named_expression": bridge.named_expression,
                "description": bridge.description,
            }
            for specifier, bridge in sorted(bridges.items())
        },
    }


def build_workflow_js_bundle_request(
    source: str,
    *,
    package_id: str,
    workflow_id: str,
    package_source_digest: str,
    payload: Any,
    bridge_imports: Optional[Mapping[str, Any]] = None,
    host_description: Optional[Mapping[str, Any]] = None,
    sandbox_policy: Optional[Mapping[str, Any]] = None,
    enabled_imports: Optional[Iterable[str]] = None,
    disabled_imports: Optional[Iterable[str]] = None,
    request_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a workflow-js-execute request around a finalized JS bundle."""

    bundle = build_workflow_js_bundle(
        source,
        bridge_imports=bridge_imports,
        host_description=host_description,
        sandbox_policy=sandbox_policy,
        enabled_imports=enabled_imports,
        disabled_imports=disabled_imports,
    )
    request: Dict[str, Any] = dict(request_fields or {})
    request.update(
        {
            "module_source": bundle["module_source"],
            "module_sha256": bundle["module_sha256"],
            "package_id": str(package_id),
            "workflow_id": str(workflow_id),
            "package_source_digest": str(package_source_digest),
            "payload": payload,
            "javascript": {
                **dict(request.get("javascript") or {}),
                "bundle": {
                    "ok": bundle["ok"],
                    "resolved_allowed_imports": bundle["resolved_allowed_imports"],
                    "resolved_disabled_imports": bundle["resolved_disabled_imports"],
                    "unresolved_imports": bundle["unresolved_imports"],
                    "import_details": bundle["import_details"],
                },
            },
        }
    )
    return request


__all__ = [
    "WorkflowJsBridgeImport",
    "build_workflow_js_bundle",
    "build_workflow_js_bundle_request",
    "workflow_js_host_bridge_imports",
]
