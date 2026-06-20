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
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
_EXPORT_NAMED_RE = re.compile(r"^[ \t]*export\s*\{(?P<items>[^}]+)\}\s*;?[ \t]*(?:\r?\n)?", re.MULTILINE)
_EXPORT_FROM_RE = re.compile(r"^[ \t]*export\s+[\s\S]*?\s+from\s*['\"][^'\"]+['\"]\s*;?[ \t]*(?:\r?\n)?", re.MULTILINE)
_EXPORT_DECL_RE = re.compile(r"^(?P<indent>[ \t]*)export\s+(?P<kind>function|class|const|let|var)\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\b", re.MULTILINE)
_EXPORT_DEFAULT_NAMED_RE = re.compile(r"^(?P<indent>[ \t]*)export\s+default\s+(?P<kind>function|class)\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\b", re.MULTILINE)
_EXPORT_DEFAULT_EXPR_RE = re.compile(r"^(?P<indent>[ \t]*)export\s+default\s+(?P<expr>[^;\r\n]+)\s*;?[ \t]*(?:\r?\n)?", re.MULTILINE)
_DYNAMIC_IMPORT_RE = re.compile(r"\bimport\s*\(")
_REQUIRE_RE = re.compile(r"\brequire\s*\(")
_SEGMENT_BEGIN_RE = re.compile(r"^/\*\s*workflow-js-bundle-segment-begin\s+(?P<meta>\{.*\})\s*\*/$")
_SEGMENT_END_RE = re.compile(r"^/\*\s*workflow-js-bundle-segment-end\s+(?P<meta>\{.*\})\s*\*/$")
_NODE_BUILTINS = {
    "assert",
    "buffer",
    "child_process",
    "cluster",
    "console",
    "constants",
    "crypto",
    "dgram",
    "dns",
    "domain",
    "events",
    "fs",
    "http",
    "https",
    "module",
    "net",
    "os",
    "path",
    "perf_hooks",
    "process",
    "punycode",
    "querystring",
    "readline",
    "repl",
    "stream",
    "string_decoder",
    "timers",
    "tls",
    "tty",
    "url",
    "util",
    "v8",
    "vm",
    "worker_threads",
    "zlib",
}


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


def _module_id(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"workflow_js_module_id_invalid:{value}")
    return "/".join(path.parts)


def _relative_module_id(importer_id: str, specifier: str) -> str:
    base = PurePosixPath(_module_id(importer_id)).parent
    normalized = posixpath.normpath(str(base.joinpath(str(specifier or "").replace("\\", "/"))))
    if normalized == "." or normalized.startswith("../"):
        raise ValueError(f"workflow_js_module_ref_escape:{specifier}")
    return _module_id(normalized)


def _candidate_ids(raw_id: str) -> List[str]:
    clean = _module_id(raw_id)
    candidates = [clean]
    if not clean.endswith(".js"):
        candidates.append(f"{clean}.js")
        candidates.append(f"{clean}/index.js")
    return list(dict.fromkeys(candidates))


def _root_rows(roots: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, value in enumerate(list(roots or [])):
        if isinstance(value, Mapping):
            raw_path = str(value.get("path") or value.get("root") or "").strip()
            name = str(value.get("name") or value.get("id") or f"root_{index}").strip() or f"root_{index}"
        else:
            raw_path = str(value or "").strip()
            name = f"root_{index}"
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        out.append({"name": name, "path": path, "index": index})
    return out


def _path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _find_relative_file(root: Path, module_id: str) -> Optional[Path]:
    for candidate in _candidate_ids(module_id):
        target = (root / candidate).resolve()
        if _path_inside(target, root) and target.exists() and target.is_file():
            return target
    return None


def _find_bare_file(root: Path, specifier: str) -> Optional[Path]:
    clean = str(specifier or "").replace("\\", "/").strip().strip("/")
    if not clean or clean.startswith(".") or clean.startswith("/"):
        return None
    for candidate in _candidate_ids(clean):
        target = (root / candidate).resolve()
        if _path_inside(target, root) and target.exists() and target.is_file():
            return target
    return None


def _lib_module_id(root_index: int, root: Path, path: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    rel_text = str(rel).replace("\\", "/")
    return f"lib:{root_index}:{rel_text}"


def _module_import_declaration(clause: str, module_expression: str, *, temp_name: str = "__workflowJsImported") -> tuple[str, List[Dict[str, str]], List[str]]:
    default_binding, namespace_binding, named_bindings = _split_import_clause(clause)
    declarations: List[str] = []
    bindings: List[Dict[str, str]] = []
    invalid: List[str] = []
    if default_binding or namespace_binding or named_bindings:
        declarations.append(f"const {temp_name} = {module_expression};")
    if default_binding:
        if _is_identifier(default_binding):
            declarations.append(f"const {default_binding} = {temp_name}.default;")
            bindings.append({"kind": "default", "local": default_binding})
        else:
            invalid.append(default_binding)
    if namespace_binding:
        if _is_identifier(namespace_binding):
            declarations.append(f"const {namespace_binding} = {temp_name};")
            bindings.append({"kind": "namespace", "local": namespace_binding})
        else:
            invalid.append(namespace_binding)
    if named_bindings:
        parts: List[str] = []
        for imported, local in named_bindings:
            if not _is_identifier(imported) or not _is_identifier(local):
                invalid.append(local or imported)
                continue
            parts.append(imported if imported == local else f"{imported}: {local}")
            bindings.append({"kind": "named", "imported": imported, "local": local})
        if parts:
            declarations.append(f"const {{ {', '.join(parts)} }} = {temp_name};")
    return "\n".join(declarations), bindings, invalid


def _is_relative_specifier(specifier: str) -> bool:
    return str(specifier or "").startswith(("./", "../"))


def _is_node_builtin_specifier(specifier: str) -> bool:
    value = str(specifier or "").strip()
    if value.startswith("node:"):
        return True
    return value in _NODE_BUILTINS


def _make_module_row(module_id: str, source: str, *, origin_path: Optional[Path] = None, kind: str = "inline", root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": _module_id(module_id) if not str(module_id).startswith("lib:") else str(module_id),
        "source": str(source or ""),
        "origin_path": origin_path,
        "kind": kind,
        "root": root,
    }


def _read_module_file(path: Path, module_id: str, *, kind: str, root: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _make_module_row(module_id, path.read_text(encoding="utf-8"), origin_path=path, kind=kind, root=root)


def _transform_exports(source: str) -> tuple[str, List[str], List[Dict[str, str]]]:
    appended: List[str] = []
    rejected: List[Dict[str, str]] = []
    default_index = 0

    for match in _EXPORT_FROM_RE.finditer(source):
        rejected.append({"reason": "export_from_unsupported", "statement": match.group(0).strip()})

    def _replace_named(match: re.Match[str]) -> str:
        for imported, exported in _parse_named_imports(match.group("items")):
            if not _is_identifier(imported) or not _is_identifier(exported):
                rejected.append({"reason": "export_binding_invalid", "statement": match.group(0).strip()})
                continue
            appended.append(f"exports.{exported} = {imported};")
        return ""

    source = _EXPORT_NAMED_RE.sub(_replace_named, source)

    def _replace_decl(match: re.Match[str]) -> str:
        name = match.group("name")
        appended.append(f"exports.{name} = {name};")
        return f"{match.group('indent')}{match.group('kind')} {name}"

    source = _EXPORT_DECL_RE.sub(_replace_decl, source)

    def _replace_default_named(match: re.Match[str]) -> str:
        name = match.group("name")
        appended.append(f"exports.default = {name};")
        return f"{match.group('indent')}{match.group('kind')} {name}"

    source = _EXPORT_DEFAULT_NAMED_RE.sub(_replace_default_named, source)

    def _replace_default_expr(match: re.Match[str]) -> str:
        nonlocal default_index
        default_index += 1
        name = f"__workflowJsDefault{default_index}"
        appended.append(f"exports.default = {name};")
        return f"{match.group('indent')}const {name} = {match.group('expr')};\n"

    source = _EXPORT_DEFAULT_EXPR_RE.sub(_replace_default_expr, source)
    if appended:
        source = source.rstrip() + "\n" + "\n".join(appended) + "\n"
    return source, appended, rejected


def _line_number_at(source: str, index: int) -> int:
    return str(source or "").count("\n", 0, max(0, int(index or 0))) + 1


def _split_lines(text: str) -> List[str]:
    if text == "":
        return []
    return str(text).splitlines()


def _append_mapped_text(parts: List[str], line_map: List[Optional[int]], text: str, *, source_line: int) -> None:
    lines = _split_lines(text)
    for offset, line in enumerate(lines):
        parts.append(line)
        line_map.append(int(source_line) + offset)


def _append_generated_text(parts: List[str], line_map: List[Optional[int]], text: str, *, source_line: Optional[int] = None) -> None:
    for line in _split_lines(text):
        parts.append(line)
        line_map.append(source_line)


def _segment_marker(event: str, *, kind: str, name: str, module: Optional[str] = None) -> str:
    meta = {"kind": str(kind), "name": str(name)}
    if module is not None:
        meta["module"] = str(module)
    return f"/* workflow-js-bundle-segment-{event} {json.dumps(meta, sort_keys=True)} */"


def describe_workflow_js_bundle_source(source: str) -> List[Dict[str, Any]]:
    """Return segment ranges embedded in a generated JS module bundle."""

    segments: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []
    lines = str(source or "").splitlines()
    for index, line in enumerate(lines, start=1):
        begin = _SEGMENT_BEGIN_RE.match(line.strip())
        if begin:
            try:
                meta = json.loads(begin.group("meta"))
            except json.JSONDecodeError:
                meta = {"kind": "unknown", "name": "unknown"}
            stack.append(
                {
                    **dict(meta or {}),
                    "generated_start_line": index,
                    "content_start_line": index + 1,
                }
            )
            continue
        end = _SEGMENT_END_RE.match(line.strip())
        if end and stack:
            segment = stack.pop()
            segment["content_end_line"] = index - 1
            segment["generated_end_line"] = index
            segments.append(segment)
    return segments


def extract_workflow_js_bundle_segment(bundle_or_source: Any, name: str) -> Optional[str]:
    """Extract a marked bundle segment by segment name or module id."""

    source = str(bundle_or_source.get("module_source") if isinstance(bundle_or_source, Mapping) else bundle_or_source or "")
    if not source:
        return None
    lines = source.splitlines()
    for segment in describe_workflow_js_bundle_source(source):
        if str(segment.get("name") or "") != str(name) and str(segment.get("module") or "") != str(name):
            continue
        start = int(segment.get("content_start_line") or 0)
        end = int(segment.get("content_end_line") or 0)
        if start <= 0 or end < start:
            return ""
        return "\n".join(lines[start - 1 : end])
    return None


def resolve_workflow_js_bundle_line(bundle_or_source: Any, line_number: int) -> Dict[str, Any]:
    """Resolve a generated bundle line to segment and original module context."""

    line = int(line_number or 0)
    if isinstance(bundle_or_source, Mapping):
        for item in list(bundle_or_source.get("bundle_line_map") or []):
            row = dict(item or {})
            if int(row.get("generated_line") or 0) == line:
                return row
        source = str(bundle_or_source.get("module_source") or "")
    else:
        source = str(bundle_or_source or "")
    for segment in describe_workflow_js_bundle_source(source):
        if int(segment.get("generated_start_line") or 0) <= line <= int(segment.get("generated_end_line") or 0):
            out = dict(segment)
            out["generated_line"] = line
            if str(segment.get("kind") or "") == "module" and line >= int(segment.get("content_start_line") or 0):
                out["original_line"] = line - int(segment.get("content_start_line") or line) + 1
            return out
    return {"generated_line": line, "kind": "unknown", "name": None, "module": None, "original_line": None}


def build_workflow_js_module_bundle(
    *,
    entry_module: str,
    modules: Optional[Iterable[Mapping[str, Any]]] = None,
    local_roots: Optional[Iterable[Any]] = None,
    allowed_lib_roots: Optional[Iterable[Any]] = None,
    disabled_lib_roots: Optional[Iterable[Any]] = None,
    bridge_imports: Optional[Mapping[str, Any]] = None,
    host_description: Optional[Mapping[str, Any]] = None,
    sandbox_policy: Optional[Mapping[str, Any]] = None,
    enabled_imports: Optional[Iterable[str]] = None,
    disabled_imports: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Bundle a constrained local JS module graph into one JS worker script.

    This helper supports relative local modules, allowed library roots, disabled
    library roots for diagnostics, and the same ``@host/...`` bridge imports as
    ``build_workflow_js_bundle``. It deliberately does not emulate Node/npm.
    """

    bridges = (
        workflow_js_host_bridge_imports(
            host_description=host_description,
            sandbox_policy=sandbox_policy,
            enabled_imports=enabled_imports,
            disabled_imports=disabled_imports,
        )
        if bridge_imports is None
        else _normalize_bridge_imports(bridge_imports)
    )
    if bridge_imports is not None and (enabled_imports or disabled_imports):
        explicit_enabled = _clean_set(enabled_imports)
        explicit_disabled = _clean_set(disabled_imports)
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

    module_map: Dict[str, Dict[str, Any]] = {}
    for row in list(modules or []):
        module_id = _module_id(str(row.get("id") or row.get("name") or row.get("path") or ""))
        module_map[module_id] = _make_module_row(module_id, str(row.get("source") or ""))

    local_root_rows = _root_rows(local_roots)
    allowed_root_rows = _root_rows(allowed_lib_roots)
    disabled_root_rows = _root_rows(disabled_lib_roots)
    entry_id = _module_id(entry_module)

    if entry_id not in module_map:
        for root in local_root_rows:
            path = _find_relative_file(root["path"], entry_id)
            if path is not None:
                module_map[entry_id] = _read_module_file(path, entry_id, kind="local", root=root)
                break

    resolved_allowed_imports: set[str] = set()
    resolved_disabled_imports: set[str] = set()
    unresolved_imports: set[str] = set()
    resolved_modules: set[str] = set()
    rejected_imports: List[Dict[str, Any]] = []
    module_edges: Dict[str, Dict[str, str]] = {}
    visiting: set[str] = set()

    def _resolve_relative(row: Dict[str, Any], specifier: str) -> Optional[str]:
        importer_id = str(row["id"])
        if str(row.get("kind")) == "lib":
            root = dict(row.get("root") or {})
            origin_path = row.get("origin_path")
            if isinstance(origin_path, Path) and root.get("path") is not None:
                base = origin_path.parent
                for suffix in ["", ".js", "/index.js"]:
                    target = (base / f"{specifier}{suffix}").resolve()
                    if _path_inside(target, root["path"]) and target.exists() and target.is_file():
                        module_id = _lib_module_id(int(root.get("index") or 0), root["path"], target)
                        if module_id not in module_map:
                            module_map[module_id] = _read_module_file(target, module_id, kind="lib", root=root)
                        return module_id
            return None
        candidate = _relative_module_id(importer_id, specifier)
        for candidate_id in _candidate_ids(candidate):
            if candidate_id in module_map:
                return candidate_id
        for root in local_root_rows:
            path = _find_relative_file(root["path"], candidate)
            if path is not None:
                module_id = _module_id(str(path.relative_to(root["path"])).replace("\\", "/"))
                if module_id not in module_map:
                    module_map[module_id] = _read_module_file(path, module_id, kind="local", root=root)
                return module_id
        return None

    def _resolve_bare(specifier: str) -> tuple[str, Optional[str]]:
        for root in disabled_root_rows:
            path = _find_bare_file(root["path"], specifier)
            if path is not None:
                return "disabled", None
        for root in allowed_root_rows:
            path = _find_bare_file(root["path"], specifier)
            if path is not None:
                module_id = _lib_module_id(int(root.get("index") or 0), root["path"], path)
                if module_id not in module_map:
                    module_map[module_id] = _read_module_file(path, module_id, kind="lib", root=root)
                return "allowed", module_id
        return "unresolved", None

    def _walk(module_id: str, stack: List[str]) -> None:
        if module_id in visiting:
            rejected_imports.append({"module": module_id, "specifier": module_id, "reason": "local_import_cycle", "path": stack + [module_id]})
            return
        row = module_map.get(module_id)
        if row is None:
            unresolved_imports.add(module_id)
            return
        if module_id in resolved_modules:
            return
        visiting.add(module_id)
        edges: Dict[str, str] = {}
        source = str(row.get("source") or "")
        if _DYNAMIC_IMPORT_RE.search(source):
            rejected_imports.append({"module": module_id, "specifier": "import(...)", "reason": "dynamic_import_unsupported"})
        if _REQUIRE_RE.search(source):
            rejected_imports.append({"module": module_id, "specifier": "require(...)", "reason": "require_unsupported"})
        for export_reject in _EXPORT_FROM_RE.finditer(source):
            rejected_imports.append({"module": module_id, "specifier": "", "reason": "export_from_unsupported", "statement": export_reject.group(0).strip()})
        for item in _import_matches(source):
            specifier = str(item["specifier"])
            bridge = bridges.get(specifier)
            if bridge is not None:
                if bridge.enabled:
                    resolved_allowed_imports.add(specifier)
                else:
                    resolved_disabled_imports.add(specifier)
                continue
            if _is_node_builtin_specifier(specifier):
                rejected_imports.append({"module": module_id, "specifier": specifier, "reason": "node_builtin_unsupported"})
                continue
            if _is_relative_specifier(specifier):
                try:
                    target_id = _resolve_relative(row, specifier)
                except ValueError:
                    rejected_imports.append({"module": module_id, "specifier": specifier, "reason": "relative_import_escape"})
                    continue
                if target_id is None:
                    unresolved_imports.add(specifier)
                else:
                    edges[specifier] = target_id
                    resolved_allowed_imports.add(specifier)
                    _walk(target_id, stack + [module_id])
                continue
            status, target_id = _resolve_bare(specifier)
            if status == "disabled":
                resolved_disabled_imports.add(specifier)
            elif status == "allowed" and target_id:
                edges[specifier] = target_id
                resolved_allowed_imports.add(specifier)
                _walk(target_id, stack + [module_id])
            else:
                unresolved_imports.add(specifier)
        visiting.remove(module_id)
        resolved_modules.add(module_id)
        module_edges[module_id] = edges

    _walk(entry_id, [])

    def _transform_module(row: Dict[str, Any]) -> tuple[str, List[Optional[int]]]:
        source = str(row.get("source") or "")
        output_lines: List[str] = []
        output_line_map: List[Optional[int]] = []
        cursor = 0
        for item in _import_matches(source):
            start = int(item["start"])
            end = int(item["end"])
            specifier = str(item["specifier"])
            bridge = bridges.get(specifier)
            _append_mapped_text(output_lines, output_line_map, source[cursor:start], source_line=_line_number_at(source, cursor))
            if bridge is not None and bridge.enabled:
                if item["kind"] == "side_effect":
                    _append_generated_text(
                        output_lines,
                        output_line_map,
                        f"/* workflow-js-module-bundle host bridge import: {specifier} */",
                        source_line=_line_number_at(source, start),
                    )
                else:
                    declaration, _bindings, invalid = _declaration_for_import(str(item["clause"]), bridge)
                    _append_generated_text(
                        output_lines,
                        output_line_map,
                        str(item["statement"]).rstrip("\r\n") if invalid else declaration,
                        source_line=_line_number_at(source, start),
                    )
            elif specifier in module_edges.get(str(row["id"]), {}):
                target_id = module_edges[str(row["id"])][specifier]
                if item["kind"] == "side_effect":
                    _append_generated_text(
                        output_lines,
                        output_line_map,
                        f"__workflowJsRequire({json.dumps(target_id)});",
                        source_line=_line_number_at(source, start),
                    )
                else:
                    declaration, _bindings, invalid = _module_import_declaration(
                        str(item["clause"]),
                        f"__workflowJsRequire({json.dumps(target_id)})",
                        temp_name=f"__workflowJsImported{len(output_lines)}",
                    )
                    _append_generated_text(
                        output_lines,
                        output_line_map,
                        str(item["statement"]).rstrip("\r\n") if invalid else declaration,
                        source_line=_line_number_at(source, start),
                    )
            else:
                _append_mapped_text(output_lines, output_line_map, str(item["statement"]), source_line=_line_number_at(source, start))
            cursor = end
        _append_mapped_text(output_lines, output_line_map, source[cursor:], source_line=_line_number_at(source, cursor))
        raw_transformed = "\n".join(output_lines)
        transformed, _exports, export_rejected = _transform_exports(raw_transformed)
        for rejected in export_rejected:
            rejected_imports.append({"module": str(row["id"]), **rejected})
        transformed_line_count = len(transformed.splitlines())
        if transformed_line_count > len(output_line_map):
            output_line_map.extend([None] * (transformed_line_count - len(output_line_map)))
        elif transformed_line_count < len(output_line_map):
            output_line_map = output_line_map[:transformed_line_count]
        return transformed, output_line_map

    ok = bool(entry_id in resolved_modules) and not resolved_disabled_imports and not unresolved_imports and not rejected_imports
    bundle_lines: List[str] = []
    bundle_line_map: List[Dict[str, Any]] = []

    def _emit_line(text: str, *, kind: str, name: str, module_id: Optional[str] = None, original_line: Optional[int] = None) -> None:
        bundle_lines.append(text)
        bundle_line_map.append(
            {
                "generated_line": len(bundle_lines),
                "kind": kind,
                "name": name,
                "module": module_id,
                "original_line": original_line,
            }
        )

    def _emit_segment_begin(kind: str, name: str, module_id: Optional[str] = None) -> None:
        _emit_line(_segment_marker("begin", kind=kind, name=name, module=module_id), kind=kind, name=name, module_id=module_id)

    def _emit_segment_end(kind: str, name: str, module_id: Optional[str] = None) -> None:
        _emit_line(_segment_marker("end", kind=kind, name=name, module=module_id), kind=kind, name=name, module_id=module_id)

    _emit_segment_begin("runtime", "runtime:prelude")
    for line in [
        "(function () {",
        "const __workflowJsModules = Object.create(null);",
        "const __workflowJsCache = Object.create(null);",
        "function __workflowJsDefine(id, factory) { __workflowJsModules[id] = factory; }",
        "function __workflowJsRequire(id) {",
        "  if (__workflowJsCache[id]) return __workflowJsCache[id].exports;",
        "  const factory = __workflowJsModules[id];",
        "  if (typeof factory !== 'function') throw new Error('workflow_js_module_not_found:' + id);",
        "  const module = { exports: {} };",
        "  __workflowJsCache[id] = module;",
        "  factory(module, module.exports, __workflowJsRequire);",
        "  return module.exports;",
        "}",
    ]:
        _emit_line(line, kind="runtime", name="runtime:prelude")
    _emit_segment_end("runtime", "runtime:prelude")
    for module_id in sorted(resolved_modules):
        row = module_map[module_id]
        transformed, transformed_line_map = _transform_module(row)
        _emit_line(f"__workflowJsDefine({json.dumps(module_id)}, function(module, exports, __workflowJsRequire) {{", kind="module", name=module_id, module_id=module_id)
        _emit_segment_begin("module", module_id, module_id)
        for index, line in enumerate(transformed.rstrip().splitlines(), start=1):
            original_line = transformed_line_map[index - 1] if index - 1 < len(transformed_line_map) else None
            _emit_line(line, kind="module", name=module_id, module_id=module_id, original_line=original_line)
        _emit_segment_end("module", module_id, module_id)
        _emit_line("});", kind="module", name=module_id, module_id=module_id)
    _emit_segment_begin("runtime", "runtime:entry")
    for line in [
        f"Object.assign(globalThis.exports, __workflowJsRequire({json.dumps(entry_id)}));",
        "})();",
        "",
    ]:
        _emit_line(line, kind="runtime", name="runtime:entry")
    _emit_segment_end("runtime", "runtime:entry")
    module_source = "\n".join(bundle_lines) + "\n"
    module_sha256 = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
    bundle_segments = describe_workflow_js_bundle_source(module_source)
    return {
        "ok": ok,
        "entry_module": entry_id,
        "module_source": module_source,
        "module_sha256": module_sha256,
        "bundle_segments": bundle_segments,
        "bundle_line_map": bundle_line_map,
        "resolved_modules": sorted(resolved_modules),
        "resolved_allowed_imports": sorted(resolved_allowed_imports),
        "resolved_disabled_imports": sorted(resolved_disabled_imports),
        "unresolved_imports": sorted(unresolved_imports),
        "rejected_imports": rejected_imports,
    }


__all__ = [
    "WorkflowJsBridgeImport",
    "build_workflow_js_bundle",
    "build_workflow_js_module_bundle",
    "build_workflow_js_bundle_request",
    "describe_workflow_js_bundle_source",
    "extract_workflow_js_bundle_segment",
    "resolve_workflow_js_bundle_line",
    "workflow_js_host_bridge_imports",
]
