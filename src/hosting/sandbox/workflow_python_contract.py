"""Workflow Python request/response contract helpers."""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

NODE_REQUEST_FIELDS = [
    "request_id",
    "module_source",
    "module_sha256",
    "code_revision",
    "package_id",
    "workflow_id",
    "package_source_digest",
    "export_name",
    "operation",
    "execution_mode",
    "instance_state_mode",
    "project",
    "payload",
    "provenance",
    "limits",
    "policy",
    "python",
    "artifact_inputs",
    "artifact_outputs",
]

NODE_RESPONSE_FIELDS = [
    "ok",
    "status",
    "output",
    "state_patch",
    "artifacts",
    "artifact_store",
    "progress",
    "logs",
    "metrics",
    "error",
    "audit",
]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _clean_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def workflow_python_node_contract() -> Dict[str, Any]:
    return {
        "profile": "node",
        "request_fields": list(NODE_REQUEST_FIELDS),
        "response_fields": list(NODE_RESPONSE_FIELDS),
        "stream_event_types": [
            "started",
            "heartbeat",
            "progress",
            "log",
            "stdout",
            "stderr",
            "artifact",
            "result",
            "error",
            "canceled",
            "done",
        ],
        "required_request_fields": [
            "module_source",
            "module_sha256",
            "package_id",
            "workflow_id",
            "package_source_digest",
            "export_name_or_operation_unless_snippet_or_project",
            "payload",
        ],
        "execution_modes": ["module", "snippet", "project"],
        "instance_state_modes": ["ephemeral", "persistent_module"],
        "request_templates": ["module_function", "snippet", "staged_project", "uv_project"],
        "limits": ["timeout_ms", "output_limit_bytes", "memory_limit_mb", "heartbeat_interval_ms", "stream_max_events"],
        "job_lifecycle_states": ["submitted", "running", "ok", "error", "timeout", "canceled"],
        "artifact_contract": {
            "ref_format": "@alias/relative/path",
            "default_roots": ["@artifacts"],
            "policy_root_field": "sandbox.artifact_roots",
            "input_kinds": ["ref", "inline", "inline_zip"],
            "output_kinds": ["ref", "inline", "inline_zip_export"],
            "path_selection": ["path_mask", "mask", "recursive"],
            "input_metadata_advisory": ["max_bytes", "count", "ttl", "encoding"],
        },
        "host_api": {
            "contract": "hosting.workflow_python.node.host_api.v1",
            "transport": "workflow_python_node_worker_ipc_control_channel",
            "transport_capabilities": {
                "framed": True,
                "host_call_id": True,
                "async_capable": True,
                "out_of_order_responses": True,
                "sync_handlers": True,
                "async_handlers": True,
            },
            "methods": ["host.describe", "fs.list", "fs.read_text", "fs.write_text", "fs.mkdir", "fs.stat", "http.fetch"],
            "policy_gated_methods": ["fs.list", "fs.read_text", "fs.write_text", "fs.mkdir", "fs.stat", "http.fetch"],
            "discovery": {
                "method": "host.describe",
                "includes": ["methods", "method_descriptions", "args_schema", "result_schema", "permissions", "roots", "policy", "transport"],
            },
            "filesystem_model": "artifact_roots",
            "readable_roots": "declared artifact inputs and outputs",
            "writable_roots": "declared artifact outputs only",
            "http": "policy_gated_brokered_http",
        },
    }


def build_workflow_python_node_module_request(
    *,
    module_source: str,
    operation: str = "",
    export_name: str = "",
    request_id: str = "",
    package_id: str = "workflow-python-node",
    workflow_id: str = "workflow",
    package_source_digest: str = "",
    payload: Any = None,
    code_revision: str = "",
    provenance: Optional[Dict[str, Any]] = None,
    limits: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
    python: Optional[Dict[str, Any]] = None,
    instance_state_mode: str = "",
    artifact_inputs: Optional[list[Dict[str, Any]]] = None,
    artifact_outputs: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    source = str(module_source or "")
    digest = _clean(package_source_digest) or _sha256_text(source)
    op = _clean(operation) or _clean(export_name)
    return normalize_workflow_python_node_request(
        {
            "request_id": request_id,
            "execution_mode": "module",
            "module_source": source,
            "module_sha256": _sha256_text(source),
            "code_revision": _clean(code_revision) or _sha256_text(source),
            "package_id": package_id,
            "workflow_id": workflow_id,
            "package_source_digest": digest,
            "operation": op,
            "export_name": _clean(export_name) or op,
            "instance_state_mode": _clean(instance_state_mode),
            "payload": payload if payload is not None else {},
            "provenance": _clean_dict(provenance),
            "limits": _clean_dict(limits),
            "policy": _clean_dict(policy),
            "python": _clean_dict(python),
            "artifact_inputs": list(artifact_inputs or []),
            "artifact_outputs": list(artifact_outputs or []),
        }
    )


def build_workflow_python_node_snippet_request(
    *,
    source: str,
    request_id: str = "",
    package_id: str = "workflow-python-snippet",
    workflow_id: str = "workflow",
    package_source_digest: str = "",
    payload: Any = None,
    code_revision: str = "",
    provenance: Optional[Dict[str, Any]] = None,
    limits: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
    python: Optional[Dict[str, Any]] = None,
    artifact_inputs: Optional[list[Dict[str, Any]]] = None,
    artifact_outputs: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    snippet = str(source or "")
    digest = _clean(package_source_digest) or _sha256_text(snippet)
    return normalize_workflow_python_node_request(
        {
            "request_id": request_id,
            "execution_mode": "snippet",
            "module_source": snippet,
            "module_sha256": _sha256_text(snippet),
            "code_revision": _clean(code_revision) or _sha256_text(snippet),
            "package_id": package_id,
            "workflow_id": workflow_id,
            "package_source_digest": digest,
            "payload": payload if payload is not None else {},
            "provenance": _clean_dict(provenance),
            "limits": _clean_dict(limits),
            "policy": _clean_dict(policy),
            "python": _clean_dict(python),
            "artifact_inputs": list(artifact_inputs or []),
            "artifact_outputs": list(artifact_outputs or []),
        }
    )


def build_workflow_python_node_project_request(
    *,
    project_ref: str,
    entrypoint: str,
    callable_name: str = "run",
    request_id: str = "",
    package_id: str = "workflow-python-project",
    workflow_id: str = "workflow",
    project_id: str = "",
    project_digest: str = "",
    package_source_digest: str = "",
    payload: Any = None,
    root_input: str = "project",
    working_directory: str = "",
    env: Optional[Dict[str, Any]] = None,
    path_mask: str = "*",
    recursive: bool = True,
    provenance: Optional[Dict[str, Any]] = None,
    limits: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
    python: Optional[Dict[str, Any]] = None,
    artifact_inputs: Optional[list[Dict[str, Any]]] = None,
    artifact_outputs: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    ref = _clean(project_ref)
    project_identity = _clean(project_digest) or _sha256_text("|".join([ref, _clean(entrypoint), _clean(callable_name), _clean(working_directory)]))
    digest = _clean(package_source_digest) or project_identity
    input_name = _clean(root_input) or "project"
    inputs = [dict(row or {}) for row in list(artifact_inputs or []) if isinstance(row, dict)]
    if ref and not any(_clean(row.get("name")) == input_name for row in inputs):
        inputs.append(
            {
                "name": input_name,
                "kind": "ref",
                "ref": ref,
                "path_mask": _clean(path_mask) or "*",
                "recursive": bool(recursive),
            }
        )
    project: Dict[str, Any] = {
        "ref": ref,
        "root_input": input_name,
        "entrypoint": _clean(entrypoint),
        "callable": _clean(callable_name) or "run",
        "project_id": _clean(project_id) or package_id,
        "project_digest": project_identity,
    }
    if _clean(working_directory):
        project["working_directory"] = _clean(working_directory)
    if isinstance(env, dict) and env:
        project["env"] = dict(env)
    return normalize_workflow_python_node_request(
        {
            "request_id": request_id,
            "execution_mode": "project",
            "module_source": "",
            "module_sha256": _sha256_text(""),
            "code_revision": project_identity,
            "package_id": package_id,
            "workflow_id": workflow_id,
            "package_source_digest": digest,
            "project": project,
            "payload": payload if payload is not None else {},
            "provenance": _clean_dict(provenance),
            "limits": _clean_dict(limits),
            "policy": _clean_dict(policy),
            "python": _clean_dict(python),
            "artifact_inputs": inputs,
            "artifact_outputs": list(artifact_outputs or []),
        }
    )


def build_workflow_python_node_uv_project_request(
    *,
    project_ref: str,
    entrypoint: str,
    callable_name: str = "run",
    pyproject_toml: str = "",
    uv_lock: str = "",
    dependency_groups: Optional[list[str]] = None,
    python: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    py = _clean_dict(python)
    uv = dict(py.get("uv") or {}) if isinstance(py.get("uv"), dict) else {}
    if pyproject_toml:
        uv["pyproject_toml"] = str(pyproject_toml)
    if uv_lock:
        uv["uv_lock"] = str(uv_lock)
    if dependency_groups is not None:
        uv["dependency_groups"] = list(dependency_groups or [])
    if uv:
        py["uv"] = uv
    else:
        py["uv_enabled"] = True
    return build_workflow_python_node_project_request(
        project_ref=project_ref,
        entrypoint=entrypoint,
        callable_name=callable_name,
        python=py,
        **kwargs,
    )


def normalize_workflow_python_node_request(request: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    req = _dict(request)
    operation = _clean(req.get("operation"))
    export_name = _clean(req.get("export_name"))
    execution_mode = _clean(req.get("execution_mode") or _dict(req.get("python")).get("execution_mode")).lower() or "module"
    if execution_mode not in {"module", "snippet", "project"}:
        execution_mode = "module"
    instance_state_mode = _clean(req.get("instance_state_mode") or req.get("state_mode") or _dict(req.get("python")).get("instance_state_mode") or _dict(req.get("python")).get("state_mode")).lower().replace("-", "_")
    if instance_state_mode in {"persistent", "module_persistent"}:
        instance_state_mode = "persistent_module"
    if instance_state_mode not in {"", "ephemeral", "persistent_module"}:
        instance_state_mode = "ephemeral"
    normalized = {
        "request_id": _clean(req.get("request_id")) or f"workflow-python-node-{int(time.time() * 1000)}",
        "module_source": str(req.get("module_source") or ""),
        "module_sha256": _clean(req.get("module_sha256")),
        "code_revision": _clean(req.get("code_revision")),
        "package_id": _clean(req.get("package_id")),
        "workflow_id": _clean(req.get("workflow_id")),
        "package_source_digest": _clean(req.get("package_source_digest")),
        "export_name": export_name or operation,
        "operation": operation or export_name,
        "execution_mode": execution_mode,
        "instance_state_mode": instance_state_mode or "ephemeral",
        "project": _dict(req.get("project")),
        "payload": req.get("payload") if "payload" in req else {},
        "provenance": _dict(req.get("provenance")),
        "limits": _dict(req.get("limits")),
        "policy": _dict(req.get("policy")),
        "python": _dict(req.get("python")),
        "artifact_inputs": list(req.get("artifact_inputs") or []) if isinstance(req.get("artifact_inputs"), list) else [],
        "artifact_outputs": list(req.get("artifact_outputs") or []) if isinstance(req.get("artifact_outputs"), list) else [],
    }
    return normalized


def validate_workflow_python_node_request(request: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = normalize_workflow_python_node_request(request)
    missing: List[str] = []
    required = ["module_sha256", "package_id", "workflow_id", "package_source_digest"]
    if normalized.get("execution_mode") != "project":
        required.insert(0, "module_source")
    for field in required:
        if not _clean(normalized.get(field)):
            missing.append(field)
    if normalized.get("execution_mode") not in {"snippet", "project"} and not (_clean(normalized.get("export_name")) or _clean(normalized.get("operation"))):
        missing.append("export_name_or_operation")
    if normalized.get("execution_mode") == "project":
        project = _dict(normalized.get("project"))
        if not _clean(project.get("entrypoint") or project.get("module")):
            missing.append("project.entrypoint")
    return {
        "status": "ok" if not missing else "error",
        "missing": missing,
        "request": normalized,
        "contract": workflow_python_node_contract(),
    }


__all__ = [
    "build_workflow_python_node_module_request",
    "build_workflow_python_node_project_request",
    "build_workflow_python_node_snippet_request",
    "build_workflow_python_node_uv_project_request",
    "normalize_workflow_python_node_request",
    "validate_workflow_python_node_request",
    "workflow_python_node_contract",
]
