"""Workflow Python request/response contract helpers."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .runtime_base import hosted_log_summary


NODE_REQUEST_FIELDS = [
    "request_id",
    "module_source",
    "module_sha256",
    "package_id",
    "workflow_id",
    "package_source_digest",
    "export_name",
    "operation",
    "execution_mode",
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


def workflow_python_node_contract() -> Dict[str, Any]:
    return {
        "profile": "node",
        "request_fields": list(NODE_REQUEST_FIELDS),
        "response_fields": list(NODE_RESPONSE_FIELDS),
        "stream_event_types": [
            "started",
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
        "limits": ["timeout_ms", "output_limit_bytes", "memory_limit_mb"],
        "artifact_contract": {
            "ref_format": "@alias/relative/path",
            "default_roots": ["@artifacts"],
            "policy_root_field": "sandbox.artifact_roots",
            "input_kinds": ["ref", "inline", "inline_zip"],
            "output_kinds": ["ref", "inline", "inline_zip_export"],
            "path_selection": ["path_mask", "mask", "recursive"],
            "input_metadata_advisory": ["max_bytes", "count", "ttl", "encoding"],
        },
    }


def normalize_workflow_python_node_request(request: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    req = _dict(request)
    operation = _clean(req.get("operation"))
    export_name = _clean(req.get("export_name"))
    execution_mode = _clean(req.get("execution_mode") or _dict(req.get("python")).get("execution_mode")).lower() or "module"
    if execution_mode not in {"module", "snippet", "project"}:
        execution_mode = "module"
    normalized = {
        "request_id": _clean(req.get("request_id")) or f"workflow-python-node-{int(time.time() * 1000)}",
        "module_source": str(req.get("module_source") or ""),
        "module_sha256": _clean(req.get("module_sha256")),
        "package_id": _clean(req.get("package_id")),
        "workflow_id": _clean(req.get("workflow_id")),
        "package_source_digest": _clean(req.get("package_source_digest")),
        "export_name": export_name or operation,
        "operation": operation or export_name,
        "execution_mode": execution_mode,
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


def workflow_python_node_not_implemented_response(
    *,
    environment_key: str = "",
    engine_id: str = "",
    request: Optional[Dict[str, Any]] = None,
    reason: str = "workflow_python_node_profile_not_implemented",
) -> Dict[str, Any]:
    validation = validate_workflow_python_node_request(request)
    normalized = dict(validation.get("request") or {})
    return {
        "status": "error",
        "ok": False,
        "profile": "node",
        "environment_key": _clean(environment_key) or None,
        "engine_id": _clean(engine_id) or None,
        "request_id": _clean(normalized.get("request_id")) or None,
        "reason": reason,
        "error": {
            "code": reason,
            "message": "workflow_python(profile=node) contract is defined, but the node worker is not implemented yet",
            "missing_request_fields": list(validation.get("missing") or []),
        },
        "output": None,
        "state_patch": None,
        "artifacts": [],
        "artifact_store": {
            "status": "unavailable",
            "reason": "artifact_store_not_implemented",
            "message": "artifact refs are part of the node-profile contract, but no workflow artifact store is wired yet",
        },
        "progress": None,
        "logs": hosted_log_summary(
            max_bytes=int(_dict(normalized.get("limits")).get("output_limit_bytes") or 4096)
        ),
        "metrics": {},
        "audit": {
            "package_id": _clean(normalized.get("package_id")) or None,
            "workflow_id": _clean(normalized.get("workflow_id")) or None,
            "package_source_digest": _clean(normalized.get("package_source_digest")) or None,
            "module_sha256": _clean(normalized.get("module_sha256")) or None,
            "provenance": _dict(normalized.get("provenance")),
        },
        "contract": dict(validation.get("contract") or workflow_python_node_contract()),
    }


__all__ = [
    "normalize_workflow_python_node_request",
    "validate_workflow_python_node_request",
    "workflow_python_node_contract",
    "workflow_python_node_not_implemented_response",
]
