"""Workflow Python request/response contract helpers."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


NODE_REQUEST_FIELDS = [
    "request_id",
    "module_source",
    "module_sha256",
    "package_id",
    "workflow_id",
    "package_source_digest",
    "export_name",
    "operation",
    "payload",
    "provenance",
    "limits",
    "policy",
    "python",
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
            "export_name_or_operation",
            "payload",
        ],
        "limits": ["timeout_ms", "output_limit_bytes", "memory_limit_mb"],
    }


def normalize_workflow_python_node_request(request: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    req = _dict(request)
    operation = _clean(req.get("operation"))
    export_name = _clean(req.get("export_name"))
    normalized = {
        "request_id": _clean(req.get("request_id")) or f"workflow-python-node-{int(time.time() * 1000)}",
        "module_source": str(req.get("module_source") or ""),
        "module_sha256": _clean(req.get("module_sha256")),
        "package_id": _clean(req.get("package_id")),
        "workflow_id": _clean(req.get("workflow_id")),
        "package_source_digest": _clean(req.get("package_source_digest")),
        "export_name": export_name or operation,
        "operation": operation or export_name,
        "payload": req.get("payload") if "payload" in req else {},
        "provenance": _dict(req.get("provenance")),
        "limits": _dict(req.get("limits")),
        "policy": _dict(req.get("policy")),
        "python": _dict(req.get("python")),
    }
    return normalized


def validate_workflow_python_node_request(request: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = normalize_workflow_python_node_request(request)
    missing: List[str] = []
    for field in ["module_source", "module_sha256", "package_id", "workflow_id", "package_source_digest"]:
        if not _clean(normalized.get(field)):
            missing.append(field)
    if not (_clean(normalized.get("export_name")) or _clean(normalized.get("operation"))):
        missing.append("export_name_or_operation")
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
        "logs": {"stdout": "", "stderr": "", "summary": ""},
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
