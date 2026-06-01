from __future__ import annotations

from hosting.sandbox.workflow_python_contract import (
    normalize_workflow_python_node_request,
    validate_workflow_python_node_request,
    workflow_python_node_contract,
    workflow_python_node_not_implemented_response,
)


def test_node_contract_lists_request_response_and_stream_fields() -> None:
    contract = workflow_python_node_contract()

    assert contract["profile"] == "node"
    assert "module_source" in contract["request_fields"]
    assert "state_patch" in contract["response_fields"]
    assert "progress" in contract["stream_event_types"]


def test_normalize_node_request_maps_export_and_operation() -> None:
    out = normalize_workflow_python_node_request(
        {
            "module_source": "def run(payload):\n    return payload\n",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "export_name": "run",
            "payload": {"value": 1},
            "limits": {"timeout_ms": 1000},
        }
    )

    assert out["request_id"].startswith("workflow-python-node-")
    assert out["operation"] == "run"
    assert out["export_name"] == "run"
    assert out["payload"] == {"value": 1}
    assert out["limits"]["timeout_ms"] == 1000


def test_validate_node_request_reports_missing_fields() -> None:
    out = validate_workflow_python_node_request({"operation": "run"})

    assert out["status"] == "error"
    assert out["missing"] == [
        "module_source",
        "module_sha256",
        "package_id",
        "workflow_id",
        "package_source_digest",
    ]


def test_node_not_implemented_response_uses_node_envelope() -> None:
    out = workflow_python_node_not_implemented_response(
        environment_key="env-node",
        engine_id="wf-node",
        request={
            "request_id": "req-node",
            "module_source": "def run(payload):\n    return payload\n",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
            "payload": {},
        },
    )

    assert out["status"] == "error"
    assert out["ok"] is False
    assert out["profile"] == "node"
    assert out["environment_key"] == "env-node"
    assert out["request_id"] == "req-node"
    assert out["output"] is None
    assert out["state_patch"] is None
    assert out["artifacts"] == []
    assert out["artifact_store"]["reason"] == "artifact_store_not_implemented"
    assert out["error"]["code"] == "workflow_python_node_profile_not_implemented"
    assert out["audit"]["package_id"] == "pkg"
