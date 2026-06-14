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
    assert "artifact_inputs" in contract["request_fields"]
    assert "artifact_outputs" in contract["request_fields"]
    assert "state_patch" in contract["response_fields"]
    assert "progress" in contract["stream_event_types"]
    assert "log" in contract["stream_event_types"]
    assert contract["artifact_contract"]["ref_format"] == "@alias/relative/path"
    assert "inline" in contract["artifact_contract"]["input_kinds"]


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
            "artifact_inputs": [{"name": "seed", "ref": "@artifacts/a/seed.txt"}],
            "artifact_outputs": [{"name": "report", "filename": "report.txt"}],
        }
    )

    assert out["request_id"].startswith("workflow-python-node-")
    assert out["operation"] == "run"
    assert out["export_name"] == "run"
    assert out["payload"] == {"value": 1}
    assert out["limits"]["timeout_ms"] == 1000
    assert out["artifact_inputs"] == [{"name": "seed", "ref": "@artifacts/a/seed.txt"}]
    assert out["artifact_outputs"] == [{"name": "report", "filename": "report.txt"}]


def test_normalize_node_request_maps_operation_from_export_name() -> None:
    out = normalize_workflow_python_node_request(
        {
            "request_id": "req",
            "module_source": "def run(payload):\n    return payload\n",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "export_name": "run",
            "payload": None,
            "provenance": "bad",
            "limits": "bad",
            "policy": "bad",
            "python": "bad",
        }
    )

    assert out["operation"] == "run"
    assert out["export_name"] == "run"
    assert out["payload"] is None
    assert out["provenance"] == {}
    assert out["limits"] == {}
    assert out["policy"] == {}
    assert out["python"] == {}


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


def test_validate_node_request_requires_export_or_operation() -> None:
    out = validate_workflow_python_node_request(
        {
            "module_source": "def run(payload):\n    return payload\n",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
        }
    )

    assert out["status"] == "error"
    assert out["missing"] == ["export_name_or_operation"]


def test_validate_node_request_accepts_payload_omission_as_empty_object() -> None:
    out = validate_workflow_python_node_request(
        {
            "module_source": "def run(payload):\n    return payload\n",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "operation": "run",
        }
    )

    assert out["status"] == "ok"
    assert out["request"]["payload"] == {}


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
            "limits": {"output_limit_bytes": 3},
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
    assert out["logs"]["output_limit_bytes"] == 3
    assert out["logs"]["stdout_truncated"] is False
    assert out["error"]["code"] == "workflow_python_node_profile_not_implemented"
    assert out["audit"]["package_id"] == "pkg"
