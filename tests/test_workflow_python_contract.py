from __future__ import annotations

import hashlib

from hosting.sandbox.workflow_python_contract import (
    build_workflow_python_node_module_request,
    build_workflow_python_node_project_request,
    build_workflow_python_node_snippet_request,
    build_workflow_python_node_uv_project_request,
    normalize_workflow_python_node_request,
    validate_workflow_python_node_request,
    workflow_python_node_contract,
    workflow_python_node_not_implemented_response,
)


def test_node_contract_lists_request_response_and_stream_fields() -> None:
    contract = workflow_python_node_contract()

    assert contract["profile"] == "node"
    assert "module_source" in contract["request_fields"]
    assert "code_revision" in contract["request_fields"]
    assert "execution_mode" in contract["request_fields"]
    assert "project" in contract["request_fields"]
    assert "snippet" in contract["execution_modes"]
    assert "project" in contract["execution_modes"]
    assert "artifact_inputs" in contract["request_fields"]
    assert "artifact_outputs" in contract["request_fields"]
    assert contract["request_templates"] == ["module_function", "snippet", "staged_project", "uv_project"]
    assert "state_patch" in contract["response_fields"]
    assert "progress" in contract["stream_event_types"]
    assert "heartbeat" in contract["stream_event_types"]
    assert "log" in contract["stream_event_types"]
    assert "heartbeat_interval_ms" in contract["limits"]
    assert "stream_max_events" in contract["limits"]
    assert contract["job_lifecycle_states"] == ["submitted", "running", "ok", "error", "timeout", "canceled"]
    assert contract["artifact_contract"]["ref_format"] == "@alias/relative/path"
    assert "inline" in contract["artifact_contract"]["input_kinds"]
    assert contract["host_api"]["http"] == "policy_gated_brokered_http"
    assert contract["host_api"]["transport_capabilities"]["out_of_order_responses"] is True


def test_build_module_request_fills_source_hash_and_defaults() -> None:
    source = "def run(payload):\n    return {'output': payload}\n"

    out = build_workflow_python_node_module_request(
        request_id="req-builder-module",
        module_source=source,
        operation="run",
        payload={"value": 3},
        package_id="pkg",
        workflow_id="wf",
    )

    assert out["request_id"] == "req-builder-module"
    assert out["execution_mode"] == "module"
    assert out["module_sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert out["code_revision"] == out["module_sha256"]
    assert out["package_source_digest"] == out["module_sha256"]
    assert out["operation"] == "run"
    assert out["payload"] == {"value": 3}


def test_build_snippet_request_does_not_require_export() -> None:
    source = "result = {'output': payload}\n"

    out = build_workflow_python_node_snippet_request(
        request_id="req-builder-snippet",
        source=source,
        payload={"ok": True},
    )

    assert out["execution_mode"] == "snippet"
    assert out["operation"] == ""
    assert out["export_name"] == ""
    assert out["module_sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert validate_workflow_python_node_request(out)["status"] == "ok"


def test_build_project_request_adds_default_project_input_and_identity() -> None:
    out = build_workflow_python_node_project_request(
        request_id="req-builder-project",
        project_ref="@project/src",
        entrypoint="pkg.runner",
        callable_name="run",
        package_id="pkg",
        workflow_id="wf",
        project_id="project-a",
        project_digest="project-digest",
        payload={"value": 1},
    )

    assert out["execution_mode"] == "project"
    assert out["module_source"] == ""
    assert out["module_sha256"] == hashlib.sha256(b"").hexdigest()
    assert out["code_revision"] == "project-digest"
    assert out["package_source_digest"] == "project-digest"
    assert out["project"]["project_id"] == "project-a"
    assert out["project"]["project_digest"] == "project-digest"
    assert out["project"]["root_input"] == "project"
    assert out["artifact_inputs"] == [
        {"name": "project", "kind": "ref", "ref": "@project/src", "path_mask": "*", "recursive": True}
    ]
    assert validate_workflow_python_node_request(out)["status"] == "ok"


def test_build_uv_project_request_adds_uv_intent() -> None:
    out = build_workflow_python_node_uv_project_request(
        request_id="req-builder-uv-project",
        project_ref="@project/src",
        entrypoint="pkg.runner",
        pyproject_toml="[project]\nname='demo'\nversion='0.0.0'\n",
        uv_lock="version = 1\n",
        dependency_groups=["dev"],
    )

    assert out["execution_mode"] == "project"
    assert out["python"]["uv"]["pyproject_toml"].startswith("[project]")
    assert out["python"]["uv"]["uv_lock"] == "version = 1\n"
    assert out["python"]["uv"]["dependency_groups"] == ["dev"]


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
    assert out["code_revision"] == ""
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


def test_validate_node_request_allows_snippet_without_export() -> None:
    out = validate_workflow_python_node_request(
        {
            "execution_mode": "snippet",
            "module_source": "result = {'output': payload}\n",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
        }
    )

    assert out["status"] == "ok"
    assert out["request"]["execution_mode"] == "snippet"


def test_validate_node_request_allows_project_without_module_source_or_export() -> None:
    out = validate_workflow_python_node_request(
        {
            "execution_mode": "project",
            "module_sha256": "sha",
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "project": {"entrypoint": "pkg.runner", "callable": "run"},
            "payload": {},
        }
    )

    assert out["status"] == "ok"
    assert out["request"]["execution_mode"] == "project"


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
