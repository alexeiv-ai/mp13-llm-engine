from __future__ import annotations

import pytest

from hosting.callable_surface import (
    bind_host_capability_provider_callback,
    extract_safe_correlation_metadata,
    host_capability_approval_decision,
    host_capability_approval_request,
    host_capability_descriptors_to_callable_schemas,
    host_capability_provider_success,
    normalize_host_capability_provider_response,
    toolbox_to_host_capability_descriptors,
)
from hosting.sandbox.host_capabilities import HostCapabilityProviderError


def test_toolbox_to_host_capability_descriptors_preserves_tools_view_flags() -> None:
    descriptors = toolbox_to_host_capability_descriptors(
        {
            "toolbox_id": "tb-1",
            "tool_metadata": {
                "customer_lookup": {
                    "description": "Look up customer.",
                    "args_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}},
                    "result_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "permissions": ["crm.customer.read"],
                    "group_path": ["CRM", "Customer"],
                }
            },
        },
        tools_view={
            "allowed_tools": ["customer_lookup"],
            "advertised_tools": ["customer_lookup"],
            "gated_tools": ["customer_lookup"],
            "tool_constraints": {"customer_lookup": {"region": "us"}},
        },
        provider_id="tb-provider",
        owner="client-a",
        namespace="crm",
    )

    row = descriptors[0].to_dict()

    assert row["name"] == "crm.customer_lookup"
    assert row["provider"] == {
        "provider_id": "tb-provider",
        "kind": "toolbox_session",
        "owner": "client-a",
        "visibility": "workflow",
    }
    assert row["approval"]["mode"] == "always"
    assert row["permissions"] == ["crm.customer.read"]
    assert row["metadata"]["toolbox"]["allowed"] is True
    assert row["metadata"]["toolbox"]["advertised"] is True
    assert row["metadata"]["toolbox"]["gated"] is True
    assert row["metadata"]["toolbox"]["constraints"] == {"region": "us"}


def test_host_capability_descriptors_to_callable_schemas_filters_disabled_and_hidden() -> None:
    descriptors = toolbox_to_host_capability_descriptors(
        {"toolbox_id": "tb-1"},
        tools_view={
            "allowed_tools": ["visible", "hidden", "disabled"],
            "advertised_tools": ["visible"],
            "hidden_allowed_tools": ["hidden"],
            "disabled_tools": ["disabled"],
        },
        namespace="tools",
    )

    visible = host_capability_descriptors_to_callable_schemas(descriptors)
    all_rows = host_capability_descriptors_to_callable_schemas(descriptors, include_hidden=True, include_disabled=True)

    assert [row["name"] for row in visible] == ["tools.visible"]
    assert sorted(row["name"] for row in all_rows) == ["tools.disabled", "tools.hidden", "tools.visible"]
    assert visible[0]["contract"] == "hosting.sandbox.callable_schema.v1"
    assert "provider" in visible[0]
    assert "approval" in visible[0]


def test_provider_callback_helper_validates_and_normalizes_responses() -> None:
    callback = bind_host_capability_provider_callback(
        lambda method, arguments, context: {"method": method, "value": arguments["value"], "request_id": context["request_id"]}
    )

    response = callback(
        {
            "contract": "hosting.sandbox.host_capability_call.v1",
            "provider_call_id": "call-1",
            "method": "demo.echo",
            "arguments": {"value": 7},
            "context": {"request_id": "req-1"},
        }
    )

    assert response == host_capability_provider_success(
        "call-1",
        {"method": "demo.echo", "value": 7, "request_id": "req-1"},
    )
    assert normalize_host_capability_provider_response(response, provider_call_id="call-1")["value"] == 7

    bad = bind_host_capability_provider_callback(lambda _row: {"status": "ok", "provider_call_id": "wrong", "result": {}})
    error = bad({"provider_call_id": "call-2", "method": "demo.echo"})
    assert error["status"] == "error"
    assert error["reason"] == "host_capability_provider_error"
    with pytest.raises(HostCapabilityProviderError):
        normalize_host_capability_provider_response(error, provider_call_id="call-2")


def test_approval_bridge_sanitizes_arguments_and_normalizes_decisions() -> None:
    request = host_capability_approval_request(
        {
            "approval_id": "approval-1",
            "provider_call_id": "provider-call-1",
            "host_call_id": "host-call-1",
            "method": "crm.customer.delete",
            "arguments": {"customer_id": "c-1", "secret": "not copied"},
            "provider": {"provider_id": "client-crm"},
            "approval": {"mode": "always"},
            "context": {"workflow_id": "wf-1", "request_id": "req-1", "actor": "client-a"},
        }
    )

    assert request["contract"] == "hosting.sandbox.host_capability_approval.v1"
    assert request["argument_keys"] == ["customer_id", "secret"]
    assert "arguments" not in request
    assert request["correlation"]["workflow_id"] == "wf-1"
    assert host_capability_approval_decision("allow_once", approval_id="approval-1")["approved"] is True
    assert host_capability_approval_decision("add_to_scope", scope_constraints={"customer_id": "c-1"})["scope_constraints"] == {"customer_id": "c-1"}
    assert host_capability_approval_decision("unexpected")["decision"] == "deny"


def test_extract_safe_correlation_metadata_omits_unapproved_fields() -> None:
    assert extract_safe_correlation_metadata(
        {"workflow_id": "wf-1", "secret": "nope"},
        {"context": {"request_id": "req-1"}, "provider": {"provider_id": "provider-1"}},
    ) == {"workflow_id": "wf-1", "request_id": "req-1", "provider_id": "provider-1"}
