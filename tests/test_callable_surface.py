from __future__ import annotations

import pytest

from hosting.callable_surface import (
    HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
    HostCapabilityApprovalCallbackRelay,
    HostCapabilityProviderCallbackRelay,
    bind_host_capability_provider_callback,
    callable_surface_digests,
    callable_surface_identity,
    extract_safe_correlation_metadata,
    host_capability_approval_decision,
    host_capability_approval_request,
    host_capability_bridge_policy,
    host_capability_descriptors_to_callable_schemas,
    host_capability_provider_success,
    normalize_host_capability_provider_response,
    toolbox_brokered_io_call_surface,
    toolbox_to_callable_schemas,
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
    assert visible[0]["group_path"] == ["Tools"]
    assert visible[0]["identity"]["provider_kind"] == "toolbox_session"
    assert visible[0]["schema_digest"]
    assert visible[0]["method_digest"]


def test_callable_surface_duplicate_names_fail_by_default() -> None:
    first = toolbox_to_host_capability_descriptors({"toolbox_id": "tb-1", "allowed_tool_names": ["lookup"]}, provider_id="provider-1", namespace="crm")
    second = toolbox_to_host_capability_descriptors({"toolbox_id": "tb-2", "allowed_tool_names": ["lookup"]}, provider_id="provider-2", namespace="crm")

    with pytest.raises(ValueError, match="callable_surface_duplicate_name:crm.lookup"):
        host_capability_descriptors_to_callable_schemas([*first, *second])

    kept = host_capability_descriptors_to_callable_schemas([*first, *second], conflict_policy="keep_first")
    assert [row["identity"]["provider_id"] for row in kept] == ["provider-1"]


def test_callable_surface_identity_and_digests_are_stable() -> None:
    descriptor = toolbox_to_host_capability_descriptors(
        {
            "toolbox_id": "tb-1",
            "tool_metadata": {
                "lookup": {
                    "args_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
                    "result_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
            },
        },
        provider_id="provider-1",
        namespace="crm",
    )[0]

    identity = callable_surface_identity(descriptor, session_id="session-1")
    digests = callable_surface_digests(descriptor)

    assert identity == {
        "provider_kind": "toolbox_session",
        "provider_id": "provider-1",
        "toolbox_id": "tb-1",
        "session_id": "session-1",
        "method": "crm.lookup",
    }
    assert len(digests["schema_digest"]) == 64
    assert len(digests["method_digest"]) == 64
    assert len(digests["policy_digest"]) == 64


def test_toolbox_to_callable_schemas_exports_adapter_metadata_without_native_migration() -> None:
    schemas = toolbox_to_callable_schemas(
        {
            "toolbox_id": "support-tools",
            "tool_metadata": {
                "lookup": {
                    "description": "Look up a ticket.",
                    "group_path": ["Support", "Tickets"],
                    "args_schema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}},
                }
            },
        },
        tools_view={"advertised_tools": ["lookup"], "allowed_tools": ["lookup"]},
        provider_id="provider-support",
        namespace="support",
        session_id="session-support",
    )

    assert schemas[0]["name"] == "support.lookup"
    assert schemas[0]["group_path"] == ["Support", "Tickets"]
    assert schemas[0]["identity"] == {
        "provider_kind": "toolbox_session",
        "provider_id": "provider-support",
        "toolbox_id": "support-tools",
        "session_id": "session-support",
        "method": "support.lookup",
    }
    assert schemas[0]["schema_digest"]
    assert schemas[0]["method_digest"]
    assert schemas[0]["policy_digest"]


def test_toolbox_callable_schemas_allow_overlapping_tools_with_provider_namespaces() -> None:
    first = toolbox_to_callable_schemas(
        {"toolbox_id": "crm-a", "advertised_tool_names": ["lookup"]},
        provider_id="crm-a",
        namespace="crm_a",
        session_id="session-a",
    )
    second = toolbox_to_callable_schemas(
        {"toolbox_id": "crm-b", "advertised_tool_names": ["lookup"]},
        provider_id="crm-b",
        namespace="crm_b",
        session_id="session-b",
    )

    merged = [*first, *second]

    assert [row["name"] for row in merged] == ["crm_a.lookup", "crm_b.lookup"]
    assert [row["identity"]["session_id"] for row in merged] == ["session-a", "session-b"]


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


def test_provider_callback_relay_binds_local_callback_transport() -> None:
    relay = HostCapabilityProviderCallbackRelay()
    binding = relay.bind_callback(lambda method, arguments, context: {"method": method, "value": arguments["value"], "request_id": context["request_id"]})
    try:
        from hosting.callable_surface import HOST_CAPABILITY_PROVIDER_CALLBACK_NAME
        from hosting.toolbox_executor_ipc import _invoke_callback_binding

        response = _invoke_callback_binding(
            binding["callback_binding"],
            callback_name=HOST_CAPABILITY_PROVIDER_CALLBACK_NAME,
            payload={
                "contract": "hosting.sandbox.host_capability_call.v1",
                "provider_call_id": "call-relay-1",
                "method": "demo.echo",
                "arguments": {"value": 11},
                "context": {"request_id": "req-relay-1"},
            },
            context={"request_id": "req-relay-1"},
        )
    finally:
        relay.release(binding)

    assert response["result"] == host_capability_provider_success(
        "call-relay-1",
        {"method": "demo.echo", "value": 11, "request_id": "req-relay-1"},
    )


def test_approval_callback_relay_binds_local_callback_transport() -> None:
    approvals: list[dict] = []
    relay = HostCapabilityApprovalCallbackRelay()
    binding = relay.bind_callback(lambda request: approvals.append(dict(request)) or host_capability_approval_decision("allow_once", approval_id=request["approval_id"]))
    try:
        from hosting.toolbox_executor_ipc import _invoke_callback_binding

        response = _invoke_callback_binding(
            binding["callback_binding"],
            callback_name=HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
            payload={
                "contract": "hosting.sandbox.host_capability_approval.v1",
                "approval_id": "approval-relay-1",
                "provider_call_id": "provider-call-1",
                "method": "demo.approve",
                "arguments": {"value": 11, "secret": "not copied"},
                "context": {"request_id": "req-relay-approval"},
            },
            context={"request_id": "req-relay-approval"},
        )
    finally:
        relay.release(binding)

    assert approvals[0]["argument_keys"] == ["secret", "value"]
    assert approvals[0]["argument_preview"]["value"] == 11
    assert approvals[0]["argument_preview"]["secret"] == {"redacted": True, "reason": "secret_key"}
    assert "arguments" not in approvals[0]
    assert response["result"]["contract"] == "hosting.sandbox.host_capability_approval_decision.v1"
    assert response["result"]["decision"] == "allow_once"
    assert response["result"]["approval_id"] == "approval-relay-1"


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
            "identity": {"provider_kind": "client_session", "provider_id": "client-crm", "session_id": "session-1", "method": "crm.customer.delete"},
            "digests": {"schema_digest": "schema-1", "method_digest": "method-1", "policy_digest": "policy-1"},
            "context": {
                "workflow_id": "wf-1",
                "request_id": "req-1",
                "actor": "client-a",
                "session_id": "session-1",
                "toolbox_id": "tb-1",
                "branch_id": "branch-1",
                "session_tree_id": "tree-1",
            },
        }
    )

    assert request["contract"] == "hosting.sandbox.host_capability_approval.v1"
    assert request["argument_keys"] == ["customer_id", "secret"]
    assert request["argument_preview"]["customer_id"] == "c-1"
    assert request["argument_preview"]["secret"] == {"redacted": True, "reason": "secret_key"}
    assert "arguments" not in request
    assert request["correlation"]["workflow_id"] == "wf-1"
    assert request["context"]["branch_id"] == "branch-1"
    assert request["context"]["session_tree_id"] == "tree-1"
    assert request["identity"]["session_id"] == "session-1"
    assert request["digests"]["method_digest"] == "method-1"
    assert host_capability_approval_decision("allow_once", approval_id="approval-1")["approved"] is True
    assert host_capability_approval_decision("add_to_scope", scope_constraints={"customer_id": "c-1"})["scope_constraints"] == {"customer_id": "c-1"}
    assert host_capability_approval_decision("unexpected")["decision"] == "deny"


def test_extract_safe_correlation_metadata_omits_unapproved_fields() -> None:
    assert extract_safe_correlation_metadata(
        {"workflow_id": "wf-1", "secret": "nope"},
        {"context": {"request_id": "req-1", "toolbox_id": "tb-1"}, "provider": {"provider_id": "provider-1"}},
    ) == {"workflow_id": "wf-1", "request_id": "req-1", "toolbox_id": "tb-1", "provider_id": "provider-1"}


def test_host_capability_bridge_policy_intersects_explicit_permissions() -> None:
    policy = host_capability_bridge_policy(
        toolbox_policy={"brokered_io": {"filesystem": True, "http": True}},
        host_capability_policy={"namespaces": {"fs": True, "http": False}},
        bridge_policy={"namespaces": {"fs": True, "http": True, "state": True}},
    )

    assert policy["contract"] == "hosting.sandbox.host_capability_bridge_policy.v1"
    assert policy["mode"] == "explicit_intersection"
    assert policy["namespaces"]["fs"] is True
    assert policy["namespaces"]["http"] is False
    assert policy["namespaces"]["state"] is False


def test_toolbox_brokered_io_call_surface_reuses_known_method_descriptors() -> None:
    surface = toolbox_brokered_io_call_surface(
        "fs.read_text",
        arguments={"root_id": "rw", "relative_path": "a.txt"},
        context={"request_id": "req-1", "toolbox_id": "tb-1"},
        toolbox_policy={"sandbox": {"brokered_io": {"filesystem": True, "http": False}}},
        provider_id="provider-1",
        toolbox_id="tb-1",
        session_id="call-1",
    )

    assert surface["contract"] == "hosting.toolbox.brokered_io.call_surface.v1"
    assert surface["method"] == "fs.read_text"
    assert surface["namespace"] == "fs"
    assert surface["argument_keys"] == ["relative_path", "root_id"]
    assert surface["identity"] == {
        "provider_kind": "toolbox_session",
        "provider_id": "provider-1",
        "toolbox_id": "tb-1",
        "session_id": "call-1",
        "method": "fs.read_text",
    }
    assert surface["digests"]["schema_digest"]
    assert surface["digests"]["method_digest"]
    assert surface["digests"]["policy_digest"]
    assert surface["bridge_policy"]["namespaces"]["fs"] is True
    assert surface["bridge_policy"]["namespaces"]["http"] is False
    assert surface["correlation"]["request_id"] == "req-1"
