from __future__ import annotations

import asyncio

import pytest

from hosting.sandbox.host_capabilities import (
    HostCapabilityApproval,
    HostCapabilityApprovalDenied,
    HostCapabilityBroker,
    HostCapabilityCanceled,
    HostCapabilityProviderCall,
    HostCapabilityProviderError,
    HostCapabilityProviderUnavailable,
    HostCapabilityDescriptor,
    HostCapabilityMethod,
    HostCapabilityProviderRef,
    HostCapabilitySession,
    HostCapabilityTimeout,
    validate_provider_response,
)


def _descriptor(name: str = "crm.customer.lookup") -> HostCapabilityDescriptor:
    return HostCapabilityDescriptor(
        name=name,
        namespace=name.split(".", 1)[0],
        group_path=["CRM", "Customer"],
        description="Lookup customer.",
        args_schema={"type": "object"},
        result_schema={"type": "object"},
        permissions=["crm.customer.read"],
        scope_requirements=[{"scope": "crm.customer", "access": "read"}],
        approval=HostCapabilityApproval(mode="none"),
        provider=HostCapabilityProviderRef(provider_id="provider-1", kind="client_session", owner="client-a", visibility="workflow"),
    )


def test_host_capability_descriptor_validation_accepts_safe_descriptor() -> None:
    row = _descriptor().to_dict()

    assert row["contract"] == "hosting.sandbox.host_capability.v1"
    assert row["name"] == "crm.customer.lookup"
    assert row["namespace"] == "crm"
    assert row["group_path"] == ["CRM", "Customer"]
    assert row["provider"]["kind"] == "client_session"


@pytest.mark.parametrize(
    "payload, reason",
    [
        ({"name": ""}, "host_capability_name_required"),
        ({"name": "Bad.Name", "namespace": "Bad"}, "host_capability_invalid_name"),
        ({"name": "crm.customer.lookup", "namespace": "other"}, "host_capability_namespace_mismatch"),
        ({"group_path": []}, "host_capability_invalid_group_path"),
        ({"provider": HostCapabilityProviderRef(provider_id="p", kind="raw_socket")}, "host_capability_invalid_provider_kind"),
        ({"provider": HostCapabilityProviderRef(provider_id="p", visibility="global")}, "host_capability_invalid_visibility"),
    ],
)
def test_host_capability_descriptor_validation_rejects_invalid_descriptor(payload: dict, reason: str) -> None:
    kwargs = _descriptor().__dict__.copy()
    kwargs.update(payload)
    with pytest.raises(ValueError, match=reason):
        HostCapabilityDescriptor(**kwargs).to_dict()


def test_host_capability_broker_dispatches_builtin_and_hides_bindings_from_discovery() -> None:
    broker = HostCapabilityBroker(request_id="req-1", workflow_id="wf", package_id="pkg", runtime_kind="workflow_python_node")
    broker.register_builtin_provider(
        provider_id="builtin.demo",
        methods=[
            HostCapabilityMethod(
                descriptor=HostCapabilityDescriptor(
                    name="demo.echo",
                    namespace="demo",
                    group_path=["Demo"],
                    args_schema={"type": "object"},
                    result_schema={"type": "object"},
                    provider=HostCapabilityProviderRef(provider_id="builtin.demo", kind="builtin", owner="service", visibility="request"),
                ),
                handler=lambda args: {"echo": args.get("value")},
            )
        ],
    )

    result = broker.dispatch({"method": "demo.echo", "arguments": {"value": 7}})
    described = broker.dispatch({"method": "sandbox.describe", "arguments": {}})

    assert result == {"echo": 7}
    assert described["contract"] == "hosting.sandbox.discovery.v1"
    assert described["methods"] == ["demo.echo"]
    assert described["host_capabilities"]["methods"][0]["provider"] == {
        "provider_id": "builtin.demo",
        "kind": "builtin",
        "owner": "service",
        "visibility": "request",
    }
    assert "binding" not in described["host_capabilities"]["providers"][0]


def test_host_capability_provider_response_validation() -> None:
    assert validate_provider_response(
        {"status": "ok", "provider_call_id": "call-1", "result": {"value": 3}},
        provider_call_id="call-1",
    ) == {"value": 3}

    with pytest.raises(ValueError, match="host_capability_provider_call_id_mismatch"):
        validate_provider_response({"status": "ok", "provider_call_id": "other"}, provider_call_id="call-1")

    with pytest.raises(HostCapabilityProviderError, match="crm_missing"):
        validate_provider_response(
            {"status": "error", "provider_call_id": "call-1", "reason": "crm_missing", "message": "missing"},
            provider_call_id="call-1",
        )


def test_host_capability_broker_invokes_client_session_provider() -> None:
    calls: list[dict] = []

    def invoke_provider(session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        calls.append({"session": session.session_id, "call": call.to_dict()})
        return {
            "status": "ok",
            "provider_call_id": call.provider_call_id,
            "result": {"customer": call.arguments["customer_id"]},
        }

    descriptor = _descriptor()
    broker = HostCapabilityBroker(
        request_id="req-1",
        workflow_id="wf-1",
        package_id="pkg-1",
        runtime_kind="workflow_python_node",
        provider_invoker=invoke_provider,
    )
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
            binding={"transport": "daemon_callback", "address": "private"},
        )
    )

    result = broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}})

    assert result == {"customer": "c-1"}
    assert calls[0]["session"] == "client-crm"
    assert calls[0]["call"]["contract"] == "hosting.sandbox.host_capability_call.v1"
    assert calls[0]["call"]["method"] == "crm.customer.lookup"
    assert calls[0]["call"]["context"]["request_id"] == "req-1"
    assert calls[0]["call"]["context"]["actor"] == "client-a"
    assert "binding" not in calls[0]["call"]


def test_host_capability_broker_times_out_async_provider() -> None:
    async def invoke_provider(_session: HostCapabilitySession, _call: HostCapabilityProviderCall) -> dict:
        await asyncio.sleep(0.05)
        return {"status": "ok", "provider_call_id": _call.provider_call_id, "result": {"late": True}}

    descriptor = _descriptor()
    broker = HostCapabilityBroker(workflow_id="wf-1", provider_invoker=invoke_provider, provider_timeout_seconds=0.001)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    with pytest.raises(HostCapabilityTimeout) as exc:
        broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}})

    assert exc.value.reason == "host_call_timeout"
    assert exc.value.detail["timeout_seconds"] == 0.001


def test_host_capability_broker_maps_provider_disconnect() -> None:
    def invoke_provider(_session: HostCapabilitySession, _call: HostCapabilityProviderCall) -> dict:
        raise BrokenPipeError("provider pipe closed")

    descriptor = _descriptor()
    broker = HostCapabilityBroker(workflow_id="wf-1", provider_invoker=invoke_provider)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    with pytest.raises(HostCapabilityProviderUnavailable) as exc:
        broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}})

    assert exc.value.reason == "host_capability_provider_unavailable"
    assert exc.value.detail["provider_id"] == "client-crm"


def test_host_capability_broker_cancellation_blocks_provider_call() -> None:
    calls: list[dict] = []

    def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        calls.append(call.to_dict())
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {}}

    descriptor = _descriptor()
    broker = HostCapabilityBroker(workflow_id="wf-1", provider_invoker=invoke_provider)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )
    broker.cancel("unit_test_cancel")

    with pytest.raises(HostCapabilityCanceled) as exc:
        broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}})

    assert calls == []
    assert exc.value.reason == "host_call_canceled"
    assert exc.value.detail["reason"] == "unit_test_cancel"


def test_host_capability_broker_cancels_inflight_async_provider_call() -> None:
    cancel_checks = {"count": 0}

    async def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        await asyncio.sleep(1.0)
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {}}

    def cancel_checker() -> bool:
        cancel_checks["count"] += 1
        return cancel_checks["count"] > 1

    descriptor = _descriptor()
    broker = HostCapabilityBroker(
        workflow_id="wf-1",
        provider_invoker=invoke_provider,
        provider_timeout_seconds=5.0,
        cancel_checker=cancel_checker,
    )
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    with pytest.raises(HostCapabilityCanceled):
        broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}})

    assert cancel_checks["count"] > 1


def test_host_capability_broker_hides_unrelated_request_session() -> None:
    descriptor = _descriptor()
    broker = HostCapabilityBroker(request_id="req-visible", provider_invoker=lambda _session, call: {})
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="request",
            scope={"request_id": "req-other"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    described = broker.describe()

    assert described["methods"] == []
    with pytest.raises(RuntimeError, match="unsupported_host_method:crm.customer.lookup"):
        broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}})


def test_host_capability_broker_matches_request_workflow_instance_and_consumer_scopes() -> None:
    calls: list[str] = []

    def invoke_provider(session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        calls.append(session.session_id)
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"session_id": session.session_id}}

    broker = HostCapabilityBroker(
        request_id="req-1",
        workflow_id="wf-1",
        instance_id="inst-1",
        consumer_id="consumer-1",
        provider_invoker=invoke_provider,
    )
    rows = [
        ("request-session", "request", {"request_id": "req-1"}, "crm.request.lookup"),
        ("workflow-session", "workflow", {"workflow_id": "wf-1"}, "crm.workflow.lookup"),
        ("instance-session", "instance", {"instance_id": "inst-1"}, "crm.instance.lookup"),
        ("consumer-session", "consumer", {"consumer_id": "consumer-1"}, "crm.consumer.lookup"),
    ]
    for session_id, visibility, scope, method_name in rows:
        descriptor = _descriptor(method_name)
        broker.register_session(
            HostCapabilitySession(
                session_id=session_id,
                owner="client-a",
                provider_kind="client_session",
                visibility=visibility,
                scope=scope,
                methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
            )
        )

    assert broker.method_names() == ["crm.consumer.lookup", "crm.instance.lookup", "crm.request.lookup", "crm.workflow.lookup"]
    for _session_id, _visibility, _scope, method_name in rows:
        assert broker.dispatch({"method": method_name, "arguments": {"customer_id": "c-1"}})["session_id"]

    assert calls == ["request-session", "workflow-session", "instance-session", "consumer-session"]


def test_host_capability_broker_duplicate_resolution_prefers_builtin_and_narrower_client_scope() -> None:
    calls: list[str] = []

    def invoke_provider(session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        calls.append(session.session_id)
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"session_id": session.session_id}}

    builtin_descriptor = HostCapabilityDescriptor(
        name="crm.customer.lookup",
        namespace="crm",
        group_path=["CRM"],
        provider=HostCapabilityProviderRef(provider_id="builtin.demo", kind="builtin", owner="service", visibility="request"),
    )
    client_descriptor = _descriptor()
    broker = HostCapabilityBroker(request_id="req-1", workflow_id="wf-1", provider_invoker=invoke_provider)
    broker.register_builtin_provider(
        provider_id="builtin.demo",
        methods=[HostCapabilityMethod(descriptor=builtin_descriptor, handler=lambda _args: {"session_id": "builtin"})],
    )
    broker.register_session(
        HostCapabilitySession(
            session_id="workflow-client",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={client_descriptor.name: HostCapabilityMethod(descriptor=client_descriptor)},
        )
    )

    assert broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}}) == {"session_id": "builtin"}
    assert calls == []

    client_only = HostCapabilityBroker(request_id="req-1", workflow_id="wf-1", provider_invoker=invoke_provider)
    for session_id, visibility, scope in [
        ("workflow-client", "workflow", {"workflow_id": "wf-1"}),
        ("request-client", "request", {"request_id": "req-1"}),
    ]:
        client_only.register_session(
            HostCapabilitySession(
                session_id=session_id,
                owner="client-a",
                provider_kind="client_session",
                visibility=visibility,
                scope=scope,
                methods={client_descriptor.name: HostCapabilityMethod(descriptor=client_descriptor)},
            )
        )

    assert client_only.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}}) == {"session_id": "request-client"}


def test_host_capability_broker_enforces_namespace_and_permission_gates() -> None:
    broker = HostCapabilityBroker(
        workflow_id="wf-1",
        allowed_namespaces=["crm"],
        approved_permissions=["crm.customer.read"],
        provider_invoker=lambda _session, call: {"status": "ok", "provider_call_id": call.provider_call_id, "result": {}},
    )
    read_descriptor = _descriptor("crm.customer.lookup")
    write_descriptor = HostCapabilityDescriptor(
        name="crm.customer.write",
        namespace="crm",
        group_path=["CRM"],
        permissions=["crm.customer.write"],
        provider=HostCapabilityProviderRef(provider_id="client-crm", kind="client_session", owner="client-a", visibility="workflow"),
    )
    erp_descriptor = HostCapabilityDescriptor(
        name="erp.customer.lookup",
        namespace="erp",
        group_path=["ERP"],
        permissions=["erp.customer.read"],
        provider=HostCapabilityProviderRef(provider_id="client-erp", kind="client_session", owner="client-a", visibility="workflow"),
    )
    for session_id, descriptor in [("client-crm-read", read_descriptor), ("client-crm-write", write_descriptor), ("client-erp", erp_descriptor)]:
        broker.register_session(
            HostCapabilitySession(
                session_id=session_id,
                owner="client-a",
                provider_kind="client_session",
                visibility="workflow",
                scope={"workflow_id": "wf-1"},
                methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
            )
        )

    assert broker.method_names() == ["crm.customer.lookup"]
    assert broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}}) == {}
    with pytest.raises(RuntimeError, match="unsupported_host_method:crm.customer.write"):
        broker.dispatch({"method": "crm.customer.write", "arguments": {}})
    with pytest.raises(RuntimeError, match="unsupported_host_method:erp.customer.lookup"):
        broker.dispatch({"method": "erp.customer.lookup", "arguments": {}})


def test_host_capability_broker_requests_approval_before_gated_provider_call() -> None:
    approval_requests: list[dict] = []
    provider_calls: list[dict] = []
    descriptor = HostCapabilityDescriptor(
        name="crm.customer.delete",
        namespace="crm",
        group_path=["CRM"],
        permissions=["crm.customer.write"],
        approval=HostCapabilityApproval(mode="always", cache_key="method+scope+actor", ttl_seconds=0),
        provider=HostCapabilityProviderRef(provider_id="client-crm", kind="client_session", owner="client-a", visibility="workflow"),
    )

    def approve(request: dict) -> dict:
        approval_requests.append(dict(request))
        return {"status": "approved", "approved": True}

    def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        provider_calls.append(call.to_dict())
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"deleted": True}}

    broker = HostCapabilityBroker(workflow_id="wf-1", provider_invoker=invoke_provider, approval_requester=approve)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    assert broker.dispatch({"method": "crm.customer.delete", "arguments": {"customer_id": "c-1"}}) == {"deleted": True}
    assert len(approval_requests) == 1
    assert len(provider_calls) == 1
    assert approval_requests[0]["contract"] == "hosting.sandbox.host_capability_approval.v1"
    assert approval_requests[0]["provider_call_id"] == provider_calls[0]["provider_call_id"]
    assert approval_requests[0]["approval"]["mode"] == "always"
    assert "binding" not in approval_requests[0]["provider"]


def test_host_capability_broker_denies_gated_provider_call_before_execution() -> None:
    provider_calls: list[dict] = []
    descriptor = HostCapabilityDescriptor(
        name="crm.customer.delete",
        namespace="crm",
        group_path=["CRM"],
        approval=HostCapabilityApproval(mode="always"),
        provider=HostCapabilityProviderRef(provider_id="client-crm", kind="client_session", owner="client-a", visibility="workflow"),
    )

    def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        provider_calls.append(call.to_dict())
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"deleted": True}}

    broker = HostCapabilityBroker(
        workflow_id="wf-1",
        provider_invoker=invoke_provider,
        approval_requester=lambda _request: {"status": "denied", "approved": False, "message": "user denied"},
    )
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    with pytest.raises(HostCapabilityApprovalDenied) as exc:
        broker.dispatch({"method": "crm.customer.delete", "arguments": {"customer_id": "c-1"}})

    assert provider_calls == []
    assert exc.value.reason == "host_call_approval_denied"
    assert exc.value.message == "user denied"


def test_host_capability_broker_requires_approval_requester_for_gated_call() -> None:
    provider_calls: list[dict] = []
    descriptor = HostCapabilityDescriptor(
        name="crm.customer.delete",
        namespace="crm",
        group_path=["CRM"],
        approval=HostCapabilityApproval(mode="always"),
        provider=HostCapabilityProviderRef(provider_id="client-crm", kind="client_session", owner="client-a", visibility="workflow"),
    )
    broker = HostCapabilityBroker(
        workflow_id="wf-1",
        provider_invoker=lambda _session, call: provider_calls.append(call.to_dict()) or {"status": "ok", "provider_call_id": call.provider_call_id},
    )
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    with pytest.raises(HostCapabilityApprovalDenied) as exc:
        broker.dispatch({"method": "crm.customer.delete", "arguments": {"customer_id": "c-1"}})

    assert provider_calls == []
    assert exc.value.detail["reason"] == "approval_requester_unavailable"
