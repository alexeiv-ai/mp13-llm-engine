from __future__ import annotations

import asyncio

import pytest

from hosting.sandbox.host_capabilities import (
    HostCapabilityApproval,
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
    broker = HostCapabilityBroker(provider_invoker=invoke_provider, provider_timeout_seconds=0.001)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
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
    broker = HostCapabilityBroker(provider_invoker=invoke_provider)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
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
    broker = HostCapabilityBroker(provider_invoker=invoke_provider)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
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
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    with pytest.raises(HostCapabilityCanceled):
        broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}})

    assert cancel_checks["count"] > 1
