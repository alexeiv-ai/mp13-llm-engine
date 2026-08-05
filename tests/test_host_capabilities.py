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
    HostCapabilitySession as _HostCapabilitySession,
    HostCapabilityTimeout,
    validate_provider_response,
)


def HostCapabilitySession(**kwargs):
    """Build a session with the descriptor's explicit logical provider identity."""
    methods = dict(kwargs.get("methods") or {})
    provider_ids = {
        method.descriptor.provider.provider_id
        for method in methods.values()
        if method.descriptor.provider.provider_id
    }
    if len(provider_ids) != 1:
        raise ValueError("test_session_requires_one_provider_id")
    provider_id = next(iter(provider_ids))
    if kwargs.get("session_id") == provider_id:
        kwargs["session_id"] = f"{provider_id}.registration"
    return _HostCapabilitySession(provider_id=provider_id, **kwargs)


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
    assert described["methods"] == ["demo.echo", "host.describe", "sandbox.describe"]
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
    events: list[tuple[str, dict]] = []

    def invoke_provider(_session: HostCapabilitySession, _call: HostCapabilityProviderCall) -> dict:
        raise BrokenPipeError("provider pipe closed")

    descriptor = _descriptor()
    broker = HostCapabilityBroker(workflow_id="wf-1", provider_invoker=invoke_provider, event_emitter=lambda kind, payload: events.append((kind, payload)))
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
    assert exc.value.detail["provider_id"] == "provider-1"
    assert [kind for kind, _payload in events] == ["host_call", "provider_failure", "host_response"]
    assert events[1][1]["reason"] == "host_capability_provider_unavailable"


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
    events: list[tuple[str, dict]] = []

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
        event_emitter=lambda kind, payload: events.append((kind, payload)),
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
    assert "canceled" in [kind for kind, _payload in events]
    assert events[-1][0] == "host_response"
    assert events[-1][1]["reason"] == "host_call_canceled"


def test_host_capability_provider_serial_policy_admits_one_call_at_a_time() -> None:
    active = 0
    max_active = 0

    async def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.03)
        active -= 1
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"key": call.arguments["key"]}}

    descriptor = HostCapabilityDescriptor(
        **{
            **_descriptor("crm.customer.mutate").__dict__,
            "metadata": {
                "concurrency": {
                    "mode": "serial",
                    "group": "crm-customer-mutations",
                    "queue_depth": 2,
                    "queue_timeout_seconds": 1,
                }
            },
        }
    )
    broker = HostCapabilityBroker(workflow_id="wf-serial", provider_invoker=invoke_provider)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm-serial",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-serial"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    async def run_calls() -> list[dict]:
        return await asyncio.gather(
            broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "a"}}),
            broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "b"}}),
        )

    results = asyncio.run(run_calls())

    assert [row["key"] for row in results] == ["a", "b"]
    assert max_active == 1
    described = broker.describe()
    policy = described["host_capabilities"]["methods"][0]["metadata"]["concurrency"]
    assert policy["mode"] == "serial"
    assert policy["max_concurrency"] == 1
    assert policy["runtime"]["active_calls"] == 0


def test_host_capability_parallel_policy_honors_max_concurrency_two() -> None:
    active = 0
    peak = 0

    async def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.03)
        active -= 1
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"key": call.arguments["key"]}}

    descriptor = HostCapabilityDescriptor(
        **{
            **_descriptor("crm.customer.lookup_parallel").__dict__,
            "metadata": {
                "concurrency": {
                    "mode": "parallel",
                    "group": "crm-customer-lookups",
                    "max_concurrency": 2,
                    "queue_depth": 2,
                    "queue_timeout_seconds": 1,
                }
            },
        }
    )
    broker = HostCapabilityBroker(workflow_id="wf-parallel", provider_invoker=invoke_provider)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm-parallel",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-parallel"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    async def run_calls() -> list[dict]:
        return await asyncio.gather(
            broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "a"}}),
            broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "b"}}),
            broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "c"}}),
        )

    results = asyncio.run(run_calls())

    assert [row["key"] for row in results] == ["a", "b", "c"]
    assert peak == 2
    policy = broker.describe()["host_capabilities"]["methods"][0]["metadata"]["concurrency"]
    assert policy["max_concurrency"] == 2


def test_host_capability_keyed_policy_overlaps_different_resources_and_blocks_same_resource() -> None:
    active_by_key: dict[str, int] = {}
    max_active_by_key: dict[str, int] = {}

    async def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        key = str(call.arguments["key"])
        active_by_key[key] = active_by_key.get(key, 0) + 1
        max_active_by_key[key] = max(max_active_by_key.get(key, 0), active_by_key[key])
        await asyncio.sleep(0.03)
        active_by_key[key] -= 1
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"key": key}}

    descriptor = HostCapabilityDescriptor(
        **{
            **_descriptor("crm.customer.update").__dict__,
            "metadata": {
                "concurrency": {
                    "mode": "keyed",
                    "group": "crm-customer-records",
                    "key_argument": "key",
                    "queue_depth": 2,
                    "queue_timeout_seconds": 1,
                }
            },
        }
    )
    broker = HostCapabilityBroker(workflow_id="wf-keyed", provider_invoker=invoke_provider)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm-keyed",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-keyed"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    async def run_calls() -> list[dict]:
        return await asyncio.gather(
            broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "a"}}),
            broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "a"}}),
            broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "b"}}),
        )

    results = asyncio.run(run_calls())

    assert [row["key"] for row in results] == ["a", "a", "b"]
    assert max_active_by_key == {"a": 1, "b": 1}


def test_host_capability_queue_full_is_reported_without_blocking_the_admitted_call() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        started.set()
        await release.wait()
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {}}

    descriptor = HostCapabilityDescriptor(
        **{
            **_descriptor("crm.customer.delete").__dict__,
            "metadata": {
                "concurrency": {
                    "mode": "serial",
                    "group": "crm-customer-delete-queue-full",
                    "queue_policy": "bounded",
                    "queue_depth": 0,
                    "queue_timeout_seconds": 1,
                }
            },
        }
    )
    broker = HostCapabilityBroker(workflow_id="wf-queue-full", provider_invoker=invoke_provider)
    broker.register_session(
        HostCapabilitySession(
            session_id="client-crm-queue-full",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-queue-full"},
            methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
        )
    )

    async def run_calls() -> tuple[object, object]:
        first = asyncio.create_task(broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "a"}}))
        await started.wait()
        second = asyncio.create_task(broker.dispatch_async({"method": descriptor.name, "arguments": {"key": "b"}}))
        await asyncio.sleep(0.02)
        release.set()
        return await asyncio.gather(first, second, return_exceptions=True)

    first, second = asyncio.run(run_calls())

    assert isinstance(first, dict)
    assert isinstance(second, HostCapabilityProviderError)
    assert second.reason == "host_call_queue_full"


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

    assert described["methods"] == ["host.describe", "sandbox.describe"]
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

    assert broker.method_names() == ["crm.consumer.lookup", "crm.instance.lookup", "crm.request.lookup", "crm.workflow.lookup", "host.describe", "sandbox.describe"]
    for _session_id, _visibility, _scope, method_name in rows:
        assert broker.dispatch({"method": method_name, "arguments": {"customer_id": "c-1"}})["session_id"]

    assert calls == ["request-session", "workflow-session", "instance-session", "consumer-session"]


def test_host_capability_broker_duplicate_resolution_requires_explicit_override() -> None:
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

    with pytest.raises(ValueError, match="host_capability_duplicate_method:crm.customer.lookup"):
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

    broker.register_session(
        HostCapabilitySession(
            session_id="workflow-client-override",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={client_descriptor.name: HostCapabilityMethod(descriptor=client_descriptor)},
            allow_override=True,
        )
    )

    assert broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}}) == {"session_id": "workflow-client-override"}
    assert calls == ["workflow-client-override"]

    client_only = HostCapabilityBroker(request_id="req-1", workflow_id="wf-1", provider_invoker=invoke_provider)
    client_only.register_session(
        HostCapabilitySession(
            session_id="workflow-client",
            owner="client-a",
            provider_kind="client_session",
            visibility="workflow",
            scope={"workflow_id": "wf-1"},
            methods={client_descriptor.name: HostCapabilityMethod(descriptor=client_descriptor)},
        )
    )
    with pytest.raises(ValueError, match="host_capability_duplicate_method:crm.customer.lookup"):
        client_only.register_session(
            HostCapabilitySession(
                session_id="request-client",
                owner="client-a",
                provider_kind="client_session",
                visibility="request",
                scope={"request_id": "req-1"},
                methods={client_descriptor.name: HostCapabilityMethod(descriptor=client_descriptor)},
            )
        )
    client_only.register_session(
        HostCapabilitySession(
            session_id="request-client",
            owner="client-a",
            provider_kind="client_session",
            visibility="request",
            scope={"request_id": "req-1"},
            methods={client_descriptor.name: HostCapabilityMethod(descriptor=client_descriptor)},
            allow_override=True,
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

    assert broker.method_names() == ["crm.customer.lookup", "host.describe", "sandbox.describe"]
    assert broker.dispatch({"method": "crm.customer.lookup", "arguments": {"customer_id": "c-1"}}) == {}
    with pytest.raises(RuntimeError, match="unsupported_host_method:crm.customer.write"):
        broker.dispatch({"method": "crm.customer.write", "arguments": {}})
    with pytest.raises(RuntimeError, match="unsupported_host_method:erp.customer.lookup"):
        broker.dispatch({"method": "erp.customer.lookup", "arguments": {}})


def test_host_capability_broker_requests_approval_before_gated_provider_call() -> None:
    approval_requests: list[dict] = []
    provider_calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    audit_records: list[dict] = []
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

    broker = HostCapabilityBroker(
        workflow_id="wf-1",
        provider_invoker=invoke_provider,
        approval_requester=approve,
        event_emitter=lambda kind, payload: events.append((kind, payload)),
        audit_emitter=lambda payload: audit_records.append(dict(payload)),
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

    assert broker.dispatch({"method": "crm.customer.delete", "arguments": {"customer_id": "c-1", "secret": "hidden"}}) == {"deleted": True}
    assert len(approval_requests) == 1
    assert len(provider_calls) == 1
    assert approval_requests[0]["contract"] == "hosting.sandbox.host_capability_approval.v1"
    assert approval_requests[0]["provider_call_id"] == provider_calls[0]["provider_call_id"]
    assert approval_requests[0]["approval"]["mode"] == "always"
    assert approval_requests[0]["argument_preview"]["customer_id"] == "c-1"
    assert approval_requests[0]["argument_preview"]["secret"] == {"redacted": True, "reason": "secret_key"}
    assert "arguments" not in approval_requests[0]
    assert "binding" not in approval_requests[0]["provider"]
    assert [kind for kind, _payload in events] == ["host_call", "approval", "approval", "host_response"]
    assert events[1][1]["status"] == "requested"
    assert events[2][1]["status"] == "approved"
    assert len(audit_records) == 1
    assert audit_records[0]["result"] == "approved"
    assert audit_records[0]["approval_id"] == events[2][1]["approval_id"]
    assert audit_records[0]["provider_call_id"] == provider_calls[0]["provider_call_id"]
    assert audit_records[0]["argument_preview"]["customer_id"] == "c-1"


def test_host_capability_broker_does_not_fail_call_when_audit_emitter_fails() -> None:
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

    def fail_audit(_payload: dict) -> None:
        raise PermissionError("access_control.json")

    broker = HostCapabilityBroker(
        workflow_id="wf-1",
        provider_invoker=invoke_provider,
        approval_requester=lambda _request: {"decision": "allow_once", "approved": True},
        audit_emitter=fail_audit,
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

    assert broker.dispatch({"method": "crm.customer.delete", "arguments": {"customer_id": "c-1"}}) == {"deleted": True}
    assert len(provider_calls) == 1


def test_host_capability_broker_denies_gated_provider_call_before_execution() -> None:
    provider_calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    audit_records: list[dict] = []
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
        event_emitter=lambda kind, payload: events.append((kind, payload)),
        audit_emitter=lambda payload: audit_records.append(dict(payload)),
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
    assert [kind for kind, _payload in events] == ["host_call", "approval", "approval", "host_response"]
    assert events[2][1]["status"] == "denied"
    assert events[-1][1]["reason"] == "host_call_approval_denied"
    assert len(audit_records) == 1
    assert audit_records[0]["result"] == "denied"
    assert audit_records[0]["reason"] == "user denied"
    assert audit_records[0]["provider_call_id"] == events[2][1]["provider_call_id"]


def test_host_capability_broker_requires_approval_requester_for_gated_call() -> None:
    provider_calls: list[dict] = []
    audit_records: list[dict] = []
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
        audit_emitter=lambda payload: audit_records.append(dict(payload)),
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
    assert len(audit_records) == 1
    assert audit_records[0]["result"] == "denied"
    assert audit_records[0]["reason"] == "approval_requester_unavailable"


def test_host_capability_broker_approval_allow_once_does_not_create_scope_grant() -> None:
    approval_requests: list[dict] = []
    provider_calls: list[dict] = []
    descriptor = HostCapabilityDescriptor(
        name="crm.customer.read_sensitive",
        namespace="crm",
        group_path=["CRM"],
        scope_requirements=[{"scope": "crm.customer", "access": "read_sensitive"}],
        approval=HostCapabilityApproval(mode="always"),
        provider=HostCapabilityProviderRef(provider_id="client-crm", kind="client_session", owner="client-a", visibility="workflow"),
    )

    def approve(request: dict) -> dict:
        approval_requests.append(dict(request))
        return {"decision": "allow_once", "approved": True}

    def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        provider_calls.append(call.to_dict())
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"ok": True}}

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

    assert broker.dispatch({"method": descriptor.name, "arguments": {"customer_id": "c-1"}}) == {"ok": True}
    assert broker.dispatch({"method": descriptor.name, "arguments": {"customer_id": "c-1"}}) == {"ok": True}

    assert len(approval_requests) == 2
    assert len(provider_calls) == 2


def test_host_capability_broker_approval_add_to_scope_reuses_matching_grant() -> None:
    approval_requests: list[dict] = []
    provider_calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    audit_records: list[dict] = []
    descriptor = HostCapabilityDescriptor(
        name="crm.customer.read_sensitive",
        namespace="crm",
        group_path=["CRM"],
        scope_requirements=[{"scope": "crm.customer", "access": "read_sensitive"}],
        approval=HostCapabilityApproval(mode="always", ttl_seconds=60),
        provider=HostCapabilityProviderRef(provider_id="client-crm", kind="client_session", owner="client-a", visibility="workflow"),
    )

    def approve(request: dict) -> dict:
        approval_requests.append(dict(request))
        return {
            "decision": "add_to_scope",
            "approved": True,
            "scope_constraints": {"customer_id": dict(request.get("argument_preview") or {}).get("customer_id")},
        }

    def invoke_provider(_session: HostCapabilitySession, call: HostCapabilityProviderCall) -> dict:
        provider_calls.append(call.to_dict())
        return {"status": "ok", "provider_call_id": call.provider_call_id, "result": {"customer_id": call.arguments["customer_id"]}}

    broker = HostCapabilityBroker(
        workflow_id="wf-1",
        provider_invoker=invoke_provider,
        approval_requester=approve,
        event_emitter=lambda kind, payload: events.append((kind, payload)),
        audit_emitter=lambda payload: audit_records.append(dict(payload)),
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

    assert broker.dispatch({"method": descriptor.name, "arguments": {"customer_id": "c-1"}}) == {"customer_id": "c-1"}
    assert broker.dispatch({"method": descriptor.name, "arguments": {"customer_id": "c-1"}}) == {"customer_id": "c-1"}
    assert broker.dispatch({"method": descriptor.name, "arguments": {"customer_id": "c-2"}}) == {"customer_id": "c-2"}

    assert len(approval_requests) == 2
    assert [call["arguments"]["customer_id"] for call in provider_calls] == ["c-1", "c-1", "c-2"]
    assert any(kind == "approval" and payload["status"] == "reused" for kind, payload in events)
    assert [row["result"] for row in audit_records] == ["approved", "reused", "approved"]
    assert audit_records[0]["decision"]["grant"]["constraints"] == {"customer_id": "c-1"}
