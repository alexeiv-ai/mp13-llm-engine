from __future__ import annotations

import pytest

from hosting.sandbox.host_capabilities import (
    HostCapabilityApproval,
    HostCapabilityBroker,
    HostCapabilityDescriptor,
    HostCapabilityMethod,
    HostCapabilityProviderRef,
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
