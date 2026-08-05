from __future__ import annotations

import pytest

from hosting.sandbox.host_capabilities import (
    HostCapabilityApproval,
    HostCapabilityBroker,
    HostCapabilityDescriptor,
    HostCapabilityMethod,
    HostCapabilityProviderRef,
    HostCapabilitySession,
)


def _session(*, session_id: str = "session-1", provider_id: str = "crm.provider") -> HostCapabilitySession:
    descriptor = HostCapabilityDescriptor(
        name="crm.lookup",
        namespace="crm",
        group_path=["CRM"],
        provider=HostCapabilityProviderRef(
            provider_id=provider_id,
            kind="client_session",
            owner="actor:a",
            visibility="workflow",
        ),
        approval=HostCapabilityApproval(mode="always"),
    )
    return HostCapabilitySession(
        session_id=session_id,
        provider_id=provider_id,
        owner="actor:a",
        provider_kind="client_session",
        visibility="workflow",
        scope={"workflow_id": "wf-1"},
        methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
    )


def test_provider_and_session_id_remain_distinct_through_discovery_dispatch_approval_and_audit() -> None:
    approvals: list[dict] = []
    audits: list[dict] = []
    calls: list[tuple[HostCapabilitySession, object]] = []

    def approve(request):
        approvals.append(dict(request))
        return {"status": "approved", "decision": "allow_once"}

    def invoke(session, call):
        calls.append((session, call))
        return {"provider_call_id": call.provider_call_id, "status": "ok", "result": {"found": True}}

    broker = HostCapabilityBroker(
        request_id="request-1",
        workflow_id="wf-1",
        provider_invoker=invoke,
        approval_requester=approve,
        audit_emitter=lambda row: audits.append(dict(row)),
    )
    session = _session()
    broker.register_session(session)

    assert broker.providers_for_discovery() == [
        {
            "provider_id": "crm.provider",
            "kind": "client_session",
            "owner": "actor:a",
            "visibility": "workflow",
            "method_count": 1,
        }
    ]
    assert broker.dispatch({"method": "crm.lookup", "arguments": {"id": "42"}}) == {"found": True}
    assert calls[0][0].session_id == "session-1"
    assert calls[0][0].provider_id == "crm.provider"
    assert approvals[0]["provider"]["provider_id"] == "crm.provider"
    assert audits[0]["provider"]["provider_id"] == "crm.provider"


def test_broker_rejects_missing_mismatched_and_duplicate_identity() -> None:
    broker = HostCapabilityBroker()
    missing = _session(provider_id="crm.provider")
    missing.provider_id = ""
    with pytest.raises(ValueError, match="provider_id_required"):
        broker.register_session(missing)

    mismatched = _session(provider_id="crm.provider")
    mismatched.provider_id = "other.provider"
    with pytest.raises(ValueError, match="provider_id_mismatch"):
        broker.register_session(mismatched)

    broker.register_session(_session())
    with pytest.raises(ValueError, match="session_already_exists"):
        broker.register_session(_session())
