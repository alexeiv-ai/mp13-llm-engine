from __future__ import annotations

import threading

import pytest

from hosting.callable_surface import ApprovalCallbackLease
from hosting.engine_host_channel import EngineHostControlChannel
from hosting.sandbox.host_capabilities import (
    HostCapabilityApproval,
    HostCapabilityBroker,
    HostCapabilityDescriptor,
    HostCapabilityMethod,
    HostCapabilityProviderRef,
    HostCapabilitySession,
)


class _Relay:
    def __init__(self) -> None:
        self.bound = 0
        self.released = 0

    def bind_callback(self, _callback, **_kwargs):
        self.bound += 1
        return {"transport": "local_ipc", "callback_binding": {"session_token": f"token-{self.bound}"}}

    def release(self, _binding):
        self.released += 1


class _Connection:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    def invoke(self, command, payload):
        self.calls.append((command, dict(payload or {})))
        if self.fail:
            raise RuntimeError("invoke_failed")
        if command.endswith("stream-open"):
            return {"status": "open", "stream_id": "stream-1"}
        return {"status": "ok"}

    def close(self):
        return None


def _channel(connection: _Connection, relay: _Relay) -> EngineHostControlChannel:
    channel = EngineHostControlChannel({"engine_host_daemon_auto_bootstrap": False})
    channel._approval_callback_relay = relay  # type: ignore[assignment]
    channel._get_connection = lambda: connection  # type: ignore[method-assign]
    return channel


def test_direct_callback_is_bound_before_sync_call_and_released_once_on_success_or_error() -> None:
    relay = _Relay()
    connection = _Connection()
    channel = _channel(connection, relay)
    channel.execute_workflow_python(request={"request_id": "request-1"}, approval_requester=lambda _row: {})
    assert relay.bound == 1 and relay.released == 1
    assert connection.calls[0][1]["approval_requester_binding"]["callback_binding"]["session_token"] == "token-1"

    failing = _channel(_Connection(fail=True), relay)
    with pytest.raises(RuntimeError, match="invoke_failed"):
        failing.execute_workflow_js(request={"request_id": "request-2"}, approval_requester=lambda _row: {})
    assert relay.bound == 2 and relay.released == 2


def test_callback_inputs_conflict_and_lease_scope_is_enforced() -> None:
    relay = _Relay()
    channel = _channel(_Connection(), relay)
    lease = ApprovalCallbackLease.bind(relay, lambda _row: {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inputs_conflict"):
        channel.execute_workflow_python(
            request={"request_id": "request-1"},
            approval_requester=lambda _row: {},
            approval_callback_lease=lease,
        )
    channel.execute_workflow_python(request={"request_id": "request-1"}, approval_callback_lease=lease)
    with pytest.raises(RuntimeError, match="scope_mismatch"):
        channel.execute_workflow_python(request={"request_id": "request-2"}, approval_callback_lease=lease)
    lease.close()
    lease.close()
    assert relay.released == 1


def test_stream_callback_lives_until_close_cancel_terminal_or_channel_shutdown() -> None:
    relay = _Relay()
    channel = _channel(_Connection(), relay)
    opened = channel.workflow_python_stream_open(
        request={"request_id": "request-stream"}, approval_requester=lambda _row: {}
    )
    assert opened["stream_id"] == "stream-1" and relay.released == 0
    channel.workflow_python_stream_close(stream_id="stream-1")
    channel.workflow_python_stream_close(stream_id="stream-1")
    assert relay.released == 1

    channel.workflow_js_stream_open(request={"request_id": "request-js"}, approval_requester=lambda _row: {})
    channel.workflow_js_stream_send(stream_id="stream-1", message={"action": "cancel"})
    assert relay.released == 2

    channel.workflow_js_stream_open(request={"request_id": "request-js-2"}, approval_requester=lambda _row: {})
    channel.close_connection()
    assert relay.released == 3


def test_stream_open_failure_releases_owned_callback() -> None:
    relay = _Relay()
    channel = _channel(_Connection(fail=True), relay)
    with pytest.raises(RuntimeError, match="invoke_failed"):
        channel.workflow_python_stream_open(
            request={"request_id": "request-stream"}, approval_requester=lambda _row: {}
        )
    assert relay.bound == 1 and relay.released == 1


def test_transport_retry_reuses_provider_call_and_approval_ids() -> None:
    approvals = []
    provider_calls = []
    descriptor = HostCapabilityDescriptor(
        name="crm.lookup",
        namespace="crm",
        group_path=["CRM"],
        provider=HostCapabilityProviderRef(
            provider_id="crm.provider", kind="client_session", owner="actor:a", visibility="workflow"
        ),
        approval=HostCapabilityApproval(mode="always"),
    )
    session = HostCapabilitySession(
        session_id="session-1",
        provider_id="crm.provider",
        owner="actor:a",
        provider_kind="client_session",
        visibility="workflow",
        scope={"workflow_id": "wf-1"},
        methods={descriptor.name: HostCapabilityMethod(descriptor=descriptor)},
    )

    def approve(row):
        approvals.append(dict(row))
        return {"status": "approved", "decision": "allow_once"}

    def invoke(_session, call):
        provider_calls.append(call.provider_call_id)
        return {"provider_call_id": call.provider_call_id, "status": "ok", "result": {"ok": True}}

    broker = HostCapabilityBroker(workflow_id="wf-1", approval_requester=approve, provider_invoker=invoke)
    broker.register_session(session)
    request = {"method": "crm.lookup", "host_call_id": "host-call-1", "arguments": {"id": "42"}}
    broker.dispatch(request)
    broker.dispatch(request)
    assert provider_calls[0] == provider_calls[1]
    assert approvals[0]["provider_call_id"] == approvals[1]["provider_call_id"]
    assert approvals[0]["approval_id"] == approvals[1]["approval_id"]

