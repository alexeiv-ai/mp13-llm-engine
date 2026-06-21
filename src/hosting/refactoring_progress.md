# Hosting Refactoring Progress

Date: 2026-06-21

Scope: implementation progress for the legacy cleanup items following the completed Sandbox Event Stream Protocol pillar.

## Completed In Current Slice

### Host Capability Slice 1: Descriptor And Built-In Broker

- [x] Added shared host capability descriptor, provider reference, approval, session, and broker primitives in `sandbox/host_capabilities.py`.
- [x] Added descriptor validation for method names, namespaces, group paths, schema size, provider kind, and visibility.
- [x] Adapted `HostApiRegistry` methods into shared descriptors with group metadata and sandbox-safe provider refs.
- [x] Added `sandbox.describe` as a host-callable discovery method.
- [x] Added Python `sandbox.describe()` and JavaScript `sandbox.describe()` harness APIs.
- [x] Wrapped request-local built-ins behind `HostCapabilityBroker`.
- [x] Routed Python and JavaScript workflow node host calls through the broker while preserving existing built-in behavior.

### Host Capability Slice 2: Client Provider Session Lifecycle

- [x] Added private binding storage and sanitized public output to `HostCapabilitySession`.
- [x] Added daemon runtime registry for client-owned provider sessions.
- [x] Added authenticated daemon commands for `host-capability-session-register`, `host-capability-session-list`, and `host-capability-session-close`.
- [x] Added typed `EngineHostControlChannel` helpers for provider session lifecycle management.
- [x] Added disconnect cleanup for sessions marked `close_on_client_disconnect`.

### Host Capability Slice 3: Provider Callback Envelope

- [x] Added canonical `hosting.sandbox.host_capability_call.v1` provider call envelope.
- [x] Added provider response validation with `provider_call_id` matching and normalized provider errors.
- [x] Added broker seam for invoking client/toolbox provider sessions through a provider invoker.
- [x] Kept provider bindings out of the callback envelope and sandbox-facing discovery.

### Host Capability Slice 4: Provider Timeout And Cancellation

- [x] Added structured provider errors for timeout, provider unavailable, and cancellation.
- [x] Added broker-level provider timeout deadlines and async provider call cancellation.
- [x] Added broker cancellation control via explicit `cancel(...)` and a `cancel_checker` hook.
- [x] Mapped broken provider callback transports to `host_capability_provider_unavailable`.
- [x] Preserved structured host-call error reasons through Python and JavaScript node runtime host responses.

### Host Capability Slice 5: Scope And Permission Resolution

- [x] Added broker visibility checks for request, workflow, instance, and consumer scoped provider sessions.
- [x] Added namespace and permission gates to broker method resolution.
- [x] Made sandbox discovery omit invisible or unauthorized provider methods.
- [x] Added deterministic duplicate resolution: built-ins win, then narrower client scopes win, then session ID tie-breaks.
- [x] Confirmed built-in workflow host API discovery and dispatch still use the broker correctly.

### Host Capability Slice 6: Gated Approval Flow

- [x] Added sandbox-safe approval request contract `hosting.sandbox.host_capability_approval.v1`.
- [x] Added broker `approval_requester` hook for outward/user-facing approval decisions.
- [x] Prevented gated provider execution until approval is granted.
- [x] Mapped approval denial and missing approval requester to structured `host_call_approval_denied` errors.
- [x] Kept durable approval audit pending for the dedicated audit slice.

### Host Capability Slice 7: Event Observations

- [x] Added broker event observations for `host_call`, `host_response`, `approval`, `provider_failure`, and `canceled`.
- [x] Added worker `host_call_id` propagation into broker observations as `call_id` so clients can correlate worker calls with broker responses.
- [x] Wired broker observations into direct and streaming Python/JavaScript node execution paths while suppressing duplicate broker `host_call` stream frames.
- [x] Verified approval request/approval/denial events include approval and provider call correlation fields.
- [x] Verified workflow JS stream subscribers receive `host_response` observations for built-in host calls.

### Host Capability Slice 8: Durable Approval Audit

- [x] Added broker `audit_emitter` hook for security-relevant host capability audit records.
- [x] Recorded approved, denied, and approval-requester-unavailable outcomes for gated host capability calls.
- [x] Persisted host capability audit rows in service control state under `audit/host_capability_audit.json`.
- [x] Kept raw provider bindings and provider session tokens out of durable audit rows.
- [x] Verified broker approval audit records and service persistence.

### Host Capability Slice 9: Public Contract Finalization

- [x] Added daemon regression coverage for SSH-bound provider session registration.
- [x] Verified missing or mismatched SSH binding denies `host-capability-session-register`.
- [x] Verified matching SSH binding allows registration while public responses still omit provider bindings.
- [x] Marked Host API implementation checklist and public breaking-change notes complete.

### Host Capability Decision Update: Client-Owned Known Methods

- [x] Recorded decision to optimize local IPC for `host.call` provider callbacks and treat SSH relay as a corner case.
- [x] Recorded decision to remove daemon-owned built-in special status from the target Host API model.
- [x] Recorded decision that hosting client library helpers should register known broker-supported methods by default, while allowing clients to omit or replace them.
- [x] Recorded decision that duplicate fully-qualified method registration fails by default unless override is explicit.
- [x] Recorded decision that approval reuse is explicit scoped-grant behavior, not an implicit broker cache.
- [x] Recorded decision that namespace hierarchy is canonical and presentation groups can be derived.

### Slice 1: Workflow Event Subscribe Cleanup

- [x] Added `HostedProcessSandboxBase.event_subscribe(...)` as the canonical workflow subscription read API.
- [x] Changed workflow Python and workflow JS service subscription methods to use `event_subscribe(...)` instead of delegating to legacy `stream_recv(...)`.
- [x] Removed public `workflow-python-stream-recv` and `workflow-js-stream-recv` command dispatch from:
  - control channel wrappers
  - daemon local IPC dispatch
  - CLI command tables
  - auth role command sets
  - daemon policy allowlists
- [x] Updated interactive CLI workflow event viewing to call `workflow-*-event-subscribe`.
- [x] Updated workflow stream tests to consume `normalized_events` and `batch.loss`.
- [x] Replaced active stream-path `HostedStreamEvent(...)` construction with direct `HostedStreamFrame`/`HostedStreamBatch` row building.
- [x] Updated public docs to describe `workflow-*-event-subscribe` as the workflow event read command.
- [x] Added client breaking-change instructions to `HOSTING_CLIENT_BREAKING_CHANGES.md`.

### Slice 2: Internal Legacy Shape Cleanup

- [x] Removed the legacy `HostedProcessSandboxBase.stream_recv(...)` response shape from the shared workflow process base.
- [x] Updated process-base tests to assert `event_subscribe(...)`, `batch`, and `normalized_events` only.
- [x] Kept `proxy-stream-recv` as a lower-level generic worker/proxy primitive rather than a workflow compatibility route.
- [x] Audited terminal output handling and removed duplicate JS terminal stdout emission when console output was already streamed live.
- [x] Marked completed cleanup items in `hosting_access_plan.md`.

## Still Pending

- [x] Original Host Capability Protocol implementation checklist is complete.
- [ ] Implement the follow-up breaking-change slice that moves known broker-supported methods out of daemon-owned built-ins and into default hosting-client registration.
- [ ] Request dependent-client adoption only after the follow-up breaking-change slice is implemented, documented, tested, and committed.

## Verification

- [x] `pytest tests/test_engine_host_channel.py tests/test_engine_host_cli_interactive.py tests/test_engine_host_cli_remote_args.py tests/test_hosting_auth_roles.py -q`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "daemon_dispatches_workflow or stream"`
- [x] `pytest tests/test_hosting_sandbox_process_base.py tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_runtime_pool.py -q`
- [x] `pytest tests/test_hosting_sandbox_process_base.py -q`
- [x] `$hostingTests = Get-ChildItem -Path tests -Filter 'test_hosting*.py' | ForEach-Object { $_.FullName }; pytest @hostingTests tests/test_engine_host_channel.py tests/test_engine_host_cli_interactive.py tests/test_engine_host_cli_remote_args.py tests/test_workflow_helper_service.py tests/test_engine_worker_ipc_streaming.py -q`
- [x] `python -m py_compile src/hosting/sandbox/host_capabilities.py src/hosting/service/workflow_helpers.py`
- [x] `pytest tests/test_host_capabilities.py -q`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "workflow_js_stream_emits_terminal_events_and_artifacts or workflow_js_node_async_host_call_uses_broker or sandbox_describe or host_api"`
- [x] `pytest tests/test_workflow_js_node_runtime.py -q -k "host_dispatcher or sandbox_describe or structured_host_api_error_reason"`
- [x] `python -m py_compile src/hosting/sandbox/host_capabilities.py src/hosting/service/workflow_helpers.py src/hosting/service/state.py`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "host_capability_audit_event_persists_in_control_state or workflow_js_stream_emits_terminal_events_and_artifacts or workflow_js_node_async_host_call_uses_broker or sandbox_describe or host_api"`
- [x] `python -m py_compile tests/test_hosting_daemon_acl.py`
- [x] `pytest tests/test_hosting_daemon_acl.py -q -k "host_capability_session"`
