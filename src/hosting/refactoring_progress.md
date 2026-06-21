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

- [ ] Continue Host Capability Protocol with daemon-mediated provider callback transport, permissions/scopes, approval routing, event observations, and durable audit.

## Verification

- [x] `pytest tests/test_engine_host_channel.py tests/test_engine_host_cli_interactive.py tests/test_engine_host_cli_remote_args.py tests/test_hosting_auth_roles.py -q`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "daemon_dispatches_workflow or stream"`
- [x] `pytest tests/test_hosting_sandbox_process_base.py tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_runtime_pool.py -q`
- [x] `pytest tests/test_hosting_sandbox_process_base.py -q`
- [x] `$hostingTests = Get-ChildItem -Path tests -Filter 'test_hosting*.py' | ForEach-Object { $_.FullName }; pytest @hostingTests tests/test_engine_host_channel.py tests/test_engine_host_cli_interactive.py tests/test_engine_host_cli_remote_args.py tests/test_workflow_helper_service.py tests/test_engine_worker_ipc_streaming.py -q`
