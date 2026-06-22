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

### Host Capability Slice 10: Known Method Registration Migration

- [x] Added hosting client helper descriptors for known broker-supported methods: `fs.list`, `fs.read_text`, `fs.write_text`, `fs.mkdir`, `fs.stat`, and `http.fetch`.
- [x] Added `EngineHostControlChannel.host_capability_session_register_known_methods(...)` for client-side registration.
- [x] Added explicit `allow_override` registration flag.
- [x] Changed duplicate fully-qualified method registration to fail by default with `host_capability_duplicate_method:<method>`.
- [x] Removed built-in precedence from broker method resolution; explicit override now wins.
- [x] Documented that dependent clients should adopt known-method registration because service-owned fallback is no longer implicit.

### Host Capability Slice 11: Callable Surface Primitives

- [x] Added `hosting.callable_surface` helpers for converting toolbox/`ToolsView` metadata to `HostCapabilityDescriptor` rows.
- [x] Added descriptor-to-callable-schema helpers for sandbox/model-facing discovery.
- [x] Added optional descriptor `metadata` so toolbox allowed/advertised/hidden/disabled/gated/constraint state can be preserved without changing required descriptor fields.
- [x] Added provider callback wrapper helpers that validate `provider_call_id` and normalize success/error/timeout/cancel envelopes.
- [x] Added approval bridge helpers that sanitize arguments to argument keys and normalize decisions to `deny`, `allow_once`, or `add_to_scope`.
- [x] Added safe correlation metadata propagation helpers.
- [x] Added `EngineHostControlChannel.host_capability_session_upsert(...)`, filtered list/close helpers, and `host_capability_session_register_toolbox(...)`.
- [x] Added public filtered Host Capability audit reads through `host_capability_audit_list(...)`.
- [x] Made service-owned `fs.*` / `http.fetch` fallback opt-in by `sandbox.host_api.service_owned_fallback_enabled=true`.
- [x] Added audit/log diagnostics for service-owned fallback use.
- [x] Updated `HOST_API_refactoring.md`, `hosting_access_plan.md`, and `HOSTING_CLIENT_BREAKING_CHANGES.md`.

### Host Capability Slice 12: Toolbox Session Execution And Fallback Removal

- [x] Threaded daemon-owned Host Capability sessions into workflow Python/JS execute and stream-open paths.
- [x] Registered visible provider sessions with the node Host Capability broker for request/workflow/instance/consumer scoped discovery and dispatch.
- [x] Added toolbox-session provider invocation through the existing `toolbox_execute(...)` harness.
- [x] Normalized toolbox `tool_call.result` JSON into sandbox-facing `host.call(...)` results.
- [x] Added private `toolbox_harness` binding support for `toolbox_session` providers.
- [x] Changed service-owned `fs.*` / `http.fetch` fallback from implicit default to explicit opt-in diagnostics.
- [x] Kept fallback audit/log marker coverage for explicitly enabled fallback calls.

### Host Capability Slice 13: Scoped Approval Grants

- [x] Added broker-local scoped approval grants for `add_to_scope` decisions.
- [x] Kept `allow_once` as a one-call approval with no reusable grant.
- [x] Matched reusable approval grants by method, provider, actor, scope requirements, and optional argument constraints.
- [x] Added grant TTL support from decision `ttl_seconds` or descriptor approval TTL.
- [x] Emitted approval/audit records for reused scoped grants.

### Host Capability Slice 13B: Client Session Callback Runtime

- [x] Added `HostCapabilityProviderCallbackRelay` as a public helper for local `client_session` callback bindings.
- [x] Wired workflow Host Capability provider dispatch to invoke `client_session` providers through the callback binding.
- [x] Preserved the normalized `hosting.sandbox.host_capability_call.v1` envelope and `provider_call_id` validation through the public callback helper.
- [x] Updated client change notes with the callback relay registration pattern.
- [x] This completes the dependent-client callable-surface feature request pause point.

### State And Long-Lived Instances Slice 14: Host-Managed State Capabilities

- [x] Added durable `sandbox_state` storage to hosting runtime state.
- [x] Added versioned JSON state helpers for backend, workflow, instance, and request partitions.
- [x] Exposed opt-in workflow node Host Capability methods for `state.<scope>.get`, `state.<scope>.set`, `state.<scope>.list`, and `state.<scope>.delete`.
- [x] Advertised available state scopes through `sandbox.describe()["state"]`.
- [x] Kept state capabilities separate from the deprecated service-owned `fs.*` / `http.fetch` fallback path.
- [x] Added workflow-node tests for default-disabled behavior and workflow state persistence across requests.

### State And Long-Lived Instances Slice 15: Python Node Pinned Instances

- [x] Added pinned Python node runtime instances in `WorkflowPythonNodeRuntimeRegistry`.
- [x] Added service APIs for Python node instance create, execute, list, and close.
- [x] Added channel, daemon, auth, and CLI command dispatch for `workflow-python-instance-*`.
- [x] Kept ordinary `workflow-python-execute` behavior unchanged; pinned routing is used through the explicit instance execute API.
- [x] Rejected project-mode Python node instances until cwd/sys.path/env/import-cache semantics are declared.
- [x] Added tests proving sequential instance calls route to the same worker process and preserve process-local mutation.

### State And Long-Lived Instances Slice 16: JS Node Pinned Instances

- [x] Added pinned JS node runtime instances in `WorkflowJsNodeRuntimeRegistry`.
- [x] Added service APIs for JS node instance create, execute, list, and close.
- [x] Added channel, daemon, auth, policy, and CLI command dispatch for `workflow-js-instance-*`.
- [x] Kept ordinary `workflow-js-execute` behavior unchanged; pinned routing is used through the explicit instance execute API.
- [x] Added JS worker PID runtime metadata so routed calls can be observed and tested.
- [x] Added tests proving sequential JS instance calls route to the same worker process.

### State And Long-Lived Instances Slice 17: Explicit State Snapshots

- [x] Added `sandbox_state_snapshot(...)` for scoped host-managed state partitions.
- [x] Added `sandbox_state_restore(...)` with `merge` and `replace` modes.
- [x] Added channel, daemon, auth, policy, and CLI command dispatch for `sandbox-state-snapshot` and `sandbox-state-restore`.
- [x] Kept snapshot/restore limited to explicit host-managed state, not arbitrary process memory.
- [x] Added tests for instance partition snapshot/restore and daemon forwarding.

### State And Long-Lived Instances Slice 18: Python Project Instance Policy

- [x] Added an explicit Python project instance policy gate for pinned project-mode instances.
- [x] Required `cwd=reset`, `sys_path=reset`, `env=reset`, and `import_cache=clear_project_modules`.
- [x] Restored cwd, `sys.path`, and `os.environ` after every project request.
- [x] Cleared project modules from `sys.modules` before and after project execution so module globals do not leak across calls.
- [x] Added tests for required policy and same-process project routing with isolated env/import state.

### State And Long-Lived Instances Slice 19: JS Project Instance Decision

- [x] Kept JS project-mode pinned instances unsupported because the current worker pins only the host worker process and creates a fresh QuickJS context for each request.
- [x] Added structured deferred detail to `workflow_js_instance_create` while preserving the existing `workflow_js_instance_project_mode_unsupported` reason.
- [x] Added a workflow-helper test for the deferred JS project-mode instance response.
- [x] Updated the main plan, Host API plan, and JS worker docs to record the decision and future runtime prerequisites.

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
- [x] Implement the follow-up breaking-change slice that exposes known broker-supported method registration through hosting client helpers.
- [x] Implement callable-surface primitives requested by the dependent client team.
- [x] Complete toolbox-backed provider callback runtime so hosted toolbox sessions can execute as Host Capability providers.
- [x] Remove implicit service-owned `fs.*` / `http.fetch` fallback from workflow node dispatch.
- [x] Implement scoped `add_to_scope` approval reuse for Host Capability calls.
- [x] Implement explicit host-managed state capability methods.
- [x] Implement explicit Python node instance create/execute/list/close APIs.
- [x] Implement explicit JS node instance create/execute/list/close APIs.
- [x] Add snapshot/restore hooks for instance-local host-managed state.
- [ ] Dependent clients should adopt explicit callable-session registration, callback relay bindings, and callable-surface helpers.
- [x] Decide and implement Python long-lived project-mode cwd/sys.path/env/import-cache policy.
- [x] Decide JS project-mode long-lived semantics.
- [ ] Implement JS project-mode long-lived runtime after persistent QuickJS context/module-cache support exists.
- [ ] Remove diagnostic service-owned `fs.*` / `http.fetch` fallback and fallback-only tests after dependent clients complete migration.

## Verification

- [x] `pytest tests/test_engine_host_channel.py tests/test_engine_host_cli_interactive.py tests/test_engine_host_cli_remote_args.py tests/test_hosting_auth_roles.py -q`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "daemon_dispatches_workflow or stream"`
- [x] `pytest tests/test_hosting_sandbox_process_base.py tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_runtime_pool.py -q`
- [x] `pytest tests/test_hosting_sandbox_process_base.py -q`
- [x] `$hostingTests = Get-ChildItem -Path tests -Filter 'test_hosting*.py' | ForEach-Object { $_.FullName }; pytest @hostingTests tests/test_engine_host_channel.py tests/test_engine_host_cli_interactive.py tests/test_engine_host_cli_remote_args.py tests/test_workflow_helper_service.py tests/test_engine_worker_ipc_streaming.py -q`
- [x] `python -m py_compile src/hosting/sandbox/host_capabilities.py src/hosting/service/workflow_helpers.py`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "workflow_js_node_instance or workflow_js_node_project_instance"`
- [x] `python -m py_compile src/hosting/sandbox/workflow_js_node_runtime.py`
- [x] `pytest tests/test_host_capabilities.py -q`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "workflow_js_stream_emits_terminal_events_and_artifacts or workflow_js_node_async_host_call_uses_broker or sandbox_describe or host_api"`
- [x] `pytest tests/test_workflow_js_node_runtime.py -q -k "host_dispatcher or sandbox_describe or structured_host_api_error_reason"`
- [x] `python -m py_compile src/hosting/sandbox/host_capabilities.py src/hosting/service/workflow_helpers.py src/hosting/service/state.py`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "host_capability_audit_event_persists_in_control_state or workflow_js_stream_emits_terminal_events_and_artifacts or workflow_js_node_async_host_call_uses_broker or sandbox_describe or host_api"`
- [x] `python -m py_compile tests/test_hosting_daemon_acl.py`
- [x] `pytest tests/test_hosting_daemon_acl.py -q -k "host_capability_session"`
- [x] `python -m pytest tests/test_callable_surface.py tests/test_engine_host_channel.py -q -k "callable_surface or host_capability"`
- [x] `python -m pytest tests/test_workflow_helper_service.py -q -k "host_capability_audit or service_owned_fallback"`
- [x] `python -m pytest tests/test_host_capabilities.py tests/test_hosting_daemon_acl.py -q -k "host_capability"`
- [x] `python -m pytest tests/test_callable_surface.py tests/test_host_capabilities.py tests/test_engine_host_channel.py tests/test_workflow_helper_service.py tests/test_hosting_daemon_acl.py -q -k "callable_surface or host_capability or service_owned_fallback or daemon_dispatches_workflow or host_api"`
- [x] `python -m pytest tests/test_host_capabilities.py -q`
