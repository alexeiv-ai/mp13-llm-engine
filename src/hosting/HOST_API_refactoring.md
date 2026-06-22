# Host API Refactoring Status

Date: 2026-06-21

This document tracks the Host Capability Protocol pillar after the dependent-client callable-surface feature request. The pillar is implemented. The remaining entries here are deferred cross-pillar work, not blockers for Host API adoption.

## Completed Pillar

- [x] Sandbox-callable Host APIs are described as Host Capability descriptors.
- [x] Sandbox code calls stable harness entry points such as Python `host.call(...)` and JavaScript `api.callAsync(...)`.
- [x] Workers do not receive provider bindings, callback addresses, provider session tokens, or approval credentials.
- [x] The service/daemon brokers capability resolution, policy checks, approval gating, provider invocation, normalized host responses, event observations, and audit rows.
- [x] Hosting clients own extension capabilities through registered provider sessions.
- [x] Provider sessions can be scoped by request, workflow, instance, consumer, owner, provider, visibility, and method.
- [x] Duplicate fully-qualified method registration fails by default unless `allow_override=true` is explicit.
- [x] Namespace hierarchy is the canonical capability hierarchy; group paths remain presentation metadata.
- [x] Service-owned `fs.*` and `http.fetch` are no longer workflow-node capabilities.
- [x] Diagnostic service-owned `fs.*` / `http.fetch` fallback registration, dispatch, audit markers, and fallback-only tests have been removed.
- [x] Legacy fallback policy keys such as `sandbox.host_api.service_owned_fallback_enabled` are ignored by workflow node Host Capability dispatch.
- [x] Worker-side `api.fs` and `api.http` convenience wrappers are preserved only as callers of advertised Host Capability methods. They do not prove that those methods exist.

## Current Programming Model

Sandboxed code still uses a simple method-call surface:

```python
result = host.call("crm.customer.lookup", {"customer_id": "123"})
```

```javascript
const result = await api.callAsync("crm.customer.lookup", {customer_id: "123"});
```

The sandbox sees method names, schemas, policy hints, and approved scoped capabilities. It does not know whether a method is implemented by a hosting client callback, a hosted toolbox session, state storage, or a future action provider.

Hosting clients use the control channel and callable-surface helpers to register providers:

- `known_host_capability_methods(...)`
- `host_capability_session_register_known_methods(...)`
- `host_capability_session_register(...)`
- `host_capability_session_upsert(...)`
- `host_capability_session_list_filtered(...)`
- `host_capability_session_close_filtered(...)`
- `host_capability_session_register_toolbox(...)`
- `host_capability_audit_list(...)`
- `HostCapabilityProviderCallbackRelay.bind_callback(...)`

Clients should use high-level helpers and avoid raw provider session tokens or callback envelopes unless they are implementing low-level transport integration.

## Callable Surface Primitives

- [x] Added `src/hosting/callable_surface.py` as the shared descriptor/callback/approval helper module.
- [x] Added `toolbox_to_host_capability_descriptors(...)` to convert toolbox descriptions plus optional `ToolsView` into Host Capability descriptors.
- [x] Added `host_capability_descriptors_to_callable_schemas(...)` to convert descriptors into sandbox/model-facing callable schemas.
- [x] Preserved allowed, advertised, hidden allowed, disabled, gated, constraints, schemas, permissions, approval metadata, provider identity, and toolbox metadata in descriptor conversion.
- [x] Added descriptor `metadata` support so adapters can carry toolbox/view information without changing the core descriptor contract.
- [x] Added `extract_safe_correlation_metadata(...)` for safe correlation propagation.
- [x] Safe correlation metadata includes workflow, instance, node, request, cursor, context, branch, session-tree, session, toolbox, actor, provider, method, approval, host-call, and provider-call ids.
- [x] Added callable-surface identity and digest helpers so merged views can preserve `provider_kind`, `provider_id`, `toolbox_id`, `session_id`, method, schema digest, method digest, and policy digest.
- [x] Added direct `toolbox_to_callable_schemas(...)` adapter export for clients that need model/sandbox-facing callable schemas without making Host Capability descriptors native toolbox storage.
- [x] Callable schemas include hierarchical `group_path` so native toolbox hierarchy can be exposed through the adapter boundary.
- [x] Merged callable-schema conversion rejects duplicate advertised method names by default; clients must namespace/alias or explicitly choose first-provider wins behavior.
- [x] Added explicit bridge-policy helper for intersecting toolbox policy, Host Capability caller policy, and bridge policy.

## Provider Callback Runtime

- [x] Added provider callback helpers for `hosting.sandbox.host_capability_call.v1`.
- [x] `bind_host_capability_provider_callback(...)` validates `provider_call_id`.
- [x] `HostCapabilityProviderCallbackRelay.bind_callback(...)` creates local callback bindings for `client_session` providers.
- [x] Workflow node Host Capability dispatch invokes `client_session` providers through local callback bindings.
- [x] Provider helper responses normalize success, error, timeout, and cancel outcomes.
- [x] Structured provider errors are surfaced without leaking private binding details to sandbox code.
- [x] Provider timeout maps to `host_call_timeout`.
- [x] Provider disconnect maps to `host_capability_provider_unavailable`.
- [x] Request cancellation cancels or closes in-flight provider calls where the broker can still act.
- [x] Local IPC remains the optimized first implementation target. SSH relay remains a corner-case path unless already available in touched code.

## Approval Bridge

- [x] Added approval helpers for `hosting.sandbox.host_capability_approval.v1`.
- [x] Approval requests expose sanitized argument keys, method, provider, approval policy, workflow/request/instance context, and safe correlation ids.
- [x] Approval decisions normalize to `deny`, `allow_once`, or `add_to_scope`.
- [x] `deny` rejects the current call before provider execution.
- [x] `allow_once` approves only the current call.
- [x] `add_to_scope` records a scoped grant with optional argument constraints and TTL for later matching calls.
- [x] There is no implicit broker approval cache. Reuse must be explicit through scoped grants.
- [x] Durable audit records remain the source of truth for security-relevant approval decisions; lossy live streams are observation only.

## Toolbox As Provider

- [x] Added a toolbox-as-provider registration helper through `host_capability_session_register_toolbox(...)`.
- [x] Toolbox-backed provider sessions use provider kind `toolbox_session`.
- [x] Toolbox-backed provider invocation executes through the existing toolbox harness.
- [x] Toolbox `ToolsView`, gating, scoped approvals, and execution accounting are reused where practical.
- [x] HostedToolbox brokered IO unification is intentionally deferred to a later uber-plan pillar.

## Audit And Diagnostics

- [x] Added filtered Host Capability audit reads through `host_capability_audit_list(...)`.
- [x] Audit filters include workflow id, instance id, request id, provider id, method, approval id, time window, limit, and offset.
- [x] Provider bindings and tokens are absent from public session listings and sandbox-facing discovery.
- [x] Host call, host response, approval, denial, provider failure, and cancellation observations are emitted through the event-stream pillar.

## Known Method Migration

- [x] Descriptor helpers for known broker-supported `fs.*` and `http.fetch` methods remain available.
- [x] Clients can explicitly register those known methods through Host Capability sessions.
- [x] Clients can omit known methods or provide custom implementations.
- [x] Diagnostic service-owned fallback has been removed; known methods exist only when a client/provider session registers them.

## Related Follow-Up Work Already Implemented

These items were enabled by the Host Capability model but belong to later plan pillars:

- [x] Host-managed sandbox state methods are exposed as opt-in Host Capabilities:
  - `state.backend.get`
  - `state.backend.set`
  - `state.backend.list`
  - `state.backend.delete`
  - `state.workflow.get`
  - `state.workflow.set`
  - `state.workflow.list`
  - `state.workflow.delete`
  - `state.instance.get`
  - `state.instance.set`
  - `state.instance.list`
  - `state.instance.delete`
- [x] Sandbox state snapshot/restore commands were added for recoverable long-lived instance state.
- [x] Python node pinned instance create/execute/list/close routing was added.
- [x] JavaScript node pinned instance create/execute/list/close routing was added.
- [x] JavaScript worker runtime metadata now reports the host worker pid for routed live instances.
- [x] Python project-mode pinned instances are allowed only with an explicit isolation policy that resets cwd, `sys.path`, env, and project import modules between calls.
- [x] JavaScript project-mode pinned instances remain unsupported until the JS runtime can preserve a QuickJS context or module graph under explicit cleanup and snapshot/restore semantics.
- [x] Service-level Python and JavaScript action manifests, card-facing discovery helpers, and action execution routing were added on top of existing worker entrypoint fields.
- [x] Action describe/execute commands were exposed through the daemon, control channel, CLI, auth, and policy command sets.

## Test Maintenance

- [x] Host Capability descriptor, broker, approval, provider callback, duplicate registration, scope matching, and audit helper tests are present.
- [x] Callable-surface adapter tests cover toolbox descriptor conversion, callable schema filtering, callback normalization, and approval request/decision shaping.
- [x] Daemon/control channel tests cover registration, duplicate override behavior, SSH auth binding preservation, disconnect cleanup, helper forwarding, filtered lifecycle helpers, audit reads, and toolbox provider registration.
- [x] Workflow helper tests cover that legacy diagnostic fallback policy keys are ignored and unsupported known methods fail through normal Host Capability errors.
- [x] Convenience-wrapper tests no longer rely on service-owned `api.fs` or `api.http` behavior.
- [x] No test should assert that `fs.*` or `http.fetch` is implicitly registered for normal workflow node dispatch.

## Deferred Cross-Pillar Work

These items are intentionally outside the completed Host Capability pillar. They should be picked up only when their owning pillar starts, because each one changes a broader runtime or toolbox lifecycle boundary.

- [x] Decide against forcing HostedToolbox brokered IO onto Host Capability dispatch as the default toolbox lifecycle direction.
  - Owning pillar: toolbox lifecycle and brokered IO simplification.
  - Current state: hosted toolbox sessions can already register as Host Capability providers and execute through the existing toolbox harness. Toolbox brokered IO remains toolbox-native.
  - Decision: use shared callable-surface descriptors, identity/digest helpers, approval/audit helper contracts, and explicit bridge policies. Keep toolbox lifecycle/execution ownership separate from sandbox Host Capability dispatch unless a later concrete duplication problem justifies deeper runtime unification.
  - Conflict rule: overlapping method names are allowed across provider sessions but must not silently collapse in a merged advertised surface. Stable identity is `provider_kind + provider_id/toolbox_id + session_id + method`.
  - Approval rule: reusable grants default to same provider/session scope. Cross-toolbox reuse requires explicit scope identity such as toolbox definition digest, owner/workspace, method/schema/policy digests, and compatible constraints.
  - Bridge rule: brokered IO permissions come from an explicit bridge policy intersected with toolbox policy and Host Capability caller policy.

- [ ] Expand long-lived routable instance state recovery policies beyond the current snapshot/restore primitives.
  - Owning pillar: stateful/recoverable sandbox instances.
  - Current state: host-managed backend/workflow/instance/request JSON state can be snapshotted and restored. Arbitrary Python/JS process memory is intentionally not captured.
  - Rework candidate: define instance recovery policy objects covering restart behavior, state partitions, mutation checkpoints, durable artifact references, and failure recovery after worker shutdown.
  - Expected benefit: long-lived routed instances can survive planned shutdowns or worker replacement without pretending raw process memory is durable.
  - Main risk: exposing too much process state would create unstable recovery semantics. Recovery should stay explicit and host-managed unless a runtime-specific checkpoint contract is added.
  - Trigger to start: when dependent workflows need routable instances to survive host restarts or planned worker recycling.

- [ ] Implement JS project-mode long-lived runtime after persistent QuickJS context or module graph cache semantics are available.
  - Owning pillar: JavaScript runtime evolution.
  - Current state: `workflow_js_instance_*` pins a host worker process, but each request creates a fresh QuickJS context from `module_source`. Project-mode instance creation returns `workflow_js_instance_project_mode_unsupported` with deferred detail.
  - Rework candidate: add a persistent QuickJS context or module graph cache, then define explicit cwd/env/import-cache cleanup rules and snapshot/restore boundaries for mutable JS state.
  - Expected benefit: JS project-style authoring can become truly warm and routable instead of only using a pinned transport process.
  - Main risk: persistent JS contexts can leak globals, async jobs, host handles, or imported module state across actions unless cleanup policy is explicit.
  - Trigger to start: after JS runtime design accepts persistent context/module-cache ownership and defines the state cleanup contract.

- [x] Decide native toolbox metadata should continue using adapters rather than directly adopting Host Capability descriptors.
  - Owning pillar: native toolbox metadata and discovery.
  - Decision: toolbox descriptions and `ToolsView` remain the native toolbox metadata model. Host Capability descriptors and callable schemas are adapter/export formats.
  - Rationale: toolbox lifecycle, execution, install/config, storage, and policy semantics remain toolbox-specific. Descriptor adoption would create broad churn without improving the immediate client integration path.
  - Implemented export behavior: `toolbox_to_callable_schemas(...)` emits namespace-qualified names, `group_path`, visibility/gating state, provider/session identity, digests, and toolbox metadata.
