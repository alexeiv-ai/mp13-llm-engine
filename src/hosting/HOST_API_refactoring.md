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
- [x] Service-owned `fs.*` and `http.fetch` are no longer implicit workflow-node capabilities.
- [x] Service-owned `fs.*` and `http.fetch` remain available only as an explicit diagnostic fallback when `sandbox.host_api.service_owned_fallback_enabled=true`.
- [x] Diagnostic service fallback emits `host_capability_service_fallback_used` audit rows and log markers.
- [x] Worker-side `api.fs` and `api.http` convenience wrappers are preserved only as callers of advertised Host Capability methods. They do not prove that those methods exist.

## Current Programming Model

Sandboxed code still uses a simple method-call surface:

```python
result = host.call("crm.customer.lookup", {"customer_id": "123"})
```

```javascript
const result = await api.callAsync("crm.customer.lookup", {customer_id: "123"});
```

The sandbox sees method names, schemas, policy hints, and approved scoped capabilities. It does not know whether a method is implemented by a hosting client callback, a hosted toolbox session, a diagnostic service fallback, state storage, or a future action provider.

Hosting clients use the control channel and callable-surface helpers to register providers:

- `known_host_capability_methods(...)`
- `host_capability_session_register_known_methods(...)`
- `host_capability_session_register(...)`
- `host_capability_session_upsert(...)`
- `host_capability_session_list_filtered(...)`
- `host_capability_session_close_filtered(...)`
- `host_capability_session_register_toolbox(...)`
- `host_capability_audit_list(...)`

Clients should use high-level helpers and avoid raw provider session tokens or callback envelopes unless they are implementing low-level transport integration.

## Callable Surface Primitives

- [x] Added `src/hosting/callable_surface.py` as the shared descriptor/callback/approval helper module.
- [x] Added `toolbox_to_host_capability_descriptors(...)` to convert toolbox descriptions plus optional `ToolsView` into Host Capability descriptors.
- [x] Added `host_capability_descriptors_to_callable_schemas(...)` to convert descriptors into sandbox/model-facing callable schemas.
- [x] Preserved allowed, advertised, hidden allowed, disabled, gated, constraints, schemas, permissions, approval metadata, provider identity, and toolbox metadata in descriptor conversion.
- [x] Added descriptor `metadata` support so adapters can carry toolbox/view information without changing the core descriptor contract.
- [x] Added `extract_safe_correlation_metadata(...)` for safe correlation propagation.
- [x] Safe correlation metadata includes workflow, instance, node, request, cursor, context, branch, session-tree, actor, provider, method, approval, host-call, and provider-call ids.

## Provider Callback Runtime

- [x] Added provider callback helpers for `hosting.sandbox.host_capability_call.v1`.
- [x] `bind_host_capability_provider_callback(...)` validates `provider_call_id`.
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
- [x] Service-owned fallback is disableable and off by default for normal workflow node dispatch.
- [x] When fallback is explicitly enabled, it is treated as a diagnostic compatibility path, not the ownership model.

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

## Test Maintenance

- [x] Host Capability descriptor, broker, approval, provider callback, duplicate registration, scope matching, and audit helper tests are present.
- [x] Callable-surface adapter tests cover toolbox descriptor conversion, callable schema filtering, callback normalization, and approval request/decision shaping.
- [x] Daemon/control channel tests cover registration, duplicate override behavior, SSH auth binding preservation, disconnect cleanup, helper forwarding, filtered lifecycle helpers, audit reads, and toolbox provider registration.
- [x] Workflow helper tests cover diagnostic service fallback audit markers and fallback-disable behavior.
- [x] JavaScript convenience-wrapper tests must opt into diagnostic service fallback explicitly when they exercise service-owned `api.fs` or `api.http` behavior.
- [x] No test should assert that `fs.*` or `http.fetch` is implicitly registered for normal workflow node dispatch.

## Deferred Cross-Pillar Work

- [ ] Rework HostedToolbox brokered IO on top of Host Capability dispatch if the later toolbox lifecycle pillar chooses that simplification.
- [ ] Add card/action discovery and invocation on top of the same descriptor/callable-surface primitives.
- [ ] Expand long-lived routable instance state recovery policies beyond the current snapshot/restore primitives.
- [ ] Decide whether native toolbox metadata should directly adopt Host Capability group/namespace descriptors or continue using adapters.
