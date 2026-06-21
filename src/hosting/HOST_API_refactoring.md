# Host API Refactoring Implementation Plan

Date: 2026-06-20

This plan is scoped to the Host Capability Protocol pillar. It starts after the event-streaming pillar and assumes dependent clients can adopt breaking API changes. It does not need compatibility fallbacks for existing hosting consumers.

The goal is to make sandbox-callable host APIs owned by hosting clients, with the daemon/service acting as broker, policy enforcer, lifecycle owner, and audit/stream observer. The worker harness should expose a stable `host.call(...)` style entry point, but should not own provider routing or user approval.

## Current Baseline

- [ ] Keep useful existing pieces:
  - Python and JavaScript node workers already send `host_call` and receive `host_response` over child IPC.
  - `host_call_id` already correlates concurrent out-of-order responses inside one worker/request IPC conversation.
  - `HostApiRegistry` already describes and dispatches service-owned built-ins such as `fs.*` and `http.fetch`.
  - Toolbox code already has discoverable tools, schemas, scopes, callback relay, approvals, and hosted execution accounting.
  - The completed event-streaming pillar now provides `host_call`, `host_response`, `approval`, `error`, and `done` observations with request/instance correlation.
- [ ] Replace the ownership model:
  - old: request-local service-owned `HostApiRegistry` is the effective host API owner
  - new: hosting clients own extension capabilities through registered provider sessions; service built-ins are just one provider class
- [ ] Do not expose provider bindings, provider session tokens, callback addresses, or approval credentials to sandbox code.
- [ ] Do not route user approvals through the sandbox. Approvals flow outward to the hosting client/user-facing side.

## Why This Pillar Next

- [ ] Host API ownership is the next blocking contract after streaming.
- [ ] Reason: later pillars need sandbox-to-host capabilities:
  - state and recovery need host-owned state read/write methods
  - card actions need discoverable callable entries
  - long-lived instances need scoped provider/session routing
  - approval and permission gates need host-side decision routing
- [ ] Constraint: this pillar implements capability ownership, routing, discovery, permissions, approvals, and callback transport. It reserves state/action hooks but does not implement durable state stores, action manifests, or routable long-lived instances.

## Target Model

Use four layers:

- [ ] Harness API: language/runtime entry points injected into sandboxed code, such as Python `host.call(...)` and JavaScript `api.callAsync(...)`.
- [ ] Capability broker: daemon/service-owned resolver, policy checker, approval gate, audit emitter, timeout/cancel coordinator, and provider invoker.
- [ ] Capability providers: built-in service providers or client-owned provider sessions that implement callable methods.
- [ ] Capability descriptors: shared discoverable metadata, using toolbox-style schemas, scopes, permissions, approval metadata, grouping, and result contracts.

Sandbox code still calls a simple method:

```python
host.call("crm.customer.lookup", {"customer_id": "123"})
```

or:

```javascript
await api.callAsync("crm.customer.lookup", {customer_id: "123"});
```

The sandbox should not know whether the method is implemented by service built-ins, a GUI backend, an orchestration process, or another authorized hosting consumer.

## Capability Descriptor Contract

### Design Goal

- [ ] Use one descriptor shape for built-ins, client-owned providers, toolbox-backed providers, state providers, and future action providers.
- [ ] Reuse toolbox concepts rather than inventing a separate capability vocabulary.
- [ ] Keep descriptors sandbox-safe: discovery exposes method metadata and policy hints, not provider connection details.
- [ ] Include hierarchical groups so navigation, scoping, and UI presentation can be organized without parsing method-name prefixes.

### Descriptor Shape

- [ ] Use this shape for host-callable capabilities:

```json
{
  "contract": "hosting.sandbox.host_capability.v1",
  "name": "crm.customer.lookup",
  "namespace": "crm",
  "group_path": ["CRM", "Customer"],
  "description": "Look up a customer record by id.",
  "args_schema": {},
  "result_schema": {},
  "permissions": ["crm.customer.read"],
  "scope_requirements": [
    {"scope": "crm.customer", "access": "read"}
  ],
  "approval": {
    "mode": "none",
    "cache_key": "method+scope+actor",
    "ttl_seconds": 0
  },
  "provider": {
    "provider_id": "provider-id",
    "kind": "builtin|client_session|toolbox_session",
    "owner": "hosting-consumer-id",
    "visibility": "request|workflow|instance|consumer"
  }
}
```

- [ ] Required fields:
  - `contract`
  - `name`
  - `namespace`
  - `group_path`
  - `args_schema`
  - `result_schema`
  - `permissions`
  - `scope_requirements`
  - `approval`
  - `provider.kind`
  - `provider.provider_id`
  - `provider.visibility`
- [ ] Descriptor validation must reject:
  - empty names
  - names outside granted namespaces
  - unsupported provider kinds
  - invalid group paths
  - oversized schemas/descriptions
  - provider transport details in sandbox-facing discovery

## Capability Sessions

### Client-Owned Provider Sessions

- [ ] Hosting clients register provider sessions through daemon/control APIs.
- [ ] Session registration is authenticated by normal hosting auth and may be bound to SSH session binding when remote.
- [ ] A provider session owns one or more capability descriptors.
- [ ] The callable implementation lives in the hosting client process or a client-owned helper process, so RPC is required.

Session shape:

```json
{
  "contract": "hosting.sandbox.host_capability_session.v1",
  "session_id": "cap-session-id",
  "owner": "client-or-actor-id",
  "scope": {
    "request_id": null,
    "workflow_id": "workflow-id",
    "instance_id": null,
    "consumer_id": "consumer-id"
  },
  "methods": [],
  "binding": {
    "transport": "daemon_callback|local_ipc|ssh_relay",
    "address": "opaque-host-only-address",
    "session_token": "opaque-host-only-token"
  },
  "lifetime": {
    "created_at_ms": 1781913600000,
    "expires_at_ms": null,
    "close_on_client_disconnect": true
  }
}
```

- [ ] The broker stores binding details, but `sandbox.describe` must never return them.
- [ ] Service-owned built-in methods should be removed from the daemon-owned provider model.
- [ ] Known broker-supported methods such as `fs.*` and `http.fetch` should be registered by the hosting client library by default, with custom client implementations allowed.
- [ ] Duplicate fully-qualified method registration must fail by default with a clear duplicate error unless the registration explicitly requests override.
- [ ] Different namespaces may use the same local method suffix because calls use fully-qualified method names.

### Scoped Toolbox Reference

- [ ] Treat the host API as a sandbox-facing toolbox:
  - descriptors look like toolbox entries
  - scopes and permissions look like toolbox scopes
  - group hierarchy looks like toolbox grouping
  - approvals reuse HostedToolbox-style approval machinery where practical
- [ ] Direction is reversed from normal hosted toolbox execution:
  - normal hosted toolbox: host/client calls into sandbox/worker
  - host capability toolbox: sandbox calls outward to host/client providers
- [ ] A scoped toolbox reference should grant only the approved subset of methods to a request/workflow/instance.
- [ ] The sandbox receives method discovery, not raw toolbox/provider handles.

## Broker Routing

### Call Flow

- [ ] Worker emits `host_call` over child IPC with:
  - `host_call_id`
  - `method`
  - `arguments`
  - `request_id`
  - optional `instance_id`
- [ ] Runtime forwards the call to the request/instance capability broker.
- [ ] Broker resolves the method against:
  - active provider sessions
  - request/workflow/instance/consumer scopes
  - namespace permissions
  - duplicate/override policy
- [ ] Broker checks static permissions and approved scopes.
- [ ] If approval is needed, broker emits an approval request toward the user-facing hosting client and waits for a decision.
- [ ] Broker invokes the provider.
- [ ] Broker maps provider result/error into normalized `host_response`.
- [ ] Runtime sends `host_response` back over the same child IPC path.
- [ ] Event stream emits observations for `host_call`, `host_response`, approval requested/resolved/denied, and provider failures.

### Correlation IDs

- [ ] Keep `host_call_id` local to the child worker/request IPC conversation.
- [ ] Introduce separate broker/provider IDs:
  - `provider_call_id`: provider callback correlation
  - `approval_id`: user approval correlation
  - `capability_id`: descriptor identity if needed for stable references
- [ ] Do not treat `host_call_id` as a daemon-global provider route.

### Provider Callback Envelope

Request:

```json
{
  "contract": "hosting.sandbox.host_capability_call.v1",
  "provider_call_id": "broker-call-id",
  "method": "crm.customer.lookup",
  "arguments": {},
  "context": {
    "request_id": "request-id",
    "instance_id": null,
    "workflow_id": "workflow-id",
    "package_id": "package-id",
    "actor": "hosting-consumer-id",
    "deadline_ms": 1781913605000,
    "permissions": ["crm.customer.read"],
    "approved_scopes": ["crm.customer:read"]
  }
}
```

Response:

```json
{
  "status": "ok",
  "provider_call_id": "broker-call-id",
  "result": {}
}
```

Error:

```json
{
  "status": "error",
  "provider_call_id": "broker-call-id",
  "reason": "crm_customer_not_found",
  "message": "Customer was not found.",
  "detail": {}
}
```

- [ ] Provider transport failures map to sandbox-visible host-call errors without leaking internal binding details by default.
- [ ] No automatic retry for provider calls after the provider has accepted a call.
- [ ] Idempotent retry can be a later descriptor field, not a first implementation requirement.

## Permission And Approval Model

- [ ] The broker is the sole authority for whether a sandbox request can call a method.
- [ ] The worker must not receive provider tokens, provider callback bindings, or approval credentials.
- [ ] Minimum policy dimensions:
  - hosting auth subject
  - provider session owner
  - workflow/package identity
  - request ID
  - optional instance ID
  - method namespace
  - permission/scope grants
  - approval cache key
- [ ] Approval direction is asymmetric:
  - request originates in sandbox
  - approval request goes outward to hosting client/user-facing process
  - decision returns to broker
  - sandbox only receives success or host-call error
- [ ] Approval events should use the completed event-stream pillar:
  - `approval` audit frames for request/resolution/denial
  - `host_call` and `host_response` observations
  - durable audit for security-relevant approval decisions
- [ ] Decision-bearing approval records must not rely only on lossy live stream retention.

## Harness Discovery

### Problem

- [ ] `host.describe()` currently mixes built-in methods, roots, policy, and transport facts.
- [ ] Sandboxed code and clients need to distinguish:
  - harness-provided APIs
  - host/client-provided extension APIs
  - built-in service APIs
  - worker-live versus host-generated event kinds
  - enabled state/action hooks
  - current scoped toolbox reference

### Discovery Shape

- [ ] Add `sandbox.describe` as the full discovery contract.
- [ ] `host.describe` can become a host-capability-only view; no old compatibility fallback is required after dependent clients migrate.

```json
{
  "contract": "hosting.sandbox.discovery.v1",
  "runtime": {
    "language": "python",
    "runtime_kind": "workflow_python_node",
    "harness_version": "1",
    "worker_contract": "hosting.workflow_python.node.v1"
  },
  "harness": {
    "execution_modes": ["module", "snippet", "project"],
    "globals": ["payload", "progress", "emit_progress", "host", "artifact_inputs", "artifact_outputs"],
    "host_api_entrypoints": ["host.call", "host.describe", "sandbox.describe"],
    "result_envelope": ["output", "state_patch", "artifacts", "progress"]
  },
  "events": {
    "worker_live": ["progress"],
    "host_generated": ["started", "heartbeat", "stdout", "stderr", "log", "artifact", "result", "error", "canceled", "done"],
    "observations": ["host_call", "host_response"],
    "reserved": ["approval", "state_notice", "action_notice"]
  },
  "host_capabilities": {
    "methods": [],
    "groups": [],
    "providers": [],
    "transport": {
      "framed": true,
      "host_call_id": true,
      "async_capable": true,
      "out_of_order_responses": true
    }
  },
  "state": {
    "available": false,
    "scopes": []
  },
  "actions": {
    "available": false,
    "entries": []
  },
  "policy": {},
  "roots": {}
}
```

- [ ] Discovery is assembled by the service because it has:
  - runtime harness facts
  - event-stream registry
  - built-in provider descriptors
  - active client-owned provider sessions
  - roots/artifact policy
  - state/action availability from later pillars
- [ ] The worker harness may expose static harness facts, but it does not discover client-owned provider sessions.

## Relationship To Toolbox

- [ ] Add hierarchical tool groups to native toolbox metadata, or first implement the grouping model in shared capability descriptors and adapt toolbox later.
- [ ] Prefer a shared descriptor module so toolbox and host capabilities converge over time.
- [ ] HostedToolbox brokered IO can later be reworked on top of host-call capability dispatch, but that is not part of the first Host API pillar slice.
- [ ] Do not import toolbox staging/repair lifecycle into node host API dispatch.
- [ ] Reuse approval and scope concepts where possible.

## Client-Facing API Strategy

- [ ] Hosting library should hide provider callback transport details.
- [ ] Client API should expose high-level operations:
  - register capability session
  - list/update/close capability session
  - provide method callback
  - approve/deny gated call
  - inspect capability audit/events
- [ ] Low-level clients may use raw daemon commands, but normal clients should not handle provider binding tokens or callback envelopes manually.
- [ ] Since dependent clients already adopted current breaking changes, no legacy host API compatibility layer is required for this pillar.

## Implementation Checklist

- [x] 1. Add shared host capability descriptor models.
- [x] 2. Add descriptor validation for names, namespaces, schemas, group paths, provider metadata, and visibility.
- [x] 3. Adapt service-owned built-ins (`fs.*`, `http.fetch`) to emit shared descriptors.
- [x] 4. Add hierarchical group metadata to descriptors and discovery output.
- [x] 5. Add `sandbox.describe` discovery assembled by the service.
- [x] 6. Convert `host.describe` / `api.describe` to the new host-capability view.
- [x] 7. Add capability broker with `describe(...)` and `dispatch_async(...)`.
- [x] 8. Wrap existing `HostApiRegistry` as a built-in provider behind the broker.
- [x] 9. Route Python and JavaScript node host calls through the broker.
- [x] 10. Add client-owned capability session registration/list/close daemon APIs.
- [x] 11. Add provider callback RPC envelopes and response validation.
- [x] 12. Add provider timeout, disconnect, and request-cancel handling.
- [x] 13. Add permission/scope checks for provider ownership, method namespace, and request/workflow/instance visibility.
- [x] 14. Add gated approval flow that routes outward to the owning hosting client/user-facing process.
- [x] 15. Emit event-stream observations for host calls, host responses, approvals, denials, provider failures, and cancellations.
- [x] 16. Add durable audit records for security-relevant approvals and denials.
- [x] 17. Update client breaking-change notes once this pillar exposes new public APIs.

## Test Checklist

- [x] Shared descriptor validation rejects invalid method names, namespaces, group paths, provider kinds, and oversized schemas.
- [x] Built-in `fs.*` descriptors normalize to shared capability descriptors.
- [x] Built-in `http.fetch` descriptor preserves policy and permission metadata.
- [x] `sandbox.describe` separates harness, events, capabilities, roots, policy, state, and actions.
- [x] Provider bindings and tokens are absent from sandbox-facing discovery.
- [x] Python `host.call(...)` dispatches a built-in through the broker wrapper.
- [x] JavaScript `api.callAsync(...)` dispatches a built-in through the broker wrapper.
- [x] Client-owned provider session registration/list/close daemon APIs sanitize provider bindings from public responses.
- [x] Provider callback responses validate `provider_call_id` and normalize provider errors.
- [x] Client-owned provider session can register one method and receive a sandbox call.
- [x] Sandbox cannot call a method registered by an unrelated client/session.
- [x] Duplicate method registration follows deterministic precedence rules.
- [x] Request-scoped capability is unavailable outside that request.
- [x] Workflow-scoped capability is available to allowed requests in that workflow.
- [x] Instance-scoped capability requires matching `instance_id`.
- [x] Provider timeout returns `host_call_timeout`.
- [x] Provider disconnect returns `host_capability_provider_unavailable`.
- [x] Request cancellation cancels or closes in-flight provider calls.
- [x] Gated method does not execute until external approval is granted.
- [x] Approval denial returns a sandbox-visible host-call error and emits durable audit.
- [x] Event stream includes `host_call`, `host_response`, and `approval` observations with correlation IDs.
- [x] Durable audit contains security-relevant approval/denial records.
- [x] SSH-bound client provider sessions preserve normal hosting auth and SSH binding checks.

## Remaining Decisions

- [x] Confirm provider callback transport for first implementation:
  - prioritize optimized local IPC because SSH is a corner case for `host.call`
  - if current code does not yet support SSH callbacks for `host.call`, leave SSH as a later relay path
  - if SSH support exists in a path being touched, keep the least invasive relay behavior
- [x] Confirm provider session lifetime defaults:
  - provider session controls registered method descriptors, owner, scope visibility, private callback binding, expiry, and disconnect cleanup
  - explicit `close()` removes the provider session from daemon registry and future discovery/call resolution
  - close does not imply worker shutdown, artifact cleanup, or guaranteed provider-side cancellation unless separately wired
- [x] Confirm duplicate method precedence:
  - remove daemon-owned built-in precedence from the target model
  - hosting client library may register known broker-supported methods by default
  - duplicate fully-qualified method names fail by default unless override is explicitly requested
  - method identity is fully qualified; same local suffix in different namespaces is not a duplicate
- [x] Confirm approval cache defaults:
  - no implicit broker approval cache
  - approval reuse must be explicit, scoped, and toolbox-style, for example by adding a grant to request/workflow/instance/actor context
  - one-time approval/rejection remains valid only for the current call
- [x] Confirm hierarchical modeling:
  - namespace hierarchy is canonical
  - group/path metadata is optional presentation metadata or derived from namespace
  - search should support namespace-only queries and name-pattern search across namespaces
- [x] Confirm HostedToolbox brokered IO unification timing:
  - later uber-plan pillar, not part of the current Host API implementation slice

## Follow-Up Work From Decisions

- [x] Remove built-in precedence from broker/provider resolution.
- [x] Move registration of known broker-supported method descriptors to hosting client library helpers, enabled by default.
- [x] Allow clients to omit known methods or provide custom implementations for those fully-qualified method names.
- [x] Add duplicate fully-qualified method registration rejection with an explicit override option.
- [x] Preserve fast local IPC as the primary provider callback transport target in the client-facing registration helper; keep SSH relay as a corner-case/later path unless already supported by the touched code.
- [x] Make namespace hierarchy the canonical capability hierarchy and derive presentation groups from namespace where possible.
- [x] Keep approval reuse explicit through scoped grants rather than an implicit broker cache.
- [x] Request dependent-client adoption after this follow-up breaking-change slice is implemented, documented, and committed.
- [ ] After dependent-client adoption, remove the remaining service-owned `fs.*` / `http.fetch` fallback registration from workflow node dispatch.
