# Hosted Sandbox Contract Feasibility Study

Date: 2026-06-20

Purpose: evaluate a broader redesign of hosted sandbox contracts around first-class streaming, client-owned host APIs, toolbox-backed capabilities, long-lived node instances, sandbox state, and workflow/card action discovery.

This replaces the previous implementation checklist. The prior work left the Python and JavaScript node runtimes functional, but the next step is architectural: decide which contracts should become stable public surfaces before adding more modes and integrations.

## Executive Summary

The proposed redesign is feasible, but it should not be attempted as one large change. The current implementation has useful building blocks:

- node child runtimes with dedicated control channels
- bounded stream sessions and request lifecycle accounting
- host API frames with `host_call_id` correlation
- host-side `HostApiRegistry`
- toolbox callback relay and gated approval machinery
- artifact prepare/collect/cleanup helpers
- warm worker reuse for compatible module/snippet requests

The largest gap is contract ownership. Today the daemon/service owns too much of the node host API and streaming behavior, while dependent clients need to own host-provided capabilities, approvals, user-facing actions, and workflow-local state. The recommended direction is to split the redesign into five explicit contracts:

1. Sandbox Control Protocol
2. Sandbox Event Stream Protocol
3. Host Capability Protocol
4. Sandbox Instance And State Protocol
5. Workflow Action Manifest Protocol

The host capability protocol should reuse toolbox concepts, but it should be a sandbox-to-host capability toolbox rather than a normal hosted toolbox worker. It needs toolbox-style discovery, schemas, permissions, scopes, and approval gates, while allowing the callable implementation to live in the hosting client process.

## Programming Model After The Proposed Changes

The target programming model should be simple for sandbox authors and explicit for hosting clients. Sandboxed code calls a small stable harness API. Hosting clients register capabilities, subscribe to events, handle approvals, and optionally manage long-lived instances or state. The daemon/service brokers those relationships, but it is not the owner of client extension APIs.

### Sandbox Author Model

Minimal nodes continue to look like a normal callable:

```python
def run(payload):
    customer = host.call("crm.customer.lookup", {"id": payload["customer_id"]})
    progress(50, "customer loaded")
    return {"customer": customer}
```

For JavaScript nodes:

```javascript
export async function run(payload, api) {
  const customer = await api.callAsync("crm.customer.lookup", {id: payload.customer_id});
  api.progress({pct: 50, message: "customer loaded"});
  return {customer};
}
```

The sandbox does not know whether `crm.customer.lookup` is a built-in, a toolbox-backed provider, a GUI-hosted callback, or a remote client-owned capability. It receives only a scoped discovery document and calls by method name. Capability routing, permission checks, approvals, and provider transport remain outside the sandbox.

Sandboxed code can discover the current contract:

```python
info = sandbox.describe()
methods = info["host_capabilities"]["methods"]
events = info["events"]
```

`sandbox.describe()` separates:

- harness APIs, globals, and execution modes
- worker-live events versus host-generated events
- host-callable capabilities currently granted to this request, workflow, or instance
- available state scopes
- available action entries
- roots and artifact policy

`host.describe()` can become a narrower host-capability view. New sandbox code should prefer `sandbox.describe()` when it needs to reason about the whole environment.

### Host Capability Provider Model

Hosting clients provide extension APIs by registering a scoped capability session. The exact client helper API can hide transport details, but conceptually it looks like:

```python
session = hosting.host_capabilities.register(
    scope={"workflow_id": workflow_id},
    methods=[
        {
            "name": "crm.customer.lookup",
            "group_path": ["CRM", "Customer"],
            "args_schema": {"type": "object"},
            "result_schema": {"type": "object"},
            "permissions": ["crm.customer.read"],
            "approval": {"mode": "none"},
        }
    ],
)

@session.method("crm.customer.lookup")
async def lookup_customer(args, context):
    return await crm.get_customer(args["id"])
```

The hosting library owns the callback binding and talks to the daemon/service. The provider implementation lives in the hosting client process or a client-owned helper process. The sandbox receives neither binding addresses nor provider tokens.

The service broker resolves each `host.call(...)` by:

1. matching the method against active provider sessions;
2. checking request/workflow/instance/consumer scope;
3. checking permissions and approved scopes;
4. requesting user approval when required;
5. invoking the provider callback;
6. returning a normalized `host_response` to the worker.

Known methods such as `fs.*`, `http.fetch`, and future state methods should be represented as explicit provider sessions behind the same broker. The hosting service should provide descriptor helpers and broker primitives, not implicit method ownership.

### Approval Model

Approvals are user-facing host decisions, not sandbox decisions. A gated call should look synchronous to sandbox code:

```python
result = host.call("billing.invoice.approve", {"invoice_id": "inv-1"})
```

If approval is required, the broker emits an approval request toward the owning hosting client or UI. The provider is invoked only after approval. Denial returns a normal sandbox-visible host-call error, and the decision is recorded in durable audit because live streams may be lossy.

### Event And Stream Model

Sandbox authors emit semantic events through harness helpers:

```python
progress(25, "loading inputs")
log.info("loaded customer", customer_id=payload["customer_id"])
```

The service emits lifecycle and observation events around the request:

- `started`, `heartbeat`, `done`, `canceled`, `error`
- `stdout`, `stderr`, `log`, `artifact`, `result`
- `host_call`, `host_response`, `approval`

Hosting clients consume these through helper APIs rather than parsing raw frames:

```python
async for event in hosting.events.subscribe(request_id):
    if event.kind == "progress":
        update_ui(event.payload)
    elif event.kind == "stream_loss":
        mark_output_partial(event.lane, event.dropped)
```

Helpers should expose simple loss behavior: either fail the stream on loss or surface a `stream_loss` event and continue with available data. Large ack-backed streams use accept/ack/close flow control; observability streams may drop according to lane policy.

### State Model

State is explicit host-managed data, not an implicit snapshot of Python or JavaScript memory. Sandboxes use granted host capabilities:

```python
profile = host.call("state.workflow.get", {"key": "customer_profile"})
host.call("state.workflow.set", {"key": "customer_profile", "value": profile})
```

The same shape can support backend-global, workflow-local, instance-local, and request-local partitions. Access is granted by capability scope and policy, not by trusting arbitrary keys from sandbox code. Long-lived workers may keep process-local caches, but restart recovery should rebuild from explicit host-managed state.

### Long-Lived Instance Model

The default remains ephemeral request execution. Long-lived execution becomes explicit:

```python
instance = hosting.instances.create(runtime="python", scope={"workflow_id": workflow_id})
await instance.call("run", payload)
await instance.call("refresh_cache", {"force": True})
await instance.close()
```

Requests can route to a live instance only when the client asks for that instance and policy allows it. Compatibility with a warm worker is not enough; the public route is `instance_id` plus matching runtime, package, scope, state policy, and capability grants.

Project mode is long-lived only when the runtime declares its isolation policy. Python project instances reset cwd, `sys.path`, environment, and project imports between calls. JavaScript project instances reuse the host worker process but create a fresh QuickJS context and project module cache per call.

### Action/Card Model

Simple nodes only need `run(payload)`. Richer nodes can expose an optional action manifest:

```json
{
  "default_action": "run",
  "actions": [
    {
      "name": "approve_invoice",
      "entrypoint": "approve_invoice",
      "args_schema": {},
      "result_schema": {},
      "permissions": ["invoice.approve"],
      "ui": {"button": true}
    }
  ]
}
```

Card designers discover available actions from the manifest and bind UI actions to sandbox entry points. The action invocation still goes through the same request/instance routing, capability, state, approval, and event contracts. This keeps `run(payload)` as the baseline while allowing composable workflow libraries to expose richer entry points.

### Operational Boundaries

This model deliberately keeps responsibilities separated:

- sandbox code owns business logic and calls stable harness APIs
- hosting clients own extension capability implementations and user-facing approval UX
- the daemon/service owns brokering, policy enforcement, event observation, lifecycle, and cleanup
- toolbox concepts provide descriptors, scopes, groups, and approvals, but node host calls do not inherit toolbox bundle staging or repair lifecycle unless a later unification explicitly chooses that

That separation is what lets streaming, host APIs, state, long-lived instances, and card actions evolve without forcing sandbox authors to learn daemon transport details.

## Current Gaps

### Streaming

Current Python node live streaming is limited. Python code can call `progress(...)` / `emit_progress(...)`, which sends a live `progress` frame. `stdout`, `stderr`, `log`, `artifact`, `result`, `error`, and `done` stream records are mostly synthesized by the service around terminal execution results.

Gap: clients see a stream API with many event types, but the worker contract does not expose first-class live event emission for most of them.

Direction: formalize event kinds as a first-class protocol. Worker harnesses should advertise which event kinds they can emit directly, and host layers should identify which events are host-generated.

### Host API Ownership

Current Python node `host.call(...)` routes through the hosting service broker. The service owns the dispatch machinery, but `fs.*`, `http.fetch`, custom methods, toolbox-backed methods, and state methods are provider-session owned capabilities. Dependent client processes register capabilities through Host Capability sessions and callback bindings.

Gap: dependent clients cannot naturally provide host API methods from their own process. Cross-process callbacks would require an RPC relay similar to the toolbox callback binding.

Direction: make host API methods client-owned capabilities. The daemon/service can broker and enforce capability sessions, but callable implementations should be able to live in the client process.

### Harness Discovery

Current `host.describe()` describes host API methods. It does not clearly separate:

- harness-provided runtime APIs
- host-provided extension APIs
- host-generated stream events
- worker-generated stream events
- supported execution modes
- action entry points

Gap: sandboxed code and host clients cannot reliably discover the boundary between the harness contract and extension capabilities.

Direction: add an explicit `sandbox.describe` or equivalent capability document that separates harness features from host extensions.

### Toolbox Reuse

Toolbox already has many of the right concepts: discoverable methods, metadata, scopes, gated approval, callback relay, and hosted execution accounting. However, normal toolbox execution is host-to-worker, while node `host.call(...)` is sandbox-to-host.

Gap: there are two similar but separate capability systems: toolbox callbacks/brokered IO and node host API.

Direction: define a "host capability toolbox" model that reuses toolbox metadata, schemas, permissions, scopes, and approval flow, but reverses the call direction.

### Long-Lived Instances

Current Python node workers can be reused sequentially for compatible module/snippet requests. Python and JavaScript node module/snippet requests now also support explicit pinned instances with create/execute/list/close APIs. Python project instances require an explicit isolation policy for cwd, `sys.path`, environment variables, and import caches. JavaScript project instances reuse the worker process with a fresh QuickJS project context per request.

Gap: clients cannot preserve arbitrary process memory through restart. Explicit host-managed state now has snapshot/restore helpers for recovery workflows.

Direction: introduce explicit sandbox instance IDs, routing policy, and state snapshot/restore hooks.

### Sandbox State

Current node responses support `state_patch`. Workflow Python node requests now also have an explicit, opt-in host-managed state provider with versioned JSON values and `state.<scope>.(get|set|list|delete)` Host Capability methods.

Gap: arbitrary Python/JS heap recovery is intentionally not supported. Host-managed JSON state has snapshot/restore hooks, and failed-request artifact handoff exists, but automatic garbage collection for retained recovery folders is still deferred.

Direction: build instance snapshot/restore and routable instance lifecycle on top of the explicit state capability model. Do not snapshot arbitrary Python memory.

### Workflow/Card Actions

Current node execution is centered on a simple callable such as `run(payload)`. Card designers cannot discover a structured action manifest from a sandbox and bind buttons to supported entry points.

Gap: workflow composition needs action discovery without abandoning simple `run(payload)` for small nodes.

Direction: support an optional action manifest. `run(payload)` remains the minimal default, while richer sandboxes can expose toolbox-like action entries.

## Feasibility By Direction

### 1. First-Class Streaming For Stdout, Stderr, Logs, And Events

Feasibility: High.

Effort: Medium to High.

Current state:

- `progress` is live from worker to host.
- Python stdout/stderr are captured and emitted after completion.
- Host emits service-level `log`, `artifact`, `result`, `error`, `canceled`, and `done` events.
- Shared stream retention already supports bounded queues and drop counts.

Needed changes:

- Add explicit worker event APIs, for example `emit_event(kind, payload)`, `log(payload)`, `metric(payload)`.
- Add live stdout/stderr forwarding with emitter-controlled chunks. Host policy should set maximum chunk size, but emitters should choose natural boundaries such as lines, log records, or semantic output records. Streams with known total size should announce that size up front.
- Keep host-generated events distinguishable from worker-generated events.
- Add lane-derived retention policy: control frames are non-droppable, ack-backed output/request streams use backpressure, non-ack observability output is bounded/droppable, progress and heartbeat are coalescible, and audit frames are bounded live with durable audit where security-sensitive.
- Add a daemon event subscription path so high-volume event reads do not block cancel/status/control commands on the same client control channel.
- Extend contract discovery to advertise supported event kinds and whether they are live, terminal, host-generated, or worker-generated.

Risks:

- Live stdout/stderr can be noisy and high-volume.
- Event ordering needs a clear rule when worker events, host heartbeats, host calls, and terminal events interleave.
- Backpressure behavior must be explicit so long-running noisy workers cannot exhaust memory.
- A verbose per-event object shape repeats request id, timestamp, sequence, type, and payload wrapper on every frame; that becomes unnecessary overhead for high-volume output.

Recommended direction:

Adopt a batched event stream shape. Compactness should come from shared batch context and kind-specific optional fields, not from unreadable field names or rigid positional payload arrays. Hosting client libraries should expose typed helper events so normal clients do not parse this wire shape directly:

```json
{
  "version": 1,
  "context": {
    "stream_id": "stream-id",
    "request_id": "request-id",
    "instance_id": "instance-id"
  },
  "base": {
    "sequence": 100,
    "timestamp_ms": 1781913600000
  },
  "loss": {
    "output": 0,
    "event": 0,
    "audit": 0
  },
  "frames": [
    {"dt_ms": 0, "kind": "progress", "pct": 40, "message": "installing"},
    {"dt_ms": 8, "kind": "stdout", "text": "Installing package\n", "boundary": true},
    {"dt_ms": 11, "kind": "done", "status": "ok"}
  ],
  "more": true
}
```

Reserve `result`, `error`, `canceled`, and `done` as terminal lifecycle events. Derive lane and retention policy from event kind instead of repeating it on every frame. Use simple loss handling for clients: helpers either raise on loss or yield a `stream_loss` marker and continue. For ack-backed request/output streams, helper-managed accept/ack/close signals should provide backpressure instead of dropping chunks. The contract should define a minimum client acceptance window, not a maximum total stream size, because artifacts/results may be large. Keep the streaming pillar focused on events: it may reserve `instance_id`, correlation metadata, `state_notice`, and `action_notice`, but state stores, host capability ownership, approval routing, and action manifests remain separate pillars.

Sequencing recommendation: implement the event stream protocol first. It is the least coupled foundation slice because every later pillar needs stable observation, terminal semantics, loss reporting, and instance/request identity. Do not implement the other pillars inside this slice; only reserve the stream semantics they will use.

### 2. Host API Owned By Hosting Clients

Feasibility: High, but requires a new public contract.

Effort: High.

Current state:

- Python node host API calls are handled by the daemon/service process.
- Known methods such as `fs.*` and `http.fetch` are no longer service-owned workflow-node capabilities; clients register them explicitly when they want those names available.
- Dependent clients can register provider sessions, and the hosting client library exposes helpers for known broker-supported method descriptors.
- Toolbox callback relay already proves callback-to-client RPC is possible.

Needed changes:

- Introduce host capability sessions registered by clients.
- Include callable endpoint binding, auth token, lifetime, close semantics, and capability descriptors.
- Route sandbox `host.call(...)` to client-owned capability endpoints, including hosting library registered default methods such as filesystem and HTTP helpers.
- Define cancellation, timeout, backpressure, and retry behavior for in-flight host calls.
- Keep the daemon as registry/policy/audit owner while optimizing local IPC as the primary provider callback transport. SSH relay remains a corner-case path.

Risks:

- Cross-process host calls need robust auth and ownership checks.
- Client process death must produce clean sandbox-visible failures.
- Host call routing must not let one sandbox call another client's capabilities without explicit scope grant.

Recommended direction:

Make client-owned host APIs use a callback-binding model similar to toolbox, but with method descriptors and scope references:

```json
{
  "contract": "hosting.sandbox.host_capabilities.v1",
  "session_id": "cap-session",
  "owner": "client-or-actor",
  "methods": [
    {
      "name": "crm.lookup_customer",
      "group": "crm",
      "args_schema": {},
      "result_schema": {},
      "permissions": ["crm.read"],
      "approval": "none|gated|required"
    }
  ],
  "binding": {
    "family": "AF_PIPE|AF_UNIX|daemon_channel",
    "address": "...",
    "session_token": "..."
  }
}
```

Default hosting clients should be able to register known broker-supported methods automatically, but may omit or replace them. Duplicate fully-qualified method names fail by default unless override is requested. Namespace hierarchy is canonical; presentation grouping can be derived from namespace.

### 3. Harness Versus Host Extension Discovery

Feasibility: High.

Effort: Medium.

Current state:

- `host.describe()` describes host API methods and transport features.
- Worker harness capabilities are implicit in docs and code.

Needed changes:

- Add a stable discovery response that separates harness features from extension features.
- Include execution modes, globals, stream event capabilities, state scopes, action entries, and host capabilities.

Recommended direction:

Add `sandbox.describe` or extend `host.describe` with top-level sections:

```json
{
  "harness": {
    "language": "python",
    "execution_modes": ["module", "snippet", "project"],
    "globals": ["payload", "progress", "emit_progress", "host", "artifact_inputs", "artifact_outputs"],
    "event_capabilities": {
      "worker_live": ["progress"],
      "host_generated": ["started", "heartbeat", "stdout", "stderr", "log", "artifact", "result", "error", "canceled", "done"]
    }
  },
  "host_extensions": {
    "capability_sets": []
  },
  "state": {
    "scopes": []
  },
  "actions": {
    "entries": []
  }
}
```

### 4. Toolbox As The Host API Model

Feasibility: High.

Effort: High.

Current state:

- `HostApiRegistry` mirrors a small part of toolbox-like discovery.
- Toolbox has richer native and hosted machinery, including scopes, gating, callback relay, and metadata.

Needed changes:

- Extract toolbox method metadata into a direction-neutral capability model.
- Define host capability toolbox refs, for example `host_toolbox://client/session/name`.
- Let sandbox code discover methods and call them via `host.call(...)`.
- Let host clients implement methods in their own process.

Risks:

- Normal toolbox and host capability toolbox have opposite call directions.
- Reusing too much hosted toolbox orchestration could import bundle staging and repair complexity where it is not needed.

Recommended direction:

Create a minimal shared capability core:

- method identity
- hierarchical group
- display metadata
- args/result schemas
- permissions
- scope requirements
- approval policy

Then adapt both normal toolbox and host capability toolbox to that core.

### 5. Permissions, Approved Scopes, And Gated User Approval

Feasibility: High.

Effort: High.

Current state:

- Toolbox already supports visible/hidden/gated states and callback processors for approval-like flows.
- Node host API has basic policy gates for Host Capability namespaces.

Needed changes:

- Define approval requests as host-directed events, not sandbox-directed events.
- Add approval result caching by scope, actor, method, and arguments if appropriate.
- Ensure sandbox cannot approve its own capability escalation.
- Add explicit denial/error semantics visible to sandbox code.

Recommended direction:

Use a policy decision pipeline:

1. Method registered with required permissions and approval policy.
2. Sandbox call arrives with instance/request identity.
3. Host broker checks static policy and scope grants.
4. If gated, host emits approval request to the owning client/user-facing process.
5. Decision is recorded with scope and TTL.
6. Handler runs only after approval.

### 6. Rework Brokered IO Around `host.call()`

Feasibility: Medium to High.

Effort: Medium.

Current state:

- Node filesystem and HTTP helpers are convenience wrappers over Host Capability calls; the implementation must come from registered provider sessions.
- The hosting client library exposes known-method registration helpers for filesystem and HTTP descriptors.
- Toolbox brokered IO has separate callback/broker code paths.

Needed changes:

- Represent fs/http as default host capability methods registered by the hosting client library.
- Align node and toolbox brokered IO schemas.
- Preserve existing toolbox callback context attribution.
- Preserve sandbox policy enforcement.

Risks:

- Existing toolbox consumers may rely on current callback shapes.
- Filesystem and HTTP policies have subtle differences between node artifact roots and toolbox sandbox roots.

Recommended direction:

Unify method metadata and dispatch, not necessarily all storage/path semantics at once. Start with shared schema and policy description; migrate transport internals after behavior parity tests.

### 7. Hierarchical Tool Groups

Feasibility: High.

Effort: Medium.

Current state:

- Methods are mostly flat names such as `fs.read_text` or toolbox tool names.
- Permissions and UI navigation would benefit from structured groups.

Needed changes:

- Add `group`, `path`, or `namespace_tree` metadata to toolbox and host capabilities.
- Support group-level permissions/scopes.
- Preserve flat method names for execution compatibility.

Recommended direction:

Use flat executable method IDs plus hierarchical metadata:

```json
{
  "name": "crm.customer.lookup",
  "group_path": ["CRM", "Customer"],
  "scope_path": ["crm", "customer", "read"]
}
```

### 8. Long-Lived Routable Node Instances

Feasibility: High.

Effort: High.

Current state:

- Warm workers are internal idle runtime instances.
- Clients cannot route to a specific worker instance.
- Project mode is one-shot.

Needed changes:

- Add public `instance_id`.
- Add create/list/status/route/close/restart APIs.
- Separate compatibility routing from explicit instance routing.
- Add per-instance capability sessions and state scopes.
- Add cooperative shutdown and mutation boundaries.

Risks:

- Long-lived instances increase leak and stale-state risk.
- Project mode requires import/cache/cwd/env reset or intentional persistence semantics.

Recommended direction:

Support two modes:

- `ephemeral`: current request-scoped behavior.
- `instance`: explicit long-lived actor with stable instance ID and declared persistence model.

### 9. State Recovery For Mutated Long-Lived Instances

Feasibility: Medium.

Effort: High.

Current state:

- Node response can include `state_patch`.
- There is no host-managed instance snapshot/restore protocol.

Needed changes:

- Define snapshotable state API.
- Decide whether state is opaque JSON, files, or both.
- Add versioning and conflict rules.
- Define when snapshots happen: explicit, on terminal result, on interval, before shutdown.
- Define restore hooks for restarted instances.

Risks:

- Python process memory is not safely serializable in general.
- File-backed state and JSON state have different lifetimes and security properties.

Recommended direction:

Do not try to snapshot arbitrary Python memory. Provide explicit state APIs:

- `state.get(scope, key)`
- `state.set(scope, key, value, version=None)`
- `state.patch(scope, patch, version=None)`
- `state.list(scope, prefix=None)`

For process-local caches, let sandbox code rebuild from host-managed state on startup.

### 10. Backend Global State And Workflow Local State

Feasibility: High.

Effort: Medium to High.

Current state:

- No first-class scoped state store exists for node sandboxes.
- Artifact refs are available, but they are not a general state API.

Needed changes:

- Define state scopes:
  - backend global
  - workflow local
  - instance local
  - request local
- Define partition keys and authorization.
- Add read/write policy per sandbox.
- Add audit trail and versioning.

Recommended direction:

Expose scoped state as a host capability group:

```text
state.backend.get
state.backend.set
state.workflow.get
state.workflow.set
state.instance.get
state.instance.set
```

Access should be granted by capability scope, not by raw key strings supplied by sandbox code.

### 11. Card Actions And Workflow Composition

Feasibility: Medium to High.

Effort: High.

Current state:

- Python node defaults to `run(payload)`.
- Project mode can select an entrypoint/callable, but there is no discoverable action manifest for UI cards.

Needed changes:

- Define action manifest schema.
- Let sandboxes advertise actions with labels, schemas, permissions, and entrypoints.
- Let card designers bind buttons to actions.
- Route action invocation to a sandbox instance or ephemeral request.
- Preserve simple `run(payload)` for minimal nodes.

Risks:

- A toolbox-like action model can become too heavy for simple workflow nodes.
- UI/card action discovery introduces versioning and compatibility concerns.

Recommended direction:

Use a layered contract:

- simple node: `run(payload)`
- action node: optional `actions.describe()` manifest
- toolbox-like node: full capability/action manifest with grouped entries and permissions

Example action manifest:

```json
{
  "default_action": "run",
  "actions": [
    {
      "name": "approve_invoice",
      "label": "Approve Invoice",
      "entrypoint": "approve_invoice",
      "args_schema": {},
      "result_schema": {},
      "permissions": ["invoice.approve"],
      "ui": {"button": true, "style": "primary"}
    }
  ]
}
```

## Proposed Target Architecture

### Contract 1: Sandbox Control Protocol

Responsible for:

- start request
- cancel request
- shutdown instance
- restart instance
- route to instance
- report status
- report resources

This remains host/runtime owned.

### Contract 2: Sandbox Event Stream Protocol

Status: completed for workflow Python/JavaScript node streams. Deferred legacy cleanup from the completed pillar is also complete; `proxy-stream-recv` remains a separate low-level generic worker/proxy primitive by design.

Responsible for:

- worker-generated events
- host-generated lifecycle events
- event ordering
- backpressure and dropped counts
- terminal event rules
- compact batched frame format
- emitter-controlled output chunking
- event subscription separate from command control

This should be stable across Python, JavaScript, and future runtimes.

### Contract 3: Host Capability Protocol

Responsible for:

- discover host-provided methods
- call methods from sandbox to host
- correlate responses
- enforce scopes and approvals
- route calls to provider-session callback endpoints

This should be toolbox-inspired and direction-neutral.

### Contract 4: Sandbox Instance And State Protocol

Responsible for:

- long-lived instance identity
- mutable instance state
- restart recovery
- backend/workflow/instance/request state scopes
- state authorization and versioning

This should be optional for ephemeral nodes.

### Contract 5: Workflow Action Manifest Protocol

Responsible for:

- discover callable actions
- expose card button/action metadata
- bind UI actions to sandbox entrypoints
- support subworkflow/library composition

This should not replace `run(payload)` for simple nodes.

## Suggested Implementation Phases

### Phase 0: Freeze Current Behavior In Docs

Goal: clarify what is current behavior versus target behavior.

Work:

- Update worker docs to distinguish worker-live events from service-generated stream records.
- Document `HostApiRegistry` as the broker-facing describe/dispatch root, not as owner of known filesystem/HTTP methods.
- Document that custom client-owned host APIs are public through Host Capability sessions.

### Phase 1: Event Protocol Cleanup

Goal: make streaming semantics first-class before adding more stateful behavior.

Work:

- [x] Add event frame schema with shared batch context, per-frame `kind`, sequence/timestamp expansion, and lane-aware loss reporting.
- [x] Add event kind registry and helper-side loss normalization.
- [x] Add live stdout/stderr/log chunk framing and progress frame conversion.
- [x] Add daemon event subscription commands separate from command RPC.
- [x] Add ack-backed output stream accept/ack/close control.
- [x] Add coverage for batch decoding, loss policy, output chunk metadata, ack flow, terminal delivery, and event subscription/control separation.

Deferred legacy cleanup from the completed streaming pillar:

- [x] Remove public `workflow-python-stream-recv` and `workflow-js-stream-recv` command paths once `workflow-*-event-subscribe` is the only documented event consumption API.
- [x] Replace `workflow_python_event_subscribe(...)` and `workflow_js_event_subscribe(...)` service implementations that currently delegate to `stream_recv(...)` with the final subscription/session implementation.
- [x] Remove `workflow-*-stream-recv` wrappers from `engine_host_channel.py`, CLI command tables, interactive CLI flows, daemon local IPC dispatch, auth allowlists, and policy allowlists.
- [x] Decide whether `proxy-stream-recv` remains a generic worker IPC primitive or is renamed/replaced by the same event subscription API; decision: keep it as a low-level generic worker/proxy primitive for now, not as a workflow compatibility route.
- [x] Replace remaining internal `HostedStreamEvent(type=..., payload=...)` construction sites with direct frame/batch builders, then remove compatibility-oriented `HostedStreamEvent.to_dict()` use from stream paths.
- [x] Update `PY_NODE_WORKER.md`, `JS_NODE_WORKER.md`, `GENERIC_WORKER.md`, `HOSTING.md`, and `ENGINE_HOST_CLI.md` so `event-subscribe` and helper-normalized events are the primary documented model.
- [x] Remove docs that describe old retained-event fields such as `dropped_event_count`, `retained_event_count`, and `next_sequence` as the public stream contract; replace them with batch `loss` and helper `stream_loss` behavior.
- [x] Remove tests that assert the legacy one-event/recv response shape after equivalent helper/batch tests are in place.
- [x] Audit terminal output summaries versus live output frames and keep only intentional post-run summary fields, avoiding duplicate compatibility copies of stdout/stderr/log data.
- [x] Remove any migration-only fallback that accepts unknown event kinds, legacy `type`/`payload` event rows, or old recv-only client shapes after all in-repo clients use helper APIs.

### Phase 2: Host Capability Toolbox Core

Goal: unify node host API and toolbox capability metadata.

Work:

- [x] Extract shared capability method model.
- [x] Add hierarchical group metadata.
- [x] Add capability descriptors to `host.describe` / `sandbox.describe`.
- [x] Add callable-surface adapters between toolbox/`ToolsView` metadata and Host Capability descriptors.
- [x] Add Host Capability descriptor-to-callable-schema helpers for sandbox/model-facing discovery.
- [x] Add direct toolbox-to-callable-schema adapter export that preserves hierarchy, identity, visibility, approval metadata, constraints, and digests.
- [x] Add stable callable-surface identity and schema/method/policy digests for approval reuse and merged-view conflict decisions.
- [x] Make merged callable-schema conversion reject duplicate advertised names by default unless the caller explicitly chooses a conflict policy.
- [x] Add explicit bridge-policy helper that intersects toolbox policy, Host Capability caller policy, and bridge policy.
- [x] Keep descriptor helpers for `fs.*` and `http.fetch` so clients can register those methods explicitly.
- [x] Remove implicit service-owned `fs.*` / `http.fetch` fallback from workflow node dispatch.
- [x] Remove diagnostic service-owned `fs.*` / `http.fetch` fallback registration, dispatch, audit/log markers, and fallback-only tests.
- [x] Treat legacy fallback policy keys such as `sandbox.host_api.service_owned_fallback_enabled` as ignored workflow-node dispatch settings.

### Phase 3: Client-Owned Host Capabilities

Goal: let dependent client processes provide host APIs.

Work:

- [x] Add host capability session registration.
- [x] Add filtered session list/close helpers and idempotent upsert registration.
- [x] Add method-scoped auth and lifecycle.
- [x] Add service broker path from sandbox `host.call` to client callback.
- [x] Add provider callback envelope helpers that validate `provider_call_id` and normalize success/error/timeout/cancel responses.
- [x] Add approval bridge helpers and filtered Host Capability audit reads.
- [x] Add tests for client provider denial, timeout, unavailable transport, cancellation, duplicate registration, scopes, and audit filtering.
- [x] Complete toolbox-backed provider callback runtime so hosted toolbox sessions can execute as Host Capability providers, not only register callable descriptors.

### Phase 4: Permission And Approval Unification

Goal: reuse toolbox-style gating for host capabilities.

Work:

- [x] Add scope refs and approval policies to capability descriptors.
- [x] Route approvals to user-facing client process.
- [x] Record approval decisions with TTL and audit.
- [x] Preserve sandbox-visible denial errors.
- [x] Make `allow_once` one-call only.
- [x] Make `add_to_scope` create a scoped broker grant that can skip later approvals for matching method, provider, actor, scope requirements, and optional argument constraints.

### Phase 5: State And Long-Lived Instances

Goal: add routable, recoverable node instances.

Work:

- [x] Add explicit instance create/route/close APIs.
  - [x] Add Python node instance create/execute/list/close APIs for module/snippet requests.
  - [x] Add JS node instance create/execute/list/close APIs for module/snippet requests.
  - [x] Reject project-mode node instances until cwd/sys.path/env/import-cache policy is explicit.
- [x] Add state scopes and state host capability methods.
  - [x] Persist host-managed sandbox state in runtime state.
  - [x] Expose `state.workflow.*`, `state.instance.*`, and explicit `state.backend.*` methods when policy grants those scopes.
  - [x] Advertise available state scopes through `sandbox.describe()`.
- [x] Add snapshot/restore hooks for instance-local state.
  - [x] Add `sandbox-state-snapshot` for scoped host-managed state partitions.
  - [x] Add `sandbox-state-restore` with `merge` and `replace` modes.
  - [x] Keep snapshots limited to explicit host-managed state, not arbitrary Python/JS process memory.
- [x] Make Python project mode long-lived only after cwd/sys.path/env/import-cache policy is explicit.
  - [x] Require `python.project_instance_policy` for pinned project instances.
  - [x] Require cwd reset, `sys.path` reset, env reset, and project import-cache clearing.
  - [x] Restore cwd, `sys.path`, env, and project modules after each project request.
- [x] Decide JS project-mode long-lived semantics separately.
  - [x] Support JS project-mode pinned instances as warm worker-process routing with a fresh QuickJS project context per request.
  - [x] Require explicit `javascript.project_instance_policy` / `project.instance_policy` reset semantics before creating a pinned project instance.
  - [x] Add explicit `instance_state_mode="persistent_module"` for pinned Python module instances and pinned JS script/module instances.
  - [x] Require explicit `replace=true` as the code-revision/state reset boundary for persistent module instances.
  - [x] Keep persistent JS project heap/module-cache state out of scope until a future runtime defines checkpoint and cleanup semantics.

### Phase 6: Action Manifest And Card Integration

Goal: support card buttons and workflow composition.

Work:

- [x] Add optional action manifest.
  - [x] Normalize `action_manifest` / `actions` into `hosting.sandbox.action_manifest.v1`.
  - [x] Preserve allowed, advertised, hidden, disabled, gated, schema, approval, permission, metadata, and entrypoint fields.
  - [x] Keep `run(payload)` as the default action when no manifest is supplied.
- [x] Add card-facing action discovery.
  - [x] Add Python and JavaScript service helpers that return advertised actions and can include hidden actions when requested.
- [x] Add action invocation routing.
  - [x] Route selected actions through existing `export_name`, `operation`, snippet, and project callable request fields before worker execution.
  - [x] Add explicit Python and JavaScript action execution helpers on top of normal execute.
  - [x] Expose action describe/execute over daemon/channel/CLI.

## Migration Notes

- Existing `workflow_python(profile=node)` and `workflow_js(profile=node)` should remain as compatibility facades while contracts are generalized.
- `fs.*` and `http.fetch` are no longer implicit workflow-node capabilities. Clients must register those methods through explicit Host Capability sessions before sandbox code calls `api.fs.*`, `host.fs_*`, or `http.fetch`.
- Event helpers should continue to expose `started`, `progress`, `stdout`, `stderr`, `log`, `artifact`, `result`, `error`, `canceled`, and `done` as normalized event kinds.
- Legacy workflow stream receive command shapes have been removed from public workflow command paths; workflow clients should use `workflow-*-event-subscribe`.
- Client-owned host APIs are opt-in through provider sessions and callable-surface helpers.
- Action manifests are additive. Existing `run(payload)` requests remain valid and become the default action when no manifest is supplied.

## Remaining Open And Deferred Items

The implementation phases above are complete except for the explicitly deferred items below. These are not blockers for the current Host Capability, streaming, state, instance, or action-manifest APIs.

### Client Adoption

- [x] Dependent clients adopted explicit callable-session registration, callback relay bindings, and callable-surface helpers for the fallback-removal slice.
  - Current state: the hosting library exposes known-method registration helpers, generic provider-session helpers, toolbox-as-provider registration, approval bridge helpers, callback relay helpers, and filtered audit reads.
  - Completed client work: `fs.*`, `http.fetch`, and custom host APIs are registered explicitly. Legacy service-owned fallback policy keys are ignored.
  - Breaking-change status: consumed by the dependent client and reset for future slices.

### Callable Surface Bridge And Toolbox Brokered IO

- [x] Decide the toolbox lifecycle pillar should not force HostedToolbox brokered IO onto Host Capability dispatch by default.
  - Current state: hosted toolbox sessions can already act as Host Capability providers, but toolbox-specific brokered IO paths still exist.
  - Implemented simplification: toolbox brokered IO remains toolbox-native, but brokered host calls now carry shared callable-surface identity, schema/method/policy digests, safe correlation fields, and effective bridge-policy metadata in callback context.
  - Decision: share callable-surface primitives and approval/audit helper contracts, while keeping HostedToolbox brokered IO toolbox-native unless a specific later maintenance problem requires deeper runtime unification.
  - Conflict rule: one workflow/session may own multiple hosted toolbox instances with overlapping method names. Duplicate advertised names must use namespaces/aliases or an explicit conflict policy; they must not silently collapse.
  - Identity rule: stable callable identity is `provider_kind + provider_id/toolbox_id + session_id + method`.
  - Approval rule: broker grants default to same provider/session scope. Cross-toolbox reuse requires explicit compatible owner/workspace, toolbox definition digest, schema/method/policy digests, method name, provider kind, and constraints.
  - Bridge rule: brokered IO permissions use an explicit bridge policy intersected with toolbox policy and Host Capability caller policy.

### Long-Lived Recovery Beyond Host-Managed State

- [x] Constrain long-lived recovery to explicit host-managed state plus crashed-request artifact handoff.
  - Current state: JSON state partitions are host-managed and recoverable through `sandbox-state-snapshot` / `sandbox-state-restore`.
  - Boundary: arbitrary Python/JS process memory, open handles, async jobs, and imported module internals are not recoverable state.
  - Decision: no heap recovery and no mutation replay. Hosting does not infer semantic validity of mutated state.
  - Implemented artifact handoff: failed workflow requests with declared output artifacts keep their run folder and expose an `artifact_recovery` notice. Clients can call `workflow-artifact-recovery-inspect`, `workflow-artifact-recovery-claim`, and `workflow-artifact-recovery-cleanup`.
  - Edit+continue rule: `instance_id` is the stable logical worker identity. Python replacement starts a fresh process under the same `instance_id`; JS module/snippet edits can reuse the same compatible worker process while updating `code_key`. Persistent module state makes replacement explicit: edited code is rejected during instance execute and accepted through `create(..., replace=true)`.
  - Artifact namespace rule: recovery claim defaults to `@artifacts/instances/<instance_id>/...` when the failed request or claim supplies an instance id.
  - Client responsibility: partial artifact validity is client-owned. Hosting provides hint labels only.
  - Cleanup boundary: successful requests clean prepared run folders after artifact collection. Failed requests with declared output artifacts intentionally keep their run folders so clients can inspect/claim recovery candidates. Those retained run folders can remain on disk until the client calls `workflow-artifact-recovery-cleanup`; automatic disk garbage collection by age/size/quota is deferred to a later task.

### JavaScript Project Instances

- [x] Implement JS project-mode long-lived runtime after project/module loading semantics are defined.
  - Current state: JS pinned module/snippet instances can either use default fresh QuickJS contexts per call or opt into `instance_state_mode="persistent_module"` for a persistent QuickJS context. Project-mode JS instances now reuse a pinned host worker process with a fresh QuickJS project context and fresh project module cache per request.
  - Implemented loading shape: project code is staged from `project.ref`; entrypoints are CommonJS-style files with `exports`, `module.exports`, and relative `require("./...")`. Dot entrypoints map to paths, for example `pkg.runner` to `pkg/runner.js`.
  - Required cleanup policy: pinned project instances must declare `context=new_per_request`, `module_cache=reset`, `globals=reset`, `async_jobs=drain_or_cancel`, and `host_handles=reset`.
  - Boundary: Node built-ins, `node_modules`, ESM imports, and persistent JS project heap/module-cache state remain out of scope.

### Native Toolbox Metadata

- [x] Decide native toolbox metadata should continue using adapters rather than directly adopting Host Capability descriptors.
  - Current state: adapters convert toolbox/`ToolsView` metadata into Host Capability descriptors.
  - Decision: toolbox descriptions and `ToolsView` remain native toolbox metadata. Host Capability descriptors and callable schemas are adapter/export formats.
  - Implemented export behavior: `toolbox_to_callable_schemas(...)` emits namespace-qualified names, `group_path`, visibility/gating state, provider/session identity, schema/method/policy digests, and toolbox metadata.

## Resolved Architecture Decisions

- Streaming/event contracts were implemented first, and workflow event consumption now uses `workflow-*-event-subscribe`.
- Host Capability provider sessions are client-owned; workers do not receive provider bindings or callback credentials.
- Provider callback relay is optimized for local IPC; SSH relay remains a corner case unless a touched transport already supports it.
- Approval reuse is explicit scoped-grant behavior, not an implicit broker cache.
- State recovery is explicit host-managed JSON state plus client-directed artifact handoff from failed request folders.
- Python project-mode pinned instances require a reset policy for cwd, `sys.path`, env, and project imports.
- JS project-mode pinned instances reuse the worker process only when reset-per-request project policy is explicit; persistent JS project heaps remain unsupported.
- Action manifests are static request metadata for now; sandbox-executed dynamic discovery is deferred unless a future workflow composition feature needs it.

## Current Recommendation

Do not start deferred runtime cleanup until there is a concrete owning feature or client migration signal. The client adoption blocker for Host Capability fallback removal is now closed. The toolbox brokered IO simplification is also closed for now: use shared callable-surface/bridge helpers and metadata, not forced runtime unification.

The next high-leverage work is to let the client integrate the callable-surface bridge helpers, adapter-based toolbox callable export, and conflict/approval policy rules.

Persistent JS project heap state should stay behind explicit runtime design work because it affects cleanup, snapshot, and mutation semantics. Broader instance recovery remains intentionally out of scope unless a future runtime adds a specific checkpoint contract.
