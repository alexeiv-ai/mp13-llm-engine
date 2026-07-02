# Host API And Brokered IO Unification Plan

Date: 2026-06-25
Scope: sandbox host-call programming model, daemon-owned brokered IO providers, node/toolbox host API discovery, approval, and client migration.

## Goal

Unify workflow node and toolbox brokered IO behind the same Host Capability callable-surface model:

1. sandbox code discovers host APIs through `host.describe()` / `sandbox.describe()`
2. sandbox code invokes host APIs through `host.call(method, args)` and typed convenience wrappers
3. daemon-owned brokered methods are exposed as normal Host Capability provider sessions
4. client-owned and toolbox-owned providers continue to use callback/provider sessions
5. approval, audit, event, descriptor, schema, and conflict behavior are shared

No legacy compatibility is required beyond clear instructions in
[HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md).

## Design Decisions

1. Add a static development-time daemon local provider registry for host-owned
   brokered methods. Runtime clients may select which registered methods are
   exposed, but they may not define new daemon-local implementations.
2. Use `provider_kind="service_broker"` for daemon-owned brokered IO methods.
3. Keep custom client/backend APIs under `provider_kind="client_session"`.
4. Keep toolbox tool export under `provider_kind="toolbox_session"`.
5. Run approval before both remote provider callbacks and daemon local
   service-broker calls.
6. Treat `host.call("fs.read_text", args)` as the canonical invocation. Typed
   wrappers are convenience aliases over the same method names.
7. Build method descriptors from callable docstrings/signatures in the same
   spirit as native toolbox callable registration in
   [../mp13_engine/mp13_toolbox.py](../mp13_engine/mp13_toolbox.py).
8. Preserve sandbox policy as the hard brokered IO boundary. Approval narrows or
   authorizes a host-call attempt but does not widen filesystem/network policy.

## Phase 1: Service Broker Registry And Discovery

- [x] Add static host-owned method registry for `fs.list`, `fs.read_text`,
  `fs.write_text`, `fs.mkdir`, `fs.stat`, and `http.fetch`.
- [x] Derive method descriptions, parameter descriptions, schemas, and required
  fields from Python callable docstrings/signatures.
- [x] Expose registry discovery as contract descriptions reusable by channel,
  daemon, workflow nodes, and toolbox workers.
- [x] Add tests for descriptor generation, docstring extraction, and stable
  contract shape.
- [x] Update client breaking-change notes with the new descriptor/discovery
  helper names.

## Phase 2: Node Service-Broker Provider Sessions

- [x] Add `provider_kind="service_broker"` registration support in daemon Host
  Capability session handling.
- [x] Add a client/channel helper to register known service-broker methods for a
  request, instance, workflow, or consumer scope.
- [x] Route service-broker provider calls to daemon local broker implementations
  through the existing Host Capability provider path.
- [x] Ensure approval, audit, and host-call events are emitted for service-broker
  calls before local broker execution.
- [x] Make workflow Python and JS node discovery show selected service-broker
  methods and hide unselected ones.
- [x] Add tests for approved, denied, unsupported, and sandbox-policy-denied
  `fs.*` calls from node workers.
- [x] Update Python/JS node docs and client migration instructions.

## Phase 3: Toolbox Host API Unification

- [x] Replace toolbox-specific hardcoded brokered IO dispatch with the shared
  service-broker registry/dispatcher.
- [x] Make toolbox `context.host.call(...)`, `context.fs.*`, and
  `context.http.*` discover through the same Host Capability description shape
  used by node workers.
- [x] Add per-IO approval support for toolbox brokered IO, independent of
  tool-level gated execution.
- [x] Preserve toolbox callable-surface metadata and bridge-policy/audit fields
  through the unified provider path.
- [x] Add tests proving toolbox filesystem/HTTP calls run through the shared
  registry, approval can deny/allow, and sandbox policy remains enforced.
- [x] Remove obsolete toolbox brokered IO helper code once tests cover the
  unified path.
- [x] Update [sandbox/TOOLBOX_WORKER.md](sandbox/TOOLBOX_WORKER.md).

## Phase 4: Typed Alias Cleanup

- [x] Standardize Python node aliases as `host.fs.read_text(...)`,
  `host.fs.write_text(...)`, `host.fs.list(...)`, `host.fs.stat(...)`,
  `host.fs.mkdir(...)`, and `host.http.fetch(...)`.
- [x] Keep `host.call(method, args)` as the canonical escape hatch.
- [x] Align JS node and toolbox naming documentation with the shared method
  names while preserving idiomatic JS casing where appropriate.
- [x] Remove redundant one-off wrappers after replacement aliases and tests are
  in place.

## Phase 5: Legacy Cleanup And Client Adoption

- [x] Remove service-owned implicit fallback paths for `fs.*` / `http.fetch`
  that bypass Host Capability descriptors.
- [x] Remove stale tests that assert toolbox-only brokered IO plumbing.
- [x] Verify full hosting sandbox and workflow test pass.
- [x] Ask the client team to adopt the final breaking-change instructions and
  then reset [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md)
  for later work.

## Phase 6: Approval Request Argument Visibility And Authority Boundaries

Status: implemented for the approval request preview and common brokered
approval helpers. This phase is not justified by the hosted chat demo bug alone,
but the bug exposed a useful design gap to close deliberately.

### Problem To Fix

Approval callbacks need enough sanitized request detail to make policy decisions
before execution. Today the canonical normalized Host Capability approval request
preserves `argument_keys`, provider/method identity, approval policy, context,
and correlation metadata, but not a stable sanitized argument preview. Some
call paths may still carry raw arguments internally, but clients should not need
to depend on transport-specific payload details.

The hosted demo failure was caused by a model-facing `root_path` argument. That
specific problem was fixed by removing `root_path` from the tool schema and
moving root authority to host/client policy plus approval scope. The broader
protocol question remains: when approval code must decide whether
`fs.read_text(root_id, relative_path)` is acceptable, it should receive a
bounded, sanitized argument view through the normalized approval contract.

### Evidence And Limits

Evidence:

- model-facing tools can accidentally expose authority-bearing fields;
- approval decisions for brokered filesystem/HTTP calls often depend on the
  requested target, not only the method name;
- current broker enforcement is correct as the final boundary, but approval UI
  cannot give useful user-facing decisions if the normalized request hides all
  argument values.

Limits:

- the demo bug itself was not strong evidence for broad protocol churn;
- raw full arguments can contain secrets, large payloads, or data that should
  not be displayed to users;
- the broker must remain the hard enforcement point even after approval helpers
  are added.

### Intended Design Direction

- Keep real filesystem roots owned by host/client configuration and sandbox
  policy.
- Keep virtual/narrowed roots owned by approval scope or client workflow state.
- Treat model-provided file paths as requests, never authority.
- Add sanitized argument previews to normalized Host Capability approval
  requests, not raw unbounded argument dumps.
- Add reusable approval helper functions for common service-broker decisions,
  especially filesystem path containment and HTTP URL-prefix checks.
- Keep approval helpers advisory for decision-making; daemon broker policy
  remains authoritative.

### Work Items

- [x] Define the sanitized approval argument preview contract:
  - include small scalar values needed for policy decisions;
  - redact known secret fields;
  - summarize or omit large payloads;
  - preserve existing `argument_keys`.
- [x] Add the preview to `host_capability_approval_request(...)` and all public
  approval callback relays.
- [x] Add service-broker approval helpers for:
  - resolving `root_id + relative_path` against the declared sandbox fs rule;
  - checking containment under configured root and optional scoped virtual root;
  - checking HTTP method and URL prefix against sandbox/network policy.
- [x] Update hosted toolbox, workflow Python, and workflow JS approval tests to
  assert callbacks receive the same normalized preview shape.
- [x] Update client-facing docs with the corrected programming model:
  roots are policy/scope-owned; sandbox/model code supplies only relative
  targets where a tool explicitly allows that.
- [x] When implementation begins, rewrite
  [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md) with
  the new approval-request payload contract and remove any compatibility
  fallback requirements.

### Completed Polish

- [x] Add registry-owned service-broker policy hints and public helper APIs so
  clients do not need to duplicate method-name to approval-check mappings.
  `service_broker_method_policy_hint(method)` exposes the method category and
  policy hints; `host_capability_approval_check_service_broker_request(...)`
  dispatches known filesystem and HTTP brokered methods to the existing preview
  validators.

## Client Programming Model After Completion

Clients choose a provider model per callable method:

1. `service_broker`: daemon-owned brokered IO. Client selects methods and
   optional approval policy; daemon executes local brokered IO after approval.
2. `client_session`: client-owned backend/API methods. Client receives provider
   callbacks and may call daemon broker commands, its own store, or another
   service.
3. `toolbox_session`: hosted toolbox methods exported as Host Capability
   callables.

Sandbox code uses the same call shape regardless of provider:

```python
result = host.call("fs.read_text", {"root_id": "project", "relative_path": "notes.txt"})
text = host.fs.read_text(root_id="project", relative_path="notes.txt")["text"]
```

Provider identity, approval policy, argument/result schemas, permissions,
scope requirements, and method digests come from Host Capability discovery.
Service-broker methods also expose registry-owned policy hints that clients can
use when implementing approval callbacks.
