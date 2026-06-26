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

- [ ] Add static host-owned method registry for `fs.list`, `fs.read_text`,
  `fs.write_text`, `fs.mkdir`, `fs.stat`, and `http.fetch`.
- [ ] Derive method descriptions, parameter descriptions, schemas, and required
  fields from Python callable docstrings/signatures.
- [ ] Expose registry discovery as contract descriptions reusable by channel,
  daemon, workflow nodes, and toolbox workers.
- [ ] Add tests for descriptor generation, docstring extraction, and stable
  contract shape.
- [ ] Update client breaking-change notes with the new descriptor/discovery
  helper names.

## Phase 2: Node Service-Broker Provider Sessions

- [ ] Add `provider_kind="service_broker"` registration support in daemon Host
  Capability session handling.
- [ ] Add a client/channel helper to register known service-broker methods for a
  request, instance, workflow, or consumer scope.
- [ ] Route service-broker provider calls to daemon local broker implementations
  through the existing Host Capability provider path.
- [ ] Ensure approval, audit, and host-call events are emitted for service-broker
  calls before local broker execution.
- [ ] Make workflow Python and JS node discovery show selected service-broker
  methods and hide unselected ones.
- [ ] Add tests for approved, denied, unsupported, and sandbox-policy-denied
  `fs.*` calls from node workers.
- [ ] Update Python/JS node docs and client migration instructions.

## Phase 3: Toolbox Host API Unification

- [ ] Replace toolbox-specific hardcoded brokered IO dispatch with the shared
  service-broker registry/dispatcher.
- [ ] Make toolbox `context.host.call(...)`, `context.fs.*`, and
  `context.http.*` discover through the same Host Capability description shape
  used by node workers.
- [ ] Add per-IO approval support for toolbox brokered IO, independent of
  tool-level gated execution.
- [ ] Preserve toolbox callable-surface metadata and bridge-policy/audit fields
  through the unified provider path.
- [ ] Add tests proving toolbox filesystem/HTTP calls run through the shared
  registry, approval can deny/allow, and sandbox policy remains enforced.
- [ ] Remove obsolete toolbox brokered IO helper code once tests cover the
  unified path.
- [ ] Update [sandbox/TOOLBOX_WORKER.md](sandbox/TOOLBOX_WORKER.md).

## Phase 4: Typed Alias Cleanup

- [ ] Standardize Python node aliases as `host.fs.read_text(...)`,
  `host.fs.write_text(...)`, `host.fs.list(...)`, `host.fs.stat(...)`,
  `host.fs.mkdir(...)`, and `host.http.fetch(...)`.
- [ ] Keep `host.call(method, args)` as the canonical escape hatch.
- [ ] Align JS node and toolbox naming documentation with the shared method
  names while preserving idiomatic JS casing where appropriate.
- [ ] Remove redundant one-off wrappers after replacement aliases and tests are
  in place.

## Phase 5: Legacy Cleanup And Client Adoption

- [ ] Remove service-owned implicit fallback paths for `fs.*` / `http.fetch`
  that bypass Host Capability descriptors.
- [ ] Remove stale tests that assert toolbox-only brokered IO plumbing.
- [ ] Verify full hosting sandbox and workflow test pass.
- [ ] Ask the client team to adopt the final breaking-change instructions and
  then reset [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md)
  for later work.

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
