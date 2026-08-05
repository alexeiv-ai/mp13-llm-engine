# Feature Request: Hosting Operation and Capability Access v2

Status: proposed for the parent hosting team

Consumer: `O:/repos/mp13-docs`

Current pinned parent: `084e559796b3ef01d94bbb749ba51aa215e79f05`

## Summary

Add a versioned parent-owned client contract that unifies durable hosted
operation identity, status, cancellation, result references, Host Capability
session identity, approval callback leases, and authority lifetime semantics.

The current durable toolbox receipt and schema-4.6 anchor work is correct and
should remain the compatibility baseline. This request is an additive v2 API,
not a request to weaken receipt guarantees or move consumer workflow policy
into the parent.

## Motivation

The consuming project now has reliable recovery, but it must maintain several
adapter layers because closely related parent surfaces have different method
names, argument shapes, callback conventions, and lifecycle representations:

- toolbox, workflow Python, and workflow JavaScript use separate execute,
  status, and cancellation APIs;
- toolbox status/cancel requires the caller to reconstruct the original
  `engine_id` or `toolbox_id` selector alongside the request ID;
- workflow execution accepts `approval_requester_binding`, while local consumer
  code commonly begins with a callable and must create/release a relay itself;
- Host Capability registration does not preserve an explicit `provider_id`
  separately from `session_id`;
- `close_on_client_disconnect` combines transport lifetime with authority
  lifetime;
- oversized receipt results expose an omission descriptor rather than a
  dereferenceable artifact reference, and digest/size spelling differs between
  parent and consumer contracts;
- the JSON receipt ledger rewrites the retained checkpoint for every lifecycle
  transition.

These differences cause consumer-side signature inspection, `TypeError`
fallbacks, selector dictionaries, method-name switching, callback-relay
lifetime code, local Host Capability session fallback, and repeated lifecycle
normalization. A stable v2 surface would let consumers delete those adapters.

## Requested capabilities

### 1. One typed hosted-operation reference

Every durable hosted execution family should return a bounded reference with a
stable contract, for example:

```python
HostedOperationRefV2 = {
    "contract": "hosting.operation_ref.v2",
    "operation_id": "...",          # parent-stable opaque identity
    "request_id": "...",            # caller idempotency identity
    "execution_kind": "toolbox|workflow_python|workflow_js",
    "selector": {"kind": "toolbox_id|engine_id", "id": "..."},
    "fingerprint": "sha256:...",
    "receipt_namespace": "...",
}
```

The opaque reference must be accepted directly by generic operations:

```python
channel.hosted_operation_status(ref=operation_ref)
channel.hosted_operation_cancel(ref=operation_ref, reason="workspace_unload")
```

The parent may retain existing family-specific APIs internally and for
compatibility. The v2 facade must normalize them to one status contract.

### 2. One canonical lifecycle/status contract

Return `hosting.operation_status.v2` for execute attachment/replay, status, and
cancel. It must preserve:

- `queued`, `running`, terminal success/failure/cancellation;
- `interrupted_before_dispatch`;
- `interrupted_after_dispatch_unknown`;
- `forgotten` and `unknown_outside_retention`;
- idempotency conflict;
- caller request ID, operation reference, timestamps, and bounded reason;
- terminal result digest and optional result reference.

Digest values should use one canonical `sha256:<hex>` spelling. Size should use
one canonical field. Consumers should not need to translate `size_bytes` to
`size`, inspect nested receipt/envelope variants, or infer lifecycle from
`status`, `outcome`, and `reason` in different orders.

### 3. Real artifact-backed terminal references

When a terminal envelope exceeds the safe inline limit, store the bounded
result in the existing parent artifact store when policy permits and return a
dereferenceable, authorization-checked reference. If policy does not permit
retention, return a distinct digest-only omission record rather than labeling
it as a usable result reference.

No raw credentials, callback bindings, unbounded arguments, or worker memory
may enter receipts or artifacts.

### 4. Transactional receipt storage

Replace the full-checkpoint JSON persistence implementation with a small
transactional indexed repository, preferably SQLite in WAL mode. Required
properties:

- atomic prepare/dispatch-claimed/terminal transitions;
- indexed `(namespace, request_id)` and opaque operation-reference lookup;
- concurrent readers and short write transactions;
- deterministic age/count pruning and tombstone retention;
- schema versioning and fail-closed corruption behavior;
- no worker or sandbox startup during ledger load;
- migration/import from the existing v1 JSON ledger.

This is a storage implementation change behind the public receipt contract. It
does not replace consumer-side journals that track local application targets.

### 5. Stable Host Capability registration identity

Add explicit `provider_id` to Host Capability session registration and preserve
it in the public provider descriptor independently of `session_id`. The parent
should reject contradictory duplicate identities rather than silently treating
the session ID as the provider ID.

All registration, list, close, filtered-close, audit, toolbox-provider, and
service-broker helpers should use the same identity model.

### 6. Parent-owned approval callback lease

Every workflow Python/JavaScript execute, action-describe, action-execute,
pinned-instance execute, and stream-open method should accept either:

- `approval_requester=<callable>` for an in-process client; or
- `approval_requester_binding=<binding>` for an already bound transport.

When given a callable, the parent client library should create the relay,
forward the binding, keep it alive for the complete request/stream lifetime,
and release it exactly once. A context-manager form is acceptable:

```python
with channel.approval_callback_lease(callback, scope=scope) as lease:
    channel.execute_workflow_python(..., approval_callback_lease=lease)
```

Duplicate delivery within one parent request must retain stable approval and
provider-call identity so a consumer can apply an allow-once decision
idempotently without granting a second request.

The parent transports approval requests and decisions. The consumer continues
to own approval UI, policy, persistence scope, and user-wait behavior.

### 7. Explicit capability-session lease semantics

Replace the ambiguous `close_on_client_disconnect` boolean in v2 with an
explicit lifetime descriptor, for example:

```python
{
    "owner_authority_id": "opaque-consumer-authority",
    "expires_at_ms": 0,
    "on_transport_loss": "retain_until_expiry|close",
    "on_authority_revoked": "close",
    "on_request_terminal": "close|retain",
}
```

Transport loss, UI detachment, authority revocation, expiry, and workflow
completion are different events. The parent need not understand browser users;
it only needs an opaque authority identifier and explicit lease transitions.
Protected lease or renewal secrets must remain process-local.

## Compatibility and migration

1. Advertise a new capability such as `hosting_operation_api_v2` only when the
   complete v2 contract is available.
2. Keep current toolbox receipt and family-specific workflow methods during a
   bounded migration window.
3. Implement v2 as a facade over existing runtimes first; migrate receipt
   storage afterward without changing the public v2 status model.
4. Provide typed Python models/helpers from the parent package so consumers do
   not reproduce lifecycle enums and normalization.
5. After a consumer repins to v2, remove signature inspection, local callback
   relay compatibility, local Host Capability session fallback, and
   family-specific status/cancel adapters from that consumer.

## Acceptance criteria

- One operation reference works for toolbox, workflow Python, and workflow
  JavaScript execute/status/cancel flows.
- Same request ID/fingerprint attaches or replays; conflicting fingerprint
  fails without dispatch.
- Every lifecycle state has the same bounded v2 response shape across families.
- Status/cancel needs only the operation reference, not reconstructed selector
  kwargs.
- Oversized retained results return a retrievable artifact reference; omitted
  results are identified as digest-only omissions.
- Receipt transitions remain correct across service recreation, concurrent
  status reads, cancellation, compaction, and v1-ledger migration.
- `provider_id` and `session_id` remain distinct through registration, list,
  dispatch, approval, audit, and close.
- Direct callback and pre-bound callback execution behave identically, remain
  live for streams, and release exactly once.
- Transport loss and authority revocation follow their explicit lease policies.
- Focused parent tests cover every contract branch without requiring a real
  daemon restart for each storage fixture; one daemon integration smoke is
  sufficient.

## Boundaries retained by consumers

The parent should not own:

- workspace admission fencing or workspace unload policy;
- consumer workflow pause/resume or absent-user dependencies;
- browser Disconnect versus Sign Out semantics;
- local turn/workflow application markers;
- replay eligibility for protected operations;
- session-tree, card, or Inspect projections;
- a consumer's SQLite journal of local application targets.

The parent supplies authoritative hosted-execution truth, transport-safe
capability sessions, and stable callback/lease primitives. Consumers decide how
that truth changes their own workspace, workflow, and UI state.
