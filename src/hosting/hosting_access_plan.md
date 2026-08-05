# Hosting Operation and Capability Access v2: Execution Plan

Status: feasible; implementation-ready after the contract decisions in Phase 0

Owner: parent hosting team

Consumer: `O:/repos/mp13-docs`

Consumer's current pinned parent: `084e559796b3ef01d94bbb749ba51aa215e79f05`

Reviewed against parent: `a74c4bdce3e0013238954808241fecd11a61e6dc`

Review date: 2026-08-04

## Verdict

The feature is feasible and fits the parent hosting boundary. It should be
implemented as an additive v2 API while retaining
`durable_toolbox_execution_receipts_v1` and the family-specific methods during
migration.

This is a multi-phase change, not only a client facade. Toolbox already has a
durable prepare/dispatch/terminal ledger, replay protection, pruning, and
restart tests. Workflow Python and JavaScript status/cancellation currently
depend on runtime-pool identity and in-memory request state. They must first be
put behind the same durable operation repository before the parent can honestly
advertise one cross-family status contract.

The remaining requests are compatible with existing building blocks:

- the client already has family-specific execute/status/cancel methods;
- the daemon already advertises versioned capabilities;
- the parent already has callback relay classes for provider and approval
  callbacks;
- Host Capability sessions already have expiry and disconnect handling;
- workflow artifacts already use host-minted `@artifacts/...` references; and
- Python's standard `sqlite3` is sufficient for a WAL-backed repository.

The highest-risk areas are workflow durability across daemon recreation,
artifact authorization/retention, callback lifetime for streams, and lease
revocation semantics. These risks are manageable with the ordering and gates
below.

## Non-negotiable contract decisions

Resolve and record these decisions before implementation. Defaults below are
recommended.

- [ ] **D-01 Operation identity:** make `operation_id` a random, parent-minted,
  globally unique opaque value persisted in the repository. Treat the selector
  embedded in a client reference as descriptive only; status and cancel must
  resolve the stored operation and authorize it against the authenticated
  caller.
- [ ] **D-02 Reference validation:** accept a typed model or mapping, require
  `contract == "hosting.operation_ref.v2"`, bound every string/map, reject
  unknown or contradictory identity fields, and never trust a caller-supplied
  selector to route cancellation.
- [ ] **D-03 Lifecycle enum:** use exactly `queued`, `running`,
  `terminal_success`, `terminal_failure`, `terminal_cancellation`,
  `interrupted_before_dispatch`, `interrupted_after_dispatch_unknown`,
  `forgotten`, `unknown_outside_retention`, and `idempotency_conflict`.
  Represent API-call success separately from operation lifecycle.
- [ ] **D-04 Digest and size:** use `digest: "sha256:<hex>"` and
  `size_bytes: <integer>` everywhere. Do not introduce the ambiguous field
  `size`.
- [ ] **D-05 Terminal payload:** define `result` for a bounded inline result,
  `result_ref` for a retrievable artifact, and `result_omission` for a
  digest-only result. These fields are mutually exclusive.
- [ ] **D-06 Retention policy:** define which execution kinds/results may be
  artifact-backed, maximum artifact size, TTL, deletion behavior, and whether
  retrieval is single- or multi-read. Default to omission when policy is
  absent or denies retention.
- [ ] **D-07 Lease expiry:** use `expires_at_ms: null` for no expiry. Do not use
  `0`, because the current session implementation interprets a past timestamp
  as immediately expired. A retained-on-transport-loss lease must either have
  a finite expiry or be explicitly renewable/revocable.
- [ ] **D-08 Authority binding:** bind `owner_authority_id` to the authenticated
  actor that registers it. Add explicit renew and revoke operations; possessing
  an opaque authority string alone must not grant control.
- [ ] **D-09 Request terminal event:** define `on_request_terminal` as the
  terminal transition of the referenced hosted operation, including
  cancellation and interrupted terminal policy. It is not merely the return of
  an execute RPC.
- [ ] **D-10 Duplicate callbacks:** derive or persist stable `approval_id` and
  `provider_call_id` per logical parent request/call. A transport retry must
  reuse those IDs; a genuinely new provider call must not.
- [ ] **D-11 Scope:** v2 durable operations initially cover toolbox execute,
  workflow Python execute, and workflow JavaScript execute. Action execute and
  pinned-instance execute may share the workflow execution kinds if their
  fingerprint includes action/instance identity. Describe calls are callback-
  lease consumers but are not durable operations. Streams get callback lease
  support in v2; durable stream recovery is out of scope unless separately
  specified.

## Target public contracts

Freeze typed Python models for the following shapes. Keep transport envelopes
separate from these domain models.

```python
HostedOperationRefV2 = {
    "contract": "hosting.operation_ref.v2",
    "operation_id": "opaque-parent-id",
    "request_id": "caller-idempotency-id",
    "execution_kind": "toolbox|workflow_python|workflow_js",
    "selector": {"kind": "toolbox_id|engine_id", "id": "..."},
    "fingerprint": "sha256:<hex>",
    "receipt_namespace": "...",
}

HostedOperationStatusV2 = {
    "contract": "hosting.operation_status.v2",
    "operation": HostedOperationRefV2,
    "lifecycle": "queued|running|terminal_success|terminal_failure|terminal_cancellation|interrupted_before_dispatch|interrupted_after_dispatch_unknown|forgotten|unknown_outside_retention|idempotency_conflict",
    "request_id": "...",
    "created_at_ms": 0,
    "updated_at_ms": 0,
    "dispatch_claimed_at_ms": None,
    "terminal_at_ms": None,
    "reason": None,
    "result": None,
    "result_ref": None,
    "result_omission": None,
}
```

All free-form reasons, selectors, metadata, and returned payloads must have
documented byte/count limits. Fingerprints are computed over canonical,
family-specific dispatch inputs and policy, never callback bindings,
credentials, or other ephemeral transport data.

## Itemized execution plan

### Phase 0 - Contract and threat-model freeze

- [ ] **P0-01** Resolve D-01 through D-11 with the consumer and add the final
  models/enums to a parent-owned module (suggested:
  `src/hosting/operation_contract.py`).
- [ ] **P0-02** Document the authorization matrix for operation status, cancel,
  artifact retrieval, session renew/revoke, administrative inspection, and
  forced close.
- [ ] **P0-03** Define per-family canonical fingerprint inputs and test vectors.
- [ ] **P0-04** Define the SQLite schema, schema version, migration transaction,
  corruption behavior, backup policy, and process-locking assumptions.
- [ ] **P0-05** Add contract serialization, validation, size-bound, and malformed
  input tests.

Exit gate: contract examples round-trip through typed models and the consumer
agrees it can delete its normalization logic once the full capability is
advertised.

### Phase 1 - Generic operation repository and read facade

- [ ] **P1-01** Generalize `ToolboxExecutionReceiptLedger` behind a repository
  interface supporting prepare, dispatch claim, terminal transition,
  pre-dispatch cancel, lookup by `(namespace, request_id)`, lookup by
  `operation_id`, wait, and prune.
- [ ] **P1-02** Mint and persist `operation_id` during the first prepare; return
  the same reference for attach, replay, conflict, and tombstone responses.
- [ ] **P1-03** Add one status normalizer that emits
  `hosting.operation_status.v2`; adapt the existing toolbox receipt states
  without changing v1 payloads.
- [ ] **P1-04** Add daemon commands and channel methods
  `hosted_operation_status(ref=...)` and
  `hosted_operation_cancel(ref=..., reason=...)`, including auth/policy/CLI
  routing where appropriate.
- [ ] **P1-05** Make generic lookup resolve the stored selector and owner. Reject
  altered refs, cross-owner access, execution-kind mismatch, and unknown
  operation IDs without probing workers.
- [ ] **P1-06** Add toolbox facade tests for new, attach, replay, conflict,
  pre-dispatch cancel, post-dispatch cancel, forgotten, unknown, and interrupted
  states.

Exit gate: toolbox can use one ref for status/cancel while all existing v1
tests and methods remain unchanged.

### Phase 2 - Durable workflow Python and JavaScript integration

- [ ] **P2-01** Define stable workflow namespaces/selectors from the resolved
  runtime registration. Do not require a caller to reconstruct
  `environment_key`, profile, or engine ID for later status/cancel.
- [ ] **P2-02** Wrap workflow Python execute with repository prepare before pool
  submission, dispatch claim immediately before worker dispatch, and terminal
  persistence on every return/error/cancel path.
- [ ] **P2-03** Apply the same wrapper to workflow JavaScript execute.
- [ ] **P2-04** Include runtime, action, pinned-instance, request body, effective
  sandbox policy, and other dispatch-affecting inputs in family-specific
  fingerprints. Exclude callback bindings and secrets.
- [ ] **P2-05** Route generic status and cancel from stored operation identity to
  the correct pool/runtime. Persist cancellation races atomically and preserve
  the existing fail-closed `interrupted_after_dispatch_unknown` behavior.
- [ ] **P2-06** Ensure attach/replay never starts an environment, worker, or
  sandbox. Preserve the existing local application-journal boundary.
- [ ] **P2-07** Add parameterized parity tests across toolbox, workflow Python,
  and workflow JavaScript for every lifecycle branch and fingerprint conflict.
- [ ] **P2-08** Add service-recreation tests proving workflow terminal replay,
  pre-dispatch recovery, post-dispatch uncertainty, and cancel correctness
  without worker startup.

Exit gate: all three execution families return the same v2 status shape and
generic status/cancel require only the operation ref.

### Phase 3 - Artifact-backed terminal results

- [ ] **P3-01** Add a dedicated terminal-result artifact manager or extend
  `HostedArtifactManager` with bounded byte writes, digest verification, TTL,
  ownership metadata, and safe deletion. Do not store arbitrary worker paths.
- [ ] **P3-02** On terminal persistence, redact first, serialize canonically,
  compute digest/size, then either store inline, write an allowed artifact, or
  emit `result_omission`. Never call an omission a reference.
- [ ] **P3-03** Define a versioned `hosting.result_ref.v2` containing an opaque
  artifact ID, digest, size, media type, and expiry. Avoid exposing host paths.
- [ ] **P3-04** Add an authorization-checked dereference endpoint/channel method
  that verifies operation ownership, retention, size, and digest before
  returning bounded bytes/content.
- [ ] **P3-05** Couple artifact pruning to receipt/tombstone retention with a
  deterministic orphan cleanup pass.
- [ ] **P3-06** Test allowed retention, denied retention, oversized artifact
  limits, credential redaction, tampering, expiry, cross-actor denial, missing
  files, digest mismatch, and cleanup.

Exit gate: every non-inline terminal result is either actually retrievable by
its authorized owner or explicitly marked digest-only.

### Phase 4 - SQLite WAL repository and v1 JSON import

- [ ] **P4-01** Implement the Phase 1 repository interface using SQLite with
  foreign keys, WAL mode, a busy timeout, short `BEGIN IMMEDIATE` write
  transactions, and read-only status transactions.
- [ ] **P4-02** Add indexed unique keys for `(namespace, request_id)` and
  `operation_id`; store lifecycle timestamps, bounded metadata, terminal
  payload/ref/omission, and tombstones in versioned tables.
- [ ] **P4-03** Make prepare, dispatch claim, terminal transition, and
  pre-dispatch cancellation compare-and-set transactions so concurrent callers
  cannot obtain two dispatch permissions.
- [ ] **P4-04** Implement deterministic age/count pruning ordered by timestamp
  then stable ID. Keep tombstones long enough to prevent unsafe re-dispatch.
- [ ] **P4-05** Import the existing
  `state/toolbox_execution_receipts.json` inside one idempotent transaction.
  Validate all rows, preserve timestamps/fingerprints/states, record import
  completion, and retain the JSON file as a backup until rollout is accepted.
- [ ] **P4-06** Fail closed on invalid schema, failed integrity check, partial
  migration, or unreadable database. Ledger initialization must not start
  workers or sandboxes.
- [ ] **P4-07** Remove the process-wide JSON-ledger cache or replace it with
  explicit repository lifecycle/close handling in `EngineHostService`.
- [ ] **P4-08** Add concurrency, crash-boundary, busy-reader, deterministic
  pruning, migration idempotency, corrupt JSON, corrupt DB, and schema-upgrade
  tests. Keep one daemon restart smoke test; use repository fixtures for the
  rest.

Exit gate: SQLite passes the same repository contract suite as the JSON
implementation, imports v1 exactly once, and no lifecycle transition rewrites
the full checkpoint.

### Phase 5 - Stable Host Capability provider identity

- [ ] **P5-01** Add `provider_id` as a first-class field on
  `HostCapabilitySession`, separate from `session_id`, and include it in public
  provider descriptors and private persistence/transport shapes.
- [ ] **P5-02** Accept `provider_id` in daemon/channel registration. Reject a
  missing ID where v2 requires it and reject contradictory duplicates according
  to the Phase 0 uniqueness rules.
- [ ] **P5-03** Update broker discovery, method resolution, callback context,
  approvals, audit, list filters, close filters, upsert, toolbox providers, and
  service-broker helpers to use `provider_id` semantically and `session_id` only
  as the registration instance identity.
- [ ] **P5-04** Preserve v1 behavior by defaulting `provider_id = session_id`
  only in the compatibility adapter, never in the v2 model.
- [ ] **P5-05** Add end-to-end tests proving distinct identities survive
  registration, list, discovery, dispatch, approval, audit, filtered close, and
  duplicate rejection.

Exit gate: no v2 code path treats `session_id` as `provider_id`.

### Phase 6 - Parent-owned approval callback lease

- [ ] **P6-01** Build `ApprovalCallbackLease` on the existing
  `HostCapabilityApprovalCallbackRelay`, with idempotent, thread-safe close and
  context-manager support.
- [ ] **P6-02** Let all Python/JavaScript execute, action-describe,
  action-execute, pinned-instance execute, and stream-open channel methods
  accept exactly one of `approval_requester`, `approval_requester_binding`, or
  `approval_callback_lease`. Reject contradictory inputs.
- [ ] **P6-03** For a callable, bind before invoking the daemon and release
  exactly once after a non-stream request returns or raises. For streams,
  transfer ownership to the stream handle and release on terminal event,
  explicit close/cancel, open failure, or channel shutdown.
- [ ] **P6-04** Ensure a pre-created lease can span multiple calls only when its
  documented scope permits it; otherwise fail closed on scope mismatch.
- [ ] **P6-05** Persist/reuse logical callback IDs as decided in D-10 and test
  allow-once idempotency under duplicate delivery.
- [ ] **P6-06** Add parity tests for direct callable versus pre-bound binding,
  synchronous success/error/timeout, stream lifetime, double close, open
  failure, disconnect, and callback exception.

Exit gate: consumer code no longer needs to construct/release approval relays,
including for streams, and leak/double-release tests pass.

### Phase 7 - Explicit capability-session authority leases

- [ ] **P7-01** Replace the v2 boolean with a validated lease model containing
  authenticated `owner_authority_id`, nullable expiry, transport-loss policy,
  authority-revocation policy, and request-terminal policy.
- [ ] **P7-02** Add authenticated renew and revoke commands and channel methods.
  Keep renewal/revocation secrets process-local and out of public descriptors,
  receipts, logs, and artifacts.
- [ ] **P7-03** Refactor disconnect cleanup to evaluate `on_transport_loss`
  rather than `close_on_client_disconnect`; preserve the boolean only in the v1
  adapter.
- [ ] **P7-04** Wire operation terminal transitions to
  `on_request_terminal=close`, authority revocation to the configured policy,
  and expiry to deterministic cleanup. Make every close path idempotent and
  audited with its cause.
- [ ] **P7-05** Define daemon-restart behavior. If sessions remain in-memory,
  report them closed on daemon loss and do not promise survival; if persistence
  is required later, make it a separate capability because callback bindings
  may not be recoverable.
- [ ] **P7-06** Test transport loss with remaining actor connections, final
  transport loss, retain-until-expiry, explicit renewal, expiry, authority
  revocation, request terminal, races among close causes, and unauthorized
  renew/revoke.

Exit gate: transport attachment and authority lifetime are independently
observable and enforceable.

### Phase 8 - Capability gate, compatibility, and consumer migration

- [ ] **P8-01** Advertise `hosting_operation_api_v2` only when Phases 1 through
  7 are enabled and their complete contract is available. Do not advertise a
  partial umbrella capability; expose narrower experimental flags during
  staged development if needed.
- [ ] **P8-02** Keep current toolbox receipt and family-specific workflow APIs
  for a documented, bounded migration window. Add deprecation telemetry before
  removal.
- [ ] **P8-03** Update parent hosting docs and breaking-change notes with v1/v2
  examples, lifecycle mapping, retention behavior, and compatibility window.
- [ ] **P8-04** Run focused parent tests plus one real-daemon integration smoke
  covering execute -> ref -> status -> cancel/replay -> artifact retrieval.
- [ ] **P8-05** Publish a consumer migration checklist: capability probe, repin,
  switch to typed refs/status, use callback leases, use explicit provider IDs
  and authority leases, then delete signature inspection, `TypeError`
  fallbacks, selector dictionaries, relay compatibility, local Host Capability
  session fallback, and family-specific status/cancel adapters.
- [ ] **P8-06** Confirm rollback: disabling v2 must leave v1 methods and the
  imported durable data usable without re-dispatching protected operations.

Exit gate: the consumer passes its recovery/approval/session tests against the
new parent pin and no longer uses the listed compatibility adapters.

## Acceptance test matrix

- [ ] A single typed operation ref works for execute attachment/replay, status,
  and cancel for all three families.
- [ ] The same request ID plus fingerprint attaches or replays; a changed
  fingerprint returns `idempotency_conflict` without dispatch.
- [ ] Every lifecycle uses the same bounded status shape, digest spelling,
  `size_bytes` field, timestamps, and reason limits.
- [ ] Status/cancel uses stored identity and does not require or trust
  reconstructed selector kwargs.
- [ ] Retained oversized results are retrievable and authorized; non-retained
  results are explicit digest-only omissions.
- [ ] Transitions remain correct under concurrent reads, cancellation races,
  service recreation, pruning, v1 import, and corruption handling.
- [ ] `provider_id` and `session_id` remain distinct through registration,
  discovery, dispatch, approval, audit, list, and close.
- [ ] Direct and pre-bound approval callbacks behave identically; stream leases
  live until terminal/close and every binding releases exactly once.
- [ ] Transport loss, expiry, request completion, and authority revocation each
  follow their explicit lease policy.
- [ ] Receipt/repository load and status/replay do not start workers or
  sandboxes.
- [ ] Existing v1 tests remain green throughout the migration window.

## Consumer-owned boundaries

The parent must not take ownership of:

- workspace admission fencing or workspace unload policy;
- consumer workflow pause/resume or absent-user dependencies;
- browser Disconnect versus Sign Out semantics;
- local turn/workflow application markers;
- replay eligibility for protected operations;
- session-tree, card, or Inspect projections; or
- a consumer's SQLite journal of local application targets.

The parent supplies authoritative hosted-execution truth, transport-safe
capability sessions, and stable callback/lease primitives. Consumers decide
how that truth changes their own workspace, workflow, and UI state.
