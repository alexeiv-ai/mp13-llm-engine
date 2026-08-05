# Hosting Operation and Capability Access: Execution Plan

Status: feasible; direct breaking replacement after the contract decisions in Phase 0

Owner: parent hosting team

Consumer: `O:/repos/mp13-docs`

Consumer's current pinned parent: `084e559796b3ef01d94bbb749ba51aa215e79f05`

Reviewed against parent: `a74c4bdce3e0013238954808241fecd11a61e6dc`

Review date: 2026-08-04

## Verdict

The feature is feasible and fits the parent hosting boundary. It should replace
the current client contract directly. Do not add a parallel version, legacy
adapter, fallback signature, migration window, or deprecation period. Record
every client-visible and persisted-format break in
`src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md` before merging it. The dependent
project will promptly repin and adopt the new contract.

This is a multi-phase change, not only a client facade. Toolbox already has a
durable prepare/dispatch/terminal ledger, replay protection, pruning, and
restart tests. Workflow Python and JavaScript status/cancellation currently
depend on runtime-pool identity and in-memory request state. They must first be
put behind the same durable operation repository before the parent can honestly
advertise one cross-family status contract.

The remaining requests are compatible with existing building blocks:

- the client already has family-specific execute/status/cancel methods;
- the daemon and channel already have routing points that can be changed
  directly;
- the parent already has callback relay classes for provider and approval
  callbacks;
- Host Capability sessions already have expiry and disconnect handling;
- workflow artifacts already use host-minted `@artifacts/...` references; and
- the existing atomic, bounded JSON receipt ledger can be generalized behind a
  storage-neutral repository interface without changing storage technology.

The highest-risk areas are workflow durability across daemon recreation,
artifact authorization/retention, callback lifetime for streams, and lease
revocation semantics. These risks are manageable with the ordering and gates
below.

SQLite is explicitly not a dependency of this plan. A future storage migration
may be evaluated from measured receipt volume and write latency, but it must not
gate the contract or dependent-project repin.

API compatibility and data safety are separate concerns. The parent will not
read or translate a legacy receipt schema. It must fail closed when one is
present and provide a documented cutover procedure that archives the old ledger
only after the operator confirms no protected operation remains inside its
replay window.

## Non-negotiable contract decisions

Resolve and record these decisions before implementation. Defaults below are
recommended.

- [x] **D-01 Operation identity:** make `operation_id` a random, parent-minted,
  globally unique opaque value persisted in the repository. Treat the selector
  embedded in a client reference as descriptive only; status and cancel must
  resolve the stored operation and authorize it against the authenticated
  caller.
- [x] **D-02 Reference validation:** accept a typed model or mapping, require
  `contract == "hosting.operation_ref"`, bound every string/map, reject
  unknown or contradictory identity fields, and never trust a caller-supplied
  selector to route cancellation.
- [x] **D-03 Lifecycle enum:** use exactly `queued`, `running`,
  `terminal_success`, `terminal_failure`, `terminal_cancellation`,
  `interrupted_before_dispatch`, `interrupted_after_dispatch_unknown`,
  `forgotten`, `unknown_outside_retention`, and `idempotency_conflict`.
  Represent API-call success separately from operation lifecycle.
- [x] **D-04 Digest and size:** use `digest: "sha256:<hex>"` and
  `size_bytes: <integer>` everywhere. Do not introduce the ambiguous field
  `size`.
- [x] **D-05 Terminal payload:** define `result` for a bounded inline result,
  `result_ref` for a retrievable artifact, and `result_omission` for a
  digest-only result. These fields are mutually exclusive.
- [x] **D-06 Retention policy:** define which execution kinds/results may be
  artifact-backed, maximum artifact size, TTL, deletion behavior, and whether
  retrieval is single- or multi-read. Default to omission when policy is
  absent or denies retention.
- [x] **D-07 Lease expiry:** use `expires_at_ms: null` for no expiry. Do not use
  `0`, because the current session implementation interprets a past timestamp
  as immediately expired. A retained-on-transport-loss lease must either have
  a finite expiry or be explicitly renewable/revocable.
- [x] **D-08 Authority binding:** bind `owner_authority_id` to the authenticated
  actor that registers it. Add explicit renew and revoke operations; possessing
  an opaque authority string alone must not grant control.
- [x] **D-09 Request terminal event:** define `on_request_terminal` as the
  terminal transition of the referenced hosted operation, including
  cancellation and interrupted terminal policy. It is not merely the return of
  an execute RPC.
- [x] **D-10 Duplicate callbacks:** derive or persist stable `approval_id` and
  `provider_call_id` per logical parent request/call. A transport retry must
  reuse those IDs; a genuinely new provider call must not.
- [x] **D-11 Scope:** durable operations initially cover toolbox execute,
  workflow Python execute, and workflow JavaScript execute. Action execute and
  pinned-instance execute may share the workflow execution kinds if their
  fingerprint includes action/instance identity. Describe calls are callback-
  lease consumers but are not durable operations. Streams get callback lease
  support; durable stream recovery is out of scope unless separately
  specified.
- [x] **D-12 Breaking cutover:** define the parent base commit, staged release
  commit placeholder, dependent repin sequence, unsupported legacy receipt
  behavior, ledger archival command, and rollback constraints. No compatibility
  code may be introduced to smooth the cutover.

## Target public contracts

Freeze typed Python models for the following shapes. Keep transport envelopes
separate from these domain models.

```python
HostedOperationRef = {
    "contract": "hosting.operation_ref",
    "operation_id": "opaque-parent-id",
    "request_id": "caller-idempotency-id",
    "execution_kind": "toolbox|workflow_python|workflow_js",
    "selector": {"kind": "toolbox_id|engine_id", "id": "..."},
    "fingerprint": "sha256:<hex>",
    "receipt_namespace": "...",
}

HostedOperationStatus = {
    "contract": "hosting.operation_status",
    "api_status": "ok|error",
    "operation": HostedOperationRef,
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

### Phase 0 - Contract, threat model, and breaking-change freeze

- [x] **P0-01** Resolve D-01 through D-12 with the consumer and add the final
  models/enums to a parent-owned module (suggested:
  `src/hosting/operation_contract.py`).
- [x] **P0-02** Document the authorization matrix for operation status, cancel,
  artifact retrieval, session renew/revoke, administrative inspection, and
  forced close.
- [x] **P0-03** Define per-family canonical fingerprint inputs and test vectors.
- [x] **P0-04** Define the storage-neutral repository interface, JSON schema
  evolution, corruption behavior, backup policy, and process-locking
  assumptions.
- [x] **P0-05** Add contract serialization, validation, size-bound, and malformed
  input tests.
- [x] **P0-06** Create the entry in
  `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md` with change IDs, affected
  methods and shapes, old-to-new call examples, persisted-ledger cutoff steps,
  required dependent changes, the parent base commit, and a release-commit
  placeholder for the committer to fill because this task remains staged.

Exit gate: contract examples round-trip through typed models and the consumer
agrees it can update immediately from the breaking-change entry. The entry must
land with or before the first breaking implementation commit.

### Phase 1 - Generic operation repository and read facade

- [ ] **P1-01** Generalize `ToolboxExecutionReceiptLedger` behind a repository
  interface supporting prepare, dispatch claim, terminal transition,
  pre-dispatch cancel, lookup by `(owner_actor_id, namespace, request_id)`, lookup by
  `operation_id`, wait, and prune.
- [ ] **P1-02** Mint and persist `operation_id` during the first prepare; return
  the same reference for attach, replay, conflict, and tombstone responses.
- [ ] **P1-03** Add one status normalizer that emits
  `hosting.operation_status`; replace the existing toolbox status and receipt
  payload shape rather than preserving two representations.
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
- [ ] **P1-07** Remove the superseded toolbox status/cancel signatures and update
  all parent call sites and tests in the same change. Do not retain aliases or
  `TypeError` fallbacks.

Exit gate: toolbox uses one ref for status/cancel, superseded APIs are absent,
and the corresponding breaking-change entry is complete.

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

Exit gate: all three execution families return the same status shape and
generic status/cancel require only the operation ref.

### Phase 3 - Artifact-backed terminal results

- [ ] **P3-01** Add a dedicated terminal-result artifact manager or extend
  `HostedArtifactManager` with bounded byte writes, digest verification, TTL,
  ownership metadata, and safe deletion. Do not store arbitrary worker paths.
- [ ] **P3-02** On terminal persistence, redact first, serialize canonically,
  compute digest/size, then either store inline, write an allowed artifact, or
  emit `result_omission`. Never call an omission a reference.
- [ ] **P3-03** Define `hosting.result_ref` containing an opaque
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

### Phase 4 - Existing JSON repository evolution

- [ ] **P4-01** Keep the atomic JSON checkpoint as the production backend and
  implement the Phase 1 repository interface over it.
- [ ] **P4-02** Evolve the bounded JSON schema to store `operation_id`, execution
  kind, owner identity, selector, lifecycle timestamps, terminal
  payload/ref/omission, and tombstones.
- [ ] **P4-03** Maintain in-memory indexes for
  `(owner_actor_id, namespace, request_id)` and
  `operation_id`, rebuilt and validated during ledger load, so generic lookup
  does not require worker discovery or repeated full scans.
- [ ] **P4-04** Preserve the existing lock plus write-temp/fsync/atomic-replace
  transition model so concurrent callers cannot obtain two dispatch
  permissions.
- [ ] **P4-05** Implement deterministic age/count pruning ordered by timestamp
  then stable ID. Keep tombstones long enough to prevent unsafe re-dispatch.
- [ ] **P4-06** Reject a legacy schema without reading, translating, deleting,
  or overwriting it. Emit a bounded diagnostic pointing to the documented
  archival/cutover procedure.
- [ ] **P4-07** Provide an explicit operator cutover command that first verifies
  the resolved ledger path and requires acknowledgement that no protected
  operation remains inside its replay window, then archives rather than deletes
  the legacy file.
- [ ] **P4-08** Fail closed on invalid schema, interrupted cutover, or unreadable
  checkpoint. Ledger initialization must not start workers or sandboxes.
- [ ] **P4-09** Add concurrency, interrupted-write, deterministic pruning,
  legacy-schema rejection, cutover archival, corrupt JSON, and index-rebuild
  tests. Keep one daemon restart smoke test; use repository fixtures for the
  rest.

Exit gate: the evolved JSON backend passes the repository contract suite,
rejects rather than adapts legacy data, remains bounded, and preserves the
current atomic receipt guarantees.

### Phase 5 - Stable Host Capability provider identity

- [ ] **P5-01** Add `provider_id` as a first-class field on
  `HostCapabilitySession`, separate from `session_id`, and include it in public
  provider descriptors and private persistence/transport shapes.
- [ ] **P5-02** Accept `provider_id` in daemon/channel registration. Reject a
  missing ID and reject contradictory duplicates according
  to the Phase 0 uniqueness rules.
- [ ] **P5-03** Update broker discovery, method resolution, callback context,
  approvals, audit, list filters, close filters, upsert, toolbox providers, and
  service-broker helpers to use `provider_id` semantically and `session_id` only
  as the registration instance identity.
- [ ] **P5-04** Remove every fallback that derives `provider_id` from
  `session_id`. Update built-in, toolbox, and service-broker constructors to
  supply both identities explicitly.
- [ ] **P5-05** Add end-to-end tests proving distinct identities survive
  registration, list, discovery, dispatch, approval, audit, filtered close, and
  duplicate rejection.

Exit gate: no code path treats `session_id` as `provider_id`, and missing
`provider_id` fails validation.

### Phase 6 - Parent-owned approval callback lease

- [ ] **P6-01** Build `ApprovalCallbackLease` on the existing
  `HostCapabilityApprovalCallbackRelay`, with idempotent, thread-safe close and
  context-manager support.
- [ ] **P6-02** Let all Python/JavaScript execute, action-describe,
  action-execute, pinned-instance execute, and stream-open channel methods
  accept exactly one of `approval_requester`, `approval_requester_binding`, or
  `approval_callback_lease`. Reject contradictory inputs and remove superseded
  signatures instead of inspecting them dynamically.
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

- [ ] **P7-01** Replace the boolean with a validated lease model containing
  authenticated `owner_authority_id`, nullable expiry, transport-loss policy,
  authority-revocation policy, and request-terminal policy.
- [ ] **P7-02** Add authenticated renew and revoke commands and channel methods.
  Keep renewal/revocation secrets process-local and out of public descriptors,
  receipts, logs, and artifacts.
- [ ] **P7-03** Refactor disconnect cleanup to evaluate `on_transport_loss`
  rather than `close_on_client_disconnect`; remove the boolean from registration
  and session models.
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

### Phase 8 - Breaking release and dependent-project handoff

- [ ] **P8-01** Remove superseded toolbox/workflow status, cancel, callback,
  provider-identity, and lifetime signatures in the same release that adds the
  replacement contract. Do not advertise parallel old/new capabilities.
- [ ] **P8-02** Update parent hosting documentation and
  `HOSTING_CLIENT_BREAKING_CHANGES.md` with final method signatures, lifecycle
  mapping, retention behavior, ledger cutover procedure, and exact parent
  commit.
- [ ] **P8-03** Run focused parent tests plus one real-daemon integration smoke
  covering execute -> ref -> status -> cancel/replay -> artifact retrieval.
- [ ] **P8-04** Publish a dependent-project adoption checklist: repin to the
  recorded commit, switch to typed refs/status, use callback leases, provide
  explicit provider IDs and authority leases, and delete signature inspection,
  `TypeError` fallbacks, selector dictionaries, local callback relays, local
  Host Capability session fallback, and family-specific status/cancel adapters.
- [ ] **P8-05** Update the dependent project promptly and run its complete
  recovery, approval, capability-session, and workflow suites against the new
  parent pin.
- [ ] **P8-06** Define rollback as a source-and-data rollback to the previous
  parent commit and its archived ledger. Do not implement runtime API fallback
  or dual-format ledger support.

Exit gate: the consumer passes its recovery/approval/session tests against the
new parent pin, superseded parent APIs are absent, and no compatibility adapter
was added.

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
  service recreation, pruning, legacy-schema rejection, and corruption
  handling.
- [ ] `provider_id` and `session_id` remain distinct through registration,
  discovery, dispatch, approval, audit, list, and close.
- [ ] Direct and pre-bound approval callbacks behave identically; stream leases
  live until terminal/close and every binding releases exactly once.
- [ ] Transport loss, expiry, request completion, and authority revocation each
  follow their explicit lease policy.
- [ ] Receipt/repository load and status/replay do not start workers or
  sandboxes.
- [ ] Tests for superseded signatures are removed or rewritten, and tests assert
  that unsupported legacy inputs fail closed.

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
