#  Hosting and Session Recovery Requirements


The consuming project needs two narrow additions. The goal is to remove replay
ambiguity after daemon or backend restart without turning the parent into a
workflow database and without copying the serialized session tree.

## Completion checklist

### Durable hosted-execution receipts

- [x] Require caller-supplied execution ids and scope them to the exact toolbox/engine host namespace.
- [x] Persist canonical tool/arguments/policy fingerprints and bounded lifecycle timestamps before dispatch.
- [x] Persist the dispatch claim immediately before tool invocation.
- [x] Attach/replay same-fingerprint duplicates without a second dispatch in queued, running, and terminal states.
- [x] Reject different-fingerprint duplicates with stable `idempotency_conflict`.
- [x] Persist terminal success, failure, and cancellation before acknowledging the client.
- [x] Recover pre-dispatch receipts as resumable once and post-dispatch receipts as fail-closed unknown.
- [x] Expose queued/running, terminal, interrupted, forgotten, and outside-retention status values.
- [x] Preserve execution-envelope fields, targeted cancellation, pool lifecycle, and separate coarse cancellation.
- [x] Use a host-owned atomic ledger without persisted worker, queue, callback, credential, argument, or stream state.
- [x] Bound/redact terminal envelopes and emit a digest/result reference for oversized results.
- [x] Configure receipt/tombstone age, receipt/tombstone count, and safe-result size.
- [x] Compact deterministically through bounded forgotten tombstones to `unknown_outside_retention`.
- [x] Load receipts in proportion to retained records without starting a sandbox or worker.
- [x] Test one-dispatch duplicates while queued, running, and terminal.
- [x] Test fingerprint conflicts, restart-safe queued cancellation, and terminal recreation.
- [x] Test both crash windows, retention compaction, credential redaction, and result bounds.
- [x] Keep restart coverage to one local-IPC daemon smoke test; use in-process unit tests for the state matrix.

### Compact try-out anchor recovery

- [x] Persist a versioned bounded descriptor for every unresolved anchor with identifiers and lifecycle facts only.
- [x] Reconcile descriptors against stable turn ids after deserialization and rebuild cursors lazily.
- [x] List unresolved anchors together with their latest reconciliation result.
- [x] Make repeated reconciliation and close/disposition idempotent.
- [x] Report missing, duplicate, or structurally ambiguous references as interrupted without guessing or copying turns.
- [x] Preserve explicit manual historical resurrection while removing automatic historical-marker fallback.
- [x] Test active serialize/load/reconcile/close and repeated reconciliation/close.
- [x] Test multiple branches/scopes without cross-binding.
- [x] Test missing/ambiguous ids and prove closed anchors are not reopened.
- [x] Test metadata size independence from message bytes and tree depth.
- [x] Preserve existing session serialization and manual resurrection behavior.

### Contract handoff and boundaries

- [x] Document public hosting/session APIs, namespaces, configuration, and status values.
- [x] Pass focused receipt, session, hosting, channel, and dependent app tests.
- [x] Run the complete parent suite and rerun the sole process-startup timeout successfully in isolation.
- [x] Document the intentional no-fallback migration for hosts/sessions that lack the new durable contracts.
- [x] Keep workflow replay policy, journals, workspace lifecycle, tree projection, and user interaction resume in the dependent project.
- [x] Avoid duplicate cursor/session trees, worker memory, callback authority, exactly-once claims for non-idempotent external tools, bulk cancellation, and distributed queues.
- [x] Provide the dependent project with a parent commit for pinning and an explicit breaking-change handoff.

## 1. Durable hosted-execution receipts

Apply this first to the hosted toolbox request path used by
`toolbox_execute(..., execution_request_id=...)`,
`toolbox_request_status(...)`, and `toolbox_cancel(...)`. Reuse the primitive
for other hosted request families only where that is a small, compatible
extension; broad migration of every runtime API is not a prerequisite.

### Required behavior

1. Treat the caller-supplied `execution_request_id` as an idempotency key within
   a documented caller/host namespace.
2. Before dispatch, durably record a compact receipt containing the request id,
   a canonical tool/arguments/policy fingerprint, lifecycle state, and bounded
   timestamps. Do not store raw credentials or unbounded arguments.
   Persist a dispatch-claimed transition immediately before invoking the tool.
3. A duplicate id with the same fingerprint must never dispatch the tool a
   second time:
   - while queued or running, return/attach to the existing lifecycle;
   - after terminal completion, return the prior terminal execution envelope
     and its bounded result or result reference;
   - after cancellation, return the prior canceled result.
4. A duplicate id with a different fingerprint must fail with a stable
   `idempotency_conflict` outcome and must not dispatch.
5. Persist the terminal transition before acknowledging it to the client. On
   restart, a receipt with no dispatch claim is
   `interrupted_before_dispatch` and may safely resume once under the same id.
   A receipt with a dispatch claim but no terminal transition is
   `interrupted_after_dispatch_unknown`; it is not permission to execute again.
6. `toolbox_request_status` must distinguish at least:
   - known queued/running;
   - known terminal success/failure/cancellation;
   - interrupted before dispatch;
   - interrupted after dispatch with completion unknown;
   - expired/forgotten while a compact tombstone is retained;
   - unknown outside the configured retention horizon.
7. Preserve the existing execution envelope fields and targeted cancellation
   semantics. This requirement adds recovery truth; it does not replace the
   current pool lifecycle or coarse running-cancel behavior.

An interrupted-unknown receipt is intentionally fail-closed. The consuming
project will not automatically replay a protected or non-restartable mutation
from that state.

### Storage and retention bounds

- Use a small append/checkpoint ledger or equivalent atomic store owned by the
  host. Do not serialize worker objects, queues, process state, raw tool
  arguments, callback/session tokens, or full stream histories.
- Store a digest plus bounded scalar metadata. Store a bounded terminal result
  only when it fits the existing safe-result limit; otherwise store an artifact
  or result reference and digest.
- Make retention configurable by age and count, with deterministic compaction.
  Expired known receipts leave a compact `forgotten` tombstone long enough for
  the supported replay window. Once that bounded tombstone also expires, report
  `unknown_outside_retention`; do not claim the id was definitely never seen.
- Loading the ledger must be proportional to retained receipts, not to session
  tree size, and must not start a sandbox or worker.

### Required tests

- same id and same fingerprint while queued, running, and terminal dispatches
  exactly once;
- same id and different fingerprint returns `idempotency_conflict`;
- queued cancellation survives restart and never reaches the tool;
- terminal success/failure/cancellation remains queryable after service
  recreation;
- crash-window fixtures recover pre-dispatch as safe-to-resume and
  post-dispatch as unknown/not-replayable;
- compaction preserves `forgotten` through the replay window and then returns
  `unknown_outside_retention` without an unbounded tombstone set;
- persisted records reject or redact credentials and obey result-size bounds.

Most of these should be storage/runtime unit tests that recreate the service
against a temporary ledger in-process. Keep only a small end-to-end daemon
restart smoke test so the parent suite does not restart the real server for
every state permutation.

## 2. Compact try-out anchor recovery

The session already serializes turns and their `$try_out` metadata, and
`ChatContext` can rebuild cursors for stable turn ids. Preserve that design.
Do not serialize cursors or duplicate any turn subtree.

### Required behavior

1. Persist one versioned, bounded descriptor for each unresolved try-out
   anchor. It may live in compact session metadata or on the existing anchor
   turn. It contains identifiers and lifecycle facts only:
   - anchor name/id and kind;
   - anchor turn id;
   - direct try-out turn ids needed to restore the anchor;
   - scope id and origin turn/cursor identity when still meaningful;
   - retry limit and remaining retries;
   - lifecycle `active|closed|interrupted`;
   - disposition/close mode and bounded reason;
   - a small generation/revision used to make close/rebind idempotent.
2. After deserialization, expose a reconciliation method that validates those
   ids against the existing turn tree and lazily rebuilds only the cursors that
   are actually needed.
3. Expose a way to list unresolved anchors and their reconciliation result.
4. Make close/disposition idempotent. Repeating close or recovery must not
   promote a branch twice, create another placeholder, decrement retry budget
   twice, or rebind the active cursor twice.
5. If referenced turns are absent or the shape is ambiguous, return an explicit
   interrupted/unresolved outcome. Do not guess a branch and do not copy turns.
6. Preserve current manual `resurrect_try_out_anchor` behavior for historical
   anchors. Automatic restart reconciliation should use the compact lifecycle
   descriptor rather than scanning and treating every historical `$try_out`
   marker as active.

The descriptor is an index into the canonical turn tree, not another tree. Its
serialized cost must be O(number of unresolved anchors), with a fixed bounded
record size. It must not include prompts, responses, tool arguments/results,
execution envelopes, cursor caches, ancestry paths, or nested turn objects.

### Required tests

- serialize/deserialize an active automatic anchor, reconcile it, and close it
  exactly once;
- repeat reconcile and close to prove idempotency;
- restore multiple anchors on different branches/scopes without cross-binding;
- report a missing or ambiguous turn id as interrupted/unresolved;
- prove closed anchors are not reopened automatically;
- add large messages and deep branches, then prove anchor metadata size does
  not grow with message bytes or tree depth;
- preserve existing session serialization and manual resurrection behavior.

Use synthetic in-memory sessions for the matrix. One serialize/load integration
test is sufficient; no daemon restart is needed for every cursor case.

## Contract handoff

The parent change is ready for the consuming project when:

1. the public hosting/session APIs and status values are documented;
2. focused and existing parent tests pass;
3. a parent commit is provided for pinning;
4. migration from sessions without descriptors and hosts without receipt
   ledgers is fail-safe and backwards compatible.

The parent does not decide whether a project workflow is replayable, does not
own workspace unload policy, and does not resume user interactions. It supplies
durable execution truth and compact anchor recovery primitives; the consuming
project owns replay eligibility, workflow pause/resume, tree projection, and
user-facing recovery.

## Explicit non-goals

- serializing the complete cursor registry or session tree again;
- storing one session-tree snapshot per anchor;
- persisting worker/sandbox process memory or live callback authority;
- providing exactly-once external side effects when a tool itself has no
  idempotency support;
- adding bulk cancel, owner/group cancel, or a distributed queue as a
  prerequisite;
- moving project workflow journals or replay policy into the parent.
