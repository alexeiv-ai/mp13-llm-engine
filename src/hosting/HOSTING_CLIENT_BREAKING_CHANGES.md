# Dependent project handoff: durable execution and try-out recovery

Date: 2026-08-03
Required action: update the dependent project before pinning this parent revision.

This release intentionally removes the previous best-effort behavior. Do not add a client-side compatibility fallback for old hosts or old session recovery.

## What changed

### Hosted toolbox calls

1. `execution_request_id` is now required for every `toolbox_execute` call. Hosting no longer generates an execution request id.
2. The idempotency namespace is determined by the selector used for execution:
   - `toolbox_id="x"` uses `toolbox:x`
   - `engine_id="y"` uses `engine:y`
   Status and targeted cancellation must use the same selector.
3. The durable fingerprint includes tool name, canonical arguments, and effective policy. Reusing an id with different content returns:

   ```json
   {"status":"error","outcome":"idempotency_conflict","reason":"idempotency_conflict"}
   ```

4. A matching duplicate never dispatches twice. It attaches while queued/running and replays the stored terminal envelope after completion.
5. `toolbox_request_status` is no longer an in-memory pool lookup. Read `lifecycle_state` (also mirrored as `outcome`):
   - `queued`, `running`
   - `terminal_success`, `terminal_failure`, `terminal_cancellation`
   - `interrupted_before_dispatch`
   - `interrupted_after_dispatch_unknown`
   - `forgotten`
   - `unknown_outside_retention`
6. Only `interrupted_before_dispatch` can be retried under the same id. Treat `interrupted_after_dispatch_unknown` as non-replayable unless project-specific recovery obtains independent proof of the external outcome.
7. Targeted cancellation now requires `toolbox_cancel(request_id=<execution_request_id>, ...)`. `tool_call_id` is not accepted as a request-id fallback. Omitting `request_id` invokes the distinct coarse executor cancellation behavior.
8. Terminal replay can contain `result_reference` instead of the full result when the safe persistence limit is exceeded. Store and surface that state; do not interpret it as a failed tool execution.

The daemon advertises `durable_toolbox_execution_receipts_v1`. A host that lacks this capability is unsupported for protected replay. Fail closed instead of issuing an unprotected call.

### Session try-out anchors

1. Serialized sessions now use schema `4.6` and carry `ChatSession.try_out_anchor_descriptors` for unresolved anchors.
2. After loading a session, call `ChatContext.reconcile_try_out_anchors()` and inspect each result before resuming workflow logic.
3. Use `ChatContext.list_unresolved_try_out_anchors()` for recovery/UI state. Do not scan every historical `$try_out` marker to infer active anchors.
4. A reconciliation result with `status="interrupted"` is unresolved. Pause or escalate according to project policy; do not guess a branch.
5. Use `decrement_try_out_anchor_retry(...)` so the serialized remaining budget and revision stay current. Do not mutate `anchor.retries_remaining` directly.
6. `close_try_out_anchor(...)` is idempotent and returns `None` when the anchor is already closed or absent.
7. `resurrect_try_out_anchor(...)` is still available only for an explicit manual resurrection of historical markers.
8. Sessions with no descriptor field have no automatically recoverable active anchors. This is intentionally fail-safe; there is no legacy-marker automatic fallback.

## Required dependent-project migration

- Generate one stable execution request id before the first host call and journal it with the project workflow operation.
- Reuse that exact id and the same toolbox/engine selector for status, cancellation, and any allowed resume attempt.
- Persist the canonical inputs used to decide whether a retry is equivalent; surface `idempotency_conflict` as a project consistency error.
- Branch recovery logic on the explicit lifecycle states above. Never automatically replay `interrupted_after_dispatch_unknown`, `forgotten`, or `unknown_outside_retention` mutations.
- Accept a bounded terminal envelope or `result_reference`.
- On session load, reconcile descriptors once, retain interrupted outcomes, and leave workflow/user-interaction resume decisions in the dependent project.
- Remove any automatic scan of historical `$try_out` metadata and any host-version fallback that calls `toolbox_execute` without an execution request id.

## Responsibility boundary

Hosting supplies durable execution truth and compact turn-tree indices. It does not decide workflow replayability, retain project journals, unload workspaces, resume user interactions, or guarantee exactly-once external side effects for tools that lack their own idempotency support.
