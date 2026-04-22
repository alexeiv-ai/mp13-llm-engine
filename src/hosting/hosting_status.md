# Hosting Status

## Daemon Operation Cancellation

Status: engine-side daemon cancellation implemented.

Implemented in this repo:

- Added daemon `op-cancel` handling for operations created through `op-start`.
- Added operation task tracking by `operation_id`.
- Added cancel metadata to operation snapshots:
  - `cancel_requested`
  - `cancel_requested_at`
  - `cancel_completed_at`
  - `cancel_reason`
  - `cancel_teardown_attempted`
  - `cancel_teardown_status`
  - `target_engine_id`
- Added best-effort cancellation behavior:
  - marks running operations as cancel requested
  - cancels the daemon wrapper task when it is still active
  - records terminal `canceled` state when the wrapper task is canceled
  - reports `already_done` for terminal operations
  - attempts worker teardown for `connect-from-config` and `spawn` operations when `engine_id` is known
  - preserves running load/spawn wrappers long enough to capture a late `engine_id` result and then tear that worker down
- Added `op-cancel` to daemon auth/policy allowlists.
- Added `EngineHostControlChannel.cancel_host_operation(operation_id=..., reason=...)`.
- Added CLI usage example and parser support for `op-cancel`.
- Added focused daemon tests for running-operation cancellation, session-token enforcement, and known-engine teardown.

Important behavior note:

`op-cancel` is best-effort. The daemon can cancel its asyncio operation wrapper, but it cannot forcibly stop synchronous work already running inside `asyncio.to_thread(...)`. For load/connect cancellation, the practical stop mechanism is worker teardown by `engine_id` when that id is known.

Remaining integration work outside the engine daemon:

- Wire backend `/api/operations/{operation_id}/cancel` to call `cancel_host_operation()` instead of only mutating backend-local operation bindings.
- Ensure backend operation bindings preserve the daemon `operation_id` and target `engine_id`.
- Refresh backend operation projections from daemon `op-status` after cancel requests.
- Keep load operations `cancelable: false` until backend cancel propagation reaches daemon `op-cancel` and worker teardown behavior is verified end to end.
- After that wiring lands, generic clients can enable cancel controls only when an operation projection reports `cancelable: true`, then continue polling or streaming operation status until a terminal state.
