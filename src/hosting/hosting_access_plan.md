# Hosting Access Plan

## Daemon Operation Cancellation

### Context

The backend already exposes operation lifecycle endpoints for generic clients:

- `GET /api/operations`
- `GET /api/operations/{operation_id}`
- `GET /api/operations/{operation_id}/events`
- `POST /api/operations/{operation_id}/cancel`

Those endpoints currently project backend-local operation bindings. Cancel is not a real daemon cancellation path yet: the backend marks its local binding as canceled, but it does not instruct the engine host daemon to stop the underlying host operation.

The engine host daemon currently supports asynchronous host operations through:

- `op-start` in `src/hosting/daemon/local_ipc.py`
- `op-status` in `src/hosting/daemon/local_ipc.py`
- `start_host_operation()` and `get_host_operation_status()` in `src/hosting/engine_host_channel.py`

There is no `op-cancel` command, no daemon auth/policy allowlist entry for it, and no channel wrapper such as `cancel_host_operation(operation_id=...)`.

Load/connect operations are deliberately reported as `cancelable: false` by the backend today because the daemon has no cancel primitive. This avoids advertising a UI or API behavior that does not actually reach the daemon.

### Scope

Add daemon-native cancellation for host operations started with `op-start`. The first target is canceling model/instance load operations that run through `connect-from-config`.

This cancellation API is for daemon operation lifecycle, not direct tool-call lifecycle.

Tool calls and toolbox execution already have separate cancellation paths such as `toolbox-cancel`, proxy stream cancel messages, and worker/runtime-specific cancellation. Do not merge those into `op-cancel` unless a future operation is explicitly started through `op-start`.

### Semantics

`op-cancel` should be a best-effort cancellation request, not a hard guarantee that every underlying blocking operation stops immediately.

For load/connect operations, cooperative cancellation inside Python, CUDA, model loading, or third-party libraries is not required. The credible behavior is process teardown:

- If the operation has not spawned a worker yet, mark it canceled and prevent further daemon-side continuation where possible.
- If the operation has spawned or selected a worker process, attempt to shut down that worker by `engine_id`.
- If the operation already finished, return an already-terminal status without changing the result.
- If teardown fails, surface that explicitly instead of reporting a clean cancel.

The API must avoid implying that cancel means synchronous hard interruption. The contract should be "cancel requested; daemon will stop waiting and will tear down a known worker process when possible."

### Engine Host Changes

Add `op-cancel` support in `src/hosting/daemon/local_ipc.py`.

Required daemon state changes:

- Track operation tasks by `operation_id`, not only as an anonymous set of tasks.
- Track per-operation metadata needed for cleanup, especially command, session token, cancel request time, target `engine_id`, and teardown outcome.
- Preserve the existing task set or equivalent cleanup behavior for shutdown drain.
- Add terminal or near-terminal status fields for cancellation:
  - `cancel_requested`
  - `canceled`
  - `cancel_failed`
  - `already_done`

Recommended operation record additions:

- `cancel_requested: bool`
- `cancel_requested_at: float | None`
- `cancel_completed_at: float | None`
- `cancel_reason: str | None`
- `cancel_teardown_attempted: bool`
- `cancel_teardown_status: str | None`
- `target_engine_id: str | None`

`op-cancel` request payload:

```json
{
  "operation_id": "required",
  "session_token": "optional, required when the operation was created with one",
  "reason": "optional"
}
```

`op-cancel` should apply the same operation ownership/session-token check as `op-status`.

Recommended response shapes:

```json
{
  "operation_id": "op_...",
  "status": "cancel_requested",
  "done": false,
  "cancel_requested": true,
  "cancel_teardown_attempted": true,
  "cancel_teardown_status": "shutdown_requested"
}
```

```json
{
  "operation_id": "op_...",
  "status": "already_done",
  "done": true
}
```

```json
{
  "operation_id": "op_...",
  "status": "cancel_failed",
  "done": true,
  "error": "worker_shutdown_failed"
}
```

Add `op-cancel` to the unsupported nested operation command set for `op-start`, alongside `op-start` and `op-status`, so users cannot recursively launch cancellation as a wrapped operation.

Add `op-cancel` to:

- `src/hosting/service/auth.py`
- `src/hosting/service/policy.py`

Add a public wrapper in `src/hosting/engine_host_channel.py`:

```python
def cancel_host_operation(self, *, operation_id: str, reason: str = "") -> Dict[str, Any]:
    ...
```

Add CLI examples for `op-cancel` in `src/hosting/engine_host_cli.py`.

### Load/Connect Cancellation Requirements

For load/connect operations, cancellation is only useful if the daemon can identify the worker to tear down.

Current `connect_from_config()` computes the effective `engine_id` inside the service call. That means the outer operation layer may not know the target process early enough to reliably clean it up.

Before exposing load operations as cancelable, choose one of these approaches:

1. Precompute or require the target `engine_id` before `op-start` creates the operation.
2. Have `connect_from_config()` publish the selected `engine_id` into operation metadata before spawn.
3. Refactor the operation runner so service calls can report lifecycle metadata back to the operation store.

The preferred first implementation is to precompute/pass `engine_id` for load/connect operations. This keeps cancellation simple and gives `op-cancel` a concrete teardown target.

When canceling a load/connect operation:

- Mark the operation as cancel requested.
- Cancel the daemon wrapper task if it is still pending.
- If a target `engine_id` is known, call the host service shutdown path for that engine.
- Record teardown status in the operation snapshot.
- Do not report final `canceled` if the worker shutdown failed.

### Backend Changes

Once daemon `op-cancel` exists, update backend cancel propagation:

- `src/backend/api/services/backend_client_auth.py::cancel_operation_payload` should call the engine host supervisor/channel cancellation method instead of only mutating `host_operation_runtime_bindings`.
- Store enough operation binding metadata for cancellation, including the daemon operation id and target engine id when known.
- Refresh the backend operation projection from daemon `op-status` after cancellation.
- Remove the backend-local operation binding when the daemon operation reaches a terminal state.

Only set load operations to `cancelable: true` after daemon cancellation and teardown behavior are implemented and tested.

Until then, keep load operations `cancelable: false` even if the cancel endpoint exists. This keeps the generic client contract honest.

### Generic Client Behavior

Generic clients must treat operation cancellation as asynchronous and best-effort.

Required client behavior:

- Discover cancel availability from each operation's `cancelable` field.
- Show or enable cancel controls only when `cancelable` is true and the operation is not terminal.
- On cancel, call `POST /api/operations/{operation_id}/cancel`.
- Do not assume the operation is canceled immediately after the POST returns.
- Continue polling `GET /api/operations/{operation_id}` or listening to `GET /api/operations/{operation_id}/events`.
- Treat these statuses as terminal:
  - `completed`
  - `failed`
  - `error`
  - `canceled`
  - `cancel_failed`
- Treat these statuses as non-terminal or pending follow-up:
  - `pending`
  - `running`
  - `in_progress`
  - `cancel_requested`
- If cancel returns `not_cancelable`, keep polling and show the operation as still owned by the daemon.
- If cancel returns `already_done`, refresh status and render the terminal result.
- If cancel returns `cancel_failed`, show the failure and allow the user to inspect or retry cleanup through a separate host/engine shutdown action if available.

Clients should not equate cancellation with rollback. A canceled load may still leave behind partial daemon state or a worker process if teardown failed. The operation status should be the source of truth.

### Testing Plan

Engine host tests:

- Start a fake long-running operation with `op-start`, call `op-cancel`, assert the operation becomes cancel requested or canceled.
- Verify `op-cancel` enforces the same session token ownership check as `op-status`.
- Verify canceling an unknown operation returns `operation_not_found`.
- Verify canceling an already completed operation returns an already-terminal response and does not corrupt the result.
- Verify cancellation records a progress event and cancel metadata in `op-status`.
- Verify shutdown drain still ignores completed/canceled operation tasks.

Load/connect tests:

- Use a fake or monkeypatched `connect_from_config` path that exposes a selected `engine_id`.
- Cancel before spawn and assert no spawn occurs when that phase is controllable.
- Cancel after spawn and assert `shutdown(engine_id)` is invoked.
- Simulate shutdown failure and assert status becomes `cancel_failed` or equivalent with details.

Backend tests:

- Verify `/api/operations/{operation_id}/cancel` calls `cancel_host_operation()` or daemon `op-cancel`.
- Verify backend operation projection reflects daemon cancellation status after refresh.
- Verify load operations remain `cancelable: false` until daemon cancellation is wired.
- After wiring teardown, flip load operations to `cancelable: true` and verify generic client payloads expose that field.

Client/browser tests:

- Verify cancel control is hidden or disabled when `cancelable` is false.
- Verify cancel control posts to the cancel endpoint when `cancelable` is true.
- Verify UI keeps polling after cancel and does not immediately remove the operation.
- Verify `cancel_requested`, `canceled`, and `cancel_failed` render distinctly.

### Recommendation

Expose `op-cancel` as a daemon operation lifecycle API, but document and implement it as best-effort cancellation with process teardown for load/connect workers.

Do not wait for cooperative cancellation inside model loading. That is likely high cost and unreliable across libraries. The practical value comes from giving users a way to stop waiting and tear down a known worker process.

Do not use `op-cancel` as the universal cancellation API for tool calls. Keep tool/runtime cancellation on the existing toolbox and worker execution paths.

The API brings more good than hassle if the contract stays narrow:

- Good: clients get a real escape hatch for long-running load/start operations.
- Good: backend cancel stops being a local-only illusion.
- Good: future daemon operations launched through `op-start` can share a common lifecycle.
- Risk: if clients treat cancel as a hard guarantee, race conditions become user-visible bugs.
- Risk: if operation metadata does not include `engine_id`, load cancellation cannot reliably tear down spawned workers.

The implementation should therefore land in this order:

1. Add daemon `op-cancel`, auth/policy allowlists, channel wrapper, and tests.
2. Add operation metadata needed for load teardown, especially target `engine_id`.
3. Wire backend cancel endpoint to daemon cancellation.
4. Keep load operations `cancelable: false` until teardown behavior is tested.
5. Flip load operations to `cancelable: true` only when cancel reaches the daemon and can tear down a known worker process.
