# Hosting Client Breaking Changes

Date: 2026-06-20

This document tracks dependent-client changes required by the sandbox event-streaming refactor. The hosting library should provide API helpers that hide the raw wire protocol for normal clients.

## Required Client Change

- [ ] Stop parsing raw stream event dictionaries directly.
- [ ] Use the hosting library stream helper/iterator once it is introduced.
- [ ] Choose a loss policy when opening a stream:
  - `on_loss="raise"`: stop with an exception when any lossy event class drops frames
  - `on_loss="mark"`: receive a `stream_loss` marker and continue with available events
- [ ] Treat helper-returned events as the public API. Raw batches are for diagnostics and low-level integrations.

Current low-level helper entry points:

- `HostedStreamBatch.from_dict(...)`: validates the raw batch version and event kinds.
- `HostedStreamBatch.expanded_frames()`: expands shared context, sequence, and timestamp values.
- `hosted_stream_normalize_batch(..., on_loss="mark")`: returns normalized events and inserts a `stream_loss` marker when loss is reported.
- `hosted_stream_normalize_batch(..., on_loss="raise")`: raises `HostedStreamLossError` when loss is reported.

Transitional stream receive responses may expose all three shapes:

- `normalized_events`: preferred helper-facing events.
- `batch`: raw compact batch for diagnostics and low-level integrations.
- `events`: legacy rows kept only while service callers migrate.

Event subscription commands:

- Use `workflow-python-event-subscribe` instead of `workflow-python-stream-recv` for the event-read path.
- Use `workflow-js-event-subscribe` instead of `workflow-js-stream-recv` for the event-read path.
- Keep `workflow-*-stream-send`, `workflow-*-stream-close`, cancel, and status calls on the control path.

## Helper-Facing Event Model

- [ ] Expect normalized events with named optional fields, similar to the `InferenceResponse` pattern in `mp13_config.py`.
- [ ] Expect enum-like string fields such as `kind`, `status`, and `level`.
- [ ] Expect optional fields to vary by kind.
- [ ] Do not expect a generic `payload` wrapper.
- [ ] Do not expect positional payload arrays.

Example normalized event:

```json
{
  "kind": "stdout",
  "request_id": "request-id",
  "instance_id": "instance-id",
  "sequence": 101,
  "timestamp_ms": 1781913600008,
  "text": "Installing package\n",
  "boundary": true,
  "dropped_before": false,
  "expected_bytes": 1048576,
  "offset": 0,
  "length": 1024
}
```

## Loss Handling

- [ ] Clients should implement one of two simple behaviors:
  - bail out on loss with `on_loss="raise"`
  - report loss and continue with `on_loss="mark"`
- [ ] Clients do not need to reconstruct missing events.
- [ ] In `mark` mode, handle a normalized event like:

```json
{
  "kind": "stream_loss",
  "request_id": "request-id",
  "instance_id": "instance-id",
  "loss": {
    "output": 12,
    "event": 1,
    "audit": 0
  }
}
```

- [ ] Treat output after loss as partial. It is valid to display it with a loss marker.
- [ ] For deterministic workflows, use `raise`.
- [ ] For ack-backed request/output streams, loss should surface as stream failure rather than silent partial output.

## Event Queue Semantics

- [ ] Do not assume every produced event is delivered.
- [ ] The runtime keeps or replaces events by kind:
  - `heartbeat`: latest replaces queued
  - `progress`: latest replaces queued per key
  - `metric`: latest replaces queued per name
  - `state_notice`: latest replaces queued per scope/partition
  - `action_notice`: latest replaces queued per action id
  - `stdout`, `stderr`, `log`: first queued is kept; later chunks are dropped until drained
  - `started`, `artifact`: first queued is kept; later duplicates may be dropped
  - `host_call`, `host_response`, `result`, `error`, `canceled`, `done`: non-droppable
- [ ] Terminal control events are never intentionally dropped.

## Output Chunks

- [ ] Stop expecting stdout, stderr, and logs to arrive only as terminal text blobs.
- [ ] Output chunk events are valid JSON objects.
- [ ] Text output uses `text`.
- [ ] Binary output uses `data_b64`.
- [ ] Chunks may include:
  - `encoding`
  - `boundary`
  - `final`
  - `truncated`
  - `dropped_before`
  - `expected_bytes`
  - `offset`
  - `length`
  - `ack_id`
  - `stream_id`
- [ ] Do not assume uniform chunk size.
- [ ] Emitters choose chunk boundaries; clients should use `boundary` only as display help.
- [ ] If `expected_bytes` is present, helpers should expose total size/progress and detect incomplete streams.
- [ ] Hosting helpers should accept and acknowledge ack-backed streams; normal clients should not manually implement ack handling.
- [ ] Helpers should expose a close/abandon operation for streams the client no longer wants to receive.
- [ ] Client close should propagate an error to the producer instead of silently discarding the stream.
- [ ] The stream protocol defines a minimum accepted credit window, not a maximum total stream size. Large artifacts/results may exceed the initial window and continue as helpers acknowledge chunks.
- [ ] Dropping applies to non-ack observability output. Ack-backed streams should pause/resume through helper-managed backpressure instead of dropping chunks.

## Raw Batch Format

Most clients should not consume raw batches. Low-level clients may use the raw format:

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
    {"dt_ms": 8, "kind": "stdout", "text": "Installing package\n", "boundary": true, "offset": 0, "length": 19},
    {"dt_ms": 11, "kind": "done", "status": "ok"}
  ],
  "more": true
}
```

- [ ] Compute default sequence from `base.sequence + frame_index`.
- [ ] Compute default timestamp from `base.timestamp_ms + frame.dt_ms`.
- [ ] Treat each frame as a valid JSON object.

## Control And Event Channels

- [ ] Stop using one daemon control channel for both long-poll streaming and responsive control commands.
- [ ] Let the hosting library helper own the channel split internally.
- [ ] Low-level clients should maintain:
  - one control channel for cancel, status, close, approval response, and future host capability registration
  - one event subscription channel for event batches
- [ ] Low-level clients that bypass helpers must implement stream flow-control messages:
  - `stream_accept`
  - `stream_ack`
  - `stream_close`
- [ ] Treat existing `workflow-*-stream-recv` command polling as transitional.
- [ ] SSH relay is not a different stream contract. If a helper supports remote targets, it should expose the same stream helper semantics and handle relay framing/timeouts internally.

## Instance Identity

- [ ] Preserve `instance_id` from helper events.
- [ ] Do not assume every stream is request-scoped. Future instance-scoped streams may omit `request_id`.
- [ ] Future cancel/status/state/action APIs may require both `instance_id` and `request_id`.

## Removed Compatibility Assumptions

- [ ] No fallback for old `type`/`payload` event objects.
- [ ] No fallback for unknown event types degrading to `log`.
- [ ] No fallback for terminal stdout/stderr as the only output delivery mode.
- [ ] No fallback for single-control-channel long-poll streaming.
- [ ] No fallback for positional payload arrays from earlier draft docs.
