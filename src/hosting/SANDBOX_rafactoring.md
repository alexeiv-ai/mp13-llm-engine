# Sandbox Event Streaming Implementation Plan

Date: 2026-06-20

This plan is scoped to the event-streaming pillar only. It defines the stream contract, batching, chunking, queue policy, and channel behavior needed before other sandbox pillars can be implemented cleanly.

It intentionally does not implement client-owned host APIs, toolbox-host capabilities, long-lived state recovery, backend/workflow state stores, or card action manifests. Those pillars may depend on stream fields such as `instance_id`, correlation ids, terminal events, and audit frames, but their implementation belongs in `hosting_access_plan.md`.

## Current Baseline

- [ ] Keep useful existing pieces:
  - Python and JavaScript node runtimes already forward live progress.
  - Python and JavaScript node runtimes already route `host_call` and `host_response` frames over child IPC.
  - Hosted process streams already retain bounded events and expose dropped-event counts.
  - Generic engine worker IPC already has bounded stream queues and max concurrent streams.
  - Daemon local IPC accepts multiple clients, though each `EngineHostControlChannel` serializes requests on one connection.
- [ ] Replace the old public event shape:
  - old: one full object per event with repeated `type`, `request_id`, `sequence`, `timestamp`, and `payload`
  - new: one batch with shared context plus independently valid JSON frame objects
- [x] Remove the old fallback where unknown event types degrade to `log`.

## Why Streaming First

- [ ] Implement the event-stream protocol first, but keep the slice narrow.
- [ ] Reason: later pillars need stable observation and lifecycle semantics:
  - client-owned host APIs need correlated `host_call`, `host_response`, denial, and approval observations
  - long-lived instances need `instance_id` in observable frames
  - state recovery needs audit/checkpoint notices, not inline state transport
  - card actions need progress/result/action notices without a new stream channel
- [ ] Constraint: this slice only reserves stream semantics used by later pillars. It does not implement host capability ownership, state stores, approval routing, or action manifests.

## Public API Strategy

- [ ] Hosting client libraries must provide helpers that hide the wire protocol.
- [ ] Dependent clients should consume typed/normalized events from helpers, not parse raw frames by default.
- [ ] Raw batch access remains available for diagnostics and low-level integrations.
- [ ] Helper APIs should expose a simple loss policy:
  - `on_loss="raise"`: stop iteration with a stream-loss exception
  - `on_loss="mark"`: yield a loss marker and continue with available events
- [ ] Default helper behavior should be `on_loss="mark"` for interactive clients and `on_loss="raise"` for deterministic automation/test clients.
- [ ] Helper output should follow the `mp13_config.py` streaming style: named fields, optional values, enum-like status/kind strings, and validation instead of positional payload decoding.

## Batch Contract

### Design Goal

- [ ] Optimize compactness by removing repeated batch context, not by making field names opaque.
- [ ] Use frame objects with optional key/value fields, similar to `InferenceResponse` chunks in `mp13_config.py`.
- [ ] Keep every frame and every output chunk valid JSON on its own.
- [ ] Avoid a generic `payload` wrapper.
- [ ] Infer retention behavior from `kind`; do not repeat lane/policy per frame.

### Batch Shape

- [ ] Use this shape for event subscription reads:

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
    {"dt_ms": 8, "kind": "stdout", "text": "Installing package\n", "boundary": true},
    {"dt_ms": 11, "kind": "done", "status": "ok"}
  ],
  "more": true
}
```

- [ ] `context` is shared identity for all frames in the batch.
- [ ] `base.sequence` plus frame index gives the default sequence.
- [ ] `base.timestamp_ms + frame.dt_ms` gives the default frame timestamp.
- [ ] `loss` reports dropped frame counts since the previous delivered batch.
- [ ] `frames` contains valid JSON objects, not positional arrays.
- [ ] Sparse per-frame overrides may include:
  - `sequence`
  - `timestamp_ms`
  - `origin`
  - `source`
  - `visibility`
  - `correlation_id`
  - `redacted`

## Event Kind Registry

- [x] Define kind policy in one registry used by runtime, service, and helpers.

| Kind | Lane | Queue Decision |
| --- | --- | --- |
| `started` | event | keep first queued; drop later duplicates |
| `heartbeat` | event | latest replaces queued |
| `progress` | event | latest replaces queued per `key`; if no key, latest replaces queued |
| `stdout` | output | keep first queued; drop later until drained |
| `stderr` | output | keep first queued; drop later until drained |
| `log` | output | keep first queued per logger/level window; drop later until drained |
| `metric` | event | latest replaces queued per `name` |
| `artifact` | event | keep first queued; later loss is acceptable because refs are queryable |
| `host_call` | control | non-droppable |
| `host_response` | control | non-droppable |
| `approval` | audit | non-droppable if decision-bearing; otherwise keep first queued |
| `result` | control | non-droppable terminal |
| `error` | control | non-droppable terminal |
| `canceled` | control | non-droppable terminal |
| `done` | control | non-droppable final |
| `state_notice` | audit | latest replaces queued per scope/partition |
| `action_notice` | event | latest replaces queued per action id |

- [ ] Reserve `state_notice` and `action_notice` only as stream hooks. Do not implement state stores or action manifests in this pillar.
- [ ] When dropping is allowed, helpers only need to surface that loss happened. They do not need to infer what was dropped.

## Frame Payload Shape

### General Rules

- [ ] Prefer named optional fields over rigid arrays for helper-facing and wire-visible frame payloads.
- [ ] Use stable enum-like strings for `kind`, `status`, and `level`.
- [ ] Use optional fields with defaults and validation, following the `mp13_config.py` pattern.
- [ ] Avoid repeated values already present in `context`.
- [ ] Avoid inlining large artifacts, large state, or large host-call bodies unless policy explicitly allows it.

### Common Fields

- [ ] All frames may include:
  - `dt_ms`
  - `kind`
  - `key`
  - `message`
  - `status`
  - `correlation_id`
  - `scope`
  - `operation`
  - `ref`
  - `expected_bytes`
  - `offset`
  - `length`
- [ ] Output frames may include:
  - `text`
  - `data_b64`
  - `encoding`
  - `boundary`
  - `final`
  - `truncated`
  - `dropped_before`
  - `ack_id`
  - `stream_id`
- [ ] Progress frames may include:
  - `key`
  - `pct`
  - `current`
  - `total`
  - `message`
  - producer-specific optional fields
- [ ] Error frames may include:
  - `reason`
  - `message`
  - `error_type`
  - `traceback_summary`
- [ ] Host-call observation frames may include:
  - `call_id`
  - `method`
  - `provider_id`
  - `capability_id`
  - `arguments_ref`
  - `result_ref`
  - `error`

### Chunking

- [ ] One chunk frame must always be a valid JSON object.
- [ ] Do not split a JSON string or object across frames in a way that requires clients to concatenate partial JSON to parse it.
- [ ] For text output, use `text`.
- [ ] For binary output, use `data_b64`.
- [ ] For structured data too large for one frame, store it as an artifact/ref and stream a notice frame with the ref.
- [ ] When total output size is known, communicate it before or on the first chunk using `expected_bytes`.
- [ ] Chunks should carry `offset` and `length` when the output stream is ordered and byte-addressable.
- [ ] Emitters control chunk size and boundaries:
  - host policy declares maximum bytes per chunk
  - emitter chooses natural chunk boundaries below that maximum
  - runtime splits only as a safety fallback
- [ ] Natural chunk boundaries:
  - stdout/stderr prefer line boundaries
  - log prefers one record per chunk
  - generated/structured output prefers semantic chunks
- [ ] No uniform chunk size requirement. Natural alignment is more important.
- [ ] Chunk maximum should align with producer write blob sizes where known. Do not choose a tiny default that forces artificial fragmentation.
- [ ] Proposed initial maximum chunk size: 64 KiB, with producer-specific override up to a host policy cap.
- [ ] Do not define a maximum total stream size in the event contract. Artifact/result streams may be much larger than memory-safe event retention limits.

## Backpressure And Loss

### Decision

- [ ] Use hybrid policy:
  - control: hard backpressure and fail-fast if undeliverable
  - ack-capable output/request streams: async backpressure controlled by acknowledgements
  - non-ack output notifications: bounded loss, preserving earliest queued chunks
  - event: kind-specific replacement or first-kept loss
  - audit: durable record for security-sensitive events, bounded live stream
- [ ] Do not block worker execution indefinitely on stdout/stderr/log delivery.
- [ ] Do not drop terminal/control frames. If they cannot be delivered, terminate the request and surface a runtime error.
- [ ] Prefer ack-backed async streams whenever output is semantically part of the request result or has known total size.
- [ ] Use dropping only for observability output where partial data is acceptable, such as noisy stdout/log tails.

### Loss Semantics For Helpers

- [ ] Helpers expose `loss_detected` on batches and normalized events.
- [ ] Helpers support:
  - `on_loss="raise"` for strict clients
  - `on_loss="mark"` for clients that can report loss and continue
- [ ] In `mark` mode, helpers yield a synthetic `stream_loss` event with counts by class, then continue.
- [ ] Clients should not need to implement per-kind loss reconstruction.

### Output Policy

- [ ] Default output target chunk size should follow producer write blob size when known.
- [ ] Default maximum chunk size: 64 KiB.
- [ ] Do not set a default maximum total stream size.
- [ ] Define a minimum acceptance contract instead:
  - clients/helpers that open an ack-backed stream must accept at least the negotiated initial credit window
  - proposed minimum initial accepted credit: 1 MiB per stream
  - clients that cannot accept the minimum should not open the stream
  - producers may stream beyond the initial credit only as acknowledgements grant more credit
- [ ] For ack-capable output/request streams:
  - require a client accept/open signal before producer sends large payload chunks
  - include accepted initial credit and optional `expected_bytes` in the stream-open/accept handshake
  - include `ack_id` on chunks that consume stream credit
  - advance producer credit only after receiver acknowledgement
  - keep bounded in-flight bytes per stream
  - pause async producers when credit is exhausted
  - fail the stream on acknowledgement timeout
  - support a client close/abandon signal that tells the producer to stop and receive a stream-abandoned error
- [ ] For known-size streams:
  - emit `expected_bytes` before or on the first chunk
  - emit `offset` and `length` on byte-addressable chunks
  - allow helpers to report progress and detect incomplete delivery
- [ ] When output queue is full:
  - keep first queued chunks
  - drop later output chunks until capacity is available
  - increment `loss.output`
  - set `dropped_before=true` on the next delivered output frame for that stream/kind
- [ ] Dropping applies only to non-ack observability output. Do not silently drop ack-backed request stream chunks.
- [ ] Preserve a bounded terminal output summary separately from live output for post-run diagnostics.

### Event Policy

- [ ] `heartbeat`: latest replaces queued.
- [ ] `progress`: latest replaces queued by `key`.
- [ ] `metric`: latest replaces queued by `name`.
- [ ] `started`, `artifact`, and unkeyed bounded events: keep first queued, drop later.
- [ ] Increment `loss.event` for dropped bounded events.

### Audit Policy

- [ ] Decision-bearing approval/audit frames are effectively control-critical and must not be silently dropped.
- [ ] Non-decision audit frames are bounded live events.
- [ ] Persist security-sensitive approval, permission, and denial records outside the live stream.
- [ ] Increment `loss.audit` when live non-decision audit frames are dropped.

## Channel Plan

### Sandbox Worker IPC

- [ ] Keep one physical child IPC connection for the first implementation.
- [ ] Add logical priority queues in the parent runtime:
  - control first
  - audit second
  - event third
  - output last
- [ ] Replace unbounded `_events` queues in Python and JavaScript node runtimes with lane-aware bounded queues.
- [ ] Continue using the existing send lock for host-to-worker writes.
- [ ] Prioritize host-to-worker `host_response`, cancel, and shutdown over optional sends.
- [ ] Add live stdout/stderr capture through emitter-controlled valid-JSON chunk frames.
- [ ] Revisit separate physical control/event IPC only if measurements show logical priority queues are insufficient.

### Daemon RPC

- [ ] Add a separate event subscription path before high-volume live output is enabled.
- [ ] Keep existing command RPC for cancel, status, stream close, approval response, and future host capability registration.
- [ ] Do not require full multiplexed request/response on one daemon connection for the first streaming slice.
- [ ] Require client helpers to maintain one control channel and one event subscription channel internally.
- [ ] Keep `workflow-*-stream-recv` as an implementation stepping stone only while the event subscription path is introduced.
- [ ] Do not restrict the first subscription path to local IPC only unless implementation proves SSH relay cannot carry the same batch/ack protocol.
- [ ] Treat SSH support as the same protocol over a different transport. The unresolved work is transport framing and timeout behavior, not the event model.

### Client Stream Signals

- [ ] Reserve client-to-host control messages for stream flow control:
  - `stream_accept`: client/helper is willing to receive the stream and grants initial credit
  - `stream_ack`: client/helper has durably consumed a chunk or byte range and grants more credit
  - `stream_close`: client/helper intentionally abandons the stream
- [ ] `stream_close` should propagate to the producer as a stream-abandoned error, not as silent cancellation.
- [ ] `stream_accept` should include at least:
  - `stream_id`
  - accepted initial credit bytes
  - optional max chunk size preference
  - optional rejection reason
- [ ] `stream_ack` should include at least:
  - `stream_id`
  - `ack_id` or byte range
  - additional credit bytes, if using credit replenishment

## Cross-Pillar Constraints

- [ ] Host API pillar may add providers, but stream observations should use `host_call` and `host_response`.
- [ ] Approval pillar may add routing, but observations should use `approval`; actual decisions travel over control-priority RPC.
- [ ] State pillar may add stores and recovery, but streaming should only emit `state_notice` summaries or refs.
- [ ] Long-lived instance pillar may add routable workers, but this pillar already reserves `context.instance_id`.
- [ ] Card action pillar may add manifests and dispatch, but streaming should only emit `action_notice` summaries.

## Cross-Pillar Fit Review

- [ ] Host Capability Protocol fit: good.
  - Reusable first-pillar code: event kind registry, control lane, non-droppable delivery, correlation handling, helper normalization.
  - Required reserved fields: `call_id`, `method`, `provider_id`, `capability_id`, `arguments_ref`, `result_ref`, `error`, `correlation_id`.
  - Constraint: host-call request/response execution must remain a control RPC concern; stream frames are observations and audit, not the callable transport itself.
- [ ] Permission and approval fit: good with one constraint.
  - Reusable first-pillar code: audit lane, durable-audit hook, non-droppable decision-bearing event policy, loss handling.
  - Required reserved fields: `approval_id` or `correlation_id`, `operation`, `scope`, `status`, `message`, `ref`.
  - Constraint: user decisions must not be sent through droppable event frames; events report requested/resolved/denied status.
- [ ] Long-lived routable instance fit: good.
  - Reusable first-pillar code: `context.instance_id`, event subscription channel, helper event normalization, terminal semantics.
  - Required reserved fields: `context.instance_id`, optional `source`, optional `correlation_id`.
  - Constraint: instance routing and lifetime management remain outside the stream contract.
- [ ] State and recovery fit: adequate if state payloads stay by reference.
  - Reusable first-pillar code: `state_notice` kind, audit lane, latest-replaces queue policy per scope/partition, helper loss semantics.
  - Required reserved fields: `scope`, `operation`, `ref`, `status`, `message`; state-specific frames may add `partition` and `version`.
  - Constraint: the stream must not become the state store. Large state and recovery snapshots must be stored separately and referenced by `ref`.
- [ ] Card action fit: adequate.
  - Reusable first-pillar code: `action_notice` kind, event lane, latest-replaces queue policy per action id, `correlation_id` for invocation/result.
  - Required reserved fields: `operation`, `status`, `message`, `ref`; action-specific frames may add `action_id` and `card_id`.
  - Constraint: action manifests and action invocation RPC remain outside the stream contract.
- [ ] Harness discovery fit: partial.
  - Reusable first-pillar code: event kind registry and helper models can be advertised by discovery.
  - Required follow-up outside this pillar: a discovery response that says which event kinds a runtime can emit live, synthesize, or only expose terminally.

Review result: the object-frame shape fits later pillars if the first implementation keeps the shared optional fields above and exposes the event kind registry as reusable code. The main risk is letting later pillars inline large or authority-bearing payloads into stream events. Use refs and control RPC for those cases.

## Implementation Checklist

- [x] 1. Define Pydantic-style stream batch and event frame models.
- [x] 2. Update `HOSTING_CLIENT_BREAKING_CHANGES.md` to direct clients to helper APIs first.
- [ ] 3. Replace `HostedStreamEvent.to_dict()` with a frame/batch builder that validates event kinds.
- [x] 4. Add event kind registry with lane, replacement key, and loss policy.
- [x] 5. Add lane-aware queues and loss counters to hosted process streams.
- [x] 6. Implement helper-side `on_loss="raise"|"mark"` behavior.
- [x] 7. Convert service-generated lifecycle events to frames.
- [ ] 8. Convert Python and JavaScript node live `progress` to frames.
- [ ] 9. Convert Python and JavaScript node `host_call` observations to control frames.
- [ ] 10. Add live stdout/stderr/log valid-JSON chunk emitters.
- [ ] 11. Add accept/ack/close credit support for output streams that require complete delivery.
- [ ] 12. Add daemon event subscription path separate from command RPC.
- [ ] 13. Add stream decoder, ordering, replacement, ack, and loss tests.

## Test Checklist

- [x] Batch decoder expands timestamps and sequences from `base`.
- [x] Unknown stream version fails validation.
- [x] Unknown event kind fails validation.
- [x] Control frames bypass output backlog.
- [x] Terminal frames are delivered when output queue is full.
- [x] Output frames are always valid JSON objects.
- [ ] Known-size output streams report `expected_bytes`.
- [ ] Byte-addressable chunks report `offset` and `length`.
- [ ] Ack-backed streams require client accept before large payload delivery.
- [ ] Ack-backed streams pause when credit is exhausted and resume after acknowledgement.
- [ ] Client close propagates a stream-abandoned error to the producer.
- [x] Output loss yields either a helper exception or a `stream_loss` marker depending on helper policy.
- [x] Latest-replaces policy works for heartbeat/progress/metric.
- [x] First-kept/drop-later policy works for stdout/stderr/log/artifact.
- [ ] Emitter-controlled chunk size is respected below policy maximum.
- [ ] Runtime splits or rejects chunks above policy maximum.
- [ ] Daemon event subscription does not block cancel/status on the control channel.
- [x] Request-scoped batches include `request_id` and `instance_id`.
- [ ] Instance-scoped batches can omit `request_id` without breaking decoding.

## Remaining Decisions

- [ ] Confirm default output target chunk size: producer write size when known.
- [ ] Confirm default maximum chunk size: proposed 64 KiB.
- [ ] Confirm minimum initial accepted credit per ack-backed stream: proposed 1 MiB.
- [ ] Confirm helper defaults: proposed `mark` for interactive clients and `raise` for deterministic automation.
- [ ] Confirm ack timeout and in-flight byte limits per stream.
- [ ] Confirm whether existing SSH relay framing can carry event subscription batches and acknowledgements without a separate implementation phase.
