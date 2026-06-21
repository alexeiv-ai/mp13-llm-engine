# Sandbox Refactoring Progress

Date: 2026-06-20

## Slice 1: Stream Contract Foundation

- [x] Added the stream contract version constant.
- [x] Added the event kind registry with lane, queue decision, replacement fields, terminal/final flags, and decision-bearing metadata.
- [x] Added compact stream context, loss, frame, and batch models.
- [x] Added batch parsing, strict version validation, strict event-kind validation, and expanded-frame normalization.
- [x] Added focused tests for registry policy, compact batch shape, timestamp/sequence expansion, version rejection, and unknown-kind rejection.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`

## Next Slice

- Replace legacy `HostedStreamEvent.to_dict()` fallback behavior with strict frame/batch construction.
- Add lane-aware queue policy to hosted process streams while preserving existing request lifecycle accounting.

## Slice 2: Strict Legacy Event Conversion

- [x] Removed the legacy behavior that degraded unknown stream event kinds to `log`.
- [x] Added `HostedStreamEvent.to_frame()` for flattening legacy payloads into the compact frame model.
- [x] Added `HostedStreamEvent.to_batch()` for one-event batch construction with shared stream/request/instance context.
- [x] Updated request lifecycle progress tracking so it can read either legacy `type`/`payload` rows or flattened frame rows.
- [x] Added focused tests for strict event validation and frame/batch conversion.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`

## Next Slice

- Move hosted process stream sessions from one FIFO event deque to lane-aware queues using the stream kind registry.
- Return compact batches from `stream_recv` while keeping request status/progress accounting coherent during migration.

## Slice 3: Hosted Process Lane Queues

- [x] Replaced the hosted process stream FIFO retention queue with lane-aware retained queues.
- [x] Added lane-specific pending loss counters and batch loss reporting.
- [x] Added compact `batch` responses beside transitional legacy `events` responses.
- [x] Added latest-replaces behavior for progress-style events.
- [x] Added first-kept/drop-later behavior for output-style events under retention pressure.
- [x] Added control-priority selection when receive limits or backlog pressure would otherwise hide control events.
- [x] Preserved chronological delivery when a receive can drain all retained events.
- [x] Prevented artifact payload metadata such as `kind` or `type` from overriding the event kind during frame conversion.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`
- `pytest tests/test_workflow_helper_service.py -k "stream"`

## Next Slice

- Introduce helper-side normalized stream iteration with `on_loss="raise"|"mark"`.
- Start moving service/runtime stream consumers from transitional `events` to `batch` helpers.

## Slice 4: Stream Batch Normalization Helper

- [x] Added `HostedStreamLossError` for strict helper-side loss handling.
- [x] Added `hosted_stream_normalize_batch(..., on_loss="mark")` to emit a synthetic `stream_loss` marker before normalized frames.
- [x] Added `hosted_stream_normalize_batch(..., on_loss="raise")` to fail deterministic clients on reported loss.
- [x] Added low-level helper guidance to `HOSTING_CLIENT_BREAKING_CHANGES.md`.
- [x] Added tests for marker mode, raise mode, invalid loss policy, and expanded request/instance context.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`
- `pytest tests/test_workflow_helper_service.py -k "stream"`

## Next Slice

- Convert service-generated lifecycle stream responses to rely on batch/helper normalization instead of transitional `events` lists.
- Start routing Python and JavaScript node live progress through frame-first emitters.

## Slice 5: Normalized Process Stream Responses

- [x] Added `normalized_events` to hosted process stream receive responses.
- [x] Kept `batch` available for diagnostics and low-level clients.
- [x] Kept transitional `events` rows so existing service call sites can migrate incrementally.
- [x] Covered normal and lossy normalized response behavior in process stream tests.
- [x] Documented the transitional response shapes for dependent clients.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`
- `pytest tests/test_workflow_helper_service.py -k "stream"`

## Next Slice

- Route Python and JavaScript node live progress through frame-first emitters instead of legacy event payloads.
- Convert host-call observation events to control frames.

## Slice 6: Node Progress And Host-Call Frames

- [x] Preserved Python node live progress as `progress` frames through process stream batch conversion.
- [x] Preserved JavaScript node live progress as `progress` frames through process stream batch conversion.
- [x] Changed JavaScript node `host_call` observations from `log` events to `host_call` control-lane events.
- [x] Added `host_call_id` to `call_id` aliasing during frame conversion for helper-facing consistency.
- [x] Covered JavaScript stream host-call observations via normalized events.

Note: this slice converts node observations at the hosted process stream boundary. A later IPC slice can make child runtimes emit frame objects directly if needed.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`
- `pytest tests/test_workflow_helper_service.py -k "stream"`

## Next Slice

- Add live stdout/stderr/log chunk metadata for valid JSON output frames, including `boundary`, `offset`, and `length` where available.

## Slice 7: Output Chunk Metadata

- [x] Added UTF-8 `encoding`, byte `offset`, byte `length`, and `boundary` defaults for text stdout/stderr/log events.
- [x] Tracked offsets independently per output kind within a process stream session.
- [x] Preserved caller-provided output metadata when present.
- [x] Added process stream tests for stdout, stderr, and log chunk metadata.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`
- `pytest tests/test_workflow_helper_service.py -k "stream"`

## Next Slice

- Design and implement accept/ack/close credit support for complete-delivery output streams.

## Slice 9: Ack-Backed Output Credit

- [x] Added `stream_accept` handling with initial credit and optional max chunk size.
- [x] Added `stream_ack` handling with consumed-byte accounting and credit replenishment.
- [x] Added `stream_close` handling for client abandon.
- [x] Added fail-fast behavior for ack-backed output before accept, when credit is exhausted, and after client abandon.
- [x] Added focused process stream tests for accept-before-output, pause/resume by credit, and stream-abandoned producer errors.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`
- `pytest tests/test_workflow_helper_service.py -k "stream"`

## Next Slice

- Add daemon event subscription path separate from command RPC.

## Slice 8: Frame-First Process Retention

- [x] Changed hosted process streams to retain expanded frame rows internally.
- [x] Changed process stream event creation to use `HostedStreamEvent.to_batch(...).expanded_frames()` instead of `HostedStreamEvent.to_dict()`.
- [x] Changed pool stream-event recording to use frame expansion for `HostedStreamEvent` inputs.
- [x] Kept transitional legacy `events` synthesized at receive boundaries while service callers migrate.
- [x] Preserved lifecycle progress/status accounting with frame rows.
- [x] Updated stream tests to assert frame-shaped `stream_emit` output.

Note: `HostedStreamEvent.to_dict()` still exists as a transitional legacy serializer, but the hosted process stream path no longer depends on it.

## Verification

- `pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py`
- `pytest tests/test_workflow_helper_service.py -k "stream"`

## Next Slice

- Design and implement accept/ack/close credit support for complete-delivery output streams.
