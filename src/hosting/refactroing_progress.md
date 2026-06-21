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
