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
