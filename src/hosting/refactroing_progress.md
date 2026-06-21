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
