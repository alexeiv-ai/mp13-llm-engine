# Hosting concurrency implementation status

This file tracks execution of `hosting_access_plan.md` for the dependent
project request: real bounded concurrency for hosted toolbox and Host API
calls.

## Plan slices

- [x] Map the existing production path: client transport, daemon dispatch,
  toolbox runtime/pool, executor IPC, callback relay, and diagnostics.
- [x] Implement atomic bounded toolbox admission with queueing/backpressure,
  cancellation, policy gates, and all-settled per-call diagnostics.
- [x] Make the production control/daemon path concurrent and keep control
  operations responsive while tools execute.
- [x] Add Host API provider/method concurrency policy, bounded execution,
  cancellation/timeout behavior, and thread-safety documentation.
- [x] Expose effective concurrency capability discovery and runtime metrics,
  including logical capacity versus worker process count.
- [x] Add focused and real-path tests, run validation, and check acceptance
  coverage.
- [x] Reply to the dependent project in `HOSTING_CLIENT_BREAKING_CHANGES.md`
  with caller stop/start guidance.
- [x] Create sliced commits for each completed implementation slice.
- [x] Correct parallel-group admission so declared `max_concurrency` is
  enforced rather than accidentally serializing parallel calls in both
  controllers.
- [x] Preserve a stable per-call execution envelope on `ToolCall` objects and
  settle harness transport/queue failures independently.
- [x] Separate internal execution request identity from model tool-call IDs,
  including status/cancellation routing and duplicate-ID coverage.
- [x] Fence cancellation before worker dispatch and report queued cancellation
  as `outcome: "canceled"`.
- [x] Make `toolbox.describe().parallel_execution` complete before execution
  and add dependent-team regression coverage.

## Progress log

### 2026-08-02

- Started from the feature request document; no prior status entries existed.
- Confirmed the current bottlenecks: persistent client connections serialize
  request/response exchanges, while the production daemon already offloads its
  synchronous service dispatcher but needed end-to-end concurrency wiring and
  status coverage. The hosted pool originally failed fast without atomic queue
  admission.
- Implemented dedicated local/SSH toolbox request transports, a thread-safe
  hosted pool with bounded queueing and serial/keyed/exclusive gates, and
  request lifecycle timing/worker/decision diagnostics.
- Added Host API controller admission with provider/method metadata, queue
  backpressure, timeout, cancellation cleanup, and discovery-time runtime
  snapshots. Service-broker methods now publish their concurrency contracts.
- Added explicit `sandbox_recycled` sibling outcomes when coarse toolbox
  cancellation recycles a worker, plus public `toolbox-request-status` control
  access for responsive inspection.
- Addressed dependent-team follow-up: parallel groups now honor declared
  limits, hosted calls carry `ToolCall.execution_envelope`, harness execution
  is all-settled per call, model IDs are separate from lifecycle IDs, queued
  cancellation is fenced and reports `canceled`, and describe metadata is
  complete before execution.
- Added regression coverage for three calls at `max_concurrency=2`, Host API
  peak concurrency two, queue-full propagation, one transport failure beside a
  successful sibling, duplicate model IDs, complete describe metadata, and
  queued-cancellation admission fencing.
- Added focused pool, Host API, daemon-dispatch, toolbox error/cancellation,
  and real local control-path tests. The real-path test exercises actual
  toolbox functions through the daemon and request transport.
- Completed the full validation matrix, checked the acceptance items below,
  and created the sliced commits listed at the end of this file.
- Full dependent-team follow-up validation passed: 280 tests across the
  sandbox process/runtime/pool, toolbox, daemon ACL, service-broker, Host API,
  and hosted chat matrices; all hosting Python files compiled successfully.

## Acceptance coverage

- [x] Actual toolbox calls overlap through local client transport, daemon,
  runtime admission, and executor IPC.
- [x] Daemon synchronous service work is offloaded and concurrent.
- [x] Serial, keyed, and exclusive policy gates are implemented and tested at
  the pool/Host API layers.
- [x] Saturation reports bounded queue/full/timeout outcomes without overbooking.
- [x] Running/queued cancellation is independent; recycled siblings are named.
- [x] Status inspection has a public control-channel path and diagnostics.
- [x] Discovery separates logical capacity, active/queued calls, worker count,
  and execution model.
- [x] Host API overlap, saturation, cancellation, provider identity, and
  thread-safety metadata are covered by focused tests/docs.
- [x] Validation complete: 280 tests passed across the sandbox process,
  runtime/pool, daemon, toolbox, Host API, service-broker, and hosted chat
  matrix; all hosting Python files compiled successfully.
- [x] Sliced commits created: `9c5663d` (pool admission), `68048f8`
  (control transport/daemon), `4023710` (toolbox policy/results), and
  `a4844ca` (Host API concurrency), `79a7d54` (parallel admission gates), and
  `db2a6bf` (stable hosted batch execution). The final documentation commit
  contains this status record and the client reply.
