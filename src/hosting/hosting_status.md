# Hosting Refactor Status

Date: 2026-06-01

This file tracks progress on the hosted sandbox runtime refactoring plan in `src/hosting/hosting_access_plan.md`.

## Current Status

- [x] Initial planning context captured.
- [x] Existing architecture reviewed at a high level.
- [x] Existing `workflow_python_helper` and `workflow_js_helper` placement decided: migrate to compatibility aliases over new workflow runtime APIs.
- [x] Existing `toolbox_executor` placement decided: migrate later onto shared base while preserving toolbox semantics.
- [x] Existing generic/model worker placement decided: remain separate; borrow IPC/streaming ideas only.
- [x] Non-Python worker scope clarified: out of this epic except for external implementation of the selected wire contract.
- [x] Implementation started.
- [x] Tests updated for the first runtime base, Python environment base, pool registry, and workflow Python compatibility facade slices.
- [x] Direct CLI updated for initial `workflow-python-*` commands.
- [x] Docs updated beyond planning/tracking files.

## Active Phase

- [x] Phase 0: Discovery And Tests Baseline.
- [x] Phase 1: Shared Base Contracts And Models.
- [x] Phase 2: Hosted Process Pool Base.
- [x] Phase 3: Python Runtime Environment Base.
- [x] Phase 4: New Workflow Python Worker.
- [ ] Phase 5: Workflow Python Node Profile.

## Progress Log

### 2026-06-01

- Added comprehensive refactoring plan to `hosting_access_plan.md`.
- Added client migration checklist to `HOSTING_CLIENT_BREAKING_CHANGES.md`.
- Seeded this status file.
- Completed Phase 0 test inventory. Existing focused helper coverage is in `tests/test_workflow_python_helper_ipc.py`, `tests/test_workflow_js_helper_ipc.py`, `tests/test_workflow_helper_service.py`, and `tests/test_engine_host_channel.py`.
- Existing sandbox navigation remains in `src/hosting/sandbox/sandbox_test_status.md`; new runtime-base tests should be added beside the helper/service tests rather than replacing the current sandbox suite.
- Started Phase 1 with `hosting.sandbox.runtime_base`: deterministic sandbox policy hashes, runtime/environment key specs, pool keys, worker slot snapshots, request lifecycle records, stream event envelopes, and pool metrics.
- Completed the first internal Phase 2 pool foundation in `hosting.sandbox.runtime_pool`: pool registry, one-worker-per-environment-key scheduling, capacity changes, saturation tracking, request lifetime completion, cancellation accounting, error grouping, and resource rollups. This is not wired into workflow routing yet.
- Completed the first internal Phase 3 Python runtime wrapper in `hosting.sandbox.python_runtime`: workflow-facing environment specs, environment-key identity, realization, prepare/lock/verify install flow, and runtime Python selection backed by the existing toolbox environment manager. This is not exposed through service/channel/CLI yet.
- Started Phase 4 with a compatibility-first `workflow_python` facade. Service, daemon, channel, and direct CLI surfaces now expose environment spec/prepare/lock/verify/install/receipt commands plus helper-profile ensure/execute/resources/capacity/cancel. The backing worker is still the existing Python helper worker; `profile=node` still returns a not-implemented error.
- Extended the Phase 4 facade to annotate helper-backed registrations with workflow runtime/environment metadata and return request lifecycle metrics for sync helper-profile execution.
- Wired the helper-backed `workflow_python(profile=helper)` facade to the internal in-memory pool registry. `ensure`, `execute`, `resources`, `set-capacity`, and `cancel-request` now maintain/report host-side pool capacity, active call counts, recent request lifetime metrics, and cancellation accounting by `environment_key`. This is accounting/scheduling around the existing helper worker, not the final new worker implementation.
- Updated the interactive CLI workflow helper management path so annotated Python helper registrations use the new `workflow-python-*` facade for resources/capacity/cancel and display workflow pool metrics by `environment_key`, while legacy helper and JS helper paths remain compatible.
- Tightened workflow Python facade migration behavior: resources/capacity/cancel can infer `environment_key` from annotated registrations, and tests now prove incompatible sandbox policies derive separate environment keys, engine IDs, and host-side pools.
- Updated `HOSTING_CLIENT_BREAKING_CHANGES.md` to reflect helper-profile workflow Python APIs and metrics that are now available, while keeping node-profile streaming and full helper implementation removal marked pending.
- Added direct CLI compatibility tests for workflow Python facade resource/capacity/cancel commands and updated `sandbox_test_status.md` with the new runtime refactor test navigation.
- Added RBAC/daemon policy support for the new `workflow-python-*` command family, with worker-user control access and diagnostic observe-only coverage.
- Started Phase 5 by adding `hosting.sandbox.workflow_python_contract`: node-profile request normalization, validation, response-envelope fields, stream event names, and a structured not-implemented response. `workflow_python(profile=node)` now returns that stable envelope instead of the older generic profile error; the streaming worker remains pending.
- Started Phase 6 compatibility rewiring: legacy Python helper resources/capacity/cancel methods now preserve old helper results while attaching `environment_key`, `workflow_runtime_kind=workflow_python`, and `workflow_pool` metadata for annotated registrations.
- Completed the interactive CLI ensure action for Python helpers: operators can annotate/use a selected legacy helper through `workflow-python-ensure` and then refresh via environment-keyed workflow resources.
- Started Phase 7 by adding `workflow_js(profile=helper)` compatibility facade surfaces in service, daemon, direct CLI, channel, RBAC, and tests. The JS facade derives environment keys from environment name, profile, Node/runtime identity, dependency hints, and sandbox policy hash, then reports environment-keyed workflow pool metadata.
- Extended Phase 7 compatibility aliases: old JS helper resources/capacity/cancel calls now preserve JS-specific fields while attaching `workflow_runtime_kind=workflow_js`, `environment_key`, and `workflow_pool` metadata for annotated registrations.
- Marked old Python/JS helper command names as compatibility aliases in `HOSTING_CLIENT_BREAKING_CHANGES.md`; new integrations should use `workflow-python-*` and `workflow-js-*`.
- Extended the workflow Python runtime environment wrapper to return a stable `install_status` summary for prepare/lock/verify/install/receipt operations, so workflow callers do not need to parse toolbox metadata directly.
- Completed the tracked Phase 7 facade/alias checklist items; full old JS helper file removal remains a later cleanup phase after dependent migration.
- Added shared pool request-status/progress snapshot plumbing and workflow-named `workflow-python-request-status` / `workflow-js-request-status` surfaces. These report request lifetime metrics plus latest progress once stream/progress events are recorded.
- Rewired old `spawn_workflow_python_helper` service calls to enter through `ensure_workflow_python(profile=helper)`, with raw helper worker spawning kept behind a private service helper for compatibility.
- Tightened workflow Python environment identity so explicit Python runtime executables or runtime hashes contribute to `environment_key`; different Python runtimes no longer share the same host-side pool identity.
- Rewired `EngineHostControlChannel.spawn_workflow_python_helper(...)` to forward to `workflow-python-ensure` while retaining the old typed method signature for dependent callers.
- Rewired direct old `proxy_rpc_call(method=execute_workflow_python_helper)` calls for Python helpers through `execute_workflow_python(profile=helper)` so legacy execution now records workflow pool/request metrics before the raw worker RPC.
- Marked `workflow_python_helper_ipc.py` as a temporary compatibility worker to remove or reduce after dependent callers complete migration to workflow Python facade APIs.
- Reconciled plan tracking for implemented foundations: temporary Python helper compatibility, workflow JS facade, generic/model separation, persisted workflow environment identity, environment-keyed pool registry, one-worker-per-key scheduling, resource reporting, capacity adjustment, and cancellation.
- Added an explicit node-profile artifact-store placeholder in the workflow Python contract envelope so clients can distinguish "no artifacts" from "artifact store not wired yet."
- Centralized the shared stream event type list and cancel control message shape in `hosting.sandbox.runtime_base`, giving future concrete sandboxes one event vocabulary.
- Added internal `HostedProcessSandboxBase` in `hosting.sandbox.process_base` as a non-public composition layer over the pool registry for shared capacity, request status, progress, and cancellation plumbing.
- Added `HostedPythonRuntimeBase` above the process base and made `HostedPythonRuntimeManager` inherit it, preserving the existing workflow environment manager behavior while exposing shared process-pool capabilities.
- Added thin `HostedJsRuntimeBase` above the process base for Node/runtime identity and environment-key derivation, and routed the workflow JS facade environment spec through it.
- Added shared runtime response helpers for registration environment metadata, resource responses, and cancellation results in `hosting.sandbox.runtime_base`.
- Centralized base IPC message family names (`hello`, `rpc_call`, stream open/recv/send/close, `shutdown`) in `hosting.sandbox.runtime_base`.
- Updated `sandbox/SANDBOX_ARCHITECTURE.md` with the new internal runtime bases and the current workflow Python/JS facade status.
- Added `HostedPythonRuntimeManager.gc_runtime_environments(...)` for dry-run or destructive cleanup of unreferenced `<hosting_root>/runtime_envs` entries by environment key/path.
- Added in-memory stream session plumbing to `HostedProcessSandboxBase`: stream open/emit/recv/send-cancel/close now records progress and request lifecycle state through the shared pool registry.
- Added workflow Python stream command surfaces for node-profile rollout (`workflow-python-stream-open/recv/send/close`). Until the real node worker lands, stream-open emits the structured pending-worker error envelope as stream events.
- Documented workflow Python/JS runtime facade commands, environment lifecycle, resource/capacity/status/cancel usage, and node-profile streaming rollout in `HOSTING.md`.
- Updated `sandbox/sandbox_test_status.md` with new process base, JS runtime base, runtime-env GC, workflow stream rollout, and auth/policy test navigation; Phase 0 inventory/characterization checklist is now current.
- Switched workflow Python runtime base to the neutral `RuntimeEnvironmentManager` adapter rather than importing `ToolboxEnvironmentManager` directly.
- Recorded open design decisions in the plan: workflow-named stream commands, helper profile sync-only behavior, one-worker-per-environment-key first pool shape, artifact-store placeholder, recent request retention, and CLI JSON/log summary posture.

## Key Design Decisions So Far

- [x] The shared base should be internal/abstract, not a public sandbox kind.
- [x] Use two main internal layers:
  - `HostedProcessSandboxBase` for process/lifecycle/IPC/pool/metrics/cancel.
  - `HostedPythonRuntimeBase` for Python runtime environments and dependency identity.

- [x] `workflow_python` should be a concrete public kind.
- [x] `workflow_python_helper` should become a compatibility alias for `workflow_python(profile=helper, environment_name=workflow-python-helper)`.
- [x] `workflow_js_helper` should migrate to the shared process/pool base as `workflow_js(profile=helper)`.
- [x] Toolbox should migrate after workflow Python proves the base.
- [x] Generic/model workers remain separate concrete workers.
- [x] Reworked sandboxes should support streaming responses where needed.
- [x] Reworked sandboxes should report latency and concurrency metrics.
- [x] Host should control capacity/concurrency by `environment_key`.
- [x] Host should track request lifetime and cancellation.

## Known Gaps Before Implementation

- [x] Internal hosted process pool abstraction exists in `hosting.sandbox.runtime_pool`.
- [x] First deterministic `environment_key` model exists in `hosting.sandbox.runtime_base`.
- [x] First-class `environment_key` routing exists for helper-profile workflow Python facade calls.
- [x] First-class `environment_key` routing exists for helper-profile workflow JS facade calls.
- [ ] Existing helper pools are tied to helper engine IDs and internal child pools.
- [ ] Existing Python helper only separates hot child checkout by Python executable, not full dependency/policy identity.
- [x] Existing workflow environment management is present mostly through toolbox-shaped APIs.
- [x] Internal workflow-facing Python environment manager exists without toolbox IDs/tool keys in its API.
- [x] Existing helper response shape is narrower than planned workflow node responses.
- [ ] Existing helper streaming support is absent.
- [x] Interactive CLI is still helper-command oriented.
- [x] Direct CLI has initial `workflow-python-*` commands.

## Next Implementation Steps

- [x] Add internal data models for environment keys, pool keys, request lifetime, stream events, and metrics.
- [x] Implement stable environment-key derivation tests before changing worker routing.
- [x] Add internal process pool registry tests for scheduling, saturation, cancellation, metrics, and resource rollups.
- [x] Add internal Python runtime environment tests for workflow spec identity, realization, install plan/lock/verify, and runtime Python selection.
- [x] Draft the new workflow Python API surface in service/channel/CLI for helper-profile compatibility.
- [x] Wire workflow Python facade to the internal pool registry and persist environment metadata on registrations.
- [x] Persist workflow Python environment metadata on helper-backed registrations.
- [x] Wire workflow Python facade to the internal pool registry for host-side scheduling/accounting.
- [x] Add interactive CLI views/actions for workflow runtime pools.
- [x] Keep `HOSTING_CLIENT_BREAKING_CHANGES.md` updated as compatibility shims land.
- [ ] Implement workflow Python node-profile streaming worker.

## Plan Audit: 2026-06-01

Completed implementation phases now cover Phase 0 through Phase 4, Phase 6,
Phase 7, the shared/base pieces of Phase 5, Phase 8 direct/channel/interactive
compatibility except a richer interactive streaming UI, and early/midpoint docs.

Remaining unchecked plan items are intentionally not marked complete:

1. Phase 5 real node-profile execution:
   - current code exposes the node contract and workflow stream commands, but
     the stream currently emits the structured pending-worker error envelope
     rather than executing node-profile workflow code.
   - remaining tests should cover real streamed progress, stdout/stderr
     summaries, result, structured error, timeout, cancel, and metrics once
     the node worker exists.
2. Phase 8 interactive streaming UI:
   - interactive helper management can ensure/show resources/set capacity/cancel
     by environment key, but it does not yet provide a dedicated node stream UI.
3. Phase 9 toolbox migration:
   - toolbox public APIs remain unchanged.
   - toolbox still needs a deliberate lifecycle mapping onto
     `HostedProcessSandboxBase`, parity tests, callback/brokered I/O regression
     coverage after migration, and docs/status updates.
4. Phase 10 cleanup/removal:
   - old Python/JS helper implementations and compatibility fields must stay
     until dependent clients migrate.
   - removal of old CLI branches and duplicate toolbox environment code should
     happen only after that migration.
5. Phase 11 final docs and client action checklist:
   - final public docs and example verification should wait until the node
     worker/toolbox migration/removal decisions are implemented.
   - client checklist items in `HOSTING_CLIENT_BREAKING_CHANGES.md` remain
     unchecked because they track dependent-project migration, not host-side
     implementation completion.
