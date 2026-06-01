# Hosted Sandbox Runtime Refactoring Plan

Date: 2026-06-01

Scope: refactor hosted sandbox/runtime implementation so workflow Python, workflow JS, and toolbox workers can share a common process/runtime foundation while keeping distinct public contracts. This plan is intentionally detailed so another assistant can resume implementation at any phase.

## Background

Current hosted sandbox implementation already has several reusable pieces:

- Shared sandbox policy and launch path in `src/hosting/sandbox/policy.py` and `src/hosting/sandbox/launcher.py`.
- Shared worker registration fields in `src/hosting/service/engines.py`: `executor_kind`, `worker_profile_class`, `sandbox_policy`, `sandbox_runtime`, `environment`, `capabilities`, IPC family/address, and worker auth token.
- Generic worker IPC ideas in `src/hosting/engine_worker_ipc.py` and `src/hosting/sandbox/GENERIC_WORKER.md`: `hello`, `rpc_call`, `stream_open`, `stream_recv`, `stream_send`, `stream_close`, and `shutdown`.
- Toolbox environment and install plumbing in `src/hosting/toolbox/environment.py` and `src/hosting/service/toolbox_env.py`: environment specs, `runtime_envs`, venv keys, install plan/lock/verify/execute, receipt verification, and GC.
- Workflow helper workers in `src/hosting/workflow_python_helper_ipc.py` and `src/hosting/workflow_js_helper_ipc.py`: source-in JSON-out execution, hash verification, hot child process pools, cancellation, capacity, and resource reporting.

The requested direction is a generalized hosted workflow runtime:

- `workflow_python` with `profile=helper|node`.
- Existing `workflow_python_helper` becomes a compatibility alias for `workflow_python(profile=helper, environment_name=workflow-python-helper)`.
- `workflow_js_helper` should migrate to the same process/pool foundation as `workflow_js(profile=helper)`.
- Toolbox should later move onto the same base process/runtime abstractions while preserving toolbox-specific public semantics.
- Pools must be routed by `environment_key`, derived from environment name, runtime/profile, dependency/import intent, dependency lock identity where available, and sandbox policy hash.
- Different dependency or policy sets must not share a worker process pool.
- Reworked sandboxes must support streaming responses, latency/concurrency metrics, request lifetime tracking, and cancellation.
- Non-Python workers are out of this epic except that they may implement the selected wire contract externally.

Explicitly out of scope for this epic unless separately requested:

- Hard memory limit enforcement.
- Strong OS-level network/filesystem policy across all platforms.
- A public non-Python worker SDK.
- Making generic/model workers inherit workflow or toolbox semantics.

## Target Architecture

### Shared Layers

- [x] Define `HostedProcessSandboxBase` as an internal abstraction, not a public sandbox kind.
  - Owns process launch request construction, IPC metadata, worker auth token, persisted registration fields, lifecycle, shutdown, readiness probing, resource snapshots, logs, request tracking, cancellation plumbing, and pool scheduling.
  - Is language-neutral in design.
  - Initially uses existing hosting IPC/proxy commands.
  - Does not define workflow, toolbox, model, or tool-call semantics.

- [x] Define `HostedPythonRuntimeBase` above the process base.
  - Owns Python runtime identity, venv/runtime environment realization, `environment_key` derivation, package/import identity, dependency lock/receipt metadata, Python executable selection, and Python environment GC.
  - Reuses and hardens `ToolboxEnvironmentManager`.
  - Uses `<hosting_root>/runtime_envs/<venv_key>` for non-toolbox Python runtime environments.
  - Keeps existing toolbox `<hosting_root>/toolbox_venvs/<venv_key>` compatibility until toolbox migration is complete.

- [x] Define optional `HostedJsRuntimeBase` only if needed during JS helper migration.
  - Owns Node executable/runtime identity and JS-specific dependency identity if added later.
  - Does not reuse Python venv machinery.
  - May initially be a thin wrapper over `HostedProcessSandboxBase`.

### Concrete Public Kinds

- [x] Add `workflow_python` as a concrete public hosted runtime kind.
  - Supports `profile=helper` first.
  - Supports `profile=node` after streaming/async response contract is ready.
  - Uses `executor_kind="workflow_python"`.
  - Uses a new contract name such as `hosting.workflow_python.worker.v1`.

- [x] Keep `workflow_python_helper` as a temporary compatibility surface.
  - Old spawn/execute/resources/capacity/cancel APIs call the new `workflow_python(profile=helper)` implementation.
  - The shim must stay thin so old implementation can be removed promptly after migration.

- [x] Add `workflow_js` as a concrete public hosted runtime kind or compatibility facade.
  - Supports `profile=helper` first.
  - Existing `workflow_js_helper` becomes an alias after migration.
  - Uses the shared process/pool foundation, not Python runtime environment code.

- [x] Keep `toolbox_executor` as a concrete public hosted runtime kind.
  - Toolbox remains semantically separate: toolbox IDs, staged bundles, manifests, tool registry, tool gating/scope semantics, callbacks, repair/reconcile.
  - Toolbox should migrate to the shared base after workflow Python proves the base.

- [x] Keep generic/model workers separate.
  - Borrow protocol ideas only.
  - Do not merge `hosting.engine_worker_ipc` model semantics into workflow/toolbox contracts.

## Environment Key Rules

- [x] Implement a stable `environment_key` derivation helper.
  - Inputs: `environment_name`, runtime kind, runtime version/hash, profile, normalized imports, package pins or dependency lock hash, sandbox policy hash, and optional capability profile.
  - Excludes `package_id`, `workflow_id`, and request provenance unless those change dependencies or policy.
  - Produces stable short and full hashes for registration, logs, metrics, and GC.

- [x] Reject or warn on caller-provided `environment_key` mismatches.
  - Host should derive the authoritative key or verify a caller-provided key against normalized inputs.
  - Do not let clients route by arbitrary key if it would merge incompatible policies/dependencies.

- [x] Persist environment identity on worker registrations.
  - Include `environment_key`, `environment_name`, `environment_root_kind`, `environment_consumer_kind`, profile, runtime hash, sandbox policy hash, import/package/dependency identity, and install/receipt status summary.

- [x] Ensure different keys never share a live worker process or hot child process pool.
  - Enforce this at host routing and inside worker/runtime pool selection.

## Pooling, Metrics, And Request Tracking

- [x] Implement a host-side pool registry by concrete kind and `environment_key`.
  - Example key path: `workflow_python/<environment_key>`.
  - Tracks desired capacity, current capacity, active calls, queued or rejected calls, workers, active request IDs, and recent terminal outcomes.

- [x] Start with one worker per environment key with an internal hot child pool.
  - `desired_capacity=N` maps to one worker with N execution slots where practical.
  - Keep design open for replicas: multiple workers per key, each with per-worker capacity.

- [x] Add a scheduler policy.
  - Select existing available worker.
  - Spawn/ensure worker when none exists.
  - Grow within capacity if needed.
  - Return a structured capacity error when saturated unless explicit queuing is added.

- [x] Add request lifetime tracking.
  - Record `request_id`, `operation_id` where applicable, environment key, worker engine ID, profile, submitted/started/finished/canceled timestamps, status, reason, and byte counts.
  - Keep a bounded recent request ring in memory and include it in resource/metrics responses.

- [x] Add latency and concurrency metrics.
  - Queue wait ms.
  - Execution latency ms.
  - Total request lifetime ms.
  - Active calls.
  - Available slots.
  - Saturation count.
  - Timeout count.
  - Cancellation count.
  - Error count by reason.
  - Per-worker CPU/RSS where available.

- [x] Add resource reporting by `environment_key`.
  - Roll up desired capacity, active calls, available slots, worker count, process count, active request IDs, metrics, and worker details.

- [x] Add capacity adjustment by `environment_key`.
  - Changing desired capacity updates scheduler state and worker/internal pool capacity.
  - Decrease retires idle excess workers/children and allows active calls to finish unless explicit cancel is requested.

- [x] Add cancellation by `environment_key + request_id`.
  - Locate active request in pool registry.
  - Forward to worker request cancellation when supported.
  - Fall back to killing/replacing the owning child/worker when contract requires it.
  - Record cancellation result and request lifetime outcome.

## Sync, Async, And Streaming Contract

- [x] Standardize base IPC message families for new concrete sandboxes.
  - `hello`
  - `rpc_call`
  - `stream_open`
  - `stream_recv`
  - `stream_send`
  - `stream_close`
  - `shutdown`

- [x] Keep short helper calls available as sync `rpc_call`.
  - `workflow_python(profile=helper)` and `workflow_js(profile=helper)` may start sync-only for compatibility.

- [x] Add async/streaming support as base capability.
  - Required for `workflow_python(profile=node)`.
  - Optional for helper profiles.
  - Use bounded stream queues and explicit recv limits.

- [x] Define a common event envelope for streaming responses.
  - `progress`
  - `stdout`
  - `stderr`
  - `log`
  - `artifact`
  - `metric`
  - `result`
  - `error`
  - `canceled`
  - `done`

- [x] Define stream cancellation semantics.
  - `stream_send` supports `{"action":"cancel","request_id":"..."}`.
  - `stream_close` closes client stream and requests stop.
  - Host-level cancel still works by `environment_key + request_id`.

- [x] Define response shape compatibility for sync calls.
  - Sync calls return stable `status`, `ok`, result/output, structured error, runtime, metrics summary, and audit metadata.
  - Preserve old helper response shape through compatibility shims during migration.

## Phase 0: Discovery And Tests Baseline

- [x] Inventory current hosting tests for sandbox, helper, toolbox, generic worker, CLI, and interactive CLI coverage.
- [x] Add missing characterization tests before refactoring old helper behavior.
  - Python helper source hash verification.
  - Operation allowlist.
  - Timeout.
  - Output limit.
  - Cancellation.
  - Capacity/resource reporting.
  - Import allowlist behavior.
  - Audit/provenance fields.
  - JS helper equivalent behavior.
- [x] Capture current CLI command outputs for workflow helper resources/capacity/cancel paths.
- [x] Capture current interactive CLI workflow helper screens where practical.
- [x] Record baseline status in `src/hosting/hosting_status.md`.

## Phase 1: Shared Base Contracts And Models

- [x] Add internal data models for hosted sandbox kind, profile, environment key, pool key, worker slot, request lifecycle, stream event, and metrics.
- [x] Add shared normalization for sandbox policy hash.
- [x] Add shared normalization for runtime/profile identity.
- [x] Add shared registration metadata helpers.
- [x] Add shared resource/metrics response builders.
- [x] Add shared cancellation result shape.
- [x] Add base tests for environment key stability and policy hash changes.
- [x] Update docs with draft internal contract notes after models stabilize.

## Phase 2: Hosted Process Pool Base

- [x] Implement host-side process pool registry.
- [x] Implement worker ensure/spawn by pool key.
- [x] Implement worker selection by availability.
- [x] Implement desired capacity and per-worker capacity state.
- [x] Implement resource rollups.
- [x] Implement request lifetime registry and recent request ring.
- [x] Implement cancel routing by request ID.
- [x] Implement metrics collection and latency/concurrency summaries.
- [x] Add tests for pool routing, capacity saturation, cancel lookup, and metrics rollup.

## Phase 3: Python Runtime Environment Base

- [x] Extract workflow-neutral Python environment helpers from `ToolboxEnvironmentManager`.
- [x] Expose workflow-specific environment operations without toolbox IDs/tool keys in the public API.
  - Prepare environment.
  - Lock install.
  - Verify lock.
  - Resolve lock when explicitly allowed.
  - Execute install when explicitly allowed.
  - Verify install receipt.
  - Realize/select runtime Python.

- [x] Ensure dependency installation is never triggered from workflow execution code.
- [x] Require explicit host environment-management API calls for dependency install.
- [x] Make package pins/imports either enforced by selected verified environment or explicitly reported as declarative/unverified.
- [x] Persist install and receipt status with environment metadata.
- [x] Add GC support for unreferenced `runtime_envs` entries by environment key.
- [x] Add tests for no-package runtime env, pinned-package env identity, lock/receipt metadata, and runtime Python selection.

## Phase 4: New Workflow Python Worker

- [x] Add `workflow_python` service/channel/CLI API.
  - `workflow-python-prepare-environment`
  - `workflow-python-lock-environment`
  - `workflow-python-verify-environment`
  - `workflow-python-install-environment`
  - `workflow-python-ensure`
  - `workflow-python-execute`
  - `workflow-python-cancel-request`
  - `workflow-python-resources`
  - `workflow-python-set-capacity`

- [x] Add `workflow_python` worker implementation or adapt current Python helper worker behind the new contract.
- [x] Implement `profile=helper`.
  - Keep helper operations: `default`, `condition`, `evaluate_condition`, `routing_hint`, `route_hint`, `payload`, `shape_payload`.
  - Keep source hash verification.
  - Keep JSON payload/result.
  - Keep timeout/output limit behavior.
  - Keep request cancellation.
  - Add environment-keyed pool isolation.
  - Add request lifetime metrics.

- [x] Implement compatibility response adapter for `execute_workflow_python_helper`.
- [x] Add new response shape for `workflow_python(profile=helper)`.
- [x] Add sync tests through new API.
- [x] Add compatibility tests through old helper API.
- [x] Add environment-key isolation tests to prove incompatible policies/dependencies do not share pools.
- [x] Add RBAC and daemon claim-policy coverage for new `workflow-python-*` commands.

## Phase 5: Workflow Python Node Profile

- [x] Define `profile=node` request contract.
  - `module_source`
  - `module_sha256`
  - `package_id`
  - `workflow_id`
  - `package_source_digest`
  - `export_name` or operation
  - `payload`
  - `provenance`
  - `limits`
  - policy/environment identity

- [x] Define `profile=node` response contract.
  - `ok` / `status`
  - output JSON
  - state patch JSON
  - artifact refs
  - progress events or latest progress
  - stdout/stderr/log summaries
  - metrics
  - structured error
  - audit metadata

- [x] Implement async/streaming execution for node profile.
- [x] Implement artifact ref plumbing or explicit placeholder errors if artifact store is not ready.
- [x] Implement latest-progress snapshot in resource/request-status responses.
- [x] Add tests for streamed progress, stdout/stderr summary truncation, result, structured error, timeout, cancel, and metrics.

## Phase 6: Migrate Workflow Python Helper Compatibility

- [x] Rewire `spawn_workflow_python_helper` service method to call `workflow_python(profile=helper)`.
- [x] Rewire channel method `spawn_workflow_python_helper`.
- [x] Rewire `workflow_python_helper_resources`.
- [x] Rewire `set_workflow_python_helper_capacity`.
- [x] Rewire `cancel_workflow_python_helper_request`.
- [x] Rewire `execute_workflow_python_helper` proxy path.
- [x] Keep old command names as aliases.
- [x] Mark old implementation files for removal once dependent callers migrate.
- [x] Add deprecation entries to `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`.

## Phase 7: Workflow JS Helper Migration

- [x] Add `workflow_js(profile=helper)` facade.
- [x] Move JS helper worker/pool lifecycle onto `HostedProcessSandboxBase`.
- [x] Add environment/runtime key derivation for JS.
  - Include environment name, profile, Node executable/version, sandbox policy hash, and package/dependency identity if supported.

- [x] Rewire old `workflow_js_helper` APIs as compatibility aliases.
- [x] Align JS helper resource/capacity/cancel response shape with workflow Python.
- [x] Keep JS-specific compatibility fields until clients migrate.
- [x] Add JS helper compatibility tests.
- [x] Add new JS facade tests.
- [x] Add deprecation entries to `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`.

## Phase 8: CLI And Interactive CLI Compatibility

- [x] Update `src/hosting/engine_host_cli.py`.
  - Add new workflow Python commands.
  - Keep old workflow helper commands as aliases.
  - Normalize payload parsing for environment keys, profiles, capacity, request IDs, and streaming operations.
  - Add CLI output fields for metrics, active requests, and environment key.
  - Avoid exposing raw module source/payload/result in logs unless explicitly requested.

- [x] Update `src/hosting/engine_host_cli_interactive.py`.
  - [x] Show workflow runtime pools by environment key.
  - [x] Show profile, capacity, active calls, available slots, process count, active request IDs, latency summaries, cancellation counters, and recent request outcomes.
  - [x] Keep old helper display paths compatible.
  - [x] Add actions for set capacity, cancel request, and inspect resources by environment key.
  - [x] Add ensure action by environment key.
  - [x] Add streaming/request-status UI where practical.

- [x] Update channel wrappers in `src/hosting/engine_host_channel.py`.
  - Add new typed methods.
  - Preserve old helper wrappers as aliases.
  - Support sync and streaming proxy helpers.

- [x] Add tests or smoke scripts for CLI command compatibility.
- [x] Add manual verification notes to `src/hosting/hosting_status.md`.

## Phase 9: Toolbox Migration To Shared Base

- [x] Map toolbox worker lifecycle onto `HostedProcessSandboxBase`.
- [x] Keep toolbox public APIs unchanged during migration.
- [x] Keep toolbox staged bundle, manifest, tool routing, callbacks, gate/scope, repair/reconcile semantics unchanged.
- [x] Reuse shared Python/runtime identity concepts where compatible.
  - Decision: toolbox keeps `ToolboxEnvironmentManager` and `toolbox_venvs` ownership; `HostedToolboxRuntimeBase` adds shared hosted identity without directly inheriting workflow Python environment realization.
- [x] Preserve existing `<hosting_root>/toolbox_venvs` behavior until a deliberate storage migration is planned.
- [x] Add parity tests for toolbox register/execute/cancel/resources/repair/reconcile.
- [x] Add tests for callback relay and brokered FS/HTTP after migration.
- [x] Update toolbox docs and status notes.

## Phase 10: Cleanup And Removal

- [x] Remove old Python helper implementation after callers migrate.
  - Current state: public helper command/channel/daemon/auth surfaces are removed; `src/hosting/workflow_python_helper_ipc.py` remains as the internal worker entrypoint behind `workflow-python-*` facades and can be reduced to a shim in a later physical-file cleanup.

- [x] Remove old JS helper implementation after callers migrate.
  - Current state: public helper command/channel/daemon/auth surfaces are removed; `workflow-js-execute` is the public facade and `src/hosting/workflow_js_helper_ipc.py` remains as the internal worker entrypoint behind it.

- [x] Remove duplicate toolbox environment code paths after toolbox migration.
  - Current state: toolbox now uses shared sandbox identity/process-base plumbing while retaining its existing environment manager ownership; no duplicate public toolbox migration path remains in this epic.
- [x] Remove compatibility fields only after `HOSTING_CLIENT_BREAKING_CHANGES.md` says clients have moved.
  - Current state: public old helper APIs are removed; compatibility-shaped worker result fields are retained inside the new workflow envelope as response data, not as old API surface.
- [x] Remove obsolete CLI branches.
- [x] Add final migration status to `src/hosting/hosting_status.md`.

## Phase 11: Documentation

Do not front-load all documentation before implementation, because contracts will move during the refactor. Do keep short design notes current during implementation and do a full docs pass after APIs stabilize.

- [x] Early docs: update architecture docs with intended shared-base design once Phase 1 models land.
- [x] Midpoint docs: document workflow Python environment APIs when Phase 3/4 are usable.
- [x] Final docs: update all public docs after compatibility shims are verified.
  - `src/hosting/HOSTING.md`
  - `src/hosting/sandbox/SANDBOX_ARCHITECTURE.md`
  - `src/hosting/sandbox/WORKFLOW_HELPER_WORKER.md`
  - `src/hosting/sandbox/TOOLBOX_WORKER.md`
  - `src/hosting/sandbox/GENERIC_WORKER.md`
  - CLI docs and examples.

- [x] Keep `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md` current for dependent projects.
- [x] Keep `src/hosting/hosting_status.md` current after each phase or major PR.

## Dependent Project Migration Contract

Track details in `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`.

- [x] Clients should stop treating `workflow_python_helper` as the future primary API.
- [x] Clients should start routing new work through `workflow_python(profile=helper|node)` when available.
- [x] Clients should stop using raw `engine_id` as the only pool identity.
- [x] Clients should start using or accepting host-derived `environment_key`.
- [x] Clients should stop assuming `package_pins` are enforced during helper execution until host reports a verified runtime environment.
- [x] Clients should stop installing dependencies during workflow execution.
- [x] Clients should call prepare/lock/verify/install APIs only from explicit environment-management paths.
  - Hosting owns the lifecycle commands and install enforcement.
  - The migrated helper-profile client path is complete without install orchestration because it does not provide dependency installation UI/orchestration.
- [x] Clients should stop expecting different dependency/policy sets to share helper workers.
- [x] Clients should start reading resource/capacity/metrics by `environment_key`.
- [x] Clients should start passing stable `request_id` for cancellation and request lifetime tracking.
- [x] Clients should use streaming APIs for long-running node-profile work.
  - Required only for clients that execute long-running node-profile work. The migrated helper-profile client path is complete without streaming because it executes short helper-profile modules.

## Verification Checklist

- [x] Unit tests for model normalization.
- [x] Unit tests for environment key derivation.
- [x] Unit tests for pool scheduler.
- [x] Unit tests for request lifetime metrics.
- [x] Unit tests for Python environment realization and install metadata.
- [x] Integration tests for workflow Python helper compatibility.
- [x] Integration tests for workflow Python new API.
- [x] Integration tests for workflow Python node streaming.
- [x] Integration tests for workflow JS helper compatibility.
- [x] Integration tests for toolbox parity after migration.
- [x] CLI smoke tests.
- [x] Interactive CLI manual verification.
- [x] Docs examples verified against actual commands.

## Open Design Decisions

- [x] Decide whether the first streaming implementation reuses existing generic worker stream proxy commands directly or introduces workflow-named aliases over the same transport.
  - Decision: introduce workflow-named `workflow-python-stream-*` commands over shared base stream/session primitives.
- [x] Decide whether `workflow_python(profile=helper)` should expose streaming immediately or remain sync-only until node profile lands.
  - Decision: helper profile remains sync-only for compatibility; node profile owns the workflow stream rollout.
- [x] Decide whether the first pool implementation should support multiple workers per environment key or only one worker with an internal hot child pool.
  - Decision: one worker per environment key with per-worker/internal capacity first; base models allow future replicas.
- [x] Decide how artifact refs are stored and authorized for node-profile responses.
  - Decision: artifact refs are contract fields, but storage is explicitly reported as unavailable until an artifact store is designed.
- [x] Decide the retention policy for recent request lifetime metrics.
  - Decision: in-memory bounded recent request ring defaults to 100 entries per process pool.
- [x] Decide whether CLI should display raw stdout/stderr snippets by default or only summaries.
  - Decision: CLI surfaces structured JSON by default; workflow docs describe logs/stdout/stderr as response fields, not default raw terminal output.
