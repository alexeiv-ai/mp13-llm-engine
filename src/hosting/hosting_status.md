# Hosting Status And Work Plan

Date: 2026-05-21

This file tracks planned hosting/sandbox work as checkbox items. Check items only when the implementation, tests, and related docs are complete.

## Shared Runtime Environment Infrastructure

- [ ] Rename the host-managed environment concept from toolbox-specific wording to a shared runtime/sandbox environment concept in docs and APIs.
- [x] Introduce a neutral environment root, preferably `<hosting_root>/runtime_envs/<env_key>`, for new non-toolbox runtime environments.
- [x] Keep `toolbox_venvs` readable for existing toolbox environments during migration so current persisted registrations and cleanup paths do not break.
- [ ] Define the compatibility/migration rule for old `toolbox_venvs` entries: whether they remain in place, are migrated lazily, or are copied into `runtime_envs`.
- [ ] Split the reusable environment manager responsibilities from toolbox ownership. Target shape: a generic runtime environment manager plus toolbox-specific convenience wrappers.
- [x] Preserve deterministic environment identity inputs: runtime hash, environment name/description hash, required imports, package pins or dependency lock hash, and sandbox/helper kind.
- [x] Update environment metadata to include a stable `environment_owner_kind` or `consumer_kind` such as `toolbox_executor`, `workflow_python_helper`, or `workflow_js_helper`.
- [x] Update reference reporting and GC so shared runtime environments are kept when referenced by toolbox state, workflow helper state, or live worker registrations.
- [ ] Update operator review/consistency output to distinguish environment consumers instead of assuming all environment users are toolboxes.
- [ ] Add tests for runtime environment path selection, legacy `toolbox_venvs` compatibility, reference reporting, and GC behavior.

## Runtime Python Selection And Verified Bootstrap

- [x] Rename "fallback" terminology in runtime Python selection to a clearer concept such as bootstrap/preverified/trusted-host Python.
- [x] Treat this path as a verification gate, not a permanent compatibility fallback.
- [x] Fix the no-package environment case so a realized environment with no install work does not stay permanently on bootstrap Python.
- [ ] Decide and document the exact statuses that allow venv activation, including no-op installs, verified receipts, and failed/stale locks.
- [x] Update metadata fields currently named `runtime_python_source = fallback` to avoid implying long-term fallback behavior.
- [x] Update tests that currently assert fallback behavior for empty workflow helper environments.
- [ ] Document why a preverified Python may be used before an environment is verified and what condition switches execution to the realized venv.

## Workflow Python Helper Support

- [ ] Decide whether workflow Python helpers are executed as toolbox tools or as a separate workflow helper executor contract.
- [ ] If workflow Python helpers are not toolbox tools, add an explicit executor kind such as `workflow_python_helper`.
- [ ] Define the workflow Python helper worker contract, including request shape, result shape, provenance, allowed operations, timeout behavior, and JSON-only input/output.
- [ ] Reuse `EngineHostService.spawn(...)`, `WorkerSandboxPolicy`, `WorkerLaunchRequest`, persisted worker registration, sandbox runtime reporting, and hosting IPC.
- [x] Reuse the shared runtime environment manager instead of adding a Python-helper-specific venv manager.
- [ ] Carry helper provenance in metadata: package id, workflow id, package source digest, helper source SHA-256, helper source path or staged source id, operation/export name, session/context/cursor ids when available, worker id, elapsed time, and reason on failure.
- [ ] Add service/channel/CLI or internal API surfaces to realize, spawn, status-check, and shut down workflow Python helper workers if a separate executor is chosen.
- [ ] Update docs to clarify that workflow helpers are not logical toolbox tools unless intentionally registered as tools.

## Workflow JS Helper Executor

- [x] Add a minimal workflow JS helper executor on the existing hosting worker/sandbox model.
- [x] Prefer a specialization of the generic worker path before adding a fully separate lifecycle manager.
- [x] Use `worker_profile_class = "generic"` unless implementation proves a narrower profile class is required.
- [x] Use `executor_kind = "workflow_js_helper"` for persisted registrations and routing.
- [x] Use sandbox profile `workflow_js_helper_v1`.
- [x] Use execution contract `hosting.workflow_helper.worker.v1`.
- [x] Spawn through `EngineHostService.spawn(...)` with a normal persisted worker registration.
- [x] Persist `sandbox_policy`, `sandbox_runtime`, command, environment, worker id, IPC family/address, auth token metadata, and capabilities as existing workers do.
- [x] Expose status/diagnostics through existing worker registration/runtime status surfaces.
- [x] Report workflow JS helper availability, Node executable path, Node version, worker id, engine id, sandbox profile, launch mode, platform sandbox status, and current capacity/busy state when available.
- [x] Add ensure-running/shutdown/status behavior using existing worker lifecycle APIs.

## Workflow JS Helper RPC Contract

- [x] Implement RPC method `execute_workflow_js_helper`.
- [x] Accept `module_source` as the public source input.
- [x] Do not accept caller-provided JS file paths.
- [x] Verify `sha256(module_source) == module_sha256` before execution.
- [x] Execute only the requested named export.
- [x] Restrict operations to `default`, `condition`, `evaluate_condition`, `routing_hint`, `route_hint`, `payload`, and `shape_payload`.
- [x] Require JSON-only input payloads.
- [x] Require JSON-only output results.
- [x] Internally materialize source into host-owned staging files only if the Node execution strategy requires files.
- [x] Keep host-owned staging paths out of the public request contract.
- [x] Enforce per-call timeout from `limits.timeout_ms`.
- [x] Enforce output size from `limits.output_limit_bytes`.
- [x] Report memory limit behavior from `limits.memory_limit_mb`, including whether enforcement is active, best-effort, or unavailable on the current platform.
- [x] Add safe concurrent-call handling through a bounded serialized queue, bounded worker pool, or explicit `workflow_sandbox_capacity_exceeded`/busy response.
- [x] Include runtime details in success and failure results: worker id, engine id, Node version, and sandbox profile.

## Workflow JS Helper Sandbox Policy

- [x] Use `WorkerSandboxPolicy` for the JS helper worker.
- [x] Set default v1 policy to sandbox enabled.
- [x] Set default v1 sandbox profile to `workflow_js_helper_v1`.
- [x] Set process subprocess spawning to disabled.
- [x] Set network mode to disabled.
- [x] Disable brokered filesystem for helper-visible access.
- [x] Disable brokered HTTP.
- [x] Disable brokered subprocess.
- [x] Ensure helper JS has no filesystem capability even if the host internally stages module files.
- [x] Document that v1 has no direct network, brokered network, brokered filesystem, helper subprocess, general Node app hosting, or long-running workflow jobs.

## Workflow JS Helper Result And Error Mapping

- [x] Return success shape with `ok: true`, `result`, and `runtime`.
- [x] Return failure shape with `ok: false`, `reason`, `detail`, and `runtime`.
- [x] Preserve or map `workflow_sandbox_invalid_module_identity`.
- [x] Preserve or map `workflow_sandbox_operation_not_allowed`.
- [x] Preserve or map `workflow_sandbox_export_not_found`.
- [x] Preserve or map `workflow_sandbox_timeout`.
- [x] Preserve or map `workflow_sandbox_output_limit_exceeded`.
- [x] Preserve or map `workflow_sandbox_invalid_json_output`.
- [x] Preserve or map `workflow_sandbox_invalid_result_shape`.
- [x] Preserve or map `workflow_sandbox_runtime_error`.
- [x] Preserve or map `workflow_sandbox_host_unavailable`.
- [x] Preserve or map `workflow_sandbox_capacity_exceeded`.
- [ ] Add tests for each stable failure reason.

## Workflow JS Helper Audit And Provenance

- [x] Carry package id through request handling, worker logs, and host-side diagnostics.
- [x] Carry workflow id through request handling, worker logs, and host-side diagnostics.
- [x] Carry package source digest.
- [x] Carry module SHA-256.
- [x] Carry operation and export name.
- [x] Carry session id, context id, cursor id, and workflow root id when provided.
- [x] Record elapsed milliseconds.
- [x] Record worker id and engine id.
- [x] Record failure reason.
- [ ] Decide whether audit events are persisted in registration metadata, logs, a bounded status cache, or a dedicated audit sink.
- [ ] Add redaction rules for source, payload, and result data in logs/status output.

## Generic Worker Extension Decision

- [x] Evaluate whether `hosting.engine_worker_ipc` can host this contract without inheriting model-worker assumptions.
- [ ] If generic worker can support it cleanly, add a contract dispatch path keyed by `MP13_WORKER_CONTRACT` or executor kind.
- [x] If generic worker cannot support it cleanly, add the smallest new worker module, likely `hosting.workflow_js_helper_ipc`, while still using shared spawn/sandbox/IPC infrastructure.
- [x] Avoid routing workflow helper calls through model tool APIs.
- [x] Avoid forcing workflow helpers into logical toolbox state unless they are intentionally exposed as toolbox tools.
- [x] Keep the public host API executor-kind based so Python and JS helpers can share lifecycle, status, and shutdown patterns.

## Documentation

- [ ] Update [sandbox/SANDBOX_ARCHITECTURE.md](sandbox/SANDBOX_ARCHITECTURE.md) to describe shared sandbox infrastructure separately from toolbox-specific execution.
- [x] Update [sandbox/GENERIC_WORKER.md](sandbox/GENERIC_WORKER.md) with the generic-worker extension decision and workflow helper relationship.
- [ ] Update [sandbox/TOOLBOX_WORKER.md](sandbox/TOOLBOX_WORKER.md) to explain toolbox workers as one consumer of shared runtime environments.
- [x] Add a workflow helper worker doc if a separate executor module is introduced.
- [ ] Document the neutral runtime environment root and the `toolbox_venvs` migration/compatibility rule.
- [x] Document default workflow JS sandbox policy and v1 non-goals.
- [x] Document the workflow JS helper RPC request/result/error contract.
- [x] Update [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md) with dependent-project migration guidance for this work.
- [x] In [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md), list what dependent projects must stop doing, including direct Node helper spawning, caller-provided helper file paths, reliance on `toolbox_venvs` naming, and treating runtime Python bootstrap as a permanent fallback.
- [x] In [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md), list what dependent projects should start doing, including calling the workflow helper RPC with `module_source`, using `executor_kind = "workflow_js_helper"`, relying on host-managed sandbox policy/lifecycle/status, and reading neutral runtime environment metadata instead of toolbox-specific paths.
- [x] In [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md), include a before/after migration example for mp13-docs workflow JS helpers.
- [ ] In [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md), include the final compatibility window and any supported legacy behavior for existing clients.

## Test Plan

- [ ] Unit test shared runtime environment identity and metadata for toolbox, workflow Python, and workflow JS consumers.
- [ ] Unit test legacy `toolbox_venvs` lookup and new `runtime_envs` creation.
- [ ] Unit test runtime Python bootstrap-to-verified transition.
- [ ] Unit test no-package/no-op verified activation.
- [x] Unit test JS module identity verification.
- [x] Unit test JS allowed operation filtering.
- [x] Unit test JS missing export behavior.
- [x] Unit test JSON input/output validation.
- [x] Unit test timeout and output limit behavior.
- [x] Unit test memory limit reporting.
- [x] Unit test capacity/busy behavior under concurrent calls.
- [x] Integration test JS helper worker spawn, RPC execution, status, ensure-running, and shutdown.
- [x] Integration test sandbox policy persistence and sandbox runtime reporting for `workflow_js_helper`.
- [ ] Integration test helper-visible filesystem/network denial for v1 policy.
- [ ] Integration test audit/provenance propagation.
- [ ] Regression test toolbox sandbox registration, execution, repair, reconcile, and GC after environment-root changes.

## Acceptance Criteria

- [x] New workflow helper support uses existing host spawn, sandbox policy, launcher, persisted registration, IPC/RPC, lifecycle, and status mechanisms.
- [ ] New runtime environment support is not toolbox-named for new non-toolbox consumers.
- [ ] Existing toolbox sandboxes continue to work without changing public toolbox APIs.
- [x] Workflow JS helpers execute from `module_source`, never caller-provided file paths.
- [x] Workflow JS helpers have no helper-visible filesystem, network, brokered I/O, or subprocess access in v1.
- [x] Workflow Python and JS helper metadata carry enough provenance for security review.
- [ ] The project no longer describes verified runtime selection as generic fallback compatibility code.
- [x] [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md) clearly tells dependent projects what to stop doing and what to start doing as a result of this task.
