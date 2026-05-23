# Hosting Status And Work Plan

Date: 2026-05-22

This file tracks the implementation plan for adding a hosted workflow Python helper executor lane and normalizing workflow helper pool reporting. Check items only when implementation, tests, and related docs are complete.

## Request Summary

- [ ] Add a dedicated hosted Python helper executor lane now that a concrete backend caller contract exists.
- [ ] Model the Python helper lane after the existing workflow JS helper worker where the lifecycle and API shape fit.
- [ ] Reuse the shared runtime environment manager already added for `workflow_python_helper`; do not add a Python-helper-specific venv manager.
- [ ] Keep this separate from toolbox execution, model execution, and unrestricted generic Python execution.
- [ ] Preserve the JS helper API while adding generic helper pool aliases so backend code can manage JS and Python helpers through one controller.

## Architecture Decisions

- [ ] Use `executor_kind = "workflow_python_helper"` for persisted registrations and routing.
- [ ] Use `worker_profile_class = "generic"` unless implementation proves a narrower profile class is required.
- [ ] Use a dedicated worker module, likely `hosting.workflow_python_helper_ipc`, instead of extending model-oriented worker IPC.
- [ ] Use execution contract `hosting.workflow_helper.worker.v1` unless Python needs a strictly versioned sub-contract.
- [ ] Use sandbox profile `workflow_python_helper_v1`.
- [ ] Reuse `EngineHostService.spawn(...)`, `WorkerSandboxPolicy`, persisted worker registration, sandbox runtime reporting, hosting IPC/RPC, lifecycle, shutdown, and ensure-running behavior.
- [ ] Keep Python helper execution out of toolbox registry/tool routing unless a helper is intentionally exposed later as a toolbox tool.
- [ ] Keep Python helper execution out of raw generic process spawn; raw process spawn remains `config_editor`/`admin` only.

## Runtime Environment Integration

- [ ] Accept Python environment requirements in request field `python`.
- [ ] Support `python.import_allowlist` as declared helper import intent.
- [ ] Support `python.package_pins` as deterministic dependency intent.
- [ ] Support `python.environment_name`, defaulting to `workflow-python-helper` when omitted.
- [ ] Map Python helper environment identity to the existing shared runtime environment manager using `workflow_python_helper` as the consumer/owner kind.
- [ ] Resolve or realize the requested runtime environment before executing helper code when package requirements are present.
- [ ] Use the verified realized environment Python when available.
- [ ] Use the existing preverified/bootstrap Python path only as a verification gate when the environment is not yet eligible for activation.
- [ ] Report runtime Python path/source in resources and per-call runtime data without using fallback terminology.
- [ ] Preserve existing `runtime_envs` and legacy readable `toolbox_venvs` compatibility behavior from the shared environment manager.

## Python Helper Worker

- [ ] Add `hosting.workflow_python_helper_ipc`.
- [ ] Add hot Python child process pool under one hosting worker id.
- [ ] Make worker `capacity` mean maximum hot Python child processes owned by the worker.
- [ ] Recycle child processes after a bounded request count to limit module/cache growth.
- [ ] Support per-request cancellation by `request_id` by killing the child process that owns the active request.
- [ ] Return `workflow_sandbox_capacity_exceeded` immediately when all slots are in use.
- [ ] Report active request ids, process ids, request counts, busy/idle state, and per-child CPU/RSS when available.
- [ ] Ensure worker shutdown terminates all hot Python child processes.
- [ ] Ensure timeout terminates the child process running the timed-out call.

## Python Execution Contract

- [ ] Implement RPC method `execute_workflow_python_helper`.
- [ ] Accept `module_source` as the public source input.
- [ ] Verify `sha256(module_source) == module_sha256` before execution.
- [ ] Accept `source_path` only as provenance; do not execute caller-provided file paths.
- [ ] Execute only the requested named export/function.
- [ ] Restrict operations to `default`, `condition`, `evaluate_condition`, `routing_hint`, `route_hint`, `payload`, and `shape_payload`.
- [ ] Require JSON-only input payloads.
- [ ] Require JSON-only output results.
- [ ] Enforce `limits.timeout_ms` per call.
- [ ] Enforce `limits.output_limit_bytes` per call.
- [ ] Report `limits.memory_limit_mb` behavior, including whether enforcement is active, best-effort, or unavailable.
- [ ] Prevent helper-visible filesystem, network, brokered I/O, and subprocess access in the v1 sandbox policy.
- [ ] Avoid logging raw helper source, payload, or result data.

## Python Result And Error Mapping

- [ ] Return success shape with `ok: true`, `result`, `runtime`, and `audit`.
- [ ] Return failure shape with `ok: false`, `reason`, `detail`, `runtime`, and `audit`.
- [ ] Preserve or map `workflow_sandbox_invalid_module_identity`.
- [ ] Preserve or map `workflow_sandbox_operation_not_allowed`.
- [ ] Preserve or map `workflow_sandbox_export_not_found`.
- [ ] Preserve or map `workflow_sandbox_timeout`.
- [ ] Preserve or map `workflow_sandbox_canceled`.
- [ ] Preserve or map `workflow_sandbox_output_limit_exceeded`.
- [ ] Preserve or map `workflow_sandbox_invalid_json_output`.
- [ ] Preserve or map `workflow_sandbox_invalid_result_shape`.
- [ ] Preserve or map `workflow_sandbox_runtime_error`.
- [ ] Preserve or map `workflow_sandbox_host_unavailable`.
- [ ] Preserve or map `workflow_sandbox_capacity_exceeded`.
- [ ] Add Python-specific stable reasons only if they are necessary and documented for clients.

## Audit And Provenance

- [ ] Carry `package_id`.
- [ ] Carry `workflow_id`.
- [ ] Carry `package_source_digest`.
- [ ] Carry `module_sha256`.
- [ ] Carry `source_path` as provenance only.
- [ ] Carry `request_id`.
- [ ] Carry `operation` and `export_name`.
- [ ] Carry `provenance.session_id`.
- [ ] Carry `provenance.context_id`.
- [ ] Carry `provenance.cursor_id`.
- [ ] Carry `provenance.workflow_root_id`.
- [ ] Record elapsed milliseconds.
- [ ] Record worker id and engine id.
- [ ] Record runtime Python path/source.
- [ ] Record failure reason.

## Public Service And Channel API

- [ ] Add `EngineHostService.spawn_workflow_python_helper(...)`.
- [ ] Add `EngineHostControlChannel.spawn_workflow_python_helper(...)`.
- [ ] Add `EngineHostService.workflow_python_helper_resources(...)`.
- [ ] Add `EngineHostControlChannel.workflow_python_helper_resources(...)`.
- [ ] Add `EngineHostService.set_workflow_python_helper_capacity(...)`.
- [ ] Add `EngineHostControlChannel.set_workflow_python_helper_capacity(...)`.
- [ ] Add `EngineHostService.cancel_workflow_python_helper_request(...)`.
- [ ] Add `EngineHostControlChannel.cancel_workflow_python_helper_request(...)`.
- [ ] Add daemon dispatch for all new workflow Python helper commands.
- [ ] Add non-interactive CLI commands for spawn, resources, capacity, and cancel.
- [ ] Update non-interactive CLI help/output paths in `engine_host_cli.py` for the new workflow Python helper commands.
- [ ] Ensure `discover-running` identifies Python helpers as workflow helper sandboxes/workers and includes process resources.

## Normalized Helper Pool Resources

- [ ] Add generic helper pool aliases to JS helper resources without removing existing Node-specific fields.
- [ ] Return top-level `capacity`.
- [ ] Return top-level `active_calls`.
- [ ] Return top-level `available_slots`.
- [ ] Return `pool.process_count`.
- [ ] Return `pool.active_process_count`.
- [ ] Return `pool.idle_process_count`.
- [ ] Return `pool.active_request_ids`.
- [ ] Return `pool.processes`.
- [ ] Include per-process `pid`, `alive`, `busy`, `active_request_id`, `request_count`, `max_requests`, `reusable`, and `resources` when available.
- [ ] Add the same normalized pool shape to Python helper resources.
- [ ] Keep JS compatibility fields such as `node_pool`, `workflow_js_node_process_count`, and `workflow_js_node_pids`.
- [ ] Add Python compatibility fields only if needed for operator diagnostics.
- [ ] Update `discover-running` flattened resource fields for Python helpers.
- [ ] Update `discover-running` flattened resource fields for JS helpers to include normalized helper pool aliases.

## Authorization

- [ ] Allow `workflow-python-helper-resources` to `diagnostic_user` and above.
- [ ] Allow `spawn-workflow-python-helper` to `worker_user`, `config_editor`, and `admin`.
- [ ] Allow `workflow-python-helper-set-capacity` to `worker_user`, `config_editor`, and `admin`.
- [ ] Allow `workflow-python-helper-cancel-request` to `worker_user`, `config_editor`, and `admin`.
- [ ] Keep execution through traffic-scoped `proxy_rpc_call` for the registered helper `engine_id`.
- [ ] Allow `model_user` traffic proxy only when scoped to the specialized `executor_kind = "workflow_python_helper"` helper engine.
- [ ] Keep raw `spawn` restricted to `config_editor` and `admin`.
- [ ] Keep toolbox mutation authority unchanged.

## Interactive CLI

- [ ] Show workflow Python helpers in the existing engines/sandboxes inventory.
- [ ] Show Python helper details in resource details, including normalized pool state and per-child metrics.
- [ ] Generalize `Manage workflow JS helpers` into a workflow helper management view, or add a sibling `Manage workflow Python helpers` view if that is clearer.
- [ ] Use the same public channel/CLI APIs in the interactive workflow helper management view that external clients use.
- [ ] Support resource refresh for Python helpers.
- [ ] Support capacity changes for Python helpers.
- [ ] Support canceling a specific active Python helper request by `request_id`.
- [ ] Keep JS helper management working through the same normalized resource path.

## Documentation

- [ ] Update [sandbox/WORKFLOW_HELPER_WORKER.md](sandbox/WORKFLOW_HELPER_WORKER.md) with the Python helper lane.
- [ ] Update [sandbox/GENERIC_WORKER.md](sandbox/GENERIC_WORKER.md) with the Python helper executor decision.
- [ ] Update [sandbox/SANDBOX_ARCHITECTURE.md](sandbox/SANDBOX_ARCHITECTURE.md) if shared helper pool/resource concepts need architecture coverage.
- [ ] Update [HOSTING.md](HOSTING.md) with the public Python helper service/channel APIs.
- [ ] Update [ENGINE_HOST_CLI.md](ENGINE_HOST_CLI.md) with non-interactive and interactive workflow helper CLI commands.
- [ ] Complete [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md) with what dependent projects should stop and start doing after implementation.
- [ ] Document that Python helper `source_path` is provenance only and not an execution path.
- [ ] Document the normalized helper resource shape shared by JS and Python helpers.
- [ ] Document authorization boundaries for Python helper resources, spawn, capacity, cancel, and execution.

## Test Plan

- [ ] Unit test Python module identity verification.
- [ ] Unit test Python allowed operation filtering.
- [ ] Unit test Python missing export behavior.
- [ ] Unit test JSON input validation.
- [ ] Unit test JSON output validation.
- [ ] Unit test timeout behavior.
- [ ] Unit test cancellation behavior and `workflow_sandbox_canceled` mapping.
- [ ] Unit test output limit behavior.
- [ ] Unit test memory limit reporting.
- [ ] Unit test capacity exceeded behavior.
- [ ] Unit test Python hot process reuse and recycling.
- [ ] Unit test Python helper resources and normalized pool shape.
- [ ] Unit test JS helper normalized pool aliases while preserving legacy JS fields.
- [ ] Unit test service/channel forwarding for Python spawn/resources/capacity/cancel.
- [ ] Unit test daemon dispatch for Python helper commands.
- [ ] Unit test RBAC for Python helper commands.
- [ ] Integration test Python helper worker spawn, RPC execution, resources, capacity change, cancellation, ensure-running, and shutdown.
- [ ] Integration test Python helper runtime environment resolution with `import_allowlist`, `package_pins`, and `environment_name`.
- [ ] Integration test sandbox policy persistence and sandbox runtime reporting for `workflow_python_helper`.
- [ ] Integration test helper-visible filesystem/network/subprocess denial for v1 policy where platform support allows.
- [ ] Regression test JS helper execution and resource reporting after normalized pool aliases.
- [ ] Regression test toolbox sandbox registration, execution, environment realization, repair, reconcile, and GC.

## Acceptance Criteria

- [ ] Backend no longer needs to spawn Python workflow helpers locally with `subprocess.run(...)`.
- [ ] Python helpers execute through hosting worker IPC/RPC with `execute_workflow_python_helper`.
- [ ] Python helper execution uses `module_source`, not caller-provided executable file paths.
- [ ] Python helper `source_path` is preserved only as provenance.
- [ ] Python helper workers use `executor_kind = "workflow_python_helper"`.
- [ ] Python helper workers reuse the shared runtime environment manager.
- [ ] Python and JS helper resources expose the normalized helper pool shape.
- [ ] JS helper existing client-visible resource fields remain compatible.
- [ ] Python helper auth boundary mirrors JS helper auth boundary.
- [ ] Interactive CLI exercises the same public APIs that clients use for helper resources, capacity, and cancellation.
- [ ] Client migration guidance is complete in [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md).
