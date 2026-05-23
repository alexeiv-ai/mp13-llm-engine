# Hosting Status And Work Plan

Date: 2026-05-22

This file tracks the implementation plan for adding a hosted workflow Python helper executor lane and normalizing workflow helper pool reporting. Check items only when implementation, tests, and related docs are complete.

## Request Summary

- [x] Add a dedicated hosted Python helper executor lane now that a concrete backend caller contract exists.
- [x] Model the Python helper lane after the existing workflow JS helper worker where the lifecycle and API shape fit.
- [x] Reuse the shared runtime environment manager already added for `workflow_python_helper`; do not add a Python-helper-specific venv manager.
- [x] Keep this separate from toolbox execution, model execution, and unrestricted generic Python execution.
- [x] Preserve the JS helper API while adding generic helper pool aliases so backend code can manage JS and Python helpers through one controller.

## Architecture Decisions

- [x] Use `executor_kind = "workflow_python_helper"` for persisted registrations and routing.
- [x] Use `worker_profile_class = "generic"` unless implementation proves a narrower profile class is required.
- [x] Use a dedicated worker module, likely `hosting.workflow_python_helper_ipc`, instead of extending model-oriented worker IPC.
- [x] Use execution contract `hosting.workflow_helper.worker.v1` unless Python needs a strictly versioned sub-contract.
- [x] Use sandbox profile `workflow_python_helper_v1`.
- [x] Reuse `EngineHostService.spawn(...)`, `WorkerSandboxPolicy`, persisted worker registration, sandbox runtime reporting, hosting IPC/RPC, lifecycle, shutdown, and ensure-running behavior.
- [x] Keep Python helper execution out of toolbox registry/tool routing unless a helper is intentionally exposed later as a toolbox tool.
- [x] Keep Python helper execution out of raw generic process spawn; raw process spawn remains `config_editor`/`admin` only.

## Runtime Environment Integration

- [x] Accept Python environment requirements in request field `python`.
- [x] Support `python.import_allowlist` as declared helper import intent.
- [x] Support `python.package_pins` as deterministic dependency intent.
- [x] Support `python.environment_name`, defaulting to `workflow-python-helper` when omitted.
- [x] Map Python helper environment identity to the existing shared runtime environment manager using `workflow_python_helper` as the consumer/owner kind.
- [x] Resolve or realize the requested runtime environment before executing helper code when package requirements are present.
- [x] Use the verified realized environment Python when available.
- [x] Use the existing preverified/bootstrap Python path only as a verification gate when the environment is not yet eligible for activation.
- [x] Report runtime Python path/source in resources and per-call runtime data without using fallback terminology.
- [x] Preserve existing `runtime_envs` and legacy readable `toolbox_venvs` compatibility behavior from the shared environment manager.

## Python Helper Worker

- [x] Add `hosting.workflow_python_helper_ipc`.
- [x] Add hot Python child process pool under one hosting worker id.
- [x] Make worker `capacity` mean maximum hot Python child processes owned by the worker.
- [x] Recycle child processes after a bounded request count to limit module/cache growth.
- [x] Support per-request cancellation by `request_id` by killing the child process that owns the active request.
- [x] Return `workflow_sandbox_capacity_exceeded` immediately when all slots are in use.
- [x] Report active request ids, process ids, request counts, busy/idle state, and per-child CPU/RSS when available.
- [x] Ensure worker shutdown terminates all hot Python child processes.
- [x] Ensure timeout terminates the child process running the timed-out call.

## Python Execution Contract

- [x] Implement RPC method `execute_workflow_python_helper`.
- [x] Accept `module_source` as the public source input.
- [x] Verify `sha256(module_source) == module_sha256` before execution.
- [x] Accept `source_path` only as provenance; do not execute caller-provided file paths.
- [x] Execute only the requested named export/function.
- [x] Restrict operations to `default`, `condition`, `evaluate_condition`, `routing_hint`, `route_hint`, `payload`, and `shape_payload`.
- [x] Require JSON-only input payloads.
- [x] Require JSON-only output results.
- [x] Enforce `limits.timeout_ms` per call.
- [x] Enforce `limits.output_limit_bytes` per call.
- [x] Report `limits.memory_limit_mb` behavior, including whether enforcement is active, best-effort, or unavailable.
- [x] Prevent helper-visible filesystem, network, brokered I/O, and subprocess access in the v1 sandbox policy.
- [x] Avoid logging raw helper source, payload, or result data.

## Python Result And Error Mapping

- [x] Return success shape with `ok: true`, `result`, `runtime`, and `audit`.
- [x] Return failure shape with `ok: false`, `reason`, `detail`, `runtime`, and `audit`.
- [x] Preserve or map `workflow_sandbox_invalid_module_identity`.
- [x] Preserve or map `workflow_sandbox_operation_not_allowed`.
- [x] Preserve or map `workflow_sandbox_export_not_found`.
- [x] Preserve or map `workflow_sandbox_timeout`.
- [x] Preserve or map `workflow_sandbox_canceled`.
- [x] Preserve or map `workflow_sandbox_output_limit_exceeded`.
- [x] Preserve or map `workflow_sandbox_invalid_json_output`.
- [x] Preserve or map `workflow_sandbox_invalid_result_shape`.
- [x] Preserve or map `workflow_sandbox_runtime_error`.
- [x] Preserve or map `workflow_sandbox_host_unavailable`.
- [x] Preserve or map `workflow_sandbox_capacity_exceeded`.
- [x] Add Python-specific stable reasons only if they are necessary and documented for clients.

## Audit And Provenance

- [x] Carry `package_id`.
- [x] Carry `workflow_id`.
- [x] Carry `package_source_digest`.
- [x] Carry `module_sha256`.
- [x] Carry `source_path` as provenance only.
- [x] Carry `request_id`.
- [x] Carry `operation` and `export_name`.
- [x] Carry `provenance.session_id`.
- [x] Carry `provenance.context_id`.
- [x] Carry `provenance.cursor_id`.
- [x] Carry `provenance.workflow_root_id`.
- [x] Record elapsed milliseconds.
- [x] Record worker id and engine id.
- [x] Record runtime Python path/source.
- [x] Record failure reason.

## Public Service And Channel API

- [x] Add `EngineHostService.spawn_workflow_python_helper(...)`.
- [x] Add `EngineHostControlChannel.spawn_workflow_python_helper(...)`.
- [x] Add `EngineHostService.workflow_python_helper_resources(...)`.
- [x] Add `EngineHostControlChannel.workflow_python_helper_resources(...)`.
- [x] Add `EngineHostService.set_workflow_python_helper_capacity(...)`.
- [x] Add `EngineHostControlChannel.set_workflow_python_helper_capacity(...)`.
- [x] Add `EngineHostService.cancel_workflow_python_helper_request(...)`.
- [x] Add `EngineHostControlChannel.cancel_workflow_python_helper_request(...)`.
- [x] Add daemon dispatch for all new workflow Python helper commands.
- [x] Add non-interactive CLI commands for spawn, resources, capacity, and cancel.
- [x] Update non-interactive CLI help/output paths in `engine_host_cli.py` for the new workflow Python helper commands.
- [x] Ensure `discover-running` identifies Python helpers as workflow helper sandboxes/workers and includes process resources.

## Normalized Helper Pool Resources

- [x] Add generic helper pool aliases to JS helper resources without removing existing Node-specific fields.
- [x] Return top-level `capacity`.
- [x] Return top-level `active_calls`.
- [x] Return top-level `available_slots`.
- [x] Return `pool.process_count`.
- [x] Return `pool.active_process_count`.
- [x] Return `pool.idle_process_count`.
- [x] Return `pool.active_request_ids`.
- [x] Return `pool.processes`.
- [x] Include per-process `pid`, `alive`, `busy`, `active_request_id`, `request_count`, `max_requests`, `reusable`, and `resources` when available.
- [x] Add the same normalized pool shape to Python helper resources.
- [x] Keep JS compatibility fields such as `node_pool`, `workflow_js_node_process_count`, and `workflow_js_node_pids`.
- [x] Add Python compatibility fields only if needed for operator diagnostics.
- [x] Update `discover-running` flattened resource fields for Python helpers.
- [x] Update `discover-running` flattened resource fields for JS helpers to include normalized helper pool aliases.

## Authorization

- [x] Allow `workflow-python-helper-resources` to `diagnostic_user` and above.
- [x] Allow `spawn-workflow-python-helper` to `worker_user`, `config_editor`, and `admin`.
- [x] Allow `workflow-python-helper-set-capacity` to `worker_user`, `config_editor`, and `admin`.
- [x] Allow `workflow-python-helper-cancel-request` to `worker_user`, `config_editor`, and `admin`.
- [x] Keep execution through traffic-scoped `proxy_rpc_call` for the registered helper `engine_id`.
- [x] Allow `model_user` traffic proxy only when scoped to the specialized `executor_kind = "workflow_python_helper"` helper engine.
- [x] Keep raw `spawn` restricted to `config_editor` and `admin`.
- [x] Keep toolbox mutation authority unchanged.

## Interactive CLI

- [x] Show workflow Python helpers in the existing engines/sandboxes inventory.
- [x] Show Python helper details in resource details, including normalized pool state and per-child metrics.
- [x] Generalize `Manage workflow JS helpers` into a workflow helper management view, or add a sibling `Manage workflow Python helpers` view if that is clearer.
- [x] Use the same public channel/CLI APIs in the interactive workflow helper management view that external clients use.
- [x] Support resource refresh for Python helpers.
- [x] Support capacity changes for Python helpers.
- [x] Support canceling a specific active Python helper request by `request_id`.
- [x] Keep JS helper management working through the same normalized resource path.

## Documentation

- [x] Update [sandbox/WORKFLOW_HELPER_WORKER.md](sandbox/WORKFLOW_HELPER_WORKER.md) with the Python helper lane.
- [x] Update [sandbox/GENERIC_WORKER.md](sandbox/GENERIC_WORKER.md) with the Python helper executor decision.
- [x] Update [sandbox/SANDBOX_ARCHITECTURE.md](sandbox/SANDBOX_ARCHITECTURE.md) if shared helper pool/resource concepts need architecture coverage.
- [x] Update [HOSTING.md](HOSTING.md) with the public Python helper service/channel APIs.
- [x] Update [ENGINE_HOST_CLI.md](ENGINE_HOST_CLI.md) with non-interactive and interactive workflow helper CLI commands.
- [x] Complete [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md) with what dependent projects should stop and start doing after implementation.
- [x] Document that Python helper `source_path` is provenance only and not an execution path.
- [x] Document the normalized helper resource shape shared by JS and Python helpers.
- [x] Document authorization boundaries for Python helper resources, spawn, capacity, cancel, and execution.

## Test Plan

- [x] Unit test Python module identity verification.
- [x] Unit test Python allowed operation filtering.
- [x] Unit test Python missing export behavior.
- [x] Unit test JSON input validation.
- [x] Unit test JSON output validation.
- [x] Unit test timeout behavior.
- [x] Unit test cancellation behavior and `workflow_sandbox_canceled` mapping.
- [x] Unit test output limit behavior.
- [x] Unit test memory limit reporting.
- [x] Unit test capacity exceeded behavior.
- [x] Unit test Python hot process reuse and recycling.
- [x] Unit test Python helper resources and normalized pool shape.
- [x] Unit test JS helper normalized pool aliases while preserving legacy JS fields.
- [x] Unit test service/channel forwarding for Python spawn/resources/capacity/cancel.
- [x] Unit test daemon dispatch for Python helper commands.
- [x] Unit test RBAC for Python helper commands.
- [x] Integration test Python helper worker spawn, RPC execution, resources, capacity change, cancellation, ensure-running, and shutdown.
- [x] Integration test Python helper runtime environment resolution with `import_allowlist`, `package_pins`, and `environment_name`.
- [x] Integration test sandbox policy persistence and sandbox runtime reporting for `workflow_python_helper`.
- [x] Integration test helper-visible filesystem/network/subprocess denial for v1 policy where platform support allows.
- [x] Regression test JS helper execution and resource reporting after normalized pool aliases.
- [x] Regression test toolbox sandbox registration, execution, environment realization, repair, reconcile, and GC.

## Acceptance Criteria

- [x] Backend no longer needs to spawn Python workflow helpers locally with `subprocess.run(...)`.
- [x] Python helpers execute through hosting worker IPC/RPC with `execute_workflow_python_helper`.
- [x] Python helper execution uses `module_source`, not caller-provided executable file paths.
- [x] Python helper `source_path` is preserved only as provenance.
- [x] Python helper workers use `executor_kind = "workflow_python_helper"`.
- [x] Python helper workers reuse the shared runtime environment manager.
- [x] Python and JS helper resources expose the normalized helper pool shape.
- [x] JS helper existing client-visible resource fields remain compatible.
- [x] Python helper auth boundary mirrors JS helper auth boundary.
- [x] Interactive CLI exercises the same public APIs that clients use for helper resources, capacity, and cancellation.
- [x] Client migration guidance is complete in [HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md).
