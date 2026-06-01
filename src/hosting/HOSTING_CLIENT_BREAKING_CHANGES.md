# Hosting Client Breaking Changes And Migration Notes

Date: 2026-06-01

Purpose: track dependent-project changes required by the hosted sandbox runtime refactor. This file should be updated as implementation lands. Entries are written as client-facing stop/start guidance.

## Planned Migration: Workflow Python

- [x] Stop treating `workflow_python_helper` as the long-term primary API.
- [x] Start using `workflow_python` once available.
  - `workflow_python(profile=helper, environment_name=workflow-python-helper)` replaces the current helper lane.
  - `workflow_python(profile=node)` is planned for long-running workflow node execution with streaming responses.
  - Helper-profile host surfaces now exist for environment spec/prepare/lock/verify/install/receipt, ensure, execute, resources, capacity, and cancel.
  - Dependent projects should use the workflow-named facade commands; helper-specific public command/channel/daemon surfaces have been removed after migration.

### Workflow Python API Navigation

Use these command names through `EngineHostControlChannel.invoke_control_command(...)`, `python -m hosting.engine_host_cli --payload-json ...`, or the corresponding typed channel methods where available.

- Environment identity:
  - Old: no standalone helper environment-key API.
  - New: `workflow-python-environment-spec` with `profile`, `environment_name`, `python`, and `sandbox_policy`.
  - Result to store/pass forward: `environment_key`, `environment_key_full`, and `environment`.

- Dependency environment lifecycle:
  - New: `workflow-python-prepare-environment`.
  - New: `workflow-python-lock-environment`.
  - New: `workflow-python-verify-environment`.
  - New: `workflow-python-install-environment` with `allow_execution=true` only from an explicit host environment-management path.
  - New: `workflow-python-verify-install-receipt`.
  - New result field: `install_status` summarizes plan/lock/verification/execution/receipt state without requiring clients to parse raw metadata.

- Worker/pool lifecycle:
  - Old: `spawn-workflow-python-helper`.
  - New: `workflow-python-ensure` with `profile=helper`, optional `environment_key`, `python`, `sandbox_policy`, `capacity`, and optional `engine_id`.
  - New ensure derives the default engine ID from `environment_key`.
  - Internal behavior: helper-profile Python workers still use the existing helper worker entrypoint behind the facade until that implementation is reduced to a thin shim.

- Execute helper-profile workflow code:
  - Old: GUI called `spawn_workflow_python_helper`, then proxy RPC `execute_workflow_python_helper`.
  - New: `workflow-python-execute` with `profile=helper` and request fields `module_source`, `module_sha256`, `package_id`, `workflow_id`, `package_source_digest`, `export_name` or `operation`, `payload`, `provenance`, `limits`, and optional `python`.
  - New result fields to consume: `status`, `ok`, `output`, `result`, `environment_key`, `metrics.workflow_pool`, and `metrics.request`.
  - Internal behavior: `workflow-python-execute` records workflow pool/request metrics, then calls the helper worker RPC through an internal bypass flag.

- Resource/capacity/cancel:
  - Old: `workflow-python-helper-resources`, `workflow-python-helper-set-capacity`, `workflow-python-helper-cancel-request`.
  - New: `workflow-python-resources`, `workflow-python-set-capacity`, `workflow-python-cancel-request`, `workflow-python-request-status`.
  - Preferred selector: `environment_key`; temporary migration selector: annotated `engine_id`.
  - Request status returns the tracked request lifetime record for a specific `environment_key + request_id`, including `latest_progress` and `stream_event_count` once streaming/progress events are recorded.

- Node-profile contract:
  - New contract path: `workflow-python-execute` with `profile=node`.
  - Current behavior: executes the requested Python export through the hosted workflow Python runtime and returns the stable node response contract.
  - Artifact refs are contract fields, but the current node response reports `artifact_store.status=unavailable` and `reason=artifact_store_not_implemented` until a store is wired.
  - Streaming rollout commands are available as `workflow-python-stream-open`, `workflow-python-stream-recv`, `workflow-python-stream-send`, and `workflow-python-stream-close`; stream-open returns immediately and background execution emits `started`, `log`, optional `progress`, `result` or structured `error`, and `done`.
  - Interactive CLI navigation: Workflow Helpers can inspect request status by `environment_key + request_id` and receive `workflow-python` stream events by `stream_id`; this is an operator aid, not the dependent-project integration path.

Minimal helper-profile execute payload:

```json
{
  "profile": "helper",
  "environment_name": "workflow-python-helper",
  "environment_key": "<optional-host-derived-key>",
  "request": {
    "request_id": "req-123",
    "module_source": "def condition(input):\n    return {\"accepted\": True}\n",
    "module_sha256": "<sha256>",
    "package_id": "pkg",
    "workflow_id": "workflow",
    "package_source_digest": "<digest>",
    "operation": "condition",
    "payload": {},
    "provenance": {},
    "limits": {
      "timeout_ms": 5000,
      "output_limit_bytes": 65536,
      "memory_limit_mb": 128
    },
    "python": {
      "import_allowlist": [],
      "package_pins": {}
    }
  }
}
```

- [x] Stop routing workflow Python pools only by `engine_id`.
- [x] Start accepting a host-derived `environment_key`.
  - The host will derive or verify the key from environment name, profile, Python runtime identity, imports, package pins or dependency lock identity, and sandbox policy hash.
  - Workflow Python environment keys now include explicit Python runtime identity from `python.runtime_hash`, `python.python_executable`, `python.bootstrap_python_executable`, or `python.fallback_python_executable` when supplied.
  - Helper-profile workflow Python calls now route host-side pool accounting by `environment_key`.
  - Different helper-profile environment keys now get separate default engine IDs and host-side pool records. Full replacement of the legacy helper worker implementation is still pending.
  - mp13-docs migration: `src/backend/api/workflows/sandbox.py` now derives and passes host environment keys for helper-profile workflow Python execution, resources, capacity, and cancellation.

- [x] Stop assuming `python.package_pins` are installed/enforced merely because they are present in an execute request.
- [x] Start using explicit workflow environment APIs for dependency-bearing environments.
  - Prepare.
  - Lock.
  - Verify.
  - Install when explicitly allowed.
  - Verify install receipt.
  - Execute only against a verified environment when dependencies matter.
  - mp13-docs migration: package pins are passed as environment identity/policy metadata only; this client does not install dependencies from workflow execution.

- [x] Stop using workflow execution code to install dependencies.
- [x] Start treating dependency installation as a host-controlled environment-management operation.
  - Host responsibility: expose and enforce `workflow-python-prepare-environment`, `workflow-python-lock-environment`, `workflow-python-verify-environment`, `workflow-python-install-environment`, and `workflow-python-verify-install-receipt`. Install execution is allowed only through the explicit environment-management command with `allow_execution=true`.
  - Client responsibility when dependencies matter: call the environment lifecycle commands before execute, persist or pass the resulting `environment_key` / install metadata, and execute only after the host reports a verified environment.
  - mp13-docs migration: no client work is missing for the current helper-profile migration because mp13-docs does not request dependency installation or dependency-bearing workflow node execution. It passes package pins/import metadata only as environment identity/policy input and leaves actual dependency realization to the hosting environment-management API when a future feature needs it.

- [x] Stop relying on package ID or workflow ID to isolate process pools.
- [x] Start using package/workflow IDs as provenance/audit fields unless they change runtime dependencies or policy.
  - mp13-docs migration: package/workflow IDs remain request provenance; environment-key derivation comes from hosting runtime identity, Python metadata, pins, and sandbox policy.

- [x] Stop assuming helper responses are the only workflow response shape.
- [x] Start handling the new workflow response envelope.
  - `status` / `ok`.
  - output or helper result.
  - structured error.
  - runtime/environment metadata.
  - metrics.
  - audit metadata.
  - progress/log/artifact events for streaming profiles.
  - Helper-profile `execute_workflow_python` now returns `metrics.workflow_pool` and `metrics.request` in addition to the compatibility helper result.
  - Node-profile returns the same top-level envelope shape for sync execution and streaming terminal events.
  - mp13-docs migration: helper output normalization now preserves workflow envelope environment and metrics metadata while keeping existing helper result semantics.

- [x] Stop omitting `request_id` for cancelable or long-running work.
- [x] Start passing stable `request_id` for request lifetime tracking and cancellation.
  - New status lookup: `workflow-python-request-status` with `environment_key` or annotated `engine_id` plus `request_id`.
  - mp13-docs migration: workflow JS/Python helper requests continue to get stable request IDs before dispatch; cancellation now prefers environment-keyed host calls when available.

## Planned Migration: Workflow JS

- [x] Stop treating `workflow_js_helper` as a separate long-term architecture.
- [x] Start using `workflow_js(profile=helper)` once available.
- [x] Start reading resources/capacity/cancellation state from the same environment-keyed resource model as workflow Python.
- [x] Continue handling JS-specific compatibility fields until they are removed after migration.
  - mp13-docs migration: JS lifecycle, execution, resources, capacity, and cancellation now use the `workflow-js-*` facade. `workflow-js-execute` is the public execution API; the old `execute_workflow_js_helper` worker RPC is host-internal only.

### Workflow JS API Navigation

- Environment identity:
  - Old: no standalone JS helper environment-key API.
  - New: `workflow-js-environment-spec` with `profile=helper`, `environment_name=workflow-js-helper`, optional `node`, and `sandbox_policy`.

- Worker/pool lifecycle:
  - Old: `spawn-workflow-js-helper`.
  - New: `workflow-js-ensure` with optional `environment_key`, `node.node_executable`, `capacity`, `sandbox_policy`, and optional `engine_id`.

- Execute helper-profile workflow code:
  - Old: GUI called `spawn_workflow_js_helper`, then proxy RPC `execute_workflow_js_helper`.
  - New: `workflow-js-execute` with `profile=helper` and request fields `module_source`, `module_sha256`, `package_id`, `workflow_id`, `package_source_digest`, `export_name` or `operation`, `payload`, `provenance`, `limits`, and optional `node`.
  - New result fields to consume: `status`, `ok`, `output`, `result`, `environment_key`, `metrics.workflow_pool`, and `metrics.request`.
  - Internal behavior: the facade currently reuses the existing helper worker RPC after `workflow-js-ensure`; callers should not call that RPC directly.

- Resource/capacity/cancel:
  - Old: `workflow-js-helper-resources`, `workflow-js-helper-set-capacity`, `workflow-js-helper-cancel-request`.
  - New: `workflow-js-resources`, `workflow-js-set-capacity`, `workflow-js-cancel-request`, `workflow-js-request-status`.
  - Preferred selector: `environment_key`; temporary migration selector: annotated `engine_id`.
  - Request status returns the tracked request lifetime record for a specific `environment_key + request_id`, including `latest_progress` and `stream_event_count` once streaming/progress events are recorded.

Minimal JS execute payload:

```json
{
  "profile": "helper",
  "environment_name": "workflow-js-helper",
  "environment_key": "<optional-host-derived-key>",
  "request": {
    "request_id": "req-123",
    "module_source": "export function condition(input) { return { accepted: true }; }",
    "module_sha256": "<sha256>",
    "package_id": "pkg",
    "workflow_id": "workflow",
    "package_source_digest": "<digest>",
    "operation": "condition",
    "payload": {},
    "provenance": {},
    "limits": {
      "timeout_ms": 5000,
      "output_limit_bytes": 65536,
      "memory_limit_mb": 128
    },
    "node": {
      "node_executable": "node"
    }
  }
}
```

## Planned Migration: Resources, Capacity, Metrics

- [x] Stop reading capacity only from a single helper engine ID.
- [x] Start reading capacity and resources by runtime kind and `environment_key`.
  - `workflow-python-resources` now reports `workflow_pool` when a helper-profile pool has been ensured.
- [x] Start consuming latency and concurrency metrics.
  - Queue wait ms.
  - Execution latency ms.
  - Total request lifetime ms.
  - Active calls.
  - Available slots.
  - Saturation count.
  - Timeout count.
  - Cancellation count.
  - Recent request outcomes.
  - Latest progress snapshots through `workflow-python-request-status` and `workflow-js-request-status`.
  - mp13-docs migration: helper execution now preserves workflow envelope metrics in runtime metadata for downstream diagnostics.

- [x] Stop assuming capacity changes apply globally to all workflow Python helpers.
- [x] Start setting capacity per `environment_key`.
  - `workflow-python-set-capacity` accepts `environment_key`; during migration it can also infer the key from an annotated helper registration when `engine_id` is supplied.
  - mp13-docs migration: capacity controller now calls `workflow-python-set-capacity` and `workflow-js-set-capacity` with `environment_key` when host support is present.

## Planned Migration: Streaming

- [x] Stop using sync helper execution for long-running workflow node work.
  - Client responsibility when long-running node-profile work is introduced: use `workflow-python-stream-open/recv/send/close` instead of synchronous helper execution.
  - mp13-docs migration: no client work is missing for the current helper-profile migration because this project executes short helper-profile workflow modules only.
- [x] Start using streaming APIs for `workflow_python(profile=node)`.
  - Host responsibility: provide workflow Python stream-open/recv/send/close commands and shared stream event types.
  - Client responsibility when node-profile streaming is adopted: open a stream, receive progress/log/artifact/result/error events, send cancel when needed, and close the stream.
  - mp13-docs migration: no node-profile streaming client path exists in this repo today, so this is not required to complete the helper-profile migration.
  - Open stream.
  - Receive progress/log/artifact/result/error events.
  - Send cancel.
  - Close stream.

- [x] Start tolerating partial progress and terminal events as separate records.
  - Client responsibility when streaming is adopted: do not assume result/error arrives in the same record as progress/log output.
  - mp13-docs migration: no client work is missing for helper-profile execution; this belongs to future node-profile integration.
- [x] Start handling node-profile execution envelopes during rollout.
  - Shared stream event type names are now centralized in `hosting.sandbox.runtime_base.HOSTED_STREAM_EVENT_TYPES`; cancel control messages use `{"action":"cancel","request_id":"..."}`.

## Planned Migration: CLI And Interactive CLI

- [x] Stop scripting only old commands after new workflow commands are available.
  - mp13-docs migration: no client work is missing here because this repo did not have hosting CLI scripts using the removed helper-only command names.
- [x] Start using new workflow commands for new integrations.
- [x] Old helper commands have been removed from public CLI/channel/daemon/auth surfaces after migration.
  - Internal helper worker entrypoints remain while the facade reuses the current worker implementations.
- [x] Interactive CLI screens will move from helper-only views to workflow runtime pool views keyed by environment.
  - Annotated Python and JS helper registrations now use `workflow-python-*` / `workflow-js-*` facade calls inside the workflow helper management screen.
  - Operators can choose "Ensure workflow runtime" for an existing Python helper to annotate it and switch the screen to environment-keyed resources.

## Removal Candidates After Migration

- [x] Remove or reduce `workflow_python_helper_ipc.py` to a thin compatibility entrypoint.
  - Host responsibility: this is mp13-llm-engine-internal cleanup. Dependent projects should not import or call the helper IPC module directly.
  - Marked in code as a temporary compatibility worker; do not add new public host-facing behavior there.
- [x] Remove or reduce `workflow_js_helper_ipc.py` to a thin compatibility entrypoint.
  - Host responsibility: this is mp13-llm-engine-internal cleanup. Dependent projects should use `workflow-js-execute`, not `execute_workflow_js_helper`.
- [x] Remove old helper-specific service branches once dependent projects use workflow runtime APIs.
  - Host responsibility: public helper command/channel/daemon/auth surfaces were removed; remaining service worker branches are internal implementation details behind the facades.
- [x] Remove compatibility response fields only after clients confirm migration.
  - Client expectation: migrated clients should consume the workflow envelope and treat compatibility-shaped helper results as nested response data, not as the API contract.

## Removed Public Command Names After Migration

- Python helper commands are no longer public host commands:
  - `spawn-workflow-python-helper`
  - `workflow-python-helper-resources`
  - `workflow-python-helper-set-capacity`
  - `workflow-python-helper-cancel-request`

- JS helper commands are no longer public host commands:
  - `spawn-workflow-js-helper`
  - `workflow-js-helper-resources`
  - `workflow-js-helper-set-capacity`
  - `workflow-js-helper-cancel-request`

- Integrations should use:
  - `workflow-python-*` for Python workflow runtimes.
  - `workflow-js-*` for JS workflow runtimes.

## Client Action Checklist

- [x] Add client-side support for host-derived `environment_key`.
- [x] Add client-side support for workflow runtime kind/profile fields.
- [x] Add client-side support for streaming workflow node responses.
  - Required only for clients that execute long-running `workflow_python(profile=node)` work. The current mp13-docs helper-profile migration is complete without this because no node-profile streaming client path exists yet.
- [x] Add client-side support for environment prepare/lock/verify/install flows.
  - Required only for clients that request dependency-bearing workflow environments. The host owns the lifecycle commands and install enforcement; those clients must call the commands and wait for host verification before execute.
  - The current mp13-docs helper-profile migration is complete without client-side install orchestration because mp13-docs does not trigger dependency installation from workflow execution.
- [x] Add client-side support for per-environment resource/capacity views.
- [x] Add client-side support for request lifetime and cancellation state.
- [x] Keep schema validation and workflow authorization in the GUI/backend.
- [x] Let host enforce runtime isolation, environment routing, and sandbox policy.
