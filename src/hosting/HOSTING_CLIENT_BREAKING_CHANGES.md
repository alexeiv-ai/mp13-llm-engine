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
  - mp13-docs migration: N/A for install flow ownership; the client has no dependency-install execution path and leaves dependency realization to hosting.

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

- [x] Stop using sync helper execution for long-running workflow node work. N/A for mp13-docs: this project currently executes helper-profile workflow modules only and does not own long-running node-profile workflow execution.
- [x] Start using streaming APIs for `workflow_python(profile=node)`. N/A for mp13-docs: no node-profile streaming client path is present in this repo.
  - Open stream.
  - Receive progress/log/artifact/result/error events.
  - Send cancel.
  - Close stream.

- [x] Start tolerating partial progress and terminal events as separate records. N/A for mp13-docs helper-profile migration; streaming event handling belongs to future node-profile integration.
- [x] Start handling node-profile execution envelopes during rollout.
  - Shared stream event type names are now centralized in `hosting.sandbox.runtime_base.HOSTED_STREAM_EVENT_TYPES`; cancel control messages use `{"action":"cancel","request_id":"..."}`.

## Planned Migration: CLI And Interactive CLI

- [x] Stop scripting only old commands after new workflow commands are available. N/A for mp13-docs: no hosting CLI scripts in this repo were using the helper-only commands.
- [x] Start using new workflow commands for new integrations.
- [x] Old helper commands have been removed from public CLI/channel/daemon/auth surfaces after migration.
  - Internal helper worker entrypoints remain while the facade reuses the current worker implementations.
- [x] Interactive CLI screens will move from helper-only views to workflow runtime pool views keyed by environment.
  - Annotated Python and JS helper registrations now use `workflow-python-*` / `workflow-js-*` facade calls inside the workflow helper management screen.
  - Operators can choose "Ensure workflow runtime" for an existing Python helper to annotate it and switch the screen to environment-keyed resources.

## Removal Candidates After Migration

- [x] Remove or reduce `workflow_python_helper_ipc.py` to a thin compatibility entrypoint. N/A for mp13-docs: host-internal removal is owned by mp13-llm-engine.
  - Marked in code as a temporary compatibility worker; do not add new public host-facing behavior there.
- [x] Remove or reduce `workflow_js_helper_ipc.py` to a thin compatibility entrypoint. N/A for mp13-docs: host-internal removal is owned by mp13-llm-engine.
- [x] Remove old helper-specific service branches once dependent projects use workflow runtime APIs. N/A for mp13-docs: public helper command/channel/daemon/auth surfaces were removed; service worker branches remain internal implementation details behind the facades.
- [x] Remove compatibility response fields only after clients confirm migration. N/A for mp13-docs: this client accepts the current workflow envelope and compatibility-shaped helper result nested inside it.

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
- [x] Add client-side support for streaming workflow node responses. N/A for mp13-docs helper-profile migration; no node-profile streaming client exists yet.
- [x] Add client-side support for environment prepare/lock/verify/install flows. N/A for mp13-docs current workflow helpers; dependency installation is host-managed and not triggered by this client.
- [x] Add client-side support for per-environment resource/capacity views.
- [x] Add client-side support for request lifetime and cancellation state.
- [x] Keep schema validation and workflow authorization in the GUI/backend.
- [x] Let host enforce runtime isolation, environment routing, and sandbox policy.
