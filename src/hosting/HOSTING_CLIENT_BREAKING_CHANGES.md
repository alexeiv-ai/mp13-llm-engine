# Hosting Client Breaking Changes And Migration Notes

Date: 2026-06-01

Purpose: track dependent-project changes required by the hosted sandbox runtime refactor. This file should be updated as implementation lands. Entries are written as client-facing stop/start guidance.

## Planned Migration: Workflow Python

- [x] Stop treating `workflow_python_helper` as the long-term primary API.
- [x] Start using `workflow_python` once available.
  - `workflow_python(profile=helper, environment_name=workflow-python-helper)` replaces the current helper lane.
  - `workflow_python(profile=node)` is planned for long-running workflow node execution with streaming responses.
  - Helper-profile host surfaces now exist for environment spec/prepare/lock/verify/install/receipt, ensure, execute, resources, capacity, and cancel.
  - Dependent projects should keep old helper calls available until the compatibility migration phase is complete.

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
  - Migration note: old spawn remains available while callers move; new ensure derives the default engine ID from `environment_key`.
  - Compatibility behavior: old Python helper spawn now delegates through `workflow-python-ensure` internally and includes `workflow_runtime_kind`, `workflow_profile`, `environment_key`, `environment`, and `workflow_ensure` alongside the raw spawn fields when it launches a worker.
  - Typed channel compatibility: `EngineHostControlChannel.spawn_workflow_python_helper(...)` now forwards to `workflow-python-ensure` while preserving its old method signature.

- Execute helper-profile workflow code:
  - Old: GUI calls `spawn_workflow_python_helper`, then proxy RPC `execute_workflow_python_helper`.
  - New: `workflow-python-execute` with `profile=helper` and request fields `module_source`, `module_sha256`, `package_id`, `workflow_id`, `package_source_digest`, `export_name` or `operation`, `payload`, `provenance`, `limits`, and optional `python`.
  - New result fields to consume: `status`, `ok`, `output`, `result`, `environment_key`, `metrics.workflow_pool`, and `metrics.request`.

- Resource/capacity/cancel:
  - Old: `workflow-python-helper-resources`, `workflow-python-helper-set-capacity`, `workflow-python-helper-cancel-request`.
  - New: `workflow-python-resources`, `workflow-python-set-capacity`, `workflow-python-cancel-request`, `workflow-python-request-status`.
  - Preferred selector: `environment_key`; temporary migration selector: annotated `engine_id`.
  - Compatibility behavior: old Python helper resource/capacity/cancel calls now include `environment_key`, `workflow_runtime_kind=workflow_python`, and `workflow_pool` when the helper registration has been annotated through `workflow-python-ensure` or `workflow-python-execute`.
  - Request status returns the tracked request lifetime record for a specific `environment_key + request_id`, including `latest_progress` and `stream_event_count` once streaming/progress events are recorded.

- Node-profile contract:
  - New contract path: `workflow-python-execute` with `profile=node`.
  - Current behavior: returns a structured `workflow_python_node_profile_not_implemented` envelope with the stable node request/response contract.
  - Future behavior: same contract will be backed by async/streaming execution.

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

- [ ] Stop routing workflow Python pools only by `engine_id`.
- [ ] Start accepting a host-derived `environment_key`.
  - The host will derive or verify the key from environment name, profile, Python runtime identity, imports, package pins or dependency lock identity, and sandbox policy hash.
  - Workflow Python environment keys now include explicit Python runtime identity from `python.runtime_hash`, `python.python_executable`, `python.bootstrap_python_executable`, or `python.fallback_python_executable` when supplied.
  - Helper-profile workflow Python calls now route host-side pool accounting by `environment_key`.
  - Different helper-profile environment keys now get separate default engine IDs and host-side pool records. Full replacement of the legacy helper worker implementation is still pending.

- [ ] Stop assuming `python.package_pins` are installed/enforced merely because they are present in an execute request.
- [x] Start using explicit workflow environment APIs for dependency-bearing environments.
  - Prepare.
  - Lock.
  - Verify.
  - Install when explicitly allowed.
  - Verify install receipt.
  - Execute only against a verified environment when dependencies matter.

- [ ] Stop using workflow execution code to install dependencies.
- [ ] Start treating dependency installation as a host-controlled environment-management operation.

- [ ] Stop relying on package ID or workflow ID to isolate process pools.
- [ ] Start using package/workflow IDs as provenance/audit fields unless they change runtime dependencies or policy.

- [ ] Stop assuming helper responses are the only workflow response shape.
- [ ] Start handling the new workflow response envelope.
  - `status` / `ok`.
  - output or helper result.
  - structured error.
  - runtime/environment metadata.
  - metrics.
  - audit metadata.
  - progress/log/artifact events for streaming profiles.
  - Helper-profile `execute_workflow_python` now returns `metrics.workflow_pool` and `metrics.request` in addition to the compatibility helper result.
  - Node-profile currently returns the same top-level envelope shape with a structured pending-worker error.

- [ ] Stop omitting `request_id` for cancelable or long-running work.
- [ ] Start passing stable `request_id` for request lifetime tracking and cancellation.
  - New status lookup: `workflow-python-request-status` with `environment_key` or annotated `engine_id` plus `request_id`.

## Planned Migration: Workflow JS

- [x] Stop treating `workflow_js_helper` as a separate long-term architecture.
- [x] Start using `workflow_js(profile=helper)` once available.
- [x] Start reading resources/capacity/cancellation state from the same environment-keyed resource model as workflow Python.
- [ ] Continue handling JS-specific compatibility fields until they are removed after migration.

### Workflow JS API Navigation

- Environment identity:
  - Old: no standalone JS helper environment-key API.
  - New: `workflow-js-environment-spec` with `profile=helper`, `environment_name=workflow-js-helper`, optional `node`, and `sandbox_policy`.

- Worker/pool lifecycle:
  - Old: `spawn-workflow-js-helper`.
  - New: `workflow-js-ensure` with optional `environment_key`, `node.node_executable`, `capacity`, `sandbox_policy`, and optional `engine_id`.

- Resource/capacity/cancel:
  - Old: `workflow-js-helper-resources`, `workflow-js-helper-set-capacity`, `workflow-js-helper-cancel-request`.
  - New: `workflow-js-resources`, `workflow-js-set-capacity`, `workflow-js-cancel-request`, `workflow-js-request-status`.
  - Preferred selector: `environment_key`; temporary migration selector: annotated `engine_id`.
  - Compatibility behavior: old JS helper resource/capacity/cancel calls now include `environment_key`, `workflow_runtime_kind=workflow_js`, and `workflow_pool` when the helper registration has been annotated through `workflow-js-ensure`.
  - Request status returns the tracked request lifetime record for a specific `environment_key + request_id`, including `latest_progress` and `stream_event_count` once streaming/progress events are recorded.

Minimal JS ensure payload:

```json
{
  "profile": "helper",
  "environment_name": "workflow-js-helper",
  "node": {
    "node_executable": "node"
  },
  "capacity": 2,
  "sandbox_policy": {
    "sandbox": {
      "enabled": true,
      "profile": "workflow_js_helper_v1"
    }
  }
}
```

## Planned Migration: Resources, Capacity, Metrics

- [ ] Stop reading capacity only from a single helper engine ID.
- [ ] Start reading capacity and resources by runtime kind and `environment_key`.
  - `workflow-python-resources` now reports `workflow_pool` when a helper-profile pool has been ensured.
- [ ] Start consuming latency and concurrency metrics.
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

- [ ] Stop assuming capacity changes apply globally to all workflow Python helpers.
- [ ] Start setting capacity per `environment_key`.
  - `workflow-python-set-capacity` accepts `environment_key`; during migration it can also infer the key from an annotated helper registration when `engine_id` is supplied.

## Planned Migration: Streaming

- [ ] Stop using sync helper execution for long-running workflow node work.
- [ ] Start using streaming APIs for `workflow_python(profile=node)`.
  - Open stream.
  - Receive progress/log/artifact/result/error events.
  - Send cancel.
  - Close stream.

- [ ] Start tolerating partial progress and terminal events as separate records.
- [x] Start tolerating node-profile pending-worker envelopes during rollout.

## Planned Migration: CLI And Interactive CLI

- [ ] Stop scripting only old commands after new workflow commands are available.
- [x] Start using new workflow commands for new integrations.
- [x] Old helper commands will remain temporary aliases during migration.
  - Python helper resources/capacity/cancel now expose workflow pool metadata for annotated registrations.
- [x] Interactive CLI screens will move from helper-only views to workflow runtime pool views keyed by environment.
  - Annotated Python helper registrations now use `workflow-python-*` facade calls inside the workflow helper management screen.
  - Operators can choose "Ensure workflow runtime" for an existing Python helper to annotate it and switch the screen to environment-keyed resources.

## Removal Candidates After Migration

- [ ] Remove or reduce `workflow_python_helper_ipc.py` to a thin compatibility entrypoint.
- [ ] Remove or reduce `workflow_js_helper_ipc.py` to a thin compatibility entrypoint.
- [ ] Remove old helper-specific service branches once dependent projects use workflow runtime APIs.
- [ ] Remove compatibility response fields only after clients confirm migration.

## Deprecated Command Names During Migration

- Python helper commands are compatibility aliases:
  - `spawn-workflow-python-helper`
  - `workflow-python-helper-resources`
  - `workflow-python-helper-set-capacity`
  - `workflow-python-helper-cancel-request`

- JS helper commands are compatibility aliases:
  - `spawn-workflow-js-helper`
  - `workflow-js-helper-resources`
  - `workflow-js-helper-set-capacity`
  - `workflow-js-helper-cancel-request`

- New integrations should use:
  - `workflow-python-*` for Python workflow runtimes.
  - `workflow-js-*` for JS workflow runtimes.

## Client Action Checklist

- [ ] Add client-side support for host-derived `environment_key`.
- [ ] Add client-side support for workflow runtime kind/profile fields.
- [ ] Add client-side support for streaming workflow node responses.
- [ ] Add client-side support for environment prepare/lock/verify/install flows.
- [ ] Add client-side support for per-environment resource/capacity views.
- [ ] Add client-side support for request lifetime and cancellation state.
- [ ] Keep schema validation and workflow authorization in the GUI/backend.
- [ ] Let host enforce runtime isolation, environment routing, and sandbox policy.
