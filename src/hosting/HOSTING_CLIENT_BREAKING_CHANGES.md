# Hosting Client Runtime Contract Checklist

Date: 2026-06-01

Purpose: track dependent-project alignment with the hosted workflow runtime contract. Entries describe the supported client-facing contract for this unreleased integration.

## Workflow Python

- [x] Use `workflow_python(profile=helper, environment_name=workflow-python-helper)` for helper-profile workflow Python execution.
- [x] Use `workflow_python(profile=node, environment_name=workflow-python-node)` for node-profile workflow Python execution when a client owns that integration.
- [x] Use workflow Python facade commands through `EngineHostControlChannel.invoke_control_command(...)`, `python -m hosting.engine_host_cli --payload-json ...`, or typed channel methods.

### Workflow Python API

- Environment identity:
  - `workflow-python-environment-spec` accepts `profile`, `environment_name`, `python`, and `sandbox_policy`.
  - Clients store or pass forward `environment_key`, `environment_key_full`, and `environment`.

- Dependency environment lifecycle:
  - `workflow-python-prepare-environment`
  - `workflow-python-lock-environment`
  - `workflow-python-verify-environment`
  - `workflow-python-install-environment` with `allow_execution=true` only from an explicit host environment-management path
  - `workflow-python-verify-install-receipt`
  - Hosting owns these commands and install enforcement.
  - Normal workflow execution clients do not install dependencies. They pass environment intent and execute against an environment that hosting has already prepared or verified.
  - Only clients that explicitly provide an environment-management UI or orchestration path call these commands and consume `install_status`.

- Worker and pool lifecycle:
  - `workflow-python-ensure` accepts `profile=helper`, optional `environment_key`, `python`, `sandbox_policy`, `capacity`, and optional `engine_id`.
  - Ensure derives the default engine ID from `environment_key`.
  - Helper-profile Python worker implementation details are host-internal.

- Helper-profile execution:
  - `workflow-python-execute` accepts `profile=helper` and request fields `module_source`, `module_sha256`, `package_id`, `workflow_id`, `package_source_digest`, `export_name` or `operation`, `payload`, `provenance`, `limits`, and optional `python`.
  - Clients consume `status`, `ok`, `output`, `result`, `environment_key`, `metrics.workflow_pool`, and `metrics.request`.

- Resource, capacity, request status, and cancellation:
  - `workflow-python-resources`
  - `workflow-python-set-capacity`
  - `workflow-python-cancel-request`
  - `workflow-python-request-status`
  - Preferred selector: `environment_key`.
  - Request status returns the tracked lifetime record for `environment_key + request_id`, including `latest_progress` and `stream_event_count` for streaming/progress records.

- Node-profile execution and streaming:
  - `workflow-python-execute` accepts `profile=node`.
  - The node response contract includes artifact reference fields. Artifact storage can report `artifact_store.status=unavailable` with `reason=artifact_store_not_implemented`.
  - Streaming commands are `workflow-python-stream-open`, `workflow-python-stream-recv`, `workflow-python-stream-send`, and `workflow-python-stream-close`.
  - Stream events include `started`, `log`, optional `progress`, `result` or structured `error`, and `done`.
  - Cancel control messages use `{"action":"cancel","request_id":"..."}`.

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

### Workflow Python Client Checklist

- [x] Route workflow Python pool accounting by `environment_key`.
  - mp13-docs: `src/backend/api/workflows/sandbox.py` derives and passes host environment keys for helper-profile workflow Python execution, resources, capacity, and cancellation.
- [x] Treat `python.package_pins` as environment identity and policy metadata, not as an installation trigger.
  - mp13-docs: package pins and import metadata are passed as environment inputs only.
- [x] Do not install dependencies from workflow execution.
  - mp13-docs: package pins and import metadata are passed as environment intent only; hosting owns any prepare/lock/verify/install operation.
- [x] Call host-controlled environment-management APIs only from an explicit environment-management path.
  - mp13-docs: not applicable for current helper-profile execution because this client does not provide dependency installation UI/orchestration.
- [x] Use package and workflow IDs as provenance/audit fields unless they affect runtime dependencies or policy.
  - mp13-docs: package/workflow IDs remain request provenance.
- [x] Handle the workflow response envelope.
  - Required fields include `status`, `ok`, output/helper result, structured error, runtime/environment metadata, metrics, audit metadata, and streaming event records where applicable.
  - mp13-docs: helper output normalization preserves workflow envelope environment and metrics metadata.
- [x] Pass stable `request_id` for tracked, cancelable, or long-running work.
  - mp13-docs: workflow JS/Python helper requests receive stable request IDs before dispatch; cancellation uses environment-keyed host calls.

## Workflow JS

- [x] Use `workflow_js(profile=helper, environment_name=workflow-js-helper)` for helper-profile workflow JS execution.
- [x] Read resources, capacity, cancellation state, and request status from the environment-keyed resource model.
- [x] Treat JS-specific worker fields as host-internal implementation data.
  - mp13-docs: JS lifecycle, execution, resources, capacity, and cancellation use the `workflow-js-*` facade. `workflow-js-execute` is the public execution API.

### Workflow JS API

- Environment identity:
  - `workflow-js-environment-spec` accepts `profile=helper`, `environment_name=workflow-js-helper`, optional `node`, and `sandbox_policy`.

- Worker and pool lifecycle:
  - `workflow-js-ensure` accepts optional `environment_key`, `node.node_executable`, `capacity`, `sandbox_policy`, and optional `engine_id`.

- Helper-profile execution:
  - `workflow-js-execute` accepts `profile=helper` and request fields `module_source`, `module_sha256`, `package_id`, `workflow_id`, `package_source_digest`, `export_name` or `operation`, `payload`, `provenance`, `limits`, and optional `node`.
  - Clients consume `status`, `ok`, `output`, `result`, `environment_key`, `metrics.workflow_pool`, and `metrics.request`.

- Resource, capacity, request status, and cancellation:
  - `workflow-js-resources`
  - `workflow-js-set-capacity`
  - `workflow-js-cancel-request`
  - `workflow-js-request-status`
  - Preferred selector: `environment_key`.
  - Request status returns the tracked lifetime record for `environment_key + request_id`, including `latest_progress` and `stream_event_count` for streaming/progress records.

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

## Resources, Capacity, Metrics

- [x] Read capacity and resources by runtime kind and `environment_key`.
- [x] Consume latency and concurrency metrics:
  - Queue wait ms
  - Execution latency ms
  - Total request lifetime ms
  - Active calls
  - Available slots
  - Saturation count
  - Timeout count
  - Cancellation count
  - Recent request outcomes
  - Latest progress snapshots through `workflow-python-request-status` and `workflow-js-request-status`
  - mp13-docs: helper execution preserves workflow envelope metrics in runtime metadata for downstream diagnostics.
- [x] Set capacity per `environment_key`.
  - mp13-docs: capacity controller calls `workflow-python-set-capacity` and `workflow-js-set-capacity` with `environment_key`.

## Streaming

- [x] Use streaming APIs for long-running `workflow_python(profile=node)` work when the client owns node-profile execution.
  - mp13-docs: not applicable for current helper-profile execution because this project executes short helper-profile workflow modules only.
- [x] Tolerate progress and terminal events as separate records.
  - mp13-docs: not applicable for current helper-profile execution; this belongs to a node-profile integration.
- [x] Handle node-profile execution envelopes where node-profile execution is present.
  - Shared stream event type names are centralized in `hosting.sandbox.runtime_base.HOSTED_STREAM_EVENT_TYPES`.

## CLI And Interactive CLI

- [x] Use workflow facade commands for integrations.
  - mp13-docs: not applicable for CLI scripting because this repo does not contain hosting CLI scripts.
- [x] Interactive CLI screens use workflow runtime pool views keyed by environment.
  - Python and JS helper registrations use `workflow-python-*` and `workflow-js-*` facade calls inside workflow helper management screens.

## Host-Internal Entry Points

- [x] Keep helper IPC worker modules host-internal.
  - `workflow_python_helper_ipc.py`
  - `workflow_js_helper_ipc.py`
- [x] Keep helper-specific worker branches behind workflow runtime facades.
- [x] Treat helper result payloads as nested response data, not as the public API contract.

## Public Command Names

- Workflow Python integrations use `workflow-python-*`.
- Workflow JS integrations use `workflow-js-*`.

## Client Action Checklist

- [x] Add client-side support for host-derived `environment_key`.
- [x] Add client-side support for workflow runtime kind/profile fields.
- [x] Add client-side support for streaming workflow node responses where node-profile execution is present.
  - mp13-docs: not applicable for current helper-profile execution.
- [x] Avoid client-side dependency installation during workflow execution.
- [x] Add prepare/lock/verify/install orchestration only in clients that explicitly manage workflow environments.
  - mp13-docs: not applicable for current helper-profile execution because it does not provide dependency installation UI/orchestration.
- [x] Add client-side support for per-environment resource/capacity views.
- [x] Add client-side support for request lifetime and cancellation state.
- [x] Keep schema validation and workflow authorization in the GUI/backend.
- [x] Let host enforce runtime isolation, environment routing, and sandbox policy.
