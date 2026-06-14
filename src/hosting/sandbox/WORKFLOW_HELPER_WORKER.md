# Workflow Helper Worker

Date: 2026-05-21
Scope: workflow helper worker contracts that reuse hosting worker, sandbox, lifecycle, and IPC infrastructure.

## Purpose

Workflow helper workers execute short, dynamic workflow helper code through the hosting worker model instead of letting dependent projects spawn runtimes directly.

V1 includes JavaScript and Python helper executors:

1. worker module: `hosting.workflow_js_helper_ipc`
2. executor kind: `workflow_js_helper`
3. worker profile class: `generic`
4. execution contract: `hosting.workflow_helper.worker.v1`
5. sandbox profile: `workflow_js_helper_v1`
6. worker module: `hosting.workflow_python_helper_ipc`
7. executor kind: `workflow_python_helper`
8. worker profile class: `generic`
9. execution contract: `hosting.workflow_helper.worker.v1`
10. sandbox profile: `workflow_python_helper_v1`

The worker is not a logical toolbox and does not participate in toolbox tool routing.
New integrations should use the workflow runtime facade commands:
`workflow-python-*` for Python and `workflow-js-*` for JavaScript. The old
`workflow_python_helper` and `workflow_js_helper` command/channel surfaces have
been removed after dependent-client migration; their IPC modules remain internal
worker entrypoints behind the facades.

Python helper requests accept `python.import_allowlist`, `python.package_pins`, and `python.environment_name` for shared runtime environment intent. The default helper environment name is `workflow-python-helper`; node-profile workflow Python uses `workflow-python-node` unless a caller supplies a specific environment name.

## Host Lifecycle

Callers should use `workflow-js-ensure`, `workflow-python-ensure`, and the matching workflow facade methods on `EngineHostControlChannel`. Do not use raw `spawn` for workflow helpers; raw process launch is a higher-trust host operation.

The convenience API uses:

```text
python -m hosting.workflow_js_helper_ipc
python -m hosting.workflow_python_helper_ipc
```

and persists a normal worker registration with:

1. `executor_kind = "workflow_js_helper"` or `executor_kind = "workflow_python_helper"`
2. `worker_profile_class = "generic"`
3. `sandbox_policy`
4. `sandbox_runtime`
5. IPC family/address and worker auth metadata
6. workflow helper capabilities

Existing hosting lifecycle APIs such as status, shutdown, and ensure-running apply to this worker like other hosted workers.

## Concurrency Model

The JS helper follows the same bounded hosting-worker pattern as toolbox sandboxes, but with a simpler v1 topology:

1. The worker IPC listener accepts multiple concurrent connections.
2. Each RPC connection is handled on a short-lived host thread.
3. `execute_workflow_js_helper` is guarded by `MP13_WORKFLOW_JS_HELPER_CAPACITY`.
4. The Python hosting worker owns a hot pool of up to `capacity` Node child processes.
5. Each admitted call checks out one hot Node process, imports the helper source by a per-request data URL, executes one named export, returns JSON, and puts the Node process back into the pool.
6. Node child processes are recycled after `MP13_WORKFLOW_JS_HELPER_MAX_REQUESTS_PER_NODE` calls, and terminated when the hosting worker exits, when a call times out, or when a specific request is canceled.
7. When all call slots are in use, the worker returns `workflow_sandbox_capacity_exceeded`.

`workflow-js-ensure(capacity=N)` sets `MP13_WORKFLOW_JS_HELPER_CAPACITY=N` on the internal worker and records the capacity in worker capabilities. The default is `1`, which gives a bounded serialized lane with one hot Node child process. Increase it only for short helpers where parallel Node child processes are acceptable for the host.

Each Node child defaults to a maximum of 256 requests before recycling. Recycling bounds long-lived module-cache growth from per-request data URL imports without changing the client contract.

Toolbox sandboxes add another layer: the client-side harness can route calls across a pool of toolbox executor registrations and uses async gather/round-robin routing. JS helper v1 does not need a toolbox-style registry/pool for correctness because helper calls are source-in, JSON-out, and isolated per call. If throughput requires it later, the same pattern can be added by registering multiple `executor_kind = "workflow_js_helper"` workers and routing by capacity/busy state.

Live pool state is available through `workflow-js-resources`, `workflow-python-resources`, or the matching channel methods. The response reports capacity, active calls, available slots, process counts, process ids, active request ids, per-process request counts, and per-process CPU/RSS when the host can sample the child process. JS responses keep `node_pool`, `workflow_js_node_process_count`, and related compatibility fields while also exposing the normalized `pool` shape.

Capacity can be changed for a loaded worker through `workflow-js-set-capacity` / `workflow-python-set-capacity` or the matching workflow channel methods. Increasing capacity allows the worker to create more hot child processes on demand. Decreasing capacity prevents new children above the new limit and retires idle excess children; active calls are allowed to finish.

Specific active requests can be canceled through `workflow-js-cancel-request`, `workflow-python-cancel-request`, or the matching channel methods. Callers should provide `request_id` in helper execution calls when they need this control. Cancellation kills the child process that owns that request and removes it from the pool; the worker creates a fresh hot child later if capacity requires it.

Environment-keyed workflow runtime commands provide the preferred management
surface:

1. `workflow-python-ensure`, `workflow-python-execute`, `workflow-python-resources`, `workflow-python-set-capacity`, `workflow-python-request-status`, and `workflow-python-cancel-request`
2. `workflow-python-stream-open`, `workflow-python-stream-recv`, `workflow-python-stream-send`, and `workflow-python-stream-close`
3. `workflow-js-ensure`, `workflow-js-execute`, `workflow-js-resources`, `workflow-js-set-capacity`, `workflow-js-request-status`, and `workflow-js-cancel-request`

These commands route and report by host-derived `environment_key` so different
runtime, dependency, or sandbox-policy identities do not share the same
host-side pool.

## Default JS Sandbox Policy

The default v1 policy is:

```json
{
  "sandbox": {
    "enabled": true,
    "profile": "workflow_js_helper_v1",
    "process": {
      "allow_subprocess": false
    },
    "network": {
      "mode": "disabled"
    },
    "brokered_io": {
      "filesystem": false,
      "http": false,
      "subprocess": false
    }
  }
}
```

Helper-visible filesystem, HTTP, direct network, and subprocess access are not part of v1. The host may use internal temporary staging as an implementation detail.

## JS RPC Contract

Call method:

```text
execute_workflow_js_helper
```

Request fields:

1. `module_source`: JavaScript module source text
2. `module_sha256`: SHA-256 hex digest of `module_source`
3. `package_id`
4. `workflow_id`
5. `package_source_digest`
6. `request_id`: optional stable caller id for cancellation/status correlation
7. `export_name`
8. `operation`
9. `payload`: JSON value
10. `provenance`: session/context/cursor/workflow root ids
11. `limits`: timeout, output limit, and memory limit request

The public contract accepts source text, not caller-provided file paths. The worker verifies `sha256(module_source) == module_sha256` before execution.

Allowed operations:

1. `default`
2. `condition`
3. `evaluate_condition`
4. `routing_hint`
5. `route_hint`
6. `payload`
7. `shape_payload`

Input and output are JSON only.

## Python RPC Contract

Call method:

```text
execute_workflow_python_helper
```

Request fields match the JS helper contract and add:

1. `source_path`: provenance only; the worker never executes caller-provided file paths
2. `python.import_allowlist`: declared helper import intent
3. `python.package_pins`: deterministic dependency intent
4. `python.environment_name`: runtime environment identity, defaulting to `workflow-python-helper`

The public contract accepts `module_source`, not an executable path. The worker verifies `sha256(module_source) == module_sha256`, executes only the requested function name, requires JSON input/output, applies per-call timeout/output limits, and reports memory limit enforcement as best-effort unavailable when the platform/runtime does not enforce it.

Allowed operations are the same as JS: `default`, `condition`, `evaluate_condition`, `routing_hint`, `route_hint`, `payload`, and `shape_payload`.

## Workflow Python Node Profile

`workflow_python(profile=node)` uses the same hosted Python execution runtime but
returns the node-profile envelope:

1. `workflow-python-execute` with `profile=node` runs the requested Python export and returns `output`, `state_patch`, `artifacts`, `progress`, `logs`, `metrics`, structured `error`, and `audit`.
2. `workflow-python-stream-open` returns immediately and starts background execution.
3. Stream events use the shared event names: `started`, `log`, optional `progress`, `artifact`, `result` or `error`, `canceled`, and `done`.
4. `workflow-python-stream-send` accepts `{"action":"cancel","request_id":"..."}` and routes cancellation through host request tracking and the worker cancel hook.
5. Dependency-bearing requests require a prepared and verified runtime environment. Normal execution does not install dependencies implicitly.

Artifact references are part of the response contract. The current node
implementation provides a local host-provisioned artifact store: input artifact
refs resolve to request-scoped sandbox paths, output artifact slots resolve to
exact writable sandbox paths, and only the host may register output files as
`workflow-artifact://...` refs. Returned values such as `{"path": "..."}`,
`{"url": "..."}`, or `{"artifact_id": "..."}` remain ordinary JSON unless the
host artifact manager validates and mints the ref.

## Result Contract

Success:

```json
{
  "ok": true,
  "result": {},
  "runtime": {
    "worker_id": "...",
    "engine_id": "...",
    "node_version": "...",
    "sandbox_profile": "workflow_js_helper_v1"
  }
}
```

Failure:

```json
{
  "ok": false,
  "reason": "workflow_sandbox_timeout",
  "detail": {},
  "runtime": {
    "worker_id": "...",
    "engine_id": "...",
    "node_version": "..."
  }
}
```

Stable failure reasons include:

1. `workflow_sandbox_invalid_module_identity`
2. `workflow_sandbox_operation_not_allowed`
3. `workflow_sandbox_export_not_found`
4. `workflow_sandbox_timeout`
5. `workflow_sandbox_canceled`
6. `workflow_sandbox_output_limit_exceeded`
7. `workflow_sandbox_invalid_json_output`
8. `workflow_sandbox_invalid_result_shape`
9. `workflow_sandbox_runtime_error`
10. `workflow_sandbox_host_unavailable`
11. `workflow_sandbox_capacity_exceeded`

## Current Limits

1. The worker uses a bounded serialized call lane by default. `workflow-js-ensure(capacity=N)` enables bounded in-worker parallelism.
2. Memory limit reporting is best-effort and currently reports unavailable enforcement.
3. V1 is for short helper calls only, not long-running jobs or general Node app hosting.
4. Audit/provenance is returned in the per-call result. There is no persistent audit sink yet.
5. Logs and status output should not include raw helper source, payload, or result data unless a future explicit audit sink adds redaction controls.
