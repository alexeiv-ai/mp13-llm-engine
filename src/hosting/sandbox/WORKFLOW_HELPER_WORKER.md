# Workflow Helper Worker

Date: 2026-05-21
Scope: workflow helper worker contracts that reuse hosting worker, sandbox, lifecycle, and IPC infrastructure.

## Purpose

Workflow helper workers execute short, dynamic workflow helper code through the hosting worker model instead of letting dependent projects spawn runtimes directly.

V1 includes a JavaScript helper executor:

1. worker module: `hosting.workflow_js_helper_ipc`
2. executor kind: `workflow_js_helper`
3. worker profile class: `generic`
4. execution contract: `hosting.workflow_helper.worker.v1`
5. sandbox profile: `workflow_js_helper_v1`

The worker is not a logical toolbox and does not participate in toolbox tool routing.

Python workflow helper support currently covers realized runtime environment metadata through `workflow_python_helper`. This slice does not define a separate Python worker execution contract; add one only when a concrete caller contract exists.

## Host Lifecycle

Callers should use `EngineHostService.spawn_workflow_js_helper(...)` or `EngineHostControlChannel.spawn_workflow_js_helper(...)`. Do not use raw `spawn` for workflow helpers; raw process launch is a higher-trust host operation.

The convenience API uses:

```text
python -m hosting.workflow_js_helper_ipc
```

and persists a normal worker registration with:

1. `executor_kind = "workflow_js_helper"`
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
6. Node child processes are recycled after `MP13_WORKFLOW_JS_HELPER_MAX_REQUESTS_PER_NODE` calls, and terminated when the hosting worker exits or when a call times out.
7. When all call slots are in use, the worker returns `workflow_sandbox_capacity_exceeded`.

`spawn_workflow_js_helper(capacity=N)` sets `MP13_WORKFLOW_JS_HELPER_CAPACITY=N` and records the capacity in worker capabilities. The default is `1`, which gives a bounded serialized lane with one hot Node child process. Increase it only for short helpers where parallel Node child processes are acceptable for the host.

Each Node child defaults to a maximum of 256 requests before recycling. Recycling bounds long-lived module-cache growth from per-request data URL imports without changing the client contract.

Toolbox sandboxes add another layer: the client-side harness can route calls across a pool of toolbox executor registrations and uses async gather/round-robin routing. JS helper v1 does not need a toolbox-style registry/pool for correctness because helper calls are source-in, JSON-out, and isolated per call. If throughput requires it later, the same pattern can be added by registering multiple `executor_kind = "workflow_js_helper"` workers and routing by capacity/busy state.

Live pool state is available through `workflow-js-helper-resources` or `EngineHostControlChannel.workflow_js_helper_resources(...)`. The response reports capacity, active calls, available slots, Node process counts, Node process ids, and per-node request counts.

Capacity can be changed for a loaded worker through `workflow-js-helper-set-capacity` or `EngineHostControlChannel.set_workflow_js_helper_capacity(...)`. Increasing capacity allows the worker to create more hot Node children on demand. Decreasing capacity prevents new children above the new limit and retires idle excess children; active calls are allowed to finish.

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
6. `export_name`
7. `operation`
8. `payload`: JSON value
9. `provenance`: session/context/cursor/workflow root ids
10. `limits`: timeout, output limit, and memory limit request

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
5. `workflow_sandbox_output_limit_exceeded`
6. `workflow_sandbox_invalid_json_output`
7. `workflow_sandbox_invalid_result_shape`
8. `workflow_sandbox_runtime_error`
9. `workflow_sandbox_host_unavailable`
10. `workflow_sandbox_capacity_exceeded`

## Current Limits

1. The worker uses a bounded serialized call lane by default. `spawn_workflow_js_helper(capacity=N)` enables bounded in-worker parallelism.
2. Memory limit reporting is best-effort and currently reports unavailable enforcement.
3. V1 is for short helper calls only, not long-running jobs or general Node app hosting.
4. Audit/provenance is returned in the per-call result. There is no persistent audit sink yet.
5. Logs and status output should not include raw helper source, payload, or result data unless a future explicit audit sink adds redaction controls.
