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

## Host Lifecycle

Callers should use `EngineHostService.spawn_workflow_js_helper(...)` or spawn the same command through the existing host/channel spawn surfaces.

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

1. The worker uses a bounded serialized call lane by default.
2. Memory limit reporting is best-effort and currently reports unavailable enforcement.
3. V1 is for short helper calls only, not long-running jobs or general Node app hosting.
