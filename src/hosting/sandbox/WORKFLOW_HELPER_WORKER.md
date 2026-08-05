# Workflow Helper Worker

Date: 2026-06-15
Scope: Python helper-profile workflow workers that reuse hosting worker,
sandbox, lifecycle, and IPC infrastructure.

## Purpose

Workflow helper workers execute short, dynamic workflow helper code through the
hosting worker model instead of letting dependent projects spawn runtimes
directly.

This document now covers Python helper-profile execution only:

1. worker module: `hosting.workflow_python_helper_ipc`
2. executor kind: `workflow_python_helper`
3. worker profile class: `generic`
4. execution contract: `hosting.workflow_helper.worker.v1`
5. sandbox profile: `workflow_python_helper_v1`

Workflow JavaScript no longer uses a helper worker. JS workflow execution is
QuickJS node-backed through `hosting.sandbox.workflow_js_node_runtime` and the
child harness `hosting.workflow_js_node_worker_ipc`. See
[JS_NODE_WORKER.md](JS_NODE_WORKER.md).

## Host Lifecycle

Callers should use `workflow-python-ensure` and the matching workflow facade
methods on `EngineHostControlChannel`. Do not use raw `spawn` for workflow
helpers; raw process launch is a higher-trust host operation.

The Python helper facade persists a normal worker registration with:

1. `executor_kind = "workflow_python_helper"`
2. `worker_profile_class = "generic"`
3. `sandbox_policy`
4. `sandbox_runtime`
5. IPC family/address and worker auth metadata
6. workflow helper capabilities

Existing hosting lifecycle APIs such as status, shutdown, and ensure-running
apply to this worker like other hosted workers.

## Workflow Runtime Facade

Environment-keyed workflow runtime commands provide the preferred management
surface:

1. `workflow-python-ensure`
2. `workflow-python-execute`
3. `workflow-python-resources`
4. `workflow-python-set-capacity`
5. `hosted-operation-status` with the execute result's complete operation ref
6. `hosted-operation-cancel` with that same ref

Resource and capacity commands route by host-derived `environment_key` so
different runtime, dependency, or sandbox-policy identities do not share the
same host-side pool. Durable status and cancellation route only from the stored
operation identity in the complete ref.

## Python RPC Contract

The helper worker RPC method remains:

```text
execute_workflow_python_helper
```

Request fields:

1. `module_source`: Python source text
2. `module_sha256`: SHA-256 hex digest of `module_source`
3. `package_id`
4. `workflow_id`
5. `package_source_digest`
6. `request_id`: optional stable caller id for cancellation/status correlation
7. `source_path`: provenance only; the worker never executes caller-provided file paths
8. `export_name`
9. `operation`
10. `payload`: JSON value
11. `provenance`: session/context/cursor/workflow root ids
12. `limits`: timeout, output limit, and memory limit request
13. `python.import_allowlist`: declared helper import intent
14. `python.package_pins`: deterministic dependency intent
15. `python.environment_name`: runtime environment identity, defaulting to `workflow-python-helper`

The public contract accepts `module_source`, not an executable path. The worker
verifies `sha256(module_source) == module_sha256`, executes only the requested
function name, requires JSON input/output, applies per-call timeout/output
limits, and reports memory limit enforcement as best-effort unavailable when the
platform/runtime does not enforce it.

Allowed operations:

1. `default`
2. `condition`
3. `evaluate_condition`
4. `routing_hint`
5. `route_hint`
6. `payload`
7. `shape_payload`

## Workflow Python Node Profile

`workflow_python(profile=node)` is not helper-backed. It has a separate
first-class node contract for `output`, `state_patch`, artifacts, streaming,
progress, logs, metrics, structured errors, and audit fields.

See [PY_NODE_WORKER.md](PY_NODE_WORKER.md) for the node-profile execution and
artifact contract.

## JavaScript Workflow

JavaScript workflow requests use `workflow_js(profile=node)`.

See [JS_NODE_WORKER.md](JS_NODE_WORKER.md) for the QuickJS-backed contract,
host API, artifact rules, streaming events, and non-goals.
