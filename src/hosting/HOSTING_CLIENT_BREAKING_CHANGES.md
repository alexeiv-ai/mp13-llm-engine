# Hosting Client Breaking Changes

Date: 2026-05-21

This document should be updated as hosting/sandbox work lands. It is the client-facing migration guide for dependent projects, including mp13-docs.

## Workflow Helper Sandbox Migration

Status: in progress.

When the workflow helper sandbox work is complete, this section must capture the final migration requirements for dependent projects.

### Stop Doing

- [x] Stop spawning Node directly for dynamic workflow JS helpers.
- [x] Stop passing caller-provided JS file paths for workflow helper execution.
- [x] Stop depending on `toolbox_venvs` as the semantic location for non-toolbox runtime environments.
- [x] Stop treating runtime Python bootstrap/preverified execution as a permanent fallback path.
- [x] Stop routing workflow helper execution through logical toolbox state unless the helper is intentionally exposed as a toolbox tool.
- [x] Stop using raw hosting `spawn` for workflow/helper runtimes unless the caller has config-editor/admin authority for arbitrary process launch.

### Start Doing

- [x] Execute workflow JS helpers through hosting worker IPC/RPC.
- [x] Send helper source as `module_source` plus `module_sha256`.
- [x] Use workflow helper execution contract `hosting.workflow_helper.worker.v1`.
- [x] Use or discover workers registered with `executor_kind = "workflow_js_helper"` for JS helpers.
- [x] Let hosting own internal helper staging paths if staging is required.
- [x] Rely on `WorkerSandboxPolicy`, persisted registration, sandbox runtime, lifecycle/status/shutdown, and ensure-running semantics from hosting.
- [x] Read neutral runtime environment metadata instead of inferring behavior from toolbox-specific directory names.
- [x] Preserve package/workflow/session/context provenance in helper requests.
- [x] Use `spawn_workflow_js_helper(...)` for the narrow JS-helper worker lane; `worker_user` may introduce this helper, while raw process spawn is restricted to `config_editor` and `admin`.
- [x] Execute approved workflow helper calls through `proxy_rpc_call(..., method="execute_workflow_js_helper", ...)` with a traffic-scoped session for the helper `engine_id`.

### Before And After Example

Before:

```python
subprocess.run(["node", helper_path], input=json_payload, ...)
```

After:

```python
host.spawn_workflow_js_helper(engine_id="workflow-js-helper")
host.proxy_rpc_call(
    engine_id="workflow-js-helper",
    method="execute_workflow_js_helper",
    params={
        "module_source": source_text,
        "module_sha256": sha256_hex,
        "package_id": package_id,
        "workflow_id": workflow_id,
        "package_source_digest": package_source_digest,
        "export_name": "condition",
        "operation": "condition",
        "payload": payload,
        "provenance": provenance,
        "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
    },
)
```

Channel clients can use the public wrapper instead of private `_invoke(...)`:

```python
channel.spawn_workflow_js_helper(engine_id="workflow-js-helper")
channel.proxy_rpc_call(
    engine_id="workflow-js-helper",
    method="execute_workflow_js_helper",
    params={...},
)
```

The daemon and CLI spawn paths preserve `worker_profile_class`, so clients do not need to compensate for generic-worker metadata loss.

Authorization boundary:

- Raw `spawn` is for arbitrary process launch and requires `config_editor` or `admin`.
- `spawn_workflow_js_helper(...)` is the constrained JS-helper introduction path and is available to `worker_user`, `config_editor`, and `admin`.
- `execute_workflow_js_helper` is reached through the normal traffic proxy path. A `model_user` session may execute it only when scoped to the registered workflow helper `engine_id`.
- Generic worker proxy remains blocked for `model_user` and `model_user_with_model_control` unless the registered worker is the specialized `executor_kind = "workflow_js_helper"` lane.

Toolbox sandbox authorization boundary:

- `diagnostic_user` is observe-only for hosted toolbox state: describe, gate status, references, consistency, review snapshot, environment list, logs, metrics, and read/stat sandbox filesystem inspection.
- `diagnostic_user` cannot execute, cancel, repair, reconcile, register/unregister toolboxes, mutate toolbox environments, run environment install phases, write sandbox files, create sandbox directories, or broker sandbox HTTP.
- `worker_user` and above can manage hosted toolbox definitions, registrations, repairs/reconciliation, and toolbox environment resolve/apply/realize/install flows.
- Normal end users should use approved hosted tools through the backend service. They do not need direct hosting toolbox mutation authority.

### Compatibility Window

- [x] No compatibility path is planned for direct Node helper spawning. Dependent projects should move to hosted workflow helper RPC before removing their old local execution path.
- [x] Existing `toolbox_venvs` entries remain readable for toolbox environments. New workflow helper runtime environments use `runtime_envs`.
- [x] Client-visible JS helper failures use stable `workflow_sandbox_*` reason values.
