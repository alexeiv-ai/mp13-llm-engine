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

### Start Doing

- [x] Execute workflow JS helpers through hosting worker IPC/RPC.
- [x] Send helper source as `module_source` plus `module_sha256`.
- [x] Use workflow helper execution contract `hosting.workflow_helper.worker.v1`.
- [x] Use or discover workers registered with `executor_kind = "workflow_js_helper"` for JS helpers.
- [x] Let hosting own internal helper staging paths if staging is required.
- [x] Rely on `WorkerSandboxPolicy`, persisted registration, sandbox runtime, lifecycle/status/shutdown, and ensure-running semantics from hosting.
- [x] Read neutral runtime environment metadata instead of inferring behavior from toolbox-specific directory names.
- [x] Preserve package/workflow/session/context provenance in helper requests.

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

### Compatibility Window

- [x] No compatibility path is planned for direct Node helper spawning. Dependent projects should move to hosted workflow helper RPC before removing their old local execution path.
- [x] Existing `toolbox_venvs` entries remain readable for toolbox environments. New workflow helper runtime environments use `runtime_envs`.
- [x] Client-visible JS helper failures use stable `workflow_sandbox_*` reason values.
