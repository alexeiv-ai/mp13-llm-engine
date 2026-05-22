# Hosting Client Breaking Changes

Date: 2026-05-21

This document should be updated as hosting/sandbox work lands. It is the client-facing migration guide for dependent projects, including mp13-docs.

## Workflow Helper Sandbox Migration

Status: planned.

When the workflow helper sandbox work is complete, this section must capture the final migration requirements for dependent projects.

### Stop Doing

- [ ] Stop spawning Node directly for dynamic workflow JS helpers.
- [ ] Stop passing caller-provided JS file paths for workflow helper execution.
- [ ] Stop depending on `toolbox_venvs` as the semantic location for non-toolbox runtime environments.
- [ ] Stop treating runtime Python bootstrap/preverified execution as a permanent fallback path.
- [ ] Stop routing workflow helper execution through logical toolbox state unless the helper is intentionally exposed as a toolbox tool.

### Start Doing

- [ ] Execute workflow JS helpers through hosting worker IPC/RPC.
- [ ] Send helper source as `module_source` plus `module_sha256`.
- [ ] Use the workflow helper execution contract selected by the implementation, expected to be `hosting.workflow_helper.worker.v1`.
- [ ] Use or discover workers registered with `executor_kind = "workflow_js_helper"` for JS helpers.
- [ ] Let hosting own internal helper staging paths if staging is required.
- [ ] Rely on `WorkerSandboxPolicy`, persisted registration, sandbox runtime, lifecycle/status/shutdown, and ensure-running semantics from hosting.
- [ ] Read neutral runtime environment metadata instead of inferring behavior from toolbox-specific directory names.
- [ ] Preserve package/workflow/session/context provenance in helper requests.

### Before And After Example

- [ ] Add an mp13-docs before/after example once the final public API is implemented.

### Compatibility Window

- [ ] Document any temporary compatibility behavior and the target removal point.
- [ ] Document whether existing `toolbox_venvs` entries remain readable, are lazily migrated, or are copied to `runtime_envs`.
- [ ] Document any client-visible status/error changes.
