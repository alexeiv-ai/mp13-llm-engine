# Hosting Client Breaking Changes

Date: 2026-06-22

## Edit And Continue Worker Identity

This slice changes the client model for edited workflow-node code. Clients should treat `instance_id` as the stable logical worker identity. Runtime processes may be reused or replaced underneath that identity depending on worker kind.

### JavaScript Nodes

Pinned JavaScript instances now separate worker-process compatibility from submitted code identity.

- `runtime_key` now identifies the reusable JS worker process: Python executable, environment key, and QuickJS runtime hash.
- `code_key` identifies the currently submitted code/package revision.
- Edited module/snippet code no longer makes a compatible pinned JS instance fail with `workflow_js_instance_incompatible_request`.
- `workflow_js_instance_create(..., replace=True, request=<edited code>)` reuses the existing worker process when the instance is idle and the worker-process `runtime_key` is unchanged.
- `workflow_js_instance_execute(instance_id=..., request=<edited code>)` also accepts edited code on the same compatible instance.
- JS project-mode pinned instances remain unsupported.

Client flow for JS edit+continue:

1. Keep using the same `instance_id`.
2. Submit edited module/snippet code with a fresh `module_sha256` or `code_revision`.
3. Optionally call `workflow_js_instance_create(..., replace=True, ...)` to update the instance code identity without restarting the worker process.
4. Call `workflow_js_instance_execute(...)`.
5. Expect the same host worker PID when only code changed and the worker compatibility key stayed unchanged.

### Python Nodes

Python edited code still requires a new Python worker process when clients replace the instance. The stable identity is the `instance_id`, not the old process heap.

Client flow for Python edit+continue:

1. Keep using the same `instance_id`.
2. Call `workflow_python_instance_create(..., instance_id=..., replace=True, request=<edited code>)`.
3. Continue with `workflow_python_instance_execute(instance_id=..., request=<edited code>)`.
4. Rehydrate from host-managed state and artifact refs. Do not expect Python globals, imported modules, open handles, threads, or C-extension state to survive replacement.

Instance-scoped host state remains keyed by `instance_id`, so replacing the Python process does not require moving supported state partitions.

## Artifact Recovery Handoff

Failed workflow requests that prepared declared artifact outputs now persist a small recovery manifest in the run folder. The manifest records safe correlation fields such as `request_id`, `instance_id`, `workflow_id`, `package_id`, and `node_id`.

### Recovery Notice

Failure responses can include:

```json
{
  "artifact_recovery": {
    "contract": "hosting.sandbox.artifact_recovery.v1",
    "request_id": "req-1",
    "instance_id": "node-inst-1",
    "cleanup_deferred": true,
    "candidates": []
  }
}
```

For streaming requests, the same shaped response is included in the terminal `error` event payload.

### Recovery Helpers

Use the typed control-channel helpers:

- `workflow_artifact_recovery_inspect(request_id=..., names=[...])`
- `workflow_artifact_recovery_claim(request_id=..., names=[...], instance_id=..., target_id=..., patch_absolute_paths=False)`
- `workflow_artifact_recovery_cleanup(request_id=...)`

When `target_id` is omitted, claim defaults to `@artifacts/instances/<instance_id>/...` if an instance id is supplied or present in the recovery manifest. Otherwise it falls back to `@artifacts/recovered/<request_id>/<timestamp>/...`.

Recommended client behavior is to keep using stable artifact refs returned under the instance namespace. The `old_path_to_new_path` and `old_path_to_new_ref` mappings in the raw claim result are low-level diagnostics and migration aids for clients that must patch their own side metadata. They are not the primary edit+continue model.

The raw daemon/CLI commands are:

- `workflow-artifact-recovery-inspect`
- `workflow-artifact-recovery-claim`
- `workflow-artifact-recovery-cleanup`

### Client Responsibilities

The client owns artifact validity decisions. Hosting labels candidates with hints such as:

- `declared_output`
- `crash_recovery_candidate`
- `partial_possible`

Do not treat those labels as proof that content is complete or semantically valid.

After a client decides it no longer needs the failed request folder, call `workflow_artifact_recovery_cleanup(...)`. Automatic garbage collection for deferred crash folders remains a later task.
