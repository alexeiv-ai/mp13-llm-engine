# Hosting Client Breaking Changes

Date: 2026-06-22

## Artifact Recovery Handoff Integration Instructions

This slice adds helper-level integration work for failed workflow requests that produced declared output artifacts. It does not recover Python/JavaScript heap state, open handles, imported module state, or replay mutations.

### Failure Responses

When a Python or JavaScript workflow node fails after artifact outputs were prepared, the service no longer deletes the request run folder immediately. The response can include:

```json
{
  "artifact_recovery": {
    "contract": "hosting.sandbox.artifact_recovery.v1",
    "request_id": "req-1",
    "cleanup_deferred": true,
    "candidates": []
  }
}
```

For streaming requests, the same shaped response is included in the terminal `error` event payload.

### New Client Helpers

Use the typed control-channel helpers rather than raw command payloads:

- `workflow_artifact_recovery_inspect(request_id=..., names=[...])`
- `workflow_artifact_recovery_claim(request_id=..., names=[...], target_id=..., patch_absolute_paths=False)`
- `workflow_artifact_recovery_cleanup(request_id=...)`

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

### Claim Semantics

Claiming copies selected recovered files into `@artifacts/<target_id>/...` and returns host-owned artifact refs plus old-path mappings. `patch_absolute_paths=True` is opt-in and only performs best-effort text replacement in small text-like files.

After a client decides it no longer needs the failed request folder, call `workflow_artifact_recovery_cleanup(...)`. Automatic garbage collection for deferred crash folders is intentionally left for a later task.
