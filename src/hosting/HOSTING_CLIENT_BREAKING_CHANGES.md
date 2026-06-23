# Hosting Client Breaking Changes

Date: 2026-06-22

## Recovery Pattern Clarification

The recommended edit+continue recovery model is instance-scoped artifact refs, not old-path remapping.

Clients should:

1. Keep using the same logical `instance_id`.
2. Claim recovered artifacts without `target_id` when an `instance_id` is available.
3. Continue with refs under `@artifacts/instances/<instance_id>/...`.
4. Treat `old_path_to_new_path` and `old_path_to_new_ref` in raw claim responses as low-level diagnostics or migration aids only.

Use old-path mappings only when client-owned metadata already persisted absolute worker-local paths and must be patched. New client flows should persist host artifact refs instead.
