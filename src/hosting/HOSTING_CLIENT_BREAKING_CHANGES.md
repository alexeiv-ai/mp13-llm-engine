# Hosting Client Remaining Changes

Date: 2026-06-14

Purpose: track only remaining dependent-project changes. Previously completed helper-profile migration items are intentionally omitted.

## Current State

- Helper-profile dependent clients have already migrated to workflow facade APIs.
- No additional client action is required for existing short helper-profile Python or JS execution.
- `workflow_python(profile=node)` now has a direct node execution path and no longer returns helper-shaped nested results.
- Dependency-bearing node-profile execution now requires host-prepared and verified runtime environments before execution.

## Remaining Client Changes

- Clients that own node-profile workflow execution must validate against the direct node response envelope and stream event model before adopting it.
- Node-profile clients must consume streaming events as separate records:
  - `started`
  - `stdout`
  - `stderr`
  - `log`
  - `progress`
  - `artifact`
  - `result`
  - `error`
  - `canceled`
  - `done`
- Node-profile clients must handle structured terminal errors for environment problems, import-policy failures, timeout, cancellation, runtime errors, and output/artifact limits.
- Node-profile clients must pass stable `request_id` values for cancellation and request-status lookup.
- Node-profile clients must use host-derived `environment_key` for resources, capacity, cancellation, and request status. Compatible node jobs route through the same environment-keyed pool; incompatible runtime/import/dependency/sandbox identities route to separate pools.
- Node-profile host callers may use capacity APIs during runtime to trim or expand reserved workers for a pool.
- Node-profile clients must not rely on helper-shaped nested result payloads. They should consume the node response envelope directly.
- Node-profile clients must stop assuming `artifact_store.status=unavailable`. They must pass input artifacts as relative alias refs such as `@artifacts/...`, declared inline payloads, or inline zip payloads; configure any non-default artifact roots such as `@project` through sandbox policy; write file outputs only to host-provided artifact output paths or output directories; declare inline outputs before returning inline artifact payloads; consume host-minted alias refs; and handle missing-artifact or unavailable-artifact responses when no refs are produced.
- Node-profile clients may select multiple artifact files with `path_mask` or `mask` and `recursive` on input or output artifact declarations. Masked inputs are exposed to Python code as directories containing matched files. Masked outputs are exposed as writable directories and return one host-minted ref per collected file, with `relative_path` populated.
- Node-profile clients may use `export_inline_zip` to export many output files as one inline zip without changing ownership. They may use `host_takeover` when the host should copy a ref output into `@artifacts/...` and own its lifetime; otherwise explicit output refs remain producer-managed.
- Clients that provide dependency-management UI or orchestration must call host-controlled prepare/lock/verify/install/receipt APIs explicitly before dependency-bearing execution. Normal workflow execution does not install dependencies implicitly.

## No Remaining Action For Already Migrated Helper Clients

- No action for clients already using `workflow-python-*` helper-profile APIs.
- No action for clients already using `workflow-js-*` helper-profile APIs.
- No action for clients already routing helper resources, capacity, request status, and cancellation by `environment_key`.
