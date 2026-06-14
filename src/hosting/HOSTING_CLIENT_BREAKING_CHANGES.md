# Hosting Client Remaining Changes

Date: 2026-06-14

Purpose: track only remaining dependent-project changes. Previously completed helper-profile migration items are intentionally omitted.

## Current State

- Helper-profile dependent clients have already migrated to workflow facade APIs.
- No additional client action is required for existing short helper-profile Python or JS execution.
- The current `workflow_python(profile=node)` behavior is a helper-backed compatibility facade, not the final first-class node sandbox.

## Remaining Client Changes

- Clients that own node-profile workflow execution must validate against the first-class node sandbox when it lands.
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
- Node-profile clients must use host-derived `environment_key` for resources, capacity, cancellation, and request status.
- Node-profile clients must not rely on helper-shaped nested result payloads. They should consume the node response envelope directly.
- Node-profile clients must stop assuming `artifact_store.status=unavailable` once artifact storage is implemented. They must handle real artifact refs, authorization failures, expiry, and missing-artifact errors.
- Clients that provide dependency-management UI or orchestration must call host-controlled prepare/lock/verify/install/receipt APIs explicitly before dependency-bearing execution. Normal workflow execution must not install dependencies implicitly.

## No Remaining Action For Already Migrated Helper Clients

- No action for clients already using `workflow-python-*` helper-profile APIs.
- No action for clients already using `workflow-js-*` helper-profile APIs.
- No action for clients already routing helper resources, capacity, request status, and cancellation by `environment_key`.
