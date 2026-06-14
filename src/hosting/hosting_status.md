# Hosted Workflow Runtime Status

Date: 2026-06-14

Purpose: record the current implementation state and the discrepancies against `src/hosting/hosting_access_plan.md`.

## Summary

- Helper-profile workflow Python facade: implemented.
- Workflow JS helper facade: implemented.
- Environment-keyed host routing/accounting: implemented for current workflow facades.
- First-class workflow Python node sandbox: not implemented.
- Node-profile compatibility facade: implemented, but helper-backed.
- Node artifact store: not implemented.
- Python helper worker cleanup: not complete because helper-profile execution and the current node facade still depend on it.

## Implemented

- `workflow_python(profile=helper)` public facade.
- Helper-profile environment spec, prepare, lock, verify, install, receipt, ensure, execute, resources, capacity, cancel, and request-status surfaces.
- Helper-profile request metrics and environment-keyed pool accounting.
- Helper-profile import allowlist behavior in the existing helper worker.
- `workflow_python(profile=node)` request/response compatibility facade.
- Node-profile stream command surfaces:
  - `workflow-python-stream-open`
  - `workflow-python-stream-recv`
  - `workflow-python-stream-send`
  - `workflow-python-stream-close`
- Host-side stream event wrapping for the current helper-backed node facade.
- `workflow_js(profile=helper)` public facade and `workflow-js-execute`.
- RBAC/daemon/channel/CLI support for the workflow command families.
- Toolbox shared identity/process-base migration while preserving toolbox semantics.

## Discrepancies

- `workflow_python(profile=node)` still executes through `execute_workflow_python_helper`.
- The current node facade is constrained by helper source-in / JSON-out execution.
- Node import restrictions are not independently implemented; they are inherited from helper execution.
- Node dependency/runtime verification is not enforced as a hard execution precondition.
- Node streaming is host-side wrapping around a synchronous helper call, not node-owned streaming execution.
- Progress is only lifted from a final return value when present.
- stdout/stderr/log capture is not node-native.
- Artifact storage is unavailable.
- Current node tests cover the helper-backed facade, not a first-class node sandbox.
- Previous tracking docs overstated node-profile execution and cleanup completion.

## Open Work

- Implement direct node-profile execution without helper request translation.
- Implement node-owned import allowlist/default-deny enforcement and tests.
- Enforce verified runtime environment selection for dependency-bearing node work.
- Implement native node streaming events for stdout, stderr, logs, progress, artifacts, result, error, cancellation, and done.
- Implement artifact storage or make a deliberate no-artifacts product decision.
- Implement node-native cancellation, request status, resources, and metrics.
- Add first-class node sandbox tests.
- Revisit Python helper worker cleanup after node execution no longer depends on it.
- Update public docs after the first-class node behavior is implemented and verified.

## Current Client Impact

- Existing helper-profile clients that already migrated to `workflow-python-*` and `workflow-js-*` do not need additional changes for the current implementation.
- Clients that will own node-profile workflow execution must not assume the current helper-backed behavior is final.
- Future node-profile clients should expect a richer streaming/runtime/artifact contract once the first-class node sandbox lands.
