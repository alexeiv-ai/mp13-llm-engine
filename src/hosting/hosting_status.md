# Hosted Workflow Runtime Status

Date: 2026-06-14

Purpose: record the current implementation state and the discrepancies against `src/hosting/hosting_access_plan.md`.

## Summary

- Helper-profile workflow Python facade: implemented.
- Workflow JS helper facade: implemented.
- Environment-keyed host routing/accounting: implemented for current workflow facades.
- First-class workflow Python node execution path: implemented.
- Full node sandbox hardening: still in progress.
- Node artifact store: deliberately unavailable for this pass.
- Python helper worker cleanup: not complete because helper-profile execution still depends on it.

## Implemented

- `workflow_python(profile=helper)` public facade.
- Helper-profile environment spec, prepare, lock, verify, install, receipt, ensure, execute, resources, capacity, cancel, and request-status surfaces.
- Helper-profile request metrics and environment-keyed pool accounting.
- Helper-profile import allowlist behavior in the existing helper worker.
- `workflow_python(profile=node)` request/response facade.
- Node-profile stream command surfaces:
  - `workflow-python-stream-open`
  - `workflow-python-stream-recv`
  - `workflow-python-stream-send`
  - `workflow-python-stream-close`
- Shared stream/session plumbing for node-profile stream events.
- `workflow_js(profile=helper)` public facade and `workflow-js-execute`.
- RBAC/daemon/channel/CLI support for the workflow command families.
- Toolbox shared identity/process-base migration while preserving toolbox semantics.
- Direct node-profile Python execution path that no longer calls `execute_workflow_python_helper`.
- Node-profile sync execution through the direct node runtime.
- Node-profile stream execution through the direct node runtime.
- Node-owned import allowlist/default-deny enforcement.
- Node runtime progress events during execution through `progress(...)` / `emit_progress(...)`.
- Node stdout/stderr capture and stream emission.
- Node resource, request-status, capacity, and metrics reporting through the workflow pool.

## Discrepancies

- Dependency-bearing node execution now rejects missing preparation and missing install receipts.
- Verified dependency runtime success now selects the verified runtime interpreter before node execution.
- Artifact I/O is intended to be host-provisioned sandbox file access, but artifact storage is still unavailable; artifact contract fields remain present.
- Node cancellation is wired through the node runtime registry, but still needs focused stream/host cancellation tests.
- Node output-limit behavior is implemented, but still needs focused node-profile tests.
- Node stderr/log truncation behavior is implemented, but still needs focused truncation tests.
- Previous tracking docs overstated node-profile execution and cleanup completion.

## Open Work

- Add deeper verified-runtime integration coverage if real dependency installs become available in CI.
- Add unavailable-artifact behavior tests and document host-provisioned input/output artifact requirements.
- Add focused node cancellation, output-limit, truncation, environment-policy, and artifact tests.
- Revisit Python helper worker cleanup after node execution no longer depends on it.
- Update public docs after the first-class node behavior is implemented and verified.

## Progress Updates

### 2026-06-14

- Added `hosting.sandbox.workflow_python_node_runtime`, a node-owned Python child runtime for node-profile execution.
- Routed `workflow_python(profile=node)` sync execution away from `execute_workflow_python_helper`.
- Routed `workflow-python-stream-open` node execution away from helper RPC and through the node runtime.
- Routed node resources, capacity, request status, and cancellation through the workflow pool/node runtime registry instead of helper worker RPCs.
- Added node import policy tests covering default-deny, wrong allowlist, and allowlisted imports.
- Added a regression test proving node execution does not call the helper proxy path.
- Added a stream test proving runtime progress can arrive before the final result and stdout is emitted as a stream event.
- Fixed stream close handling so a stream already marked closed by terminal execution does not emit a second `done` event.
- Verified focused hosting workflow tests: `87 passed`.
- Updated the checked plan items to reflect only the node behavior implemented or verified in this pass.
- Updated `HOSTING_CLIENT_BREAKING_CHANGES.md` so dependent-project actions no longer describe node execution as helper-backed.
- Clarified the artifact decision: artifact I/O belongs in the node sandbox contract, but must be host-provisioned through input refs, read-only input paths, output paths or brokered writes, host validation, and host-minted refs.
- Clarified the artifact sandbox boundary: sandboxed code cannot mint trusted artifact refs by returning paths, URLs, or tokens; only the host artifact manager may create artifact refs after validating files from allowed output locations.
- Changed node responses to drop sandbox-returned `artifacts` while the artifact manager is unavailable.
- Added artifact-safety tests proving unavailable artifact behavior and preventing sandbox-returned refs from becoming stream `artifact` events.
- Added focused node tests for output-limit errors and stdout/stderr log truncation.
- Implemented structured node cancellation results for active runtime cancellation.
- Added active host-level and stream-send cancellation tests, including request-status checks while execution is running.
- Added dependency-environment policy checks that reject node execution when preparation or install receipt verification is missing.
- Added node environment-key mismatch and incompatible-identity pool isolation tests.
- Routed verified dependency-bearing node execution through selected runtime Python and added selection coverage.
- Added node resource metrics coverage after success, error, timeout, and cancellation.
- Added focused node request normalization and validation contract tests.
- Added channel/daemon forwarding coverage for node sync execution and node stream commands.
- Documented the host-provisioned artifact boundary and requirements in sandbox architecture/workflow docs.
- Updated remaining client-change notes to reflect enforced dependency-environment prechecks.
- Added active node runtime process resource reporting with host CPU/RSS snapshots where available.

## Current Client Impact

- Existing helper-profile clients that already migrated to `workflow-python-*` and `workflow-js-*` do not need additional changes for the current implementation.
- Clients that will own dependency-bearing node-profile workflow execution must wait for verified-environment enforcement before relying on package/runtime isolation guarantees.
- Future node-profile clients should treat `artifact_store.status=unavailable` as deliberate current behavior and expect artifact behavior to change to host-provisioned input refs plus host-minted output refs when artifact storage is implemented.
