# Hosted Workflow Runtime Status

Date: 2026-06-14

Purpose: record the current implementation state and the discrepancies against `src/hosting/hosting_access_plan.md`.

## Summary

- Helper-profile workflow Python facade: implemented.
- Workflow JS helper facade: implemented.
- Environment-keyed host routing/accounting: implemented for current workflow facades.
- First-class workflow Python node execution path: implemented.
- Full node sandbox hardening: still in progress.
- Node artifact store: local host-provisioned refs implemented for declared input refs and output slots, including inline artifacts, inline zip inputs, alias refs, file masks, recursive path matching, host takeover, and inline zip export.
- Python helper worker cleanup: reviewed; it remains intentionally required for helper-profile execution.

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
- Node compatible-work routing through environment-keyed pools, with runtime capacity controls for reserved slots.
- Node artifact file-mask and recursive input/output collection support.
- Shared active child runtime registry for active child resources/cancel tracking.
- Shared host artifact manager for artifact prepare, collect, zip, ownership, and request-local cleanup.
- Snippet execution with `execution_mode=snippet`, using `module_source` directly and no required export.
- Multi-module project execution with `execution_mode=project`, staged project refs, entrypoint module/callable selection, working directory, environment variables, and project-local import allowance.
- uv availability/version reporting and uv intent in Python environment specs.
- Deterministic non-executing uv install plans from `pyproject_toml`, `uv_lock`, and dependency groups.
- uv install plan locking/verification, explicit uv execution through install APIs, uv install receipts, uv receipt verification, and uv-managed interpreter selection for dependency-bearing node execution.
- Toolbox executor runtime execution, cancellation, request-status, and resource accounting through the shared hosted pool lifecycle layer, while toolbox registration/repair/GC orchestration remains toolbox-specific.
- Python node host API back channel for discoverable, dispatcher-based cooperative host calls over the built-in node harness control channel, currently scoped to artifact-root filesystem operations.
- Node host API discovery now exposes method descriptions, argument schemas, result schemas, permissions, roots, policy, and transport capabilities through `host.describe`.
- Python node workers now support warm sequential reuse for compatible module/snippet requests through a long-lived harness control loop. Project requests remain one-shot until project state recycling is implemented.
- Module/snippet warm worker routing now includes code revision identity using explicit `code_revision` or `module_sha256`; edited source reroutes to a new worker and old idle revisions are trimmed to configured capacity.
- Python node request lifecycle states are exposed as `submitted`, `running`, `ok`, `error`, `timeout`, and `canceled`; long-running node requests can opt into host-side `heartbeat` stream events with `limits.heartbeat_interval_ms`.
- Python node streams use bounded per-request retention through `limits.stream_max_events`; stream receives report retained and dropped event counts so callers can detect backpressure loss.
- Pending-cancel handling in the shared active child runtime registry so host cancellation is not lost while a node harness child is still starting.

## Discrepancies

- Dependency-bearing node execution now rejects missing preparation and missing install receipts.
- Verified dependency runtime success now selects the verified runtime interpreter before node execution.
- Artifact I/O is host-provisioned local sandbox file access: alias-ref and inline inputs are copied into request input paths, output slots become exact writable paths, inline outputs require matching declarations, and host-minted alias refs such as `@artifacts/...` are returned only for declared output files.
- File-mask artifact I/O is supported for local alias roots. Input masks copy matching files into a request-scoped input directory, and output masks collect matching files from a request-scoped output directory while preserving relative paths in returned refs.
- Inline zip inputs are expanded into request-scoped input directories. Multi-file outputs can be exported as inline zip without taking over artifact ownership. Explicit ref outputs remain producer-owned unless `host_takeover` is requested; takeover copies outputs into `@artifacts/...`.
- Artifact authorization, lifetime, cleanup, and external read APIs remain basic/local rather than a full durable artifact service.
- Previous tracking docs overstated node-profile execution and cleanup completion.

## Open Work

- Add deeper verified-runtime integration coverage if real dependency installs become available in CI.
- Add deeper artifact authorization, expiry, cleanup, and external read/API coverage when dependent clients consume refs.
- Generalize the Python node runtime for long-running job lifecycle/heartbeat behavior and uv-managed environments.
- Extend warm long-lived Python node harness workers beyond sequential compatible module/snippet reuse, including project-mode recycling.
- Add worker recycling for warm node workers, including explicit unhealthy-worker, policy-change, and project invalidation behavior.
- Decide whether helper-compatible runtimes should adopt the shared child-runtime/artifact helpers without changing helper response compatibility.
- Treat any future Python helper worker reduction as a separate helper-profile replacement project.
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
- Changed node responses to drop sandbox-returned `artifacts` unless the host collected declared output files and minted refs.
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
- Implemented local host-provisioned node artifacts: declared input refs resolve to request input paths, declared output slots expose exact writable paths, and successful outputs are copied into host-controlled local artifact storage.
- Added sync artifact tests for output collection, input-ref consumption, and rejection of undeclared file writes.
- Added stream artifact-event coverage for host-minted refs.
- Reviewed Python helper worker cleanup after node decoupling; no code removal is part of this node plan because `workflow_python(profile=helper)` still intentionally depends on that worker.
- Expanded node artifacts to support inline inputs, declared inline outputs, and relative alias refs such as `@artifacts/...` and policy-configured `@project/...` roots.
- Updated the interactive CLI stream event renderer to summarize artifact events.
- Added `sandbox/PY_NODE_WORKER.md` documenting the Python node execution API, artifact contract, and comparison with helper/toolbox workers.
- Investigated worker architecture: Python node uses shared runtime/pool primitives but is not a registered IPC worker; toolbox and helper remain separate worker entrypoints with distinct orchestration and compatibility roles.
- Added the next-phase plan for base-class completeness, long-running/concurrent node jobs, arbitrary snippets, multi-module project execution, and uv-managed runtime environments.
- Added `hosting.sandbox.child_runtime.HostedChildRuntime` and made the Python node runtime registry implement the shared `execute` / `cancel` / `resources` interface.
- Added node pool routing coverage proving compatible but different jobs share capacity and incompatible identities remain isolated.
- Added same-code concurrency coverage proving multiple instances of one `module_sha256` can run concurrently up to configured capacity.
- Added artifact input/output `path_mask` / `mask` and `recursive` support for alias refs, including tests for recursive masked input consumption and recursive masked output collection.
- Verified broader hosting workflow tests: `161 passed`.
- Added `hosting.sandbox.child_runtime.HostedActiveChildRuntimeRegistry` so direct host-managed runtimes share active child tracking, cancellation lookup, and active process resource listing.
- Added `hosting.sandbox.artifacts.HostedArtifactManager` and routed Python node artifact prepare/collect/cleanup through it.
- Added inline zip input expansion for multi-file artifacts, inline zip export for multi-file outputs, explicit `host_takeover` for selected ref outputs, and request-local artifact cleanup after collection.
- Added tests for active child runtime tracking, inline zip input, inline zip output export, and host takeover.
- Verified broader hosting workflow tests: `165 passed`.
- Added `execution_mode=snippet` for arbitrary source snippets that do not require `operation` or `export_name`.
- Added `execution_mode=project` for staged multi-module projects using `project.ref`, `project.entrypoint`, `project.callable`, optional `working_directory`, and optional `env`.
- Added project-local import handling so staged modules can import each other without allowing unlisted global imports.
- Added snippet, multi-module project, and project import-escape tests.
- Verified broader hosting workflow tests: `170 passed`.
- Fixed leaked workflow Python node child processes by reaping/killing children after terminal results and registering an interpreter-exit cleanup hook. Cleaned up orphaned node child processes found during investigation.
- Added uv availability/version detection, uv intent identity, and deterministic uv install-plan metadata without executing uv.
- Added uv environment-spec and prepare-plan tests.
- Added uv install lock/verify/execute/receipt lifecycle metadata and kept uv execution blocked unless explicitly requested through install APIs.
- Updated node dependency-environment checks so uv intent uses uv plan/execution/receipt verification fields and can select a verified uv runtime interpreter.
- Added uv lifecycle tests for missing uv, deterministic prepare plans, lock verification, blocked execution, verified receipt/runtime selection, service-side uv dependency routing, and uv-shaped runtime GC.
- Verified focused runtime/service tests: `83 passed`.
- Verified broader hosting workflow tests: `180 passed`.
- Migrated toolbox executor runtime calls onto the shared hosted pool lifecycle for execute, cancel, request status, and resource reporting.
- Added toolbox hosted-pool tests for execute lifecycle recording and cancellation lifecycle recording.
- Verified full toolbox sandbox tests: `124 passed`.
- Verified broader hosting workflow tests after toolbox lifecycle migration: `180 passed`.
- Added `hosting.workflow_python.node.host_api.v1` with `host.describe`, `host.call`, and artifact-scoped `fs.*` helpers available to node code through a bidirectional host-call protocol.
- Replaced the initial stdout/stdin host-call bridge with `hosting.workflow_python_node_worker_ipc`, a built-in Python node harness that uses a dedicated multiprocessing control channel for request, event, result, and host RPC messages.
- Routed node harness startup through the built-in worker module while keeping the same dedicated control-channel protocol.
- Fixed the child-runtime cancellation race where a host cancel issued while the node harness was still starting could be lost before active runtime registration.
- Removed the legacy embedded `python -c` node runner from the node runtime after harness parity verification.
- Documented the node host API, back-channel transport, and code-edit strategy for future long-lived workers in `sandbox/PY_NODE_WORKER.md`.
- Investigated node code editing: current one-child-per-request execution naturally handles fixed snippets as new `module_sha256` revisions; future long-lived workers should use explicit code revisions with restart/reroute as the conservative default, not uv as the code-edit mechanism.
- Added node host API tests for discovery, artifact-root reads/writes, and rejected input-root writes.
- Added a reusable native scoped host API registry for node built-ins and future host-registered functions, with sync/async handler support and sandbox-visible schemas.
- Added warm node harness reuse across compatible sequential requests and resource reporting for idle warm workers.
- Added capacity-shrink cleanup for idle warm node workers through the node capacity API.
- Added explicit node lifecycle states and opt-in heartbeat stream events for long-running node requests.
- Added module/snippet code-revision routing for warm workers and post-run idle trimming to capacity.
- Added bounded stream retention metadata and dropped-event accounting for node streams.
- Verified focused node harness lifecycle tests after control-channel startup/cancel changes: `3 passed`, repeated twice.
- Verified broader hosting workflow tests after node host API, warm-worker lifecycle, heartbeat, code-revision routing, and stream retention changes: `188 passed`.
- Verified toolbox host-call smoke tests after node host API changes: `2 passed`.

## Current Client Impact

- Existing helper-profile clients that already migrated to `workflow-python-*` and `workflow-js-*` do not need additional changes for the current implementation.
- Clients that own dependency-bearing node-profile workflow execution must prepare and verify runtime environments before execution.
- Node-profile clients should pass input artifacts as alias refs, inline payloads, or inline zip payloads, optionally use `path_mask` / `mask` and `recursive` for multi-file refs, write file outputs only to provided `artifact_outputs` paths or output directories, declare inline outputs before returning inline artifact payloads, use `export_inline_zip` for many output files when ownership should stay with the producer, request `host_takeover` only when the host should own returned ref lifetime, consume host-minted output refs, and still handle unavailable/missing artifact cases when no refs are produced.
- Node-profile clients may use `execution_mode=snippet` for source snippets or `execution_mode=project` with `project.ref` / `project.entrypoint` / `project.callable` for staged projects.
- Node-profile callers can use capacity APIs at runtime to trim or expand reserved workers for an environment-keyed pool; compatible jobs route through that pool while incompatible environment/import/dependency/sandbox identities route to separate pools.
