# Hosted Workflow Runtime Goals And Discrepancies

Date: 2026-06-14

Purpose: keep the implementation pointed at the intended hosted workflow runtime behavior. This file intentionally avoids prescribing internal module names, worker contract strings, or transport details unless they are already part of the public API.

## Goals

- `workflow_python` is the public Python workflow runtime with two profiles:
  - `profile=helper`: short helper execution, source-in / JSON-out, compatible with existing helper behavior.
  - `profile=node`: first-class workflow node execution with a richer node response and streaming model.
- `workflow_js(profile=helper)` remains the public JavaScript helper runtime facade.
- Runtime routing is by host-derived `environment_key`, not by raw engine ID alone.
- Environment identity includes runtime intent, dependency/import intent, and sandbox policy identity so incompatible work does not share a worker pool.
- Dependency installation is not triggered implicitly by workflow execution. Dependency-bearing work must execute against an explicitly prepared/verified runtime environment or fail with a structured environment error.
- Shared hosting concerns remain common across runtimes: process lifecycle, sandbox policy attachment, pool capacity, cancellation, request status, metrics, and resource reporting.
- Node-profile Python must not be limited by the narrower helper input/output contract. Helper compatibility may reuse implementation where appropriate, but it must not define node semantics.

## Current State

- `workflow_python(profile=helper)` exists as a public facade and is backed by the existing Python helper worker.
- Helper-profile execution has source hash verification, operation allowlisting, JSON payload/result handling, import allowlist enforcement, timeout/output limits, capacity, cancellation, and request metrics.
- `workflow_python(profile=node)` now has a direct node execution path:
  - It validates node-profile request fields.
  - It executes node requests through node-owned Python child runtime code.
  - It no longer translates node execution through `execute_workflow_python_helper`.
  - It preserves the node response envelope.
  - Its stream API routes execution through the node runtime and emits node events through the shared stream/session plumbing.
- `workflow_js(profile=helper)` exists as a public facade over the JS helper implementation.
- Host-side environment-key routing and pool/request accounting exist for the workflow facades.
- Node-profile artifact storage has a local host-provisioned implementation for declared input refs and output slots, including inline artifacts, alias refs, file masks, and recursive path collection. It returns `artifact_store.status=ok` only when refs are minted from declared output files or declared inline outputs.
- Node-profile routing now uses host-derived `environment_key` plus shared pool capacity. Compatible node requests share the same pool, incompatible environment/import/dependency/sandbox identities route to separate pools, and runtime capacity can be adjusted through host capacity APIs. The current node harness is still started per request; long-lived warm worker reuse is a next-phase implementation item.

## Current Discrepancies

- Node-profile artifact refs are implemented as local host-controlled alias refs such as `@artifacts/...`, but authorization, lifetime, cleanup, and external read APIs remain basic/local rather than a full durable artifact service.
- Node-profile cancellation, output-limit, truncation, environment-policy, and artifact behavior now have focused coverage; broader integration coverage can still be added when real dependency installs and artifact consumers exist.
- Cleanup is incomplete: the Python helper worker remains the actual execution substrate for helper-profile execution.

## Work Items

### Baseline Already Present

- [x] Keep `workflow_python(profile=helper)` available as the short helper execution profile.
- [x] Keep `workflow_js(profile=helper)` available as the JavaScript helper execution profile.
- [x] Keep public workflow Python facade commands available for helper-profile execution.
- [x] Keep public workflow JS facade commands available for helper-profile execution.
- [x] Derive and report `environment_key` for workflow facade calls.
- [x] Record host-side request lifecycle metrics for current helper-backed workflow execution.
- [x] Preserve node-profile public request normalization and response envelope scaffolding.
- [x] Preserve node-profile stream command surfaces while replacing the backing implementation.

### Node Execution Path

- [x] Add a direct `workflow_python(profile=node)` execution path that does not call `execute_workflow_python_helper`.
- [x] Route node-profile sync execution through the new node execution path.
- [x] Route node-profile stream execution through the new node execution path.
- [x] Keep helper-profile execution behavior unchanged while node execution is replaced.
- [x] Keep the public node request fields stable:
  - [x] `request_id`
  - [x] `module_source`
  - [x] `module_sha256`
  - [x] `package_id`
  - [x] `workflow_id`
  - [x] `package_source_digest`
  - [x] `export_name` or `operation`
  - [x] `payload`
  - [x] `provenance`
  - [x] `limits`
  - [x] `policy`
  - [x] `python`
- [x] Keep the public node response fields stable:
  - [x] `status`
  - [x] `ok`
  - [x] `output`
  - [x] `state_patch`
  - [x] `artifacts`
  - [x] `artifact_store`
  - [x] `progress`
  - [x] `logs`
  - [x] `metrics`
  - [x] `error`
  - [x] `audit`
- [x] Return a structured error when required node request fields are missing.
- [x] Return a structured error when `module_sha256` does not match `module_source`.
- [x] Return a structured error when the requested export/operation cannot be executed.

### Import Policy

- [x] Move node-profile import enforcement into node-owned execution code.
- [x] Default-deny imports when `python.import_allowlist` is empty or absent.
- [x] Allow only explicitly listed root modules when `python.import_allowlist` is present.
- [x] Reject imports whose root module is not allowlisted.
- [x] Preserve safe builtin behavior appropriate for node execution.
- [x] Ensure import policy failures produce structured node errors.
- [x] Add tests for default-deny imports.
- [x] Add tests for allowlisted imports.
- [x] Add tests for unallowlisted imports.

### Runtime Environment Policy

- [x] Derive node `environment_key` from environment name, runtime intent, import intent, dependency intent, and sandbox policy identity.
- [x] Reject caller-supplied `environment_key` values that do not match host-derived identity.
- [x] Ensure different dependency/import/runtime/sandbox-policy identities do not share node workers.
- [x] Ensure compatible node requests route through the same environment-keyed worker pool.
- [x] Ensure host runtime capacity controls trim or expand the reserved slots for a node pool.
- [x] Ensure dependency-bearing node requests execute only against a selected verified runtime environment.
- [x] Return a structured environment error when required dependency environment preparation is missing.
- [x] Return a structured environment error when install receipt verification failed or is absent.
- [x] Keep normal node execution from installing dependencies implicitly.
- [x] Add tests for environment-key mismatch.
- [x] Add tests for dependency-bearing execution without verified environment.
- [x] Add tests proving incompatible identities do not share live workers or hot child pools.

### Streaming And Events

- [x] Emit `started` when node execution begins.
- [x] Capture and emit bounded `stdout` events from executed Python code.
- [x] Capture and emit bounded `stderr` events from executed Python code.
- [x] Emit `log` events for host/runtime diagnostics that are safe to expose.
- [x] Emit `progress` during execution, not only after final return.
- [x] Emit `artifact` events when artifact refs are created.
- [x] Emit `result` for successful terminal output.
- [x] Emit `error` for structured terminal failures.
- [x] Emit `canceled` when cancellation wins.
- [x] Emit `done` exactly once for each stream.
- [x] Keep stream queues bounded and enforce `max_items` on receive.
- [x] Add tests proving progress can be observed before final result.
- [x] Add tests for stdout/stderr/log truncation.
- [x] Add tests for terminal event ordering.

### Result Semantics

- [x] Preserve node `output` as the primary successful result value.
- [x] Preserve `state_patch` as JSON object or `null`.
- [x] Preserve `progress` as latest progress snapshot where available.
- [x] Preserve `logs` as bounded summaries that do not expose raw source by default.
- [x] Preserve `audit` fields for package, workflow, source digest, module hash, provenance, runtime, and request identifiers.
- [x] Return structured runtime errors with safe traceback/message summaries.
- [x] Return structured timeout errors.
- [x] Return structured output-limit errors.
- [x] Add tests for successful output/state patch.
- [x] Add tests for structured runtime errors.
- [x] Add tests for timeout.
- [x] Add tests for output limit.

### Artifacts

Decision: artifact I/O belongs in the first-class node sandbox contract, and this pass implements the local host-provisioned version. The node response keeps `artifacts` and `artifact_store` stable; `artifact_store.status=ok` is returned only when the host mints refs from declared output files or declared inline outputs, while requests with no declared output artifacts still report the store as unavailable for that response.

How artifacts fit the sandbox model: input artifacts are either alias refs such as `@artifacts/...` or `@project/...`, inline bytes/text, or inline zip bundles that the host expands into sandbox-visible input paths before execution. Output artifacts are either files written by sandboxed code only to host-provided output paths/directories, inline outputs returned by sandboxed code only when a matching inline output declaration exists, or multi-file outputs exported as one inline zip. Ref outputs remain producer-owned when an explicit output ref is used. The host takes over a ref output only when the output declaration asks for takeover or omits `ref`, in which case the host copies files into `@artifacts/...`. Request-local worker output paths are cleaned after collection. The sandbox should never let code mint artifact identity by returning a path, URL, or opaque token directly.

Rationale: this keeps artifact management aligned with sandbox file access. The sandbox may consume files and produce files, but the host owns the capability boundary: which configured alias refs are readable, which exact output files are writable, what inline bytes cross back out, and which refs clients may later read. Direct filesystem paths are not promoted as artifact refs.

Untrusted artifact refs means any artifact-looking value produced by sandboxed code rather than by the host artifact manager. Examples include returned dicts such as `{"path": "/tmp/report.csv"}`, `{"url": "file:///..."}`, `{"artifact_id": "abc"}`, or `{"ref": "../other-run/output"}`. These values may be useful as ordinary JSON output if the workflow wants them, but the host must not treat them as authorized downloadable artifacts, emit them as `artifact` stream events, or store them in the response `artifacts` list until the host has verified the file came from an allowed output path and has created/registered the reference.

- [x] Decide whether first-class node supports artifacts in this implementation pass.
- [x] Choose host-provisioned artifact I/O as the intended sandbox model.
- [x] Keep a deliberate structured unavailable response when no host-minted artifact refs exist for a response.
- [x] Define request fields for input artifact refs and host-provided output artifact slots/directories.
- [x] Support alias-ref artifact inputs with relative refs such as `@artifacts/...` or policy-configured roots such as `@project/...`.
- [x] Support input artifact file masks with `path_mask` or `mask`.
- [x] Support recursive input artifact matching with `recursive=true`.
- [x] Support inline artifact inputs by writing declared bytes/text to sandbox input paths.
- [x] Support inline zip artifact inputs by expanding zip members into sandbox input directories.
- [x] Support alias-ref artifact outputs by returning host-minted or host-validated relative alias refs.
- [x] Support output artifact file masks with `path_mask` or `mask`.
- [x] Support recursive output artifact collection with `recursive=true`.
- [x] Support host takeover of selected output ref artifacts with `host_takeover`.
- [x] Support multi-file output export as inline zip without changing producer ownership.
- [x] Support declared inline artifact outputs without trusting undeclared sandbox artifact returns.
- [x] Configure artifact root alias to physical path mappings through sandbox policy.
- [x] Treat input-side size/count/lifetime/encoding metadata as optional advisory metadata.
- [x] Resolve input artifact refs into sandbox-visible input paths before execution.
- [x] Provide output artifact paths scoped to the current request.
- [x] Collect only files written under host-provided output locations.
- [x] Register collected output files into host-controlled artifact storage and return host-minted refs.
- [x] Clean request-local worker artifact directories after collection.
- [x] Ensure direct node execution ignores or rejects untrusted returned artifact refs instead of treating them as host-created artifacts.
- [x] Ensure stream execution emits `artifact` events only for host-minted refs.
- [x] Keep artifact-looking values from sandbox code as ordinary `output` only, unless the host artifact manager creates the reference.
- [x] Add tests for no-host-minted-artifact behavior on successful node execution.
- [x] Add tests proving returned artifact-like data from user code is not promoted to host artifact refs.
- [x] Add tests for input artifact ref resolution to sandbox paths.
- [x] Add tests for recursive masked input artifact refs.
- [x] Add tests for inline zip input artifact expansion.
- [x] Add tests for output artifact collection from allowed output paths.
- [x] Add tests for recursive masked output artifact collection.
- [x] Add tests for host takeover of ref outputs.
- [x] Add tests for multi-file inline zip export.
- [x] Add tests rejecting artifact collection from paths outside host-provided output locations.
- [x] Document the artifact-storage requirements before enabling artifacts:
  - [x] host-controlled storage root
  - [x] stable reference shape
  - [x] authorization model for reads and writes
  - [x] lifetime/expiry policy
  - [x] cleanup policy
  - [x] size/count limits
  - [x] input-ref-to-read-only-path mapping
  - [x] output-slot-to-writable-path mapping
  - [x] brokered write API from sandboxed execution if path-based output is insufficient
  - [x] stream `artifact` event semantics
  - [x] response artifact ref semantics

### Cancellation, Status, And Resources

- [x] Make `workflow-python-stream-send` cancellation interrupt active node execution.
- [x] Make host-level `workflow-python-cancel-request` cancellation interrupt active node execution.
- [x] Record terminal request state for canceled node executions.
- [x] Report active node request status by `environment_key + request_id`.
- [x] Report latest progress in node request status.
- [x] Report node resources by `environment_key`.
- [x] Report node capacity, active calls, available slots, active request IDs, and process count.
- [x] Report latency, timeout, cancellation, saturation, and error counters.
- [x] Report per-process CPU/RSS where the host can sample them.
- [x] Add tests for stream cancellation.
- [x] Add tests for host-level cancellation.
- [x] Add tests for request status during active execution.
- [x] Add tests for resource metrics after success, error, timeout, and cancel.

### Compatibility And Cleanup

Cleanup decision: the Python helper worker is no longer part of node-profile execution, but it is still the intentional backing implementation for `workflow_python(profile=helper)`. Reducing or removing it would now be a separate helper-profile replacement project, not required for first-class node sandboxing.

- [x] Keep existing helper-profile clients working while node implementation changes.
- [x] Remove the current helper-backed node facade as the temporary compatibility path during migration.
- [x] Remove helper-backed node execution once direct node execution is verified.
- [x] Revisit whether the Python helper worker can be reduced after node no longer depends on it.
- [x] Document why node does not fully subsume helper-profile compatibility yet.
- [x] Compare Python node, Python helper, and toolbox worker architecture for shared base/refactoring opportunities.
- [x] Update `HOSTING_CLIENT_BREAKING_CHANGES.md` only for remaining dependent-project actions.
- [x] Update public hosting docs after first-class node behavior is implemented and tested.
- [x] Remove or rewrite stale docs that imply helper-backed node execution is complete first-class node sandboxing.

### Verification

- [x] Add focused unit tests for node request normalization and validation.
- [x] Add focused unit tests for node import policy.
- [x] Add focused unit tests for node runtime environment policy.
- [x] Add service-level sync execution tests for node success and failure.
- [x] Add service-level streaming tests for node events.
- [x] Add service-level streaming tests for node cancellation.
- [x] Add CLI/channel payload forwarding tests for node sync and stream commands.
- [x] Add resource/request-status tests for node metrics.
- [x] Add regression tests proving helper-profile behavior remains unchanged.
- [x] Run the focused hosting workflow test suite.
- [x] Record the verified behavior in `hosting_status.md`.

## Non-Goals

- Do not require a specific internal worker file name or contract string in the planning docs.
- Do not merge generic/model worker semantics into workflow runtime semantics.
- Do not promise strong OS-level filesystem/network isolation beyond what the shared sandbox launcher and brokered I/O actually enforce.
- Do not implement implicit dependency installation during normal workflow execution.

## Next Phase: Python Node Runtime Generalization

These items are intentionally separate from the completed first-class node contract above. They address the next concern: Python node should become a general hosted Python runtime for concurrent, long-running, snippet, and project execution while sharing more host-side sandbox management code.

### Base Class Completeness

Current assessment: `HostedProcessSandboxBase` plus the shared child/artifact helpers now cover the lean host-side lifecycle layer: pool, request lifecycle, request status, stream queues, capacity, cancellation bookkeeping, pending-cancel handling during child startup, active child tracking, child cancel/resource listing, and artifact prepare/collect/cleanup. Toolbox executor runtime calls now also record execute/cancel/request-status/resource accounting through the shared hosted pool layer. Node execution now launches a built-in Python harness file with a dedicated control channel instead of using stdout as the host RPC transport. Runtime-specific code still owns child process launch details, control-channel protocol parsing, import policy, result normalization, warm process reuse, project staging, venv selection, and toolbox registration/repair/GC orchestration.

- [x] Define a small hosted child-runtime interface with `execute`, `cancel`, and `resources`.
- [x] Move active child cancel/resource tracking behind the shared child-runtime base.
- [x] Move reusable artifact preparation/collection/cleanup into a shared host-side component.
- [x] Make node runtime use the shared child-runtime interface.
- [x] Keep child process launch and control-channel protocol parsing runtime-specific for now.
- [ ] Decide whether Python helper can use the same child-runtime implementation while preserving helper response compatibility.
- [x] Route toolbox executor execute/cancel/request-status/resource accounting through the same normalized host pool/resource shapes while preserving toolbox registration/repair orchestration.
- [ ] Decide whether persisted toolbox registration/repair/GC state should gain shared lifecycle metadata or remain toolbox-specific.
- [x] Add base-layer tests for pool/request/status/cancel behavior and active child resource/cancel tracking.

### Node Host API Back Channel

Target assumption: node workers may need cooperative host interactions like toolbox brokered filesystem/http callbacks, especially once node workers become long-lived. The host API should be discoverable from Python code and should use a dispatcher-based request/response protocol that can be reused by one-shot and future long-lived workers.

- [x] Define a node host API contract exposed through `host.describe`.
- [x] Add a built-in Python node worker harness with a dedicated control channel for node execution.
- [x] Add bidirectional child-process protocol messages for `host_call` and `host_response` without using stdout as the host RPC transport.
- [x] Route active node execution through the built-in harness control channel instead of the embedded `python -c` stdout event bridge.
- [x] Launch the harness as an explicit worker file for fast cold-start while keeping stdout/stderr reserved for user logs.
- [x] Remove the legacy embedded `python -c` node runner after harness parity coverage remains stable.
- [x] Expose a Python `host` object with `call`, `describe`, and filesystem convenience methods.
- [x] Implement artifact-scoped `fs.list`, `fs.read_text`, `fs.write_text`, `fs.mkdir`, and `fs.stat` through the host dispatcher.
- [x] Enforce read-only input roots and writable output roots for node host API filesystem calls.
- [x] Include host API metadata in the machine-readable node contract.
- [ ] Add policy-gated HTTP host API support using the same dispatcher shape.
- [ ] Add a long-lived worker transport loop that reuses the same host API protocol across many requests and avoids per-request cold-start cost.
- [x] Add tests for host API discovery, artifact-root read/write, and rejected input-root writes.

### Long-Running And Concurrent Node Jobs

Target assumption: many different Python node jobs may run concurrently, and several instances of the same Python node code may run concurrently. Long-running node jobs are expected and should be managed explicitly by host pool capacity, request IDs, status, stream backpressure, cancellation, and resource reporting.

- [ ] Define node job lifecycle states for long-running execution beyond short helper calls.
- [x] Ensure concurrent requests for the same `environment_key` are admitted up to configured capacity.
- [x] Ensure multiple instances of the same `module_sha256` can run concurrently with distinct `request_id` values.
- [ ] Keep warm harness workers per environment-keyed pool so capacity represents reusable reserved workers, not only per-request slots.
- [ ] Define code revision lifecycle for long-lived workers so edited snippets/modules run as new revisions instead of mutating loaded code in place.
- [ ] Decide and implement restart/reroute versus explicit module unload/reload for long-lived worker code edits.
- [ ] Add per-request stream backpressure and bounded event retention policy suitable for long-running jobs.
- [ ] Add long-running progress heartbeat/status behavior.
- [x] Add tests for concurrent different node jobs.
- [x] Add tests for concurrent same-code node jobs.
- [ ] Add tests for long-running stream/status/cancel behavior under capacity pressure.

### Snippets And Python Projects

Target assumption: node execution should support both arbitrary Python snippets and Python projects made of multiple modules. `module_source` remains useful for single-file execution, but it is not enough for project execution.

- [x] Define request shape for snippet execution where source is arbitrary Python code and not necessarily a named workflow export.
- [x] Define request shape for project execution with a project root artifact/ref, entrypoint module, callable, argv, environment variables, and working directory.
- [x] Stage project files into a request/runtime workspace using artifact refs or configured alias roots.
- [x] Support multi-module imports from the staged project root without weakening global import policy.
- [x] Preserve source/package digest audit fields for staged projects.
- [x] Add tests for snippet execution.
- [x] Add tests for multi-module project execution.
- [x] Add tests that project import paths cannot escape the staged project root.

### uv-Managed Environments

Target assumption: Python node projects need deterministic, host-managed environments. Dependency installation remains explicit; normal execution must not install implicitly.

- [x] Add uv availability detection and version reporting.
- [x] Extend environment specs to represent uv-managed environments.
- [x] Support `pyproject.toml`, `uv.lock`, and dependency-group inputs.
- [x] Prepare deterministic uv install plans without executing them.
- [x] Lock/verify uv plans before execution.
- [x] Execute uv environment creation only through explicit prepare/install APIs.
- [x] Select the uv-managed Python interpreter for dependency-bearing node execution.
- [x] Record uv lock/install receipts and verify them before execution.
- [x] Add cleanup/GC for stale uv-managed runtime environments.
- [x] Add tests for missing uv, prepared uv plan, verified uv receipt, selected uv runtime, and uv-shaped runtime GC.
