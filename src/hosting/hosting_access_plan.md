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
- `workflow_python(profile=node)` currently exists as a compatibility facade:
  - It validates node-profile request fields.
  - It translates the request into a helper-style call.
  - It executes through `execute_workflow_python_helper`.
  - It wraps the returned JSON into the node response envelope.
  - Its stream API is host-side wrapping around a synchronous helper execution.
- `workflow_js(profile=helper)` exists as a public facade over the JS helper implementation.
- Host-side environment-key routing and pool/request accounting exist for the workflow facades.
- Artifact storage for node-profile responses is not implemented; current responses use an explicit unavailable placeholder.

## Current Discrepancies

- Node-profile Python is not a first-class sandbox yet. It still depends on the helper execution path.
- Node-profile imports are not independently implemented. Node inherits helper import behavior.
- Node-profile dependency/runtime enforcement is incomplete. Package/import intent contributes to identity, but execution must still be tightened so dependency-bearing node work runs only in a verified runtime or fails explicitly.
- Node-profile streaming is not native. Progress, artifact, stdout, stderr, and logs are not emitted by a node-owned execution loop.
- Node-profile progress is lifted from the final returned JSON, not streamed during execution.
- Node-profile logs are host-created summaries, not captured Python execution logs.
- Node-profile artifact refs are contract fields only; there is no storage, authorization, lifetime, or reference implementation.
- Cleanup is incomplete: the Python helper worker remains the actual execution substrate for helper-profile execution and the current node facade.
- Previous plan/status text over-marked node-profile execution and cleanup as complete. The corrected status is that helper/facade work is complete, but first-class node sandbox work remains open.

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

- [ ] Add a direct `workflow_python(profile=node)` execution path that does not call `execute_workflow_python_helper`.
- [ ] Route node-profile sync execution through the new node execution path.
- [ ] Route node-profile stream execution through the new node execution path.
- [ ] Keep helper-profile execution behavior unchanged while node execution is replaced.
- [ ] Keep the public node request fields stable:
  - [ ] `request_id`
  - [ ] `module_source`
  - [ ] `module_sha256`
  - [ ] `package_id`
  - [ ] `workflow_id`
  - [ ] `package_source_digest`
  - [ ] `export_name` or `operation`
  - [ ] `payload`
  - [ ] `provenance`
  - [ ] `limits`
  - [ ] `policy`
  - [ ] `python`
- [ ] Keep the public node response fields stable:
  - [ ] `status`
  - [ ] `ok`
  - [ ] `output`
  - [ ] `state_patch`
  - [ ] `artifacts`
  - [ ] `artifact_store`
  - [ ] `progress`
  - [ ] `logs`
  - [ ] `metrics`
  - [ ] `error`
  - [ ] `audit`
- [ ] Return a structured error when required node request fields are missing.
- [ ] Return a structured error when `module_sha256` does not match `module_source`.
- [ ] Return a structured error when the requested export/operation cannot be executed.

### Import Policy

- [ ] Move node-profile import enforcement into node-owned execution code.
- [ ] Default-deny imports when `python.import_allowlist` is empty or absent.
- [ ] Allow only explicitly listed root modules when `python.import_allowlist` is present.
- [ ] Reject imports whose root module is not allowlisted.
- [ ] Preserve safe builtin behavior appropriate for node execution.
- [ ] Ensure import policy failures produce structured node errors.
- [ ] Add tests for default-deny imports.
- [ ] Add tests for allowlisted imports.
- [ ] Add tests for unallowlisted imports.

### Runtime Environment Policy

- [ ] Derive node `environment_key` from environment name, runtime intent, import intent, dependency intent, and sandbox policy identity.
- [ ] Reject caller-supplied `environment_key` values that do not match host-derived identity.
- [ ] Ensure different dependency/import/runtime/sandbox-policy identities do not share node workers.
- [ ] Ensure dependency-bearing node requests execute only against a selected verified runtime environment.
- [ ] Return a structured environment error when required dependency environment preparation is missing.
- [ ] Return a structured environment error when install receipt verification failed or is absent.
- [ ] Keep normal node execution from installing dependencies implicitly.
- [ ] Add tests for environment-key mismatch.
- [ ] Add tests for dependency-bearing execution without verified environment.
- [ ] Add tests proving incompatible identities do not share live workers or hot child pools.

### Streaming And Events

- [ ] Emit `started` when node execution begins.
- [ ] Capture and emit bounded `stdout` events from executed Python code.
- [ ] Capture and emit bounded `stderr` events from executed Python code.
- [ ] Emit `log` events for host/runtime diagnostics that are safe to expose.
- [ ] Emit `progress` during execution, not only after final return.
- [ ] Emit `artifact` events when artifact refs are created.
- [ ] Emit `result` for successful terminal output.
- [ ] Emit `error` for structured terminal failures.
- [ ] Emit `canceled` when cancellation wins.
- [ ] Emit `done` exactly once for each stream.
- [ ] Keep stream queues bounded and enforce `max_items` on receive.
- [ ] Add tests proving progress can be observed before final result.
- [ ] Add tests for stdout/stderr/log truncation.
- [ ] Add tests for terminal event ordering.

### Result Semantics

- [ ] Preserve node `output` as the primary successful result value.
- [ ] Preserve `state_patch` as JSON object or `null`.
- [ ] Preserve `progress` as latest progress snapshot where available.
- [ ] Preserve `logs` as bounded summaries that do not expose raw source by default.
- [ ] Preserve `audit` fields for package, workflow, source digest, module hash, provenance, runtime, and request identifiers.
- [ ] Return structured runtime errors with safe traceback/message summaries.
- [ ] Return structured timeout errors.
- [ ] Return structured output-limit errors.
- [ ] Add tests for successful output/state patch.
- [ ] Add tests for structured runtime errors.
- [ ] Add tests for timeout.
- [ ] Add tests for output limit.

### Artifacts

- [ ] Decide whether first-class node supports artifacts in this implementation pass.
- [ ] If artifacts are supported, define storage root, reference shape, authorization, lifetime, and cleanup.
- [ ] If artifacts are supported, write artifacts only through host-controlled storage.
- [ ] If artifacts are supported, emit artifact events during streaming.
- [ ] If artifacts are supported, return artifact refs in node responses.
- [ ] If artifacts are not supported, keep a deliberate structured unavailable response and document it as a product decision.
- [ ] Add tests for artifact refs or explicit unavailable-artifact behavior, depending on the decision.

### Cancellation, Status, And Resources

- [ ] Make `workflow-python-stream-send` cancellation interrupt active node execution.
- [ ] Make host-level `workflow-python-cancel-request` cancellation interrupt active node execution.
- [ ] Record terminal request state for canceled node executions.
- [ ] Report active node request status by `environment_key + request_id`.
- [ ] Report latest progress in node request status.
- [ ] Report node resources by `environment_key`.
- [ ] Report node capacity, active calls, available slots, active request IDs, and process count.
- [ ] Report latency, timeout, cancellation, saturation, and error counters.
- [ ] Report per-process CPU/RSS where the host can sample them.
- [ ] Add tests for stream cancellation.
- [ ] Add tests for host-level cancellation.
- [ ] Add tests for request status during active execution.
- [ ] Add tests for resource metrics after success, error, timeout, and cancel.

### Compatibility And Cleanup

- [ ] Keep existing helper-profile clients working while node implementation changes.
- [ ] Keep the current helper-backed node facade only as a temporary compatibility path during migration.
- [ ] Remove helper-backed node execution once direct node execution is verified.
- [ ] Revisit whether the Python helper worker can be reduced after node no longer depends on it.
- [ ] Update `HOSTING_CLIENT_BREAKING_CHANGES.md` only for remaining dependent-project actions.
- [ ] Update public hosting docs after first-class node behavior is implemented and tested.
- [ ] Remove or rewrite stale docs that imply helper-backed node execution is complete first-class node sandboxing.

### Verification

- [ ] Add focused unit tests for node request normalization and validation.
- [ ] Add focused unit tests for node import policy.
- [ ] Add focused unit tests for node runtime environment policy.
- [ ] Add service-level sync execution tests for node success and failure.
- [ ] Add service-level streaming tests for node events and cancellation.
- [ ] Add CLI/channel payload forwarding tests for node sync and stream commands.
- [ ] Add resource/request-status tests for node metrics.
- [ ] Add regression tests proving helper-profile behavior remains unchanged.
- [ ] Run the focused hosting workflow test suite.
- [ ] Record the verified behavior in `hosting_status.md`.

## Non-Goals

- Do not require a specific internal worker file name or contract string in the planning docs.
- Do not merge generic/model worker semantics into workflow runtime semantics.
- Do not promise strong OS-level filesystem/network isolation beyond what the shared sandbox launcher and brokered I/O actually enforce.
- Do not implement implicit dependency installation during normal workflow execution.
