# QuickJS Workflow JS Node Task Plan

Date: 2026-06-15

Purpose: define the implementation plan for the first-class QuickJS-backed
workflow JS node runtime. This is a clean JavaScript workflow contract with no
Node.js executable requirement.

## Target Outcome

- `workflow_js` becomes a first-class hosted workflow node runtime, analogous to
  `workflow_python(profile=node)`.
- The runtime executes JavaScript through a Python-owned QuickJS child harness,
  not a Node.js subprocess.
- Host interaction uses the same dispatcher pattern as the Python node worker:
  framed child-to-host calls, `host_call_id` correlation, policy-gated
  filesystem/artifact and HTTP services, and request-scoped capability roots.
- The public result shape aligns with Python node: `output`, `state_patch`,
  `artifacts`, `progress`, `logs`, `metrics`, structured `error`, and `audit`.
- QuickJS is positioned as workflow JavaScript, not a Node.js compatibility
  layer. Node built-ins and npm package execution are non-goals unless exposed
  through explicit host APIs or a future bundling step.

## Execution Discipline

- Execute this plan in small reviewable slices.
- Each slice should complete one coherent group of work, update this document by
  checking the completed work item boxes, and commit the code and matching plan
  updates together.
- Do not mark a work item complete until the implementation and relevant
  verification for that item are done.
- Keep documentation-only preparation separate from runtime implementation
  commits.
- Prefer preserving a working tree that can run the focused workflow hosting
  tests after each implementation slice.

## Design Decisions

- Use a clean JS node contract with `exports.run(input, api)` as the default
  entrypoint.
- Keep public workflow facade command names such as `workflow-js-execute`,
  `workflow-js-resources`, `workflow-js-set-capacity`,
  `workflow-js-request-status`, and `workflow-js-cancel-request`, but make them
  route to the JS node runtime.
- Default `workflow_js` profile to `node`.
- Start with a single-script QuickJS contract to avoid depending on Python
  QuickJS binding ESM loader support.
- Allow modern ESM authoring later by adding a host-side bundle/transform step
  that emits the single-script runtime format.
- Provide a host-side bridge import finalizer for already-composed JS source so
  callers can patch allowed `@host/...` imports and inspect disabled or
  unresolved imports before submitting the single script.
- Expose Node-like conveniences only as host-owned capabilities, for example
  artifact filesystem methods, brokered HTTP, codec helpers, selected crypto
  helpers, and console/progress capture.
- Keep UI web component execution separate from QuickJS. QuickJS may validate,
  lint, or bundle UI code, but browser components should execute in a browser
  context loaded dynamically by the Python web host.

## Phase 1: Contract And Runtime Shape

- [x] Add `src/hosting/sandbox/JS_NODE_WORKER.md` as the canonical JS node
      worker contract document.
- [x] Define the JS node request contract with required fields:
      `module_source`, `module_sha256`, `package_id`, `workflow_id`,
      `package_source_digest`, and `payload`.
- [x] Define optional request fields: `request_id`, `provenance`, `limits`,
      `policy`, `javascript`, `artifact_inputs`, `artifact_outputs`,
      `execution_mode`, `project`, and `code_revision`.
- [x] Define the initial supported execution modes:
      `script` for single-source execution and `snippet` for assigning a global
      result.
- [x] Defer multi-module/project execution until a bundling or module-loader
      strategy is selected.
- [x] Add a JS bundle finalizer helper that rewrites only enabled host bridge
      imports and reports allowed, disabled, and unresolved imports.
- [x] Add a constrained JS multi-module bundle helper that resolves passed
      modules, local roots, allowed library roots, disabled library roots, and
      host bridge imports into one deterministic worker script.
- [x] Define the runtime authoring shape for the first implementation:

```javascript
exports.run = function(input, api) {
  return { output: input };
};
```

- [x] Define async semantics explicitly before exposing `async` examples:
      either synchronous host APIs only for v1 or a bounded promise/job pump
      with host-call promise resolution.
- [x] Define the normalized JS return shape:
      plain values become `output`, while objects may carry `output`,
      `state_patch`, `artifacts`, and `progress`.
- [x] Define structured error reasons for invalid source hash, missing export,
      runtime exception, timeout, cancellation, invalid output, output limit,
      unsupported host method, and policy denial.

## Phase 2: Runtime Base And Environment Identity

- [x] Rework `HostedJsRuntimeBase` so it represents the QuickJS workflow node
      runtime instead of the legacy helper compatibility environment.
- [x] Update JS environment identity to include runtime kind, profile,
      QuickJS binding/runtime identity, dependency/bundle intent, host API
      policy, and sandbox policy hash.
- [x] Add a `javascript` policy block parallel to Python policy, with fields
      such as `runtime`, `runtime_hash`, `bundle_hash`, `allowed_host_modules`,
      and future package/bundle metadata.
- [x] Ensure incompatible QuickJS runtime identity, host API policy, sandbox
      policy, or bundle identity creates a distinct `environment_key`.
- [x] Preserve shared pool, request lifecycle, capacity, status, resource, and
      cancellation accounting through `HostedProcessSandboxBase`.

## Phase 3: QuickJS Child Harness

- [x] Add a QuickJS node runtime module, for example
      `hosting.sandbox.workflow_js_node_runtime`.
- [x] Add a built-in child harness entrypoint, for example
      `hosting.workflow_js_node_worker_ipc`.
- [x] Launch the child harness with the selected Python executable and imported
      QuickJS binding; do not launch `node`.
- [x] Track active JS child processes through the shared active child runtime
      registry used by Python node.
- [x] Implement source hash verification before executing QuickJS code.
- [x] Build a fresh QuickJS context per request at first; add warm context reuse
      only after cleanup and global-state policy are defined.
- [x] Inject a minimal global surface:
      `exports`, `console`, `api`, `progress` or `emitProgress`, and selected
      codec helpers.
- [x] Keep ambient host access out of the QuickJS global scope:
      no `process`, unrestricted `require`, unrestricted `import`, direct
      filesystem, direct subprocess, or direct network.
- [x] Capture console output into bounded logs.
- [x] Enforce timeout, output limit, and cancellation at the child-process
      level even when QuickJS binding-level limits are available.
- [x] Add memory limit reporting based on what the selected QuickJS binding and
      platform can actually enforce; report unavailable when not enforced.

## Phase 4: Host API Dispatcher

- [x] Reuse or generalize the Python node host dispatcher for JS node host
      calls.
- [x] Expose `api.describe()` with methods, schemas, permissions, roots,
      policy, and transport capabilities.
- [x] Implement artifact filesystem methods:
      `fs.list`, `fs.read_text`, `fs.write_text`, `fs.mkdir`, and `fs.stat`.
- [x] Enforce read-only artifact input roots and writable artifact output roots.
- [x] Reject relative paths that escape the selected root.
- [x] Implement brokered `http.fetch` through the existing host HTTP policy
      checks when sandbox policy enables brokered HTTP.
- [x] Add `codec` helpers for UTF-8 and base64.
- [x] Add selected `crypto` helpers only when they are deterministic and
      policy-approved, starting with hashing.
- [x] Decide whether host APIs are synchronous-only for v1 or promise-based.
- [x] If promise-based APIs are supported, implement QuickJS job pumping and
      host-call response correlation tests before exposing the public contract.
      Not applicable for v1 because promise-based host APIs are not exposed.
- [x] Ensure host API failures return structured JS node errors and stream
      events rather than raw Python exceptions.

## Phase 5: Artifacts, Streaming, And Results

- [x] Reuse the shared host-side artifact preparation, collection, cleanup, and
      ref-minting helpers used by Python node.
- [x] Support declared artifact inputs: alias refs, inline text/base64/data,
      masked refs, recursive refs, and inline zip inputs.
- [x] Support declared artifact outputs: file refs, masked refs, host takeover,
      producer-owned refs, inline outputs, and inline zip exports.
- [x] Keep sandbox-returned artifact-looking data as ordinary output unless it
      matches a declared output and the host validates it.
- [x] Emit stream events: `started`, `heartbeat`, `log`, `stdout` or
      `console`, `progress`, `artifact`, `result`, `error`, `canceled`, and
      `done`.
- [x] Keep stream retention bounded and report dropped event counts as Python
      node does.
- [x] Return the same top-level response envelope and artifact-store semantics
      as Python node.

## Phase 6: Public Facade And Cleanup

- [x] Change `workflow_js_environment_spec` to derive QuickJS node environment
      specs.
- [x] Change `ensure_workflow_js` to reserve or start QuickJS JS node workers.
- [x] Change `execute_workflow_js` to call the JS node runtime directly instead
      of a proxy RPC path.
- [x] Change `workflow_js_resources`, `set_workflow_js_capacity`,
      `cancel_workflow_js_request`, and `workflow_js_request_status` to use the
      JS node pool.
- [x] Keep runtime selection host-owned for JS workflow execution.
- [x] Replace Node subprocess expectations with QuickJS JS node tests.
- [x] Update worker discovery/status labels so JS workflow workers are reported
      as JS node sandboxes.
- [x] Update CLI and interactive UI wording for JS node workflow execution.

## Phase 7: Tests

- [x] Add contract tests for JS node request normalization and validation.
- [x] Add source hash verification tests.
- [x] Add tests for successful `exports.run` execution.
- [x] Add tests for missing `exports.run` or requested export.
- [x] Add tests for structured runtime errors with safe stack/message summaries.
- [x] Add timeout, cancellation, output-limit, and invalid-output tests.
- [x] Add environment-key tests for QuickJS runtime identity and sandbox policy
      isolation.
- [x] Add host API discovery tests.
- [x] Add artifact filesystem read/write/list/stat/mkdir tests.
- [x] Add tests that input roots are read-only and output roots are writable.
- [x] Add brokered HTTP allowed and denied tests.
- [x] Add progress and console/log stream tests.
- [x] Add artifact input/output collection tests matching Python node coverage.
- [x] Add resource/status/capacity tests for success, error, timeout, and
      canceled JS node requests.
- [x] Add regression tests for host-owned QuickJS runtime execution.
- [x] Add tests for JS host bridge import finalization, disabled imports,
      unresolved imports, and request hash construction.
- [x] Add tests for JS multi-module bundling from passed modules, local roots,
      allowed library roots, disabled library roots, and unsupported Node-style
      imports.

## Phase 8: Documentation Updates

- [x] Update `src/hosting/hosting_access_plan.md` so current state says JS
      workflow is QuickJS node-backed.
- [x] Update `src/hosting/hosting_status.md` after implementation with verified
      behavior and the focused test command used.
- [x] Update `src/hosting/HOSTING.md` examples to use the clean JS node
      contract.
- [x] Update `src/hosting/ENGINE_HOST_CLI.md` and CLI help text for JS node
      workflow execution.
- [x] Keep `src/hosting/sandbox/WORKFLOW_HELPER_WORKER.md` focused on Python
      helper behavior and link JS workflow readers to `JS_NODE_WORKER.md`.
- [x] Add or update `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md` with JS
      workflow client migration guidance.
- [x] Add `src/hosting/sandbox/JS_NODE_WORKER.md` to describe the QuickJS JS
      node contract, host API, artifact rules, stream events, and non-goals.
- [x] Document the JS bridge import finalizer and caller diagnostics in
      `src/hosting/sandbox/JS_NODE_WORKER.md`.
- [x] Update any config/setup docs that currently tell users to install Node.js
      for workflow JS.
- [x] Document the separate plan for browser-executed custom UI components:
      dynamic module serving, registry/versioning, preview iframe, and reload
      without Python web server restart. Keep that separate from QuickJS
      workflow execution.

## Open Questions

- [x] Which QuickJS binding should be the default dependency, and does it expose
      reliable time/memory limits on every supported platform?
      Decision: use the Python `quickjs` package. Memory limits are reported as
      enforced only when `Context.set_memory_limit(...)` succeeds. Binding-level
      time limits are not used while Python callbacks are required; wall-clock
      timeout remains parent-process enforced.
- [x] Should v1 host APIs be synchronous-only, or should promise-based APIs be
      required before landing?
      Decision: v1 host APIs are synchronous-only; promise-returning JS results
      are rejected with `workflow_sandbox_async_unsupported`.
- [x] Should `exports.run(input, api)` be the only v1 entrypoint, or should
      callers be allowed to name exports through `export_name`?
      Decision: `exports.run(input, api)` is the documented default, and
      `export_name` may select another property on `exports`.
- [x] Should bundled ESM authoring be included in the first implementation or
      deferred until after the single-script runtime is stable?
      Decision: defer bundled ESM authoring until the single-script runtime is
      stable. Follow-up implemented a constrained host-side multi-module helper
      that still emits the single-script runtime contract and does not add
      Node/npm compatibility.
- [x] How should QuickJS runtime version and binding version be recorded in
      `runtime_hash` and audit output?
      Decision: `runtime_hash` stays host-policy derived; audit runtime records
      the QuickJS binding name/version where the selected Python package exposes
      it, otherwise `unknown`.

## Non-Goals

- Do not emulate Node.js wholesale.
- Do not support arbitrary npm package execution in the QuickJS runtime.
- Do not rely on QuickJS Python binding ESM module-loader behavior for v1.
- Do not use QuickJS as the browser runtime for custom UI web components.
- Do not promise stronger OS-level isolation than the shared sandbox launcher
  and host dispatcher actually enforce.
