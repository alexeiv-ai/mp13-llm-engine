# Hosting Client Remaining Changes

Date: 2026-06-15

Purpose: track only remaining dependent-project changes. Previously completed helper-profile migration items are intentionally omitted.

## Current State

- Helper-profile dependent clients have already migrated to workflow facade APIs.
- No additional client action is required for existing short helper-profile Python execution.
- JavaScript workflow helper-profile execution has been replaced by `workflow_js(profile=node)` with the QuickJS JS node contract.
- `workflow_python(profile=node)` now has a direct node execution path and no longer returns helper-shaped nested results.
- Dependency-bearing node-profile execution now requires host-prepared and verified runtime environments before execution.

## Remaining Client Changes

- Clients that own node-profile workflow execution must validate against the direct node response envelope and stream event model before adopting it.
- JS workflow clients must stop using `workflow_js(profile=helper)`, helper operation allowlists, Node ESM source loading behavior, `node_executable`, or Node.js runtime/version assumptions. Use `exports.run = function(input, api) { ... }` with `workflow-js-execute`.
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
- `@artifacts/...` refs are local host artifact aliases. On a local host
  endpoint, a returned ref is resolvable to an absolute path on the
  worker-process host by the host artifact manager: `@artifacts` is always
  registered to the default workflow artifact root, and non-default prefixes
  such as `@project` or `@home` must be registered in
  `sandbox_policy.sandbox.artifact_roots`. This is a local resolver contract,
  not a remote download contract; dependent backends that need HTTP download
  behavior still need explicit auth, metadata, range, lifetime, and error
  semantics.
- Node-profile clients should prefer artifact helper constructors in `hosting.sandbox.artifacts` for common artifact input/output rows instead of hand-authoring every low-level artifact field.
- Node-profile clients may select multiple artifact files with `path_mask` or `mask` and `recursive` on input or output artifact declarations. Masked inputs are exposed to Python code as directories containing matched files. Masked outputs are exposed as writable directories and return one host-minted ref per collected file, with `relative_path` populated.
- Node-profile clients may use `export_inline_zip` to export many output files as one inline zip without changing ownership. They may use `host_takeover` when the host should copy a ref output into `@artifacts/...` and own its lifetime; otherwise explicit output refs remain producer-managed.
- Node-profile clients may use `execution_mode=snippet` for arbitrary source snippets without `operation` / `export_name`, or `execution_mode=project` with `project.ref`, `project.entrypoint`, and `project.callable` for staged multi-module project execution. Prefer the host-owned request builders in `hosting.sandbox.workflow_python_contract` for module, snippet, staged-project, and uv-project requests instead of hand-authoring hashes, project identity, and default artifact inputs.
- Node-profile clients that use the Python node host API should call `host.describe()` and handle policy-gated methods. Artifact filesystem methods can be disabled with `sandbox_policy.sandbox.host_api.namespaces.fs=false`; brokered HTTP is exposed as `http.fetch` / `host.http_fetch(...)` only when sandbox policy enables brokered HTTP with `sandbox.enabled=true`, `sandbox.brokered_io.http=true`, and `sandbox.network.mode=brokered_only`. Custom host API transport consumers must correlate responses by `host_call_id` because the node transport now advertises out-of-order-safe responses.
- Clients that provide dependency-management UI or orchestration must call host-controlled prepare/lock/verify/install/receipt APIs explicitly before dependency-bearing execution. Normal workflow execution does not install dependencies implicitly.
- Clients must treat Python helper internals, toolbox worker internals, and toolbox persisted registration/repair/GC state as implementation details. The maintained public surface is the workflow/toolbox host API and this change log, not internal worker or persisted-state compatibility.

## Migrating From The Old JS Helper API

The old JavaScript helper API was removed rather than wrapped for compatibility.
Clients must migrate to the QuickJS-backed `workflow_js(profile=node)` contract.
The new runtime is intentionally not a Node.js subprocess and does not provide
Node ESM loading, npm package resolution, `require`, Node built-ins, or
`node_executable` selection.

Replace old entry points as follows:

| Old JS helper usage | New QuickJS node usage |
| --- | --- |
| `workflow_js(profile=helper)` | `workflow_js(profile=node)` |
| `execute_workflow_js_helper` internal RPC | public `workflow-js-execute` / `execute_workflow_js(profile="node", ...)` |
| `spawn_workflow_js_helper` | `workflow-js-ensure` |
| `workflow_js_helper_resources` | `workflow-js-resources` |
| `set_workflow_js_helper_capacity` | `workflow-js-set-capacity` |
| `cancel_workflow_js_helper_request` | `workflow-js-cancel-request` |
| helper operation allowlists | explicit `exports.run(input, api)` or `export_name` |
| Node ESM / `require` / npm packages | bundle to one script before submission, or use host APIs on `api` |
| `node_executable`, `node_version`, `MP13_WORKFLOW_JS_NODE` | remove; the host owns the Python QuickJS child runtime |

Build requests with the node envelope:

```json
{
  "profile": "node",
  "request": {
    "request_id": "req-123",
    "module_source": "exports.run = function(input, api) { return {output: input}; };",
    "module_sha256": "<sha256 of module_source>",
    "package_id": "pkg",
    "workflow_id": "wf",
    "package_source_digest": "digest",
    "payload": {"value": 7},
    "limits": {
      "timeout_ms": 5000,
      "output_limit_bytes": 65536,
      "memory_limit_mb": 128
    }
  }
}
```

Rewrite helper code to use the injected `api` object instead of ambient Node
globals. For example:

```javascript
exports.run = function(input, api) {
  api.progress({ step: "start" });
  const seed = api.fs.readText("seed");
  api.fs.writeText("report", "", seed.toUpperCase());
  return {
    output: { ok: true },
    state_patch: { report_written: true }
  };
};
```

Artifact migration is declaration-driven. Clients must declare readable inputs
in `artifact_inputs` and writable outputs in `artifact_outputs`; JS code uses
the root names from those declarations through `api.fs.*`. Do not pass raw host
paths to JS and do not expect JS to dereference `@artifacts/...` or
`@project/...` aliases directly.

Streaming clients should consume node stream events directly. The JS node
runtime can emit `started`, `stdout`, `log`, `progress`, `artifact`, `result`,
`error`, `canceled`, and `done`; stream receives are bounded and include
drop-count metadata so clients can detect backpressure loss.

`host_call_id` values are per worker/request IPC correlation IDs. They are not
globally routable callback IDs across daemon control channels. Cross-channel
callback routing would require a separate protocol for session/channel identity,
worker ownership, auth/capabilities, cancellation, close semantics,
backpressure, and response routing.

QuickJS can run bundled single-file JavaScript, but it should not be treated as
Node.js. If clients want ESM-style authoring, bundle or transform modules before
submission into one deterministic `module_source` that assigns `exports.run`;
then submit the matching `module_sha256`.

For host-backed imports in already-composed JS source, clients can call
`hosting.sandbox.build_workflow_js_bundle(...)`. The helper rewrites enabled
`@host/...` bridge imports, reports `resolved_allowed_imports`,
`resolved_disabled_imports`, and `unresolved_imports`, and sets `ok=false` when
anything remains disabled or unresolved. It does not resolve Node built-ins,
npm packages, arbitrary ESM graphs, or QuickJS module loader behavior.

Error handling should move from helper-shaped nested results to the direct node
response envelope. Terminal failures use structured reasons such as
`workflow_sandbox_runtime_error`, `workflow_sandbox_timeout`,
`workflow_sandbox_canceled`, `workflow_sandbox_invalid_output`, artifact errors,
or host API policy failures.

## No Remaining Action For Already Migrated Helper Clients

- No action for clients already using `workflow-python-*` helper-profile APIs.
- JS clients already using `workflow-js-*` command names must update request payloads to `profile:"node"` and the QuickJS `exports.run(input, api)` contract.
- No action for clients already routing helper resources, capacity, request status, and cancellation by `environment_key`.
