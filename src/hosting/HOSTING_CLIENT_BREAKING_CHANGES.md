# Hosting Client Breaking Changes

Date: 2026-06-19

## JavaScript Helper Removed In Favor Of QuickJS Node Worker

The old Node.js-based JS helper API is removed. Clients should use the hosted
`workflow_js` node worker facade instead of depending on a Node install,
Node-style module loading, or helper-specific filesystem and HTTP behavior.

The JS worker should now be evaluated as a narrower, safer peer of the Python
node worker:

1. both use the hosted node envelope, request lifecycle, stream events,
   cancellation, resource reporting, artifact preparation, and host dispatcher
   model
2. both route host API calls through worker/request-scoped `host_call_id`
   correlation
3. both support host dispatchers implemented as sync functions or awaitable
   Python callables
4. JS script and snippet results may now be promises
5. JS exposes explicit async host APIs such as `api.callAsync(...)`,
   `api.fs.readTextAsync(...)`, and `api.http.fetchAsync(...)`

## Migration Shape

Old JS helper clients must submit one finalized JS source to the workflow
facade:

```javascript
exports.run = async function(payload, api) {
  const seed = await api.fs.readTextAsync("seed", "");
  await api.fs.writeTextAsync("report", "report.txt", seed.toUpperCase());
  return { output: { ok: true } };
};
```

The request must include `module_source`, `module_sha256`, `package_id`,
`workflow_id`, `package_source_digest`, `payload`, and any declared artifact
inputs or outputs. `script` mode calls `exports[export_name || "run"]`.
`snippet` mode evaluates one source and reads global `result`; `result` may be
a promise.

Multi-file JS authoring is producer-side only. Use the JS bundle helpers to
turn constrained local modules and allowed `@host/...` bridge imports into one
auditable `module_source`. The QuickJS child does not load files, resolve npm,
interpret `package.json`, or implement `require`.

## Usage Differences From Python Node

The JS worker is closer to PY node after the async changes, but it is not a
drop-in replacement for all PY node modes.

If a dependent backend already integrates with the PY node worker, it can treat
the JS worker as the same hosted node shape for request routing, lifecycle, host
dispatch, artifacts, and cancellation. The drift is in the language runtime and
entrypoint surface, not in the parent host contract.

Same parent-host behavior as PY node:

1. hosted process pooling and request status
2. node envelope output, state patch, progress, artifacts, logs, metrics, and
   structured errors
3. host API back channel with sync or awaitable host dispatchers
4. declared artifact input/output handling through host-owned roots
5. request cancellation and timeout behavior
6. worker/request-scoped `host_call_id` response correlation

JS async parity:

1. `exports.run(payload, api)` may return a value or a promise.
2. `snippet` mode may assign `result` to a value or a promise.
3. `api.callAsync(method, arguments)` returns a promise for the same host
   dispatcher used by sync calls.
4. `api.fs.*Async` and `api.http.fetchAsync(...)` are convenience wrappers over
   `api.callAsync(...)`.
5. parent host dispatchers may be regular Python callables or awaitables, as
   with PY node.

JS-specific differences:

1. JS has only `script` and `snippet` execution modes.
2. JS runs in a static QuickJS environment selected by source/runtime/policy
   identity.
3. JS does not have Python project mode, uv-project execution, dependency
   installs, Python import allowlists, or direct guarded `open(...)`.
4. JS does not expose Node built-ins, npm packages, browser DOM APIs, direct
   filesystem, direct network, subprocess, or a Node/libuv event loop.
5. JS async means QuickJS promise jobs plus host-call promises. It does not
   imply timers, streams, background tasks after the terminal result, or Node
   event-loop compatibility.

Migration from PY-node assumptions:

1. replace direct file paths or guarded `open(...)` with declared roots and
   `api.fs.*` / `api.fs.*Async`
2. replace Python import/dependency expectations with pre-bundled JS source
3. replace host helper calls with `api.call(...)` or `await api.callAsync(...)`
4. keep artifact refs as host-owned values; JS should not translate
   `@artifacts/...` or other registered prefixes itself

Artifact refs remain host-owned. Sandboxed JS should use declared root names
through `api.fs.*`; it should not resolve aliases such as `@artifacts/...`
itself. On the host/harness side, registered artifact prefixes resolve to
absolute paths on the worker-process host.
