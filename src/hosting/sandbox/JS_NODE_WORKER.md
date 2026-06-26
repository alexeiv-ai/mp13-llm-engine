# JavaScript Node Worker

Date: 2026-06-15
Scope: `workflow_js` node-profile execution contract for a QuickJS-backed
hosted JavaScript runtime.

## Purpose

The JavaScript node worker is the first-class hosted JavaScript workflow
runtime. It uses a Python-owned QuickJS child harness and returns the
node-profile envelope:

1. `output`
2. `state_patch`
3. `artifacts`
4. `progress`
5. `logs`
6. `metrics`
7. structured `error`
8. `audit`

The public entrypoints are the workflow facade commands and channel methods:

1. `workflow-js-execute`
2. `workflow-js-stream-open`
3. `workflow-js-event-subscribe`
4. `workflow-js-stream-send`
5. `workflow-js-stream-close`
6. `workflow-js-resources`
7. `workflow-js-set-capacity`
8. `workflow-js-request-status`
9. `workflow-js-cancel-request`

This runtime is not a Node.js runtime. It does not expose Node built-ins or npm
package execution by default. It executes QuickJS workflow code with explicit
host-provided capabilities.

## Host Lifecycle

JS node execution is host-owned:

1. The client calls a workflow facade API.
2. The host derives an `environment_key` from QuickJS runtime identity, JS
   policy, bundle/source identity, host API policy, and sandbox policy.
3. The host validates request fields and source identity.
4. The host prepares artifact inputs and output slots.
5. The host starts or reuses a Python child harness that owns a QuickJS context.
6. The child executes the requested JS source with only injected host APIs.
7. The host records lifecycle, progress, stream events, cancellation, metrics,
   logs, and artifacts.

The runtime exposes workflow facade APIs rather than raw worker spawn APIs to
dependent projects.

Request lifecycle states follow the hosted pool model:

1. `submitted`
2. `running`
3. `ok`
4. `error`
5. `timeout`
6. `canceled`

## Request Contract

Required fields:

1. `module_source`: JavaScript source text
2. `module_sha256`: SHA-256 hex digest of `module_source`
3. `package_id`
4. `workflow_id`
5. `package_source_digest`
6. `payload`

Optional fields:

1. `request_id`
2. `provenance`
3. `limits`
4. `policy`
5. `javascript`
6. `artifact_inputs`
7. `artifact_outputs`
8. `execution_mode`
9. `code_revision`
10. `export_name`
11. `instance_state_mode`
12. `action_manifest` / `actions`
13. `action_name`

The host verifies `sha256(module_source) == module_sha256` before execution.

Supported JS worker execution modes:

1. `script`, the default: execute one finalized JS source and call
   `exports[export_name || "run"](payload, api)`.
2. `snippet`: execute one finalized JS source and read global `result`.

The material difference is the entrypoint contract. `script` mode is for
reusable node code with explicit callable exports and `export_name` selection.
`snippet` mode is for short evaluated source that writes one global `result`;
it does not call an exported function and ignores `export_name`.

`instance_state_mode` defaults to `ephemeral`. For pinned script/module
instances, clients may set `instance_state_mode="persistent_module"` to keep a
QuickJS context and its globals alive across sequential
`workflow-js-instance-execute` calls. Persistent module state is not valid for
`snippet` or `project` execution.

Script mode should use a single-script contract:

```javascript
exports.run = function(input, api) {
  return { output: input };
};
```

The default export name is `run`. If `export_name` is supported in v1, it should
select a property on `exports`, not a Node ESM export.

Snippet request:

```json
{
  "execution_mode": "snippet",
  "module_source": "result = { output: payload };",
  "module_sha256": "...",
  "package_id": "pkg",
  "workflow_id": "wf",
  "package_source_digest": "digest",
  "payload": {"ok": true}
}
```

Snippet code can read the global `payload` and should assign `result`.

The JS worker does not support Python node `project` or uv-project execution
modes. Multi-file JS authoring is handled before execution by
`build_workflow_js_module_bundle(...)`, which still emits one `module_source`
submitted as normal `script` mode.

## Action Manifest

Requests may include an optional `action_manifest` or `actions` field using the
`hosting.sandbox.action_manifest.v1` shape. Each action has a stable `name`,
display metadata, visibility flags, schemas, approval/permission metadata, and
an `entrypoint`. When no manifest is supplied, the host exposes one default
`run` action that routes to the request's existing `export_name` or to
`exports.run`.

Card-facing discovery uses `workflow_js_action_describe(...)`, which returns
advertised actions and can include `hidden_allowed` actions when requested.
Execution can select an action by passing `action_name` on the normal workflow
JavaScript request or by calling `execute_workflow_js_action(...)`. The host
routes the selected action into the existing worker contract by setting
`export_name` or `execution_mode="snippet"` before the worker receives the
request.

Dynamic discovery is opt-in. Calling
`workflow_js_action_describe(dynamic=True, request=...)` executes
`exports.describe_actions(payload)` by default, or the entrypoint configured
under `request["action_discovery"]["entrypoint"]`, and normalizes the returned
manifest through `hosting.sandbox.action_manifest.v1`. The discovery callable
may return either `{"output": {"actions": [...]}}`,
`{"output": {"action_manifest": {...}}}`, or a raw list/dict manifest from the
worker response. Passing `instance_id` targets an already-created pinned
instance; it does not implicitly create one.

## JavaScript Execution API

The QuickJS context should expose a small global surface:

1. `exports`: object containing callable entrypoints.
2. `payload`: request payload for snippet mode.
3. `api`: host capability object.
4. `progress(payload)` or `emitProgress(payload)`: emits a stream progress
   event.
5. `console`: bounded log capture.

Example:

```javascript
exports.run = function(input, api) {
  const seed = api.fs.readText("seed");
  api.progress({ message: "writing report" });
  api.fs.writeText("report", "", seed.toUpperCase());
  return {
    output: { ok: true },
    state_patch: { report_written: true }
  };
};
```

Artifact refs are host-side capabilities. JS code should not dereference
`@project/...` or `@artifacts/...` refs itself. It receives request-scoped root
names through `api.describe()` and host API calls.

## Host API Back Channel

JS code calls the host through the injected `api` object:

```javascript
exports.run = function(input, api) {
  const described = api.describe();
  const seed = api.fs.readText("seed");
  api.fs.mkdir("reports", "nested");
  api.fs.writeText("reports", "nested/report.txt", seed.toUpperCase());
  return { output: { methods: described.methods } };
};
```

The planned host API contract is `hosting.workflow_js.node.host_api.v1`.

`api.describe()` returns:

1. `methods`
2. `method_descriptions`
3. `roots`
4. `policy`
5. `transport`
6. `runtime`

Base discoverable methods:

1. `host.describe`
2. `codec.base64_encode`
3. `codec.base64_decode`
4. `crypto.sha256`

Known methods such as `fs.list`, `fs.read_text`, `fs.write_text`, `fs.mkdir`,
`fs.stat`, and `http.fetch` appear only when a Host Capability session registers
and advertises them for the request. The daemon-owned implementation uses
`provider_kind="service_broker"` and is registered through the known
service-broker method helpers. Client-owned callback APIs still use
`provider_kind="client_session"`. Sandbox policy can further disable
namespaces, but policy no longer causes the hosting service to register
service-owned `fs.*` or `http.fetch` methods by itself.

Convenience methods on `api` may wrap dispatcher methods:

1. `api.describe()`
2. `api.call(method, arguments)`
3. `api.callAsync(method, arguments)`
4. `api.describeAsync()`
5. `api.fs.readText(rootId, relativePath="", encoding="utf-8")`
6. `api.fs.readTextAsync(rootId, relativePath="", encoding="utf-8")`
7. `api.fs.writeText(rootId, relativePath="", text="", encoding="utf-8")`
8. `api.fs.writeTextAsync(rootId, relativePath="", text="", encoding="utf-8")`
9. `api.fs.list(rootId, relativePath="")`
10. `api.fs.listAsync(rootId, relativePath="")`
11. `api.fs.stat(rootId, relativePath="")`
12. `api.fs.statAsync(rootId, relativePath="")`
13. `api.fs.mkdir(rootId, relativePath="", options={})`
14. `api.fs.mkdirAsync(rootId, relativePath="", options={})`
15. `api.http.fetch(url, options={})`
16. `api.http.fetchAsync(url, options={})`
17. `api.http.fetchJsonAsync(url, options={})`

Transport should reuse the framed host-call pattern from Python node: the child
harness sends `host_call` messages with `host_call_id`, the host dispatcher
evaluates them, and the host returns matching `host_response` messages.
`host_call_id` is scoped to that worker/request IPC conversation. It correlates
responses inside the child runtime; it is not a global route across arbitrary
daemon control channels. Cross-channel callback routing would need an explicit
session/channel/worker ownership protocol with auth, cancellation, close,
backpressure, and response-routing rules.

The JS worker does not need a different parent IPC contract for async support.
It uses the same `host_call` and `host_response` message shapes as Python node.
The implementation difference is child-local: the QuickJS harness can keep
multiple JS promises pending and resolve or reject them when the matching
`host_response` arrives.

When a client registers known artifact filesystem methods as a `service_broker`
session, the daemon maps `fs.*` calls to the worker engine's declared artifact
roots after Host Capability approval:

1. readable roots: declared artifact inputs and declared artifact outputs
2. writable roots: declared artifact outputs only
3. input roots are read-only
4. output roots may be exact files or directories depending on output
   declaration
5. relative paths cannot escape the selected root

The HTTP namespace is disabled unless sandbox policy enables brokered HTTP and
a Host Capability session registers `http.fetch`. When enabled and registered
as `service_broker`, the daemon validates URL scheme, host allowlist, URL
prefix allowlist, method, headers, timeout, and response size from the worker
sandbox policy. Client-owned providers must enforce their own backend policy
before returning results.

## Async Semantics

JS worker async support is QuickJS promise support, not Node.js event-loop
compatibility. The child harness owns the QuickJS context and pumps
`execute_pending_job()` while a script or snippet result promise is pending.
During that pump it also polls the worker IPC channel for `host_response`
messages and resolves or rejects the matching JS promise by `host_call_id`.
The daemon/parent runtime never pumps QuickJS jobs; it only dispatches host
calls and sends responses back over the worker IPC channel.

Supported async forms:

1. `exports.run(payload, api)` may return a value or a promise.
2. snippet mode may assign `result` to a value or a promise.
3. `api.callAsync(method, arguments)` returns a promise for a host dispatcher
   response.
4. async convenience wrappers such as `api.fs.readTextAsync(...)` and
   `api.http.fetchAsync(...)` return promises.
5. `api.http.fetchJsonAsync(...)` returns a promise that parses the brokered
   HTTP response `body_b64` as JSON.

The synchronous wrappers remain available. `api.call(...)`, `api.fs.readText`,
and related sync helpers block the child harness until the matching
`host_response` arrives. Async wrappers allow multiple in-flight host calls;
out-of-order responses are correlated by `host_call_id`.

Limits and caveats:

1. no Node timers, Node streams, libuv handles, or npm event-loop behavior
2. no background task lifetime after the terminal worker result
3. timeout and cancellation are request-scoped; pending promises fail with the
   same terminal request failure semantics as the worker, and timeout details
   include pending async `host_call_id` values when available
4. `host_call_id` remains scoped to the worker/request IPC conversation, not a
   globally routable daemon identifier
5. sync and async host calls share one response-correlation buffer, so an
   out-of-order async response observed during a sync wait can still be applied
   to the right pending promise later
6. a `host_response` whose `host_call_id` does not match a pending JS host call
   fails the request with `host_response_unknown_host_call_id`

## Module And Import Policy

The JS worker executes one already-finalized script. The runtime does not ask
QuickJS to resolve modules, load files, load npm packages, emulate Node.js, or
interpret `require`. Code submitted to `workflow-js-execute` must already be a
single `module_source` that assigns callable exports such as `exports.run`.

QuickJS core has ES module support, but this host runtime deliberately keeps
module loading outside the child process. That keeps filesystem access,
dependency selection, host bridges, policy checks, and source hashing in the
Python host where they can be audited before execution.

Runtime import policy:

1. no unrestricted runtime `import`
2. no unrestricted `require`
3. no Node built-ins such as `fs`, `path`, `buffer`, or `node:*`
4. no npm package resolution
5. host APIs are available only through injected globals such as `api`

## Long-Lived Instance Semantics

`workflow_js_instance_create`, `workflow_js_instance_execute`,
`workflow_js_instance_list`, and `workflow_js_instance_close` route compatible
module/snippet calls to a pinned host worker process. By default the pinned
process is a transport and startup optimization only: each execution creates a
fresh QuickJS context from the submitted `module_source`, so module globals,
closures, and other in-context JS values are not reused between requests.

When clients set `instance_state_mode="persistent_module"` on an explicit
pinned script/module instance, the child harness creates one QuickJS context for
the current code revision and reuses that context for sequential calls.
Top-level JS state, closures, and exported functions persist until close,
crash, cancel-driven process termination, or explicit replacement. Request-local
bindings such as `payload`, `api` callbacks, progress, console, and host-call
correlation are reset for each call.

Pinned module/snippet instances separate worker-process compatibility from code
identity. `runtime_key` identifies the reusable worker process, while `code_key`
identifies the currently submitted code/package revision. Edited code with a new
`module_sha256` or `code_revision` can run on the same pinned worker process
when the instance is idle and the worker-process `runtime_key` is unchanged.
`workflow_js_instance_create(..., replace=True, ...)` updates `code_key` without
restarting that worker in this compatible case. For persistent module
instances, executing a different code revision without `replace=true` returns
`workflow_js_instance_code_replacement_required`; replacement is the explicit
state reset boundary.

Project-mode JS instances can reuse the pinned host worker process, but they do
not preserve a project heap. Each project execution creates a fresh QuickJS
context and fresh project module cache. Clients must declare this reset policy
with `javascript.project_instance_policy` or `project.instance_policy`:

```json
{
  "context": "new_per_request",
  "module_cache": "reset",
  "globals": "reset",
  "async_jobs": "drain_or_cancel",
  "host_handles": "reset"
}
```

Project code is loaded from `project.ref` through the artifact staging path.
Entrypoints are CommonJS-style files using `exports`, `module.exports`, and
relative `require("./...")`. Dot entrypoints map to paths, for example
`pkg.runner` maps to `pkg/runner.js`. Node built-ins, `node_modules`, ESM
imports, and persistent project module state are not part of this contract.

Authoring with imports is supported by host-side helpers that emit the
single-script worker contract. These helpers are preprocessing tools; their
output is still ordinary JS source plus `module_sha256`.

### Host Bridge Finalizer

`hosting.sandbox.build_workflow_js_bundle(...)` finalizes one already-composed
source string. It rewrites static imports that target enabled host bridge
specifiers into bindings against injected QuickJS globals, then emits
deterministic `module_source` and `module_sha256`.

Default host bridge specifiers:

1. `@host/api` -> `api`
2. `@host/fs` -> `api.fs`
3. `@host/http` -> `api.http`
4. `@host/codec` -> `api.codec`
5. `@host/crypto` -> `api.crypto`
6. `@host/console` -> `console`
7. `@host/progress` -> `api.progress` for default imports
8. `@host/call` -> `api.call` for default imports
9. `@host/describe` -> `api.describe` for default imports

The helper accepts `host_description`, usually from `api.describe()`, and
`sandbox_policy` so policy-gated bridges match the effective host toolbox.
Callers may also pass an explicit `bridge_imports` map for custom host-backed
imports. Custom bridge expressions must still point at host-provided globals or
methods; they are not arbitrary package imports.

Example source:

```javascript
import fs, { readText } from "@host/fs";
import { sha256 } from "@host/crypto";

exports.run = function(input) {
  const seed = readText("seed", "");
  return { output: { digest: sha256(seed), fs_available: !!fs } };
};
```

The finalizer rewrites those imports to single-script bindings:

```javascript
const fs = api.fs;
const { readText } = api.fs;
const { sha256 } = api.crypto;
```

Diagnostics:

1. `resolved_allowed_imports`: bridge specifiers that were enabled and patched
2. `resolved_disabled_imports`: known bridge specifiers disabled by policy
3. `unresolved_imports`: imports not present in the bridge table

`ok` is true only when every static import is resolved and enabled.

### Multi-Module Bundler

`hosting.sandbox.build_workflow_js_module_bundle(...)` accepts a constrained
module set and emits one worker script. It is intended for producers that want
to author multiple JS files with import/export syntax while still submitting a
single runtime payload.

Inputs:

1. `entry_module`: normalized entry module id, for example `main.js`
2. `modules`: passed module rows with `id` and `source`
3. `local_roots`: optional folders for resolving missing relative imports
4. `allowed_lib_roots`: optional folders for known allowed bare imports
5. `disabled_lib_roots`: optional folders for known but disabled bare imports
6. host bridge policy inputs shared with `build_workflow_js_bundle(...)`

Resolution rules:

1. passed modules are matched by normalized `id`
2. relative imports such as `./x.js` and `../shared/x.js` resolve against
   passed modules first, then `local_roots`
3. bare imports such as `math` resolve only from `allowed_lib_roots`
4. bare imports found under `disabled_lib_roots` are reported as disabled
5. allowed library modules may import other modules under the same allowed root
6. all modules may import enabled `@host/...` bridges

Rejected forms:

1. Node built-ins and `node:*`
2. `require(...)`
3. dynamic `import(...)`
4. `export ... from` re-export syntax
5. dependency cycles

The bundler returns `module_source`, `module_sha256`, `resolved_modules`,
`resolved_allowed_imports`, `resolved_disabled_imports`, `unresolved_imports`,
`rejected_imports`, `bundle_segments`, and `bundle_line_map`. Callers can add
missing modules or adjust allowed and disabled roots, call the helper again, and
submit the resulting script only when `ok=true`.

The emitted script includes comment markers around shared runtime segments and
each bundled module segment. The returned line map lets callers resolve a bundle
line number from a QuickJS exception back to the original module id and source
line where available. Helper functions are available for diagnostics:

1. `describe_workflow_js_bundle_source(source)` reads segment markers from a
   bundle source string
2. `extract_workflow_js_bundle_segment(bundle_or_source, name)` extracts a
   marked module or shared runtime segment
3. `resolve_workflow_js_bundle_line(bundle_or_source, line_number)` resolves a
   generated bundle line to the segment, module id, and original line when the
   returned `bundle_line_map` is available

This is intentionally a small authoring subset. It supports local JS module
composition and host bridges, but it does not imply Node.js compatibility, npm
installation, package.json interpretation, browser module loading, or QuickJS
module-loader behavior.

## Artifact Contract

Artifacts should match the Python node contract.

Artifact input kinds:

1. alias refs such as `@artifacts/...` or policy-configured roots such as
   `@project/...`
2. inline text/base64/data
3. masked refs
4. recursive refs
5. inline zip inputs

Artifact output kinds:

1. exact file refs
2. declared inline outputs
3. masked refs
4. recursive output collection
5. host takeover refs
6. producer-owned refs
7. inline zip exports

Sandboxed JS cannot mint trusted artifact refs. Returned values such as
`{ ref: "../other-run/output" }`, `{ path: "/tmp/report.csv" }`, or
`{ url: "file:///tmp/report.csv" }` remain ordinary output unless the host has
validated a declared output slot and minted or accepted the artifact ref.

`@artifacts/...` aliases are currently local host-controlled artifact refs.
They can be consumed by later workflow requests as declared inputs, but they are
not resolved by sandboxed JS code. On the host/harness side, every valid ref
with a registered prefix resolves to an absolute path on the worker-process
host. `@artifacts` is always registered to the default workflow artifact root;
non-default prefixes such as `@project` or `@home` must be registered in
`sandbox_policy.sandbox.artifact_roots`. A separate remote download route, if
needed, still needs explicit helper naming, authorization, metadata shape,
range/streaming limits, expiry/cleanup semantics, and errors for missing,
expired, unauthorized, or unavailable refs.

## Streaming

JS node stream event types:

1. `started`
2. `heartbeat`
3. `log`
4. `console`
5. `progress`
6. `artifact`
7. `result`
8. `error`
9. `canceled`
10. `done`

`artifact` events are emitted only for host-minted or host-accepted artifacts,
before the terminal `result` event.

## QuickJS Runtime Limits

The runtime should report:

1. QuickJS binding name and version when available
2. QuickJS engine version when available
3. timeout limit and enforcement mode
4. memory limit and enforcement mode
5. output limit
6. stream retention limits
7. host API namespace policy

When memory limits or instruction/time limits are not actually enforced by the
selected binding on the current platform, the runtime must report that clearly
instead of implying enforcement.

## Relationship To Python Node

JS node should be on par with Python node for the host-side node envelope and
pooling model:

1. environment-keyed routing
2. hosted process pool accounting
3. request lifecycle states
4. request status
5. stream sessions
6. host dispatcher
7. artifact preparation, collection, and cleanup
8. cancellation and resource reporting

Supported JS worker modes are narrower than Python node modes:

1. `script`: one finalized JS source, usually either hand-authored,
   bridge-finalized, or emitted by `build_workflow_js_module_bundle(...)`
2. `snippet`: one finalized JS source that assigns global `result`

Within those modes, choosing JS node instead of Python node mainly changes:

1. async: both runtimes can use sync or awaitable parent host dispatchers. JS
   additionally supports promise-returning script/snippet results and explicit
   `api.callAsync(...)` wrappers by pumping QuickJS jobs in the child harness.
   This is QuickJS promise support, not Node/libuv compatibility.
2. host calls: both runtimes use the same host dispatcher pattern. JS exposes
   `api.call(...)`, `api.callAsync(...)`, and `api.fs/http/codec/crypto`
   wrappers; Python exposes `host.call(...)`, `host.fs.*`, and
   `host.http.fetch(...)`.
3. imports and code shape: JS worker execution uses one finalized source and
   has no runtime loader; local JS modules must be bundled first. Python
   module/snippet mode can use allowlisted Python imports at runtime.
4. environment: JS currently runs in a static QuickJS environment selected by
   runtime/source/policy identity. Python node can also run dependency-bearing
   requests against prepared runtime environments.
5. artifacts: both use host-prepared artifact inputs/outputs. JS accesses them
   through `api.fs.*`; Python can use host-provisioned paths and guarded
   `open(...)`, plus host API calls.

Not supported by the JS worker runtime, compared with broader Python node
features:

1. Python node `project` mode
2. Python uv-project/dependency environment execution
3. Python import allowlist semantics at runtime
4. loading multiple source files directly in the QuickJS child
5. resolving Node/npm packages at runtime
6. treating browser components as QuickJS execution targets

The JS multi-module helper is a producer-side authoring convenience. It may
read passed module rows, local roots, and allowed/disabled library roots, but
its output is still one script submitted to the JS worker. It is not equivalent
to Python project execution because the QuickJS child does not receive a project
directory, mutate `cwd`, manage `sys.path`, or import files during execution.

Runtime-specific JS code still owns:

1. QuickJS context creation
2. JS global injection
3. JS result normalization
4. JS error formatting
5. QuickJS job pumping for promise results and async host APIs
6. bundle/transform behavior for constrained JS authoring helpers

## Relationship To Custom UI Components

QuickJS is not the browser runtime for custom UI web components. Browser-side
components need DOM, CSS, events, Shadow DOM, and browser module loading.

The Python host can separately support UI components by storing component
source, serving dynamic browser ES modules by version/hash, sending reload
events over WebSocket or SSE, and rendering previews in a browser or iframe
without restarting the Python web server. QuickJS may validate, lint, or bundle
that source, but the component execution target should remain the browser.

## Non-Goals

1. Preserve Node ESM `import(data:)` behavior.
2. Emulate Node.js wholesale.
3. Run arbitrary npm packages.
4. Expose direct filesystem, direct network, or subprocess access.
5. Use QuickJS as a browser DOM/web-component runtime.
