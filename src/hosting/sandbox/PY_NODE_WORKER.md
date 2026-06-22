# Python Node Worker

Date: 2026-06-14
Scope: `workflow_python(profile=node)` execution contract, artifact I/O contract, and relationship to helper/toolbox workers.

## Purpose

The Python node worker is the first-class hosted Python workflow node runtime. It executes one requested Python export from caller-provided source and returns the node-profile envelope:

1. `output`
2. `state_patch`
3. `artifacts`
4. `progress`
5. `logs`
6. `metrics`
7. structured `error`
8. `audit`

The public entrypoints are the workflow facade commands and channel methods:

1. `workflow-python-execute` with `profile=node`
2. `workflow-python-stream-open`
3. `workflow-python-event-subscribe`
4. `workflow-python-stream-send`
5. `workflow-python-stream-close`
6. `workflow-python-resources`
7. `workflow-python-set-capacity`
8. `workflow-python-request-status`
9. `workflow-python-cancel-request`

The implementation lives in `hosting.sandbox.workflow_python_node_runtime`, launches the built-in `hosting.workflow_python_node_worker_ipc` harness, and is called by `WorkflowHelperMixin`. It is not an externally registered IPC worker module. The host starts a child Python harness process for execution and tracks it through the shared hosted workflow pool and request lifecycle.

## Host Lifecycle

Node-profile execution is host-owned:

1. The client calls a workflow facade API.
2. The host derives an `environment_key` from runtime profile, Python dependency/import intent, and sandbox policy.
3. The host validates dependency-bearing work against prepared and verified runtime environments.
4. The host prepares artifact inputs and output slots.
5. The host starts a child Python runtime using the selected Python executable.
6. The host records request lifecycle, progress, stream events, cancellation, metrics, logs, and artifacts.

The node runtime intentionally does not use `execute_workflow_python_helper`. It also does not expose a raw `python -m ...` worker entrypoint to dependent projects.

Request lifecycle states are shared with the hosted pool model:

1. `submitted`: accepted by the host but not yet assigned to a worker slot.
2. `running`: assigned to a node runtime and currently executing.
3. `ok`: completed successfully.
4. `error`: failed before or during execution.
5. `timeout`: exceeded `limits.timeout_ms`.
6. `canceled`: canceled by host request, stream command, or worker shutdown.

Long-running requests can opt into host-side liveness events by setting `limits.heartbeat_interval_ms`. Heartbeats are emitted by the host wait loop as `heartbeat` stream events with `request_id`, `status=running`, `elapsed_ms`, and `remaining_ms`. They do not require sandbox code cooperation and are separate from user progress events.

Stream retention is bounded per request. `limits.stream_max_events` sets the retained live-event queue size used by `workflow-python-event-subscribe`; the host caps this value to a finite range. Subscription responses return a compact `batch` plus helper-normalized `normalized_events`. Loss is reported in `batch.loss` and as a helper `stream_loss` event when helper policy is `mark`. Request status still records total `stream_event_count` for lifecycle metrics.

## Request Contract

Required fields:

1. `module_source`: Python source text
2. `module_sha256`: SHA-256 hex digest of `module_source`
3. `package_id`
4. `workflow_id`
5. `package_source_digest`
6. `operation` or `export_name`
7. `payload`

Optional fields:

1. `request_id`
2. `provenance`
3. `limits`
4. `policy`
5. `python`
6. `artifact_inputs`
7. `artifact_outputs`
8. `execution_mode`
9. `project`
10. `action_manifest` / `actions`
11. `action_name`

The host verifies `sha256(module_source) == module_sha256` before execution. In default module mode, the requested function is found by `export_name` or `operation`, and is called with `payload`.

Snippet request:

```json
{
  "execution_mode": "snippet",
  "module_source": "result = {'output': {'ok': True}}",
  "module_sha256": "...",
  "package_id": "pkg",
  "workflow_id": "wf",
  "package_source_digest": "digest",
  "payload": {}
}
```

Snippet code can read the global `payload` and should assign `result`. `result` follows the same return normalization as a module export.

Project request:

```json
{
  "execution_mode": "project",
  "module_source": "",
  "module_sha256": "e3b0c44298fc1c149afbf4c8996fb924...",
  "package_id": "pkg",
  "workflow_id": "wf",
  "package_source_digest": "project-digest",
  "project": {
    "ref": "@project/src",
    "entrypoint": "pkg.runner",
    "callable": "run",
    "working_directory": ".",
    "env": {"MODE": "test"}
  },
  "payload": {}
}
```

For `project.ref`, the host stages files into the request workspace as an artifact input named by `project.root_input` or `project` by default. Project-local imports are allowed only when the imported module resolves under the staged project root. Global imports still require `python.import_allowlist`.

## Action Manifest

Requests may include an optional `action_manifest` or `actions` field using the
`hosting.sandbox.action_manifest.v1` shape. Each action has a stable `name`,
display metadata, visibility flags, schemas, approval/permission metadata, and
an `entrypoint`. When no manifest is supplied, the host exposes one default
`run` action that routes to the request's existing `export_name`, `operation`,
or project callable.

Card-facing discovery uses `workflow_python_action_describe(...)`, which returns
advertised actions and can include `hidden_allowed` actions when requested.
Execution can select an action by passing `action_name` on the normal workflow
Python request or by calling `execute_workflow_python_action(...)`. The host
routes the selected action into the existing worker contract by setting
`export_name` / `operation`, `execution_mode="snippet"`, or
`project.callable` before the worker receives the request.

Host-side callers can import request builders from `hosting.sandbox.workflow_python_contract` instead of hand-authoring low-level request fields:

1. `build_workflow_python_node_module_request(...)`
2. `build_workflow_python_node_snippet_request(...)`
3. `build_workflow_python_node_project_request(...)`
4. `build_workflow_python_node_uv_project_request(...)`

The builders return normal node request dictionaries accepted by `execute_workflow_python(profile="node", request=...)` and stream open. They fill source hashes, code revisions, package source digests, execution mode, and default payloads. Project builders also fill the empty project `module_source` hash, default `root_input="project"`, recursive `path_mask="*"` project artifact input, and explicit `project_id` / `project_digest` fields. Existing callers may keep providing raw request dictionaries.

## Python Execution API

Node Python code can use these globals:

1. `progress(payload)` or `emit_progress(payload)`: emits a stream progress event.
2. `artifact_inputs`: mapping from declared input artifact name to a sandbox-visible file path or directory path.
3. `artifact_outputs`: mapping from declared file output artifact name to an exact writable file path or directory path.
4. `host`: discoverable cooperative host API client.

Example:

```python
def run(payload):
    with open(artifact_inputs["seed"], "r", encoding="utf-8") as f:
        seed = f.read()

    progress({"message": "writing report"})

    with open(artifact_outputs["report"], "w", encoding="utf-8") as f:
        f.write(seed.upper())

    return {"output": {"ok": True}, "state_patch": {"report_written": True}}
```

Artifact refs are host-side capabilities. Python node code should not dereference `@project/...` or `@artifacts/...` refs itself. It receives host-provisioned file paths through `artifact_inputs` and `artifact_outputs`.

## Host API Back Channel

Node code can call the host through the `host` global:

```python
def run(payload):
    described = host.describe()
    seed = host.fs_read_text("seed")["text"]
    host.fs_mkdir("reports", "nested")
    host.fs_write_text("reports", "nested/report.txt", seed.upper())
    return {"output": {"methods": described["methods"]}}
```

The current host API contract is `hosting.workflow_python.node.host_api.v1`.

`host.describe()` returns:

1. `methods`: available method names
2. `method_descriptions`: descriptions, argument schemas, result schemas, permissions, and async handler metadata
3. `roots`: readable/writable artifact root names for this request
4. `policy`: enabled host API namespaces for this request
5. `transport`: control-channel capabilities

Base discoverable methods:

1. `host.describe`

Known methods such as `fs.list`, `fs.read_text`, `fs.write_text`, `fs.mkdir`,
`fs.stat`, and `http.fetch` appear only when a hosting client/provider session
registers and advertises them for the request. Sandbox policy can further
disable namespaces, but policy no longer causes the hosting service to register
service-owned `fs.*` or `http.fetch` methods by itself.

Convenience methods on `host` call those dispatcher methods:

1. `host.describe()`
2. `host.call(method, arguments)`
3. `host.fs_read_text(root_id, relative_path="", encoding="utf-8")`
4. `host.fs_write_text(root_id, relative_path="", text="", encoding="utf-8", create_parents=True)`
5. `host.fs_list(root_id, relative_path="")`
6. `host.fs_stat(root_id, relative_path="")`
7. `host.fs_mkdir(root_id, relative_path="", parents=True, exist_ok=True)`
8. `host.http_fetch(url, method="GET", headers=None, body_b64="", timeout_seconds=30.0, max_response_bytes=1048576)`

Transport: the host starts the built-in `hosting.workflow_python_node_worker_ipc` harness with a dedicated multiprocessing control channel. The worker sends framed `host_call` messages with `host_call_id` on that channel, the host dispatcher evaluates them, and the host sends matching `host_response` messages back on the same channel. User stdout/stderr remain ordinary execution logs and are not the host RPC transport. The host-side dispatcher supports synchronous and asynchronous handlers. Worker host calls correlate responses by `host_call_id`, so concurrent blocking calls can receive out-of-order host responses safely.

When a client registers the known artifact filesystem methods, the expected
provider behavior maps `fs.*` calls to declared artifact roots:

1. readable roots: declared artifact inputs and declared artifact outputs
2. writable roots: declared artifact outputs only
3. input roots are read-only
4. output roots may be exact files or directories depending on the output declaration
5. relative paths cannot escape the selected root

A node sandbox policy can disable the artifact filesystem namespace with either
shape:

```json
{
  "sandbox": {
    "host_api": {
      "namespaces": {
        "fs": false
      }
    }
  }
}
```

or:

```json
{
  "sandbox": {
    "host_api": {
      "fs": false
    }
  }
}
```

When disabled, `host.describe()` remains available, reports
`policy.artifact_fs=false`, and omits the `fs.*` methods even if a provider
session registered them. Calling a disabled or unregistered method returns an
unsupported-host-method error through the normal host response path.

The HTTP namespace is disabled unless the same sandbox broker policy used by
generic workers allows brokered HTTP and a client/provider session registers
`http.fetch`:

```json
{
  "sandbox": {
    "enabled": true,
    "network": {
      "mode": "brokered_only",
      "allow_hosts": ["example.com"],
      "allow_url_prefixes": ["https://example.com/api/"]
    },
    "brokered_io": {
      "filesystem": false,
      "http": true,
      "subprocess": false
    }
  }
}
```

When enabled and registered, `host.describe()` includes `http.fetch`. Node code
can call either `host.call("http.fetch", {...})` or `host.http_fetch(...)`. The
provider must enforce the URL scheme, host allowlist, URL prefix allowlist,
request method, request headers, response size limit, and timeout. Response
bodies are returned as `body_b64`.

Single-file inline inputs resolve to the file itself. Directory-like inputs and outputs are created by masked/recursive declarations, inline zip inputs, or output declarations with `path_mask` / `mask`.

The current node host dispatcher does not enable arbitrary subprocess or
unrestricted filesystem access. Additional host services should be added only
through policy-gated Host Capability provider sessions.

## Artifact Contract

Artifacts are part of the node sandbox API contract. They are not just response metadata.

Artifact input kinds:

1. `ref`: host resolves an alias ref into a request-scoped input file.
2. `inline`: host writes declared inline text/base64/data into a request-scoped input file.
3. masked `ref`: host copies matching files into a request-scoped input directory and preserves relative paths.
4. inline zip: host expands declared zip bytes into a request-scoped input directory and preserves relative paths.

Artifact output kinds:

1. `ref`: host exposes an exact writable path, validates the written file, and returns a host-minted or host-validated alias ref.
2. `inline`: sandboxed code returns inline bytes/text in `artifacts`, and the host promotes it only when a matching output declaration exists.
3. masked `ref`: host exposes a writable output directory, collects matching files after execution, and returns one ref per collected file.
4. inline zip export: host packs matching output files into one inline zip response without changing artifact ownership.

Alias ref format:

```text
@alias/relative/path
```

Rules:

1. Refs must be relative alias refs beginning with `@`.
2. Absolute paths, `..`, empty path parts, and URL-like refs are rejected.
3. `@artifacts` is the default host-controlled artifact alias.
4. Additional aliases such as `@project` or `@home` are configured through `sandbox_policy.sandbox.artifact_roots`.
5. Policy-configured alias-to-physical-path mappings are part of sandbox policy normalization and therefore part of `environment_key` identity.

`@artifacts/...` and other registered alias refs are resolvable to absolute
paths on the worker-process host by the host artifact manager. `@artifacts` is
always registered to the default workflow artifact root. Additional prefixes
such as `@project` or `@home` are valid only after registration in
`sandbox_policy.sandbox.artifact_roots`. Python node code should still use the
host-provisioned `artifact_inputs` and `artifact_outputs` paths instead of
resolving alias refs itself; alias resolution belongs to the host/harness side.

Sandbox policy example:

```json
{
  "sandbox": {
    "artifact_roots": {
      "project": "O:/repos/example-project/artifacts",
      "home": "C:/Users/me/workflow-artifacts"
    }
  }
}
```

Input ref example:

Host-side callers can build these rows with stable helper constructors from `hosting.sandbox.artifacts`:

1. `artifact_inline_input(...)`
2. `artifact_inline_zip_input(...)`
3. `artifact_ref_input(...)`
4. `artifact_masked_ref_input(...)`
5. `artifact_file_output(...)`
6. `artifact_host_takeover_output(...)`
7. `artifact_producer_owned_output(...)`
8. `artifact_inline_zip_output(...)`

Each helper returns a plain dictionary accepted in `artifact_inputs` or `artifact_outputs`. The helpers fill reasonable defaults for `kind`, filenames, media types, recursive masks, ownership, and inline zip export flags while leaving advisory metadata such as `ttl`, `count`, `max_bytes`, and `encoding` optional.

```json
{
  "name": "seed",
  "kind": "ref",
  "ref": "@project/input/seed.txt",
  "filename": "seed.txt",
  "media_type": "text/plain"
}
```

Inline input example:

```json
{
  "name": "seed",
  "kind": "inline",
  "filename": "seed.txt",
  "text": "hello",
  "media_type": "text/plain",
  "encoding": "utf-8"
}
```

Inline zip input example:

```json
{
  "name": "project",
  "kind": "inline",
  "filename": "project.zip",
  "media_type": "application/zip",
  "encoding": "zip",
  "base64": "..."
}
```

For inline zip inputs, `artifact_inputs["project"]` is a request-scoped directory containing the extracted files. Zip entries with absolute paths or `..` are rejected.

File output declaration:

```json
{
  "name": "report",
  "kind": "ref",
  "filename": "report.txt",
  "media_type": "text/plain"
}
```

Explicit output ref declaration:

```json
{
  "name": "report",
  "kind": "ref",
  "ref": "@project/output/report.txt",
  "filename": "report.txt",
  "media_type": "text/plain"
}
```

Masked input declaration:

```json
{
  "name": "dataset",
  "kind": "ref",
  "ref": "@project/input",
  "path_mask": "*.txt",
  "recursive": true,
  "media_type": "text/plain"
}
```

For masked inputs, `artifact_inputs["dataset"]` is a request-scoped directory containing only matched files. Relative paths under the declared base directory are preserved.

Masked output declaration:

```json
{
  "name": "reports",
  "kind": "ref",
  "ref": "@project/output",
  "path_mask": "*.txt",
  "recursive": true,
  "media_type": "text/plain"
}
```

For masked outputs, `artifact_outputs["reports"]` is a request-scoped writable directory. The host collects files matching `path_mask` after execution. With `recursive=true`, nested matches are included and returned with `relative_path`; explicit refs are expanded by appending the relative path, for example `@project/output/nested/report.txt`.

Host takeover ref output:

```json
{
  "name": "report",
  "kind": "ref",
  "ref": "@project/worker/report.txt",
  "filename": "report.txt",
  "host_takeover": true,
  "media_type": "text/plain"
}
```

With `host_takeover=true`, the returned artifact ref is minted under `@artifacts/...` and its lifetime is host-managed. Without takeover, an explicit output `ref` remains producer-managed.

Inline zip export:

```json
{
  "name": "bundle",
  "kind": "ref",
  "ref": "@project/producer-owned",
  "path_mask": "*.py",
  "recursive": true,
  "export_inline_zip": true,
  "filename": "bundle.zip"
}
```

With `export_inline_zip=true`, the host packs matching files into one inline `application/zip` artifact and does not copy them into `ref`; ownership remains with the producer.

Inline output declaration:

```json
{
  "name": "summary",
  "kind": "inline",
  "filename": "summary.txt",
  "media_type": "text/plain"
}
```

Inline output return:

```python
def run(payload):
    return {
        "output": {"ok": True},
        "artifacts": [
            {"name": "summary", "text": "done", "media_type": "text/plain"}
        ],
    }
```

Input-side `max_bytes`, `count`, `ttl`, `lifetime`, `expires_at`, and `encoding` metadata are accepted as optional advisory metadata. `path_mask` / `mask` and `recursive` are also accepted on input refs to select files from a configured alias root. Inline artifact payloads are receiver-managed. Ref artifacts are producer-managed unless the output declaration asks for `host_takeover` or omits `ref`, in which case the host owns the returned `@artifacts/...` ref. Artifact access control currently uses existing hosting roles plus sandbox policy and configured artifact-root aliases. The current local implementation carries metadata where useful, but it does not implement a separate durable artifact authorization, expiry, cleanup, or external read API.

## Trust Boundary

Sandboxed code cannot mint trusted artifact refs.

The host ignores artifact-looking values unless they match a declared output slot. Examples that remain ordinary JSON output unless declared and host-validated:

1. `{"path": "/tmp/report.csv"}`
2. `{"url": "file:///tmp/report.csv"}`
3. `{"artifact_id": "abc"}`
4. `{"ref": "../other-run/output"}`
5. `{"ref": "workflow-artifact://old/ref"}`

The host mints response artifacts only after one of these happens:

1. A declared file output path was written and collected.
2. A declared inline output name matched returned inline artifact bytes/text.
3. A declared multi-file output was packed as inline zip.

Request-local worker artifact directories are cleaned after collection. Producer-owned explicit refs remain outside host lifetime management; host-takeover refs are copied into the host `@artifacts` root.

## Import And Builtin Policy

Imports are default-deny. Node code may import only root modules listed in `python.import_allowlist`.

The child runtime provides a small safe builtin set. `open` is available only when artifact inputs or outputs are declared, and it is guarded:

1. Read mode is allowed only for declared input paths, declared output paths, and descendants of declared masked input/output directories.
2. Write mode is allowed only for declared output paths or descendants of declared masked output directories.
3. Any other path raises `PermissionError`.

Filesystem access outside artifact paths is not a node-profile API feature.

## Streaming

Node stream event types:

1. `started`
2. `heartbeat`
3. `log`
4. `stdout`
5. `stderr`
6. `progress`
7. `artifact`
8. `result`
9. `error`
10. `canceled`
11. `done`

`artifact` events are emitted only for host-minted or host-accepted artifacts, before the terminal `result` event.

## Long-Lived Workers And Code Edits

Current implementation: the host keeps warm child harness processes for compatible sequential module/snippet requests under the same environment/import/revision identity. A fixed `module_source` plus `module_sha256` is the default source revision identity for each request; callers may pass explicit `code_revision` when they need a host-defined revision label. Corrected snippets/modules should be submitted as new requests with new digests or revision labels. Those requests route to a different warm worker instead of mutating already-loaded code in place. In-flight requests keep their original revision, and idle workers from older revisions are trimmed back to configured capacity after completion. Project requests remain one-shot for now because they can mutate `cwd`, `sys.path`, environment variables, and import caches.

Idle warm workers are recycled when:

1. the same logical `environment_name` derives a different `environment_key`, including sandbox-policy identity changes
2. capacity shrinks below the current idle worker count
3. resource inspection finds an unhealthy idle child process

The recycling path stops only idle workers. In-flight requests keep their original runtime identity and finish or cancel through normal request lifecycle handling.

For a future long-lived Python node worker, code updates should not rely on uv. uv manages dependencies and interpreter environments; it is not the right mechanism for hot-editing workflow source modules.

Implemented module/snippet model:

1. Every loaded snippet/project/module gets a code revision identity such as `module_sha256`, `package_source_digest`, or an explicit `code_revision`.
2. The worker routes execution by `(environment_key, code_revision, entrypoint)` rather than only by environment.
3. When module/snippet code changes, the host starts or reuses a worker for the new revision and lets old in-flight requests finish.
4. In-flight requests keep their original revision.
5. Failed snippets are fixed by submitting a new revision; the old revision is not mutated in place.

The restart/reroute approach used by toolbox remains the conservative default for correctness. A hot-reload path is still future work, and it must be explicit and revision-scoped; otherwise Python import caches and module globals will make bug-fix behavior ambiguous.

## Relationship To Helper And Toolbox Workers

### Python Helper Worker

`hosting.workflow_python_helper_ipc` is still the backing implementation for `workflow_python(profile=helper)`. It has a narrower source-in / JSON-out contract:

1. allowed helper operations only
2. `result` rather than node `output` / `state_patch`
3. no artifact input/output API
4. no node stream model
5. no node response envelope

The node runtime covers the richer Python workflow-node use case. It should eventually support long-running jobs, many different jobs concurrently, and multiple concurrent instances of the same node code. Current decision: keep the helper worker minimally changed for now because it is already small and hot for source-in / JSON-out helper calls. Do not adapt helper over node unless maintaining the separate helper process becomes more expensive than retiring or replacing it.

### Toolbox Worker

`hosting.toolbox_executor_ipc` is a long-lived registered worker for tool execution. It loads toolbox manifests, routes named tools, supports callback/broker clients, and participates in toolbox registration/orchestration state.

The Python node runtime is not a toolbox:

1. no toolbox manifest
2. no tool routing
3. no callback contract
4. no advertised/allowed tool views
5. no toolbox assignment rollout

Both systems share hosting concepts: sandbox policy, environment identity, process lifecycle, request status, cancellation, and resource reporting. They do not currently share a single base worker class. Toolbox and helper are registered IPC worker entrypoints. Python node is a direct host-managed child runtime using `HostedPythonRuntimeBase`, `HostedProcessSandboxBase`, `HostedProcessPoolRegistry`, and `WorkflowPythonNodeRuntimeRegistry`.

## Refactoring Assessment

The current shared base layer is useful but incomplete:

1. `HostedProcessSandboxBase` already centralizes pool, request status, stream queue, capacity, and cancellation bookkeeping.
2. `HostedPythonRuntimeBase` centralizes Python environment identity and runtime environment management.
3. Toolbox still has separate orchestration because it owns tool assignment, bundle staging, callback routing, and persistent toolbox state.
4. Python helper still has separate IPC-worker internals because minimal helper change is currently lower maintenance than adapting helper over node.
5. Python node has runtime-specific launch/protocol internals because it is not a registered IPC worker and has node-specific import/result semantics.

The incomplete part is not the pool concept. The pool concept is the right host-side shape for long-running and concurrent node work. The shared child-runtime layer now owns active child tracking, cancellation lookup, and active resource listing. Runtime-specific code still owns launch, child protocol parsing, hot reuse/recycling, import policy, and result normalization.

Recommended future simplification:

1. Keep Python helper internals minimally changed unless helper-profile maintenance cost justifies retiring or replacing the helper facade.
2. Add explicit long-running/concurrent node job lifecycle and heartbeat behavior on top of the shared pool model.
3. Add Python snippet and multi-module project execution modes.
4. Add uv-managed environment preparation, lock/receipt verification, interpreter selection, and cleanup.
5. Keep toolbox registration/repair/GC persisted state toolbox-specific; use shared lifecycle only for runtime request/resource accounting where practical.
6. Replace Python helper internals only if a later maintenance review shows a concrete cost benefit.

Do not delete the Python helper worker solely because node exists. The current maintenance choice is to leave it small and isolated while new first-class workflow runtime work moves through `profile=node`.
