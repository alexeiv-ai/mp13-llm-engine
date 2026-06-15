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
3. `workflow-python-stream-recv`
4. `workflow-python-stream-send`
5. `workflow-python-stream-close`
6. `workflow-python-resources`
7. `workflow-python-set-capacity`
8. `workflow-python-request-status`
9. `workflow-python-cancel-request`

The implementation lives in `hosting.sandbox.workflow_python_node_runtime` and is called by `WorkflowHelperMixin`. It is not an externally registered IPC worker module. The host starts a child Python process for execution and tracks it through the shared hosted workflow pool and request lifecycle.

## Host Lifecycle

Node-profile execution is host-owned:

1. The client calls a workflow facade API.
2. The host derives an `environment_key` from runtime profile, Python dependency/import intent, and sandbox policy.
3. The host validates dependency-bearing work against prepared and verified runtime environments.
4. The host prepares artifact inputs and output slots.
5. The host starts a child Python runtime using the selected Python executable.
6. The host records request lifecycle, progress, stream events, cancellation, metrics, logs, and artifacts.

The node runtime intentionally does not use `execute_workflow_python_helper`. It also does not expose a raw `python -m ...` worker entrypoint to dependent projects.

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

The host verifies `sha256(module_source) == module_sha256` before execution. The requested function is found by `export_name` or `operation`, and is called with `payload`.

## Python Execution API

Node Python code can use these globals:

1. `progress(payload)` or `emit_progress(payload)`: emits a stream progress event.
2. `artifact_inputs`: mapping from declared input artifact name to a sandbox-visible file path or directory path.
3. `artifact_outputs`: mapping from declared file output artifact name to an exact writable file path or directory path.

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

## Artifact Contract

Artifacts are part of the node sandbox API contract. They are not just response metadata.

Artifact input kinds:

1. `ref`: host resolves an alias ref into a request-scoped input file.
2. `inline`: host writes declared inline text/base64/data into a request-scoped input file.
3. masked `ref`: host copies matching files into a request-scoped input directory and preserves relative paths.

Artifact output kinds:

1. `ref`: host exposes an exact writable path, validates the written file, and returns a host-minted or host-validated alias ref.
2. `inline`: sandboxed code returns inline bytes/text in `artifacts`, and the host promotes it only when a matching output declaration exists.
3. masked `ref`: host exposes a writable output directory, collects matching files after execution, and returns one ref per collected file.

Alias ref format:

```text
@alias/relative/path
```

Rules:

1. Refs must be relative alias refs beginning with `@`.
2. Absolute paths, `..`, empty path parts, and URL-like refs are rejected.
3. `@artifacts` is the default host-controlled artifact root.
4. Additional aliases such as `@project` or `@home` are configured through `sandbox_policy.sandbox.artifact_roots`.
5. Alias-to-physical-path mappings are part of sandbox policy normalization and therefore part of `environment_key` identity.

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

Input-side `max_bytes`, `count`, `ttl`, `lifetime`, `expires_at`, and `encoding` metadata are accepted as optional advisory metadata. `path_mask` / `mask` and `recursive` are also accepted on input refs to select files from a configured alias root. The current local implementation carries metadata where useful, but it does not implement a durable artifact authorization, expiry, cleanup, or external read API.

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
2. `log`
3. `stdout`
4. `stderr`
5. `progress`
6. `artifact`
7. `result`
8. `error`
9. `canceled`
10. `done`

`artifact` events are emitted only for host-minted or host-accepted artifacts, before the terminal `result` event.

## Relationship To Helper And Toolbox Workers

### Python Helper Worker

`hosting.workflow_python_helper_ipc` is still the backing implementation for `workflow_python(profile=helper)`. It has a narrower source-in / JSON-out contract:

1. allowed helper operations only
2. `result` rather than node `output` / `state_patch`
3. no artifact input/output API
4. no node stream model
5. no node response envelope

The node runtime covers the richer Python workflow-node use case. It should eventually support long-running jobs, many different jobs concurrently, and multiple concurrent instances of the same node code. It does not completely subsume the helper profile yet because helper clients may depend on the narrower operation allowlist and compatibility response shape. The correct migration path is to keep helper-profile compatibility until dependent projects explicitly move to `profile=node` or the helper facade is replaced by a compatibility adapter over the node runtime.

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
4. Python helper still has separate IPC-worker internals because it remains a compatibility process for helper-profile clients.
5. Python node has separate child-runtime internals because it is not a registered IPC worker and has node-specific artifact/import/result semantics.

The incomplete part is not the pool concept. The pool concept is the right host-side shape for long-running and concurrent node work. The missing part is a shared child-runtime layer underneath the pool that can own launch, child protocol, hot reuse/recycling, cancellation, resource sampling, and result normalization across Python node and helper-compatible runtimes.

Recommended future simplification:

1. Extract a small hosted child-runtime interface for `execute`, `cancel`, and `resources` so Python node and helper child pools can share process lifecycle mechanics.
2. Add explicit long-running/concurrent node job support on top of the shared pool model.
3. Add Python snippet and multi-module project execution modes.
4. Add uv-managed environment preparation, lock/receipt verification, interpreter selection, and cleanup.
5. Keep toolbox orchestration separate, but adapt toolbox executor registrations to report through the same normalized pool/resource models where possible.
6. Replace Python helper internals with either a thin compatibility adapter over node runtime or a shared child-runtime implementation only after dependent helper-profile clients no longer require the current response and operation semantics.

Do not delete the Python helper worker solely because node exists. It still has a compatibility niche for helper-profile workflow calls and already migrated dependent projects that expect helper semantics.
