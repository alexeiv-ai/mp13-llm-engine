# Sandbox Architecture

Date: 2026-05-21
Scope: shared sandbox foundation used by hosted workers. Worker-specific details live in [TOOLBOX_WORKER.md](TOOLBOX_WORKER.md) and [GENERIC_WORKER.md](GENERIC_WORKER.md).

## Purpose

The sandbox layer is the common host-side boundary for launching worker processes with an explicit policy, registering their runtime metadata, and brokering selected I/O back through the host. It is not only a toolbox feature anymore.

Current worker/runtime families:

1. toolbox executors, described in [TOOLBOX_WORKER.md](TOOLBOX_WORKER.md)
2. generic/model workers, described in [GENERIC_WORKER.md](GENERIC_WORKER.md)
3. workflow Python helper workers, described in [WORKFLOW_HELPER_WORKER.md](WORKFLOW_HELPER_WORKER.md)
4. workflow JS QuickJS node runtime, described in [JS_NODE_WORKER.md](JS_NODE_WORKER.md)

Workflow Python helper workers use the same launch, registration, policy, and
runtime metadata foundation, but expose `hosting.workflow_helper.worker.v1`
instead of the model-worker contract. Workflow JS uses a Python-owned QuickJS
child harness for JS node requests.

The host remains the trust boundary and lifecycle authority. Workers are separate processes that expose RPC over hosting IPC; the host decides what process is launched, what sandbox policy is attached to the registration, and what brokered callbacks are allowed.

## Shared Runtime Model

The shared path is:

1. caller asks `EngineHostService.spawn(...)` or a higher-level helper to start a worker
2. host normalizes `sandbox_policy` with `WorkerSandboxPolicy.from_mapping(...)`
3. host allocates IPC metadata and worker auth token
4. host calls `launch_worker_process(WorkerLaunchRequest(...))`
5. launcher starts either a restricted Windows process or a plain subprocess
6. host persists the registration, including `sandbox_policy` and `sandbox_runtime`
7. host proxies worker RPC or brokered I/O by looking up the persisted registration

Primary implementation files:

1. [policy.py](policy.py): shared policy dataclasses and normalization
2. [launcher.py](launcher.py): `WorkerLaunchRequest`, `WorkerLaunchResult`, `launch_worker_process`
3. [windows.py](windows.py): Windows restricted-token / low-integrity launch support
4. [broker_fs.py](broker_fs.py): host-side filesystem broker
5. [broker_http.py](broker_http.py): host-side HTTP broker
6. [worker_fs.py](worker_fs.py): transport-agnostic worker-side filesystem client
7. [worker_http.py](worker_http.py): transport-agnostic worker-side HTTP client
8. [../service/sandbox_api.py](../service/sandbox_api.py): host service methods exposed to callback paths
9. [../service/engines.py](../service/engines.py): spawn, registration, shutdown, and broker lookup integration
10. [../service/workflow_helpers.py](../service/workflow_helpers.py): workflow helper spawn convenience API

## Internal Runtime Bases

The refactoring introduces internal bases for new hosted runtime kinds. These
are implementation layers, not public sandbox kinds:

1. [runtime_base.py](runtime_base.py): deterministic environment identity,
   pool/request/worker models, shared stream event names, IPC message family
   names, registration metadata helpers, resource response helpers, and
   cancellation result shapes.
2. [runtime_pool.py](runtime_pool.py): in-memory process pool registry keyed by
   concrete runtime kind and `environment_key`; tracks desired capacity,
   workers, active requests, recent request outcomes, latency metrics, progress
   snapshots, and cancellation/error counts.
3. [process_base.py](process_base.py): `HostedProcessSandboxBase`, an internal
   language-neutral composition layer over the pool registry for capacity,
   request status, progress, and cancellation plumbing.
4. [python_runtime.py](python_runtime.py): `HostedPythonRuntimeBase` and
   `HostedPythonRuntimeManager`, which add Python runtime/environment identity,
   runtime environment realization, install plan/lock/verify/receipt hooks, and
   Python executable selection.
5. [js_runtime.py](js_runtime.py): `HostedJsRuntimeBase`, a thin QuickJS
   workflow node identity layer for JavaScript requests. It intentionally does
   not reuse Python venv machinery.
6. [toolbox_runtime.py](toolbox_runtime.py): `HostedToolboxRuntimeBase`, an
   internal toolbox identity adapter. It keeps toolbox staging, tool routing,
   callbacks, brokered I/O, and `toolbox_venvs` ownership in toolbox code while
   adding shared `environment_key` / `environment_identity` registration
   metadata.

Concrete workflow facades currently use these layers incrementally:

1. `workflow_python(profile=helper)` exposes environment spec, environment
   lifecycle hooks, ensure, execute, resources, capacity, cancel, and request
   status while keeping the old Python helper worker as temporary execution
   compatibility code.
2. `workflow_python(profile=node)` executes Python exports through the hosted
   workflow Python runtime and wraps results in the node response contract.
   Stream-open returns immediately and emits `started`, `log`, optional
   `progress`, `result` or structured `error`, `canceled`, and `done`.
   Dependency-bearing node execution requires an explicitly prepared and
   verified runtime environment; normal execution does not install packages
   implicitly. Declared artifact inputs and outputs use the host-provisioned
   local artifact data plane described below and in
   [PY_NODE_WORKER.md](PY_NODE_WORKER.md).
3. `workflow_js(profile=node)` exposes environment spec, ensure, execute,
   resources, capacity, cancel, and request status through the JS runtime base
   and QuickJS node runtime.
4. `toolbox_executor` registrations now include shared hosted environment
   identity through `HostedToolboxRuntimeBase`; toolbox public APIs and
   lifecycle semantics are otherwise unchanged.

Generic/model workers remain separate. They share IPC vocabulary ideas and
proxy commands, but their model-worker semantics do not become workflow or
toolbox semantics.

## Workflow Artifact Boundary

Artifact I/O is part of the node sandbox contract, but artifact authority stays
with the host. The current implementation provides local host-controlled alias
refs such as `@artifacts/...` for declared output files, supports
policy-configured roots such as `@project/...`, resolves declared input refs
into request-scoped input paths, supports inline zip inputs, and supports declared inline inputs/outputs.
The remaining durable-service concerns are explicit:

1. A host-controlled storage root outside arbitrary sandbox paths.
2. Stable relative alias reference shape, including request/workflow/package identity.
3. Read and write authorization rules for refs.
4. Lifetime, expiry, cleanup, and garbage-collection policy.
5. Size and count limits per request.
6. Input-ref, inline-input, and inline-zip resolution into sandbox-visible input paths.
7. Output-slot resolution into exact sandbox-visible writable paths.
8. A brokered write API if path-based output slots are insufficient.
9. Stream `artifact` events only for host-minted refs.
10. Response `artifacts` entries only for host-validated output files, declared inline outputs, or declared inline zip exports.

Sandboxed code may return ordinary JSON values that look like paths, URLs, or
artifact IDs, but those values are not trusted artifact refs. The host only
mints artifact refs after validating files from declared output locations,
registering them in host-controlled storage, accepting inline bytes for a
declared inline output name, or packing declared output files into inline zip.
Explicit ref outputs remain producer-managed unless the host is asked to take
over ownership, in which case they are copied into `@artifacts/...`.
Input-side size/count/lifetime/encoding fields are
advisory metadata. Stronger authorization, expiry, cleanup, and external
artifact-read APIs remain future durable-service work.

Alias refs are local resolver inputs for the host/harness artifact manager.
`@artifacts` is always registered to the default workflow artifact root, and
policy-configured aliases such as `@project` or `@home` are registered through
`sandbox_policy.sandbox.artifact_roots`. A valid ref under a registered prefix
resolves to an absolute path on the worker-process host. Sandboxed code should
not perform that resolution itself; it receives request-scoped input/output
paths or root IDs from the host. Co-located dependent backends that need local
file access must use the same registered-prefix resolver context rather than
treating unregistered or sandbox-returned strings as trusted artifact refs.

## Shared API

### `WorkerSandboxPolicy`

`WorkerSandboxPolicy` is the persisted and runtime policy envelope. It accepts either a nested `{"sandbox": ...}` mapping or the sandbox body directly.

Current fields:

1. `enabled`: enables restricted launch where supported and enables broker checks
2. `profile`: policy profile name, default `generic_worker_v1`
3. `filesystem.rules`: brokered roots with `root_id`, `path`, and `access`
4. `artifact_roots`: alias-to-path mappings for refs such as `@project/...`
5. `process`: `allow_subprocess`, `inherit_parent_handles`, and platform status metadata
6. `network`: `mode`, `allow_hosts`, `allow_url_prefixes`, and platform status metadata
7. `platform_policy.windows`: `restricted_token`, `integrity_level`, `job_object`
8. `brokered_io`: `filesystem`, `http`, `subprocess`

The policy object is deliberately pragmatic: it records platform support status in the policy shape, but only the implemented launch and broker paths enforce behavior.

### `WorkerLaunchRequest`

`WorkerLaunchRequest` contains:

1. `engine_id`
2. `command`
3. `cwd`
4. `env`
5. `log_path`
6. `sandbox_policy`

`launch_worker_process(...)` returns `WorkerLaunchResult` with:

1. `pid`
2. launched `command`
3. `persisted_env`
4. `runtime` diagnostics

On Windows with `policy.enabled=True`, the launcher calls `launch_restricted_worker(...)`. Otherwise it uses a normal subprocess. Normal Windows launches still attach a kill-on-close job object where possible.

### Runtime Environments

Host-managed Python runtime environments now have a shared model even though toolbox APIs still expose toolbox-oriented wrapper names.

Current roots:

1. `<hosting_root>/toolbox_venvs/<venv_key>` for existing toolbox executor environments
2. `<hosting_root>/runtime_envs/<venv_key>` for new non-toolbox runtime environments

Compatibility rule:

1. existing `toolbox_venvs` entries remain readable
2. toolbox executor environments continue to use `toolbox_venvs`
3. new workflow helper environments use `runtime_envs`
4. entries are not eagerly copied between roots

Runtime environment metadata includes `environment_root_kind` and `environment_consumer_kind` so review, GC, and dependent projects do not need to infer semantics from directory names.

Runtime Python selection uses a bootstrap/preverified interpreter only while a dependency-bearing environment is not verified. A no-package/no-op environment can activate its realized venv immediately. A dependency-bearing environment switches to its realized venv only after install execution and install receipt verification are both recorded as `ok`.

### Brokered Filesystem

Host-side API:

1. `sandbox_fs_list`
2. `sandbox_fs_read_text`
3. `sandbox_fs_write_text`
4. `sandbox_fs_mkdir`
5. `sandbox_fs_stat`

The broker looks up the worker registration by `engine_id`, reconstructs `WorkerSandboxPolicy`, requires `sandbox.enabled`, requires `brokered_io.filesystem`, resolves `root_id` to a configured filesystem rule, rejects path traversal, and checks read/write access per operation.

Worker-side helpers:

1. shared transport-agnostic helper: `BrokeredFilesystemClient` in [worker_fs.py](worker_fs.py)
2. toolbox execution-context helper: `context.fs` in [../toolbox_executor_ipc.py](../toolbox_executor_ipc.py)

### Brokered HTTP

Host-side API:

1. `sandbox_http_fetch`

The broker requires:

1. `sandbox.enabled`
2. `brokered_io.http`
3. `network.mode == "brokered_only"`
4. `http` or `https` URL
5. optional host allowlist match
6. optional URL-prefix allowlist match

It strips unsafe request headers such as `Host`, `Content-Length`, and `Connection`, applies timeout and response-size limits, and returns base64 response bodies.

### Callback Context

Brokered service methods accept optional `callback_context`. When present, the host returns it in the broker response. Toolbox workers use this for attribution to the originating toolbox, tool name, call id, arguments, and callback signature.

Generic/model workers can call the same service commands through the host CLI or channel, but the current generic worker contract does not inject a rich per-tool callback context.

## Platform Boundaries

Windows sandboxing is implemented as restricted-token / low-integrity launch plus job-object cleanup. It is useful as a process boundary and write boundary, but it should not be presented as strong read isolation.

Linux and other POSIX launches currently use `plain_subprocess` in the shared launcher. Brokered filesystem and HTTP still enforce host-side checks, but there is no implemented OS-level Linux sandbox wrapper in this package yet.

Direct network allowlisting is not enforced as an OS-level network filter. The trustworthy network path is brokered HTTP with `network.mode="brokered_only"`.

## Registration Contract

The persisted engine registration is the host's lookup point for sandbox behavior. Relevant fields include:

1. `engine_id`
2. `pid`, and optionally `worker_pid` / `launcher_pid`
3. `command`, `cwd`, `env`
4. `worker_transport`, `worker_ipc_family`, `worker_ipc_address`
5. `worker_auth_token`, `worker_auth_header`
6. `worker_profile_class`
7. `sandbox_policy`
8. `sandbox_runtime`
9. optional worker-specific metadata such as `executor_kind`, `bundle`, `environment`, `tool_access`, and `capabilities`

`ensure_running(...)` respawns from the persisted command/env/policy for generic registrations. Toolbox replacement and repair are handled by toolbox-specific persisted state, because one logical toolbox can span multiple worker registrations.

## Current Limits

1. Windows Low IL is not strong read isolation.
2. POSIX workers currently launch as plain subprocesses.
3. Direct network policy is metadata unless the worker uses brokered HTTP.
4. Brokered filesystem is path-root based and does not replace OS ACLs.
5. Brokered HTTP supports simple fetch semantics, not streaming or a general browser/network stack.
6. Sandbox policy is attached to worker registrations; per-request narrowing is owned by higher-level toolbox scope/constraint APIs.
