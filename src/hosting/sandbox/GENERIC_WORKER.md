# Generic Worker

Date: 2026-05-21
Scope: generic worker registrations, the built-in model/generic IPC worker, and
their public host proxy APIs. Shared sandbox policy, launch, and broker APIs
are described in [SANDBOX_ARCHITECTURE.md](SANDBOX_ARCHITECTURE.md).

## Purpose

`worker_profile_class="generic"` is a broad registration class, not one fixed
worker implementation. A generic-profile registration can point at any
configured IPC worker command. The profile itself gives the host process
lifecycle, persisted registration, sandbox policy lookup, and proxy routing; it
does not by itself inject workflow-node APIs, toolbox APIs, or host callbacks
into the worker process.

The built-in [../engine_worker_ipc.py](../engine_worker_ipc.py) implementation
can be launched under either a model profile or a generic-style command. In
model mode it drives an in-process `mp13_engine` instance through
`mp13_engine.mp13_engine_api.handle_call_tool(...)`. A config whose
`worker_kind` is `generic` instead supplies its own `worker_command` /
`spawn.command`; hosting starts that process and treats it as generic if it
speaks the expected IPC protocol.

They share the sandbox launch and broker foundation, but they are not staged toolbox executors. They do not load toolbox manifests and do not have logical-toolbox routing.

Workflow Python helper workers are a separate specialization that also uses the generic worker lifecycle and sandbox foundation. See [WORKFLOW_HELPER_WORKER.md](WORKFLOW_HELPER_WORKER.md). JavaScript workflow execution uses the QuickJS node contract documented in [JS_NODE_WORKER.md](JS_NODE_WORKER.md).

## Main Implementation

Primary files:

1. [../engine_worker_ipc.py](../engine_worker_ipc.py): built-in model/generic-compatible IPC server
2. [../service/engines.py](../service/engines.py): spawn, connect-from-config, registration, respawn, shutdown, and RPC proxy helpers
3. [../engine_host_channel.py](../engine_host_channel.py): client/channel wrappers for spawn, proxy, sandbox broker commands, and config-based connect
4. [../engine_host_cli.py](../engine_host_cli.py): CLI command surface
5. [../engine_process_supervisor.py](../engine_process_supervisor.py): older persisted process registration helper used by legacy surfaces

## Startup Model

The host normally starts generic-profile workers through
`EngineHostService.spawn(...)`, often via `connect-from-config`.

The host:

1. allocates IPC family/address
2. generates `MP13_ENGINE_HOST_TOKEN`
3. appends `--ipc-family` and `--ipc-address` to the command if missing
4. sets `MP13_ENGINE_TRANSPORT=ipc`
5. sets `MP13_WORKER_IPC_FAMILY` and `MP13_WORKER_IPC_ADDRESS`
6. normalizes and persists `sandbox_policy`
7. records `sandbox_runtime` from the launcher

For any generic-profile registration, hosting persists the command, IPC
metadata, auth token, sandbox policy, and environment. The worker command must
cooperate with the IPC contract if callers expect `proxy-*` operations to work.

The built-in `engine_worker_ipc` implementation reads:

1. `MP13_ENGINE_ID`
2. `MP13_MODEL_PATH`
3. `MP13_ENGINE_CONFIG_PATH`
4. `MP13_WORKER_CONTRACT`, default `mp13.worker.rpc.v1`
5. stream limit environment variables

In model mode, `connect-from-config` builds the command as
`python -m hosting.engine_worker_ipc`, sets `MP13_MODEL_PATH`, and waits for RPC
readiness. At startup that worker calls `initialize-engine`. Model RPC methods
can later load or unload model instances inside the worker.

For `worker_kind="generic"`, `connect-from-config` builds the command from
`worker_command` / `spawn.command`, does not require a model path, and does not
wait for model-worker readiness. The process must implement the IPC operations
that the client intends to call.

## IPC Message API

The host proxy APIs send these IPC message kinds. A generic-profile worker only
supports the ones implemented by its command. The built-in `engine_worker_ipc`
accepts all of them:

1. `hello`
2. `rpc_call`
3. `stream_open`
4. `stream_recv`
5. `stream_send`
6. `stream_close`
7. `http_request`
8. `shutdown`

The IPC auth key is `MP13_ENGINE_HOST_TOKEN`.

## Generic Profile Operation Surface

For a bare `worker_profile_class="generic"` registration, hosting supports:

1. lifecycle: `spawn`, `ensure-running`, `discover-running`, `shutdown`
2. HTTP-style proxying: `proxy-request`, which sends `kind="http_request"`
3. synchronous RPC proxying: `proxy-rpc-call`, which sends `kind="rpc_call"`
4. async stream proxying: `proxy-rpc-open`, `proxy-rpc-send`,
   `proxy-rpc-recv`, and `proxy-rpc-close`
5. sandbox broker calls: `sandbox-fs-*` and `sandbox-http-fetch`, authorized by
   the persisted `sandbox_policy` for the worker `engine_id`

Those are host-side capabilities. They do not prove that an arbitrary generic
worker command implements a matching in-process API. If the worker does not
listen on the registered IPC endpoint or does not implement a message kind,
the matching proxy operation fails.

The profile does not provide these workflow/toolbox features:

1. no injected `host.call(...)`
2. no Host Capability session discovery or approval requester
3. no workflow node artifact input/output contract
4. no workflow event-subscribe batch contract
5. no toolbox manifest, logical toolbox routing, tool views, or toolbox
   callback harness
6. no custom callback registration from the worker back into the host

## Sync RPC Methods

For the built-in `engine_worker_ipc`, `rpc_call` supports:

1. `rpc.describe`, `describe`, `capabilities`
2. `worker.resources`
3. `worker.resource-status`
4. `model.list`, `model.describe`
5. `model.load`
6. `model.unload`
7. engine tool names handled by
   `mp13_engine.mp13_engine_api.handle_call_tool(...)`

Describe responses include protocol version, contract, sync/async support, cancellation support, model-management support, and configured limits.

## Streaming RPC

For the built-in `engine_worker_ipc`, streaming uses:

1. `stream_open`: starts an async method call and requires `request_id`
2. `stream_recv`: drains queued events with timeout and max item limits
3. `stream_send`: currently supports `{"action": "cancel", "request_id": ...}`
4. `stream_close`: marks the stream closed and requests stop

Stream execution calls `handle_call_tool(method, params)` in a background
thread. Events are lower-level proxy events such as `accepted`, `result`,
`chunk`, `error`, and `final`; they are not the hosted workflow stream event
schema.

Current limit environment variables:

1. `MP13_WORKER_MAX_CONCURRENT_STREAMS`
2. `MP13_WORKER_STREAM_QUEUE_MAX_ITEMS`
3. `MP13_WORKER_MAX_STREAM_RECV_ITEMS`

The CLI/channel exposes these through `proxy-rpc-open`, `proxy-rpc-recv`, `proxy-rpc-send`, and `proxy-rpc-close`.

Workflow runtime facades do not expose the generic worker `stream_recv` shape as their public event read API. Workflow Python and JavaScript node streams use `workflow-*-event-subscribe`, which returns the hosted event batch contract plus helper-normalized events. The generic/proxy `stream_recv` primitive remains a lower-level model-worker IPC surface until the proxy stream cleanup decision is made.

## Workflow Runtime Relationship

Workflow Python helper workers intentionally stay outside `hosting.engine_worker_ipc` so workflow execution does not inherit model-worker routing or `mp13_engine` tool dispatch. Workflow JavaScript is launched as a request-scoped QuickJS child harness by the workflow JS node runtime.

The shared pieces are:

1. `EngineHostService.spawn(...)`
2. persisted worker registration
3. `WorkerSandboxPolicy`
4. `WorkerLaunchRequest`
5. sandbox runtime reporting
6. hosting IPC/RPC
7. lifecycle APIs such as status, ensure-running, and shutdown

The Python helper specialized pieces are:

1. worker module `hosting.workflow_python_helper_ipc`
2. executor kind `workflow_python_helper`
3. execution contract `hosting.workflow_helper.worker.v1`
4. RPC method `execute_workflow_python_helper`

The JavaScript specialized pieces are:

1. runtime module `hosting.sandbox.workflow_js_node_runtime`
2. child harness `hosting.workflow_js_node_worker_ipc`
3. execution contract documented in [JS_NODE_WORKER.md](JS_NODE_WORKER.md)

## HTTP Compatibility API

For the built-in `engine_worker_ipc`, `http_request` is a compatibility shim,
not the preferred new integration API.

Supported routes:

1. `GET /health`
2. `GET /capabilities`
3. `POST /inference`

New integrations should prefer `proxy-rpc-*` for generic workers.

## Host/Public APIs

Relevant host/channel/CLI commands include:

1. `connect-from-config`
2. `spawn`
3. `ensure-running`
4. `discover-running`
5. `shutdown`
6. `proxy-request`
7. `proxy-rpc-call`
8. `proxy-rpc-open`
9. `proxy-rpc-send`
10. `proxy-rpc-recv`
11. `proxy-rpc-close`

Sandbox broker commands are shared with toolbox workers:

1. `sandbox-fs-list`
2. `sandbox-fs-read-text`
3. `sandbox-fs-write-text`
4. `sandbox-fs-mkdir`
5. `sandbox-fs-stat`
6. `sandbox-http-fetch`

These broker commands authorize by the persisted `engine_id` registration policy. The generic worker implementation does not inject the toolbox-specific context helpers; callers invoke broker commands through host APIs or use the shared worker-side clients where they wire an RPC invoker.

`callback_context` on sandbox broker calls is attribution metadata supplied by
the caller and echoed in the broker response. Hosting does not deliver it to a
generic worker process and does not use it as an in-process callback channel.
Toolbox workers attach richer callback context because the toolbox harness owns
that execution context; bare generic workers do not.

## Registration And Respawn

Generic worker registrations persist the command, environment, IPC metadata, auth token, worker profile, sandbox policy, and sandbox runtime. `ensure_running(...)` can respawn a dead registration from this persisted data.

Model workers additionally persist model/config binding metadata such as:

1. `worker_id`
2. `model_instance_id`
3. `model_path`
4. `canonical_model_path`
5. `config_path`
6. `canonical_config_path`
7. `loaded_models`
8. `config_bindings`

The host can reuse running model workers when their runtime profile and loaded model/config bindings match the requested connection.

## Shutdown And Cancellation

Worker shutdown is process-level. The host terminates the process tree and, on Windows, closes the tracked job object when available.

Stream cancellation is request-scoped from the protocol perspective: `stream_send` with `action=cancel` calls the engine `cancel-request` tool and sets the stream stop event. This is distinct from killing the worker process.

## Current Limits

1. Generic workers do not use `ToolboxWorkerStartupSpec`.
2. Generic workers do not have logical toolbox state, bundle staging, or profile routing.
3. POSIX launch is currently a plain subprocess even when sandbox metadata is present.
4. HTTP compatibility routes are intentionally narrow.
5. Brokered I/O is available through host APIs but not automatically surfaced as rich injected execution context like toolbox tools get.
