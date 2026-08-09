# Toolbox Worker

Date: 2026-05-21
Scope: sandboxed toolbox executor implementation and APIs. Shared sandbox policy, launch, and broker APIs are described in [SANDBOX_ARCHITECTURE.md](SANDBOX_ARCHITECTURE.md).

## Purpose

Toolbox workers run staged tool code in host-managed worker processes. They preserve the native toolbox programming model while allowing the host to route each tool through a sandbox profile, broker filesystem and HTTP access, and rebuild executors from persisted logical toolbox state.

Important distinction:

1. a logical toolbox is user-facing state
2. a toolbox worker is one live executor for one staged bundle/profile
3. one logical toolbox can span multiple toolbox workers

## Main Implementation

Primary files:

1. [../toolbox_executor_ipc.py](../toolbox_executor_ipc.py): toolbox worker process and RPC handlers
2. [../toolbox/bundle_models.py](../toolbox/bundle_models.py): bundle/profile/startup/environment dataclasses
3. [../toolbox/staging.py](../toolbox/staging.py): bundle staging
4. [../toolbox/manifest.py](../toolbox/manifest.py): staged manifest loading
5. [../toolbox/orchestration.py](../toolbox/orchestration.py): grouping, staging, environment selection, and spawn orchestration
6. [../toolbox/hosted_ref.py](../toolbox/hosted_ref.py): hosted toolbox ref and builder API
7. [../toolbox/execution.py](../toolbox/execution.py): hosted execution harness
8. [../service/toolbox_runtime.py](../service/toolbox_runtime.py): host service registration, routing, gate, execute, and cancel APIs
9. [../service/toolbox_env.py](../service/toolbox_env.py): environment descriptions, repair, reconcile, GC, and reference reporting

`hosting.toolbox_harness` is a compatibility import path that re-exports the public `hosting.toolbox` package API.

## Data Model

### Logical Toolbox

A logical toolbox is keyed by `toolbox_id`. Persisted logical toolbox state is the source of truth. Live executor registrations are derived runtime state.

Routing is toolbox-scoped:

1. host receives `toolbox_id + tool_name`
2. host resolves which sandbox profile owns that tool
3. host forwards to the active executor registration for that profile

Version-2 definition rollout supplies host-resolved profile assignments. An
exactly reused profile is neither staged nor spawned. Added and replaced
profiles are staged and registered with `routing_state="candidate"`; scan-based
routing excludes that state. Before a candidate can be published, the host
requires successful RPC readiness, an exact (not subset) tool inventory, the
planned resolved-profile and environment identities, and the matching verified
hermetic-environment receipt.

### Sandbox Profile

`SandboxProfileSpec` contains:

1. `profile_id`
2. `environment_name`
3. `required_imports`
4. `sandbox_policy`

If `profile_id` is absent, the profile id is derived from a stable fingerprint of environment name, required imports, and sandbox policy.

### Bundle And Manifest

`ToolboxBundleSpec` produces a staged manifest with:

1. `executor_kind="toolbox_executor"`
2. `bundle_id`
3. `toolbox_id`
4. `sandbox_profile`
5. `bundle_revision`
6. `manifest_hash`
7. staged source file hashes
8. manual tool definitions
9. auto-callable entries
10. intrinsic tool activation
11. hidden tool state
12. callback signatures and `non_restartable` metadata

The executor loads only the staged manifest and staged bundle contents. It does not discover tools from ambient host process state.

### Startup Spec

`ToolboxWorkerStartupSpec` is the structured worker startup contract. It carries:

1. `worker_id`
2. `sandbox_id`
3. `toolbox_revision`
4. `manifest_path`
5. `scratch_root`
6. optional `engines_state_file`
7. optional `control_state_file`
8. optional `venv_path`
9. `ipc_family`
10. `ipc_address`
11. `auth_token_env`
12. `execution_contract`
13. `callback_contract`
14. `policy`

The host writes this spec under hosting state and passes its path through `MP13_TOOLBOX_WORKER_SPEC_PATH`. Legacy manifest/env fallbacks still exist but should not be treated as the preferred contract.

## Worker RPC API

Toolbox workers speak the common hosting IPC message shape with `kind="hello"` and `kind="rpc_call"`.

Supported RPC methods:

1. `rpc.describe`, `describe`, `capabilities`: returns protocol metadata, executor kind, registered tool names, and tool metadata
2. `toolbox.describe`: returns bundle identity, registered tool names, metadata, and parallel-execution notes
3. `toolbox.execute`: executes one staged `ToolCall`
4. `host.call`: invokes supported host callbacks directly

`toolbox.execute` rejects unstaged tool names inside the executor even if the host should normally route before dispatch.

## Execution Context API

Tool code can receive context/helper objects through normal toolbox execution injection:

1. `context.host.call(method, arguments)`
2. `context.fs.list_dir/read_text/write_text/mkdir/stat`
3. `context.http.fetch`
4. `context.callbacks.invoke(callback_name, payload)`

Host callback methods supported by the worker:

1. `fs.list`
2. `fs.read_text`
3. `fs.write_text`
4. `fs.mkdir`
5. `fs.stat`
6. `http.fetch`
7. `callback.invoke`

Brokered filesystem and HTTP calls use the shared Host Capability
`provider_kind="service_broker"` route. `context.host.call(...)`,
`context.fs.*`, and `context.http.*` all dispatch through a local Host
Capability broker, then into the shared service-broker registry/dispatcher.
The daemon-owned broker still authorizes the actual IO from the persisted
sandbox policy for the worker `engine_id`.

`context.host.describe()` returns the same Host Capability discovery shape used
by node workers. Worker `rpc.describe` / `toolbox.describe` responses also
include `host_capabilities` for the advertised service-broker host-call
surface.

Toolbox host API approval is independent from toolbox tool gating. Public
toolbox execution entrypoints may provide `host_api_approval` with a Host
Capability approval policy. When that policy requires approval, service-broker
`fs.*` / `http.fetch` calls request approval through the hosted callback
binding before brokered IO executes. Approval denial prevents the brokered IO
call. Approval does not widen sandbox filesystem or network policy.
Approval callbacks receive normalized
`hosting.sandbox.host_capability_approval.v1` payloads. Use
`argument_preview` for bounded policy-relevant values such as `root_id`,
`relative_path`, `url`, and `method`; do not depend on raw `arguments` in
client approval code.

The worker also attaches shared callable-surface metadata to callback context
under `callable_surface`. That metadata uses
`hosting.toolbox.brokered_io.call_surface.v1` and includes method identity,
schema/method/policy digests, safe correlation fields, and the effective
bridge-policy intersection. Approval/audit events use the Host Capability
approval/audit shape.

## Hosted Callback Relay

Generic hosted callbacks use a per-execute callback binding. The worker connects to that binding and sends:

1. `callback_name`
2. callback payload
3. context with `engine_id`, `toolbox_id`, `tool_name`, `tool_call_id`, `tool_arguments`, optional `callback_signature`, and `callable_surface`

The caller-side hosted execution harness processes callbacks concurrently. A blocked callback processor blocks only that callback response, not the entire worker callback path.

## Host/Public APIs

Host service and channel APIs include:

1. `toolbox-describe`
2. `toolbox-gate`
3. `toolbox-execute`
4. `toolbox-cancel`
5. `toolbox-register-auto`
6. `toolbox-unregister-auto`
7. `toolbox-register-manual`
8. `toolbox-unregister-manual`
9. `toolbox-register-intrinsics`
10. `toolbox-unregister-intrinsics`
11. `toolbox-references`
12. `toolbox-consistency`
13. `toolbox-review-snapshot`
14. `toolbox-repair`
15. `toolbox-reconcile`
16. `toolbox-gc`

Environment APIs include description list/upsert/clone, requirement resolution, apply, realize, install plan/lock/verify/execute flows, and receipt verification.

App-facing helpers include:

1. `create_hosted_control_channel(...)`
2. `attach_existing_hosted_toolbox(...)`
3. `create_hosted_toolbox_ref(...)`
4. `register_hosted_tool_callable(...)`
5. `create_hosted_toolbox_executor(...)`
6. `execute_tool_round_on_cursor(...)`

`HostedToolBoxRef.mutate()` returns a pending builder so multiple registrations can be resolved with one backend sandbox rebuild.

## Gate And Scope Semantics

Hosted toolbox execution follows the native `Toolbox` semantics first:

1. registered vs missing
2. advertised vs hidden
3. allowed vs gated or blocked by scope
4. static guide tools are separate tools and can be gated independently

The hosted layer then adds backend outcomes such as missing executor, sandbox-policy denial, and cancellation.

Current dynamic constraints live in `ToolsScope` / `ToolsView`, not in sandbox policy. The implemented shared subset supports:

1. `argument_policy.implied_args`
2. `argument_policy.locked_args`
3. `path_under_implied_root`
4. `url_under_implied_prefix`
5. kwargs injection of `tool_constraints`, `tools_view`, and `tool_constraints_view`

Sandbox policy remains the hard outer boundary. Scope constraints narrow a call within that boundary.

## Environment And Rollout

Toolbox environments are one consumer of the shared host-managed runtime environment model.

Existing toolbox executor environments remain under:

```text
<hosting_root>/toolbox_venvs/<venv_key>
```

New non-toolbox runtime environments use:

```text
<hosting_root>/runtime_envs/<venv_key>
```

Environment identity is based on runtime hash, consumer kind, intrinsic dependency profile where applicable, required imports, environment description identity, and optional dependency-lock identity. Metadata includes `environment_root_kind` and `environment_consumer_kind`.

Workers use a bootstrap/preverified Python only while a dependency-bearing environment has not been verified. No-package/no-op environments can activate the realized venv immediately. Dependency-bearing environments switch to the realized venv after install execution and receipt verification are both recorded as ok.

Rollout is intentionally simple:

1. stage bundle
2. realize/select environment
3. spawn replacement executor
4. wait for readiness
5. verify registered tool inventory
6. persist new profile registration
7. retire old registrations
8. rollback on failed warmup

Repair/reconcile rebuild from persisted logical toolbox state and serialize per targeted `toolbox_id`.

## Cancellation

`toolbox.cancel` is coarse executor-level cancellation. It kills the targeted sandbox worker and can respawn replacement workers from persisted toolbox state. Harness boundaries normalize worker loss into canceled tool-call errors, and wrappers can consult `should_resubmit_canceled_tool_call(...)` plus persisted `non_restartable` metadata before resubmitting.

## Current Limits

1. Chat integration supports parallel tool calls for a single response. Batch tool rounds currently invoke the executor serially; the underlying toolbox and hosted/non-chat harness support parallel calls, but cross-prompt batch tool parallelism is not enabled by the chat batch path.
2. One worker process serves one staged profile; there is no sandbox worker pool or replica set.
3. Rollout has no percentage cutover or soak window.
4. Static sandbox policy is not mutated for per-request approvals; use scope constraints.
5. Environment locking is usable but not a fully mature package-management platform.
