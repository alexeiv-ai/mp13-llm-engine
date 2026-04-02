# Toolbox Sandbox Architecture

Date: 2026-04-01
Scope: current repository design for manifest-driven toolbox sandboxes, native toolbox fallback, hosting API integration, and toolbox-worker startup/callback contracts

## 1. Overview

The sandbox design for toolbox execution is intentionally split into two layers:

1. host-side management and authorization
2. executor-side loading and execution

The host remains the trust boundary.

The executor is a hosted worker process that loads a staged toolbox revision and executes user-provided tool code over the existing IPC transport.

Important terminology:

1. "worker" here means the generic hosting worker-process concept
2. it does not mean the trusted `mp13engine` model-serving worker
3. the toolbox executor should be built on the existing generic hosting spawn/control/IPC machinery

Current implementation entry points:

1. host-side bundle/harness helpers:
   - [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py)
2. sandbox executor worker:
   - [toolbox_executor_ipc.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_executor_ipc.py)
3. host-side authorization and RPC:
   - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py)
   - [engine_host_channel.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_channel.py)
   - [engine_host_cli.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_cli.py)
   - [engine_host_daemon.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_daemon.py)

## 2. Main Components

### 2.1 Toolbox Revision

A sandboxed toolbox does not discover callables from ambient Python scope.

Instead, the host stages an explicit toolbox revision made of:

1. toolbox module files
2. tool definitions
3. callable entrypoints
4. optional resources
5. derived manifest metadata

Core types:

1. `ToolboxBundleFile`
2. `ToolboxBundleTool`
3. `ToolboxBundleSpec`
4. `StagedToolboxBundle`
5. `ToolboxBundleStager`

The staged manifest contains:

1. `executor_kind=toolbox_executor_v1`
2. `bundle_id`
3. `bundle_revision`
4. `manifest_hash`
5. optional `dependency_lock_hash`
6. tool inventory
7. file content hashes

For toolbox scope, "bundle" should be read narrowly:

1. one staged revision of a toolbox runtime payload
2. not a generic worker packaging abstraction
3. the unit of add/remove/reload for sandboxed tools

That revision must cover both kinds of toolbox content:

1. staged user-defined callables loaded from manifest-declared modules
2. auto-discovered user-defined callables loaded from staged modules by callable name
3. built-in intrinsic tools loaded from `INTRINSICS_REGISTRY` in [mp13_tools_builtin.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_tools_builtin.py)

So a toolbox revision is not just "files on disk".

It is the complete materialized toolbox state required by the executor:

1. user-tool manifest entries
2. auto-tool discovery entries
3. intrinsic-tool enablement state
4. guide-tool inclusion state
5. active/hidden tool lists
6. dependency and environment provenance

### 2.2 Toolbox Executor Worker

The worker process in [toolbox_executor_ipc.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_executor_ipc.py):

1. reads `MP13_TOOLBOX_WORKER_SPEC_PATH` when available
2. falls back to `MP13_TOOLBOX_MANIFEST_PATH` for compatibility
2. loads staged Python files from the bundle root
3. constructs a `Toolbox`
4. registers only tools declared in the manifest
5. serves RPC over existing IPC

The intended long-term startup model should move from ad hoc env variables toward one structured startup spec.

Recommended shape:

```python
@dataclass
class ToolboxWorkerStartupSpec:
    worker_id: str
    sandbox_id: str
    toolbox_revision: str

    manifest_path: str
    scratch_root: str
    venv_path: str | None = None

    ipc_family: str = "AF_PIPE"
    ipc_address: str = ""
    auth_token_env: str = "MP13_ENGINE_HOST_TOKEN"

    execution_contract: str = "hosting.toolbox.worker.v1"
    callback_contract: str = "hosting.toolbox.callbacks.v1"
    policy: dict[str, Any] = field(default_factory=dict)
```

Recommended transport of the startup spec:

1. host writes a JSON startup spec file under hosting-managed state
2. host sets one env var such as `MP13_TOOLBOX_WORKER_SPEC_PATH`
3. worker loads the spec from that path on startup

Reason:

1. keeps worker startup versionable
2. avoids expanding a long list of worker-specific env vars
3. preserves reuse of existing hosting auth and IPC setup

Current implementation status:

1. `ToolboxWorkerStartupSpec` now exists in [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py)
2. staged bundles can emit startup-spec files through `worker_env_with_startup_spec(...)`
3. the worker now resolves manifest path from `MP13_TOOLBOX_WORKER_SPEC_PATH`
4. the startup spec now carries hosting-state metadata used for host callback resolution
5. toolbox sandbox orchestration now uses the startup-spec path when spawning profile-specific executors
6. legacy `MP13_TOOLBOX_MANIFEST_PATH` and related env fallback are still accepted for compatibility

Current worker RPC methods:

1. `rpc.describe`
2. `toolbox.describe`
3. `toolbox.execute`
4. `host.call`

Recommended next worker RPC methods:

1. `toolbox.cancel`

Current worker-side defense in depth:

1. unknown tool names are rejected
2. only manifest entrypoints are loaded
3. no ambient `search_scope` relinking is used

### 2.3 Toolbox Execution Harness

`ToolboxExecutionHarness` is the wiring layer between a caller and either:

1. a native in-process `Toolbox`
2. one sandbox executor
3. a pool of sandbox executors

Supported execution modes:

1. `native`
2. `sandbox`

Supported parallelism today:

1. async parallel dispatch within one executor
2. round-robin dispatch across multiple sandbox executors
3. both at once if multiple calls are dispatched while the harness has a pool configured

What the harness should hide from normal callers:

1. staging/revision churn
2. executor restart/switchover
3. pool membership changes
4. future `.venv` selection details

What the harness should not own:

1. sandbox process lifecycle policy
2. bundle garbage collection
3. low-level executor health orchestration

That means the harness is a user-facing logical layer, but not the full lifecycle manager.

### 2.4 Hosting API Layer

The host registration model now includes toolbox-specific metadata:

1. `executor_kind`
2. `bundle`
3. `environment`
4. `tool_access`
5. `capabilities`

The host service exposes:

1. `toolbox_describe(...)`
2. `toolbox_execute(...)`
3. `toolbox_register_auto(...)`
4. `toolbox_unregister_auto(...)`

The control channel mirrors those methods, and the daemon/CLI expose:

1. `toolbox-describe`
2. `toolbox-execute`
3. `toolbox-register-auto`
4. `toolbox-unregister-auto`

## 3. Capability Summary

### 3.1 Supported Now

1. manifest-driven loading of sandboxed toolbox callables
2. host-side allowlist gating before tool dispatch
3. dedicated toolbox executor IPC worker
4. native toolbox mode without sandbox
5. harness-managed pool of sandbox executors
6. async parallel dispatch of tool calls
7. host registration metadata for bundle provenance

### 3.1A Important Clarification

The current dedicated executor path is an initial toolbox-worker slice built on the generic hosting worker mechanism.

That means:

1. hosting already provides the relevant generic spawn/control/IPC substrate
2. toolbox execution is the specialized worker role layered on top of it
3. the trusted engine worker remains a separate runtime role

### 3.2 Supported In Adjacent Sandbox Infrastructure

These capabilities already exist in the generic sandbox layer and are reusable by toolbox sandboxes:

1. Windows spawn hardening
2. Low Integrity worker launch path
3. brokered filesystem policy on the host
4. brokered HTTP allowlist enforcement on the host

See:

1. [policy.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/policy.py)
2. [launcher.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/launcher.py)
3. [broker_fs.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/broker_fs.py)
4. [broker_http.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/broker_http.py)

## 4. Execution Model

### 4.1 Native Mode

Native mode is for trusted in-process execution.

The harness calls `Toolbox.execute(...)` directly.

Use native mode when:

1. the toolbox is trusted
2. sandbox isolation is not required
3. you still want the same harness interface as sandbox mode

### 4.2 Sandbox Mode

Sandbox mode is for manifest-driven execution in a separate worker process.

The flow is:

1. stage bundle
2. spawn toolbox executor worker
3. register bundle/environment/tool-access metadata with hosting
4. use host RPC for `toolbox.describe` and `toolbox.execute`
5. optionally place multiple executors behind one harness

The user-facing intent is still simple:

1. add tool
2. remove tool
3. execute tool
4. list tools

But under sandbox mode, those simple operations map to host-managed revision and worker lifecycle changes.

Example: sandboxing builtin intrinsic tools

The toolbox already supports builtin/intrinsic functions through [mp13_toolbox.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_toolbox.py), so sandbox mode should treat them as revision state, not as a separate side path.

Example desired logical operation:

```python
toolbox.add_tool_callable(
    ["scriptable_calculator", "scriptable_calculator_guide"],
    is_intrinsic=True,
    include_guides=True,
    activate=True,
)
```

Sandbox interpretation:

1. host records the requested intrinsic names in the toolbox revision
2. host resolves the required environment profile for those intrinsics
3. replacement toolbox executor starts from that revision
4. executor constructs `Toolbox(with_intrinsics=True, with_intrinsic_guides=True)`
5. executor restores revision state and exposes only the requested intrinsic entries
6. host routes `toolbox.execute("scriptable_calculator", ...)` to that executor after authorization checks

This keeps the user-facing toolbox API simple while still making intrinsic enablement auditable and restart-driven under sandbox management.

### 4.3 Parallel Execution

Parallel execution can come from either of these paths:

1. async dispatch to one executor
2. a pool of multiple executors
3. both

Current pool behavior in the harness is round-robin selection across `sandbox_engine_ids`.

## 5. Wiring A Toolbox Instance To Sandboxes

### 5.1 Native Wiring

If you already have a `Toolbox` instance and do not want sandboxing:

```python
from hosting.toolbox_harness import ToolboxExecutionHarness, ToolboxHarnessConfig

harness = ToolboxExecutionHarness(
    config=ToolboxHarnessConfig(mode="native"),
    native_toolbox=toolbox,
)

results = await harness.execute_calls(tool_calls, parallel=True)
```

This preserves the option of native toolbox operation mode without sandbox.

### 5.2 Sandbox Wiring

To wire toolbox execution to sandbox workers, you need:

1. staged bundle content
2. spawned toolbox executor registration(s)
3. a control channel
4. a harness configured for `mode="sandbox"`

Example:

```python
import sys

from hosting import EngineHostControlChannel, EngineHostService
from hosting.toolbox_harness import (
    ToolboxBundleFile,
    ToolboxBundleSpec,
    ToolboxBundleStager,
    ToolboxBundleTool,
    ToolboxExecutionHarness,
    ToolboxHarnessConfig,
)

service = EngineHostService()
stager = ToolboxBundleStager(service.hosting_root)

bundle = stager.stage_bundle(
    ToolboxBundleSpec(
        bundle_id="user-tools",
        files=[
            ToolboxBundleFile(
                relative_path="user_tools.py",
                content=(
                    "def hello(name='world'):\n"
                    "    return {'greeting': f'hi {name}'}\n"
                ),
            ),
        ],
        tools=[
            ToolboxBundleTool(
                definition={
                    "type": "function",
                    "function": {
                        "name": "hello_tool",
                        "description": "Return a greeting.",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": [],
                        },
                    },
                },
                entrypoint="user_tools:hello",
            ),
        ],
    )
)

registration = service.spawn(
    engine_id="toolbox-user-tools-1",
    command=bundle.worker_command(python_executable=sys.executable),
    env=bundle.worker_env(),
    worker_profile_class="generic",
    executor_kind="toolbox_executor_v1",
    bundle=bundle.registration_bundle(),
    environment=bundle.registration_environment(),
    tool_access=bundle.registration_tool_access(),
    capabilities={
        "brokered_filesystem": False,
        "brokered_http": False,
        "dynamic_reload": False,
    },
)

channel = EngineHostControlChannel()
harness = ToolboxExecutionHarness(
    config=ToolboxHarnessConfig(
        mode="sandbox",
        sandbox_engine_ids=[registration["engine_id"]],
    ),
    control_channel=channel,
)
```

The current repo now supports both startup styles, but the preferred direction is the startup spec.

The recommended future equivalent should look more like:

```python
startup = ToolboxWorkerStartupSpec(
    worker_id="toolbox-user-tools-1",
    sandbox_id="sandbox-user-tools",
    toolbox_revision=bundle.manifest["bundle_revision"],
    manifest_path=str(bundle.manifest_path),
    scratch_root=str((service.hosting_root / "toolbox_scratch" / "toolbox-user-tools-1").resolve()),
    venv_path=None,
    ipc_family="AF_PIPE",
    ipc_address="<allocated by hosting>",
)
```

Hosting should serialize that spec and inject only a pointer to it into the worker startup environment.

Current helper example:

```python
env = bundle.worker_env_with_startup_spec(
    worker_id="toolbox-user-tools-1",
    sandbox_id="sandbox-user-tools",
    scratch_root=service.hosting_root / "toolbox_scratch" / "toolbox-user-tools-1",
    engines_state_file=service.engines_state_file,
    control_state_file=service.control_state_file,
)
```

## 6. Injecting And Removing Functions As Toolbox Operations

### 6.1 Important Design Rule

For sandboxed execution, injection and removal are host-managed bundle operations.

They are not ambient `Toolbox.add_tool_callable(...)` mutations inside the worker process.

They are also intentionally symmetric:

1. add function => create new toolbox revision => roll traffic to new executor revision
2. remove function => create new toolbox revision => roll traffic to new executor revision

### 6.2 Inject / Add A Function

To inject a new function into a sandboxed toolbox:

1. add the Python source file or module content to a new `ToolboxBundleSpec`
2. add a new `ToolboxBundleTool` entry with:
   - tool definition
   - `module:function` entrypoint
3. stage the new bundle revision
4. spawn a new executor or refresh an existing pool member against the new revision
5. update host registration metadata
6. switch traffic to the new executor(s)

Example delta:

```python
bundle = stager.stage_bundle(
    ToolboxBundleSpec(
        bundle_id="user-tools",
        files=[
            ToolboxBundleFile(
                relative_path="user_tools.py",
                content=(
                    "def hello(name='world'):\n"
                    "    return {'greeting': f'hi {name}'}\n"
                    "\n"
                    "def goodbye(name='world'):\n"
                    "    return {'farewell': f'bye {name}'}\n"
                ),
            ),
        ],
        tools=[
            ToolboxBundleTool(definition=hello_def, entrypoint="user_tools:hello"),
            ToolboxBundleTool(definition=goodbye_def, entrypoint="user_tools:goodbye"),
        ],
    )
)
```

That produces a new `bundle_revision` and `manifest_hash`.

Recommended logical API expectation:

```python
sandbox_toolbox.add_tool(...)
```

What should happen under the hood:

1. mutate logical toolbox definition
2. generate new staged toolbox revision
3. prepare runtime metadata and environment selection
4. start replacement toolbox worker
5. switch dispatch to the new revision

The user should not manually perform those lifecycle steps in normal use.

Automatic callable discovery from staged modules is also supported now.

Example:

```python
ToolboxBundleSpec(
    bundle_id="user-tools",
    files=[
        ToolboxBundleFile(
            relative_path="user_tools.py",
            content=(
                "def hello_auto(name: str = 'world'):\n"
                "    \"\"\"Return a greeting.\n\n"
                "    Args:\n"
                "        name (str): Name to greet.\n"
                "    \"\"\"\n"
                "    return {'greeting': f'hi {name}'}\n"
            ),
        ),
    ],
    auto_tools=[
        ToolboxBundleAutoTool(
            module_name="user_tools",
            callable_name="hello_auto",
        ),
    ],
)
```

Worker behavior:

1. worker imports the staged module
2. worker resolves the named callable from that module scope
3. worker calls `Toolbox.add_tool_callable(callable_name, search_scope=module.__dict__, ...)`
4. `Toolbox` derives the tool definition from signature and docstring
5. resulting tool is exposed through normal `toolbox.describe` / `toolbox.execute`

Builtin and guide activation should be described with the same workflow shape.

Example:

```python
sandbox_toolbox.add_tool("symbolic_algebra", is_intrinsic=True, include_guides=True)
```

What should happen under the hood:

1. mutate logical toolbox definition to include the intrinsic and optional guide
2. generate new staged toolbox revision metadata even if no new Python file is added
3. re-resolve environment selection because builtin dependencies may have changed
4. start replacement toolbox worker
5. switch dispatch to the new revision

### 6.3 Remove A Function

To remove a function from a sandboxed toolbox:

1. build a new bundle spec without that tool
2. stage the new revision
3. register or refresh executor(s) with the reduced tool inventory
4. stop routing calls to the old executor(s)
5. remove stale registrations
6. garbage-collect stale bundle content when safe

Current implementation status:

1. staging the new revision is implemented
2. host-side tool allowlist enforcement is implemented
3. automatic stale bundle GC is not implemented yet

Recommended logical API expectation:

```python
sandbox_toolbox.remove_tool("goodbye_tool")
```

What should happen under the hood:

1. remove tool from logical toolbox definition
2. build a new staged revision
3. start replacement worker set
4. switch dispatch
5. retire old revision later

### 6.4 Trusted Native Toolbox Mutation

For native mode only, it is still valid to use:

1. `Toolbox.add_tool_callable(...)`
2. `Toolbox.add_tool_external(...)`
3. `Toolbox.delete_tool(...)`

That is the trusted local path, not the sandbox authority path.

## 7. Generic Host Callback RPC

### 7.1 Core Design

The missing executor-side brokering problem should be solved as a generic host callback RPC design.

Recommended worker RPC method:

1. `host.call`

Recommended request shape:

```json
{
  "kind": "rpc_call",
  "method": "host.call",
  "params": {
    "method": "fs.read_text",
    "arguments": {
      "root_id": "tool_data",
      "relative_path": "config.json"
    }
  }
}
```

Recommended response shape:

```json
{
  "status": "ok",
  "result": {
    "text": "{...}"
  }
}
```

This keeps the transport generic while still allowing host policy to dispatch explicit broker methods.

### 7.2 Toolbox-Side Convenience Layer

Toolbox execution code should not need to manually construct low-level callback payloads.

Instead, the execution context may expose:

```python
context.host.call("fs.read_text", {...})
context.host.call("http.fetch", {...})
```

And optional convenience wrappers:

```python
context.fs.read_text(root_id="tool_data", relative_path="config.json")
context.http.fetch(url="https://example.com/api/test", method="GET")
```

That is the right place to "hide" generic callback transport from tool code.

Current implementation status:

1. `host.call` now exists on the dedicated toolbox executor worker
2. execution context wrappers now expose `context.host`, `context.fs`, and `context.http`
3. brokered filesystem callback use is covered by end-to-end tests
4. startup-spec path now carries the hosting-state metadata needed for callback routing
5. compatibility env wiring still exists but is no longer the preferred path

### 7.3 Why This Does Not Eliminate Lifecycle Separation

Even with those convenience handlers, the following still belong outside `Toolbox` and `ToolsAccess`:

1. revision creation
2. staging paths
3. worker spawn/restart
4. registration metadata
5. garbage collection
6. health and switchover

So generic callback wrappers help simplify tool code, but they do not replace host/runtime lifecycle management.

## 8. Permission Wiring

### 8.1 Host-Side Permission Gate

The host is the primary permission gate.

Before dispatching `toolbox.execute`, the host checks:

1. executor registration exists
2. transport is IPC
3. tool name is present in registration `tool_access.allowed_tool_names` when an allowlist is present

Current host methods:

1. `EngineHostService.toolbox_execute(...)`
2. `EngineHostControlChannel.toolbox_execute(...)`

Persisted-policy direction:

1. the effective permission policy should be attached to the persisted sandbox instance itself
2. that sandbox instance should also carry its environment-description linkage
3. toolbox refs should depend on one or more sandbox instances
4. rehydrating a toolbox ref should therefore rehydrate the dependent sandbox instances and their persisted permission policy

Example allowlist metadata:

```python
tool_access = {
    "allowed_tool_names": ["hello_tool", "goodbye_tool"],
    "advertised_tool_names": ["hello_tool"],
}
```

### 8.2 Sandbox-Side Defense In Depth

The worker also enforces:

1. only manifest tools are loaded
2. only manifest tool names can execute

If the host mistakenly dispatches a non-staged tool, the worker still rejects it.

### 8.3 Filesystem And Network Permissions

Today, toolbox executor registration can carry sandbox policy using the existing worker sandbox schema.

That allows host policy to express:

1. filesystem roots
2. brokered filesystem enablement
3. brokered HTTP enablement
4. network mode
5. Windows Low IL launch settings

Example:

```python
sandbox_policy = {
    "sandbox": {
        "enabled": True,
        "platform_policy": {
            "windows": {
                "restricted_token": True,
                "integrity_level": "low",
                "job_object": True,
            }
        },
        "filesystem": {
            "rules": [
                {
                    "root_id": "scratch",
                    "path": "C:\\sandbox\\scratch",
                    "access": ["read", "write"],
                }
            ]
        },
        "network": {
            "mode": "brokered_only",
            "allow_hosts": ["example.com"],
            "allow_url_prefixes": ["https://example.com/api/"],
        },
        "brokered_io": {
            "filesystem": True,
            "http": True,
            "subprocess": False,
        },
    }
}
```

Important current limitation:

1. toolbox executor bundle loading, tool execution, and initial host callbacks are implemented
2. host broker enforcement exists and is now callable from toolbox execution context
3. callback contract is still incomplete for cancellation, streaming callback patterns, and structured startup-spec integration

So permission metadata can already be registered, and initial executor-side callback usage now works, but the contract is not yet complete.

## 9. Management Model

### 9.1 Registration Metadata

Recommended host registration fields for toolbox executors:

1. `executor_kind="toolbox_executor_v1"`
2. `bundle`
   - `bundle_id`
   - `bundle_revision`
   - `manifest_hash`
   - `bundle_root`
   - `manifest_path`
3. `environment`
   - `venv_key`
   - `venv_lock_hash`
   - `venv_mutable`
4. `tool_access`
   - `allowed_tool_names`
   - `advertised_tool_names`
5. `capabilities`
   - `brokered_filesystem`
   - `brokered_http`
   - `dynamic_reload`

### 9.2 Pool Management

A pool is currently managed by the harness configuration, not by a dedicated daemon-side pool manager.

That means:

1. you can register multiple executor workers manually
2. you can pass their engine ids to `ToolboxHarnessConfig.sandbox_engine_ids`
3. the harness will distribute requests round-robin

Example:

```python
harness = ToolboxExecutionHarness(
    config=ToolboxHarnessConfig(
        mode="sandbox",
        sandbox_engine_ids=[
            "toolbox-user-tools-1",
            "toolbox-user-tools-2",
            "toolbox-user-tools-3",
        ],
    ),
    control_channel=channel,
)
```

### 9.3 Revision Rollout

A practical rollout sequence is:

1. stage a new bundle revision
2. spawn a new executor pool for that revision
3. wait for the new executor registrations to answer `toolbox.describe`
4. switch the harness to the new pool ids
5. drain old traffic
6. remove old registrations
7. garbage-collect stale bundle content later

Current implementation status:

1. host-side toolbox register/unregister now waits for newly spawned profile-specific executors to become ready before retiring replaced registrations
2. if readiness fails, the new registrations are rolled back and old registrations remain in place
3. successful profile rollouts now persist basic rollout metadata such as:
   - `engine_id`
   - `ready_at`
   - `warmup_ms`
4. persisted profile state now also keeps a bounded `rollout_history` trail for successful cutovers
5. register/unregister results now also return that rollout metadata for newly readied executors
6. rollout policy is still single-step and does not yet include replica warmup, staged cutover percentages, or longer-lived health history

### 9.4 One Logical Toolbox Across Multiple Sandbox Specs

This design point is not fully implemented yet and must remain explicit.

Current implementation shape:

1. one toolbox executor registration corresponds to one staged toolbox revision
2. one harness instance can round-robin across multiple equivalent executors
3. host-side routing by logical `toolbox_id` now works when tools are grouped into separate profile-specific revisions
4. automatic sandbox-profile assignment at registration time is now implemented for auto-callable requests

Required future shape:

1. one logical toolbox may contain functions assigned to different sandbox specs
2. host must group tools by sandbox policy and dependency profile
3. each group should stage to its own sandbox revision and executor pool
4. a routing layer should dispatch each tool call to the correct sandbox pool
5. higher-level registration should assign callables to sandbox profiles automatically

Ideal user-facing direction:

1. user supplies a callable plus a permissions/dependency spec
2. hosting decides whether an existing sandbox profile matches
3. hosting either assigns the callable to an existing sandbox or stages a new one
4. logical toolbox APIs stay simple while host routing hides per-sandbox placement

Tool identity rule:

1. tool names are scoped to a live toolbox ref
2. they do not need to be globally unique across all sandboxes
3. the effective invocation identity should therefore be `toolbox_ref + tool_name`
4. a separate implementation registry may still exist behind the scenes, but runtime dispatch should stay toolbox-ref-scoped because tools are always invoked through a live toolbox ref

Current first-slice implementation:

1. [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py) now provides:
   - `SandboxProfileSpec`
   - `ToolboxAutoAssignmentRequest`
   - `ToolboxSandboxOrchestrator`
2. `SandboxProfileSpec` can derive a stable `profile_id` from:
   - required imports
   - sandbox policy
3. `ToolboxSandboxOrchestrator` can:
   - group auto-callable requests by derived profile
   - stage one toolbox revision per profile
   - spawn one toolbox executor per staged profile revision
4. host routing by logical `toolbox_id` can then dispatch calls to the correct profile-specific executor

What is still missing:

1. mutation/update flows that merge new tools into an already-running profile revision automatically
2. persistent host-side management of profile membership beyond the current orchestration helper
3. higher-level public APIs that let callers register callables without directly constructing orchestration requests

The next slice is now partially implemented:

1. [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now provides `toolbox_register_auto(...)`
2. host persists logical toolbox membership in:
   - [toolbox_sandboxes.json](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_architecture.md)
   - actual runtime location: `<hosting_root>/state/toolbox_sandboxes.json`
3. a registration update now works like this:
   - load persisted requests for `toolbox_id`
   - merge new auto-callable requests into that logical toolbox state
   - regroup requests by derived sandbox profile
   - stage one new revision per affected profile
   - spawn replacement executor registrations
   - remove replaced profile registrations from the live registry
   - persist the new profile membership and engine ids

What is still not done after this slice:

1. richer rollout policies such as health-checked cutover, staged warmup, or replica pools per profile
2. reference-counted garbage collection beyond the current retired-bundle cleanup path
3. a higher-level user-facing facade that can hide orchestration request construction entirely

Removal lifecycle is now also implemented at the host-service layer:

1. `EngineHostService.toolbox_unregister_auto(...)`
2. remove one or more persisted callable keys from a logical toolbox
3. regroup remaining requests by sandbox profile
4. rebuild only the affected profile-specific revisions
5. retire replaced registrations
6. delete retired bundle roots under `<hosting_root>/toolbox_bundles`
7. remove logical toolbox state entirely when the last tool is removed

That lifecycle is now also exposed through:

1. `EngineHostControlChannel.toolbox_register_auto(...)`
2. `EngineHostControlChannel.toolbox_unregister_auto(...)`
3. `toolbox-register-auto`
4. `toolbox-unregister-auto`

## 10. User-Facing Simplicity Vs Hidden Lifecycle

### 10.1 What Users Should See

Normal callers should be able to think in toolbox terms:

1. `add_tool(...)`
2. `remove_tool(...)`
3. `execute(...)`
4. `list_tools(...)`

Current implementation note:

1. [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py) now provides `SandboxedToolboxFacade`
2. the facade hides low-level `ToolboxAutoAssignmentRequest` shaping for common auto-callable registration/removal flows
3. callers can now use:
    - `register_auto_callable(...)`
    - `register_python_callable(...)`
    - `register_intrinsic_tools(...)`
    - `unregister_intrinsic_tools(...)`
    - `register_manual_tool(...)`
    - `unregister_manual_tool(...)`
    - `unregister_auto_callable(...)`
    - `describe(...)`
    - `execute(...)`
4. `register_python_callable(...)` can now stage a real module-backed Python callable by reading its source module automatically
5. builtin intrinsic tools can now be added and removed through the same facade against sandbox hosting
6. explicit manual tool definitions can now be registered against a module-backed Python implementation through the same facade
7. richer facade coverage beyond these core flows remains future work

Public API naming direction:

1. `SandboxedToolboxFacade` should be treated as an implementation-stage name
2. the intended public API should preserve the existing `Toolbox` / `ToolBoxRef` programming model
3. the preferred public hosting-side name is `HostedToolBoxRef`
4. that better reflects how users already operate on toolbox refs, especially for dynamic active-tool management

### 10.2 What Hosting Should Hide

Hosting should hide:

1. revision generation
2. worker restart and switchover
3. bundle path bookkeeping
4. environment bookkeeping
5. pool churn

### 10.3 Why `ToolBoxRef` / `ToolsAccess` Alone Are Not Enough

`ToolBoxRef` and `ToolsAccess` are the right logical layer for:

1. tool visibility
2. scoped permission views
3. execution-time access shaping

They are not sufficient for:

1. cross-process startup
2. sandbox registration provenance
3. host authorization boundary
4. worker lifecycle and restart
5. bundle/reference garbage collection

So the correct design is:

1. keep toolbox logical APIs simple
2. let a sandbox-aware facade or manager hide lifecycle complexity
3. do not overload `ToolBoxRef` / `ToolsAccess` with process-orchestration responsibilities

Refined interpretation:

1. `ToolBoxRef` remains the primary user mental model
2. hosting should expose a public hosted-ref type that behaves like normal toolbox-ref programming
3. sandbox lifecycle, env selection, and routing stay behind that hosted ref
4. compatibility preservation is not required for the current helper name if the public API is renamed

## 11. Integrating With The Hosting API

### 11.0 Simple Facade Integration

If you want a simpler toolbox-facing API on top of the service or control channel:

```python
from hosting import EngineHostService
from hosting.toolbox_harness import SandboxedToolboxFacade

service = EngineHostService()
toolbox = SandboxedToolboxFacade(
    toolbox_id="user-tools",
    host=service,
)

toolbox.register_auto_callable(
    relative_path="user_tools.py",
    content=(
        "def hello_auto(name: str = 'world'):\n"
        "    return {'greeting': f'hi {name}'}\n"
    ),
    module_name="user_tools",
    callable_name="hello_auto",
    required_imports=["requests"],
    sandbox_policy={"sandbox": {"enabled": True}},
)

desc = toolbox.describe()
result = toolbox.execute(tool_name="hello_auto", arguments={"name": "Sam"})
```

If you already have a module-backed Python function object, the facade can register it directly:

```python
toolbox.register_python_callable(
    hello_auto,
    required_imports=["requests"],
    sandbox_policy={"sandbox": {"enabled": True}},
)
```

Intrinsic tools can now use the same facade:

```python
toolbox.register_intrinsic_tools(
    ["symbolic_algebra"],
    include_guides=True,
    sandbox_policy={"sandbox": {"enabled": True}},
)
```

Explicit tool definitions can also be staged through the same facade:

```python
toolbox.register_manual_tool(
    tool_definition={
        "type": "function",
        "function": {
            "name": "hello_manual",
            "description": "Return a greeting.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": [],
            },
        },
    },
    implementation=hello_auto,
    required_imports=["requests"],
    sandbox_policy={"sandbox": {"enabled": True}},
)
```

The same facade can wrap `EngineHostControlChannel` because it only depends on the high-level toolbox registration/describe/execute methods.

Public API direction:

1. the repo currently uses `SandboxedToolboxFacade` in examples because that is what exists today
2. the intended public API direction is a hosted toolbox-ref type, likely `HostedToolBoxRef`
3. when renamed, the goal should be to preserve the toolbox-ref programming model rather than expose lifecycle machinery directly

### 11.0A Registration Validation Strength

Not all registration modes can provide the same validation guarantees before sandbox warmup.

Recommended distinction:

1. explicit manual definition + live Python implementation object
   - strong pre-staging validation
   - host can verify callable object, source file, and optional schema/signature consistency
2. live Python callable registration through `register_python_callable(...)`
   - strong pre-staging validation
   - host can verify callable object and source module before staging
3. name-based auto-discovery from staged module/callable names only
   - structural pre-staging validation only
   - full resolution happens during sandbox warmup

Operational implication:

1. do not import arbitrary candidate modules in the host merely to prove name-based resolution
2. let sandbox warmup be the authoritative resolution step for that mode
3. keep readiness-gated rollback as the safety mechanism when sandbox resolution fails

### 11.1 Direct Service Integration

If you are local to the host process, use `EngineHostService`:

```python
service = EngineHostService()

desc = service.toolbox_describe(engine_id="toolbox-user-tools-1")
result = service.toolbox_execute(
    engine_id="toolbox-user-tools-1",
    tool_call={
        "name": "hello_tool",
        "arguments": {"name": "Sam"},
    },
)
```

### 11.2 Daemon / Control Channel Integration

If you are talking to the host through the daemon, use `EngineHostControlChannel`:

```python
channel = EngineHostControlChannel(control_settings=control_settings)

desc = channel.toolbox_describe(engine_id="toolbox-user-tools-1")
result = channel.toolbox_execute(
    engine_id="toolbox-user-tools-1",
    tool_call={
        "name": "hello_tool",
        "arguments": {"name": "Sam"},
    },
)
```

### 11.3 CLI Integration

Examples:

```powershell
@'{"engine_id":"toolbox-user-tools-1"}'@ |
python -m hosting.engine_host_cli --payload-stdin toolbox-describe
```

```powershell
@'{"engine_id":"toolbox-user-tools-1","tool_call":{"name":"hello_tool","arguments":{"name":"Sam"}}}'@ |
python -m hosting.engine_host_cli --payload-stdin toolbox-execute
```

## 12. Current Limitations

### 12.1 Bundle Lifecycle

Not yet complete:

1. automatic executor reload on bundle change
2. reference-tracked bundle garbage collection
3. automatic removal of stale bundle revisions

### 12.2 `.venv` Management

The environment metadata shape exists, but the final operational model should now be read with these design decisions:

1. there is one host-managed base toolbox environment
2. that base environment assumes none of the optional supported permissions are granted
3. that base environment carries the standard package set needed for the toolbox runtime itself
4. additional named environment descriptions may extend that base environment
5. toolbox functions may be linked explicitly to one of those named environment descriptions

This is intentionally simpler than a heavy fully-general package-lock distribution model.

Current status:

1. `venv_key` is now derived deterministically from:
   - toolbox runtime hash
   - intrinsic dependency tier
   - required imports
   - optional dependency lock hash
2. host now materializes environment metadata roots under `<hosting_root>/toolbox_venvs/<venv_key>`
3. host now creates a real Python virtual environment root there using stdlib `venv`
4. profile-specific toolbox workers are now spawned through that environment's Python executable
5. compatible toolbox revisions can now reuse the same environment root by `venv_key`
6. unreferenced toolbox environment roots can now be garbage-collected when logical toolbox state no longer references them
7. executor registrations now carry:
   - `venv_key`
   - `venv_path`
   - `python_executable`
   - `venv_lock_hash`
   - `intrinsics_profile_id`
   - `required_imports`
8. locked dependency installation and fully reproducible environment build policy are still pending
9. no live dependency installation should be assumed inside the sandbox executor path

Recommended design direction from current decisions:

1. keep one small distributable base toolbox environment in the repository/runtime story
2. let users define a few named environment descriptions rather than forcing fully automatic environment synthesis for every tool
3. let toolbox functions be linked manually to one of those named environments
4. keep environment growth as a host lifecycle operation, not a normal tool-execution operation
5. prefer cloning or extending an existing named environment over inventing a new one for every change

That means the important identity split should be:

1. toolbox revision identity
2. environment description identity
3. actual realized environment instance/path

Those should remain related, but not collapsed into one hash.

Current builtin dependency signal from [mp13_tools_builtin.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_tools_builtin.py):

1. `scriptable_calculator` uses `numpy` and can optionally use `numexpr`
2. `symbolic_algebra` uses `sympy`
3. builtin guides are lighter logically, but their activation still belongs to the same revision model

Recommended interpretation:

1. intrinsic activation can affect `.venv` provenance even when no user file changes
2. builtin guide activation should be tracked in revision state
3. the current implementation already derives `venv_key` from runtime base plus intrinsic profile plus required imports and optional dependency lock
4. the current implementation now builds and reuses host-managed Python environment roots, but still relies on the base interpreter package set
5. the remaining step is adding named-environment description management, package resolution/update APIs, and stronger immutable environment provenance

### 12.2A Named Environment Descriptions

Before deeper operational implementation, the architecture should assume:

1. host can persist several named toolbox environment descriptions
2. each description may declare:
   - base environment name
   - extra packages
   - install policy metadata
   - whether online resolution/install is allowed
3. a toolbox function may be linked explicitly to one environment description
4. multiple functions may share the same environment description

This keeps the model understandable:

1. user chooses or creates an environment description
2. user links a function to it
3. hosting resolves or reuses the realized environment behind that description

### 12.2B Package Resolution And Update API

The package-resolution path should be explicit host API, not an implicit tool-execution side effect.

Recommended host-owned APIs:

1. resolve currently linked functions against a named environment description
   - return missing packages or compatibility gaps
2. resolve an arbitrary set of functions to the extra packages they would require beyond a given environment
3. update an existing named environment description
4. clone an existing environment description into a new one, then apply package changes there

The important policy rule is:

1. package resolution and installation belong to hosting management code
2. they are not normal tool-function decisions
3. a tool function should not silently mutate its environment during normal execution

If automatic environment update is ever allowed, it should still be:

1. host-mediated
2. policy-gated
3. observable in rollout/status history
4. separate from normal sandboxed tool execution

### 12.2C Online Install Policy

Online package resolution/install should be a hosting configuration choice, not a tool-level permission.

That means:

1. a host deployment may forbid online install entirely
2. another deployment may allow host-managed online install for environment maintenance
3. this choice belongs to hosting config and environment-management APIs
4. it should not be exposed as ordinary sandboxed tool network access

So even if online install is enabled:

1. the host performs it
2. the environment is updated or cloned as a managed lifecycle operation
3. sandboxed tool execution still runs against a prepared environment, not an actively mutating one

### 12.3 Brokered Callback Contract

The plan calls for generic executor callbacks such as `host.call`, with convenience wrappers for:

1. `fs.list`
2. `fs.read_text`
3. `fs.write_text`
4. `fs.mkdir`
5. `fs.stat`
6. `http.fetch`

Host enforcement for those brokered paths already exists in the generic sandbox layer.

Current status:

1. the dedicated toolbox executor worker now exposes `host.call`
2. tool code can use `context.host.call(...)` directly
3. tool code can use `context.fs.*` and `context.http.fetch(...)` convenience wrappers
4. richer callback semantics such as streaming callback responses are not implemented yet

### 12.4 Cancellation

`toolbox.cancel` is not implemented yet.

### 12.5 Windows Boundary Limits

These still apply:

1. Low IL is primarily a write boundary, not strong same-account read isolation
2. direct-network restrictions on Windows are still partial or unsupported beyond brokered paths
3. direct hostname / URL allowlisting for arbitrary worker traffic is not a trustworthy current claim

### 12.6 Multi-Sandbox Toolbox Routing

Not yet implemented:

1. richer replica and health-managed routing policies within each sandbox profile pool
2. automatic profile assignment for non-auto registration flows such as future explicit/manual tool-definition APIs

## 13. Recommended Usage Pattern Today

For current repo state, the recommended pattern is:

1. use native mode for trusted local callables
2. use `toolbox_register_auto(...)` when registering isolated auto-discovered callables from staged modules
3. register toolbox executors with explicit `tool_access`
4. treat the host as the real policy gate
5. use a harness-managed pool when you need parallel sandbox throughput
6. avoid claiming immutable `.venv` isolation or strong direct-network control until those parts are completed

## 14. Related Docs

1. [hosting_access_plan.md](/o:/repos/mp13-llm-engine/src/hosting/hosting_access_plan.md)
2. [hosting_status.md](/o:/repos/mp13-llm-engine/src/hosting/hosting_status.md)
3. [sandbox_test_status.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_test_status.md)
