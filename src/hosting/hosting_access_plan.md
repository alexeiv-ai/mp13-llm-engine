# Hosting Toolbox Sandbox Spec And Plan

Date: 2026-04-01
Scope: Windows-first sandboxing for toolbox-compatible executors, not trusted engine instances.

This document is both:

1. a normative specification for the first sandbox consumer and its enforcement model
2. an implementation plan with explicit limitations and rollout gates

## 1. Scope Correction

This plan supersedes the earlier generic-worker framing.

Normative scope:

1. Trusted engine instances such as [engine_worker_ipc.py](/o:/repos/mp13-llm-engine/src/hosting/engine_worker_ipc.py) are not the first sandbox target.
2. The first sandbox consumer is a toolbox-compatible executor process that runs user-provided tool code.
3. Sandbox lifecycle includes staging and removing Toolbox objects plus supporting Python modules as part of sandbox management.
4. Sandbox execution may communicate back to the host over existing IPC for brokered operations and control-plane callbacks.

Out of scope for the first cut:

1. Sandboxing the main engine instance.
2. Claiming strong direct-network allowlisting at the OS boundary on Windows.
3. In-sandbox package installation directly from the public network at execution time.
4. Arbitrary same-account read isolation on Windows.

## 2. Problem Statement

The relevant dynamic behavior lives in [mp13_toolbox.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_toolbox.py):

1. tools can be added through `add_tool_external(...)` and `add_tool_callable(...)`
2. tool links can be repaired dynamically through `resolve_tool_link(...)`
3. user-defined callable implementations are tracked in `user_tool_callables`
4. tools can be deleted through `delete_tool(...)`
5. execution permission is checked at call time in `execute(...)`

That is useful for trusted in-process operation, but not sufficient as the sandbox authority boundary.

Normative conclusion:

1. `Toolbox` remains the logical tool registry and execution contract.
2. Host sandbox management must become the authority for which user-defined tools are staged into a sandbox executor.
3. Tool permission checks must not rely only on `Toolbox.execute(...)` inside sandboxed user code.

Intrinsic tools are part of the same planning surface:

1. [mp13_toolbox.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_toolbox.py) already supports intrinsic discovery and activation through:
   - `available_intrinsics(...)`
   - `add_tool_callable(..., is_intrinsic=True, include_guides=...)`
   - `activate_tool(...)`
   - `deactivate_tool(...)`
2. [mp13_tools_builtin.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_tools_builtin.py) defines built-in callable implementations and optional guide tools through `INTRINSICS_REGISTRY`.
3. Sandbox planning must therefore cover both:
   - user-staged Python callables
   - auto-discovered callables loaded from staged modules by name
   - builtin intrinsic tools and optional intrinsic guide tools

## 3. Target Architecture

The intended architecture is:

1. host daemon
2. trusted engine worker
3. one or more sandboxed toolbox executor workers

The trusted engine worker may request tool execution, but the actual untrusted tool code runs in the sandbox executor.

### 3.1 Executor Contract

The sandbox executor should expose a toolbox execution API over existing IPC:

1. `toolbox.describe`
2. `toolbox.execute`
3. `toolbox.cancel` if cancellation is supported
4. generic host callback RPC:
   - `host.call`
5. convenience callback methods wrapped on top of that path:
   - `fs.list`
   - `fs.read_text`
   - `fs.write_text`
   - `fs.mkdir`
   - `fs.stat`
   - `http.fetch`

No separate sandbox-facing HTTP listener should be introduced.

Recommended startup contract:

1. hosting constructs one structured toolbox-worker startup spec
2. hosting serializes it to a JSON file under hosting-managed state
3. hosting injects only a pointer such as `MP13_TOOLBOX_WORKER_SPEC_PATH`
4. worker loads manifest path, scratch root, optional `.venv`, and IPC metadata from that spec

This should be built on the existing generic hosting worker spawn/control/IPC mechanism, not on the trusted engine worker path.

### 3.2 Toolbox Revision Model

User-defined tools should be staged as a host-managed toolbox revision:

1. manifest describing tool names, callable entrypoints, dependency set, and content hash
2. Python source/modules required by those callables
3. optional static assets
4. executor-local writable scratch root kept separate from staged revision content

The host, not the user tool code, owns bundle staging and removal.

For toolbox scope, the revision model must also be able to represent intrinsic-tool activation state, not just staged Python files.

Recommended revision payload:

1. manifest-driven user tool entries:
   - tool name
   - declared entrypoint
   - source file/module mapping
2. auto-tool discovery entries:
   - module name
   - callable name
   - optional guide content metadata
3. intrinsic-tool activation entries:
   - intrinsic names requested
   - whether intrinsic guides are included
   - intrinsic override metadata when applicable
4. toolbox-wide state:
   - active tool names
   - hidden tool names
   - active intrinsic tool names
   - hidden intrinsic tool names

## 4. Access Gating And Permission Checks

### 4.1 Primary Gate Location

Primary access gating must be implemented on the host before dispatch to the sandbox executor.

Reason:

1. host is the trust boundary and policy authority
2. sandbox code may be buggy or malicious
3. permission denials must happen before untrusted code starts
4. host already has session/role/access context

Required host-side checks before `toolbox.execute`:

1. requested sandbox executor is registered and alive
2. requested tool name is present in the staged manifest for that sandbox
3. caller/session is authorized to use that tool or toolbox bundle
4. toolbox scope permits the tool for the current request
5. sandbox policy allows the requested brokered operations implied by the call

### 4.2 Defense In Depth Inside The Sandbox

Sandbox-side checks are still required, but only as a secondary line:

1. executor loads only callables declared in the staged manifest
2. executor rejects unknown tool names even if host dispatch is buggy
3. executor refuses entrypoints outside bundle roots
4. executor refuses to mutate the active registry from user code unless the host explicitly sends a management command

### 4.3 Toolbox-Level Permission Use

`Toolbox.execute(...)` in [mp13_toolbox.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_toolbox.py) should remain the local execution gate for allowed/active tools, but it must not be the only security check for sandbox use.

Normative rule:

1. `Toolbox.execute(...)` is an execution-time consistency check
2. host-side sandbox authorization is the real permission gate

## 5. Adding And Removing Tools

### 5.1 Host-Managed Add Flow

Adding tools to a sandbox should work like this:

1. host receives a new tool bundle definition
2. host validates manifest, entrypoints, dependency metadata, and policy labels
3. host computes a revision hash
4. host stages the revision into a managed sandbox root
5. host resolves or builds the executor `.venv` for that bundle revision
6. host starts or refreshes the sandbox executor against that exact bundle revision
7. host updates registration metadata and diagnostics

Important first-cut rule:

1. do not let the sandbox executor discover new user callables from arbitrary ambient Python scope
2. for sandboxed use, tools should be loaded from a staged bundle manifest, not from ad hoc `search_scope`

Allowed discovery form:

1. host may stage Python modules and explicitly list callable names to discover from those modules
2. worker may then use `Toolbox.add_tool_callable(callable_name, search_scope=module.__dict__)`
3. that remains manifest-driven because the host declared both the module and the callable names up front

Intrinsic-tool activation should follow the same host-managed model.

Example: activating builtin calculator tools in a sandbox revision

1. host receives a logical toolbox change such as:
   - enable `scriptable_calculator`
   - include `scriptable_calculator_guide`
2. host records this as toolbox revision state, not as in-sandbox mutation
3. host stages or updates the revision manifest with:
   - `with_intrinsics=true`
   - requested intrinsic names
   - `with_intrinsic_guides=true` when guides are desired
4. toolbox executor starts from that revision and materializes the toolbox by:
   - creating `Toolbox(with_intrinsics=True, with_intrinsic_guides=True)`
   - restoring revision state through `from_dict(...)` or equivalent startup materialization
   - loading only the requested intrinsic entries from `INTRINSICS_REGISTRY`
5. host dispatches `toolbox.execute(name="scriptable_calculator", ...)` only after allowlist checks pass

The same pattern applies to `symbolic_algebra` and its guide tool.

### 5.2 Host-Managed Remove Flow

Removing tools should work like this:

1. host marks the tool or bundle inactive in sandbox management metadata
2. host stops routing new executions to that tool
3. host instructs executor to unload or restarts executor without that bundle
4. host removes staged bundle content when no active sandbox references it
5. host garbage-collects unused `.venv` directories only after reference checks pass

Normative rule:

1. removal must be host-driven and observable in status/diagnostics
2. sandbox user code must not delete or replace its own registered tool bundle

### 5.3 Mapping To Current `Toolbox`

Current `Toolbox` operations should be interpreted this way:

1. `add_tool_callable(...)` and `add_tool_external(...)` remain valid for trusted in-process workflows
2. sandbox-managed user tools should instead be materialized from host-staged manifests
3. `delete_tool(...)` remains a logical registry mutation, but sandbox staging cleanup belongs to hosting
4. `resolve_tool_link(...)` is not the sandbox authority path because it depends on ambient Python scope
5. intrinsic-tool enable/disable operations remain valid logical API operations, but sandboxed execution must persist them into toolbox revision state and roll out a replacement executor revision
6. `add_tool_callable("name", search_scope=...)` is the preferred sandbox-worker implementation path for auto-discovered user callables loaded from staged modules

## 6. Sandbox Filesystem And `.venv` Strategy

### 6.1 Recommended `.venv` Model

Sandbox executors should use immutable, host-built environments keyed by content.

Recommended shape:

1. one bundle manifest hash
2. one dependency lock hash
3. one resolved environment key derived from both

Recommended directories:

1. read-only staged bundle root
2. read/execute `.venv` root
3. writable scratch/work root
4. optional host-controlled cache root outside the sandbox write path

Grounding this in current builtin tools:

1. [mp13_tools_builtin.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_tools_builtin.py) imports:
   - standard-library modules such as `json`, `re`, and `codecs`
   - `numpy`
   - optional `numexpr`
   - `sympy`
2. That means intrinsic-tool sandbox environments cannot be treated as “pure stdlib only”.
3. The first `.venv` spec should therefore distinguish at least:
   - toolbox runtime base requirements
   - intrinsic-tool dependency set
   - staged user-tool dependency set
4. A sandbox revision that enables `scriptable_calculator` or `symbolic_algebra` must resolve to an environment key that includes the required builtin dependency tier, even if no user-provided dependencies are present.

### 6.2 Locking Strategy

Normative first-cut rules:

1. sandbox `.venv` must be built by the host, not by tool code at runtime
2. sandbox `.venv` must be treated as immutable after activation
3. writable runtime state must go to scratch, not to `.venv`
4. dependency resolution should come from a lockfile or frozen wheel set, not live unconstrained installs

Recommended first cut:

1. host-managed wheelhouse or pinned requirements input
2. build `.venv` outside the sandbox
3. stamp manifest and dependency hashes into executor registration metadata
4. mount or expose `.venv` as read/execute only to the sandbox worker where possible

Recommended environment-key shape:

1. `toolbox_runtime_hash`
2. `intrinsics_profile_hash`
3. `user_dependency_lock_hash`
4. derived `venv_key = hash(toolbox_runtime_hash, intrinsics_profile_hash, user_dependency_lock_hash)`

Example intrinsic profiles:

1. `none`
   - no intrinsic tools enabled
2. `calculator`
   - includes `numpy`
   - includes `numexpr` when present in the locked environment
3. `symbolic_math`
   - includes `sympy`
4. `calculator+symbolic_math`
   - includes both dependency groups

The exact grouping can change, but the plan should preserve one rule:

1. enabling intrinsic tools changes environment provenance when those tools require non-stdlib packages

### 6.3 Limitations To Document Explicitly

The plan must not over-claim the `.venv` story.

Known limitations:

1. Windows cannot provide a perfect trusted read allowlist for same-account files in this first cut
2. Low IL is primarily a write-protection boundary
3. if `.venv` build requires live network dependency resolution, reproducibility and policy enforcement become weak
4. mutable `.venv` patching in place makes provenance and rollback harder

Normative recommendation:

1. first trustworthy version should avoid live dependency installs during sandbox execution

## 7. Network Control Model

### 7.1 What We Can Trust In First Cut

Trusted first-cut network policy for sandbox executors:

1. brokered HTTP over host IPC
2. host-enforced hostname and URL-prefix allowlists on that brokered path
3. optional “no brokered network” mode

### 7.2 What We Must Not Claim

Do not claim these as fully supported on Windows first cut:

1. direct-socket hostname allowlists
2. direct-socket URL allowlists
3. route-specific network enforcement for arbitrary worker traffic
4. complete outbound block of all possible direct egress without additional OS/network infrastructure

### 7.3 Selected Route Clarification

If “selected route” means brokered outbound HTTP through the host:

1. that is in scope
2. host can enforce host/prefix policy there
3. status should be `supported` for brokered HTTP policy only

If “selected route” means direct worker networking through a particular NIC, proxy, or OS route:

1. that is not a trustworthy first Windows promise
2. it should remain `partial` or `unsupported` unless an explicit WFP/firewall/proxy enforcement layer is added

### 7.4 Communication Back To Host

Sandbox execution may need host callbacks for:

1. brokered filesystem
2. brokered HTTP
3. structured logs
4. cancellation/progress events
5. future capability requests

Normative rule:

1. all such callbacks must stay on existing IPC transport
2. no new sandbox-facing local web server should be added

Recommended callback request example:

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

## 8. Policy Model For Toolbox Sandboxes

Existing sandbox policy fields remain useful, but the semantics shift from “generic worker” to “toolbox executor”.

Additional recommended metadata:

1. `executor_kind`
   - `toolbox_executor_v1`
2. `bundle`
   - `bundle_id`
   - `bundle_revision`
   - `manifest_hash`
3. `environment`
   - `venv_key`
   - `venv_lock_hash`
   - `venv_mutable=false`
4. `tool_access`
   - allowed tool names
   - advertised tool names if needed for describe flows
5. `capabilities`
   - `brokered_filesystem`
   - `brokered_http`
   - `dynamic_reload`

## 9. Revised Platform Support Matrix

### 9.1 Windows

| Capability | Status | Notes |
|---|---|---|
| deny parent handle inheritance | supported | existing spawn hygiene work |
| deny writes to medium-integrity host files | supported | Low IL boundary |
| direct same-account read isolation | unsupported | not solved by Low IL |
| immutable host-built `.venv` exposure | partial | can be made practical, but trusted read isolation remains weak |
| brokered filesystem policy | supported | host policy decision point |
| brokered HTTP host/prefix allowlists | supported | on broker path only |
| direct worker networking disabled | partial | trustworthy only with stronger OS/network controls |
| direct worker hostname/URL allowlists | unsupported | do not claim without mediation |
| selected-route direct network enforcement | unsupported | not a first-cut promise |

### 9.2 Linux

| Capability | Status | Notes |
|---|---|---|
| filesystem containment with dedicated launcher | planned_supported | `bwrap` path not implemented yet |
| immutable host-built `.venv` exposure | planned_supported | mount/namespace model is stronger |
| brokered filesystem policy | planned_supported | same IPC contract |
| brokered HTTP host/prefix allowlists | planned_supported | same IPC contract |
| direct network disablement | planned_supported | namespace model |

## 10. Implementation Plan

### Phase 1: Scope Correction And Policy Terminology

1. Update docs/status to say trusted engine workers are not sandbox targets.
2. Rename the first consumer to toolbox sandbox executor.
3. Record Windows/network/.venv limitations explicitly.

Exit criteria:

1. no status doc claims the real engine worker is the production sandbox target
2. plan clearly places sandboxing around toolbox executors

### Phase 2: Host-Side Sandbox Registry And Authorization

1. Add sandbox bundle metadata model.
2. Add host-side authorization checks before sandbox dispatch.
3. Define manifest-driven tool inventory for executor registration.

Exit criteria:

1. host can say which tools are staged in which sandbox
2. unauthorized tool execution is denied before sandbox code starts

### Phase 3: Bundle Staging And Removal

1. Add host-managed bundle staging root.
2. Add add/remove lifecycle for staged bundles.
3. Add executor restart or reload semantics tied to bundle revision.

Exit criteria:

1. adding a tool means staging a new bundle revision
2. removing a tool stops new dispatch and cleans staged state when safe

Current implementation note:

1. staged revisions now support:
   - manual `ToolboxBundleTool` entries
   - auto-discovered `Toolbox.add_tool_callable(...)` entries from staged modules
   - intrinsic-tool revision state

### Phase 4: Immutable `.venv` Management

1. Define lockfile/wheelhouse input.
2. Define intrinsic dependency profiles based on builtin tool imports.
3. Build `.venv` out of band under host control.
4. Register executor with `venv_key` and provenance metadata.

Exit criteria:

1. sandbox execution does not mutate `.venv`
2. status can report bundle and environment hashes
3. intrinsic-tool activation is reflected in environment provenance, not just tool metadata

### Phase 5: Toolbox Executor IPC Contract

1. Add `toolbox.describe`
2. Add `toolbox.execute`
3. Add optional `toolbox.cancel`
4. Add generic `host.call`
5. Keep brokered `fs.*` and `http.fetch` on the same IPC transport through that callback path

Exit criteria:

1. toolbox execution can be requested over existing pipe/socket RPC
2. sandbox callbacks to host remain on existing IPC transport
3. toolbox-side execution context can expose simple callback helpers without exposing worker lifecycle details

### Phase 5A: Logical Toolbox Routing Across Sandbox Specs

1. Add sandbox-profile metadata for staged tool entries.
2. Allow one logical toolbox to span multiple sandbox executor pools.
3. Route each tool call to the correct pool based on tool name and sandbox profile.
4. Retire direct single-executor assumptions from higher-level toolbox APIs.

Exit criteria:

1. user can attach a permissions/dependency spec to a callable registration request
2. hosting can assign that callable to an existing sandbox profile or stage a new one
3. one logical toolbox can expose tools backed by different sandbox specs without user-managed routing

Current implementation note:

1. the first assignment slice now exists through:
   - `SandboxProfileSpec`
   - `ToolboxAutoAssignmentRequest`
   - `ToolboxSandboxOrchestrator`
2. profile ids can now be derived deterministically from required imports and sandbox policy
3. orchestration can now group requests by profile, stage one revision per profile, and spawn routed executor registrations under one logical toolbox id
4. persistent lifecycle now has a first host-service slice through `EngineHostService.toolbox_register_auto(...)`
5. removal lifecycle now also has a first host-service slice through `EngineHostService.toolbox_unregister_auto(...)`
6. daemon/control-channel/CLI surfaces now expose the same high-level registration and removal operations
7. environment identity now has a first host-managed slice through:
   - `ToolboxEnvironmentSpec`
   - `ToolboxEnvironmentManager`
8. environment metadata is now derived from toolbox runtime hash, intrinsic profile, required imports, and optional dependency lock hash
9. host now creates toolbox environment roots with stdlib `venv`, spawns workers via the environment Python executable, and reuses those roots by `venv_key`
10. unreferenced toolbox environment roots can now be garbage-collected from hosting state
11. staged toolbox executor registrations now carry `venv_key`, `venv_path`, `python_executable`, `venv_lock_hash`, `intrinsics_profile_id`, and `required_imports`
12. host-side register/unregister now performs a first readiness-gated cutover by waiting for new executors to answer `toolbox.describe` before retiring replaced registrations
13. successful cutovers now persist basic rollout metadata (`ready_at`, `warmup_ms`) per profile and return it from register/unregister operations
14. successful cutovers now also append to a bounded per-profile `rollout_history`
15. a first higher-level user-facing facade now exists through `SandboxedToolboxFacade`, hiding common auto-callable request construction on top of the service/channel methods
16. that facade can now also stage a real module-backed Python callable through `register_python_callable(...)`
17. that facade can now also register and unregister builtin intrinsic tools through sandbox hosting
18. that facade can now also register and unregister explicit manual tool definitions backed by Python implementations
19. remaining work is about richer rollout policies, stronger garbage-collection semantics, broader facade coverage, and locked immutable dependency installation/provenance, not the basic routing model itself

### Phase 6: Windows First Enforcement Slice

1. Reuse current restricted-token / Low IL / Job Object launcher work for toolbox executor workers.
2. Preserve existing IPC contract.
3. Validate write-up denial and handle inheritance controls.

Exit criteria:

1. sandbox executor still answers IPC requests
2. sandbox executor cannot modify medium-integrity host files

### Phase 7: Brokered Filesystem And HTTP For Toolbox Executors

1. Attach logical roots to toolbox executor bundles and scratch space.
2. Enforce brokered HTTP allowlists for executor callbacks.
3. Keep direct network claims explicitly limited.

Exit criteria:

1. toolbox executor can only use approved brokered roots and URLs through host checks
2. status clearly distinguishes brokered support from direct-network limitations

### Phase 8: Linux Backend

1. Add Linux sandbox launcher backend.
2. Map the same toolbox executor contract onto Linux.

Exit criteria:

1. same host-side bundle and authorization model works on Linux
2. Linux support labels are updated honestly

## 11. Recommended First Trustworthy Promise

The first trustworthy promise should be:

1. trusted engine workers are not sandboxed
2. user-defined toolbox callables can run in a separate sandbox executor
3. host decides which tools are staged and callable
4. sandbox executor cannot modify normal medium-integrity host files on Windows
5. sandbox executor filesystem and HTTP access must go through brokered host policy for the supported path
6. direct-network restrictions beyond the broker path remain partial or unsupported unless stronger OS/network enforcement is added

That is a coherent first design and does not over-claim the Windows boundary.
