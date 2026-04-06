# Toolbox Sandbox Architecture

Date: 2026-04-04
Scope: current implemented architecture for sandboxed toolbox execution, hosted lifecycle, chat/runtime integration, operator workflow, and known limits.

## 1. Purpose

This document is the architecture guide for the toolbox sandbox feature as it exists now.

It is meant to be sufficient for a new thread or a new contributor to understand:

1. what problem this feature solves
2. what the current architecture is
3. what the main runtime objects and responsibilities are
4. how chat/runtime integration works
5. how operator/admin recovery works
6. what is intentionally still incomplete

This document describes current reality first.

It also records current pitfalls and short improvement bullets, but it is not the step-by-step plan document. Use [sandbox_plan.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_plan.md) for that.

## 2. Problem And Scope

The toolbox subsystem allows dynamic tool registration and execution.

That is useful for trusted in-process use, but it is not enough for untrusted or semi-trusted tool code because:

1. tools may be user-defined Python callables
2. tools may need narrower filesystem or HTTP access than the host process has
3. tools may need to be routed by different permission profiles
4. tool lifecycle should be host-managed rather than ambient in-process mutation

So the sandbox feature exists to make toolbox execution host-managed and policy-aware.

Important scope correction:

1. trusted model-serving workers are not the sandbox target
2. toolbox executors are the sandbox target
3. the host remains the trust boundary and lifecycle authority

## 3. Core Design Principles

The implemented design follows these rules.

1. The host is authoritative.
   - host decides what tool code is staged
   - host decides which sandbox profile a tool belongs to
   - host decides whether a call is allowed before dispatch

2. Toolbox executors are disposable.
   - live executor registrations are runtime state
   - persisted logical toolbox state is the source of truth
   - repair/reconcile should rebuild rather than patch in place

3. One logical toolbox can span many sandbox profiles.
   - users think in terms of a logical toolbox
   - host routes each tool name to the right sandbox executor

4. Tool availability and tool executability are separate concerns.
   - a tool may be logically present
   - a given call may still be gated or denied

5. Brokered I/O is the trustworthy path.
   - brokered filesystem and brokered HTTP are supported
   - direct-network Windows controls are not treated as a trustworthy promise

6. Operator UX should be compact by default.
   - review, repair, and reconcile should not force operators to inspect raw low-level ids
   - deep details are opt-in

## 4. High-Level Runtime Model

The runtime has two primary layers.

### 4.1 Host Layer

The host layer owns:

1. persisted logical toolbox state
2. toolbox registration and removal
3. sandbox-profile assignment and routing
4. environment identity and named environment descriptions
5. worker spawn and replacement rollout
6. tool gating before dispatch
7. brokered filesystem and HTTP enforcement
8. references, consistency, repair, reconcile, and GC

### 4.2 Sandbox Executor Layer

The sandbox executor layer owns:

1. loading one staged toolbox revision
2. materializing a `Toolbox` from that revision
3. exposing RPC over the existing hosting IPC transport
4. executing allowed tool calls
5. making brokered callback requests to the host when tool code uses brokered fs/http

The executor is intentionally narrow. It does not own long-term lifecycle policy.

## 5. Main Runtime Pieces

Primary implementation entry points:

1. [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py)
   - staging helpers
   - sandbox profile types
   - environment types
   - hosted toolbox proxy
   - execution harness
2. [toolbox_executor_ipc.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_executor_ipc.py)
   - sandbox worker process
   - toolbox RPC handling
3. [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py)
   - authoritative host-side lifecycle and policy
4. [engine_host_channel.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_channel.py)
   - Python client wrapper over the host command surface
5. [engine_host_cli.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_cli.py)
   - operator-facing CLI
6. [engine_host_daemon.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_daemon.py)
   - daemon command transport
7. [toolbox_admin.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_admin.py)
   - long-lived server/operator convenience wrapper
8. [hosted_toolbox_api.py](/o:/repos/mp13-llm-engine/src/app/hosted_toolbox_api.py)
   - app-facing hosted toolbox helpers
9. [hosted_tool_runtime.py](/o:/repos/mp13-llm-engine/src/app/hosted_tool_runtime.py)
   - lightweight hosted tool-round runtime helper
10. [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py)
    - hosted demo and hosted execution wiring

## 6. Main Concepts And Data Model

The architecture revolves around a few key concepts.

### 6.1 Logical Toolbox

A logical toolbox is the user-facing identity.

Key property:

1. one logical toolbox may span many sandbox executors

Current key:

1. `toolbox_id`

Invocation identity is toolbox-scoped, not globally unique tool names.

That means:

1. tool names only need to be unique within a logical toolbox
2. routing is by `toolbox_id + tool_name`

### 6.2 Staged Toolbox Revision

A sandbox executor never loads tools from ambient Python scope.

Instead, it loads a staged toolbox revision.

A staged revision may contain:

1. staged Python source files
2. manual tool definitions
3. module/callable auto-discovery entries
4. intrinsic-tool activation state
5. active/hidden tool state
6. manifest hashes and provenance metadata

The revision is the unit of replacement rollout.

### 6.3 Sandbox Profile

A sandbox profile captures the execution boundary for a set of tools.

Current inputs include:

1. sandbox policy
2. brokered fs/http configuration
3. required imports
4. named environment description

The implementation uses `SandboxProfileSpec`.

One logical toolbox may have many profiles.

Each profile typically maps to one current active sandbox executor registration.

### 6.4 Environment Description

Environment descriptions are host-managed named descriptions, not ambient runtime state.

They exist to keep environment management understandable and explicitly controlled.

Current model:

1. one host-managed base environment concept
2. additional named environment descriptions may extend that base
3. tools can be linked to a named environment description
4. environment descriptions can inherit from base descriptions

### 6.5 Realized Environment

A realized environment is the actual venv root used to start a sandbox executor.

Current location:

1. `<hosting_root>/toolbox_venvs/<venv_key>`

Current identity inputs:

1. toolbox runtime hash
2. intrinsic dependency profile
3. required imports
4. environment-description identity
5. optional dependency-lock identity

### 6.6 Live Executor Registration

A live toolbox executor registration ties together:

1. engine id
2. logical toolbox id
3. sandbox profile id
4. staged bundle root and revision
5. environment metadata
6. allowed tool names

This is runtime state, not the authoritative model.

## 7. Staging And Loading Model

### 7.1 Why Staging Exists

Staging exists so that sandbox executors load exactly what the host declared.

That avoids:

1. ambient scope discovery
2. accidental dependency on unrelated in-process modules
3. unclear provenance of what code was executed

### 7.2 Supported Registration Styles

Current registration styles:

1. manual tool-definition registration
2. live Python callable registration
3. auto-callable registration from staged module/callable name
4. intrinsic-tool enablement

These are all normalized into staged toolbox revision state.

### 7.3 Validation Strength

Not all registration paths can validate equally before sandbox warmup.

Current intended distinction:

1. live callable or manual-definition-backed registration
   - stronger pre-staging validation
2. name-based auto-discovery
   - structural pre-staging validation
   - authoritative resolution happens during sandbox warmup

### 7.4 Worker Startup

The worker is started from a structured startup spec.

That spec carries:

1. manifest path
2. hosting-state pointers
3. scratch root
4. optional venv path
5. worker identity
6. IPC metadata

Legacy env-var fallbacks still exist, but the architecture should be understood as startup-spec driven.

## 8. RPC And Callback Model

### 8.1 Current Executor RPC Surface

Current executor RPC surface includes:

1. `rpc.describe`
2. `toolbox.describe`
3. `toolbox.execute`
4. `host.call`

### 8.2 Callback Model

Tool code can make host-mediated or client-mediated callback requests through execution context helpers.

Current convenience helpers:

1. `context.host.call(...)`
2. `context.fs.*`
3. `context.http.fetch(...)`
4. `context.callbacks.invoke(...)`

The current callback contract splits into two categories:

1. brokered host callbacks
   - filesystem reads/writes within brokered roots
   - HTTP fetches within brokered allowlists
   - host still authorizes these requests primarily by sandbox worker `engine_id`
   - but the request envelope now also carries per-call callback context:
     - `toolbox_id`
     - `tool_name`
     - `tool_call_id`
     - tool arguments
     - persisted `callback_signature`
     - caller-supplied hosted callback context when present
2. generic hosted callbacks
   - tool code can invoke `context.callbacks.invoke(name, payload)`
   - hosted caller code can pass a `callback_processor` when executing through:
     - `HostedToolBoxRef.execute(...)`
     - `ToolboxExecutionHarness.execute_calls(...)`
     - hosted `ToolboxExecutionHarness.execute_request_tools(...)`
     - the lightweight hosted runtime helper `execute_tool_round_on_cursor(...)`
   - the callback is processed on the hosted-caller side rather than by the host service itself

Generic hosted callbacks currently carry `HostedToolCallbackContext` with:

1. `toolbox_id`
2. `tool_name`
3. `tool_call_id`
4. `tool_arguments`
5. `engine_id`
6. `callback_name`
7. `callback_payload`
8. persisted `callback_signature`
9. caller-supplied `user_context`

Tool registration can now persist optional `callback_signature` metadata for:

1. hosted auto-callable registration
2. hosted Python-callable registration
3. hosted manual tool registration

Hosted `describe` now surfaces that metadata through `tool_metadata`.

Current brokered FS/HTTP follow-through:

1. brokered filesystem and brokered HTTP service methods now accept the same callback-context envelope
2. brokered callback responses now preserve that envelope in their returned payloads
3. this gives the host and the tool result path direct attribution back to the originating tool call

### 8.3 Callback Concurrency Contract

Generic hosted callbacks are intentionally not serialized through one blocking callback loop.

Current behavior:

1. each hosted execute call binds a callback relay session
2. the relay accepts callback connections concurrently
3. each callback connection is handled in its own thread
4. a blocked callback processor only blocks the callback waiting on that processor result
5. other callbacks on the same hosted execute call can still proceed
6. unrelated tool calls can still proceed according to the normal sandbox execution contract

### 8.4 Important Constraint

No separate sandbox-facing HTTP server is introduced for callbacks.

All callback traffic stays on existing hosting IPC or the per-execute hosted callback relay path.

## 9. Logical Toolbox Routing

### 9.1 Why Routing Exists

Different tools may need:

1. different permissions
2. different brokered roots
3. different network policy
4. different dependency sets

That means one logical toolbox cannot always map to one sandbox process.

### 9.2 Current Routing Rule

Host routes by:

1. `toolbox_id`
2. requested tool name

Host then resolves:

1. which sandbox profile owns that tool
2. which active executor registration currently serves that profile

### 9.3 What The User Sees

The user still sees one hosted toolbox proxy.

The routing is hidden behind:

1. `HostedToolBoxRef`
2. hosted execution harness
3. chat/runtime wiring

## 10. Call Gating Model

### 10.1 Reason For Gating

The architecture now distinguishes:

1. logical tool membership
2. execution-time authorization for a specific call

This is necessary because a tool can be visible while still being denied or gated in the current hosted context.

### 10.2 Current Implemented Slice

Current first slice includes:

1. native/toolbox-facing `Toolbox.gate_call(...)`
2. hosted `toolbox_gate(...)`
3. hosted execution preflight before dispatch
4. gated errors surfaced distinctly from tool crashes
5. hosted `describe` now reports:
   - `all_registered_tool_names`
   - `advertised_tool_names`
   - `hidden_allowed_tool_names`
6. hosted app/runtime helpers now preserve the native distinction between:
   - advertised and allowed
   - hidden but allowed
   - blocked in scope

### 10.3 Intended Semantic Relationship To Native Toolbox Access Control

The hosted sandbox model is not intended to define a second access-control system.

The original/native `Toolbox` contract remains the semantic baseline:

1. visibility
   - what the LLM sees
   - determined by native toolbox mode, hidden state, and `ToolsScope`
2. execution
   - what is allowed to run
   - determined by `ToolsView.is_allowed(...)` and `Toolbox.gate_call(...)`

Hosted sandbox execution is intended to extend that contract, not duplicate it:

1. native toolbox remains responsible for logical visibility and scoped execution semantics
2. hosted sandbox adds backend-specific outcomes:
   - routed sandbox-profile ownership
   - sandbox policy denials
   - unavailable backend / missing executor
3. the effective hosted contract should therefore be:
   - first apply native toolbox access resolution
   - then apply hosted backend routing and policy checks

This means the long-term hosted model should preserve native categories such as:

1. advertised and allowed
2. hidden but allowed
3. disabled / blocked in scope

without redefining them in a sandbox-only vocabulary

### 10.4 Current Limits

The broader prompt/runtime stack is still not fully gate-driven everywhere.

Current state is better than the initial hosted slice:

1. hosted gate/execute now accept request-scoped `ToolsView` payloads
2. hosted user-tool hidden state is now representable in persisted requests
3. hosted visibility reporting is now split into registered vs advertised vs hidden-allowed sets

Remaining architectural limit:

1. not every app/runtime path has been proven end to end against the full native toolbox contract
2. the most polished slices are:
   - hosted chat inspection and `/t` presentation
   - focused hosted tool-round runtime
   - operator review/admin surfaces
3. chat itself is still intentionally serial in the current implementation even though hosted/native toolbox execution can support parallel multi-call execution underneath

### 10.5 Execution Granularity And Multi-Call Contract

Current execution granularity is explicit and should not be inferred from the chat UX.

Single-call contract:

1. native `Toolbox.execute(...)` executes one `ToolCall`
2. hosted worker RPC `toolbox.execute` executes one `ToolCall`
3. `HostedToolBoxRef.execute(...)` executes one `ToolCall`

Multi-call contract:

1. multiple tool calls from one LLM response are handled by:
   - native `Toolbox.execute_request_tools(...)`
   - hosted `ToolboxExecutionHarness.execute_request_tools(...)`
   - hosted `ToolboxExecutionHarness.execute_calls(...)`
2. both native and hosted support sequential or parallel execution controlled by `serial_execution` or `parallel`
3. non-chat hosted runtime now defaults to parallel multi-call execution
4. `mp13chat` intentionally still forces serial tool execution for now

Response-boundary contract:

1. multiple tool calls from one LLM response can be processed in one native or hosted tool round
2. multiple separate LLM responses are not merged into one atomic toolbox RPC
3. each response or auto-tool round is parsed and executed as its own round, even if the same logical toolbox continues across turns

## 11. Environment Architecture

### 11.1 Current Design Choice

The environment model intentionally stays simpler than a fully generalized package-management platform.

The intended mental model is:

1. base toolbox environment
2. named environment descriptions
3. tools linked to those descriptions
4. host-managed apply/realize/install lifecycle

### 11.2 Current Host APIs

Current environment-related host APIs cover:

1. description list/get/upsert/clone
2. requirement resolution against linked tools
3. apply description to linked toolbox profiles
4. realize environment metadata
5. prepare install plan
6. lock install plan
7. resolve stronger exact lock
8. execute install
9. verify lock
10. verify observed receipt

### 11.3 What Is Strong Today

What is already good enough:

1. deterministic environment identity
2. host-built venv roots
3. named environment descriptions
4. explicit apply/rebuild flows
5. explicit install planning and provenance recording

### 11.4 What Is Still Weak

What is still not the final story:

1. resolver-backed immutable lock policy
2. full reproducibility guarantees
3. mature upgrade/re-resolution policy

So the current environment model is usable, but should not be oversold as a production-grade package-management system.

## 12. Rollout And Replacement Model

### 12.1 Current Rollout Policy

Current rollout is intentionally minimal.

Implemented checks:

1. spawn replacement executor
2. wait for readiness
3. verify reported tool inventory matches expected allowlist
4. cut over
5. rollback on failed warmup
6. structured rollout error reporting

### 12.2 Why It Is Acceptable For Now

This is enough to support:

1. real replacement rollouts
2. predictable failure behavior
3. compact operator recovery flows

It is not intended to be a full deployment/orchestration platform.

### 12.3 What Is Not Implemented

Not implemented:

1. replicas
2. staged percentage cutover
3. soak windows
4. post-cutover health orchestration beyond the current simple model

## 13. Operator And Recovery Model

### 13.1 Source Of Truth

Persisted logical toolbox state is the source of truth.

Live registrations, bundle roots, and realized env roots are operational state derived from that truth.

### 13.2 Current Operator Surfaces

Current operator surfaces:

1. `toolbox-references`
2. `toolbox-consistency`
3. `toolbox-review-snapshot`
4. `toolbox-repair`
5. `toolbox-reconcile`
6. `toolbox-gc`

### 13.3 Current Intended Meanings

1. `toolbox-references`
   - what is referenced vs stale
2. `toolbox-consistency`
   - whether referenced state is coherent
3. `toolbox-review-snapshot`
   - compact pre-action summary
4. `toolbox-repair`
   - rebuild inconsistent toolboxes from persisted state
5. `toolbox-reconcile`
   - consistency + selective repair + GC
6. `toolbox-gc`
   - cleanup of stale artifacts

Repair/reconcile are now serialized per targeted `toolbox_id`, so concurrent rebuild attempts against the same logical hosted toolbox do not overlap state rewrite or sandbox respawn work.

### 13.4 Current UX Rule

Default output should be compact.

Operators should mostly see:

1. requested toolbox ids
2. target toolbox ids
3. repaired toolbox ids
4. removed artifact counts
5. before/after issue counts
6. overall outcome

Deep internals are opt-in via `details=true`.

### 13.5 Current Admin Helper

`HostedToolboxAdmin` is the small long-lived-process wrapper over the same contract.

It currently supports:

1. review snapshot
2. startup reconcile
3. periodic consistency check
4. optional auto-repair-if-needed

## 14. Chat And App Integration

### 14.1 Public Hosted Proxy

`HostedToolBoxRef` is the public hosted toolbox proxy type.

It preserves the toolbox-ref programming model while hiding most lifecycle detail.

### 14.2 App Helpers

Current app-facing helpers:

1. `create_hosted_toolbox_ref(...)`
2. `register_hosted_tool_callable(...)`
3. `create_hosted_toolbox_executor(...)`
4. `HostedToolExecutionRouter`
5. `create_hosted_control_channel(...)`
6. `attach_existing_hosted_toolbox(...)`
5. `execute_tool_round_on_cursor(...)`
6. `is_hosted_tool_call_canceled(...)`
7. `should_resubmit_hosted_tool_call(...)`

### 14.2.1 Hosted Toolbox Ref Builder API

To minimize sandbox rebuild penalties when registering multiple tools, `HostedToolBoxRef` implements a builder pattern.

1. Call `ref.mutate()` to create a `PendingHostedToolboxRef`.
2. Accumulate changes locally using builder registration methods.
3. Call `builder.resolve_sandbox()` explicitly to trigger one synchronous backend update.
4. Tool registrations can now persist `non_restartable=false|true` so later coarse-cancel recovery policy can distinguish tools that should not be auto-resubmitted.

### 14.3 Chat Integration

`mp13chat` can now:

1. configure hosted execution
2. keep local toolbox-ref state
3. route actual tool execution through hosted sandbox executors
4. expose hosted-aware tool inspection

Important current split:

1. non-chat hosted tool rounds now default to native-style parallel multi-call execution
2. `mp13chat` remains intentionally serial for tool execution at the moment
3. the chat/runtime split is therefore policy, not capability:
   - native toolbox supports multi-call rounds
   - hosted harness supports multi-call rounds
   - chat currently chooses serial execution explicitly

### 14.4 Hosted Demo

The hosted demo is the main polished scenario at the moment.

There is now also a lighter non-demo adoption path in `mp13chat` itself:

1. attach to an already provisioned hosted toolbox by passing:
   - `--hosted-toolbox-id`
   - `--hosted-engines-state-file`
   - `--hosted-control-state-file`
2. this reuses the same hosted execution router without provisioning demo tools or demo hosting state
3. this is the first broader app/runtime adoption slice beyond the built-in hosted demo

The same attach path is also exposed as a public app helper:

1. `attach_existing_hosted_toolbox(...)` builds:
   - the local hosted control channel
   - a `HostedToolBoxRef`
   - a hosted execution harness
   - the current hosted toolbox summary
2. wrapper apps can therefore attach to existing hosted toolbox deployments without copying `mp13chat` CLI-specific glue
3. the same hosted attach helpers are also re-exported from the top-level `app` package for cleaner wrapper imports
4. `demo/demo_hosted_toolbox_attach.py` is the minimal sample wrapper showing this non-demo attach flow against an existing hosted toolbox deployment

It validates:

1. multi-profile routing in chat
2. hosted-visible tool advertisement
3. brokered filesystem tool behavior
4. brokered HTTP tool behavior
5. clean deny paths
6. compact operator review/repair/reconcile while chat is live

Important limitation of the current polished chat slice:

1. chat currently compensates for some hosted contract gaps by filtering inference payloads to a hosted-visible set
2. that makes the user-facing hosted demo work well
3. but it should not be mistaken for full hosted/native semantic parity underneath

## 15. Remote / Thin-Client Model

The current architecture supports a remote thin client.

Supported model:

1. client uses `EngineHostControlChannel`
2. client constructs `HostedToolBoxRef`
3. client registers tools or executes tools through the hosted proxy
4. hosting server performs staging, lifecycle, policy, and execution

Current caveat:

1. some registration modes still read source on the client before uploading staged content
2. there is not yet a separate “register by module path already present only on the hosting server” mode

Still, the thin-client model is already real and usable.

Coarse cancellation contract in the thin-client model:

1. `toolbox.cancel` is coarse executor-level cancellation, not per-call cooperative cancellation
2. a successful cancel kills the targeted sandbox worker and can optionally respawn healthy replacement workers from persisted toolbox state
3. in-flight hosted tool calls that lose their worker are normalized to `canceled` tool-call errors at the harness boundary
4. wrappers can use `should_resubmit_hosted_tool_call(...)` plus the persisted `non_restartable` flag to choose whether to resubmit on a fresh sandbox

Generic callback contract in the thin-client model:

1. wrappers can register hosted tools with optional `callback_signature`
2. wrappers can pass a `callback_processor` plus optional caller `callback_context` at execute time
3. the tool can call back through `context.callbacks.invoke(...)`
4. the wrapper receives structured callback context identifying the toolbox, tool, call id, and caller-supplied context

## 16. Current Pitfalls And Limits

These are the main caveats that still matter architecturally.

1. Windows Low IL is mainly a write boundary, not strong read isolation.
2. Direct-network route control on Windows is not a trustworthy promise.
3. Brokered HTTP is the supported network path.
4. `.venv` lifecycle is usable, but not yet a mature immutable dependency-management system.
5. GC/reference tracking is coherent, but not yet deeply production-style.
6. Rollout policy is intentionally minimal.
7. `toolbox.cancel` is only coarse executor-level cancellation.
8. Linux backend is now viable and validated in the WSL shared-shadow model, but not yet broadly proven across a wider Linux distro/platform matrix.
9. Some hosted chat/runtime behavior is polished only in the current hosted slices, not universally across every app path.
10. Hosted access control is much closer to the original native `Toolbox` design, but broader end-to-end coverage is still more mature in the focused hosted slices than in every possible consumer.
11. Hosted user-tool hidden state is first-class in persisted requests, but a full native hidden/silent vocabulary is still not expanded into a richer separate hosted reporting contract than the current registered/advertised/hidden split.

## 17. What The Current Polished Scenarios Prove

The current polished scenarios prove two important contracts.

### 17.1 User-Facing Contract

1. hosted tools can feel like normal chat tools
2. hosted advertisement can stay aligned with hosted executability
3. deny paths can appear as tool-result failures rather than app/runtime crashes
4. shared hosted toolbox refs can tolerate concurrent read/execute/cancel use
5. non-chat hosted execution can execute multiple tool calls from one response in parallel

### 17.2 Operator-Facing Contract

1. hosted toolbox state can be reviewed while live
2. healthy systems produce compact no-op repair/reconcile output
3. operators do not need low-level ids on the default path
4. concurrent repair attempts against the same toolbox serialize instead of racing

## 18. Short Improvement Bullets

Near-term:

1. keep validating remaining app/runtime consumers against the current hosted visibility and gate contract
2. decide whether client-facing helper coverage around coarse cancel should expand beyond the current retry-decision helpers
3. keep compact operator UX consistent across all admin outputs
4. decide whether request-level cancellation is worth the deeper executor redesign
5. widen locking further only if housekeeping paths beyond registration/repair/reconcile become a real concurrency hotspot

Medium-term:

1. stronger immutable env/provenance model
2. deeper reference-tracked GC semantics
3. broader long-lived server automation/runbook guidance
4. broader app/runtime adoption beyond the current hosted demo slice

## 19. Related Sandbox Docs

1. [sandbox_test_status.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_test_status.md)
2. [sandbox_plan.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_plan.md)
3. [sandbox_status.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_status.md)

## 20. WSL Shared Validation Model

Current Linux validation is centered on a WSL-native shadow root rather than on sharing one repo-local `.venv` across Windows and Linux.

Recommended structure:

1. keep the primary Windows checkout unchanged
2. create a WSL-native shadow root such as `~/mp13-wsl`
3. symlink live code/content from the Windows checkout into that root
4. keep the Linux `poetry.lock` and Linux `.venv` owned by the WSL shadow root

Why this model is preferred:

1. it avoids the common broken state where Windows creates `.venv/Scripts/python.exe` and WSL expects `.venv/bin/python`
2. it keeps Linux dependency installation and Linux test execution in the Linux filesystem
3. it still allows live source edits from the main Windows checkout to be reflected in WSL tests immediately

Operationally, this means:

1. Windows remains the main editing/control-plane environment
2. WSL is the Linux validation environment
3. the helper `misc/wsl_shared_test_setup.py` is the quick check for whether the WSL shadow root is usable before running Linux pytest slices
4. current Linux worker launch policy is intentionally pragmatic:
   - sandbox workers stay on the active project interpreter until the realized environment has both successful install execution and successful receipt verification
   - once that install state is verified, worker launch can switch to the realized environment's Python
