# Sandbox Status

Date: 2026-04-04
Purpose: context needed before continuing execution from the current sandbox/toolbox plan.

## 1. Current State Summary

The sandbox/toolbox effort is past the fundamentals stage.

Current status:

1. hosted sandboxed toolbox execution is real and working
2. one logical toolbox can route tools across multiple sandbox profiles
3. chat can run a hosted demo end to end
4. operator flows now have compact defaults
5. env management is usable, though not fully production-hardened
6. hosted gate/execute now accept request-scoped hosted `ToolsView` payloads so host-side dispatch can deny `blocked_in_scope` before sandbox IPC
7. hosted `describe` now reports separated `all_registered_tool_names`, `advertised_tool_names`, and `hidden_allowed_tool_names`
8. staged/persisted hosted user-tool requests can now carry hidden membership, and staged manifests preserve hidden user-tool state
9. app-facing hosted visibility helpers now distinguish hosted advertised tools from hosted hidden-but-allowed tools instead of collapsing both into one advertised-only set
10. focused hosted runtime tests now cover one round with both `blocked_in_scope` denial and hidden-but-allowed hosted execution using the same hosted `ToolsView`
11. toolbox consistency/review now probe live executor IPC reachability and surface dead-but-registered workers as explicit consistency issues
12. env install verification now treats `resolved_install_lock` as a first-class provenance artifact, and receipt verification refuses to certify installs against stale lock state
13. real chat command coverage now includes `/t` and `/t sc` presentation for hosted visible, hosted hidden-but-allowed, and hosted-gated tools in one integrated flow
14. `mp13chat` `/t` command handling now delegates to a shared lightweight tools-CLI handler, so the hosted chat test path and live chat tool-management path are aligned instead of duplicated
15. operator review/admin profile rows now carry `all_registered_tool_names`, `advertised_tool_names`, and `hidden_allowed_tool_names`, so non-chat hosted inspection surfaces reflect the same hosted visibility split
16. `HostedToolBoxRef` now supports a `.mutate()` builder API, allowing clients to aggregate multiple tool registrations into a single synchronous backend update, significantly minimizing sandbox rollout penalty.
17. hosted sandbox clients now have a coarse `toolbox.cancel` path that stops targeted executor workers and repairs persisted toolbox state, giving client code a real abort-and-recover operation without requiring per-tool cooperative cancellation
18. hosted toolbox mutation flows for auto/manual/intrinsic registration are now serialized per `toolbox_id`, reducing state races when the same hosted toolbox ref is used concurrently across client threads
19. coarse `toolbox.cancel` can now record optional tool-call identity and persist recent cancel events in toolbox runtime metadata so later restart policy can know what caused sandbox recycling
20. hosted auto/manual tool registrations can now persist a `non_restartable` flag, defaulting to `false`, so future sandbox-restart policy can distinguish tools that should not be auto-resumed
21. non-chat hosted tool-runtime execution now defaults to parallel multi-call dispatch, so one hosted response can execute multiple tool calls with native-style concurrency even while `mp13chat` remains explicitly serial for now
22. hosted repair/rebuild flows now serialize per targeted `toolbox_id`, so concurrent repair attempts against the same hosted toolbox do not overlap sandbox respawn or state rewrite work
23. lightweight hosted tool-round execution now returns coarse-cancel guidance directly in `ToolRoundResult`, including canceled tool names and resubmittable tool names after applying `non_restartable`
24. hosted demo helpers now expose the exact `execute_tool_round_on_cursor(...)` options a thin client should pass for coarse-cancel retry policy, using the plan's persisted `non_restartable` flags
25. remaining app-facing hosted presentation helpers no longer fall back to coarse `all_registered_tool_names`; UI-facing hosted visibility now reads the explicit advertised/hidden split only
26. environment resolution/execution policy is tighter: resolver work and install execution now require an explicit locked install plan instead of proceeding from plan-only state
27. resolved-lock provenance now includes the resolver report artifact itself: the stored `pip --report` JSON hash is verified alongside the resolved package list, and tampered/missing report artifacts now stale the resolved lock
28. successful install execution now immediately certifies and persists `install_receipt_verification`, so observed install state is checked against the lock source as part of execution rather than only by a later manual verify step
29. rollout readiness now treats successful install execution as incomplete unless `install_receipt_verification` is also `ok`, so stale observed install state can block a toolbox from being treated as ready
30. GC/reference tracking now protects realized toolbox environments by resolved `venv_path` as well as `venv_key`, so referenced environments are not misclassified as stale when the on-disk folder name differs from the logical environment key
31. `toolbox_references()` now exposes explicit reference reasons for retained environment keys, realized environment roots, and bundle roots, so operators can see why an artifact was kept instead of only seeing stale lists
32. `toolbox_gc()` now returns collection-side provenance for removed registrations, bundle directories, and environment directories, so GC results explain why stale artifacts were collected instead of only listing names

## 2. Most Important Current Contracts

### 2.1 Trust Contract

1. trusted engine workers are not the sandbox target
2. toolbox executors are the sandbox target
3. host is the policy and lifecycle authority

### 2.2 User Contract

1. hosted tools should feel like normal chat tools
2. hosted-visible advertisement should align with executability closely enough for practical use
3. deny paths should show up as tool-result failures, not runtime crashes
4. hosted execution should preserve native `Toolbox` access semantics, not silently broaden them
5. the same hosted toolbox ref may be shared across client threads for concurrent describe/gate/execute/cancel use
6. hosted toolbox mutation is expected to serialize per logical toolbox id rather than racing persisted state

### 2.2A Access-Control Alignment Contract

The original/native `Toolbox` contract has two axes:

1. visibility
   - governed by global mode, hidden state, and per-turn `ToolsScope`
2. execution
   - governed by `ToolsView.is_allowed(...)` and `Toolbox.gate_call(...)`

Hosted sandbox execution is expected to support and extend that model:

1. native `Toolbox` remains the semantic source of truth for:
   - advertised tools
   - hidden-but-allowed tools
   - disabled tools
   - scoped per-turn overlays
2. hosted sandbox adds only backend-specific concerns:
   - routed sandbox profile ownership
   - sandbox policy denials
   - unavailable executor/backend states
3. hosted gating must therefore extend native gating, not duplicate it in a separate incompatible form

### 2.3 Operator Contract

1. persisted logical toolbox state is authoritative
2. live sandbox workers are disposable
3. review/repair/reconcile should be compact by default
4. deep internals are opt-in with `details=true`

## 3. Important Files

Architecture and status docs:

1. [sandbox_architecture.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_architecture.md)
2. [sandbox_test_status.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_test_status.md)
3. [sandbox_plan.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_plan.md)

Core implementation:

1. [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py)
2. [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py)
3. [toolbox_executor_ipc.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_executor_ipc.py)
4. [toolbox_admin.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_admin.py)

Chat/runtime slice:

1. [hosted_toolbox_api.py](/o:/repos/mp13-llm-engine/src/app/hosted_toolbox_api.py)
2. [hosted_tool_runtime.py](/o:/repos/mp13-llm-engine/src/app/hosted_tool_runtime.py)
3. [hosted_chat_demo.py](/o:/repos/mp13-llm-engine/src/app/hosted_chat_demo.py)
4. [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py)

## 4. Latest Validated Flows

Validated in the user environment:

1. hosted demo chat works end to end
2. hosted tool visibility in `/t` and `/t sc` is aligned with the hosted backend
3. brokered HTTP deny path gives:
   - `PermissionError - brokered_http_url_not_allowed:https://example.org/`
4. brokered filesystem traversal deny path gives:
   - `BrokeredFsError - path_traversal_denied`
5. healthy `toolbox-review-snapshot` recommends `observe`
6. healthy `toolbox-repair` returns compact `noop`
7. healthy `toolbox-reconcile` returns compact `noop`

## 5. Known Gaps That Still Matter

1. hosted semantic parity is improved, but still not fully proven across every app/runtime path
   - host-side dispatch now accepts request-scoped `ToolsView`
   - app-side visibility helpers now preserve hosted hidden-but-allowed tools
   - app-facing hosted presentation helpers now avoid coarse full-membership fallback for visible-tool lists
   - focused runtime coverage now exercises scoped deny plus hidden-but-allowed hosted execution in one hosted round
   - real chat coverage now exercises `/t` and `/t sc` presentation for hosted visible, hidden-allowed, and gated states
   - `mp13chat` tool-management handling now shares the lightweight handler used by hosted tests
   - operator review/admin now expose the same registered/advertised/hidden split
   - remaining risk is mostly future consumers, not the current app/admin helpers that have now been re-audited
2. hosted user-tool hidden state is now representable in staged/persisted hosted requests
   - remaining gap is broader operator/runtime surface adoption, not manifest-state absence
3. live dead-worker detection is now present in consistency/review, but operator UX may still want small polish
   - dead-but-registered executors now surface as explicit consistency issues
   - remaining question is whether default review output needs any more compact liveness wording
4. Windows direct-network enforcement is still not a trustworthy claim
5. env/provenance is still short of a full immutable dependency-management story
   - resolved lock hash and resolved requirements file are now verified against the current install plan and environment identity
   - resolved lock now also verifies the persisted resolver report artifact hash
   - receipt verification now short-circuits on stale lock state instead of validating against a drifted lock source
   - resolution/execution now require an explicit locked plan instead of tolerating plan-only installs
   - successful install execution now also records immediate receipt verification
   - rollout readiness now enforces verified observed install state when an environment has recorded successful install execution
   - remaining gap is policy depth, not total absence of lock verification
6. Linux backend is still missing
7. `toolbox.cancel` now exists only as coarse executor-level cancellation
   - it cancels by stopping sandbox executor worker(s) and repairing persisted toolbox state
   - cancel events can now persist `tool_name`, optional `tool_call_id`, and `non_restartable` state for later restart-policy work
   - remaining gap is finer-grained in-flight request cancellation if that becomes necessary
8. concurrent execution is better than concurrent mutation
   - same sandbox executor can serve overlapping tool calls
   - same hosted toolbox ref can now tolerate concurrent read/execute-style use better
   - non-chat hosted tool rounds no longer force serial execution, so multiple tool calls in one response can overlap through the sandbox harness
   - repair/rebuild now also serialize per targeted toolbox id
   - broader housekeeping paths outside the explicit per-toolbox registration and repair flows may still want more locking if concurrency pressure increases
9. GC/reference tracking is more robust, but still intentionally simple
   - stale environment detection now considers both logical `venv_key` references and resolved `venv_path` references under `toolbox_venvs`
   - `toolbox_references()` now also reports retention reasons for referenced env keys, env roots, and bundle roots
   - `toolbox_gc()` now also reports collection reasons for removed registrations, removed bundle directories, and removed environments
   - remaining gap is richer lifecycle history if operators later need GC actions tied back to prior rollout/repair events

## 6. Starting Point For The Next Thread

If starting fresh from these docs, assume:

1. architecture is coherent enough to continue implementation
2. the next recommended execution item is native/hosted access-control parity
3. the highest-priority semantic fixes are:
   - verify any remaining downstream consumers use hosted `advertised_tool_names` and `hidden_allowed_tool_names` rather than coarse full membership
   - treat gated tool-call completeness and hosting integration as explicit follow-through work, not just helper parity
   - re-check whether any non-chat app path still bypasses hosted request-scoped `ToolsView`
4. env/provenance is somewhat tighter now:
   - `resolved_install_lock` drift blocks execution
   - stale lock state also blocks receipt certification
   - the next env step, if worth doing, is policy hardening such as requiring resolver work to start from an already-locked plan or storing a stronger external resolver provenance model
5. otherwise move on to the next investment rather than expanding sandbox operator output unnecessarily
6. if client-facing cancellation needs become more demanding, the next step is request-level cancellation rather than another coarse worker-restart layer
7. if concurrency becomes a primary product concern, the next step is to widen per-toolbox serialization coverage and then let chat/runtime exploit parallel tool execution more aggressively
8. if continuing strictly by the plan, the next near-term bucket after env/provenance is deeper GC/reference provenance rather than another client/runtime feature
