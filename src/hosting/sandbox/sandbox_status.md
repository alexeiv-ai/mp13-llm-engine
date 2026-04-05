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

1. hosted execution is not yet fully scope-equivalent to native `Toolbox` execution
   - request-scoped `ToolsView` restrictions are not the authoritative gate on the hosted path yet
2. hosted user tools do not yet fully support native hidden/silent semantics
   - hidden intrinsics are representable
   - hidden hosted auto/manual user tools are not a first-class hosted state yet
3. hosted `describe` currently exposes tool membership more strongly than true advertised visibility
   - practical chat UX is aligned in the polished hosted slice
   - the underlying hosted contract is still coarser than native `ToolsView`
4. live dead-worker detection is still weaker than ideal
5. Windows direct-network enforcement is still not a trustworthy claim
6. env/provenance is still short of a full immutable dependency-management story
7. Linux backend is still missing
8. `toolbox.cancel` is still missing

## 6. Starting Point For The Next Thread

If starting fresh from these docs, assume:

1. architecture is coherent enough to continue implementation
2. the next recommended execution item is native/hosted access-control parity
3. the highest-priority semantic fixes are:
   - hosted enforcement of request-scoped `ToolsView`
   - hosted hidden/silent parity for user tools
   - hosted `describe` separation of allowed vs advertised visibility
4. after that, add live worker liveness probing in consistency/review/reconcile
5. then reassess whether further operator polish is needed before tackling deeper env/provenance or Linux work
