# Sandbox Status

Date: 2026-04-06
Purpose: fresh status snapshot for the gated-tool roadmap after Phase 1 implementation and callback-relay validation updates.

## 1. Current Position

The foundational sandbox/toolbox work is largely complete.

At a coarse level, the current system already provides:

1. real hosted sandboxed toolbox execution
2. logical-toolbox routing across sandbox profiles
3. hosted/native access-control parity first slice
4. hidden vs advertised hosted visibility split
5. compact operator lifecycle flows
6. coarse executor-level cancellation
7. generic hosted callbacks
8. brokered fs/http callback attribution follow-through
9. WSL-validated Linux backend coverage
10. app/runtime adoption beyond the original hosted demo path
11. native Windows generic hosted callback relay restored for low-IL sandbox workers

So the active work is no longer “finish the sandbox foundation”. The active work is the next semantic feature:

1. gated tools
2. hosted approval flow for gated tools

## 2. Active Plan Status

### 2.1 Phase 1: Semantic Gated State

Status: implemented first slice

What exists already:

1. `disabled` and scoped-deny semantics exist
2. hidden vs advertised semantics exist
3. hosted/native gate surfaces already support distinct gate outcomes

What is missing:

1. broader doc consolidation across all sandbox/user-facing guides
2. deeper end-to-end coverage beyond the currently focused hosted slices
3. final polish around every presentation surface

### 2.2 Phase 2: Interactive Hosted Approval

Status: first hosted slice implemented

What exists already:

1. hosted callback relay
2. generic callback processor contract
3. per-call callback context:
   - toolbox id
   - tool name
   - tool call id
   - tool arguments
   - caller context
4. approval callback contract:
   - callback name: `tool_requires_confirmation`
   - payload kind: `tool_approval_request`
   - decisions: `deny`, `allow_once`, `add_to_scope`
5. current execution semantics:
   - `allow_once` mutates only the current execution view
   - `add_to_scope` mutates the current execution view and persists through a provided `ToolBoxRef`
   - app-level hosted runtime now auto-forwards the active cursor plus context `toolbox_ref` for hosted rounds
   - delayed or missing approval now defaults to deny by timeout
   - repeated gated calls in the same hosted round dedupe by tool name for sticky decisions:
     - `deny`
     - `add_to_scope`
   - `allow_once` is intentionally not cached because it is per-call only
   - default behavior remains deny when no approval processor is present or the decision is invalid

What is missing:

1. final docs for public approval-callback usage
2. broader wrapper adoption beyond the hosted runtime helper and direct hosted ref path
3. confirmation of the preferred public timeout override surface

Wrapper-consistency audit result:

1. `execute_tool_round_on_cursor(...)` is the only public helper that auto-forwards a durable scope target for `add_to_scope`
2. direct `HostedToolBoxRef.execute(...)` and raw hosted harness usage are consistent only when the caller passes `callback_context` with:
   - `toolbox_ref`
   - or `cursor` whose context owns a `toolbox_ref`
3. current docs now state that explicitly; behavior was not changed in this audit

### 2.2A Dynamic Constraint Layer

Status: first slice implemented

Motivation:

1. binary `add_to_scope` is too weak for tools that need contextual narrowing
2. examples include:
   - implied filesystem roots
   - narrowed URL prefix usage
   - data-subscope restrictions
3. static sandbox policy is intentionally too coarse and too lifecycle-bound to carry every per-context adjustment

Implemented design:

1. extend `ToolsScope` with per-tool `tool_constraints`
2. extend `ToolsView` with resolved effective constraints
3. keep the same scope-stack model:
   - `set`
   - `add`
   - `pop`
   - `reset`
4. let hosted approval optionally return `scope_constraints`
5. on `add_to_scope`, persist both:
   - ungating
   - tool constraints
6. native `Toolbox.execute(...)` now applies a minimal `argument_policy` slice:
   - `implied_args` fill missing tool arguments
   - `locked_args` reject conflicting overrides before tool execution
   - `normalizers` now support a first shared domain-aware subset:
     - `path_under_implied_root`
     - `url_under_implied_prefix`
   - kwargs-capable tools now receive:
     - resolved `tool_constraints`
     - current `tools_view`
     - `tool_constraints_view` helper
       The helper currently exposes `resolve_argument(...)`, `resolve_filesystem_root(...)`, and `resolve_url(...)`.

Security split:

1. static sandbox policy stays the hard outer boundary
2. dynamic constraints become the fine-grained contextual narrowing layer
3. brokered execution still enforces the outer sandbox policy

### 2.3 Phase 3: Guide Policy

Status: unresolved design question

What exists already:

1. guides can be surfaced separately from main tools
2. hosted/native visibility model already distinguishes executability from presentation

What is missing:

1. explicit trust model for guides when the paired tool is gated
2. decision on stripped-sandbox vs safe in-proc guide execution

## 3. Main Risks

Current implementation risk is no longer foundational breakage. The main risk is semantic inconsistency if gated tools are added without a strict precedence model.

Highest-risk areas:

1. disabled vs gated precedence
2. hidden vs gated presentation
3. whether every hosted wrapper path consistently provides a stable scope target for `add_to_scope`
4. keeping the new constraint layer generic without turning it into a second ad-hoc policy engine
5. guide execution trust model
6. public approval callback ergonomics

## 4. Recommended Next Step

Recommended next implementation step:

1. decide how much of `normalizers` should stay generic vs tool-specific helper code
2. add one or two app-level examples that consume persisted constraints explicitly
3. decide whether path/url provenance should be surfaced to tools or only enforced silently
4. keep the Windows callback relay fix and the approval slices covered in regression runs

The hosted approval callback is now stable enough to persist contextual narrowing without re-asking on every gated call. The remaining work is to refine how far the shared normalizer layer should go before tool-specific helper APIs take over.

## 5. Key References

1. [sandbox_architecture.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_architecture.md)
2. [sandbox_plan.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_plan.md)
3. [sandbox_test_status.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_test_status.md)
