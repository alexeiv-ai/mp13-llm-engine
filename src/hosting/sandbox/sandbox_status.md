# Sandbox Status

Date: 2026-04-05
Purpose: fresh status snapshot for the new gated-tool roadmap.

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

So the active work is no longer “finish the sandbox foundation”. The active work is the next semantic feature:

1. gated tools
2. hosted approval flow for gated tools

## 2. Active Plan Status

### 2.1 Phase 1: Semantic Gated State

Status: not started

What exists already:

1. `disabled` and scoped-deny semantics exist
2. hidden vs advertised semantics exist
3. hosted/native gate surfaces already support distinct gate outcomes

What is missing:

1. gated state in `ToolsView`
2. gated state in `ToolsScope`
3. native `gated_requires_confirmation` outcome
4. hosted/native presentation of gated state

### 2.2 Phase 2: Interactive Hosted Approval

Status: blocked on Phase 1 semantics

What exists already:

1. hosted callback relay
2. generic callback processor contract
3. per-call callback context:
   - toolbox id
   - tool name
   - tool call id
   - tool arguments
   - caller context

What is missing:

1. approval callback schema
2. allow-once semantics
3. add-to-scope semantics
4. dedupe and timeout rules

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
3. per-round repeated gated calls
4. scope mutation semantics for `allow_once` vs `add_to_scope`
5. guide execution trust model

## 4. Recommended Next Step

Implement Phase 1 only:

1. add gated state to `ToolsView` and `ToolsScope`
2. define precedence rules explicitly
3. extend native `Toolbox.gate_call(...)`
4. mirror the outcome through hosted gate/execute and `/t` presentation

Do not start the interactive approval callback until that semantic base is stable.

## 5. Key References

1. [sandbox_architecture.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_architecture.md)
2. [sandbox_plan.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_plan.md)
3. [sandbox_test_status.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_test_status.md)
