# Sandbox Plan

Date: 2026-04-05
Scope: forward-looking plan after the foundational sandbox/toolbox build-out is largely complete.

## 1. Completed Foundation

Completed at a coarse level:

1. dedicated sandboxed toolbox executor path
2. manifest-driven staged toolbox revisions
3. hosted logical-toolbox routing across multiple sandbox profiles
4. persistent register/unregister lifecycle
5. compact operator workflow:
   - review snapshot
   - repair
   - reconcile
   - gc
6. hosted chat/runtime slice and non-demo hosted attach path
7. hosted/native access-control alignment first slice
8. coarse executor-level cancellation
9. callback contract first slice:
   - generic hosted callbacks
   - brokered fs/http follow-through with per-call attribution context
10. Linux backend validation in WSL shared-shadow tests

This means the next plan should no longer focus on “make sandboxed hosting real.” It should focus on the next client-facing promotion of tool execution semantics.

## 2. Active Plan: Gated Tools

The next active investment is a gated execution state between “allowed” and “disabled”.

Target user-facing outcome:

1. a tool may be advertised or hidden as usual
2. a tool may still be executable only with explicit caller approval
3. a hosted client can approve:
   - deny
   - allow once
   - add permission scope for future calls
4. disabled tools still override gating entirely

## 3. Phase 1: Semantic Gated State

Goal: make gated state part of the native `Toolbox` contract before adding interactive approval.

### 3.1 Data Model

Add gated state to the native semantic layer:

1. extend `ToolsView`
2. extend `ToolsScope`
3. define the effective-view merge rules

Required precedence:

1. disabled wins over everything
2. hidden controls advertisement, not executability
3. gated controls executability, not visibility by itself

Open point to preserve explicitly:

1. hidden + gated is valid
2. visible + gated is valid
3. disabled + gated behaves as disabled

### 3.2 Native Execution Contract

Extend native gating so `Toolbox.gate_call(...)` can return a distinct gated outcome rather than collapsing it into deny.

Target outcomes:

1. `allowed`
2. `blocked_in_scope`
3. `gated_requires_confirmation`
4. existing backend-specific hosted denials remain hosted-only extensions

### 3.3 Hosted Parity

Hosted execution must reuse the native gated semantics rather than invent a second model.

Required surfaces:

1. hosted `toolbox_gate(...)`
2. hosted `toolbox_execute(...)`
3. hosted summaries and describe output
4. `/t`
5. `/t sc`
6. prompt-shaping / hosted visible-tool filtering

### 3.4 Presentation Contract

Decide how gated tools should appear:

1. advertised + gated
2. hidden + gated
3. disabled + gated suppressed to disabled

At the end of Phase 1, the system should be able to report gated tools consistently even if no interactive approval exists yet.

## 4. Phase 2: Interactive Hosted Approval

Goal: use the callback contract to let hosted clients decide gated-tool execution in real time.

### 4.1 Approval Flow

When a gated tool is about to execute, hosted code should be able to ask the client for a decision.

Supported decisions:

1. `deny`
2. `allow_once`
3. `add_to_scope`

The recommended first implementation is not “block the tool call indefinitely inside execution.” It is:

1. detect gated outcome
2. invoke hosted approval callback
3. apply returned decision
4. either continue execution or return a gated failure/result

### 4.2 Scope Mutation Contract

Approval decisions need a precise state target.

Required rules:

1. `allow_once` is per-call only and does not mutate future scope
2. `add_to_scope` updates the active request/session scope used for future calls
3. once the gating bit is removed from effective view/scope, execution becomes fixed until gating is re-enabled

### 4.3 Multi-Call / Multi-Round Behavior

Need an explicit contract for repeated gated calls.

Decisions required:

1. dedupe policy for repeated gated calls in one tool round
2. whether dedupe is per tool or per tool+arguments
3. timeout/default-deny policy when client does not answer

Recommended default:

1. dedupe by tool name per round
2. default deny on timeout

### 4.4 Dynamic Fine-Grained Constraints

The first approval slice solved binary execution approval, but not contextual
execution constraints such as implied roots, allowed URL prefixes, or dataset
sub-scopes.

The recommended next step is not a separate parallel grant framework. It is:

1. extend native `ToolsScope`
2. extend native `ToolsView`
3. persist per-tool dynamic constraint payloads in the same scope stack model

Recommended shape:

1. keep existing scope stack semantics:
   - `set`
   - `add`
   - `pop`
   - `reset`
2. add `tool_constraints` keyed by tool name
3. materialize effective constraints into `ToolsView`
4. let hosted approval optionally return `scope_constraints`
5. on `add_to_scope`, store both:
   - tool ungating
   - effective per-tool constraints for future calls
6. first helper slice is now in place in native `Toolbox.execute(...)`:
   - `argument_policy.implied_args`
   - `argument_policy.locked_args`
   - kwargs-capable tools receive resolved `tool_constraints`, `tools_view`, and `tool_constraints_view`
     with convenience accessors such as `resolve_argument(...)`, `resolve_filesystem_root(...)`, and `resolve_url(...)`

Design intent:

1. minimize API surface expansion
2. avoid pushing dynamic path/url/data policy into static sandbox policy
3. preserve one logical global tool repository while still supporting per-context implied arguments and narrowing rules

Constraint payloads should stay generic and domain-oriented rather than broker-specific.

Suggested envelope:

```json
{
  "domains": {
    "filesystem": {...},
    "network": {...},
    "data": {...}
  },
  "argument_policy": {
    "implied_args": {...},
    "locked_args": [...],
    "normalizers": {...}
  }
}
```

Security model:

1. static sandbox policy remains the hard outer boundary
2. dynamic scope constraints are the finer contextual narrowing layer
3. tool/runtime helpers apply implied/locked argument behavior before brokered execution
4. the first shared `normalizers` slice now covers:
   - `path_under_implied_root`
   - `url_under_implied_prefix`
5. brokered host policy remains the final physical stop

## 5. Phase 3: Guides And Safe Execution Policy

The current feature idea needs a stricter guide policy decision.

Question to resolve:

1. should guides execute freely even when the paired tool is gated?

Likely safe answers:

1. guide content is static/lightweight and can execute freely
2. or guide execution moves to stripped sandbox / safe in-proc path

Avoid:

1. treating arbitrary Python guide code as always safe in-proc just because the tool itself is gated

## 6. Phase 4: Tests And Documentation

Required test slices:

1. native gated precedence tests:
   - visible + gated
   - hidden + gated
   - disabled + gated
2. hosted parity tests:
   - `toolbox_gate`
   - `toolbox_execute`
   - `/t`
   - `/t sc`
   - prompt shaping
3. approval-flow tests:
   - deny
   - allow once
   - add to scope
   - timeout
   - repeated gated calls in one round
4. guide-policy tests once guide behavior is chosen

Required doc updates:

1. architecture
2. test status
3. user-facing wrapper guidance if approval callbacks become public app contract

## 7. Open Design Questions

These should be kept explicit while implementing:

1. Should gated tools be advertised normally by default, or visually distinguished in every presentation surface?
2. Should approval be keyed by tool name only, or tool name plus arguments/profile?
3. What exact callback payload should be sent for an approval decision?
4. Should native non-hosted callers also receive the same gated outcome, even without interactive approval?
5. What is the guide execution trust model?

## 8. Suggested Execution Order

1. Phase 1 semantic gated state
2. Phase 1 presentation parity in hosted/native surfaces
3. Phase 2 interactive hosted approval
4. Phase 2 dedupe / timeout / scope-mutation rules
5. Phase 3 guide execution policy
6. final doc/test consolidation
