# Sandbox Status

Date: 2026-04-06
Purpose: concise status snapshot after the gated-tool, hosted-approval, dynamic-constraint, and guide-hardening slices.

## 1. Current Position

The sandbox/toolbox foundation is complete enough that the main work is no longer “make hosted sandboxing real.”

Implemented and validated:

1. hosted sandboxed toolbox execution
2. logical-toolbox routing across sandbox profiles
3. hosted/native gated-tool semantics
4. hosted approval callbacks:
   - `deny`
   - `allow_once`
   - `add_to_scope`
5. dynamic per-tool scope constraints through `ToolsScope` / `ToolsView`
6. shared execution helpers for implied args, locked args, and first normalizers
7. kwargs-facing `tool_constraints_view` helper injection
8. static-content-only guide execution
9. Windows and WSL/Linux callback validation

## 2. Settled Semantics

The detailed reference is now [sandbox_architecture.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_architecture.md).

Settled points:

1. disabled overrides gated
2. hidden controls advertisement, not executability
3. `add_to_scope` persists only through a durable `ToolBoxRef`
4. dynamic contextual narrowing lives in `tool_constraints`, not static sandbox-policy mutation
5. guides are separate tools and do not inherit parent-tool gating implicitly
6. all guides are static-content-backed only

## 3. Wrapper Consistency

Current public wrapper behavior:

1. `execute_tool_round_on_cursor(...)` auto-forwards a durable scope target
2. direct `HostedToolBoxRef.execute(...)` persists `add_to_scope` only when the caller passes:
   - `callback_context["toolbox_ref"]`
   - or `callback_context["cursor"]` whose context owns a `toolbox_ref`
3. `create_hosted_toolbox_executor(...)` builds the harness only; it does not auto-supply persistence context

## 4. Remaining Work

The remaining work is polish and consolidation, not a new semantic subsystem.

Recommended next items:

1. decide how much more of `normalizers` should stay generic vs tool-specific
2. add one or two compact app/reference examples for scoped tools
3. decide whether path/url provenance should be surfaced to tools or only enforced silently
4. keep the main hosted regression slice passing

## 5. Key References

1. [sandbox_architecture.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_architecture.md)
2. [sandbox_plan.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_plan.md)
3. [sandbox_test_status.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_test_status.md)
