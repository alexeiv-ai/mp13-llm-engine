# Sandbox Plan

Date: 2026-04-06
Scope: forward-looking follow-up after the major sandbox/toolbox semantics are implemented.

This file is now intentionally short.

The settled model lives in [sandbox_architecture.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_architecture.md).
Use this file only for the remaining forward plan.

## 1. Already Landed

These are no longer active design items:

1. hosted sandboxed toolbox execution
2. gated tool semantics in native and hosted flows
3. hosted approval callbacks with:
   - `deny`
   - `allow_once`
   - `add_to_scope`
4. dynamic per-tool `tool_constraints`
5. first shared constraint helpers:
   - implied args
   - locked args
   - `path_under_implied_root`
   - `url_under_implied_prefix`
6. kwargs-facing `tool_constraints_view`
7. static-content-only guide execution
8. shallow partial-merge support for `tool_constraints` across stacked scopes
9. explicit constraint-clear marker via `tool_constraints={tool_name: None}`

## 2. Active Follow-Up

The next active work is polish, not a new subsystem.

### 2.1 Constraint Helper Boundary

Need to decide how far the shared helper layer should go.

Open items:

1. which future `normalizers` should stay generic
2. which behaviors should move into tool-specific helper code instead
3. whether path/url provenance should be surfaced back to tools
4. whether the current shallow-merge plus replay semantics need anything beyond the new explicit clear marker

### 2.2 Reference Examples

Add one or two compact examples of the intended usage pattern:

1. hosted approval returning `scope_constraints`
2. a kwargs-capable tool using `tool_constraints_view`

### 2.3 Wrapper Polish

Keep wrapper behavior explicit and coherent:

1. runtime helper path remains the recommended persistent-approval entrypoint
2. direct hosted-ref execution must keep documenting its `callback_context` persistence requirement

## 3. Validation

Keep the main hosted regression slice green:

```powershell
pytest -q tests/test_hosted_tool_visibility.py tests/test_hosting_toolbox_sandbox.py tests/test_mp13chat_hosted_toolbox_api.py
```

For Linux validation, keep using the WSL shadow-root flow documented in [sandbox_architecture.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_architecture.md) and [sandbox_test_status.md](/o:/repos/mp13-llm-engine/src/hosting/sandbox/sandbox_test_status.md).
