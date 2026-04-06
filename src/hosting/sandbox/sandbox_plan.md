# Sandbox Plan

Date: 2026-04-04
Scope: coarse completion map for sandbox/toolbox work, with emphasis on what is done and what should happen next.

## 1. Done

Completed at a coarse level:

1. dedicated sandboxed toolbox executor path
2. manifest-driven staged toolbox revisions
3. hosted logical toolbox routing across multiple sandbox profiles
4. persistent register/unregister lifecycle
5. named environment description and realized-venv lifecycle
6. brokered fs/http execution paths
7. hosted chat/runtime slice
8. toolbox call-gating first slice
9. compact operator workflow:
   - review snapshot
   - repair
   - reconcile
   - gc
10. thin-client / remote hosted-toolbox proxy path
11. Linux backend validated in a WSL shared-shadow environment
12. first broader app/runtime adoption slice:
   - `mp13chat` can now attach to an already provisioned hosted toolbox via explicit host state files and toolbox id
   - wrappers can now reuse the same attach flow through `attach_existing_hosted_toolbox(...)`

## 2. Next Steps

### 2.1 Immediate Next Steps

1. Restore native `Toolbox` access-control parity in hosted execution.
   - hosted execution must enforce request-scoped `ToolsView` decisions before sandbox dispatch
   - `ToolsScope.disabled_tools`, scoped mode overrides, and native `blocked_in_scope` semantics must survive hosted mode
   - hosted gating should extend native `Toolbox.gate_call(...)`, not replace it with a second incompatible policy model
   - the target contract is:
     - native `Toolbox` remains the source of truth for visibility/executability semantics
     - hosted execution adds backend-specific outcomes such as `unavailable_backend` and sandbox policy denials
     - prompt shaping, `/t`, `/t sc`, and direct hosted execution should all reflect the same effective view

2. Add hidden/silent parity for hosted user tools.
   - hosted auto/manual user tools currently lack the full native hidden-but-allowed model
   - staged/persisted hosted toolbox state should carry hidden user-tool membership, not only hidden intrinsics
   - hosted `describe` should separate:
     - all registered tool names
     - advertised tool names
     - hidden-but-allowed tool names

3. Add live worker liveness probing into consistency/review.
   - detect dead-but-registered executors
   - make review/reconcile catch the class of failure seen when a pipe exists in state but the worker is gone

4. Keep compact operator UX consistent everywhere.
   - review any remaining operator outputs for raw internal-state leakage by default
   - preserve `details=true` for deep diagnostics

5. Lock down gated tool-call completeness and hosted integration.
   - add a broader real-chat integration test around `/t` and `/t sc` presentation
   - verify hosted-gated tools, hosted hidden-but-allowed tools, and hosted advertised tools present consistently across:
     - prompt shaping
     - `/t`
     - `/t sc`
     - direct hosted gate/execute
   - audit any remaining non-chat hosted consumers for coarse full-membership fields or bypassed request-scoped `ToolsView`

6. Decide whether to add `toolbox.cancel`.
   - only worth doing if real hosted tool calls can be long-running enough that cancellation matters operationally

### 2.2 Near-Term Improvements

1. Strengthen env/provenance rigor.
   - current env flow is usable
   - next real step would be a stronger immutable resolver/lock policy

2. Strengthen GC/reference tracking.
   - current state is coherent
   - next step is deeper reference-tracked cleanup semantics if needed

3. Broaden long-lived server automation guidance.
   - `HostedToolboxAdmin` exists
   - may still want stronger admin/runbook guidance once liveness probing lands

### 2.3 Medium-Term Work

1. broader app/runtime adoption beyond the current hosted demo slice and the new direct `mp13chat` attach path
2. callback-contract refinement if a concrete need appears

## 3. Suggested Priority Order

1. native/hosted access-control parity
2. hosted hidden/silent parity for user tools
3. worker liveness in review/consistency
4. gated tool-call completeness and hosted integration validation
5. small operator UX polish only if still needed after that
6. stop and review whether env/provenance depth is worth more investment
7. only then take on larger backend/platform work

## 4. Why This Priority

1. access-control mismatch is the biggest semantic risk because it can make hosted execution disagree with the original `Toolbox` contract
2. hidden/silent parity matters because the native design explicitly separates visibility from executability for all tools, not just intrinsics
3. liveness gaps are the next practical operational risk once semantic parity is fixed
4. gated tool-call completeness needs explicit validation because chat presentation can still drift from hosted execution even after helper-level parity fixes
5. operator UX is already mostly good, so only small targeted polish should remain after the core contract is aligned
6. deeper env/provenance work can become expensive quickly
7. Linux and bigger runtime adoption are meaningful, but not needed to validate the current Windows-first hosted-toolbox architecture
