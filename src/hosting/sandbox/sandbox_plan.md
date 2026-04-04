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

## 2. Next Steps

### 2.1 Immediate Next Steps

1. Add live worker liveness probing into consistency/review.
   - detect dead-but-registered executors
   - make review/reconcile catch the class of failure seen when a pipe exists in state but the worker is gone

2. Keep compact operator UX consistent everywhere.
   - review any remaining operator outputs for raw internal-state leakage by default
   - preserve `details=true` for deep diagnostics

3. Decide whether to add `toolbox.cancel`.
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

1. Linux backend
2. broader app/runtime adoption beyond the current hosted demo slice
3. callback-contract refinement if a concrete need appears

## 3. Suggested Priority Order

1. worker liveness in review/consistency
2. small operator UX polish only if still needed after that
3. stop and review whether env/provenance depth is worth more investment
4. only then take on larger backend/platform work

## 4. Why This Priority

1. liveness gaps are the most practical remaining operational risk
2. operator UX is already mostly good, so only small targeted polish should remain
3. deeper env/provenance work can become expensive quickly
4. Linux and bigger runtime adoption are meaningful, but not needed to validate the current Windows-first hosted-toolbox architecture
