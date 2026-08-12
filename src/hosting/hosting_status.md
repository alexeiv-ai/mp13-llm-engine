# Unified hosting cutover status

Last updated: 2026-08-12

Status: R7.2 active on durable candidate records

This is the current execution ledger for
[`hosting_access_plan.md`](hosting_access_plan.md). Detailed completed-slice
narratives were intentionally removed to keep resume context small; they remain
available in Git history. This file records only the current gate, open work,
compact evidence index, and external completion gates.

## 1. Current gate and resume order

**R7.2 is active at the durable candidate-record boundary.** R7.1 planning and selective revision is complete:
generic locks/requests, CAS changes, evidence, child replanning, exact public
projections, and stale/retry/restart/concurrency safety proofs are committed.
The strict v3 candidate retention/quota policy, durable candidate repository,
pre-publication rollout seam, candidate operation kinds, and shared validated
candidate preparation, bounded get/renew, and ordinary gated candidate
execution are committed. Next, wire exact publication, discard, expiry cleanup,
and restart reconciliation.

Resume in this order:

1. If no equivalent post-`757c13f` receipt exists, run once:
   `python -m pytest tests/test_workflow_helper_service.py -q`.
2. Finish remaining R3.2–R4/R6 P0 cuts and focused proofs.
3. Implement R7.1/R7.2 against the frozen tool-change/candidate design.
4. Complete R9 public surfaces, removals, permanent docs, dependent adoption,
   and final matrices.

Do not run the repository-wide aggregate at resume. Its last diagnostic run
stopped after 509 passes and 100 expected legacy-fixture failures; rerun it only
after R9.4 removals make R9.7 evidence meaningful.

## 2. Progress

| Work | Status | Remaining boundary |
|---|---|---|
| R0 Contract freeze | Complete | Final tool-change schemas were revised in place before adoption. |
| R1 Inventory/handoff | Complete | Refresh removal scans and dependent receipt at R9. |
| R2 Shared paths/setup | Complete | Final aggregate/platform proof at R9.7. |
| R3 Unified configuration | Complete | Final aggregate/platform proof at R9.7. |
| R4 Single startup path | Complete | Final aggregate/platform proof at R9.7. |
| R5 Generic packages | Complete | Final aggregate/platform proof at R9.7. |
| R6 Generic environments | Active | Remove final legacy receipt/reference readers; add candidate lifecycle policy fields in R7.2. |
| R7 Toolbox adoption | R7.1 complete | R7.2 durable candidate/materialization lifecycle. |
| R8 Worker-neutral state | Complete | One focused workflow-helper checkpoint rerun remains as noted above. |
| R9 Acceptance/handoff | Partial | Public signature alignment, lifecycle matrices, removals, docs, dependent receipt, and full/platform lanes. |

The plan progress table records completed blocks; its remaining-work checkboxes
are authoritative for unfinished item-level work.

## 3. Frozen steering relevant to resume

- Keep `hosting.control.v3`, `hosting.toolbox.definition_plan.v2`, and
  `hosting.toolbox.confirmation_receipt.v1`; do not add another version solely
  for the unadopted field-shape revision.
- The host owns atomic add/update/rename/remove batches, per-tool import/package
  analysis, and immutable child replanning after selective rejection.
- The candidate lifecycle reuses ordinary `toolbox_worker` candidate routing:
  prepare, get, renew, execute, publish, and discard. It is not a dry run.
- Candidate preparation/renewal accepts a requested lifetime within host policy.
  In-flight execution leases prevent expiry cleanup from retiring resources
  underneath a long-running tool; ordinary execution timeout remains decisive.
- `HOSTING_CLIENT_BREAKING_CHANGES.md` contains the exact consumer payloads.
  Permanent docs remain at shipped behavior until their owning production slice
  lands.

## 4. Open implementation focus

### P0 before R7

The pre-R7 P0 clean-cut items are complete.

### R7.2 candidate lifecycle

- Expose exact publication and discard through authorized
  daemon/channel/CLI surfaces without adding a worker kind or generic endpoint.
- Add restart reconciliation, resource cleanup, stale-pin revalidation, and
  exact warmed-candidate publication.
- Enforce the same execution gates, sandbox, host-API/data/network approvals,
  callbacks, audit, timeout, and cancellation policy as active tools.
- Add `toolbox_candidate_retention_ms` and
  `toolbox_candidate_limit_per_actor` to strict host-local lifecycle policy.

### R9 closeout

- Align channel/CLI/capabilities and typed client methods with the final handoff.
- Complete lifecycle/no-double/security/removal/documentation tests.
- Update permanent docs only when behavior is shipped.
- Obtain dependent implementation pin/receipt, then run required aggregate and
  native/platform lanes.

## 5. Compact evidence index

| Evidence | Result retained |
|---|---|
| DOC-R0/R1 | Control-v3 contract, inventory, consumer handoff, and doc ownership frozen. |
| CODE-R2 | Shared roots, logical labels, safe host-local setup/journal implemented. |
| CODE-R3.1/R4A | Strict unified configuration and single-path startup implemented; focused startup/security lanes passed. |
| TEST-R3.2 | Legacy authority scan leaves only explicit negative/removal evidence; strict configuration/startup/setup lane passed 27 tests. |
| CODE/TEST-R4.1 | Unified revision pins cover toolbox plans and all hosted-operation creation paths; plan/operation lanes passed 67 tests and helper operations passed 116 tests. |
| CODE/TEST-R4.2 | Removed startup flags/settings fail fast; app attachment uses only the top-level config input; client/startup lane passed 67 tests. |
| CODE-R5 | Generic package ingress, daemon hashing, locks, commands, and denial paths passed focused coverage. |
| CODE-R6A–D | Generic environment contracts/manager/commands and legacy-root cut passed focused coverage. |
| CODE/TEST-R6.5 | Generic lock/receipt/reference validation replaced legacy readers and builder references; lifecycle/runtime lanes passed 27 tests, atomic publication 7, and maintenance 15. |
| CODE-R7A/B | Generic materializer/package/reference bridges and atomic candidate publication remain valid foundations; old plan/confirmation field assertions are provisional. |
| CODE/TEST-R7.1A | Planning persists strict generic lock/request records and apply consumes the selected record without late lock creation; package/plan/rollout lane passed 29 tests and atomic routing passed 10. |
| CODE/TEST-R7.1B | Strict CAS tool-change merge supports atomic add/update/rename/remove batches and deterministic complete-definition change IDs; focused lane passed 12 tests. |
| CODE/TEST-R7.1C | Tool-change planning is exposed through authorized daemon/channel/CLI/reference surfaces and persists normalized proposal metadata; combined focused lane passed 109 tests (one removed-constructor fixture deselected). |
| CODE/TEST-R7.1D | Immutable plans persist bounded per-change import evidence, mapped distributions, environment grouping, preferred exact package mutations, and approval flags; combined focused lane passed 110 tests. |
| CODE/TEST-R7.1E | Authorized selective revision validates complete decisions/evidenced denials, preserves active exclusions, cascades dependents, and fully replans parent-bound immutable children (including empty results); expanded lane passed 143 tests. |
| CODE/TEST-R7.1F | Public plans enforce the frozen exact v2 projection with nested change IDs; confirmation emits the exact v1 projection or revision-required failure without a receipt; expanded lane passed 153 tests. |
| CODE/TEST-R7.1G | Planned artifacts are rehashed at confirmation; stale approvals do not consume, same-request restart retry is idempotent, changed requests fail, and materialization failure acquires no reference/spawns no worker; expanded lane passed 156 tests. |
| CODE/TEST-R7.2A | Strict v3 lifecycle policy normalizes bounded candidate retention/quota defaults and exposes sanitized health; configuration/startup/setup lane passed 31 tests. |
| CODE/TEST-R7.2B | Actor-bound candidate records persist bounded projections, repeated idempotent renewal, quotas, terminal states, and expiry-deferring execution leases across restart; focused lane passed 25 tests. |
| CODE/TEST-R7.2C | Rollout preparation is split before publication while one-shot apply preserves atomic routing and failure cleanup; rollout/atomic lane passed 11 tests. |
| CODE/TEST-R7.2D | Candidate prepare/publish/discard have distinct durable operation kinds and strict phase sets; operation contract/repository lane passed 55 tests. |
| CODE/TEST-R7.2E | One-shot apply and candidate preparation share actor/revision/pin/exact-approval validation; selected generic records remap to confirmed profile IDs, preparation persists the exact warmed payload without publication, retry is stable, and approval cannot be reused; focused lane passed 21 tests (one recorded legacy daemon fixture deselected). |
| CODE/TEST-R7.2F | Authorized prepare/get/renew daemon, channel, and CLI surfaces expose only bounded actor-scoped records; renewal is idempotent and revalidates active/config/catalog/host-policy/source pins plus candidate workers; combined transport/auth/service/repository/operation lane passed 89 tests with no deselections. |
| CODE/TEST-R7.2G | Candidate execution permits only changed-profile routes, selects the retained candidate worker, forwards ordinary tool/effect/callback/host-API/timeout policy into durable toolbox execution, and holds a lease until terminal cleanup; service/transport/auth/repository lane passed 57 tests. |
| CODE-R8A–C | Versioned neutral state, Python/JS shared manager adoption, GC/repair controls passed focused coverage. |
| CODE/TEST-R9 partial | Structured auth, role/hash authority, redaction, startup modes, generic lifecycle, and retry/restart identity passed focused coverage. |
| TEST-R8.2C checkpoint | Workflow operation suite passed; full helper suite passed 116 tests on resume. |

## 6. External completion gates

- Parent implementation pin and dependent owner/revision/test receipt.
- Local real-daemon R7 acceptance currently requires the unavailable
  `typing_inspection` fixture dependency; rerun in the declared dependency lane.
- Required Windows/POSIX and relay-equivalent lanes.
- Final secret/path/process-argument redaction proof.
- R9.7 aggregate, lint, type, and required native/platform results.

## 7. Ledger rules

1. Record one active slice before production changes.
2. Update code, focused tests, owning permanent docs, handoff, plan checkbox, and
   one concise evidence row together when they form one contract slice.
3. Preserve the clean cut: no aliases, fallbacks, dual reads/writes, automatic
   legacy migration, or dependent-repository edits.
4. Keep this file current and compact; completed detail belongs in Git history,
   not an ever-growing transcript.
