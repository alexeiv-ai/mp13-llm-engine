# Unified hosting cutover status

Last updated: 2026-08-12

Status: paused at clean commit `757c13f`; consumer-contract steering is
documented but not implemented

This is the current execution ledger for
[`hosting_access_plan.md`](hosting_access_plan.md). Detailed completed-slice
narratives were intentionally removed to keep resume context small; they remain
available in Git history. This file records only the current gate, open work,
compact evidence index, and external completion gates.

## 1. Current gate and resume order

No slice is active while paused. The working tree intentionally changes only:

- `hosting_access_plan.md` — execution/design plan;
- `HOSTING_CLIENT_BREAKING_CHANGES.md` — exact external-consumer handoff; and
- this status ledger.

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
| R3 Unified configuration | Active | Remove remaining old-file fixtures/docs and close aggregate proof. |
| R4 Single startup path | Active | Pin configuration revision in long operations; finish removed flag/setting scans. |
| R5 Generic packages | Complete | Toolbox planning origin moves to generic locks in R7.1. |
| R6 Generic environments | Active | Remove final legacy receipt/reference readers; add candidate lifecycle policy fields in R7.2. |
| R7 Toolbox adoption | R7.0 design complete; implementation pending | R7.1 planning/revision and R7.2 candidate/materialization work. Existing package/reference/publication bridges remain useful. |
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

- R3.2: eliminate remaining `access_control.json` readers/fixtures/docs.
- R4.1: capture stable configuration revision in long-running plans/operations.
- R4.2: finish removal of old startup flag/setting/launcher vocabulary.
- R6.5: remove final legacy receipt/reference readers and aliases.

### R7.1 planning and selective revision

- Plan exact generic package locks and `EnvironmentRequest` records directly.
- Implement server-side tool change sets, deterministic change IDs, per-tool
  import evidence, and atomic rename.
- Implement immutable child plans with full closure/lock recomputation; never
  truncate an existing lock.
- Replace provisional field-level plan/confirmation assertions while retaining
  the existing record identifiers.

### R7.2 candidate lifecycle

- Split existing candidate preparation from publication without adding a worker
  kind or generic code-execution endpoint.
- Implement durable candidate records, bounded read, requested lifetime,
  renewal, in-flight execution leases, restart reconciliation, quotas, expiry,
  discard, and exact warmed-candidate publication.
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
| CODE-R5 | Generic package ingress, daemon hashing, locks, commands, and denial paths passed focused coverage. |
| CODE-R6A–D | Generic environment contracts/manager/commands and legacy-root cut passed focused coverage. |
| CODE-R7A/B | Generic materializer/package/reference bridges and atomic candidate publication remain valid foundations; old plan/confirmation field assertions are provisional. |
| CODE-R8A–C | Versioned neutral state, Python/JS shared manager adoption, GC/repair controls passed focused coverage. |
| CODE/TEST-R9 partial | Structured auth, role/hash authority, redaction, startup modes, generic lifecycle, and retry/restart identity passed focused coverage. |
| TEST-R8.2C checkpoint | Workflow operation suite passed; helper suite recorded 114 passes plus both fixes passing together. One full 116-test focused rerun remains. |

## 6. External completion gates

- Parent implementation pin and dependent owner/revision/test receipt.
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
