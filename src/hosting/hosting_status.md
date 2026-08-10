# Hosting toolbox corrective status

Last updated: 2026-08-09

## Overall status

Status: Active corrective work

The complete-definition migration remains adopted, but the environment/template
implementation is not accepted as complete. A post-acceptance audit found that
tests proved isolated planner and builder components without proving the real
daemon/control-channel bridge. The corrected plan now also requires an explicit
multi-tool package-confirmation protocol and removes all superseded compatibility
paths because the product is unreleased. The corrective scope is defined in
[hosting_access_plan.md](hosting_access_plan.md).

## Retained completed baseline

The following foundations remain implemented and are not reopened wholesale:

- complete toolbox definition read/plan/approval/durable-apply APIs;
- canonical definition, profile, environment, manifest, and lock identities;
- source import analysis and reviewed import/distribution mapping;
- strict immutable plans, approval bindings, and digest-validated toolbox state;
- non-inheriting hermetic environment builder primitives with exact wheel,
  receipt, import-probe, quarantine, reference, and GC behavior;
- candidate rollout and atomic complete route publication;
- removal of version-1 toolbox mutation/environment APIs;
- dependent adoption of the complete-definition protocol.

Release/adoption pins retained for traceability:

- parent release adopted by the dependent:
  `83b35e20604c8f0c2fbe27467980b6a49385d918`;
- `mp13-docs` adoption commit:
  `125d20f232bf5b755d18c1b23bc1e4b8929edf21`;
- breaking-change handoff reset commit: `99b79e0`;
- stale Python-node fallback removal commit: `3752118`.

## Confirmed open gaps

1. Custom dependency planning does not produce the complete resolved lock and
   artifacts consumed by the real builder.
2. Shipped template artifact references are not a real per-distribution wheel
   closure, and normal daemon startup does not configure the real materializer.
3. Built-ins are not constructed for the daemon's current target during real
   hosting setup.
4. ARM64 and macOS target detection/validation/support are absent.
5. Persisted manifest normalization, identical reapply, candidate identity, and
   concurrent healing are unsafe or incomplete.
6. Repair reports missing registrations but cannot safely heal them.
7. Abrupt POSIX daemon death can leave toolbox workers outside daemon ownership.
8. Dependency approval authority does not match the distinct role described by
   the contract.
9. Plans do not present exact direct/transitive package mutations or bounded
   configured-source alternatives, and no consumer confirmation/skip receipt
   exists for multi-tool changes.
10. Hosting configuration does not own built-in intent, source modes/revisions,
    air-gap ingestion, or non-built-in environment retention/removal.
11. No real-daemon end-to-end test covers built-in setup, multi-tool package
    confirmation, custom package add/removal, or restart healing.
12. **Blocking interactive/network API defect.** Plan and apply preflight run
    synchronously; GC/repair/reconcile, duplicate toolbox-execution attach,
    worker describe, and hosted cancellation can hold the client while waiting
    on IPC, process, lock, or filesystem work. Adding HTTPS resolution to that
    path would worsen the defect, and no request may remain open for human
    confirmation/approval. `op-start`/`op-status` exist but currently use a
    separate 200-entry daemon snapshot store without canonical hosted-operation
    restart recovery. Workflow/proxy streams are in-memory and unsuitable for
    durable management progress.
13. Online and air-gapped package ingress do not yet converge on a configured,
    verified content-addressed wheel store through durable progress-reporting
    operations.

## Progress ledger

Only corrective work is tracked here. The former phase-by-phase historical test
transcript was intentionally removed because it obscured current truth.

| Work group | Status | Outcome/evidence |
| --- | --- | --- |
| R0 Corrective contract baseline | Active | R0-01 is complete and was corrected on 2026-08-09 to prohibit dependent edits, remove compatibility preservation, reference production seams, specify package ingress and confirmation/config behavior, file the blocking interactive/network defect, select the generic operation façade, allocate average/medium/high work with code guidance, and require checkbox-bearing commits at every slice/expertise boundary. R0-02 and R0-03 remain cross-slice obligations. No runtime behavior changed. |
| R1 Canonical current-host target | Pending | No implementation started. |
| R2 Revisioned hosting configuration and built-ins | Pending | No implementation started. |
| R3 Multi-tool planning and consumer confirmation | Pending | No implementation started. |
| R4 Privileged approval and immutable apply | Pending | No implementation started. |
| R5 Removal, retention, and administrator environments | Pending | No implementation started. |
| R6 Restart-safe consumer healing | Pending | No implementation started. |
| R7 Breaking-change handoff and acceptance | Pending | No implementation started. |

## Next implementation slice

No code slice is active. The next declared slice is R0-03 (`medium`): replace
the breaking-change reset marker with the exact consumer/admin migration for
the first client-visible target/config replacement. Its commit must check R0-03
only after the handoff is complete. Verify a clean worktree before switching
expertise to the following `high` slice, R1-01 and R1-02, which establishes one
canonical current-host target model and removes x86-only defaults/validators.

Before either slice begins, replace this paragraph with its active slice ID(s),
required expertise, production boundary, and exact focused, native-platform,
regression, and documentation test commands.

## Status update rules

1. Record active slice IDs, required expertise, boundary, and tests before
   changing code.
2. Commit one coherent slice at a time; never mix expertise labels in a commit.
3. Switching expertise requires validated checkbox/status updates, a commit, and
   a clean worktree before the next slice begins.
4. Check completed plan items/subitems in the same slice commit after the real
   production boundary passes; test doubles
   may supplement but never replace the boundary test.
5. Split oversized items into ordered verifiable sub-checkboxes before
   implementation; never use a later bulk commit to mark prior work complete.
6. Record concise test results and the commit subject after completion.
7. Keep failures and partial work visible as Active or Blocked with boxes
   unchecked.
8. Put durable behavior in the contracts, not in this ledger.
9. Populate `HOSTING_CLIENT_BREAKING_CHANGES.md` before releasing any required
   consumer or administrator migration.
10. Never edit a dependent project; record all dependent work in the handoff and
   accept adoption/test evidence produced by that project.
11. Remove superseded code, tests, commands, and documentation in the same slice;
   do not add compatibility adapters or deprecated aliases.

## Documentation correction

This reset supersedes the previous claim that phases 0-7 and the acceptance
audit were wholly complete. It does not invalidate the retained foundation or
the dependent's completed definition-protocol migration. It corrects the false
conclusion that shipped-template setup, custom package materialization, and
restart-safe healing had been proven end to end.
