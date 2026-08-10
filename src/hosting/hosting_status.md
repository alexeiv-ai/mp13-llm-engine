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

## Progress ledger

Only corrective work is tracked here. The former phase-by-phase historical test
transcript was intentionally removed because it obscured current truth.

| Work group | Status | Outcome/evidence |
| --- | --- | --- |
| R0 Corrective contract baseline | Active | R0-01 is complete and was corrected on 2026-08-09 to prohibit dependent edits, remove compatibility preservation, reference production seams, and specify confirmation/config behavior. R0-02 and R0-03 remain cross-slice obligations. No runtime behavior changed. |
| R1 Canonical current-host target | Pending | No implementation started. |
| R2 Revisioned hosting configuration and built-ins | Pending | No implementation started. |
| R3 Multi-tool planning and consumer confirmation | Pending | No implementation started. |
| R4 Privileged approval and immutable apply | Pending | No implementation started. |
| R5 Removal, retention, and administrator environments | Pending | No implementation started. |
| R6 Restart-safe consumer healing | Pending | No implementation started. |
| R7 Breaking-change handoff and acceptance | Pending | No implementation started. |

## Next implementation slice

No code slice is active. The next slice should be R1-01 and R1-02: establish one
canonical current-host target model and replace the existing x86-only defaults
and validators. Because those replacements are client-visible, R0-03 must first
replace the breaking-change reset marker with exact migration instructions.
Before implementation, record the exact focused, native-platform, regression,
and documentation test commands in this section.

## Status update rules

1. Record an active slice and its test commands before changing code.
2. Commit one coherent slice at a time.
3. Check plan items only after the real production boundary passes; test doubles
   may supplement but never replace the boundary test.
4. Record concise test results and the commit subject after completion.
5. Keep failures and partial work visible as Active or Blocked.
6. Put durable behavior in the contracts, not in this ledger.
7. Populate `HOSTING_CLIENT_BREAKING_CHANGES.md` before releasing any required
   consumer or administrator migration.
8. Never edit a dependent project; record all dependent work in the handoff and
   accept adoption/test evidence produced by that project.
9. Remove superseded code, tests, commands, and documentation in the same slice;
   do not add compatibility adapters or deprecated aliases.

## Documentation correction

This reset supersedes the previous claim that phases 0-7 and the acceptance
audit were wholly complete. It does not invalidate the retained foundation or
the dependent's completed definition-protocol migration. It corrects the false
conclusion that shipped-template setup, custom package materialization, and
restart-safe healing had been proven end to end.
