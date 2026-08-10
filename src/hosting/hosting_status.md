# Hosting toolbox corrective status

Last updated: 2026-08-10

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
| R0 Corrective contract baseline | Active | R0-01 is complete. R0-03 populated the corrective consumer/administrator handoff before implementation: removed target/configuration and mutation surfaces, replacement payload sequence, durable retry/watch/recovery behavior, dependent removals/additions, rollout order, and explicit pending implementation/adoption gates are recorded. Focused docs: 10 passed in 0.12s; `git diff --check` passed. R0-02 remains a cross-slice obligation. No runtime behavior changed. |
| R1 Canonical current-host target | Active | R1-01/R1-02 add the canonical CPython 3.12 detector and use it across configuration, policy, catalog/cache, hermetic construction, orchestration, runtime, and model-runtime validation. Five target families are strict; configured cross-target construction and incompatible pinned wheels fail before build. Focused target/config/catalog/policy/builder/rollout/hash suite: 62 passed in 108.45s; docs: 10 passed in 0.05s; broader definition/model boundary suite: 54 passed in 3.59s; `git diff --check` passed. One earlier focused run exposed the existing nondeterministic concurrent-publication path check (61 passed, 1 failed); its isolated rerun passed in 12.74s and the complete rerun passed. R1-03a implements a five-runner GitHub Actions matrix; the local Windows x64 target/native-wheel probe passed and target/workflow tests are 9 passed in 0.73s. R1-03a remains unchecked until all native jobs execute successfully. R1-03b remains gated on R6-04, and no new target family is advertised. |
| R2 Revisioned hosting configuration and built-ins | Active | R2-01a removes the shipped-catalog-only parser and service arguments. Strict built-in/source/resolution/retention models compute config/source-set revisions, reject target selection and invalid mode/source combinations, and redact credentials/paths. Service readiness uses `toolbox_host_project` plus `toolbox_readiness`; the handoff and normative contract were updated in-slice. Host-config: 7 passed in 2.50s; docs: 10 passed in 0.13s; shipped-template integration: 6 passed in 1.52s; removed-schema audit passed. R2-01b atomically persists revision history; changes invalidate unconsumed plans and non-active receipts while active catalog/environment-reference state remains unchanged. Focused config/plan/receipt/docs: 34 passed in 3.22s. R2-02a wires strict config, logical source bindings, policy, and detected target through normal `EngineHostDaemon` into the real hermetic materializer. R2-02b keeps the control plane available with stable unavailable diagnostics and zero catalog publication for absent/partial/invalid setup. Combined daemon/config/docs boundary: 23 passed in 2.68s; `git diff --check` passed. R2-03a removes packaged realized catalog/lock resources, their initializer and runtime fallback; config now carries intent only, while planner/service tests publish explicit test-only realized fixtures. Focused migration: 62 passed in 8.45s. R2-03b1 adds bounded deterministic read-only air-gap closure resolution and stable path-redacted failure results; focused resolver/daemon/config: 13 passed in 15.48s. R2-05b1 adds canonical raw-Ed25519 bundle verification, adversarial ZIP/wheel/closure validation, and atomic content-addressed indexing. Focused signed-bundle/resolver/config/docs matrix: 29 passed in 13.40s; Poetry lock/check and `git diff --check` passed (existing Poetry deprecation warnings only). |
| R3 Multi-tool planning and consumer confirmation | Pending | No implementation started. |
| R4 Privileged approval and immutable apply | Pending | No implementation started. |
| R5 Removal, retention, and administrator environments | Pending | No implementation started. |
| R6 Restart-safe consumer healing | Pending | No implementation started. |
| R7 Breaking-change handoff and acceptance | Pending | No implementation started. |

R2 evidence continuation: R2-05b2 binds the exact daemon public-key set,
discovers direct signed bundles, and resolves only rehashed CAS objects with
bounded degraded diagnostics. The expanded signed-ingress/daemon/config/docs
matrix passed 45 tests in 17.92s; `git diff --check` passed.

## Active implementation slice

Active slice: R2-05b2 (`high`). Bind the exact configured Ed25519 public-key set
at normal daemon construction, discover only direct signed ZIPs in read-only
air-gap roots, and import them through R2-05b1. Feed verified CAS object paths
to the built-in resolver internally while normal projections expose only
logical sources and digests. Invalid bundles degrade toolbox setup without
taking down the control plane or publishing any catalog entry.

Required validation:

- normal daemon signed-bundle discovery/resolution and restart test
- missing/extra/malformed key binding and invalid-bundle degraded diagnostics
- no unsigned raw-wheel fallback, path/key redaction, and zero catalog publish
- `python -m pytest tests/test_hosted_toolbox_contract_docs.py -q`
- `git diff --check`

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
