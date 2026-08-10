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
| R2 Revisioned hosting configuration and built-ins | Complete | R2-01a removes the shipped-catalog-only parser and service arguments. Strict built-in/source/resolution/retention models compute config/source-set revisions, reject target selection and invalid mode/source combinations, and redact credentials/paths. Service readiness uses `toolbox_host_project` plus `toolbox_readiness`; the handoff and normative contract were updated in-slice. Host-config: 7 passed in 2.50s; docs: 10 passed in 0.13s; shipped-template integration: 6 passed in 1.52s; removed-schema audit passed. R2-01b atomically persists revision history; changes invalidate unconsumed plans and non-active receipts while active catalog/environment-reference state remains unchanged. Focused config/plan/receipt/docs: 34 passed in 3.22s. R2-02a wires strict config, logical source bindings, policy, and detected target through normal `EngineHostDaemon` into the real hermetic materializer. R2-02b keeps the control plane available with stable unavailable diagnostics and zero catalog publication for absent/partial/invalid setup. Combined daemon/config/docs boundary: 23 passed in 2.68s; `git diff --check` passed. R2-03a removes packaged realized catalog/lock resources, their initializer and runtime fallback; config now carries intent only, while planner/service tests publish explicit test-only realized fixtures. Focused migration: 62 passed in 8.45s. R2-03b1 adds bounded deterministic read-only air-gap closure resolution and stable path-redacted failure results; focused resolver/daemon/config: 13 passed in 15.48s. R2-05b1 adds canonical raw-Ed25519 bundle verification, adversarial ZIP/wheel/closure validation, and atomic content-addressed indexing. Focused signed-bundle/resolver/config/docs matrix: 29 passed in 13.40s; Poetry lock/check and `git diff --check` passed (existing Poetry deprecation warnings only). |
| R3 Multi-tool planning and consumer confirmation | Pending | No implementation started. |
| R4 Privileged approval and immutable apply | Pending | No implementation started. |
| R5 Removal, retention, and administrator environments | Pending | No implementation started. |
| R6 Restart-safe consumer healing | Pending | No implementation started. |
| R7 Breaking-change handoff and acceptance | Pending | No implementation started. |

R2 evidence continuation: R2-05b2 binds the exact daemon public-key set,
discovers direct signed bundles, and resolves only rehashed CAS objects with
bounded degraded diagnostics. The expanded signed-ingress/daemon/config/docs
matrix passed 45 tests in 17.92s; `git diff --check` passed.
R2-04a1 constructs single-bundle signed-provenance candidates and passes exact
verified CAS paths through the real hermetic install/lock/import-probe boundary
without catalog or public receipt mutation. The focused success/corruption/
runtime-artifact/ambiguous-evidence/probe-failure matrix passed 19 tests in
33.92s. The expanded artifact/hermetic/prewarm/docs suite passed 48 tests in
138.19s; `git diff --check` passed.
R2-04a2 atomically replaces the complete receipt set and complete active
catalog set, with rollback of newly inserted receipts/references on ordinary
failure and idempotent restart/retry. It also fixes physical environment reuse
across logical templates by comparing only physical receipt identity and
rerunning each template's import probes. The expanded catalog/prewarm/hermetic/
docs suite passed 61 tests in 171.78s; `git diff --check` passed.
R2-06a adds the strict system-owned `toolbox_setup` hosted execution kind with
the sole `host_scope: toolbox-host` selector, canonical
config/source-set/target fingerprint, immediate duplicate attachment, fixed
non-cancellable progress phases, verified-byte acquisition units, and bounded
terminal success/failure. Toolbox readiness becomes ready only after complete
atomic publication. The operation/repository/artifact-store/docs suite passed
80 tests in 84.23s; `git diff --check` passed.
R2-04b/R2-06b moves verified bundle ingestion and exact built-in resolution
behind the automatically dispatched canonical setup worker, so normal daemon
construction does not wait for source I/O or hermetic builds. Configured
readiness is gated on the canonical operation and real active receipts. Restart
redispatches an interrupted-before-dispatch record once on its original ID;
interrupted-after-dispatch succeeds only from a durable committed-publication
checkpoint plus current receipts and otherwise terminally fails on that same
record. The focused daemon/artifact/config/restart suite passed 41 tests in
98.24s. The expanded operation-repository/service/daemon/artifact/config/docs
suite passed 107 tests in 101.56s; `git diff --check` passed.
R2-05a1 adds redirect-controlled signed PEP 691 metadata fetch, exact
daemon-owned Authorization binding, streamed byte/time bounds, and verified
wheel download into artifact-store v2. Signed HTTPS manifests and signed
air-gap bundles share immutable CAS objects but retain distinct evidence; any
signature, redirect, size, digest, tag, namespace, or metadata failure leaves
the index unchanged. The HTTPS/store suite passed 31 tests; compile and
`git diff --check` passed. The expanded HTTPS/store/config/docs suite passed 47
tests in 97.91s.
R2-05a2/R2-03b2 adds bounded transitive candidate discovery from signed wheel
metadata, exact daemon credential wiring, offline-only pip resolution over
verified CAS paths, deterministic aggregate HTTPS evidence, and normal
nonblocking daemon build/probe/publication. Missing transitive content remains
not-ready with zero catalog publication. Online and air-gap fixtures with the
same logical source and wheel bytes produce identical lock/artifact digests.
The expanded online/air-gap/resolver/hermetic/config/docs suite passed 72 tests
in 224.43s; the final PEP 503/691 acquisition matrix passed 10 tests in
17.85s. Compile and `git diff --check` passed.
The final shared HTTPS/signed-bundle store regression passed 35 tests in
112.37s after bounding each signed project alternative set and source-scoping
HTTPS manifest identities.
R2-05c1 adds process-safe untrusted upload staging with exact
owner/request/source/config/source-set/target/size/digest identity, idempotent
begin and chunk retry, contiguous 1 MiB chunk bounds, restart continuation,
15-minute expiry, synchronous idempotent cancel, and zero trusted-store
visibility. The focused upload/compile/docs boundary passed 5 tests in 1.52s;
the expanded upload/docs suite passed 15 tests in 1.44s; `git diff --check`
passed.
R2-05c2 adds five admin-only upload commands and one durable non-cancellable
`toolbox_artifact_import` operation per complete upload. The worker rehashes the
declared archive, verifies/imports the signed current-target closure through the
atomic CAS boundary, removes successful staging, and reconciles restart from a
durable committed checkpoint without a parallel operation. The focused
operation/upload boundary passed 35 tests; expanded validation is recorded in
the slice commit. The expanded operation-repository/service, daemon-startup,
auth-role, shared artifact-store, upload, and contract-doc suite passed 150
tests in 113.22s.

## Active implementation slice

Active slice: none. R5-01 (`high`) is complete; pause before R5-02 (`average`).
Remaining profiles are resolved from only their post-removal requirements, so a
custom requests/urllib3 closure contracts to the exact built-in closure when
those requirements disappear. Exact profiles reuse their engine, environment,
and actual builder reference. Replaced/removed references survive the atomic
publication boundary and are released afterward, including while an old worker
finishes draining; physical deletion remains separate and grace-period guarded.
Active state now requires its complete profile-reference set, and resolved
assignments declare the materialization reference explicitly. The
planner/resolution, real hermetic reference-index, maintenance, atomic-routing,
state, and contract matrix passed 51 tests in 126.94s. The expanded surrounding
definition, rollout, catalog, transport, maintenance, and public-guarantee suite
passed 92 tests in 40.08s; final model/atomic/state/docs validation passed 32
tests in 7.34s. Compile and `git diff --check` passed. No public API was removed
or replaced, so the existing breaking-change handoff already covers this
explicit-removal behavior and required no new migration entry.

Completed slice evidence: R3 and R4 are complete. R4-03/R4-04/R4-05
(`high`) carry the selected exact verified CAS closure from confirmation through
orchestration into the real offline hermetic builder. Active custom state keeps
the exact resolved input for later comparisons. Apply builds/stages/warms
before one atomic effective-definition publication, returns confirmed tool and
package outcomes, and cleans candidate registrations/references on failure.
The real builder/air-gap/corrupt-artifact/atomic-routing matrix passed 45 tests
in 128.36s. The expanded authenticated daemon, definition service/transport,
public guarantees, app migration, auth, operation, state, rollout, and atomic
routing suite passed 140 tests in 48.75s; compile and `git diff --check` passed.
Repository-wide validation reached 1189 passed and 3 skipped in 562.56s; two
pre-existing workflow-helper timing tests failed in that suite order, while
their immediate isolated reruns passed (the stream case in the two-test rerun
and the capacity-shrink case alone in 3.96s). The preceding repository-wide run
reached 1190 passed and 3 skipped with only the now-corrected stale contract
signature assertion; the contract-doc suite then passed 10 tests in 0.09s.

Completed slice evidence: R4-01/R4-02 (`average`) are complete. The ordinary
consumer approval command and hosted-reference helper are removed. A distinct
dependency-approver role and `toolbox-approve-confirmed-definition-plan`
surface mint approval only after confirmation. The durable approval binds the
confirmation owner/authority, approving actor, confirmation-ref digest,
effective definition, selected exact locks/artifacts digest, complete plan-pin
digest, expiry, and first apply request. Focused service, authenticated role,
channel/daemon/CLI transport validation passed 13 tests in 5.57s. The expanded
auth-role, service, transport, and operation-repository suite passed 78 tests
in 19.47s; compile and `git diff --check` passed.

Completed slice evidence: R3-04b/R3-05 (`average`) are complete. Apply now accepts
only `plan_id`, the actor-bound immutable `confirmation_ref`, and a request ID;
the caller-supplied definition and apply-time re-resolution path are removed.
Complete runnable bundle inputs survive restart inside the immutable plan and
confirmation records, while consumer choices remain limited to offered IDs and
one boolean. Durable results carry accepted/skipped/preserved/removed tools and
logical package mutations. The focused plan repository, definition matrix,
service, and transport suite passed 32 tests in 9.00s; compile and
`git diff --check` passed.

Completed slice evidence: R3-02/R3-03/R3-07 (`high`) are complete. Planning and
confirmation/acquisition now use canonical hosted execution kinds and the
shared operation repository; identical request IDs return current status and
one deterministic immutable confirmation receipt. Daemon `op-*` routes these
families without `operations.json`, raw long-command dispatch fails closed,
and the channel watch emits changed snapshots until a bounded timeout or
terminal state. The focused service/operation/transport/config suite passed 43
tests in 5.37s. The expanded daemon startup/ACL/auth, CLI, transport,
operation-repository, and apply-operation suite passed 179 tests in 27.23s;
compile and `git diff --check` passed.

Completed slice evidence: R3-01c (`high`) is complete. Definition planning now
resolves exact wheel-only candidates from configured verified CAS sources,
reconstructs the active template closure from verified catalog references, and
persists the complete v2 plan with target/config/source/policy pins. Sanitized
source-set alternatives expose exact template/direct/transitive artifacts
without credentials, signed queries, filesystem paths, or installed-
environment discovery. The focused configured resolver, definition service,
host-config transition, planner matrix, and repository suite passed 31 tests
in 9.91s; `git diff --check` passed.

Completed slice evidence: R3-01a added the strict v2 complete-plan record and
process-safe immutable repository. Focused planner/repository/model/service/
contract validation passed 43 tests in 5.69s; `git diff --check` passed.

Completed slice evidence: R3-04a added strict offered artifact, package
mutation, alternative, environment, dependency-edge, and pin models plus the
pure deterministic confirmation reducer. It accepts only complete offered
choices, implements decline/skip, preserved active updates, explicit removal
and shared-environment propagation, reconstructs the exact effective
definition, and revalidates namespace conflicts before any worker. Focused
validation passed 37 planner/model/repository/contract tests in 3.10s;
`git diff --check` passed.

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
