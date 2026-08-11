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

## Remaining external acceptance gates

The corrective implementation gaps above are closed by R1 through R7 parent
work. Two release gates cannot be manufactured from this checkout:

1. The five native GitHub Actions jobs must report successful target detection,
   native-extension import, sandbox worker execution, restart healing, and
   cleanup. The workflow contains those commands, but only Windows x64 has been
   run locally.
2. Each dependent maintainer must supply an adoption receipt pinned to the
   committed parent implementation and containing its own migration command and
   result. No dependent repository is modified or treated as adopted here.

## Progress ledger

Only corrective work is tracked here. The former phase-by-phase historical test
transcript was intentionally removed because it obscured current truth.

| Work group | Status | Outcome/evidence |
| --- | --- | --- |
| R0 Corrective contract baseline | Active | R0-01 is complete. R0-03 populated the corrective consumer/administrator handoff before implementation: removed target/configuration and mutation surfaces, replacement payload sequence, durable retry/watch/recovery behavior, dependent removals/additions, rollout order, and explicit pending implementation/adoption gates are recorded. Focused docs: 10 passed in 0.12s; `git diff --check` passed. R0-02 remains a cross-slice obligation. No runtime behavior changed. |
| R1 Canonical current-host target | Active | R1-01/R1-02 add the canonical CPython 3.12 detector and use it across configuration, policy, catalog/cache, hermetic construction, orchestration, runtime, and model-runtime validation. Five target families are strict; configured cross-target construction and incompatible pinned wheels fail before build. Focused target/config/catalog/policy/builder/rollout/hash suite: 62 passed in 108.45s; docs: 10 passed in 0.05s; broader definition/model boundary suite: 54 passed in 3.59s; `git diff --check` passed. One earlier focused run exposed the existing nondeterministic concurrent-publication path check (61 passed, 1 failed); its isolated rerun passed in 12.74s and the complete rerun passed. R1-03a implements a five-runner GitHub Actions matrix; the local Windows x64 target/native-wheel probe passed and target/workflow tests are 9 passed in 0.73s. R1-03a remains unchecked until all native jobs execute successfully. R1-03b remains gated on R6-04, and no new target family is advertised. |
| R2 Revisioned hosting configuration and built-ins | Complete | R2-01a removes the shipped-catalog-only parser and service arguments. Strict built-in/source/resolution/retention models compute config/source-set revisions, reject target selection and invalid mode/source combinations, and redact credentials/paths. Service readiness uses `toolbox_host_project` plus `toolbox_readiness`; the handoff and normative contract were updated in-slice. Host-config: 7 passed in 2.50s; docs: 10 passed in 0.13s; shipped-template integration: 6 passed in 1.52s; removed-schema audit passed. R2-01b atomically persists revision history; changes invalidate unconsumed plans and non-active receipts while active catalog/environment-reference state remains unchanged. Focused config/plan/receipt/docs: 34 passed in 3.22s. R2-02a wires strict config, logical source bindings, policy, and detected target through normal `EngineHostDaemon` into the real hermetic materializer. R2-02b keeps the control plane available with stable unavailable diagnostics and zero catalog publication for absent/partial/invalid setup. Combined daemon/config/docs boundary: 23 passed in 2.68s; `git diff --check` passed. R2-03a removes packaged realized catalog/lock resources, their initializer and runtime fallback; config now carries intent only, while planner/service tests publish explicit test-only realized fixtures. Focused migration: 62 passed in 8.45s. R2-03b1 adds bounded deterministic read-only air-gap closure resolution and stable path-redacted failure results; focused resolver/daemon/config: 13 passed in 15.48s. R2-05b1 adds canonical raw-Ed25519 bundle verification, adversarial ZIP/wheel/closure validation, and atomic content-addressed indexing. Focused signed-bundle/resolver/config/docs matrix: 29 passed in 13.40s; Poetry lock/check and `git diff --check` passed (existing Poetry deprecation warnings only). |
| R3 Multi-tool planning and consumer confirmation | Complete | Durable planning and confirmation expose bounded exact alternatives, direct/transitive mutations, decline/skip/preserve/remove semantics, request recovery, and changed-snapshot watching. |
| R4 Privileged approval and immutable apply | Complete | Dependency approval is a distinct authority bound to the confirmation and exact artifacts; apply consumes only immutable plan/confirmation/approval receipts and atomically publishes the confirmed effective definition. |
| R5 Removal, retention, and administrator environments | Complete | R5-01 through R5-05 are complete. Explicit tool removal contracts shared profiles after atomic publication; exact environment deletion is reference-safe; administrator construction publishes inactive verified revisions with explicit lifecycle transitions; mutating maintenance is canonical, durable, idempotent, cancellable before mutation, and restart-recoverable. |
| R6 Restart-safe consumer healing | Complete | R6-01 through R6-06 are implemented. Manifest digests are normalized at plan/state/registration boundaries; concrete toolbox candidates use unique runtime IDs and immutable binding digests; duplicate registrations are rejected instead of replaced; recovery reports missing/mismatched runtime bindings for explicit reapply; Linux workers request native parent-death termination, Windows retains job containment, and retirement removes bounded worker spec/scratch artifacts. Duplicate toolbox execution attaches return the current durable snapshot immediately, cancellation acknowledges before asynchronous teardown, and `toolbox-describe-refresh` is a separate durable operation while `toolbox-describe` is bounded to persisted registration state. Poetry-based focused rollout/atomic/sandbox/operation and state archive tests passed. |
| R7 Breaking-change handoff and acceptance | Parent complete; external gates pending | The parent handoff inventory and receipt schema are populated. The real-daemon no-double suite covers signed configured setup, decline/skip, distinct approval, custom add/execute/remove, environment-removal safety, restart healing, GC, and request recovery. Native workflow commands now include sandbox/restart/cleanup coverage. Full dependent adoption and five-runner evidence remain external release gates. |

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

Active slice: R7 acceptance audit. Parent implementation and local acceptance
are complete; five-runner native evidence and dependent adoption receipts remain
external release gates.

R7 parent evidence: `tests/test_hosting_r7_acceptance.py` constructs a normal
configured daemon over a signed current-target source and the real hermetic
materializer. It proves durable duplicate plan/execution attachment, package
decline/skip, separate dependency approval, exact apply, real worker execution,
custom add/remove, referenced-environment removal safety, restart healing with
the same semantic revision, maintenance, and request-ID terminal recovery. The
suite exposed and fixed restart cleanup treating an already-absent old worker as
a post-publication error. `.github/workflows/hosting-native-targets.yml` now runs
the R6/R7 sandbox, worker, restart, and cleanup boundary on all five declared
runner families. Local Poetry evidence and the complete regression result are
recorded here: the focused R7/R6/rollout/contract matrix passed 47 tests in
72.57s; the contract/removal/public-guarantee docs matrix passed 22 tests in
6.86s; compile, `git diff --check`, and the local Windows x64 native-extension
probe passed. Two complete Poetry regressions collected 1,221 tests. The first
reached 1,219 passed and one skipped with one workflow-JS subprocess startup
timeout that passed alone in 2.12s. The timed run reached 1,218 passed and one
skipped with two workflow-Python timing/isolation failures; fixed durable
request IDs and unconditional registry cleanup reduced their combined rerun to
two passes in 4.57s. The five-runner native result and dependent migration
receipt remain unavailable, so R7-01/R7-03 and the corresponding acceptance
criteria stay unchecked.

R7 commits: client handoff `eb3c631`; restart-safe runtime, native-workflow
wiring, and real-daemon acceptance `d689bda`; the reconciled documentation and
surface audit are committed separately. The measured full-suite hotspot list
is led by the real R7 lifecycle (67.47s), workflow-node environment/process
tests (21.34-32.51s), hermetic builders (11.82-23.73s), and artifact-store
construction (12.92-15.90s). A follow-up optimization should cache immutable
wheel/seed fixtures and replace polling/fixed joins with explicit worker events
while retaining one serial real-process case per production boundary.

Completed R6 evidence: R6-01/R6-02 add canonical manifest normalization,
runtime-binding digests, unique concrete candidate IDs, and fail-closed
registration replacement protection. R6-03 recovery validates the persisted
binding against the registration and reports `runtime_repair_required` without
creating a worker or rewriting semantic rollout history. R6-04 adds Linux
`PR_SET_PDEATHSIG`, preserves Windows job-object containment, and cleans
bounded toolbox worker spec/scratch artifacts during retirement. R6-05 covers
the existing restart/atomic-routing boundaries plus the Poetry live-worker
smoke path. R6-06 removes synchronous duplicate attach waiting, acknowledges
active cancellation before background teardown, and adds the durable
`toolbox-describe-refresh` operation. The focused R6 rollout/atomic/sandbox
and operation checks passed through `poetry run`; the complete
`test_hosting_toolbox_state_v2.py` state/archive coverage also passed.

Completed slice evidence: R5-05 (`high`) converts mutating GC, repair, and
reconcile into canonical `toolbox_maintenance` operations selected by the
`toolbox-host` scope. High-level channel/CLI helpers submit through `op-start`;
raw long dispatch is rejected. Stable request IDs deduplicate and recover the
same operation, including one safe replay after an interrupted dispatch.
Validation is cancellable; recovery/repair/GC/cleanup are durable
non-cancellable mutation checkpoints, so cancellation returns immediately and
never races a filesystem or worker mutation. Reference, consistency, and review
remain bounded reads. The focused maintenance/operation/auth/contract matrix
passed 125 tests in 18.08s. The combined R5 removal, construction, lifecycle,
maintenance, transport, repository, and exact physical builder matrix passed
191 tests in 78.28s. Compile and `git diff --check` passed. The client handoff
for the synchronous-to-hosted replacement was committed first in `2bd22ce`.

Completed slice evidence: R5-03/R5-04 (`high`) remove the raw caller-supplied
template/artifact/signature publication surface. Administrator construction is
one canonical `toolbox_template_construct` operation selected by the new
logical template ID and pinned to an exact non-revoked base digest, config/source
revisions, and current target. It retains the exact base distribution pins,
resolves only configured sources, verifies signed artifacts, uses the real
hermetic builder and complete import probes, commits the exact receipt, and
publishes an immutable inactive revision. Separate exact activate and
compare-and-swap replace operations prevent implicit active replacement;
deprecate, revoke, and prewarm remain final exact-revision APIs. The focused
catalog, prewarm, shared resolver, definition resolution/config/transport,
removed-surface, operation/auth, contract, and real signed-artifact/materializer
matrix passed 148 tests in 60.84s. Compile and `git diff --check` passed. The
breaking-change handoff was committed first in `2bd22ce`.
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

Completed slice evidence: R5-02 (`average`) adds the revision-bound,
administrator-only `toolbox_environment_remove` durable operation with an exact
`environment_digest` selector. It checks active profiles, candidate
registrations, unexpired plans and confirmations, active operations, built-in
references, protected retention digests, and builder references before any
physical deletion. The builder performs one exact locked removal and reports
`removed`/`already_absent`; GC now carries the revisioned retention policy and
protected digests. Operation progress is strict and durable through validation,
reference check, removal, and cleanup. The R5-02 maintenance/operation/auth/
transport/contract suite passed 96 tests in 10.76s; exact physical builder
removal passed 1 test in 13.70s; the earlier R5-01 boundary and surrounding
regression remain recorded above. Compile and `git diff --check` passed. No
public API was removed or replaced, so no breaking-change handoff update was
required.

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
