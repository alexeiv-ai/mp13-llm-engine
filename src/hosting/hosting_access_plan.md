# Hosting toolbox completion plan

Status: Active corrective work

This plan replaces the completed 2026-08-08 ledger. It describes only the
remaining corrective work and the code boundaries that must change. Progress is
recorded in [hosting_status.md](hosting_status.md); normative behavior belongs in
[HOSTED_TOOLBOX_CONTRACT.md](HOSTED_TOOLBOX_CONTRACT.md).

The product is unreleased. When a final design supersedes an existing public or
internal compatibility path, the implementation slice must remove the old path,
tests, commands, models, aliases, fallbacks, and documentation. It must not keep
a legacy adapter or deprecation period.

## Ownership and dependent-project rule

This repository must not modify a dependent project. In particular, no slice in
this plan may edit `mp13-docs`. Read-only inspection may be used to understand a
consumer, but its maintainers perform and validate their own migration.

Before a parent change requires any consumer or administrator action,
[HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md) must be
populated with all of the following:

- removed contract names, payload fields, commands, error codes, and behavior;
- the exact replacement request/response sequence with representative payloads;
- changes to client branching, retry, approval, confirmation, and recovery logic;
- code, configuration, tests, and documentation the dependent must remove;
- code, configuration, tests, and documentation the dependent must add or change;
- parent release/commit pin, rollout order, and an adoption receipt supplied by
  the dependent project.

The handoff file remains populated until every listed dependent confirms
adoption. Parent code must not infer adoption by changing a dependent worktree.

## Objective

An authenticated consumer submits a complete desired toolbox definition that
may add, update, or remove more than one tool. The daemon analyzes imports,
offers exact current-host package mutations for confirmation, obtains distinct
privileged approval when policy requires it, constructs immutable environments,
warms changed workers, and atomically publishes the confirmed effective
definition.

The result must support:

- packages absent from built-in templates without a local terminal session;
- deterministic notification and confirmation of package additions, version
  transitions, and removals, including transitive packages;
- partial consumer decline: affected proposed tools are skipped and identified
  without silently removing their currently active versions;
- host-configured built-in realization, package sources, air-gapped artifacts,
  and safe removal of unreferenced non-built-in environments;
- CPython 3.12 on Windows x64/ARM64, Linux glibc x64/ARM64, and macOS ARM64;
- consumer-triggered, conflict-safe healing after daemon restart without
  restoring stale workers or leaking daemon/runtime state.

## Existing code map and defects

Implementation work must start from these seams rather than introduce a second
parallel subsystem.

1. Definition and dependency request models are in
   `toolbox/bundle_models.py` (`ToolboxDependencyRequest`,
   `ToolboxDefinitionSpec`, and the V2 assignment requests). They currently
   represent one complete requested definition but not confirmation choices or
   an effective definition after skips.
2. Import analysis and reviewed import-to-distribution mapping are in
   `toolbox/dependency_analysis.py` and `toolbox/catalog.py`. They select a
   template or unresolved custom delta; they do not produce bounded alternative
   exact locks from configured sources.
3. `toolbox/definition_planner.py::_resolve_member` computes a custom lock digest
   from the direct delta and literal `"artifacts": []`. It therefore cannot be
   passed to the real hermetic builder.
4. `service/toolbox_plans.py::PersistedToolboxDefinitionPlan` pins catalog and
   package-policy revisions but has no source/config revision, exact resolved
   artifacts, confirmation choices, effective definition, or skip receipt.
5. `service/toolbox_runtime.py::toolbox_plan_definition`,
   `toolbox_approve_definition_plan`, and `toolbox_apply_definition` expose the
   current sequence. Approval is minted through the ordinary consumer route and
   apply requires the original definition, so consumer confirmation and
   privileged dependency approval are incorrectly conflated.
6. `toolbox/host_project_config.py::ToolboxHostProjectConfiguration` accepts a
   shipped catalog resource, two x86 targets, and source IDs. It does not model
   built-in intent, source definitions/mode/revision, resolver policy, imported
   air-gap artifacts, or non-built-in environment retention/removal.
7. `daemon/local_ipc.py::EngineHostDaemon` constructs `EngineHostService`
   without the configuration and source inputs accepted by
   `service/host_service.py`, leaving normal daemon startup disconnected from
   real materialization.
8. `service/toolbox_catalog.py::materialize_toolbox_environment_for_bundle`
   accepts a verified template. `toolbox/shipped_templates.py` publishes lock
   JSON as an artifact, while `toolbox/hermetic_environment.py` requires one
   compatible exact wheel per locked distribution. Existing setup tests bridge
   this mismatch with doubles.
9. Target defaults and validators are duplicated in
   `toolbox/host_project_config.py`, `toolbox/dependency_policy.py`,
   `toolbox/catalog.py`, `toolbox/hermetic_environment.py`,
   `toolbox/orchestration.py`, and `service/toolbox_runtime.py`. Non-Windows is
   often assumed to be Linux x64; ARM64 and macOS are not modeled consistently.
10. Rollout and persisted-state paths in `toolbox/orchestration.py`,
    `service/toolbox_rollout.py`, `service/toolbox_state_v2.py`,
    `service/toolbox_env.py`, and `service/engines.py` have manifest-normalization,
    identical-reapply, deterministic candidate-ID, and runtime-repair defects.
11. POSIX `sandbox/launcher.py` uses `plain_subprocess`; daemon-death containment
    is absent. Orphan scanning and cleanup must cover
    `hosting.toolbox_executor_ipc` and its IPC/spec/candidate resources before a
    POSIX target can be advertised for untrusted toolbox execution.

## Final mutation protocol

### Roles are separate

The final protocol has three distinct authorities:

- the toolbox consumer requests a complete definition and confirms offered
  package choices;
- a dependency approver authorizes the exact accepted custom lock when host
  policy requires privileged review;
- a host administrator configures built-ins, artifact sources, trust, air-gap
  ingestion, retention, and explicit environment removal.

No operation may silently borrow another role. The existing consumer-callable
approval minting path is removed when the replacement approver path lands.

### Plan response and bounded alternatives

One plan may contain multiple tool additions, updates, and removals. Planning
must resolve the complete definition against the active definition and return
an ordered `environment_mutations` offer. Each proposed environment entry must
contain:

- stable affected tool keys and whether each is added, updated, unchanged, or
  explicitly removed;
- selected base template ID and immutable revision;
- exact direct and transitive package additions and removals compared with the
  active lock, with upgrades/downgrades represented as an explicit version
  transition;
- import root, mapped distribution, dependency reason, version, wheel filename,
  artifact digest, current-host compatibility tags, provenance, and logical
  source ID for every exact artifact;
- at most three deterministic viable resolution/source alternatives, including
  the policy-preferred selection; and
- whether consumer confirmation and separate privileged approval are required.

Alternatives may use only administrator-configured sources. Responses expose a
logical source ID and sanitized origin URL, never credentials, signed query
parameters, or daemon filesystem paths. If the solver has more viable outcomes,
it reports that alternatives were truncated; it does not enumerate an
unbounded dependency solution space. A missing compatible exact wheel is a
bounded planning/setup error rather than permission to compile an sdist.

The persisted plan in `service/toolbox_plans.py` must pin the definition, active
revision, target identity, catalog revision, host-config revision, dependency
policy revision, source-set revision, every offered exact lock/artifact digest,
and expiry. Any pin change makes confirmation or apply stale and requires a new
plan.

### Consumer confirmation and skip semantics

Add a confirmation operation between plan and approval/apply. Its request names
the plan and, for every offered environment, selects one offered alternative
and accepts or declines its package additions. It cannot submit a new version,
URL, source, lock, artifact, path, or install command.

The final consumer sequence is `toolbox_get_definition`,
`toolbox_plan_definition`, `toolbox_confirm_definition_plan`, and
`toolbox_apply_definition`. The confirmation response supplies an opaque
`confirmation_ref`. Apply accepts `plan_id`, `confirmation_ref`, `request_id`,
and, only when required, `dependency_approval_ref`; it no longer accepts or
re-resolves a second copy of the definition. The privileged operation is
`toolbox_approve_confirmed_definition_plan`. Remove
`toolbox_approve_definition_plan` and its dispatch/channel/CLI surface when the
replacement is available.

The daemon returns a durable, idempotently recoverable confirmation receipt
bound to the actor, plan, target, selected exact resolutions, accepted and
declined package groups, and resulting effective-definition hash. The receipt
must list:

- accepted tool keys;
- skipped tool keys with stable reason codes and the declined direct or
  transitive package choice that affected each tool;
- explicit tool removals that will proceed;
- exact package additions, removals, and version transitions for the effective
  definition; and
- whether privileged dependency approval is still required.

Skip behavior is deterministic:

1. Declining any required package skips every proposed new tool that depends on
   it directly or transitively.
2. If an update to an active tool is skipped, the previous active tool remains
   in the effective definition; it is not treated as an implicit removal.
3. Explicit removals still proceed because they require no package install.
4. Accepted tools may proceed only if their complete shared environment remains
   resolvable after all declines. Otherwise they are also skipped with
   `shared_environment_incomplete`.
5. Namespace/file conflicts are revalidated on the effective definition. A
   conflict cannot be resolved by arbitrary tool ordering; affected entries are
   rejected with a stable diagnostic and apply does not start.
6. Apply accepts the plan plus confirmation receipt and publishes exactly the
   pinned effective definition. It must not reinterpret the original request.

An offer containing only removals still produces a notification receipt but
requires no install acceptance. Package removal means removal from the new
logical lock; physical wheel/environment deletion remains reference-safe GC.

### Privileged approval and apply

After confirmation, policy may require a dependency approver to authorize the
exact effective custom locks and artifacts. Approval binds the confirmation
receipt and all plan/config/source/policy pins. It cannot approve declined or
unoffered choices. Apply validates and consumes the confirmation and approval
receipts, builds or reuses immutable environments, probes required imports,
warms unique candidates, and atomically publishes the complete route map.

No environment is mutated or uninstalled in place. Failure before publication
leaves the previous definition active. Publication failure drains candidates and
releases only their references.

## Host configuration and setup contract

Extend `ToolboxHostProjectConfiguration` rather than add terminal-only state.
The strict, revisioned host-owned configuration must model:

- current-target detection policy; never a configured cross-target build;
- built-in template intents (`template_id`, imports, package requirements,
  sandbox policy, required/prewarm flags, and provenance);
- ordered package sources with logical ID, kind (`https_index`,
  `https_artifact`, or `airgap_store`), sanitized origin, credential reference,
  allowed package namespaces, priority, trust keys, and download bounds;
- resolution mode (`online`, `prefer_airgap`, or `air_gapped`), timeouts, maximum
  bytes/artifacts, allowed redirects/origins, and wheel-only policy;
- immutable artifact-cache and non-built-in-environment retention policy,
  including grace period, byte/count bounds, protected digests, and whether
  unreferenced custom revisions are removed on config apply.

Configuration apply is atomic and creates a new config/source-set revision. It
invalidates unused plans and confirmation/approval receipts, but never mutates
an active environment. Existing definitions remain pinned until explicitly
replanned. Source credentials stay daemon-owned.

Air-gapped operation has two administrator paths: a configured read-only
artifact store, or a bounded signed artifact bundle uploaded through an
authenticated chunked admin control operation and committed into that store.
Normal toolbox consumers cannot supply archives or paths. Setup verifies the
manifest, hashes, signatures, exact target tags, and complete closure before
publication. If a required built-in wheel is unavailable, setup reports the
missing distribution/tags/source IDs and does not enter toolbox-ready state.

The upload lifecycle is begin/chunk/commit/cancel, with one durable operation ID,
declared total size and archive digest, bounded chunk and archive sizes, expiry,
and idempotent commit. It stages outside the trusted store and publishes only
after full verification; interruption leaves no partially visible source.

Environment deletion is not encoded as a repeatedly executed list of paths in
configuration. The configuration authorizes retention/automatic GC, while a
separate authenticated admin operation removes an exact non-built-in
environment digest. Removal is refused while any active, candidate, persisted
operation, confirmation, or plan reference exists. Built-in revisions cannot be
explicitly removed; they are replaced through built-in config revision and
setup, with old revisions reclaimed only when unreferenced.

Name the explicit operation `toolbox_environment_remove`. Its result reports
`removed`, `already_absent`, or a stable list of blocking reference kinds. It
never accepts a path, glob, logical template ID, or `force` bypass.

## Target and artifact contract

The initial target set is CPython 3.12 Windows x64/ARM64, Linux glibc
x64/ARM64, and macOS ARM64. One detector must derive interpreter ABI and
compatible `packaging.tags.sys_tags()` from the running daemon. Exact internal
target identity includes Python ABI, OS, architecture, and the platform baseline
needed for wheel compatibility.

Setup resolves and downloads only for that identity. A daemon never builds or
ships a venv for another host. Signed locks and wheels may be transported; the
destination always constructs and probes its own environment. Native CI must
prove resolution, compatible wheel selection, native-extension import, sandbox
containment, restart, and cleanup on every advertised target.

Source builds are intentionally outside this plan. If an allowed package has no
compatible verified wheel for the daemon target, the resolution fails. Adding
reproducible sdist compilation would require a separately reviewed compiler,
toolchain, build-sandbox, provenance, and cache contract.

## Itemized corrective work

Every slice declares focused tests before implementation, updates normative and
breaking-change documentation in the same slice, passes `git diff --check`, and
is committed separately. A checkbox is completed only after its production
boundary—not a double—passes.

### R0 - Corrective contract baseline

- [x] **R0-01** Replace the obsolete ledger with this code-referenced plan and
  compact current-state ledger; record that no runtime behavior changed.
- [ ] **R0-02** Update `HOSTED_TOOLBOX_CONTRACT.md` and
  `sandbox/TOOLBOX_WORKER.md` as each replacement slice becomes real. Remove
  superseded normative text rather than retain compatibility notes.
- [ ] **R0-03** Before the first client-visible implementation break, replace
  the reset marker in `HOSTING_CLIENT_BREAKING_CHANGES.md` with the complete
  dependent handoff described above. Do not edit dependent repositories.

### R1 - Canonical current-host target

- [ ] **R1-01** Add one detector module using the running interpreter and
  `packaging.tags.sys_tags()`. Replace target defaults in
  `host_project_config.py`, `dependency_policy.py`, `catalog.py`,
  `hermetic_environment.py`, `orchestration.py`, and `toolbox_runtime.py`.
- [ ] **R1-02** Update strict target/lock/catalog/cache models for the five
  target families listed above and reject cross-target wheels before download or
  build.
- [ ] **R1-03** Add native CI jobs and production-boundary tests. A target is not
  advertised until its sandbox, worker ownership, restart, and cleanup tests
  pass natively.

### R2 - Revisioned hosting configuration and built-ins

- [ ] **R2-01** Replace the current shipped-catalog-only schema in
  `toolbox/host_project_config.py` with the strict built-in/source/mode/retention
  schema above; remove old schema parsing and standard-config fixtures.
- [ ] **R2-02** Wire configuration, sources, policy, and target detection through
  `daemon/local_ipc.py::EngineHostDaemon` into `EngineHostService`; refuse
  toolbox readiness on invalid or incomplete setup.
- [ ] **R2-03** Replace realized shipped lock resources in
  `toolbox/shipped_templates.py` and `resources/toolbox_templates/` with built-in
  intent. Resolve exact transitive wheel closures for the current host.
- [ ] **R2-04** Materialize, probe, publish, and optionally prewarm built-ins via
  the real `toolbox_catalog.py` and `hermetic_environment.py` boundary. Remove
  the lock-JSON-as-wheel bridge and tests that normalize it.
- [ ] **R2-05** Implement revisioned source changes and both air-gap ingestion
  paths. Prove missing built-in wheels prevent readiness without partial catalog
  publication.

### R3 - Multi-tool planning and consumer confirmation

- [ ] **R3-01** Extend `bundle_models.py`, `definition_planner.py`, and
  `toolbox_plans.py` with exact complete resolutions, package diffs, bounded
  alternatives, source/config pins, and affected-tool dependency edges.
- [ ] **R3-02** Add the confirmation request/receipt repository and authenticated
  control-channel operation in `toolbox_runtime.py`, `daemon/local_ipc.py`,
  `engine_host_channel.py`, and `engine_host_cli.py`.
- [ ] **R3-03** Implement accepted, declined, skipped, preserved-active-update,
  explicit-removal, shared-environment, and namespace-conflict semantics exactly
  as specified above. Remove the old apply-original-definition behavior.
- [ ] **R3-04** Return sanitized source alternatives and exact direct/transitive
  additions, removals, and transitions. Reject arbitrary URLs, paths, locks, and
  install commands supplied by consumers.
- [ ] **R3-05** Prove multi-tool add/update/remove and idempotent confirmation
  recovery through the real authenticated daemon channel.

### R4 - Privileged approval and immutable apply

- [ ] **R4-01** Move approval minting behind distinct dependency-approver
  authorization in `service/auth.py`, `service/policy.py`, daemon dispatch, and
  channel/CLI surfaces. Remove ordinary-consumer approval minting.
- [ ] **R4-02** Bind approval to the confirmation receipt and exact complete
  locks/artifacts/config/source/policy revisions; reject stale or mismatched
  receipts before worker spawn.
- [ ] **R4-03** Pass accepted custom resolved inputs from the persisted plan
  through `toolbox/orchestration.py` to the real hermetic builder. Remove the
  template-only custom rejection.
- [ ] **R4-04** Atomically publish the confirmed effective definition and return
  accepted/skipped/removed tools plus logical package mutations in durable apply
  results.
- [ ] **R4-05** Prove denied, missing, incompatible, corrupt, and air-gapped
  artifacts leave the previous active definition unchanged.

### R5 - Removal, retention, and administrator environments

- [ ] **R5-01** Recompute complete closure after tool/package removal, reuse
  unaffected immutable environments, release references after publication, and
  prove custom-to-built-in contraction.
- [ ] **R5-02** Implement revisioned retention/GC config and the exact-digest
  non-built-in environment removal operation with active/candidate/plan/receipt/
  operation reference checks.
- [ ] **R5-03** Add administrator construction of a named template from an exact
  base revision plus imports/package requirements using the same resolver,
  sources, builder, probes, and immutable publication path.
- [ ] **R5-04** Keep publication inactive until explicit activation; support
  prewarm, replace, deprecate, and revoke as final APIs. Remove superseded raw
  publication payloads and commands rather than preserve both designs.

### R6 - Restart-safe consumer healing

- [ ] **R6-01** Normalize manifest identities across plan and persisted state;
  classify missing/mismatched registrations as runtime repair even when the
  semantic definition is unchanged.
- [ ] **R6-02** Give candidates unique runtime IDs, forbid implicit registration
  replacement, and compare the prior runtime-binding digest atomically. Two
  healers must yield one repair and one already-healthy/conflict result.
- [ ] **R6-03** Keep repair out of semantic rollout history while preserving
  durable operation status. Startup validates state but does not restore workers;
  consumer reapply safely reconstructs them.
- [ ] **R6-04** Add OS-native parent-death/process containment, correct orphan
  scans, and cleanup of worker IPC/spec/candidate artifacts.
- [ ] **R6-05** Test graceful restart, abrupt death, identical reapply,
  concurrent healers, and failures immediately before and after route
  publication on every advertised host.

### R7 - Breaking-change handoff and acceptance

- [ ] **R7-01** For every removed/replaced API, populate the breaking-change
  handoff before its implementation commit and obtain dependent-provided
  adoption evidence; never modify the dependent project.
- [ ] **R7-02** Add one no-double end-to-end suite covering configured daemon
  startup, built-ins, source alternatives, confirmation decline/skip, approval,
  custom add/remove, restart healing, environment removal, and GC.
- [ ] **R7-03** Run focused parent tests, native target suites, and complete
  parent regression. Dependents run their own migration tests and report pins.
- [ ] **R7-04** Reconcile plan, status, contracts, setup docs, worker
  architecture, and breaking-change handoff with actual code; remove obsolete
  tests/docs/code and commit the audit separately.

## Acceptance criteria

- [ ] Normal daemon setup constructs and probes required built-ins for only its
  current target from the configured source mode.
- [ ] Missing exact wheels stop air-gapped setup with stable bounded diagnostics.
- [ ] One definition can add, update, and remove multiple tools; its plan offers
  bounded exact package/source alternatives and complete package notifications.
- [ ] Consumer confirmation can decline packages; the receipt identifies every
  skipped affected tool, preserves skipped active updates, and applies explicit
  removals without ambiguity.
- [ ] Privileged approval is distinct from consumer confirmation and binds only
  accepted exact locks and artifacts.
- [ ] Package/source/config changes invalidate stale receipts but never mutate an
  active environment.
- [ ] Removal releases references only after publication; non-built-in deletion
  cannot remove referenced or built-in environments.
- [ ] Windows x64/ARM64, Linux glibc x64/ARM64, and macOS ARM64 pass native setup,
  wheel, sandbox, worker, restart, and cleanup tests.
- [ ] Identical reapply after restart heals missing runtime state without a new
  semantic revision, state corruption, leaked workers, or healer conflict.
- [ ] Normal consumers cannot publish templates, choose arbitrary sources/URLs,
  upload artifacts, supply filesystem/interpreter paths, or install packages.
- [ ] Superseded compatibility code and documentation are removed, and every
  dependent action is recorded in the breaking-change handoff with independent
  adoption evidence.
