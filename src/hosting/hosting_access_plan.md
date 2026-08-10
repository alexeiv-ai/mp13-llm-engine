# Hosting toolbox completion plan

Status: Active corrective work

This plan replaces the completed 2026-08-08 implementation ledger. It keeps
the useful toolbox-definition foundation, records the integration defects found
after acceptance, and defines the remaining work needed for usable package
environments, target-local templates, ARM64 hosts, and safe restart recovery.

Progress is recorded in [hosting_status.md](hosting_status.md). Durable public
behavior belongs in [HOSTED_TOOLBOX_CONTRACT.md](HOSTED_TOOLBOX_CONTRACT.md).
Client migrations belong in
[HOSTING_CLIENT_BREAKING_CHANGES.md](HOSTING_CLIENT_BREAKING_CHANGES.md).

## Objective

An authenticated consumer submits one complete desired toolbox definition. The
daemon must analyze its source and dependency intent, prepare an isolated
environment for the daemon's current host target, warm changed workers, and
atomically publish the complete route map.

The supported result must include all of the following:

- a new tool can add packages not present in a built-in template;
- removing a tool or removing its package requirement releases packages that
  are no longer needed without mutating a live environment;
- built-in and administrator-created templates are constructed for the one
  architecture on which the daemon is running;
- Linux ARM64, macOS ARM64, and Windows ARM64 are supported alongside existing
  x64 hosts;
- air-gapped setup fails clearly when an exact required artifact is absent;
- daemon restart does not restore workers automatically, corrupt persisted
  truth, leak workers, or make competing consumer healing operations conflict.

## Fixed design decisions

### Complete definitions remain the consumer contract

The existing `get_definition`, `plan_definition`,
`approve_definition_plan`, and durable `apply_definition` sequence remains the
normal toolbox mutation boundary. Consumers submit source plus dependency
intent; they never submit a venv, interpreter path, wheel path, lockfile, or
installation command.

`declared_imports` is for dynamic, optional, or conditional imports that source
analysis cannot prove. `package_requirements` contains reviewed PEP 508
distribution requirements. Import names and distribution names remain separate.

### Only the daemon's current target is realized

A daemon never builds or distributes environments for another architecture.
Setup detects the current Python ABI, OS, architecture, and compatible wheel
tags. A stable logical template such as `core` may therefore have different
immutable realized revision digests on different hosts.

Cross-platform support still requires native CI, because compatible wheel
selection, native extensions, process containment, and sandbox enforcement must
work on every advertised host. It does not require a daemon-side cross-target
template matrix.

### Templates are immutable realized environments

Built-in configuration supplies logical template intent: stable ID, imports,
package requirements, sandbox policy, and provenance. Hosting setup resolves
that intent for the current target, obtains exact artifacts, creates a complete
lock, materializes and probes the environment, and then publishes its immutable
local revision.

An administrator may construct another named template from a base template and
additional import/package intent. Publication and activation remain separate.
Existing revisions are never modified in place.

### Custom environments solve the one-tool long tail

A tool does not need a named template merely because it requires an additional
package. Planning may derive a complete custom environment from an allowed base
template. The complete base-plus-delta lock and artifacts are parent-resolved,
approval-bound when policy requires it, independently materialized, and cached
by immutable identity.

Frequently reused custom locks may later seed an administrator-created template,
but promotion is explicit and never triggered by usage frequency alone.

### Restart recovery is consumer-triggered

Daemon startup validates persisted state but does not automatically restore
toolbox workers. After reconnect, a consumer may safely reapply its complete
desired definition. An identical reapply must repair missing runtime bindings
without creating a new semantic definition revision.

## Retained foundation

The following implemented behavior remains in scope and must not be replaced:

- strict complete definition/request models and canonical identities;
- source import analysis and reviewed import-to-distribution mapping;
- immutable definition plans and exact approval-reference binding;
- durable hosted apply operations and terminal result recovery;
- non-inheriting venv construction with offline `--no-index --no-deps`
  installation from a complete exact wheel set;
- exact lock verification, final-interpreter import probes, quarantine, atomic
  cache publication, reference tracking, and grace-period GC;
- candidate/active/retired worker states and atomic complete route publication;
- digest-validated, process-safe persisted toolbox state;
- the dependent project's adopted complete-definition integration.

## Confirmed defects

These are acceptance failures, not optional enhancements.

1. **Custom plan/build disconnect.** Planning records a custom requirement
   delta but does not resolve a complete transitive lock or artifact set. The
   production materialization path accepts only an already verified template
   and rejects the custom delta.
2. **Invalid shipped artifact bridge.** Shipped template publication references
   lock JSON as the artifact, while the real materializer requires one exact
   compatible wheel per locked distribution.
3. **Daemon configuration disconnect.** The ordinary daemon constructs its host
   service without the toolbox project configuration and artifact sources needed
   by the real materializer.
4. **No target-local setup resolver.** `online_resolution_allowed` is policy
   data only; there is no controlled current-target resolver/downloader. The
   air-gapped local wheel path exists only after an exact resolved input has
   already been manufactured elsewhere.
5. **Persisted reuse mismatch.** Planned bundle manifest hashes and persisted
   prefixed hashes are compared in different forms, so an unchanged persisted
   profile can be classified as replaced.
6. **Unsafe identical replacement.** Semantic revision identity does not change
   for an identical reapply, candidate engine IDs are deterministic, and a new
   registration can overwrite an existing registration without first owning or
   retiring its process.
7. **Repair does not heal.** Consistency and repair report
   `definition_reapply_required`, but the reapply path is not yet a safe
   non-conflicting runtime repair.
8. **Abrupt-process leakage.** POSIX toolbox workers lack parent-death/process
   containment, runtime registrations are not persisted, and the external
   orphan scan does not include `hosting.toolbox_executor_ipc`.
9. **Target validation is x86-only.** Target regexes, defaults, catalogs, and
   runtime checks do not model Linux, macOS, or Windows ARM64 correctly.
10. **Approval authority mismatch.** The contract describes dependency approval
    as distinct parent authority, while an ordinary worker user can currently
    request and mint the approval through the deployment channel.
11. **False end-to-end acceptance.** Builder tests hand-construct complete
    resolved inputs and setup tests use materializer doubles, so the real
    daemon/control-channel path was never proven.

## Required add/remove behavior

### Adding a tool with additional packages

1. The consumer reads the active definition and submits the complete replacement
   with the new tool, source, imports, and package requirements.
2. Planning analyzes all tools, selects the smallest compatible active template,
   or derives a custom environment from an allowed base.
3. The daemon resolves a complete exact lock and compatible artifact closure for
   its current target. Missing, ambiguous, denied, incompatible, or unavailable
   artifacts fail during planning/setup, before worker spawn.
4. If policy requires review, approval binds the exact plan, definition, target,
   complete custom lock, artifact digests, catalog revision, and policy revision.
5. Apply builds or reuses the immutable environment, probes every required
   import, warms a unique candidate worker, and atomically publishes the new
   complete route map.

### Removing a tool or package

The daemon never uninstalls a package from a live environment.

- Removing a tool means omitting it from the next complete definition.
- Removing or changing a remaining tool's dependency means changing its source
  and dependency intent in that definition.
- Planning recomputes requirements across every remaining tool. Tools that still
  resolve to the same environment retain it; tools whose complete lock changes
  move to a newly materialized or already cached immutable environment.
- Publication removes the route atomically, drains replaced/removed workers,
  releases their environment references, and leaves physical deletion to
  grace-period GC.
- If no remaining tool needs a custom delta, the replacement resolves back to a
  built-in or administrator template.
- Removing a package from a named template creates and activates a new immutable
  template revision. Existing definitions remain pinned until explicitly
  replanned and applied.

## Target and artifact behavior

Initial supported host targets after this plan are:

- CPython 3.12 Windows x64 and Windows ARM64;
- CPython 3.12 Linux glibc x64 and Linux glibc ARM64;
- CPython 3.12 macOS ARM64.

Target detection must use the interpreter's compatible packaging tags rather
than infer every non-Windows host as Linux x64. Internal target identity must be
canonical and must distinguish ABI, OS, architecture, and minimum platform tag
where wheel compatibility requires it.

Online setup may contact only configured HTTPS indexes/artifact origins using
daemon-owned credentials and bounded downloads. Air-gapped setup resolves only
from configured local or imported immutable artifacts. If any required built-in
package lacks a compatible verified artifact, standard hosting setup reports a
stable bounded error and the daemon does not enter ready service.

No prebuilt venv is portable or accepted across hosts. Artifact bundles may
transport signed manifests, locks, and wheels, but environments are always
constructed and verified on the destination daemon host.

## Itemized corrective work

Every slice must declare focused tests before implementation, update durable
documentation with behavior changes, pass `git diff --check`, and be committed
separately. A checkbox is completed only after its production boundary—not a
test double—passes.

### R0 - Documentation reset

- [x] **R0-01** Replace the obsolete long plan and historical status ledger with
  this corrective plan and compact current-state ledger.
- [ ] **R0-02** Update the normative toolbox contract and worker architecture as
  each implementation slice establishes replacement behavior.

### R1 - Current-target platform model

- [ ] **R1-01** Add one canonical current-host target detector based on Python
  ABI and compatible packaging tags. Remove duplicated OS-name defaults.
- [ ] **R1-02** Support Windows x64/ARM64, Linux glibc x64/ARM64, and macOS
  ARM64 in catalog, policy, configuration, lock, materializer, and cache models.
- [ ] **R1-03** Reject cross-target artifacts and unsupported Python/OS targets
  with stable setup diagnostics.
- [ ] **R1-04** Add native platform test jobs; do not claim sandbox support from
  emulation-only results.

### R2 - Target-local built-in construction

- [ ] **R2-01** Replace shipped realized cross-target manifests with strict
  built-in intent resources for `core` and `py-compute`.
- [ ] **R2-02** Load toolbox project/artifact configuration in the real daemon
  startup path and validate it before accepting clients.
- [ ] **R2-03** Implement controlled current-target dependency resolution and
  artifact acquisition with exact transitive pins, hashes, provenance, bounds,
  and policy enforcement.
- [ ] **R2-04** Construct, sign/attest, publish, materialize, probe, and prewarm
  current-target built-in revisions during setup.
- [ ] **R2-05** In air-gapped mode, use only approved local/imported artifacts;
  fail startup when a required compatible artifact is absent.

### R3 - End-to-end custom environments

- [ ] **R3-01** Extend internal plan state with the complete resolved
  base-plus-delta lock, compatible artifacts, provenance, and import obligations.
- [ ] **R3-02** Bind dependency approval to that exact complete resolution and
  revalidate every pin at apply.
- [ ] **R3-03** Pass the pinned custom resolved input through rollout to the real
  hermetic builder; remove the template-only rejection on this path.
- [ ] **R3-04** Prove add-package behavior through authenticated daemon control:
  plan, approval, materialization, import, execution, and durable result.
- [ ] **R3-05** Prove denied, missing, incompatible, corrupt, and air-gapped
  custom artifacts fail before worker spawn and leave the old revision active.

### R4 - Package and tool removal

- [ ] **R4-01** Recompute complete dependency closure after tool removal and
  after a remaining tool removes or changes package intent.
- [ ] **R4-02** Reuse unaffected profiles, replace only changed locks/policies,
  atomically remove routes, and drain removed workers.
- [ ] **R4-03** Release obsolete environment references only after publication;
  verify shared references and grace-period GC prevent premature deletion.
- [ ] **R4-04** Prove custom-to-template contraction when the last additional
  package requirement disappears.

### R5 - Safe consumer-triggered healing

- [ ] **R5-01** Normalize manifest identities at every plan/state boundary and
  add a persisted real-state unchanged-profile reuse test.
- [ ] **R5-02** Classify a missing/mismatched live registration as runtime repair
  even when semantic definition content is unchanged.
- [ ] **R5-03** Give candidates unique runtime IDs and forbid registration
  replacement without explicit ownership and cleanup.
- [ ] **R5-04** Add atomic comparison of the old runtime-binding digest as well
  as semantic revision. A competing healer must replan or return already healthy.
- [ ] **R5-05** Keep runtime-only healing out of semantic rollout history while
  preserving durable operation status and bounded diagnostics.
- [ ] **R5-06** Contain toolbox workers on daemon death on every supported OS,
  scan the correct worker command, and clean stale IPC/spec/candidate resources.
- [ ] **R5-07** Test graceful restart, abrupt daemon death, identical reapply,
  two concurrent healers, and failures on both sides of route publication.

### R6 - Administrator template construction

- [ ] **R6-01** Add an authenticated additive admin operation accepting a new
  logical template ID, exact base revision, imports, and package requirements.
- [ ] **R6-02** Use the same current-target resolver, artifact policy, builder,
  probes, and immutable publication used by custom environments.
- [ ] **R6-03** Keep publication inactive until explicit activation; support
  prewarm, deprecate, revoke, and immutable replacement.
- [ ] **R6-04** Allow an approved custom lock to seed construction intent, but
  re-resolve/revalidate it and never promote automatically.

### R7 - Authority and consumer compatibility

- [ ] **R7-01** Preserve existing normal consumer definition payloads, operation
  refs, logical template IDs, and opaque digest semantics.
- [ ] **R7-02** Decide and implement the distinct dependency-approver authority.
  If ordinary worker approval is removed, publish the complete migration in the
  breaking-change handoff before release.
- [ ] **R7-03** Keep raw immutable template publication for existing admins and
  add construction as a new API rather than changing the old payload in place.
- [ ] **R7-04** Validate `mp13-docs` add/remove, approval, restart reapply,
  operation recovery, projection, and empty-definition behavior against the
  real parent daemon.

### R8 - Acceptance and closeout

- [ ] **R8-01** Add one no-double end-to-end suite covering configured daemon
  startup, built-ins, custom add/remove, restart healing, and GC.
- [ ] **R8-02** Run focused parent tests, native platform tests, complete parent
  regression, and affected dependent tests with exact results in the status
  ledger.
- [ ] **R8-03** Reconcile plan, status, durable contracts, setup documentation,
  worker architecture, and any breaking-change handoff with actual code.
- [ ] **R8-04** Check every acceptance item only from reviewable production-path
  evidence and commit the final audit separately.

## Acceptance criteria

- [ ] Real daemon setup constructs and probes `core` and `py-compute` for only
  its current target.
- [ ] Missing required artifacts stop air-gapped setup with a stable error.
- [ ] A new tool can add a previously absent allowed package through the normal
  control channel and execute from a verified isolated environment.
- [ ] Removing a tool or package requirement never mutates a live environment,
  preserves shared users, and eventually garbage-collects unreferenced state.
- [ ] Custom locks are complete, exact, approval-bound when required, and do not
  inherit host or base-environment packages.
- [ ] Linux ARM64, macOS ARM64, and Windows ARM64 pass native setup, worker,
  package, sandbox, restart, and cleanup tests.
- [ ] An identical consumer reapply after restart heals missing workers without
  changing semantic definition revision or conflicting with another healer.
- [ ] Graceful and abrupt daemon termination leave no live unowned toolbox
  worker, stale routable registration, partial cache publication, or leaked
  candidate reference.
- [ ] Normal consumers do not gain template publication, artifact-source,
  interpreter, filesystem-path, or package-install authority.
- [ ] Every client-visible break, if any, has complete migration instructions
  and dependent adoption evidence before the handoff is reset again.
