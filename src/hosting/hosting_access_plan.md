# Unified hosting configuration and package/environment cutover plan

Status: active breaking-change plan; R9.7 acceptance complete, with R9.6
consumer adoption still gated

This is the current execution plan, not a history log. Completed-slice detail is
retained in Git history and summarized in `hosting_status.md`. Exact external
payloads belong in `HOSTING_CLIENT_BREAKING_CHANGES.md`; permanent documents
change only when their owning behavior ships.

## 1. Target and ownership

This repository owns the daemon, hosting client/channel, CLI, setup library,
configuration contracts, and package/environment implementation. The dependent
consumer repository is inspection-only; its exact adoption instructions and
receipt requirements belong in `HOSTING_CLIENT_BREAKING_CHANGES.md`.

The completed cutover must provide:

- one top-level path map for `@hosting`, `@packages`, and `@environments`;
- one strict `<config root>/hosting/hosting_config.json` authority for static
  control, package, environment, and lifecycle policy;
- a single daemon startup input: the top-level configuration location;
- server-side role authorization and daemon-computed SHA-256 package identity;
- worker-neutral package acquisition, locking, environment construction,
  references, reuse, verification, and garbage collection;
- toolbox-specific definition, review, candidate, publication, and execution
  APIs over that shared subsystem; and
- a clean major-version break: removed files, fields, commands, state formats,
  roots, and aliases fail fast without fallback or automatic migration.

Secrets, audit logs, mutable state, scratch data, packages, and built
environments remain separate records/data, not additional configuration
authorities:

```text
<hosting root>/keyring|audit|state|scratch
<packages root>/artifacts|locks
<environments root>/...
```

## 2. Locked architecture

### 2.1 Paths and configuration

- Extend the existing `category_dirs`/`PathResolver` model with
  `hosting_root_dir`, `packages_root_dir`, and `environments_root_dir`.
- Normal configuration uses logical references. Persistent roots may anchor to
  stable `@home`, `@config`, or `@temp` locations, never `@project`.
- `hosting_config.py` stays a thin entry point; the importable hosting setup
  library owns safe host-local root/config planning and application.
- `hosting.configuration.v3` is strict. It contains `control`,
  `package_management`, and `environment_management`; it never restores the
  five toolbox startup maps or embeds toolbox definitions.
- Static policy changes take effect after deliberate daemon restart. Pin the
  active configuration revision into long-running plans and operations so
  changed policy makes stale work explicit.
- Candidate lifecycle policy is host-local:
  `toolbox_candidate_retention_ms` defaults to `1800000` and permits
  `300000`–`14400000`; `toolbox_candidate_limit_per_actor` defaults to `3` and
  permits `1`–`16`.

### 2.2 Authority, packages, and environments

- The authenticated server-side role authorizes package, environment, toolbox,
  and worker mutations; equivalent roles granted by password or public key have
  equivalent authority.
- Publisher signatures are optional policy, not a baseline requirement.
- The daemon streams ingress to bounded scratch storage, computes SHA-256 from
  received bytes, checks any caller expectation, atomically promotes complete
  content, creates an exact lock, and records secret-free audit/receipt data.
- Shared contracts remain worker-neutral: `PackageSource`, `PackagePolicy`,
  `EnvironmentTemplate`, `EnvironmentRequest`, `EnvironmentLock`,
  `EnvironmentReceipt`, `EnvironmentReference`, and `EnvironmentManager`.
- References carry `consumer_kind`, `consumer_id`, and `revision`. Toolboxes are
  one consumer kind; toolbox code still owns toolbox definition semantics.
- Generic public command families are `package-artifact-upload-*`,
  `environment-template-*`, and `environment-remove`. Their toolbox-owned
  predecessors are removed rather than aliased.

### 2.3 Tool changes, selective rejection, and candidates

Keep `hosting.control.v3`, `hosting.toolbox.definition_plan.v2`, and
`hosting.toolbox.confirmation_receipt.v1`. The revised plan/receipt shapes were
not adopted externally, so R7.0 replaces them in place rather than inventing a
new record version.

The complete definition remains authoritative and `toolbox-plan-definition`
remains the advanced surface. The richer host contract additionally provides:

- `toolbox-plan-tool-changes`: atomic add/update/rename/remove batches merged
  server-side with stable change IDs and per-tool import, distribution,
  evidence, package, and environment analysis;
- `toolbox-revise-definition-plan`: immutable child replanning after selective
  rejection, with full dependency/lock recomputation and explicit preserved,
  skipped, and cascade outcomes; it never mutates source, its parent, or a lock;
- confirmation only for the reviewed exact plan; package/import rejection at
  confirmation returns `tool_change_revision_required`; and
- one-shot apply plus a try-before-publish lifecycle: prepare, get, renew,
  execute, publish, and discard.

Candidate validation reuses ordinary `toolbox_worker` candidate routing, not a
new worker kind or generic execution endpoint. Execution uses the same sandbox,
tool gates, host-API/data/network approvals, timeout, cancellation, callbacks,
and audit policy as an active tool; it is not a dry run. Publication revalidates
all pins and publishes the exact warmed candidate without rebuilding.

Preparation may request a lifetime; renewal requires one. Each requested window
is host-bounded to 5 minutes–4 hours, defaults to 30 minutes, and may be renewed
repeatedly while authorization and pins remain current. Dispatch acquires an
in-flight execution lease so expiry cleanup cannot retire candidate resources
under a long-running tool; normal execution timeout remains independent and
decisive. Exact payloads, results, errors, and side-effect language are frozen
only in `HOSTING_CLIENT_BREAKING_CHANGES.md`.

## 3. Progress and resume gate

| Work | State | Remaining boundary |
|---|---|---|
| R0–R1 freeze/inventory | Complete | Refresh handoff/removal evidence at R9. |
| R2 paths/setup | Complete | Final aggregate/platform evidence at R9.7. |
| R3 unified configuration | Complete | Final aggregate/platform evidence at R9.7. |
| R4 startup cut | Complete | Final aggregate/platform evidence at R9.7. |
| R5 generic packages | Complete | Final aggregate/platform evidence at R9.7. |
| R6 generic environments | Complete | Final aggregate/platform evidence at R9.7. |
| R7 toolbox adoption | Complete | Public acceptance/removal evidence at R9. |
| R8 neutral state/workers | Complete | One conditional focused checkpoint below. |
| R9 acceptance/handoff | Partial | Public surfaces, matrices, removals, docs, receipt. |

Resume in this order:

1. Obtain the R9.6 dependent-owner adoption receipt.
2. Reconcile that receipt, then perform R9.8 closeout. Do not repeat locally
   green lanes unless intervening code changes or an external lane exposes a
   regression.

Completed foundations that remain usable include shared roots/setup, strict v3
configuration loading, single-path startup, generic package ingress/locks,
generic environments/references/commands, atomic candidate publication bridges,
versioned worker-neutral state, Python/JS shared-manager adoption, structured
authentication, security/redaction proofs, and startup-mode coverage. Existing
R7 plan/confirmation field assertions are provisional and must be replaced.

## 4. Remaining work

### R3/R4/R6 — finish the P0 clean cut

- [x] **R3.2** Remove every remaining production reader, fixture, example, and
  document that treats `access_control.json` as an authority. Prove the strict
  repository and setup flow are the only path.
- [x] **R4.1** Capture and validate a stable configuration revision in every
  long-running plan/operation affected by restart or policy change.
- [x] **R4.2** Remove `--toolbox-config-file`,
  `engine_host_toolbox_config_file`, ephemeral launcher JSON behavior, and all
  remaining help/fixture vocabulary. Reject old arguments; do not absorb them
  through permissive `**kwargs`.
- [x] **R6.5** Remove final legacy environment receipt/reference readers and
  compatibility aliases. Legacy roots must not affect resolution, reuse,
  references, GC, or execution.

### R7.1 — plan tool changes and selective revision

- [x] Resolve exact generic package locks and an `EnvironmentRequest` during
  toolbox definition planning.
- [x] Implement compare-and-swap server-side tool-change merge, deterministic
  change IDs, atomic rename, and strict add/update/remove validation.
- [x] Emit bounded per-tool import/source evidence, mapped distributions,
  environment grouping, exact package mutations, and approval requirements.
- [x] Build immutable child plans for selective rejection, recomputing the
  complete closure and locks and reporting preserved/skipped/cascade outcomes.
- [x] Replace provisional plan/confirmation field-level assertions while
  retaining the frozen v2/v1 identifiers and exact handoff schemas.
- [x] Prove stale approval, source mutation, retry, restart, and concurrent
  revision cannot execute unapproved bytes or leak references.

### R7.2 — materialize, try, and publish exact candidates

- [x] Make materialization consume immutable generic environment
  receipts/references while preserving runtime, proxy, sandbox, exposure, and
  execution constraints.
- [x] Split existing candidate rollout at its pre-publication boundary into
  durable prepare/get/renew/execute/publish/discard operations.
- [x] Add requested lifetime, repeated renewal, in-flight execution leases,
  quotas, expiry/discard cleanup, restart recovery, and stale authorization,
  definition, environment, package-lock, and configuration-pin checks.
- [x] Add the frozen retention/limit fields to strict v3 lifecycle validation
  and sanitized health without creating another configuration authority.
- [x] Reuse normal `toolbox_worker` protocol and candidate routing; prove a
  candidate route is never visible through active toolbox execution.
- [x] Publish the exact warmed candidate without rebuild/reresolution, retain
  one-shot apply, and keep handoff result/error/no-double semantics exact.

Frequent-edit acceptance must exercise source-only edits with environment
reuse, atomic multi-tool add/update/rename/remove, selective import denial and
child replan, candidate try-out, long execution across nominal expiry, repeated
renewal, exact publication, discard, expiry, restart, and stale pins. Candidate
execution must visibly use ordinary effect approvals and never be described as
side-effect-free.

### R7.3 — adapt toolbox maintenance

- [x] Route toolbox consistency, gate, reconcile, repair, review snapshot,
  references, GC, and archive behavior through generic package/environment
  operations where appropriate.
- [x] Keep toolbox-only and generic shared state in separately versioned
  repositories with explicit cross-references.
- [x] Prove maintenance cannot remove content referenced by a toolbox, another
  worker kind, an active execution, or a live candidate lease.

### R9 — public acceptance and handoff

- [x] **R9.1 Public surfaces:** align channel, API, CLI, help, capabilities,
  request/result/error shapes, and add typed high-level tool-change and
  candidate-session methods while retaining advanced full-definition planning.
- [x] **R9.3 Lifecycle/no-double:** cover upload, resolution, build/reuse,
  replacement, release, removal, repair, GC, toolbox/second-worker concurrency,
  the frequent-edit candidate matrix above, disconnect/retry/restart, and prove
  one authorized logical request causes at most one durable/execution effect.
- [x] **R9.4 Removal:** run the R1 searches over production, tests, docs, and
  examples; remove old fields, commands, aliases, codes, contract IDs,
  filenames, roots, fallbacks, and mandatory-signing language. Inspect any
  intentionally retained historical match.
- [x] **R9.5 Permanent docs:** update configuration, setup, startup/CLI,
  security, package/environment, toolbox, candidate, and worker guidance only
  as behavior ships. Executable examples must pass; permanent docs must not
  depend normatively on the transient plan/status/handoff.
- [ ] **R9.6 Consumer adoption:** deliver the final handoff and record the
  dependent owner, implementation revision, and tests. A daemon shim is not an
  adoption receipt.
  - [x] Final parent handoff is frozen against implementation commit
    `4d01307f664366c3149bef539aaa1b4e3f98a82f`.
  - [ ] Named dependent owner, full dependent revision, and reproducible test
    receipt remain required.
- [x] **R9.7 Full matrix:** after removals, run required aggregate, lint, type,
  integration, Windows/POSIX, relay-equivalent, and affected native/platform
  lanes. Record commands/results; skipped required lanes remain open with owner
  and reason.
  - [x] Local aggregate: `python -m pytest -q` passed 1257 tests with the two
    declared platform/external-resource skips.
  - [x] Lint: `python -m pylint src/hosting` passed at 10.00/10.
  - [x] Type: `python -m mypy` passed all 130 hosting source files.
  - [x] Focused security/redaction proof passed 172 tests.
  - [x] Windows x64 native and relay-equivalent lanes passed as recorded in
    `hosting_status.md`.
  - [x] External native workflow run `31608128010` passed Windows x64/ARM64,
    Linux x64/ARM64, and macOS ARM64 at implementation commit
    `4d01307f664366c3149bef539aaa1b4e3f98a82f`.
  - [x] Opt-in Windows sensitive-sandbox lane passed with a real external tiny
    Llama model under a restricted token, Low Integrity, and a job object.
- [ ] **R9.8 Closeout:** reconcile every remaining checkbox and every target or
  locked rule in Sections 1–2 with evidence, verify schema/capability/docs/
  handoff agreement, verify no compatibility code escaped review, and mark
  complete only after R9.6/R9.7 pass.

## 5. Navigation map for remaining work

Search call sites before changing a contract.

| Boundary | Primary seams |
|---|---|
| Configuration remnants/pins | `src/hosting/hosting_setup_api.py`; `src/hosting/hosting_config_cli.py`; `src/hosting/service/host_service.py`; `src/hosting/daemon/foreground.py`; `src/hosting/daemon/background.py`; `src/hosting/daemon/local_ipc.py` |
| Client/startup remnants | `src/hosting/engine_host_channel.py`; `src/hosting/engine_host_cli.py`; `src/hosting/transport_bootstrap_api.py`; `src/app/hosted_chat_demo.py` |
| Legacy environment records | `src/hosting/toolbox/environment.py`; `src/hosting/toolbox/hermetic_environment.py`; `src/hosting/toolbox/bundle_models.py`; `src/hosting/sandbox/toolbox_runtime.py` |
| Tool-change planning | `src/hosting/toolbox/hosted_ref.py`; `src/hosting/toolbox/definition_planner.py`; `src/hosting/service/toolbox_plans.py`; `src/hosting/service/toolbox_confirmations.py` |
| Generic plan/materialization bridge | `src/hosting/toolbox/orchestration.py`; `src/hosting/toolbox/staging.py`; `src/hosting/service/toolbox_materialization.py`; `src/hosting/service/toolbox_runtime.py` |
| Candidate lifecycle | `src/hosting/service/toolbox_rollout.py`; `src/hosting/service/toolbox_runtime.py`; `src/hosting/service/toolbox_materialization.py`; `src/hosting/service/toolbox_state_v2.py` |
| Maintenance/shared state | `src/hosting/service/proxy.py`; package/environment repositories; toolbox consistency/reconcile/repair/GC services |
| Public surfaces | `src/hosting/engine_host_channel.py`; `src/hosting/engine_host_cli.py`; `src/hosting/daemon/local_ipc.py`; capability/version declarations |
| Permanent docs/tests | `CONFIG.md`; `src/hosting/HOSTING_CONFIG_SCRIPT.md`; `src/hosting/HOSTED_TOOLBOX_CONTRACT.md`; focused hosting tests and removal scans |

## 6. Execution and evidence rules

1. Record one active slice in `hosting_status.md` before production changes.
2. A slice closes only with focused positive/negative proof and coordinated
   updates to code, owning tests/docs, handoff, this plan, and the compact status
   evidence row where applicable.
3. Do not edit the dependent repository, introduce fallbacks/aliases/dual
   reads/writes, or update permanent docs ahead of implementation.
4. Prefer focused tests during implementation. Run aggregate/platform lanes at
   R9.7 or when a slice specifically needs broader regression evidence.
5. Treat exact external schemas in `HOSTING_CLIENT_BREAKING_CHANGES.md` as the
   implementation oracle for R7/R9; do not duplicate them here.
6. Keep this file current: remove completed checklist detail after preserving a
   concise status/evidence summary and rely on Git history for the transcript.

## 7. Explicitly out of scope

- compatibility readers, shims, aliases, dual commands, or automatic legacy
  environment reuse/migration;
- editing the dependent repository;
- mandatory publisher signatures or private-key distribution in the baseline;
- remote relocation of host roots or embedded toolbox definitions in static
  configuration;
- trusting caller hashes without hashing received bytes;
- credential values in CLI arguments, logs, audit, receipts, or remote status;
- a new candidate worker kind or generic code-execution endpoint; and
- cross-filesystem transaction claims without journaled recovery.
