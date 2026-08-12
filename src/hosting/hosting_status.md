# Unified hosting cutover status

Last updated: 2026-08-12

Status: R9.6 consumer adoption gated; R9.7 sensitive Windows lane gated

This is the current execution ledger for
[`hosting_access_plan.md`](hosting_access_plan.md). Detailed completed-slice
narratives were intentionally removed to keep resume context small; they remain
available in Git history. This file records only the current gate, open work,
compact evidence index, and external completion gates.

## 1. Current gate and resume order

**R9.6 consumer adoption is gated; every locally executable R9.7 lane is now
green.** The repository aggregate, actionable lint, declared type check,
security/redaction proof, relay-equivalent lane, and five-target external native
workflow pass. The opt-in Windows sensitive-sandbox lane still requires an
external engine model or configuration.

Resume in this order:

1. Obtain the dependent owner/revision/test adoption receipt.
2. Run the external sensitive-sandbox lane.
3. Reconcile all remaining evidence and close R9.8.

Do not repeat the repository aggregate unless code changes after the recorded
R9.7 run or a remaining lane exposes a regression.

## 2. Progress

| Work | Status | Remaining boundary |
|---|---|---|
| R0 Contract freeze | Complete | Final tool-change schemas were revised in place before adoption. |
| R1 Inventory/handoff | Complete | Refresh removal scans and dependent receipt at R9. |
| R2 Shared paths/setup | Complete | Final aggregate/platform proof at R9.7. |
| R3 Unified configuration | Complete | Final aggregate/platform proof at R9.7. |
| R4 Single startup path | Complete | Final aggregate/platform proof at R9.7. |
| R5 Generic packages | Complete | Final aggregate/platform proof at R9.7. |
| R6 Generic environments | Complete | Final aggregate/platform proof at R9.7. |
| R7 Toolbox adoption | Complete | Public acceptance/removal proof at R9. |
| R8 Worker-neutral state | Complete | Final local aggregate proof recorded at R9.7. |
| R9 Acceptance/handoff | Partial | Dependent receipt, sensitive Windows sandbox, and closeout remain. |

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

### R9 closeout

- Obtain the dependent implementation pin/receipt.
- Run the opt-in sensitive Windows sandbox lane.
- Reconcile external evidence and close R9.8.

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
| CODE/TEST-R7.2H | Publish consumes only retained warmed runtime state and emits the existing apply result without rebuild/spawn; discard atomically becomes terminal before retiring non-active workers/references, and both deny in-flight leases; focused lifecycle/transport lane passed 13 tests. |
| CODE/TEST-R7.2I | Recovery loads durable candidate truth before reconciling workers, preserves complete ready candidates, expires missing-worker candidates, cleans only non-active workers/references, and validates live generic reference/environment/package-lock identity; expanded R7.2 matrix passed 190 tests (one recorded real-daemon dependency fixture deselected). |
| CODE/TEST-R7.3A | Generic environment authority exposes a versioned protection snapshot; toolbox references/consistency/repair/GC consume durable candidate truth and explicit candidate→engine/reference plus generic reference/execution cross-links; focused maintenance lane passed 34 tests. |
| TEST-R7.3B | Generic GC and explicit removal preserve toolbox/other-worker references, active executions, and live candidate leases; environment/maintenance lane passed 31 tests. |
| CODE/TEST-R7.3C | Gate and consistency fail closed for released declared generic references; review exposes generic protection counts; offline v1 archive refuses newer lifecycle repositories and excludes shared state; R7.3 authority lane passed 54 tests. |
| CODE/TEST-R9.1 | Channel/daemon/CLI candidate commands retain frozen payloads; capabilities advertise tool-change and candidate support; `HostedToolBoxRef` exposes typed atomic changes, candidate preparation, and reconnectable candidate sessions; public/auth/CLI lane passed 117 tests. |
| TEST-R9.3 | Package/environment lifecycle, atomic tool changes, candidate prepare/renew/execute/publish/discard/expiry, maintenance, and durable-operation concurrency/restart/no-double lane passed 158 tests; the authenticated daemon fixture now uses strict v3 configuration and the frozen confirmation projection. |
| CODE/TEST-R9.4A | Removed the `toolbox-gc` command/method alias; `hosting-gc` now uses the frozen `hosting_gc`, `host_scope=hosting`, mark/sweep phase contract and generic terminal result. Toolbox readiness no longer emits the four removed configuration-code family names. Focused command/auth/operation/docs lane passed 126 tests. |
| CODE/TEST-R9.4B | `environment-remove` now exclusively enters the durable generic operation with stable request ID, `environment_id` selector, full blocker checks, generic result contract, and op-start-only daemon/channel/CLI dispatch; the orphaned toolbox-prefixed service path was removed. Focused lifecycle lane passed 86 tests plus the exact daemon proof. |
| CODE/TEST-R9.4C | Worker startup specs and hosted references no longer carry control-state or host paths; remote authentication and metrics omit host paths and interpreter details; strict v3 sandbox fixtures replace the removed startup inputs. Focused sandbox, worker, security, and cutover lane passed 141 tests with 1 skipped. |
| CODE/TEST-R9.4D | Removed the `engines_state_file` startup override across service, daemon, channel, CLI, relay, worker spec, and hosted-reference serialization. The registry path is now derived from v3 configuration, while the old CLI/profile/spec fields fail fast. Focused configuration, daemon, client-realm, CLI, and sandbox lane passed 163 tests. |
| TEST-R9.4E | Removed 108 redundant engine-state overrides from workflow-helper fixtures that already use strict v3 configuration. The file reached 115 passes with one existing concurrency timing failure; the exact failed process-backed concurrency test passed on immediate rerun. |
| TEST-R9.4F | Removed obsolete engine/control-state fields from interactive CLI fixtures; the complete interactive menu test file passed 40 tests. |
| CODE/TEST-R9.4G | Removed the engine-state attachment surface from app helpers, hosted chat demo, and `mp13chat`; existing toolbox attachment now takes only the top-level configuration. Removed redundant state keywords from modern fixtures. The focused modern hosting lane passed 337 tests with 1 skipped, and app helper coverage passed 20 tests. |
| TEST-R9.4H | Hermetic builder fixtures now test rejection using only the removed toolbox map and no longer model engine/control-state paths; the full file passed 11 tests. |
| TEST-R9.4I | Toolbox host-project semantic tests now install their catalog intent explicitly after strict v3 service startup; removed setup-summary projections stay absent and revision invalidation remains covered. The file passed 6 tests. |
| TEST-R9.4J | Removed the legacy signed-bundle daemon/setup, HTTPS-acquisition, and pre-generic R7 acceptance modules that depended on the five deleted startup maps and mandatory signing baseline. Native CI now exercises strict-v3 definition resolution; the replacement package/definition/cutover/workflow lane passed 21 tests. |
| DOC/TEST-R9.4K | Permanent worker guidance now pins templates through generic package locks and environment receipts and describes verification as optional policy, not a signed-manifest baseline. Contract/removal documentation coverage passed 13 tests. |
| CODE/TEST-R9.4L | Toolbox template semantic provenance now uses an evidence digest and optional verifier ID; required signing-key identity and readiness trust-key gating were removed. Catalog, dependency, materialization, definition-resolution, and host-transition coverage passed 85 tests. |
| CODE/TEST-R9.4M | Catalog publication now stores optional verification evidence instead of a required manifest signature. Generic templates publish with no verifier/evidence, while configured verifier evidence remains validated. Focused catalog, resolver, prewarm, transition, and definition coverage passed 61 tests. |
| CODE/TEST-R9.4N | Base-template definition planning no longer requires the removed toolbox host-project configuration. Source metadata and all confirmation/approval/apply/candidate pins now derive from v3 configuration and its generic package source set; custom exact-wheel resolution remains separately explicit. Focused planning, candidate, routing, cutover, and transport coverage passed 34 tests. |
| CODE/TEST-R9.4O | Custom exact-wheel planning now resolves only from the generic package manager's daemon-hashed CAS and source filename index under strict v3 source/policy authority. Confirmation, locks, and real venv materialization consume those same bytes without a toolbox-store copy or signed-config fixture; package/resolution and candidate/routing/revision lanes passed 62 tests. |
| CODE/TEST-R9.4P | Removed the unreachable `toolbox_setup` execution kind, phases, reconciliation/worker service path, daemon diagnostic, and normative contract section. Template/package/environment lifecycle remains on generic operations; operation, service, contract-doc, and cutover coverage passed 53 tests. |
| CODE/TEST-R9.4Q | Removed the unreachable configured-startup resolve/prepare/batch-publish helpers and their obsolete host-project semantic test module. Active catalog publication, prewarm, generic resolution, cutover, and strict-v3 definition planning coverage passed 60 tests. |
| CODE/TEST-R9.4R | Required-template readiness now derives from the active catalog and unified configuration revision; maintenance removal/GC/fingerprints use generic environment retention and the unified revision rather than host-project retention/source pins. Catalog, prewarm, environment, maintenance, and cutover coverage passed 60 tests. |
| CODE/TEST-R9.4S | Removed the orphan toolbox-specific template constructor and its host-project config repository, signed artifact store/acquirer, built-in resolver/template adapters, private host fields, and legacy resolver fixtures. The supported generic environment-template API remains; plan, rollout, catalog, maintenance, definition, and revision coverage passed 97 tests. |
| CODE/TEST-R9.4T | Removed the obsolete `toolbox_artifact_import` hosted-operation kind, selector namespace, progress phases, cancellation branch, and contract fixture. Generic package upload/commit and environment operations remain; focused operation/package/environment coverage passed 64 tests. |
| DOC/TEST-R9.4U | Replaced the permanent contract's signed bundle/setup/built-in authority with the shipped generic package CAS/index, locks, active catalog, prewarm, and readiness behavior; removed the retired host-config state filename and mandatory-signing fixture vocabulary. Contract/catalog/plan/definition/rollout coverage passed 57 tests. |
| TEST-R9.4V | Final production/test/doc vocabulary and constructor scans leave only explicit rejection, absence, and historical handoff matches. Removed-surface, strict-startup, generic-root, CLI/settings rejection, builder, contract, and definition-resolution coverage passed 102 tests. |
| DOC/TEST-R9.5 | Configuration, setup, startup/CLI, security/access, generic package/environment, toolbox/candidate, and worker guidance now describe only shipped v3 and generic lifecycle behavior. Executable durable-doc, public guarantee, startup, configuration, and setup coverage passed 56 tests. |
| TEST-R9.7A | After the final local type/lint remediation slices, `python -m pytest -q` passed 1257 tests with 2 identified environment/platform skips. |
| TEST-R9.7B | The Windows x64 native probe confirmed `cp312-win_amd64` and imported `pydantic_core` from a matching native extension; restart healing, definition resolution, and target coverage passed 20 tests. Relay autostart/configuration, strict-v3 startup, and callback-lease coverage passed 25 tests. |
| CODE/TEST-R9.7C | Declared actionable lint policy preserves fatal/error and useful warning checks while documenting intentional mixin/runtime architecture exceptions; `python -m pylint src/hosting` passed at 10.00/10. |
| CODE/TEST-R9.7D | The declared mypy lane checks `src/hosting`, follows imported dependencies in skip mode, and disables only the deliberate sibling-mixin `attr-defined` pattern; `python -m mypy` passed all 130 source files. |
| TEST-R9.7E | Focused authentication, daemon-state, operation/result, callable-surface, host-capability, catalog, and identity security/redaction proof passed 172 tests. |
| CODE/TEST-R9.7F | Draft PR #2 executed native workflow run `31605912784` at `cb39c0ee73d0beb7f1b852b4e4a9e80a5220883e`; Windows x64/ARM64, Linux x64/ARM64, and macOS ARM64 all passed. The first Windows ARM64 attempt exposed the obsolete, unused mandatory-signing `cryptography` dependency; removing it from package/workflow/lock state and regenerating the lock produced the green rerun. |
| DOC-R9.6A | The final parent handoff is pinned to accepted implementation commit `cb39c0ee73d0beb7f1b852b4e4a9e80a5220883e`; only the named dependent owner/revision/test receipt remains open in R9.6. |
| CODE-R8A–C | Versioned neutral state, Python/JS shared manager adoption, GC/repair controls passed focused coverage. |
| CODE/TEST-R9 partial | Structured auth, role/hash authority, redaction, startup modes, generic lifecycle, and retry/restart identity passed focused coverage. |
| TEST-R8.2C checkpoint | Workflow operation suite passed; full helper suite passed 116 tests on resume. |

## 6. External completion gates

- Dependent owner/revision/test receipt against parent implementation
  `cb39c0ee73d0beb7f1b852b4e4a9e80a5220883e`.
- Opt-in Windows sensitive-sandbox validation with an external engine model or
  configuration; the local aggregate skip records the exact requirement.
- R9.8 reconciliation after R9.6 and every required R9.7 lane pass.

## 7. Ledger rules

1. Record one active slice before production changes.
2. Update code, focused tests, owning permanent docs, handoff, plan checkbox, and
   one concise evidence row together when they form one contract slice.
3. Preserve the clean cut: no aliases, fallbacks, dual reads/writes, automatic
   legacy migration, or dependent-repository edits.
4. Keep this file current and compact; completed detail belongs in Git history,
   not an ever-growing transcript.
