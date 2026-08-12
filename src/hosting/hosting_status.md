# Unified hosting cutover status

Last updated: 2026-08-11

Status: active; R0–R3.1 and R5 complete; R3.2–R4 active

This is the fresh execution ledger for
[`hosting_access_plan.md`](hosting_access_plan.md). The prior toolbox corrective
ledger and its test transcript remain available in Git history. Their completed
work may be useful implementation foundation, but it is not evidence that the
new unified configuration, generic APIs, state cut, or dependent adoption is
complete.

## 1. Current gate

Current continuous block: C — implementation and acceptance (`high`, P0–P2)

Active slice: R6 generic environment subsystem (`high`, P0), with R3.2–R4
aggregate acceptance still open

R0.1–R0.7, R1.1–R1.5, R2.1–R2.3, R3.1, and R5.1–R5.4 are complete. R3.2–R4
remain the active authority/startup acceptance slice; no dependent adoption is
claimed.

In-progress R3.2–R4 production state: daemon/service/foreground/background/
HTTP/relay/channel startup now accepts the top-level MP13 configuration and
loads immutable v3 policy before listener construction; static and dynamic
records are separated; legacy launcher inputs and module are removed from
production. New focused tests pass. The aggregate lane currently reports 100
failures before its configured stop after 509 passes, predominantly deliberate
legacy fixture/signature expectations. R3.2/R4 remain open until those fixtures
and remaining permanent documents are migrated and the named proofs pass.

## 2. Progress ledger

| Work | Priority | Expertise | Status | Evidence |
|---|---:|---|---|---|
| R0.1 File ownership/layout | P0 | average | Complete | DOC-R0 |
| R0.2 Root-label semantics | P0 | average | Complete | DOC-R0 |
| R0.3 Authority/artifact identity | P0 | average | Complete | DOC-R0 |
| R0.4 Generic contracts | P0 | average | Complete | DOC-R0 |
| R0.5 Commands/readiness/version | P0 | average | Complete | DOC-R0 |
| R0.6 Clean-cut state behavior | P0 | average | Complete | DOC-R0 |
| R0.7 Host-local root customization | P0 | average | Complete | DOC-R0 |
| R1.1 Exact dependent handoff | P0 | medium | Complete | DOC-R1 |
| R1.2 Production inventory | P0 | medium | Complete | DOC-R1 |
| R1.3 Tests/fixtures inventory | P0 | medium | Complete | DOC-R1 |
| R1.4 Dependent read-only inventory | P0 | medium | Complete | DOC-R1 |
| R1.5 Documentation cutover map | P0 | medium | Complete | DOC-R1 |
| R2 Shared paths/config foundation | P0 | high | Complete | CODE-R2 |
| R3 Unified hosting configuration | P0 | high | Active (legacy fixture/doc removal open) | CODE-R3.1/R4A |
| R4 Single-path daemon startup | P0 | high | Active (legacy fixture/doc removal open) | CODE-R4A |
| R5 Generic package subsystem | P0 | high | Complete | CODE-R5 |
| R6 Generic environment subsystem | P0 | high | Active (foundation complete) | CODE-R6A |
| R7 Toolbox adoption | P1 | high | Not started | — |
| R8 Worker-neutral state/operations | P1 | high | Complete | CODE-R8A/R8B/R8C |
| R9 Public surfaces/acceptance/handoff | P2 | high | Not started | — |

Do not mark a row complete merely because an older toolbox-specific
implementation exists. Record proof against the new contract and exact plan
checkboxes.

## 3. Locked direction

The current plan has locked these architectural boundaries; R0 turns them into
exact implementable contracts:

- one static authority at `<config root>/hosting/hosting_config.json`;
- top-level `category_dirs` roots and `@hosting`, `@packages`, and
  `@environments` labels;
- host-local setup/config library writes static configuration; daemon reads it;
- static policy is activated by deliberate restart;
- dynamic package, template, environment, tool, and toolbox management remains
  available through the control channel after startup;
- authenticated server-side role/effective scope authorizes mutations;
- daemon-computed SHA-256 identifies received artifacts;
- mandatory publisher signing is absent from the baseline;
- one worker-neutral package/environment subsystem serves toolbox, Python
  workflow helper, and JavaScript/Node worker consumers; and
- clean major-version cut with no compatibility readers, aliases, or legacy
  environment reuse.

## 4. R1 owned inventory

This is the R1.2–R1.5 implementation, test, dependent, and documentation map.
Searches covered production, tests, docs, examples, re-exports, constructor
calls, command strings, readiness codes, serialized contracts, and legacy
paths. R2–R9 owners must repeat the relevant searches at completion because
this inventory is a starting snapshot, not zero-result proof.

### Configuration and startup seams

- `src/mp13_engine/mp13_config_paths.py`
- `src/app/config.py`
- `src/app/mp13chat.py`
- `hosting_config.py`
- `src/hosting/hosting_config_cli.py`
- `src/hosting/hosting_setup_api.py`
- `src/hosting/transport_bootstrap_api.py`
- `src/hosting/daemon/foreground.py`
- `src/hosting/daemon/background.py`
- `src/hosting/daemon/local_ipc.py`
- `src/hosting/daemon/toolbox_launch_config.py`
- `src/hosting/engine_host_channel.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/service/host_service.py`
- `src/hosting/service/constants.py`
- `src/hosting/service/state.py`

### Package/environment/toolbox seams

- `src/hosting/toolbox/host_project_config.py`
- `src/hosting/toolbox/dependency_policy.py`
- `src/hosting/toolbox/environment.py`
- `src/hosting/toolbox/hermetic_environment.py`
- `src/hosting/toolbox/bundle_models.py`
- `src/hosting/service/toolbox_artifact_store.py`
- `src/hosting/service/toolbox_artifact_uploads.py`
- `src/hosting/service/toolbox_artifact_upload_service.py`
- `src/hosting/service/toolbox_catalog.py`
- `src/hosting/service/toolbox_definition_resolution.py`
- `src/hosting/service/toolbox_host_config_state.py`
- `src/hosting/service/toolbox_materialization.py`
- `src/hosting/service/toolbox_runtime.py`
- `src/hosting/sandbox/runtime_base.py`
- `src/hosting/sandbox/python_runtime.py`
- `src/hosting/sandbox/js_runtime.py`
- `src/hosting/sandbox/toolbox_runtime.py`

### Known dependent seams (inspection only)

- `src/backend/platform/hosting/hosting_admin.py`
- `src/backend/platform/capabilities/parent_truth.py`
- `src/backend/platform/toolboxes/definition_coordinator.py`
- `src/ui/web/static/js/features/chat/CapabilityToolsPanel.js`
- `tests/backend_infra/test_parent_toolbox_truth.py`

### 4.1 Production ownership by implementation slice

| Owner | Production files/symbol families |
|---|---|
| R2 shared roots/setup | `src/mp13_engine/mp13_config_paths.py`; `src/app/config.py`; `src/app/mp13chat.py`; `hosting_config.py`; `src/hosting/hosting_setup_api.py`; `src/hosting/hosting_config_cli.py`; `src/hosting/transport_bootstrap_api.py`; `src/app/hosted_chat_demo.py` |
| R3 unified configuration | `src/hosting/service/constants.py`; `service/host_service.py`; `service/state.py`; `daemon/diagnostics.py`; the configuration readers/writers in `hosting_config_cli.py`, `hosting_setup_api.py`, and `transport_bootstrap_api.py` |
| R4 single startup input | `src/hosting/daemon/foreground.py`; `background.py`; `local_ipc.py`; delete `daemon/toolbox_launch_config.py`; `src/hosting/engine_host_channel.py`; `engine_host_cli.py`; `service/host_service.py` constructors |
| R5 generic packages | `src/hosting/toolbox/host_project_config.py`; `dependency_policy.py`; `builtin_resolver.py`; `src/hosting/service/toolbox_artifact_store.py`; `toolbox_artifact_uploads.py`; `toolbox_artifact_upload_service.py`; `toolbox_https_acquisition.py`; package portions of `toolbox_catalog.py`, `toolbox_definition_resolution.py`, `toolbox_host_config_state.py`, `auth.py`, `policy.py`, `local_ipc.py`, channel and CLI |
| R6 generic environments | `src/hosting/toolbox/environment.py`; `hermetic_environment.py`; environment identities in `bundle_models.py`; `src/hosting/service/toolbox_catalog.py`; `toolbox_env.py`; `proxy.py`; `src/hosting/sandbox/runtime_base.py`; `python_runtime.py`; `js_runtime.py`; `toolbox_runtime.py`; generic command portions of daemon/channel/CLI/auth/policy |
| R7 toolbox adoption | `src/hosting/toolbox/orchestration.py`; `staging.py`; `hosted_ref.py`; toolbox fields in `bundle_models.py`; `src/hosting/service/toolbox_definition_resolution.py`; `toolbox_plans.py`; `toolbox_confirmations.py`; `toolbox_rollout.py`; `toolbox_materialization.py`; `toolbox_runtime.py`; `toolbox_env.py`; `toolbox_state_v2.py`; `sandbox/toolbox_runtime.py` |
| R8 worker-neutral state | `src/hosting/service/operation_repository.py`; `hosted_operations.py`; generic package/environment repositories created by R5/R6; `workflow_helpers.py`; `src/hosting/sandbox/python_runtime.py`; `js_runtime.py`; `workflow_python_contract.py`; `workflow_python_node_runtime.py`; `workflow_js_node_runtime.py`; workflow IPC modules; maintenance/GC in `toolbox_env.py` and state cutover readers |
| R9 public/acceptance | `src/hosting/client_realm_api.py`; `engine_host_channel.py`; `engine_host_cli.py`; `engine_host_cli_interactive.py`; `daemon/local_ipc.py`; `service/auth.py`; `policy.py`; `control.py`; `core.py`; capability declarations, generated CLI examples/help, and every permanent document in §4.4 |

Re-exports in `src/hosting/__init__.py` and `src/hosting/toolbox/__init__.py`
belong to the slice that removes or adds the exported contract. A file touching
multiple rows is owned by the earliest dependency slice for its foundational
change and revisited by later consumers; it is not evidence that later work is
already complete.

### 4.2 Test and fixture ownership

| Owner | Primary proof locations | Required negative proof |
|---|---|---|
| R2 | `tests/test_config_paths_remote_model.py`, `test_app_config_host_auth.py`, `test_hosting_config.py`, `test_mp13chat_hosted_toolbox_api.py` | unknown label, cycle, traversal, overlap, Windows/POSIX normalization, interrupted journal phases, active-daemon/non-empty/cross-volume refusal |
| R3 | `tests/test_hosting_config.py`, `test_hosting_daemon_startup.py`, `test_hosting_service_security.py`, `test_hosting_secure_state.py` | only-old-file startup, unknown/secret fields, invalid version/type, sentinel secret and resolved-path redaction |
| R4 | `tests/test_hosting_daemon_startup.py`, `test_hosting_daemon_pidfile.py`, `test_engine_host_cli_remote_args.py`, `test_engine_host_channel.py` | old constructor/settings/flag rejection, pre-bind validation, direct/background/service/relay-equivalent parity |
| R5 | `tests/test_hosting_toolbox_artifact_uploads.py`, `test_hosting_toolbox_artifact_store.py`, `test_hosting_toolbox_https_acquisition.py`, `test_hosted_toolbox_dependency_policy.py`, `test_hosting_toolbox_host_config.py` | denied role, size/order/hash/source/credential/policy failures, disconnect/restart/concurrent commit, old command rejection |
| R6 | `tests/test_hosted_toolbox_hermetic_environment_contract.py`, `test_hosted_toolbox_hermetic_builder.py`, `test_hosted_toolbox_catalog_control.py`, `test_hosted_toolbox_template_prewarm.py`, `test_hosting_toolbox_maintenance_v2.py`, runtime-base tests | key separation, build coalescing, referenced/active removal denial, incomplete build, legacy-root non-discovery, old type/command rejection |
| R7 | `tests/test_hosting_toolbox_definition_resolution.py`, `test_hosting_toolbox_definition_transport.py`, `test_hosting_toolbox_definition_service.py`, `test_hosting_toolbox_definition_matrix.py`, `test_hosting_toolbox_atomic_routing.py`, `test_hosting_r7_acceptance.py`, `test_hosting_toolbox_sandbox.py` | mutation between plan/apply, stale approval/config revision, retry/restart, reference leak, duplicate build/execution |
| R8 | `tests/test_hosting_operation_repository.py`, `test_hosting_operation_contract.py`, `test_hosting_toolbox_state_v2.py`, `test_workflow_helper_service.py`, `test_workflow_js_node_runtime.py`, `test_hosting_python_runtime_base.py`, `test_hosting_js_runtime_base.py`, `test_hosting_workflow_operations.py` | corrupt/truncated/old state, concurrent writer/create/GC, all-consumer mark, active execution, bounded listing/quota |
| R9 | `tests/test_engine_host_channel.py`, `test_engine_host_cli_remote_args.py`, `test_engine_host_cli_interactive.py`, `test_hosting_auth_roles.py`, `test_hosting_daemon_acl.py`, `test_hosting_toolbox_removed_surface.py`, `test_hosted_toolbox_contract_docs.py`, aggregate repository lanes | fresh/cached auth equivalence, lower-role denials, secret/path/process-argument redaction, old vocabulary zero results, no-double lifecycle |

Serialized-fixture decisions: fixtures embedded in
`test_hosted_toolbox_hermetic_environment_contract.py`,
`test_hosting_python_runtime_base.py`, `test_hosting_resolved_toolbox_rollout.py`,
`test_hosting_toolbox_definition_resolution.py`,
`test_hosting_toolbox_host_config.py`, `test_hosting_toolbox_sandbox.py`, and
`test_workflow_helper_service.py` become new generic fixtures when testing the
new path; each suite keeps one minimal old-contract fixture solely to prove
`state_contract_unsupported`. No JSON resource outside those test modules was
found carrying the removed environment contracts. The home-state audit found
no implicit `Path.home()`/default hosting-root test dependency; the sole
`expanduser()` match operates on an already explicit fixture path.

Native/platform lanes: Windows is mandatory for path/permissions, service and
venv behavior; POSIX is mandatory for resolver normalization, permissions,
foreground/background startup, Python environments, and Node environments.
SSH relay-equivalent tests are required on both; real SSH/service-manager lanes
are recorded with an owner if unavailable. R1 does not claim these lanes ran.

### 4.3 Dependent repository inspection

Inspected read-only: `O:/repos/mp13-docs`, branch `redesign/cards_workflows`,
revision `a36400e8af908f702a4db84e4fdb1894ac28da36`. Its pre-existing untracked
`parent_project_feature.md` was not inspected or modified.

- Contract/version ownership: `src/backend/app/factory.py` constants and
  `src/backend/platform/hosting/daemon_contract.py::ensure_min_daemon_contract`.
- Structured auth ownership:
  `hosting_admin.py::_require_authentication_result_mapping`,
  `public_key_session_payload`, and
  `daemon_sessions.py::ensure_daemon_session_for_backend_client`. The current
  fresh and cached paths already consume mappings; adoption must prove all five
  authority fields plus `reused`, not merely a token.
- Readiness ownership: `capabilities/parent_truth.py`,
  `capabilities/runtimes.py`,
  `toolboxes/definition_coordinator.py::_PARENT_RUNTIME_FAILURE_CODES`, and
  `CapabilityToolsPanel.js::normalizeRuntime/readinessRemediation`.
- Generic identity/response ownership:
  `toolboxes/definition_coordinator.py`, `toolboxes/hosted_store.py`,
  `app/routers/capabilities.py`, `CapabilityToolsController.js`, and
  `CapabilityToolsPanel.js`.
- No removed generic command is a production literal today. The residual test
  `tests/backend_infra/test_toolbox_replacement_residuals.py` intentionally
  constructs a removed name and must remain a negative scan.

Owner: the `mp13-docs` maintainer team; no individual CODEOWNER/MAINTAINERS
entry exists. Required evidence: named maintainer, full dependent revision,
parent pin, focused auth/readiness/coordinator/UI/residual tests, and aggregate
dependent suite receipt. This repository does not edit that worktree.

### 4.4 Permanent documentation and example ownership

| Document/example | Owner | Required cutover |
|---|---|---|
| `CONFIG.md` | R2 | three top-level roots/labels, anchors, resolution, containment, local customization |
| `src/hosting/HOSTING_CONFIG_SCRIPT.md` | R2–R3 | one authority, logical/resolved status, plan/apply/recovery/reset, data/record layout |
| `src/hosting/HOSTING_ACCESS.md` | R3, R5–R6, R9 | role authority, credential handling, daemon hashing, generic lifecycle, audit/redaction, v3 operations |
| `src/hosting/HOSTED_TOOLBOX_CONTRACT.md` | R5–R7 | generic package locks/environment requests/references beneath retained toolbox semantics; remove signing baseline |
| `src/hosting/ENGINE_HOST_CLI.md` | R4, R9 | one config argument and exact generic commands/examples; remove old flags/names |
| `src/hosting/sandbox/SANDBOX_ARCHITECTURE.md` | R6–R8 | neutral environment ownership, content keys, adapters, references, retention/GC |
| `src/hosting/HOSTING.md` | R4, R8–R9 | developer orientation, startup examples, generic worker/package/environment lifecycle |
| `src/hosting/sandbox/GENERIC_WORKER.md`, `WORKFLOW_HELPER_WORKER.md`, `JS_NODE_WORKER.md`, `PY_NODE_WORKER.md`, `TOOLBOX_WORKER.md` | R7–R9 | consumer kinds, manager handoff, references/release, no toolbox-owned environment assumptions |
| `README.md` and `demo/demo_hosted_toolbox_attach.py` | R7–R9 | hosted attach terminology/examples and v3 startup expectations |
| generated help/examples in `src/hosting/engine_host_cli.py` and `engine_host_cli_interactive.py` | owning R4/R5/R6/R9 slice | exact flags, commands, payloads, codes, no credentials or absolute mainstream paths |
| `INSTALL.md`, `GOTCHAS.md`, `APPLAYERS.md` | R9 review | current matches are generic project environment/toolbox prose; update only if final terminology/search proves affected |

Removal scan patterns are the §5 list plus `hosting.toolbox.artifact_store.v2`,
`hosting.toolbox.artifact_uploads.v1`, `hosting.toolbox.environment.v2`,
`hosting.toolbox.environment_references.v1`,
`hosting.toolbox.template_catalog_state.v1`, mandatory `trust_key_ids`/signed
manifest language, and the v1/unversioned operation/result contracts. Permanent
documents must not link to this plan, status ledger, or dependent handoff.

## 5. Required removal searches

Record zero-result evidence in the owning completion slice for runtime/API
occurrences of:

```text
toolbox_host_project_configuration
toolbox_artifact_sources
toolbox_trust_public_keys
toolbox_source_credentials
toolbox_dependency_policy
--toolbox-config-file
engine_host_toolbox_config_file
access_control.json
ToolboxEnvironmentManager
RuntimeEnvironmentManager
toolbox_venvs
runtime_envs
toolbox_environment_cache
toolbox-artifact-upload-
toolbox-template-
toolbox-environment-remove
toolbox_configuration_missing
toolbox_configuration_incomplete
toolbox_configuration_invalid
toolbox_source_binding_invalid
```

Historical migration documents may retain a match only when the final R9 audit
identifies it explicitly as non-runtime history.

## 6. Active slice record

Populate this section before changing production code.

```text
Slice ID:
Plan IDs:
Priority:
Minimum expertise:
Owner:
Production boundary:
Expected removals:
Dependent handoff impact:
Positive tests:
Negative/security tests:
Native/platform lanes:
Status: Active | Blocked | Complete
```

Only one slice may be active. Consecutive plan IDs may share a slice when they
are tightly coupled and require the same expertise. A change in expertise ends
the slice.

## 7. Evidence log

### DOC-RESET — planning ledger reset

- Date: 2026-08-11
- Scope: documentation only; no production behavior changed
- Files:
  - `src/hosting/hosting_access_plan.md`
  - `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`
  - `src/hosting/hosting_status.md`
  - `tests/test_hosted_toolbox_contract_docs.py`
- Outcome: superseded completed ledgers replaced with the unified cutover plan,
  active dependent handoff, and fresh all-open status ledger; permanent contract
  tests no longer read the transient handoff
- Validation: permanent-to-transient inbound-reference audit returned zero;
  Markdown structure checks and `git diff --check` passed;
  `poetry run pytest -q tests/test_hosted_toolbox_contract_docs.py` passed 9/9
- Plan completion credited: none

### DOC-R0 — contract freeze

- Date: 2026-08-11
- Scope: R0.1–R0.7 documentation freeze only; no production behavior changed
- Files:
  - `src/hosting/hosting_access_plan.md`
  - `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`
  - `src/hosting/hosting_status.md`
- Outcome: froze `hosting.control.v3`, the one-file authority and root-label
  semantics, role/artifact rules, neutral package/environment record schemas,
  generic command/readiness cutover, clean-cut rejection behavior, and the
  journaled host-local root setup contract. Marked R0.1–R0.7 complete.
- Validation: `poetry run pytest -q tests/test_hosted_toolbox_contract_docs.py`
  passed 9/9; `git diff --check` passed; no production files changed.
- Negative-path contract proof recorded: unsupported control major,
  unsupported state contract, legacy `access_control.json`, legacy environment
  roots, traversal roots, and remote root relocation are all fail-closed.
- Dependent handoff impact: exact client-visible contract is frozen; R1.1 has
  not started and no dependent owner or adoption receipt is claimed.

### DOC-R1 — handoff and inventory

- Date: 2026-08-11
- Scope: R1.1–R1.5 documentation/inventory only; no production or dependent
  files changed
- Parent searches: all named legacy fields, flags, classes, paths, command
  strings, readiness codes, re-exports, constructors, state contracts, tests,
  fixtures, permanent docs, examples, and generated CLI help sources
- Dependent inspection: `O:/repos/mp13-docs` at
  `a36400e8af908f702a4db84e4fdb1894ac28da36`, read-only; its existing untracked
  `parent_project_feature.md` remained untouched
- Outcome: exact stable dependent symbols and proof requirements added to the
  handoff; production ownership assigned to R2–R9; test/fixture/native lanes
  assigned; documentation and final removal patterns assigned
- Validation: all Markdown JSON examples parsed; R0/R1 plan sections contain no
  unchecked item; permanent-to-transient backlink scan returned zero;
  `poetry run pytest -q tests/test_hosted_toolbox_contract_docs.py` passed 9/9;
  `git diff --check` passed
- Dependent handoff impact: implementable handoff is ready, but delivery,
  named owner acknowledgment, revision, tests, and receipt remain R9.6 work

### CODE-R2 — shared roots and host-local setup

- Date: 2026-08-11
- Plan IDs: R2.1–R2.3; P0; high expertise
- Production boundary: shared `category_dirs`/`PathResolver`, both config UIs,
  and host-local setup plan/apply/inspect/status/reset
- Outcome: added `@hosting`, `@packages`, and `@environments`; strict
  label/cycle/traversal/type/overlap validation; logical round trips; local
  preflight; optimistic revisions; locked restrictive atomic writes; and
  four-phase idempotent recovery without remote relocation
- Positive/negative proof: 90 focused path, UI, setup, legacy setup, and doc
  tests passed, including all journal phases, unsafe roots, non-empty targets,
  remote denial, and stale revisions; `git diff --check` passed
- Native lane: Windows path and file-replace behavior passed locally; POSIX and
  real service/SSH lanes remain required by R9.7
- Dependent handoff impact: root fields and `hosting.setup.v1` are now available;
  dependent adoption remains external R9.6 work

### CODE-R3.1 — strict unified configuration repository

- Date: 2026-08-11
- Plan IDs: R3.1; P0; high expertise
- Production boundary: immutable `hosting.configuration.v3` model, sanitized
  inspection model, and the sole locked/atomic `hosting_config.json` repository
- Outcome: strict control/package/environment sections preserve logical paths,
  resolve at the host boundary, compute a canonical revision, and reject
  unsupported contracts, unknown security fields, wrong types, invalid labels,
  policy conflicts, missing credential refs, and non-SHA-256 baseline policy
- Positive/negative proof: 17 focused configuration/repository/setup recovery
  tests passed; sentinel credentials, query tokens, and resolved paths were
  absent from remote inspection; invalid writes left the authority unchanged
- Dependent handoff impact: the v3 static schema is executable; daemon startup
  and dependent adoption remain R3.2–R4 and R9.6 work

### CODE-R4A — single-path production startup and readiness

- Date: 2026-08-11
- Plan IDs: R3.2–R3.3 and R4.1–R4.3 (partial); P0; high expertise
- Outcome: all production startup modes load the top-level MP13 configuration
  before listener construction and inject one immutable configuration; static
  authority and mutable key/session/audit/runtime records are separate; the
  launcher JSON module and five startup mappings are absent; generic readiness
  keeps control diagnosis distinct from package/environment degradation
- Proof: 21 focused configuration/startup tests passed at the production cut,
  production searches found no old startup flags or authority filename, and
  sentinel secrets/resolved paths were absent from remote inspection
- Remaining: migrate or remove the inventoried legacy fixtures and permanent
  documentation before closing the clean-cut checkboxes

### CODE-R5 — generic package management

- Date: 2026-08-11
- Plan IDs: R5.1–R5.4; P0; high expertise
- Production boundary: neutral package source/policy/lock contracts,
  daemon-owned artifact ingress, deterministic resolution, public package
  commands, role policy, capability advertisement, and sanitized audit records
- Outcome: bounded resumable uploads compute their identity from staged bytes
  and promote atomically into content-addressed storage; optional verification
  is outside the baseline; credential values remain host-local; deterministic
  locks contain only reproducible, non-secret source metadata
- Positive/negative proof: 20 focused package/configuration/startup tests passed,
  covering authorization, bounds/order/idempotency, cancellation/disconnect,
  mismatch isolation, restart, concurrent commit, source/policy/credential
  failures, offline reuse, deterministic output, and legacy command rejection;
  the production legacy command search returned zero results
- Dependent handoff impact: generic package commands and locks are available;
  toolbox adoption and external dependent receipt remain R7 and R9.6 work

### CODE-R6A — generic environment foundation

- Date: 2026-08-11
- Plan IDs: R6.1 (partial), R6.2, R6.3, R6.4 (public names); P0; high
  expertise
- Production boundary: strict neutral template/request/lock/receipt/reference
  records, content-addressed manager, runtime adapter boundary, service facade,
  public command names, role policy, and capability advertisement
- Positive/negative proof: 13 focused package/environment tests passed; same-key
  concurrent requests coalesced to one build; toolbox and workflow consumers
  acquired separate references to identical content; active/retained content
  resisted removal; legacy-only roots were ignored
- Follow-up proof: hosted operation enum values, namespaces, validation codes,
  and worker names now use generic environment kinds; 17 focused package,
  environment, and daemon-startup tests passed after the cut
- Aggregate migration check collected successfully after updating the operation
  contract fixture, then reached 336 passes and the 100-failure stop; failures
  remain concentrated in legacy configuration/startup/toolbox fixtures
- Remaining in R6: remove the toolbox manager/builder aliases and legacy readers,
  then generalize the remaining toolbox-owned audit/receipt implementation

### CODE-R8A — versioned and bounded neutral state

- Date: 2026-08-11
- Plan IDs: R8.1 and R8.3 (partial); P1; high expertise
- Outcome: package/environment repositories use exact versioned contracts and
  fail closed on prior or foreign state; writes are locked and atomic; template
  and reference counts are bounded; reference listing is cursor-paginated;
  environment removal and GC re-check builds, executions, references, and
  retention under lock
- Proof: 14 focused package/environment tests passed, including unsupported
  state version, pagination, concurrent build/commit, active execution,
  reference, and retention denial paths

### DOC-R6/R9A — permanent environment and toolbox guidance cut

- Date: 2026-08-11
- Plan IDs: R6.5 cleanup guidance and R9.5 (partial); P2; high expertise
- Outcome: sandbox guidance now describes only `@environments` reusable content,
  `@hosting/scratch` staging, and generic consumer references; it explicitly
  requires stopped-host operator archival/rebuild for legacy directories;
  toolbox documentation uses generic public commands, package locks,
  daemon-computed hashes, optional verification, and one MP13 configuration
- Remaining: finish the implementation alias cut, then run the final permanent
  document search and executable example tests

### CODE-R6B — legacy environment roots disabled

- Date: 2026-08-11
- Plan IDs: R6.5 (root discovery); P0; high expertise
- Outcome: toolbox, workflow Python, workflow JavaScript, orchestration, bundle
  metadata, and hermetic resolution use the shared environments root; production
  Python contains no legacy environment directory name and never discovers old
  content
- Proof: root-name production search returned zero; 26 focused hermetic and
  neutral environment tests ran with 24 passing before two deliberately stale
  fixtures, which were migrated to shared-root and legacy-constructor rejection

### CODE-R6C — compatibility environment type names removed

- Date: 2026-08-11
- Plan IDs: R6.1 alias cut and R6.5 compatibility readers; P0; high expertise
- Outcome: the toolbox and workflow mechanics are explicit runtime adapters and
  the Python materializer is a runtime-specific builder; the retired manager
  and hermetic-builder class names have no production or test imports
- Proof: prohibited type-name search returned zero; 26 hermetic/neutral tests
  and three focused toolbox runtime-adapter tests passed

### CODE-R6D — exact generic environment public payloads

- Date: 2026-08-11
- Plan IDs: R6.4; P0; high expertise
- Outcome: daemon dispatch, channel methods, and CLI routing accept exact generic
  template records, integer revisions, environment requests, and environment
  IDs; no toolbox-era template/environment digest field remains on those public
  surfaces
- Proof: production public-surface search returned zero for the retired fields;
  18 focused environment/package/startup tests passed

### TEST-R9A — internal security/removal checkpoint

- Date: 2026-08-11
- Plan IDs: R9.2/R9.4 (partial); P2; high expertise
- Proof: 28 focused configuration, startup, package, environment, role, denial,
  redaction, hashing, retry, and concurrency tests passed; diagnostic/worker
  roles cannot mutate generic templates, environments, references, or GC
- Production removal search: zero Python matches for the old authority filename,
  upload/template/environment commands, three legacy environment roots, and
  three retired environment class names
- Remaining: migrate/delete inventoried legacy tests and permanent-doc matches,
  exercise password/public-key policy parity, and run process/log/path leakage
  acceptance before closing R9.2/R9.4

### TEST-R9B — aggregate and dependent gate audit

- Date: 2026-08-11
- Aggregate: `python -m pytest -q --tb=no --maxfail=200` collected cleanly,
  reached 509 passes, then stopped at 200 failures; the failure list is dominated
  by pre-v3 service/daemon constructors, mutable static-authority expectations,
  and retired toolbox catalog/template fixtures. No compatibility shim was added.
- Dependent read-only audit: `O:/repos/mp13-docs` remains at
  `a36400e8af908f702a4db84e4fdb1894ac28da36` with the pre-existing untracked
  `parent_project_feature.md`; searches found no new generic config/package/
  environment/readiness adoption and did find the retired command family in its
  refactoring plan
- Gate impact: R9.6 cannot pass without a dependent-team revision, tests, named
  owner, and receipt. The dependent worktree was not modified.

### CODE-R7A — toolbox materializer configuration bridge

- Date: 2026-08-11
- Plan IDs: R7.2 (partial); P1; high expertise
- Outcome: service construction now configures the retained Python environment
  builder exclusively from generic package sources and the resolved shared
  environment root; toolbox materialization no longer depends on launcher maps
  or legacy directory conventions
- Proof: 33 package/environment/startup/hermetic tests executed with 32 passing;
  the sole failure was a test-only string-to-Path assertion and was corrected
- Remaining: make plan/apply consume generic package-lock and environment
  request/receipt/reference records transactionally before closing R7.2

### CODE-R8B — workflow runtimes adopt shared environment manager

- Date: 2026-08-11
- Plan IDs: R8.2; P1; high expertise
- Outcome: Python-helper and JavaScript/Node runtime bases receive the service's
  `EnvironmentManager`, enforce stable consumer kinds, acquire exact package-
  lock-backed environments, and release generic references
- Proof: 17 focused package/environment tests passed; matching Python-helper and
  JS-node inputs reused one physical build while retaining independent references,
  and removal remained denied until both consumers released

### CODE-R8C — repair mutation authorization

- Date: 2026-08-11
- Plan IDs: R8.3; P1; high expertise
- Outcome: `toolbox-repair` is an observational review by default; mutation
  requires both explicit `apply: true` and a server-supplied authorization bit;
  payloads cannot assert mutation authority
- Proof: 10 focused environment/maintenance tests passed, covering observation,
  denied unauthorized mutation, and authorized dispatch

### CODE-R9.1A — structured authentication cache results

- Date: 2026-08-11
- Plan IDs: R9.1 authentication result preservation; P2; high expertise
- Outcome: shared-secret and public-key cache reads return complete structured
  authentication metadata; adoption preserves token, role, auth method, scope,
  key ID, and expiry, and the channel exposes the adopted structured result
- Proof: three focused fresh/empty/cached metadata tests passed; token-only cache
  return annotations are absent

### TEST-R9.2A — role, authentication-method, and hash authority

- Date: 2026-08-11
- Plan IDs: R9.2 (first three items); P2; high expertise
- Proof: 19 focused tests passed; diagnostic authority is disjoint from package
  and environment mutation surfaces, equal admin roles backed by shared-secret
  and public-key records produce identical command policy, and mismatched caller
  expectations cannot publish or resolve bytes under either digest
- Remaining: complete the process-argument/log/audit/receipt/error/path sentinel
  sweep before closing R9.2

### TEST-R9.2B — secret and path leakage acceptance

- Date: 2026-08-11
- Plan IDs: R9.2 redaction; P2; high expertise
- Proof: 23 focused startup/configuration/package tests passed with sentinel
  credential/query/path values; detached process arguments and environment
  contained only the top-level configuration location, while configuration
  bodies, credentials, and package policy were absent; remote inspection,
  errors, receipts, and audit omitted secret values and resolved host paths

### TEST-R9.3A — unified lifecycle startup surface

- Date: 2026-08-11
- Plan IDs: R9.3 startup modes; P2; high expertise
- Proof: 25 focused direct/background/HTTP-service/relay-equivalent startup,
  package, and environment tests passed; each startup callable exposes exactly
  the top-level MP13 configuration input and no toolbox mapping or mutable
  control-state input

### CODE-R9.3B — public generic environment lifecycle

- Date: 2026-08-11
- Plan IDs: R9.1/R9.3 (partial); P2; high expertise
- Outcome: reference list/release, execution begin/end, exact removal, and GC
  are available through authenticated policy, daemon dispatch, channel, CLI,
  and capability negotiation using only generic payload identities
- Proof: 11 focused lifecycle/role/channel tests passed with exact payload
  comparisons and lower-role denial coverage

### CODE-R7B — generic toolbox package and reference authority

- Date: 2026-08-11
- Plan IDs: R7.1 (identity and reference transaction); P1; high expertise
- Outcome: confirmed toolbox bytes are rehashed into the generic CAS, expressed
  as a deterministic `hosting.package_lock.v1`, and attached to a generic
  environment receipt/reference before candidate registration; failed rollout
  releases the generic reference and successful replacement releases displaced
  references only after publication
- Proof: 22 package/environment tests passed, including daemon-local rehash,
  mismatch denial, deterministic lock creation, adopted receipt idempotence,
  independent references, and removal guards
- Remaining: remove the retained toolbox resolver/artifact repository and make
  planning originate the generic lock/request directly before closing R7.1/R7.2

### TEST-R7B — rollout transaction regression proof

- Date: 2026-08-11
- Scope: resolved candidate registration plus definition apply recovery,
  cancellation, publication checkpoint, and cleanup
- Proof: seven rollout/apply tests passed after migration to the canonical v3
  hosting fixture; candidate routing remains non-public, duplicate request
  identity is stable, cancellation cleans generic references before publication,
  and the publication checkpoint remains atomic

### TEST-R9.3C — operation retry/restart identity

- Date: 2026-08-11
- Plan IDs: R9.3 no-double/restart (partial); P2; high expertise
- Proof: eight operation-service tests passed after v3 fixture migration,
  covering duplicate replay without redispatch, actor isolation, response-loss
  recovery, queued cancellation, Python/JS recreation, and real daemon restart
  with the original terminal operation and external result identity

### CODE/TEST-R3.2D — immutable lifecycle and claim policy fixture cut

- Date: 2026-08-12
- Plan IDs: R3.1/R3.2 fixture migration and R9.3 lifecycle acceptance;
  P0/P2; high expertise
- Outcome: the strict v3 `control` schema now owns lifecycle and claim-retention
  policy; daemon ACL tests configure static policy before construction and use
  resolved hosting state paths instead of mutating policy after startup or
  assuming the retired root layout
- Proof: `python -m pytest tests/test_hosting_configuration_v3.py
  tests/test_hosting_daemon_acl.py -q --maxfail=20` passed 59 tests, including
  invalid lifecycle/type rejection, authenticated ACL denials, restart-activated
  SSH connectivity policy, operation persistence, and lifecycle shutdown gates
- Dependent impact: consumers that author full v3 control policy may add the
  optional `lifecycle` and `claims` objects; static changes still require an
  explicit daemon restart

### TEST-R4.2C — CLI startup fixture clean cut

- Date: 2026-08-12
- Plan IDs: R4.1/R4.2 and R9.4 (partial); P0/P2; high expertise
- Outcome: daemon and relay CLI tests now pass only the top-level MP13
  configuration, local workflow facade tests load the canonical v3 authority,
  and the removed control-state/toolbox-launcher flags fail with
  `hosting_startup_option_removed` instead of being silently ignored
- Proof: `python -m pytest tests/test_engine_host_cli_remote_args.py -q
  --maxfail=20` passed 13 tests, including exact production/background
  forwarding, absence of launcher kwargs, removed-flag rejection, and Python/JS
  local facade payload preservation
- Remaining: R4.2 stays open until removal searches clear the other fixtures,
  help text, and documentation matches

### TEST-R3.2E — shared configuration auth fixture cut

- Date: 2026-08-12
- Plan IDs: R3.2 fixture removal and R9.4 (partial); P0/P2; high expertise
- Outcome: the shared MP13 configuration CLI auth tests now load the v3 static
  authority through `--host-mp13-config-file`; the remote shared-secret denial
  activates changed connectivity policy by rewriting v3 configuration before a
  new service invocation, without editing runtime control state
- Proof: `python -m pytest tests/test_app_config_host_auth.py -q --maxfail=20`
  passed 10 tests, including key/status persistence and remote shared-secret
  denial guidance

### TEST-R4.1D — channel bootstrap and recovery fixture cut

- Date: 2026-08-12
- Plan IDs: R4.1/R4.2 and R9.3/R9.4 (partial); P0/P2; high expertise
- Outcome: channel bootstrap forwards only the MP13 configuration path; its
  unconfigured-host probe is observational and preserves immutable v3 auth
  policy, while reset and force-stop helpers construct services from the same
  v3 authority
- Proof: `python -m pytest tests/test_engine_host_channel.py -q --maxfail=20`
  passed 43 tests, including exact one-path background startup, no toolbox
  launcher kwargs, v3 shared/no-auth snapshots, local-only reset, registered
  worker shutdown, and orphan worker termination
- Remaining: broader channel command removal and public-surface alignment remain
  open under R9.1/R9.4

### TEST-R3.2F — model runtime boundary fixture cut

- Date: 2026-08-12
- Plan IDs: R3.2 fixture removal and R9.4 (partial); P0/P2; high expertise
- Outcome: read-only model-runtime boundary services and daemon projections now
  receive the immutable v3 authority; no model-runtime test constructs a legacy
  control-state authority or filename
- Proof: `python -m pytest tests/test_hosted_model_runtime_boundary.py -q
  --maxfail=20` passed 12 tests, covering bounded status, unconfigured/degraded
  states, selector-smuggling denials, role/channel projection, daemon dispatch,
  and SSH CLI routing

### TEST-R6/R9C — generic template catalog public-surface cut

- Date: 2026-08-12
- Plan IDs: R6.1 and R9.1/R9.4 (partial); P1/P2; high expertise
- Outcome: catalog channel, role, daemon, operation, and remote CLI acceptance
  now uses only `environment-template-*` methods and exact v1 template payloads;
  authenticated fixtures receive v3 policy without post-start mutation
- Proof: `python -m pytest tests/test_hosted_toolbox_catalog_control.py -q
  --maxfail=20` passed 13 tests, including exact channel payloads, lower-role
  mutation denial, daemon/CLI routing, generic operation dispatch, and retained
  internal catalog atomicity/redaction tests
- Remaining: the retained toolbox-specific catalog repository/materializer is
  still an internal R7 removal item and is not treated as completed by this
  public-surface migration

### CODE/TEST-R3.1G — immutable auth and role acceptance cut

- Date: 2026-08-12
- Plan IDs: R3.1/R3.2 and R9.2/R9.3/R9.4 (partial); P0/P2; high expertise
- Outcome: the v3 parser rejects unauthenticated shared endpoints; auth/role
  acceptance now supplies authentication, connectivity, endpoint, and lifecycle
  policy before service construction, activates policy changes through a new
  service instance, and proves runtime state cannot override static authority
- Removal proof: `tests/test_hosting_auth_roles.py` contains zero
  `set_control_config`, `control_state_file`, or `access_control.json` matches
- Proof: `python -m pytest tests/test_hosting_configuration_v3.py
  tests/test_hosting_auth_roles.py tests/test_hosting_daemon_acl.py -q
  --maxfail=30` passed 105 tests, including role denials, local/remote bootstrap
  policy, SSH binding, pre-restart session rejection, lifecycle defaults,
  static-over-runtime precedence, and daemon ACL behavior
- Remaining: production removal of the legacy control-config command/method and
  migration of other fixtures/docs remain open under R9.1/R9.4

### CODE/TEST-R3.1H — static traffic policy and security fixture cut

- Date: 2026-08-12
- Plan IDs: R3.1/R3.2 and R9.2/R9.4 (partial); P0/P2; high expertise
- Outcome: strict v3 control configuration now owns default and per-engine
  traffic policy; policy types and unknown fields are rejected before startup,
  and service security acceptance no longer edits mutable control state
- Removal proof: `tests/test_hosting_service_security.py` contains zero
  `set_control_config`, `control_state_file`, or `access_control.json` matches
- Proof: `python -m pytest tests/test_hosting_configuration_v3.py
  tests/test_hosting_service_security.py -q --maxfail=30` passed 53 tests,
  covering auth/session/challenge redaction and denials, SSH binding, config and
  engine allowlists, discovery/process validation, default/per-engine traffic
  policy, proxy bounds, metrics, and strict traffic-policy rejection
- Aggregate checkpoint before this slice: 693 passed and 120 failed at the
  bounded `--maxfail=120` stop; remaining failures were concentrated in legacy
  toolbox repositories/startup fixtures plus service/list-config constructors

### TEST-R3.2I — hosted config service fixture cut

- Date: 2026-08-12
- Plan IDs: R3.2 fixture removal and R9.4/R9.7 (partial); P0/P2; high expertise
- Outcome: hosted config discovery, spawn-spec, model-worker reuse/readiness, and
  generic-worker tests construct the service exclusively from v3 configuration
- Proof: `python -m pytest tests/test_hosting_service_list_configs.py -q
  --maxfail=30` passed 10 tests; the file contains zero legacy control-state
  constructor or authority filename matches

### TEST-R3.2J — secure-state daemon fixture cut

- Date: 2026-08-12
- Plan IDs: R3.2 fixture removal and R9.2/R9.4 (partial); P0/P2; high expertise
- Outcome: secure-state daemon projection loads v3 configuration while keeping
  keyring, sessions, challenges, audits, and bootstrap records under the
  resolved hosting root; remote setup status omits the resolved hosting path and
  the retired monolithic authority record
- Proof: `python -m pytest tests/test_hosting_secure_state.py -q --maxfail=20`
  passed three tests, including local/remote path projection and restrictive
  state-file handling

### TEST-R4.1E — daemon startup and pidfile fixture clean cut

- Date: 2026-08-12
- Plan IDs: R4.1/R4.2 and R9.2/R9.3/R9.4 (partial); P0/P2; high expertise
- Outcome: foreground/background startup, PID lifecycle, shutdown checkpoints,
  terminal-disconnect policy, and startup recovery tests now pass one MP13
  configuration path; former launcher tests instead prove exact v3 package
  source/policy wiring, secret-free argv, and generic readiness degradation
- Removal proof: both startup files contain zero `control_state_file`,
  `access_control.json`, removed launcher-file, trust-key, source-credential, or
  five toolbox startup-mapping matches
- Proof: `python -m pytest tests/test_hosting_daemon_pidfile.py
  tests/test_hosting_daemon_startup.py tests/test_hosting_daemon_startup_v3.py
  -q --maxfail=30` passed 43 tests, including listener-before-PID ordering,
  duplicate-daemon rejection, background ping readiness, crash progress,
  lifecycle signal behavior, exact foreground handoff, strict signature
  rejection, and bounded package/environment readiness
- Remaining: R4.2 remains open until repository-wide fixture/doc removal searches
  clear; the generic builder still bridges retained toolbox internals under R7

### TEST-R4.1F — HTTP ingress fixture clean cut

- Date: 2026-08-12
- Plan IDs: R4.1/R4.2 and R9.2/R9.4 (partial); P0/P2; high expertise
- Outcome: authenticated HTTP proxying, engine-specific traffic overrides, and
  cross-transport daemon-version checks now use the same immutable v3 hosting
  configuration as local IPC startup
- Removal proof: the HTTP ingress suite contains zero `control_state_file`,
  `set_control_config`, or `access_control.json` matches
- Proof: `python -m pytest tests/test_hosting_http_ingress.py -q --maxfail=20`
  passed all three tests, including 401/403 enforcement and static per-engine
  path-policy behavior

### TEST-R6.4C — shared-template prewarm surface cut

- Date: 2026-08-12
- Plan IDs: R4.1/R4.2, R6.4, and R9.1/R9.4 (partial); P0/P2; high expertise
- Outcome: shared-template resolver fixtures load v3 hosting configuration;
  public role, channel, daemon, and SSH-CLI prewarm coverage uses only
  `environment-template-prewarm` with a worker-neutral `EnvironmentRequest`
- Removal proof: the touched resolver/prewarm suites and catalog contain zero
  `control_state_file`, `access_control.json`, `toolbox-template-prewarm`, or
  `TOOLBOX_TEMPLATE_PREWARM` matches
- Proof: `python -m pytest tests/test_hosted_toolbox_shared_template_resolver.py
  tests/test_hosted_toolbox_template_prewarm.py
  tests/test_hosted_toolbox_shipped_templates.py -q --maxfail=30` passed all 21
  tests, including generic operation-kind persistence and fail-closed materialization

Append one concise entry per completed slice. Include exact commands, counts,
durations where useful, negative-path results, commit pin, and dependent receipt
impact. Do not paste an unstructured full test transcript.

## 8. Blockers and external evidence

Current blockers: none recorded

Execution pause: none; R3.2 is active.

External evidence required before final completion:

- dependent revision and adoption receipt for the new major contract;
- required native/platform lane results identified by R1.3/R9.7; and
- proof that remote output and process arguments contain no sentinel secrets or
  unrestricted host-local paths.

An unavailable external lane is recorded as not run with an owner and reason;
it is not reported as a pass.

## 9. Status update rules

1. Record an active slice before changing production code.
2. Keep one coherent slice per commit and use the highest expertise touched.
3. Update production code, tests, docs, handoff, status, and plan checkboxes in
   the same slice when they form one contract.
4. Mark a plan item complete only after its named proof passes.
5. Keep failures and partial work visible with checkboxes open.
6. Record removal `rg` results for every clean-cut boundary.
7. Include denial and redaction tests for authorization/security boundaries.
8. Never edit the dependent repository; record its team-supplied evidence.
9. Do not infer adoption from a parent compatibility shim.
10. Do not restore old fallbacks, aliases, dual commands, or migration readers.
11. Do not link permanent docs to transient plan/status/handoff records or test
    those records as normative contracts.
