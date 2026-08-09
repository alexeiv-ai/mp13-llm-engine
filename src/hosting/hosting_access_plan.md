# Hosted Toolbox Reconfiguration and Environment Templates

Status: proposed direct breaking replacement

Owner: parent hosting team

Primary consumer: `O:/repos/mp13-docs`

Plan date: 2026-08-08

## Goal

Replace the current hosted-toolbox mutation and environment workflow with one
code-derived definition/apply flow:

1. The consumer submits the complete desired toolbox definition.
2. The host analyzes the submitted bundle imports.
3. The host maps imports to an immutable built-in environment template or to an
   approved custom locked environment.
4. The host prepares and verifies the environment before importing tool code.
5. The host stages and warms only changed sandbox profiles.
6. The host atomically publishes the new profile/tool routing revision.
7. The host drains replaced workers and later garbage-collects unreferenced
   bundles and environments.
8. Apply runs as an actor-owned durable hosted operation with progress,
   reconnect recovery, idempotent retry, terminal diagnostics, and explicitly
   bounded cancellation semantics.

Adding, changing, and removing functions use the same operation. Legacy APIs,
state readers, environment-description behavior, and fallback execution paths
are removed rather than adapted. Every consumer requirement must be entered in
`src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md` before the corresponding code
change merges.

## Documentation policy

- [ ] `src/hosting/HOSTED_TOOLBOX_CONTRACT.md` is durable normative
  documentation. It describes only the supported contract as implemented. It
  must not mention removed methods, legacy fields, old state schemas, migration
  aliases, historical behavior, cutover instructions, parent pin transitions,
  or compatibility comparisons.
- [ ] `src/hosting/sandbox/TOOLBOX_WORKER.md` is durable implementation
  documentation. It describes only the current worker architecture and links to
  the normative contract. It must not retain a legacy-behavior or migration
  section.
- [ ] `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md` is the sole transient
  migration handoff. All removed APIs/fields, old-to-new examples, unsupported
  state formats, archival/cutover steps, release commits, dependent-project
  requirements, and historical references belong there.
- [ ] Remove `HOSTING_CLIENT_BREAKING_CHANGES.md` after every listed dependent
  project confirms adoption and the handoff is no longer needed. Durable
  documentation must remain complete without it.

## Execution, sliced-commit, and completion policy

- [ ] Implement the plan as small dependency-closed slices. Each commit must
  name the plan item IDs it advances and contain one coherent contract, model,
  implementation, test, documentation, or removal slice rather than an entire
  phase-sized change.
- [ ] Every slice must leave the parent repository in an internally consistent
  state. Do not commit calls to APIs that are not present, state writers without
  strict readers, routable candidates without active-route selection, or new
  contract fields without validation and focused tests.
- [ ] A breaking replacement and removal of the superseded behavior should land
  in the same dependency-closed slice when possible. If sequencing requires
  intermediate commits on the implementation branch, those commits must not be
  released or presented as supported states and must not add compatibility
  adapters.
- [ ] Update `HOSTING_CLIENT_BREAKING_CHANGES.md` in the same commit as, or
  before, the first client-visible break in a slice. Keep durable documentation
  current in the slice that establishes the replacement behavior.
- [ ] Do not mark a plan item `[x]` merely because code exists. Before marking
  it complete, inspect the implemented code against the full item, run the
  focused tests required by that item, confirm removal requirements by search,
  and verify any required durable/transient documentation changes.
- [ ] Record completion evidence next to each completed item or in a phase
  evidence block: commit hash, focused test commands/results, relevant contract
  or state-schema artifact, and dependent-project impact. An item without
  reviewable evidence remains incomplete.
- [ ] At the start of every resumed implementation session and before each phase
  exit, audit all previously checked items affected by subsequent changes.
  Compare the current code and tests with their recorded evidence; do not trust
  checkmarks or old summaries alone.
- [ ] Reopen a checked item immediately if its implementation was removed,
  weakened, contradicted by a later slice, lacks the promised tests, or no
  longer satisfies the current contract. Record why it was reopened.
- [ ] A phase exit gate may be marked complete only after every item in that
  phase and every prerequisite defect fix has current evidence, focused tests
  pass, and no known blocking TODO is deferred to a later phase without being
  explicitly represented there.
- [ ] Parent and dependent-project changes use separate sliced commits. Record
  the finalized parent commit in the transient breaking-change handoff before
  the dependent project repins; then record the dependent adoption commit and
  verification evidence before deleting the handoff file.

### Completed-slice test, checkbox, status, and commit protocol

Every implementation slice must close one coherent server-side outcome and map
to explicit unchecked item IDs in this plan. If an item cannot be completed in
one reviewable slice, split it into independently checkable plan items before
implementation; do not check a parent item for partial work.

Before implementation, write the slice and its exact required test commands in
`hosting_status.md`. Select tests from every affected category:

1. Focused unit and contract tests for each changed type, planner rule, state
   transition, policy decision, or diagnostic.
2. Clean-environment build, import, probe, and worker-launch tests for template,
   package, intrinsic, sandbox, or materialization changes.
3. Daemon/control-channel and client integration tests for public API,
   operation, authorization, projection, or reconnect behavior.
4. Persistence, restart, migration, rollback, repair, and concurrency tests for
   state or rollout changes.
5. Existing regression tests covering changed server and dependent-facing
   behavior. Record a justified `not applicable` for a category that the slice
   cannot affect; absence of a test command is not evidence of completion.

Close and commit a slice only in this order:

1. Complete the implementation and remove superseded behavior required by the
   slice.
2. Run all predeclared focused and regression commands and record their results.
3. Update durable contracts and worker documentation to describe only the new
   supported behavior.
4. Update `HOSTING_CLIENT_BREAKING_CHANGES.md` with any dependent-project
   adoption delta introduced by the slice.
5. Check only the plan boxes whose full acceptance criteria are satisfied.
6. Move the slice from Active to Completed in `hosting_status.md`, listing the
   checked item IDs, delivered outcome, exact passing test commands, and planned
   commit subject.
7. Review the staged slice for unrelated or incomplete changes, then create one
   non-amended commit whose subject or body includes the completed plan item
   IDs. Do not mix unchecked follow-up work into that commit.

Git history is the source of commit hashes; the status ledger records the commit
subject because a commit cannot contain its own final hash. A failed test,
unresolved diagnostic, missing required documentation, or unchecked acceptance
criterion keeps the slice Active or Blocked and must not produce a completion
commit.

## Implementation baseline from the current code

The plan is based on these current implementation facts.

- [ ] **B-01 Models:** `src/hosting/toolbox/bundle_models.py` represents auto
  and manual functions as `ToolboxAutoAssignmentRequest` and
  `ToolboxManualAssignmentRequest`. Their stable keys are respectively
  `module_name:callable_name` and `manual:module_name:callable_name`.
- [ ] **B-02 Profiles:** `SandboxProfileSpec` currently hashes
  `environment_name`, `required_imports`, and `sandbox_policy` to derive a
  profile ID. An explicit `profile_id` bypasses that derivation.
- [ ] **B-03 Bundles:** `ToolboxBundleSpec.manifest_payload()` already computes
  a deterministic full manifest hash and 16-character bundle revision from
  files, tool metadata, intrinsics, profile, and dependency-lock hash.
- [ ] **B-04 Staging:** `ToolboxBundleStager` writes immutable revision-named
  bundle directories and verifies conflicting existing content. This remains
  the bundle materialization boundary.
- [ ] **B-05 Loading:** `load_toolbox_from_manifest()` imports every staged auto
  and manual module while constructing the worker toolbox. A missing top-level
  import therefore fails worker startup/warmup.
- [ ] **B-06 Grouping:** `ToolboxSandboxOrchestrator.build_assignments()` groups
  requests by normalized profile ID. It does not reject two requests that use
  the same explicit profile ID with contradictory profile contents.
- [ ] **B-07 Rollout:** every register/unregister implementation reconstructs
  all assignments, calls `spawn_assignments()`, waits for readiness and exact
  tool inventory, retires replaced registrations, then writes logical state.
- [ ] **B-08 Routing:** `_route_toolbox_registration()` scans all live toolbox
  executor registrations and requires exactly one registration containing the
  requested tool. Candidate and active registrations are not distinguished.
  Overlapping old/new workers can therefore make routing ambiguous during a
  rollout.
- [ ] **B-09 Mutation state:** version-1 `toolbox_sandboxes.json` stores
  `requests`, `manual_requests`, `intrinsics`, derived `profiles`, `runtime`, and
  global mutable `environment_descriptions`.
- [ ] **B-10 State safety:** `_read_toolboxes()` delegates to `_read_json()`,
  which returns default empty state on invalid JSON. `_write_toolboxes()` writes
  the file directly. The per-toolbox `RLock` is process-local.
- [ ] **B-11 Environment identity:** `environment_spec_for_bundle()` hashes the
  environment name/description, intrinsic dependency profile, raw required
  imports, runtime hash, and optional dependency-lock hash into `venv_key`.
- [ ] **B-12 Environment isolation:** `ensure_environment()` creates venvs with
  `system_site_packages=True` and no dedicated pip bootstrap. Host packages can
  satisfy undeclared imports.
- [ ] **B-13 Runtime selection:** `runtime_python_executable()` runs a
  dependency-bearing toolbox worker with the bootstrap interpreter until both
  install execution and receipt verification are `ok`.
- [ ] **B-14 Dependency workflow:** toolbox consumers currently manage mutable
  descriptions and separate resolve, apply, realize, prepare, lock, resolve
  lock, verify, execute, and receipt-verification calls.
- [ ] **B-15 Package naming:** `required_imports` are copied directly into pip
  requirement planning. There is no import-root to distribution mapping.
- [ ] **B-16 Intrinsics:** `SandboxProfileSpec.intrinsics_profile_id()` contains
  hard-coded calculator/symbolic dependency categories rather than dependency
  metadata owned by each intrinsic.
- [ ] **B-17 Cleanup:** mutation and failed-rollout paths can immediately delete
  unreferenced environment directories; repair/reconcile/GC derive references
  from version-1 logical profiles and live registrations.
- [ ] **B-18 Public surface:** the old commands are duplicated across
  `HostedToolBoxRef`, `EngineHostControlChannel`, daemon dispatch, subprocess
  CLI dispatch, authorization/policy lists, service methods, and tests.

## Confirmed defects that the replacement must fix

These are required correctness fixes, not optional benefits of the redesign.
Each must have a focused regression test in addition to the broader phase exit
tests.

- [ ] **F-01 Conflicting explicit profile IDs:** remove consumer-supplied
  `profile_id` and derive resolved profile identity from canonical sandbox
  policy plus resolved environment identity. Until the old model is removed,
  reject duplicate explicit IDs with differing profile contents. This prevents
  tools from being grouped under the first request's environment or sandbox
  policy. Address in P3-01 through P3-03 and verify in P7-01.
- [ ] **F-02 Intrinsic mutation deletes manual tools:** the replacement
  definition must always contain and validate auto requests, manual requests,
  and intrinsics together. Applying an intrinsic-only difference must preserve
  every manual tool, its bundle membership, dependency intent, and route.
  Delete the current intrinsic register/unregister implementations rather than
  repairing their partial merge behavior. Address in P4, P6-04/P6-05, and add a
  mixed-tool regression in P7-01.
- [ ] **F-03 Corrupt state becomes empty state:** add a strict toolbox-state
  reader that fails closed on malformed, truncated, non-object, wrong-version,
  or digest-invalid state. A failed read must never produce default empty state
  when the state file exists. Address in P5-02 and verify in P7-06.
- [ ] **F-04 Non-atomic and cross-process-unsafe state updates:** write toolbox
  state through temp-file, flush, fsync, and atomic replace under a process-safe
  lock. Perform expected-revision compare-and-swap inside that transaction.
  Address in P5-03/P5-04 and verify concurrent-process and interrupted-write
  behavior in P7-06.
- [ ] **F-05 Candidate/active routing ambiguity:** candidate registrations must
  remain non-routable during environment preparation, spawn, and warmup.
  Execution must route through the persisted active `tool_routes` map rather
  than all live registrations. Address in P4-02 through P4-05 and verify with
  continuous execution during rollout in P7-03.
- [ ] **F-06 Duplicate tool names create permanent ambiguous routes:** validate
  advertised tool-name uniqueness across auto, manual, intrinsic, and guide
  tools before staging. Return a definition validation error identifying every
  conflicting stable key/profile. Address in P3-04 and verify in P7-01.
- [ ] **F-07 Auto mutation drops manual profile membership:** resolved profile
  state must be generated once from the complete definition and must contain
  all assigned auto/manual/intrinsic tool keys. Environment planning, expected
  inventory, consistency, repair, references, and GC must consume that same
  canonical membership instead of independently reconstructed partial lists.
  Address in P3-03/P3-06, P4-03, and P6-10; verify in P7-01/P7-07.
- [ ] **F-08 New top-level dependencies cannot reach installation:** complete
  dependency analysis and template/custom environment preparation before
  `load_toolbox_from_manifest()` imports staged modules. Environment build must
  not depend on a previously persisted toolbox profile. Address in P1, P2-06/
  P2-07, and P4-04; verify missing and newly approved top-level imports in
  P7-04.
- [ ] **F-09 Ambient packages and bootstrap fallback defeat dependency
  isolation:** create toolbox environments without `system_site_packages` and
  remove the toolbox bootstrap-interpreter fallback. Spawn only with the final
  receipt-verified interpreter. Address in P2-04/P2-05 and verify in P7-05 and
  P7-08.
- [ ] **F-10 Retirement precedes durable logical publication:** do not retire
  active workers before the new definition/routes are durably published. After
  publication, retirement must be recoverable and idempotent so a crash leaves
  either serving old routes or serving new routes with stale old workers ready
  for cleanup. Address in P4-05 through P4-07 and P5-05; verify every crash
  point in P7-06.

## Code to retain and build on

- [ ] Keep `ToolboxBundleFile`, auto/manual tool metadata, callback signatures,
  guide metadata, visibility, `non_restartable`, and concurrency metadata.
- [ ] Keep deterministic manifest hashing and bundle revision generation, but
  feed them the new resolved profile/environment identity.
- [ ] Keep bundle path validation, staging, startup specs, worker IPC, manifest
  loading, and exact worker tool-inventory warmup checks.
- [ ] Keep the per-toolbox mutation serialization concept, extending it to a
  process-safe state transaction.
- [ ] Keep hosted execution, gate, cancel, callback relay, and scope behavior.
- [ ] Keep `toolbox_consistency`, `toolbox_review_snapshot`, `toolbox_repair`,
  `toolbox_reconcile`, `toolbox_references`, and `toolbox_gc`, rewriting their
  state/reference logic for the new schema.
- [ ] Keep shared runtime-environment functionality used by workflow Python and
  JavaScript. Toolbox environment replacement must not remove those consumers.

## Replacement contracts

### Cross-project vocabulary and scope

| Term | Meaning |
| --- | --- |
| Tool runtime | Configured hosting/execution target to which a client connects |
| Toolbox | One deployed tool namespace governed by one `ToolboxDefinitionSpec` |
| Package environment | Verified dependency environment inside the tool runtime |
| Environment template | Parent-owned immutable dependency base |
| Resolved profile | Parent-internal grouping by package environment and sandbox policy |

Environment templates and resolved profiles are implementation/deployment
details. They are not tool categories, user-saved runtime selections, or
consumer-visible toolbox identities.

Atomicity, revision compare-and-swap, and advertised-name uniqueness are scoped
to one `toolbox_id` on one tool runtime. Multiple toolbox references remain
concurrently executable. Routing always includes toolbox identity, and the same
advertised tool name is valid in different toolboxes.

### Complete toolbox definition

Add a typed `ToolboxDefinitionSpec` in
`src/hosting/toolbox/bundle_models.py`. It replaces all incremental register and
unregister payloads.

```python
ToolboxDefinitionSpec = {
    "contract": "hosting.toolbox.definition",
    "toolbox_id": "workspace-tools",
    "expected_revision": "sha256:<current-definition-hash>",  # null only on create
    "auto_requests": [ToolboxAutoAssignmentRequestV2],
    "manual_requests": [ToolboxManualAssignmentRequestV2],
    "intrinsics": {
        "names": ["symbolic_algebra"],
        "include_guides": True,
        "sandbox_policy": {...},
    },
}
```

Continue using the current auto/manual stable-key rules unless Phase 0 finds a
real collision in the consumer. Reject duplicate stable keys, duplicate tool
names across profiles, conflicting staged paths/content, and contradictory
profiles before environment work starts.

Remove `python_executable` from consumer toolbox definitions. The host chooses
the runtime interpreter and records its runtime/ABI identity. The tool runtime
is selected by the configured host connection/`HostedToolBoxRef`, not by a
field inside the toolbox definition.

### Per-request dependency intent

Replace `SandboxProfileSpec.environment_name` and `required_imports` with a
dependency request attached to each auto/manual request:

```python
ToolboxDependencyRequest = {
    "mode": "auto|template|custom",
    "template_id": None,
    "declared_imports": [],
    "package_requirements": [],
}
```

- `auto` scans source and chooses the smallest compatible template.
- `template` requires the named immutable template and verifies coverage.
- `custom` uses a selected base template plus explicit package requirements.
- `declared_imports` handles dynamic/optional imports that static analysis
  cannot prove.
- `package_requirements` contains installable distribution requirements and is
  never inferred merely by copying an import name.

Sandbox policy remains per request. The host derives the resolved profile ID
from canonical sandbox policy plus resolved environment identity. Remove
consumer-supplied `profile_id` so contradictory profile aliases cannot group
together.

### Plan and apply APIs

The consumer flow is:

```python
plan = toolbox.plan_definition(definition)
approval = None
if plan["user_projection"]["state"] == "approval_required":
    approval = toolbox.approve_definition_plan(plan_id=plan["plan_id"])

started = toolbox.apply_definition(
    definition=definition,
    plan_id=plan["plan_id"],
    request_id=stable_request_id,
    dependency_approval_ref=(approval or {}).get("approval_ref"),
)
operation_ref = started["operation"]
```

Planning returns detected imports, import evidence, mapped distributions,
selected templates, custom deltas, unresolved imports, policy denials, profile
diffs, and whether apply can proceed. Planning does not write logical toolbox
state, spawn workers, or install packages.

`approve_definition_plan()` is authenticated parent behavior, not a client
assertion. If policy permits approval, it returns an opaque parent-minted
`ToolboxDependencyApprovalRef`. The parent stores and validates its binding to
the authenticated actor/authority, exact toolbox/plan/definition, exact custom
delta digest, catalog and package-policy revisions, decision, and expiry.
Apply accepts no approval Boolean. It rejects expired, cross-actor, wrong-plan,
changed-delta, changed-policy, and changed-catalog references without exposing
another actor's approval state.

Apply revalidates the definition hash, expected active revision, catalog,
policy, and approval reference before dispatch. It returns immediately using
the existing `hosting.operation_status` shape and a parent-minted
`HostedOperationRef`. Extend `HostedExecutionKind` with
`toolbox_definition_apply`, use a `toolbox_id` selector, and include all
dispatch-affecting definition/plan/approval identities in its fingerprint.

Extend the strict hosted-operation status with optional bounded progress for
apply: stable phase/code, completed/total units where meaningful, update time,
and user-safe summary. Status, reconnect recovery, result retrieval, and retry
use the existing generic hosted-operation APIs and repository.

Cancellation is allowed while queued and during pre-publication build/stage/
warmup when candidate cleanup is safe. Publication is a non-cancellable commit
boundary. Cancellation after publication begins must not roll back the active
definition; the operation completes draining/cleanup idempotently.

### Authoritative read API

Add `toolbox.get_definition()` / `toolbox-get-definition`. It returns the
complete canonical active definition, active revision used by
`expected_revision`, toolbox and tool-runtime identity, active tool inventory,
bounded rollout state, and stable user-safe diagnostics.

The read is actor-authorized and side-effect-free. It does not discover, start,
repair, or reconcile workers. Revision-conflict responses direct the client to
read again instead of embedding a potentially large stale definition.

### Template administration and physical materialization

Template control and package installation are separate responsibilities:

- An authenticated hosting administrator manages immutable template manifests
  and lifecycle through daemon control-channel APIs. Normal toolbox consumers
  may list/describe/select templates but cannot publish, mutate, deprecate, or
  revoke them.
- Control-channel requests carry template identity, immutable lock/manifests,
  and approved package-artifact or repository references. They do not carry a
  prebuilt venv, arbitrary host paths, or authority to run unrestricted pip.
- The daemon-owned environment builder on the target tool runtime performs the
  physical artifact download, local venv materialization, installation, receipt
  verification, and import probes. The worker later performs ordinary Python
  imports only from that verified local package environment.
- Interactive or physical access to the target box is not required when the
  daemon can reach policy-approved artifact sources. Air-gapped/offline hosts
  require the locked artifacts to be preseeded through the host's approved
  deployment/artifact channel; this is provisioning, not a toolbox-client
  installation fallback.
- Apply may lazily materialize a published template, and administrators may
  prewarm it. Both paths use the same lock verification and content-addressed
  cache and are observable as durable host operations where execution may be
  long-running.

### Required initial Python environments

The parent release must ship two visible, parent-owned environment templates.
Their stable logical names do not contain version suffixes:

1. `core` is the smallest supported Python execution template. It contains the
   installed hosting/worker artifact and its required protocol, serialization,
   validation, and sandbox-harness dependencies, but no optional mathematical,
   data, document, network-client, or model package set. It is a useful public
   template for standard-library-only toolbox functions, Python node modules
   and snippets, and workflow helper workers whose declared imports fit it; it
   is not merely a hidden bootstrap venv.
2. `py-compute` contains the complete `core` dependency specification plus
   pinned NumPy, SymPy, NumExpr, and every third-party import required to load
   and execute the project built-ins. It is the standard compute-friendly
   template and must run all shipped built-ins without a custom environment.

Logical template names remain stable. Template digest, complete lock digest,
catalog revision, Python ABI, platform, parent worker artifact digest, and
isolation version identify immutable revisions and invalidate materializations.
The relationship between `core` and `py-compute` is dependency provenance, not
physical venv cloning, shared `site-packages`, or runtime inheritance. Every
resolved environment is independently materialized and verified from its
complete lock.

Each template is paired by default with the compute-only sandbox policy:
sandbox enforcement enabled, no filesystem rules or artifact roots, subprocess
disabled, network disabled, and all brokered filesystem/HTTP/subprocess
capabilities disabled. Package availability conveys no sandbox or host-API
authority. If the target platform cannot enforce this policy, the host must
refuse to advertise or launch either template rather than silently weaken it.

The exact template manifests/locks and compute-only policy are shipped as
parent project resources and referenced from host project configuration. Daemon
startup validates and materializes/prewarms both before reporting standard
Python deployment readiness. Missing, stale, or unverifiable required
materializations produce degraded/not-ready host status with stable
diagnostics. Physical-box access is not required after daemon administration is
configured.

`auto` dependency planning selects the smallest compatible environment: `core`
for standard-library-only work, then `py-compute`, then another allowed
parent-owned template, and only then an approved custom environment. Planning
unions imports and packages from user code and selected built-ins. A selected
tool import that is absent from the resolved environment is a planning/build
failure, never a post-rollout worker failure. Custom environments use a
complete approved resolved lock and are independently materialized; they do not
inherit a live template venv.

The local model worker environment is a separate exclusive runtime, not an
environment template. It is assembled from the project's `pyproject.toml` lock
and the configured optional model packages required to run local `mp13-engine`
models. Its preinstalled activation path is an internal model-worker launch
detail. Toolbox, Python node, snippet, helper, and custom-environment requests
must not select it, use it as a base, receive its interpreter path, or execute
arbitrary code through it.

Expose only a bounded read-only model-runtime projection through model/host
status APIs: readiness, Python/runtime compatibility, engine artifact and lock
digests, configured optional-package set, materialization revision, and stable
diagnostics. Model operations are the only execution route for this runtime and
retain their own authorization and resource policy. Generic environment APIs
must not turn the convenience of a preinstalled model venv into execution
authority.

Omitting sandbox policy from a toolbox or compatible Python worker request
selects compute-only. Any wider filesystem, network, brokered I/O, artifact,
host-API, or subprocess policy must be explicit, validated separately from
dependencies, and authorized by host policy.

### User-safe and operator projections

Plan, read, apply-progress, and terminal responses include a stable
`user_projection` suitable for `ready`, `setup_needed`, `approval_required`,
`deploying`, and `deployment_failed`. Diagnostics contain stable codes,
bounded summaries, affected tool keys where safe, and remediation categories.

Raw profile IDs, engine IDs, package/filesystem paths, environment keys,
installer output, and internal locks are excluded. They may appear only in a
separately authorized bounded `operator_details` projection or administrative
review API.

## Itemized implementation plan

### Phase 0 - Freeze code-derived contracts and breaking handoff

- [x] **P0-01** Inventory current parent and `mp13-docs` uses of auto, manual,
  intrinsic, environment-description, and install APIs. Record every method,
  command, payload field, and persisted field that will disappear.
- [x] **P0-02** Inventory imports used by actual hosted functions and intrinsics.
  Use that inventory, not speculative package groupings, to choose the first
  template set and package locks.

  Evidence (2026-08-08): the `HOSTED-TOOLBOX-DEFINITION` handoff inventories
  all parent hosted-ref/channel/service/command/payload/state removals, the
  concrete `mp13-docs` deployment/authoring/persistence call sites, and actual
  parent-intrinsic/demo/dependent-starter imports with import-to-distribution
  distinctions. Verification passed both predeclared repository-wide `rg`
  searches and an AST assertion covering all 19 old dispatch commands, every
  deprecated hosted-ref method, NumPy/SymPy/NumExpr intrinsic imports, the
  Matplotlib starter import, and the stale non-imported Requests declaration.
  Commit: recorded by git history under `docs: freeze hosting inventories
  (P0-01 P0-02)`.
  Resumed-session audit: verified against commit `4efc3f0`; the subsequent
  public-contract slice did not remove or weaken the inventory or handoff
  instructions.
- [x] **P0-03** Freeze `ToolboxDefinitionSpec`, version-2 request models,
  `ToolboxDependencyRequest`, template descriptors, plan/apply results, strict
  validation limits, and error codes.
- [x] **P0-04** Freeze canonical hashes for definition revision, resolved
  profile identity, environment identity, bundle manifest, template lock, and
  custom lock. Add cross-process test vectors.

  Evidence (2026-08-08): `hosting.toolbox.identity` implements Unicode-NFC,
  finite-JSON, canonical-order, domain-separated SHA-256 identities for all six
  required domains. `HOSTED_TOOLBOX_HASH_VECTORS.json` publishes fixed inputs
  and digests. `python -m pytest tests/test_hosted_toolbox_identity.py
  tests/test_hosted_toolbox_contract_docs.py -q` passed (10 tests), including
  fresh interpreters with `PYTHONHASHSEED=1` and `987654`; three focused
  existing bundle staging/manifest regressions also passed. Clean-environment,
  daemon, and persistence categories were not applicable because the helpers
  are not yet wired into those runtime paths. The dependent handoff states that
  predictive digests use these vectors but never replace authoritative reads.
  Commit: recorded by git history under `feat: add canonical toolbox identities
  (P0-04)`.
  Subsequent-slice audit: verified against commit `cfbaf1b`; the operation
  contract extension did not change any canonicalization/domain/input rule or
  published vector, and all identity/doc tests remained in the required suite.
- [x] **P0-05** Decide package index/artifact policy, online build approval,
  template administration roles, remote control-channel management methods,
  signed/immutable manifest and artifact requirements, offline artifact
  preseeding, supported Python ABI/platform combinations, build/prewarm timeout,
  lifecycle/revocation behavior, audit, and cache retention. Physical box login
  must not be part of the normal management contract.

  Evidence (2026-08-08): the durable contract freezes four distinct roles, six
  consumer/admin control methods, Ed25519-signed canonical manifests,
  digest/size-verified allowlisted HTTPS or offline-preseeded artifacts,
  default-denied online resolution with exact parent approval, CPython 3.12
  Windows/Linux x64 targets, bounded fetch/resolve/build/probe/operation
  timeouts, immutable active/deprecated/revoked lifecycle behavior, audit event
  contents/redactions, quarantine, and reference/grace/LRU cache retention.
  The handoff explicitly removes dependent-side installation, venv/path, and
  artifact administration. Contract-doc command passed 6 tests and the exact
  forbidden-history search returned no matches; executable test categories were
  not applicable to this policy-only slice. Commit: recorded by git history
  under `docs: freeze template deployment policy (P0-05)`.
- [x] **P0-06** Freeze `ToolboxDependencyApprovalRef`: minting authority,
  authenticated actor binding, plan/definition/delta/policy/catalog scope,
  expiry, revocation, retry/consumption behavior, audit fields, and
  unauthorized-response behavior. Prohibit Boolean approval authority.
- [x] **P0-07** Freeze authoritative `get_definition()` snapshot semantics,
  source visibility/size limits, revision-conflict recovery, and side-effect-
  free behavior.
- [x] **P0-08** Extend the existing hosted-operation contract with
  `toolbox_definition_apply`, bounded progress, request recovery, terminal
  diagnostics, and the pre-publication/post-publication cancellation boundary.

  Evidence (2026-08-08): `HostedExecutionKind` now includes
  `toolbox_definition_apply`; `HostedOperationProgress` strictly bounds and
  round-trips phase/code/count/time/summary/cancellability; the atomic
  repository persists monotonic checkpoints and rejects terminal updates,
  timestamp regression, and reversal of the cancellation boundary. The generic
  contract freezes the full apply fingerprint input, authenticated request
  recovery, terminal diagnostic placement, and publication boundary. Contract,
  repository, and toolbox-doc command passed 52 tests, including repository
  recreation/interruption and existing multi-process idempotency; service and
  workflow regression command passed 15 tests. Clean-environment coverage was
  not applicable. Commit: recorded by git history under `feat: extend hosted
  operation progress (P0-08)`.
- [x] **P0-09** Freeze per-toolbox scope: atomicity and name uniqueness are per
  toolbox; duplicate names across toolboxes are valid; routing includes
  `toolbox_id`; multiple toolbox references remain concurrent.
- [x] **P0-10** Freeze the cross-project vocabulary and user-safe/operator
  projection split defined above.
- [x] **P0-11** Create a durable parent-owned public contract document,
  `src/hosting/HOSTED_TOOLBOX_CONTRACT.md`, covering typed models, strict field
  validation, actor authorization, plan/approval/read/apply semantics, durable
  operation behavior, progress/cancellation, per-toolbox scope, user/operator
  projections, retention, and error codes. This document is the normative
  specification for the parent and dependent projects. Write only the target
  supported contract; do not include legacy names, migration notes, old state
  formats, historical references, or compatibility language.

  Evidence (2026-08-08): `HOSTED_TOOLBOX_CONTRACT.md` freezes strict JSON
  models and limits, exact public client signatures, validation and stable
  errors, per-toolbox scope, actor/source authorization, immutable bounded
  plans, opaque actor/request-bound approval references, side-effect-free
  authoritative reads, durable apply/progress/cancellation/result behavior,
  retention, and the user/operator projection split. The transient handoff
  links the durable contract while retaining concrete dependent rewrite and
  removal instructions. `python -m pytest
  tests/test_hosted_toolbox_contract_docs.py -q` passed (5 tests); the
  predeclared forbidden-vocabulary `rg` search returned no matches. Runtime test
  categories were not applicable because this slice changes documentation only.
  Commit: recorded by git history under `docs: freeze toolbox public contract`
  with the completed item IDs in its body.
  Subsequent-slice audit: verified against commit `ec0bd66`; adding the
  canonical-identity section and vectors preserved every strict model,
  authorization, scope, projection, and client-flow requirement, and the full
  contract-doc suite remained green.
  Deployment-policy audit: the P0-05 additions describe supported
  administration only and preserve the public model/client algorithm and the
  durable document's no-history rules; the expanded suite remained green.
- [x] **P0-12** Add migration change set `HOSTED-TOOLBOX-DEFINITION` to
  `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md` with removed APIs/fields,
  old-to-new examples, state archival procedure, parent baseline, release
  placeholder, hosted-operation extension, approval-reference flow,
  authoritative read contract, vocabulary/scope rules, projection contract,
  exact dependent-project requirements, and a link to the durable contract.
  Treat this entry as the adoption delta and release handoff, not as the
  normative contract specification.

  Evidence (2026-08-08): the handoff now contains syntactically valid old/new
  Python examples for complete definitions, approval, durable apply/recovery,
  empty-definition teardown, and conflict replanning; it freezes the exact
  `toolbox-state-archive-v1` command, path/digest/daemon/locking/archive safety
  checks, and code-matched rollback boundary. Its inventory matrix maps every
  prior call-path category to replacement logic and behavior to delete. The
  focused document suite passed 12 tests, the independent identity/operation
  audit passed 52 tests, and the exit audit covered 21 public operation names,
  all 19 daemon commands, and all six dependent file groups. The durable
  contract forbidden-vocabulary search returned no matches.
- [x] **P0-13** Freeze the initial environment catalog contract: stable `core`
  and `py-compute` names without version suffixes, shipped complete manifests
  and locks, immutable revision identity, project-config keys, compute-only
  sandbox policy, supported-platform enforcement requirements, startup
  validation/prewarm behavior, readiness diagnostics, smallest-compatible
  selection rules, and authorization required to widen sandbox capabilities.
- [x] **P0-14** Freeze cross-worker use of `core`: standard-library-only toolbox
  functions, `workflow_python(profile=node)` modules and snippets, and workflow
  helper workers may resolve to it, while each retains its own execution
  contract, import allowlist, sandbox policy, pool identity, and lifecycle.
- [x] **P0-15** Freeze the model-runtime boundary: derive its complete lock from
  `pyproject.toml` plus the configured optional model package set; keep it
  exclusive to model operations; expose bounded readiness/capability metadata
  but no venv path, generic interpreter selection, arbitrary-code route, or use
  as a template/custom-environment base.

  Evidence (2026-08-08): the durable contract freezes the two signed complete
  initial templates, immutable revision tuple, exact host config keys,
  compute-only policy, supported-target/prewarm/readiness behavior, and
  smallest-template selection. It separately freezes cross-worker `core`
  resolution without process/protocol sharing and an exclusive model-runtime
  lock, configuration, authorization, and bounded status surface. The dependent
  handoff gives exact environment/readiness/model-runtime code removal and
  replacement rules. The predeclared contract-doc command passed 10 tests and
  its forbidden-vocabulary search returned no matches; the Phase 0
  contract/identity/operation audit passed 62 tests. Runtime test categories
  were not applicable to this documentation-only freeze.

Exit gate: the breaking-change entry and typed examples cover every current
call path found in the inventory.

### Phase 1 - Template catalog and dependency analysis

- [x] **P1-01** Add a parent-owned `ToolboxEnvironmentTemplateSpec` containing
  immutable template ID, Python/runtime constraints, platform constraints,
  locked distributions, exposed import roots, lock digest, and provenance.
- [x] **P1-02** Add a reviewed import-to-distribution catalog supporting aliases,
  package extras, and version constraints. Seed it only from Phase 0 inventory;
  likely candidates visible in current code/tests include calculator/symbolic,
  NumPy/SymPy, and HTTP packages.

  Evidence (2026-08-08): `toolbox/catalog.py` adds frozen strict template,
  locked-distribution, provenance, reviewed-rule, and deterministic catalog
  models. Template data includes Python/runtime/platform constraints, complete
  exact distributions, exposed roots, lock/worker digests, isolation version,
  and provenance. The reviewed seed contains only the inventoried NumPy, SymPy,
  NumExpr, Requests, and Matplotlib roots while the model supports distinct
  import/distribution names, package aliases, extras, and normalized version
  constraints. Focused tests passed 16 cases; identity/runtime-key regressions
  passed 20 tests; compile and diff checks passed.
- [x] **P1-03** Move intrinsic dependency knowledge out of
  `SandboxProfileSpec.intrinsics_profile_id()` and into dependency-only
  intrinsic registry metadata that can be read on `core` without importing
  implementations. Remove its hard-coded calculator/symbolic branching,
  eliminate eager optional-package loading during registry inspection, and
  declare and pin the currently undeclared SymPy requirement.

  Evidence (2026-08-08): `mp13_intrinsics_metadata.py` is a standard-library-
  only discovery/dependency registry with exact module-load roots and pinned
  distribution requirements for both intrinsic families and their guides.
  Discovery, target validation, and loaded-tool listing no longer import the
  implementation registry; only intrinsic initialization does. Environment
  identity/probes merge metadata roots and use a canonical dependency-derived
  profile ID; the calculator/symbolic branching was deleted from
  `SandboxProfileSpec`. SymPy 1.14.0 is now an exact direct dependency and the
  Poetry lock is current. Focused tests passed 7 cases, the complete toolbox
  sandbox suite passed 138 tests, `poetry check --lock`, compile, and diff checks
  passed. The real-daemon concurrency regression now asserts worker-reported
  interval overlap instead of a scheduler-sensitive wall-clock ceiling.
- [x] **P1-04** Add an AST analyzer over the existing `ToolboxBundleFile`
  contents. Classify standard-library, local staged modules, parent runtime,
  known third-party, declared dynamic/optional, and unresolved imports.
- [x] **P1-05** Resolve source evidence plus explicit declarations into
  distribution requirements. Reject incompatible declarations and unresolved
  required imports with file/line diagnostics.
- [x] **P1-06** Select the smallest allowed template covering the complete
  resolved requirement set. If none matches, produce a custom delta from an
  allowed base template.

  Evidence (2026-08-08): `dependency_analysis.py` implements a bounded,
  side-effect-free AST pipeline over normalized staged files. It classifies
  standard, local/relative, parent, reviewed third-party, explicitly declared
  optional/dynamic, and unresolved imports with deterministic file/line
  evidence. Resolution combines reviewed mappings, aliases/extras, explicit
  PEP 508 requirements, and real PEP 440 intersections; syntax, duplicate
  paths, dynamic expressions, unreviewed roots/packages/extras, and conflicting
  constraints fail with stable diagnostics. Selection filters exact ABI/
  platform/allowlist, chooses the smallest complete template, or chooses the
  compatible base minimizing the exact custom delta. Focused tests passed 15
  cases; catalog/identity regressions passed 21 tests; the Poetry lock, compile,
  and diff checks passed. No worker or mutable host state is touched.
- [x] **P1-07** Validate requested templates and custom packages against runtime,
  platform, package allow/deny policy, index policy, and intrinsic requirements.

  Evidence (2026-08-08): `dependency_policy.py` adds a strict immutable policy
  and fail-closed validation of exact ABI/platform, selected/requested template,
  custom-delta permission and approval classification, normalized distribution
  allow/deny rules with deny precedence, normalized HTTPS-only index origins,
  online-resolution policy, and import-root/version-complete intrinsic
  requirements. Recursive dependency metadata cannot assert approval or
  sandbox capability. Focused tests passed 11 cases; dependency/catalog
  regressions passed 31 tests; compile and diff checks passed. The validator is
  pure and starts no resolution, build, or worker action.
- [x] **P1-08** Add actor-authorized template list/describe APIs for consumers
  and operator-authorized publish/deprecate/revoke APIs over the daemon control
  channel. Publishing accepts immutable lock/manifests and approved artifact
  references, never a mutable package list bound to an existing template ID.

  Evidence (2026-08-08): `service/toolbox_catalog.py` adds a strict
  process-locked, fsync/atomic-replace catalog with immutable canonical revision
  identities, signed manifest value, digest/size/source artifact references,
  multiple stable-ID revisions, active pointer, lifecycle transitions, catalog
  revision, bounded audit, and fail-closed reads. Consumer list/describe returns
  only logical bounded descriptors. Publish/deprecate/revoke and reads are wired
  through service, daemon, channel, CLI/SSH routing, command policy, and roles;
  worker/config/diagnostic roles read while only admin mutates. Focused tests
  passed 11 cases including multi-process/restart/corruption/role/transport
  coverage; existing channel/CLI/auth regressions passed 95 tests; compile and
  diff checks passed. Materialization remains P1-09.
- [x] **P1-09** Add an operator prewarm/materialize operation that executes on
  the target runtime host and returns a durable operation ref with progress and
  terminal verification diagnostics. Apply may invoke the same builder lazily.
  Evidence (2026-08-08): added the administrator-only daemon/channel/CLI
  `toolbox-template-prewarm` path, `toolbox_template_prewarm` hosted-operation
  kind and template selector, strict target-host materializer interface,
  process-safe exact-revision verification receipts, bounded phase progress,
  terminal diagnostics, request idempotency/recovery identity, and fail-closed
  unconfigured-builder behavior. Consumer readiness changes only after exact
  template/target, complete artifact/import coverage, and derived environment
  identity are verified and atomically committed. Focused tests passed 6 cases;
  operation/catalog regressions passed 58, channel/CLI/auth regressions passed
  95, contract documentation passed 12, and compile/diff checks passed.
- [x] **P1-10** Add strict tests for relative/local imports, alias mappings such
  as import name versus distribution name, optional imports, dynamic imports,
  duplicate staged paths, intrinsic requirements, role separation, immutable
  publish conflicts, remote management, and offline artifact availability.
  Evidence (2026-08-08): the strict dependency analysis/policy suites cover
  relative/local, alias, optional/dynamic, duplicate-path, and intrinsic cases;
  catalog/materialization suites cover role separation, immutable conflicts,
  daemon/channel/SSH management, and fail-closed offline availability. Added
  explicit prewarm exclusions to every non-admin catalog role and an exact SSH
  prewarm payload assertion. Required matrices passed 26, 18, and 95 tests;
  diff checks passed.
- [x] **P1-11** Add shipped `core` and `py-compute` descriptors and complete
  immutable locks plus the compute-only sandbox policy preset. Materialize and
  probe both during normal host setup, advertise only verified templates, make
  the planner choose the smallest compatible template, and reject package
  metadata that attempts to grant sandbox capabilities.
  Evidence (2026-08-08): shipped strict package resources for exactly `core`
  and `py-compute` with independent complete locks derived from the parent
  validation closure and exact intrinsic metadata/Poetry versions. Added
  canonical lock/manifest/worker/resource identity validation, recursive
  capability-metadata denial, the exact enforceable compute-only preset,
  normal setup publication/prewarm through the shared durable materializer,
  bounded required-template readiness, and smallest exact planner assertions.
  Focused tests passed 6; planner/policy/catalog passed 42;
  catalog/prewarm/setup-state passed 21; contract docs passed 12; compile and
  diff checks passed. Physical artifact installation remains Phase 2.
- [x] **P1-12** Use the same template resolver and materialization receipts for
  toolbox, Python node, snippet, and helper worker classes without merging their
  worker processes or public contracts. Probe standard-library execution on
  `core` and every shipped built-in on `py-compute` in clean environments.
  Evidence (2026-08-08): added a shared pure resolver over staged-source
  analysis, intrinsic requirements, active templates, and exact verified target
  receipts. It produces consumer-specific binding identities for toolbox, node,
  snippet, and helper while preserving existing runtime families, worker pools,
  protocols, and APIs; resolution performs no worker discovery/start or state
  mutation. Isolated interpreter processes executed standard-library work and
  every shipped compute intrinsic/guide. Focused tests passed 9; dependency and
  shipped-template regressions passed 32; existing workflow/helper/operation
  regressions passed 37; contract docs passed 12; compile/diff checks passed.
- [x] **P1-13** Add the read-only model-runtime status projection and enforce
  that generic template planning, custom environment building, Python worker
  launch, and control-channel requests cannot select or reveal the exclusive
  model environment. Test denial even when its venv is already installed and
  healthy.
  Evidence (2026-08-08): added the exact strict ten-field bounded
  `ModelRuntimeStatus`, read-only authorized daemon/channel/CLI routing, and a
  central generic-selection guard covering template, dependency/custom-build,
  workflow Python environment/launch, and control inputs without intercepting
  legitimate model commands. Paths/interpreters/raw locks are never projected;
  every explicit/alias selector is denied even with a verified healthy model
  identity. Focused tests passed 12; shared resolver/policy/catalog passed 32;
  channel/CLI/auth/model-service regressions passed 133; docs/shipped-template
  tests passed 18; compile/diff checks passed.

Exit gate: the same source/metadata/catalog inputs produce the same resolved
environment request and diagnostics without starting a worker.
Evidence (2026-08-08): a fresh service over persisted catalog/materialization
receipts returned an identical `core` resolution/binding for the same staged
relative/local source; engine registrations remained empty and hosted-operation
checkpoint bytes were unchanged. The shared resolver suite passed 10 tests.

### Phase 2 - Toolbox-specific hermetic environment builder

- [x] **P2-01** Split toolbox-specific environment behavior from the shared
  runtime environment code in `src/hosting/toolbox/environment.py`. Preserve or
  relocate primitives still used by `RuntimeEnvironmentManager` and workflow
  Python/JavaScript.
- [x] **P2-02** Replace toolbox environment-description lookup with a resolved
  template/custom-lock input. Remove environment-name inheritance from toolbox
  environment identity.
- [x] **P2-03** Derive toolbox `venv_key` from runtime/ABI/platform identity,
  immutable template lock digest, optional custom resolved-lock digest, and
  isolation policy. Do not include each function's raw import subset.
- [x] **P2-04** Create toolbox venvs without `system_site_packages`; ensure pip
  or the selected installer exists inside the build environment explicitly.
- [x] **P2-05** Remove `runtime_python_executable()` bootstrap fallback for
  toolbox executors. Do not alter workflow fallback behavior unless separately
  required by workflow contracts.
- [x] **P2-06** On the target runtime host, materialize built-in templates from
  policy-approved locked artifacts and resolve, lock, install, and receipt-
  verify approved custom deltas before worker spawn. Never accept a client-
  supplied venv or execute installation on the dependent-project machine.
- [x] **P2-07** Run import probes with the final environment interpreter for all
  resolved import roots before staging a candidate worker.
- [x] **P2-08** Publish a cache entry only after lock, receipt, and import-probe
  verification. Quarantine partial/failed builds.
- [x] **P2-09** Deduplicate concurrent builds by environment key using a
  process-safe lock. Track references and defer deletion to GC with a grace
  period instead of deleting immediately in mutation failure paths.
- [x] **P2-10** Materialize and verify derived environments from the complete
  base-plus-delta resolved lock. Prove that they do not read or inherit another
  venv's `site-packages`, and report required-base readiness during daemon
  startup/configuration checks.

Exit gate: no toolbox worker can start with ambient host packages, a bootstrap
interpreter, an unverified lock, or a failed import probe.

Evidence (2026-08-08): the toolbox-only builder created real venvs with
in-environment pip and `include-system-site-packages = false`, installed only
digest/size-verified administrator-source wheels with `--no-index --no-deps`,
verified the exact complete distribution lock and all import roots under the
final interpreter, and atomically published a strict receipt. An ambient
parent-only `pytest` probe was absent. Failed final-interpreter probes produced
one quarantined candidate and no published path. Four concurrent threads and
two fresh spawned processes converged on one physical environment and retained
all references; grace-period GC removed it only after release. A complete
base-plus-delta build imported both packages while its `sys.path` excluded the
base venv. Catalog prewarm used the same builder, required-template readiness
was receipt-gated, and a configured orchestrator launched only the resolved
published Python rather than its deliberately invalid bootstrap executable.
The hermetic builder suite passed 11 tests; environment/prewarm/shipped
template regressions passed 19; workflow/runtime regressions passed 29; and
focused legacy toolbox environment regressions passed 3 with 135 deselected.

### Phase 3 - Definition planner and resolved profile model

- [x] **P3-01** Add strict `ToolboxDefinitionSpec` parsing and canonicalization
  to `bundle_models.py`; reject unknown version-1 profile/dependency fields.
- [x] **P3-02** Replace public `SandboxProfileSpec` use with an internal
  `ResolvedToolboxProfileSpec` containing host-derived profile ID, resolved
  environment key/lock digest, canonical sandbox policy, and assigned tool keys.
- [x] **P3-03** Group requests only after dependency resolution. Functions with
  different import subsets share a profile when the resolved environment and
  sandbox policy are identical.
- [x] **P3-04** Validate advertised tool names as unique within one toolbox
  definition before grouping so `_route_toolbox_registration()` cannot face a
  legitimate duplicate route. Duplicate names in different toolboxes are
  valid and must not be compared.
- [x] **P3-05** Continue producing `ToolboxBundleSpec`, but set
  `dependency_lock_hash` from the resolved environment and serialize the new
  resolved profile shape into the manifest.
- [x] **P3-06** Compare proposed profiles with persisted active profiles by
  manifest hash, environment key, and policy digest. Classify each as reused,
  added, replaced, or removed.
- [x] **P3-07** Persist bounded expiring plans keyed by `plan_id`, definition
  hash, expected revision, catalog revision, and package-policy revision.

Exit gate: planning identifies unchanged profiles without staging, spawning, or
changing registrations.

Evidence (2026-08-08): exact manifest/environment/policy triples classified as
reused; source and policy changes retaining assigned tool ownership classified
as replaced; unrelated profiles classified as added/removed. Strict plan IDs
changed independently with expected active, catalog, and package-policy pins.
Atomic records survived repository recreation, expired without refresh, failed
closed on corrupt/truncated/wrong-contract JSON, and retained two distinct
plans written by fresh concurrent processes. Planning/persistence created no
bundle directory or engine state. The plan-repository suite passed 10 tests;
planner/identity regressions passed 16; operation-repository regressions passed
23; compile/diff checks passed.

### Phase 4 - Candidate rollout and atomic active routing

- [x] **P4-01** Extend `HostedExecutionKind`, hosted-operation fingerprinting,
  repository metadata, status normalization, request recovery, and cancellation
  dispatch for durable `toolbox_definition_apply` operations. Add bounded
  persisted progress checkpoints for validation, environment build, staging,
  warmup, publication, draining, and cleanup.

  Evidence (2026-08-08): definition applies recover only through the exact
  toolbox-scoped receipt namespace and reject engine selectors. Queued/running
  cancellation and publication checkpoints serialize under the same durable
  repository lock; a pre-publication cancellation performs candidate cleanup
  and persists a bounded terminal result, while a publication winner returns
  `apply_publication_committed` without cleanup or rollback. Focused operation,
  contract, and repository tests passed 51 including the cancellation/
  publication race; operation/workflow regressions passed 28; compile and diff
  checks passed.
- [x] **P4-02** Refactor `ToolboxSandboxOrchestrator` to accept resolved profile
  assignments and skip staging/spawn for profiles classified as reused.
- [x] **P4-03** Spawn added/replaced workers as non-routable candidates. Add an
  explicit candidate/active registration state or ensure routing never scans
  candidate registrations.
- [x] **P4-04** Reuse `_ensure_toolbox_assignments_ready()` to verify RPC
  readiness, exact expected tool inventory, environment receipt status, and
  candidate metadata before publication.

  Evidence (2026-08-08): the resolved orchestrator matched every proposed
  profile to its pinned bundle/change classification, skipped all stage/
  materialize/spawn work for reused profiles, and registered added/replaced
  workers only as candidates. Candidate registrations were excluded from
  scan-based routing. Readiness rejected extra inventory and a corrupt/mismatched
  hermetic receipt, and required exact resolved profile/environment metadata.
  Focused rollout/planner/hermetic tests passed 25; the complete toolbox sandbox
  suite passed 138 after correcting its legacy environment-description rebuild
  adapter; compile and diff checks passed.
- [x] **P4-05** Persist an explicit active `tool_routes` map from tool name to
  profile ID/engine ID in toolbox state. Change execute/describe/gate/cancel
  routing to use that map instead of scanning all live registrations.
- [x] **P4-06** Publish definition revision, active profiles, and `tool_routes`
  in one state transition. Only after publication may candidate registrations
  become active routes.
- [x] **P4-07** Treat publication as the operation's non-cancellable commit
  boundary. Persist progress before and after it so restart/cancel handling
  cannot mistake a committed revision for a candidate.
- [x] **P4-08** Drain and retire replaced/removed engines after publication.
  Preserve `non_restartable` metadata and cancellation policy for in-flight
  work.
- [x] **P4-09** If environment build, staging, spawn, readiness, inventory, or
  publication fails, retire candidates and leave old active routes untouched.
- [x] **P4-10** Treat an empty definition as a valid active revision with no
  routes. Do not delete all toolbox history during apply.
- [x] **P4-11** Persist terminal apply results through the hosted-operation
  repository. Put stable user-safe diagnostics in the normal result and expose
  raw deployment details only through authorized operator details.

Exit gate: execution sees either the complete old routing map or the complete
new routing map and can never encounter candidate/active ambiguity.

Evidence (2026-08-08): strict version-2 snapshots publish canonical definition,
resolved profiles, environment references, and a complete tool-name route map
under one process-safe transaction. Toolbox-scoped route, describe, and gate
selection use only that map. Warmup failure and pre-publication cancellation
removed candidates with byte-for-byte old active state retained; publication
made the complete replacement visible before old retirement. Busy old workers
were marked retired but left alive for in-flight work, including preserved
`non_restartable` route policy. Empty teardown published a new no-route revision
while retaining history. Normal terminal results excluded engine/profile/
environment identities; raw rollout details required explicit operator access.
Focused routing/apply/operation tests passed 11; operation-repository and full
toolbox regressions passed 161; compile and diff checks passed.

### Phase 5 - Version-2 state and transaction safety

- [x] **P5-01** Replace version-1 toolbox state with a strict version-2 schema:
  active definition revision, canonical desired definition, resolved profiles,
  explicit tool routes, bounded rollout history, and environment references.
- [x] **P5-02** Add a toolbox-specific strict reader. Invalid JSON, wrong schema,
  wrong version, and digest mismatch must fail closed; do not inherit
  `_read_json()` behavior that returns empty defaults on corruption.
- [x] **P5-03** Replace direct toolbox state writes with temp-file write, flush,
  fsync, and atomic replace under a process-safe file lock.
- [x] **P5-04** Keep the current per-toolbox in-process lock for local
  serialization, and add expected-revision compare-and-swap inside the
  process-safe state transaction.
- [x] **P5-05** Write candidate intent only if restart recovery needs it; never
  expose candidate routes as active. Define recovery for crashes before publish,
  during atomic replace, and after publish but before old-worker retirement.
- [x] **P5-06** Reject version-1 `toolbox_sandboxes.json` without translation.
  Add an operator command that validates the exact path and archives version-1
  state/bundles before initializing version 2.
- [x] **P5-07** Define rollback as code rollback plus restoration of matching
  archived state. Do not implement dual-schema reads or writes.

Exit gate: corruption and concurrent writers cannot silently reset, merge, or
partially publish toolbox state.

Evidence (2026-08-08): the version-2 repository strictly validates schema,
version, canonical definition/profile/route relations, bounded history, and a
whole-state digest; malformed/truncated/legacy/digest-mismatched files failed
closed. File and parent-directory durability plus a process lock protected
atomic replace, and two spawned writers produced one CAS winner and one revision
conflict. An injected interrupted replace preserved the prior file byte-for-
byte. Recovery removed orphan candidates before publication, activated
published route targets, retired non-routed workers, and completed operations
only when both committed progress and active revision matched. The local-only
archive command verified exact resolved root, state digest/version, stopped
daemon, release commit and bundle containment; it inventoried/fsynced and moved
version-1 state/bundles before initializing empty version 2. Focused state/
routing tests passed 10; full toolbox and CLI regressions passed 187 after the
temporary procedural adapter was explicitly isolated from strict definition
state reads; compile and diff checks passed.

### Phase 6 - Replace service, transport, client, and admin paths

- [ ] **P6-01** Add service methods `toolbox_get_definition()`,
  `toolbox_plan_definition()`, `toolbox_approve_definition_plan()`, and
  `toolbox_apply_definition()` in `toolbox_runtime.py`. Consolidate mutation
  logic there and route apply through the hosted-operation repository.
- [ ] **P6-02** Add `toolbox-get-definition`, `toolbox-plan-definition`,
  `toolbox-approve-definition-plan`, and `toolbox-apply-definition` to daemon
  dispatch, subprocess CLI dispatch, `EngineHostControlChannel`, authorization,
  audit, and policy routing. Add template list/describe for authorized consumers
  and template publish/deprecate/revoke/prewarm for hosting administrators over
  the same control transport, with distinct role/policy checks.
- [ ] **P6-03** Add `HostedToolBoxRef.get_definition()`, `plan_definition()`,
  `approve_definition_plan()`, and `apply_definition()`. Apply returns a durable
  operation status/ref; observation and recovery use generic hosted-operation
  methods. Expose template list/describe as read-only client helpers; keep
  publication/lifecycle/prewarm on an administrative channel surface. Keep
  execution/describe/gate/cancel methods unchanged except for their internal
  use of active routes.
- [ ] **P6-04** Remove `register_auto_callable`, `register_python_callable`,
  `register_manual_tool`, intrinsic register methods, every unregister/remove
  method, aliases, `mutate()`, `PendingHostedToolboxRef`, and
  `resolve_sandbox()`.
- [ ] **P6-05** Remove service/channel/daemon/CLI/auth/policy support for
  `toolbox-register-*` and `toolbox-unregister-*` commands.
- [ ] **P6-06** Remove hosted-ref and transport APIs for environment list,
  upsert, clone, resolve, apply, realize, sync, install planning, lock,
  resolution, verification, execution, and receipt verification.
- [ ] **P6-07** Remove `ToolboxEnvironmentMixin` methods and version-1
  environment-description state once workflow/shared callers are proven not to
  depend on them.
- [ ] **P6-08** Remove `environment_name`, `required_imports`, consumer
  `profile_id`, and toolbox `python_executable` from runtime payloads and tests.
  Reject these legacy fields rather than ignoring them.
- [ ] **P6-09** Update hosted chat/demo setup to submit one complete definition;
  update teardown to apply an empty definition instead of unregistering tools.
- [ ] **P6-10** Rewrite consistency, review, repair, reconcile, references, and
  GC against active revisions/routes and candidate/retired registrations.
- [ ] **P6-11** Add host project configuration for the exact required standard
  base template and compute-only policy IDs, startup materialization/prewarm,
  readiness reporting, and administrative replacement with a new immutable
  version. Do not expose these settings as mutable per-toolbox environment
  descriptions.

Exit gate: repository search finds no old toolbox mutation command, public
method, environment-description API, install sequence, field fallback, or
version-1 toolbox state path.

### Phase 7 - Tests and dependent-project cutover

- [ ] **P7-01** Replace current register/unregister/builder tests with complete
  definition create, code update, add/remove combination, intrinsic update, and
  empty-definition tests. Include a mixed auto/manual/intrinsic toolbox and
  prove that changing each category independently preserves the other two,
  their dependencies, profile membership, inventory, and routes. Also reject
  conflicting profile identities and duplicate advertised tool names.
- [ ] **P7-02** Add profile-diff tests proving unchanged profiles and
  environments are reused while only changed profiles are staged and spawned.
- [ ] **P7-03** Add routing-concurrency tests that execute continuously during
  candidate warmup and publication and never observe ambiguity, missing tools,
  or a partial definition.
- [ ] **P7-04** Add dependency tests for template hits, import/distribution
  aliases, custom fallback, denied online build, lock mismatch, failed import
  probe, and top-level missing imports before worker spawn.
- [ ] **P7-05** Add hermeticity tests proving packages available only in the
  host interpreter are unavailable to toolbox workers unless present in the
  resolved template/custom lock.
- [ ] **P7-06** Add state tests for stale expected revisions, concurrent
  processes, corrupt/truncated JSON, interrupted write, version-1 rejection,
  archival cutover, and each crash-recovery point before/during/after route
  publication and before/during/after old-worker retirement.
- [ ] **P7-07** Update repair/GC tests for candidates, active routes, retired
  workers, environment reference grace periods, and orphaned bundles.
- [ ] **P7-08** Add absence tests for every removed API, command, payload field,
  state schema, bootstrap fallback, and ambient-site-package behavior.
- [ ] **P7-09** Add approval tests for cross-actor denial, wrong plan/definition/
  delta, changed catalog/policy revision, expiry, retry, duplicate apply, and
  rejection of client-fabricated Boolean or mapping evidence.
- [ ] **P7-10** Add authoritative-read and durable-apply tests for side-effect-
  free snapshots, revision-conflict recovery, immediate operation return,
  progress persistence, reconnect/request recovery, idempotent retry, terminal
  diagnostics, safe pre-publication cancellation, denied post-publication
  cancellation, and daemon restart in every phase.
- [ ] **P7-11** Add multi-toolbox tests proving duplicate names across toolboxes
  are valid, references execute concurrently, updates are atomic only within
  the target toolbox, and routing always includes toolbox identity.
- [ ] **P7-12** Add projection tests proving stable user codes/summaries and no
  engine IDs, profile IDs, environment keys, package paths, or installer output
  leak without operator authorization.
- [ ] **P7-13** Finalize `HOSTED_TOOLBOX_CONTRACT.md` against the implemented
  public models and behavior. Rewrite
  `src/hosting/sandbox/TOOLBOX_WORKER.md` to describe the new internal worker,
  template/custom package environment, candidate rollout, active routing,
  durable apply, recovery, and GC architecture. Both documents must describe
  only the supported implementation, contain no migration/history sections,
  and remain complete after the transient breaking-changes file is deleted.
  Link the worker document to the normative contract instead of duplicating it.
- [ ] **P7-14** Complete `HOSTING_CLIENT_BREAKING_CHANGES.md` with final method
  signatures, models, template catalog, custom-build approval semantics,
  authoritative read, durable apply/progress/cancellation behavior, projection
  vocabulary, operator cutover command, release commit, adoption checklist, and
  durable-contract link.
- [ ] **P7-15** Update `mp13-docs` to build complete definitions, persist active
  revision hashes and apply operation refs, handle user-safe plan/progress/
  terminal diagnostics, recover revision conflicts through authoritative reads,
  and remove procedural mutation/environment-management logic.
- [ ] **P7-16** Repin `mp13-docs` and run parent focused/full suites plus the
  dependent project's complete hosted toolbox, workflow, recovery, approval,
  and sandbox suites.
- [ ] **P7-17** Add standard-base tests for clean-host bootstrap, project-config
  validation, eager prewarm and lazy materialization, offline preseed, missing/
  corrupt lock, unsupported sandbox enforcement, compute-only defaults,
  explicit authorized policy widening, base-plus-delta derivation, and proof
  that derived environments do not inherit another venv or host packages.
- [ ] **P7-18** After every listed dependent project confirms adoption, remove
  `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md` and remove any transient
  references to it. Confirm the durable contract and worker documentation are
  independently complete before deletion.

Exit gate: the direct replacement passes both repositories, all old behavior is
absent, and a dependent project can migrate using only the breaking-change
entry and public examples.

## Acceptance checklist

- [ ] Built-in functions declare dependencies through intrinsic registry
  metadata and load in the selected verified environment.
- [ ] Source imports map to the smallest compatible immutable template.
- [ ] A clean configured host can validate/materialize the shipped standard
  base and run a standard-library compute tool without online package access.
- [ ] The default base uses compute-only sandbox policy; package templates
  cannot grant filesystem, network, brokered I/O, artifact, or subprocess
  capabilities.
- [ ] Every derived package environment has a complete independently verified
  lock and does not inherit host or base-venv `site-packages`.
- [ ] Import names and distribution names are modeled separately.
- [ ] Unknown/dynamic imports require explicit declarations and never trigger
  silent package guesses.
- [ ] A custom package environment is locked, installed, receipt-verified, and
  import-probed before worker spawn.
- [ ] Code-only updates reuse their environment.
- [ ] Different import subsets covered by one template reuse one environment.
- [ ] Unchanged profiles remain running through another profile's update.
- [ ] Additions and removals in one definition become visible atomically.
- [ ] Candidate workers are never selected by execution routing.
- [ ] Failed preparation/warmup leaves the complete prior revision active.
- [ ] Apply returns an actor-owned durable operation ref and survives reconnect,
  duplicate retry, and daemon restart with bounded progress and diagnostics.
- [ ] Only an exact host-minted, actor-bound, unexpired approval reference can
  authorize the custom delta covered by its plan.
- [ ] `get_definition()` returns canonical active definition/revision without
  starting or repairing workers.
- [ ] Name uniqueness and atomicity are per toolbox; duplicate names in
  different toolboxes route and execute correctly.
- [ ] User projections contain stable translatable diagnostics and exclude
  internal deployment identities and paths.
- [ ] State corruption fails closed instead of producing an empty toolbox.
- [ ] Toolbox venvs cannot import undeclared ambient host packages.
- [ ] Toolbox workers never use the bootstrap interpreter as a dependency
  fallback.
- [ ] Version-1 APIs, commands, fields, state, and compatibility code are absent.
- [ ] All dependent-project requirements are recorded in
  `HOSTING_CLIENT_BREAKING_CHANGES.md` before release.

## Consumer-owned boundaries

The parent owns import analysis, template/package mapping, environment build and
verification, candidate rollout, atomic route publication, active toolbox
revision truth, authoritative reads, approval-reference validation, durable
apply operation truth/progress, repair, and derived-resource cleanup.

Consumers own:

- the complete desired auto/manual/intrinsic tool set;
- staged source and explicit dependency declarations when analysis is
  insufficient;
- sandbox policies and function metadata;
- requesting user/operator approval for policy-allowed custom builds and
  returning only the parent-minted approval reference;
- retry after an expected-revision conflict using the newly read definition;
- persisting apply operation refs and recovering status through generic hosted-
  operation APIs;
- workspace lifecycle decisions that apply a replacement or empty definition;
  and
- persistence/UI projection of the parent-returned revision and rollout state.
