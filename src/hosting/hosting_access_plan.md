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
12. **Blocking interactive/network boundary.** Existing management commands are
    dispatched through synchronous `engine_host_channel.py::_invoke` calls.
    `toolbox_plan_definition` and the pre-dispatch part of
    `toolbox_apply_definition` analyze synchronously; `toolbox_describe` may wait
    ten seconds on worker IPC; GC/repair/reconcile and hosted cancellation may
    wait on process/filesystem work. The planned resolver would add HTTPS work
    to the synchronous planning path. A request must never hold its connection
    while waiting for a human package confirmation or dependency approval.
    `daemon/local_ipc.py` already has `op-start`/`op-status`, but that store is
    separate from `service/operation_repository.py`, has only 200 snapshots, and
    does not reconcile an in-flight record after restart. Workflow/proxy stream
    sessions are execution-specific and in-memory, so they are not a durable
    substitute.

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

The final semantic sequence is `toolbox_get_definition`,
`toolbox_plan_definition`, `toolbox_confirm_definition_plan`, and
`toolbox_apply_definition`. Consumers submit the three potentially long mutation
commands through the generic `op-start` façade described below. Planning's
terminal hosted result contains the plan and alternatives. Confirmation's
terminal hosted result supplies an opaque `confirmation_ref` after acquiring
and verifying selected package artifacts. Apply accepts `plan_id`,
`confirmation_ref`, `request_id`, and, only when required,
`dependency_approval_ref`; it no longer accepts or re-resolves a second copy of
the definition. The privileged, bounded synchronous operation is
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

## Package ingress vehicle

New packages always arrive as exact wheel artifacts and converge on the daemon's
content-addressed artifact store before any environment is built.

### Online source

The daemon reads PEP 503/691 metadata from configured HTTPS sources, resolves
current-host wheel alternatives, and includes source-provided hashes in the
plan. After consumer confirmation, the daemon downloads only the selected exact
wheels using daemon-owned credentials. It stages each download, enforces size
and time bounds, verifies filename/tags/size/digest/provenance, and atomically
publishes it under the digest. A source that cannot provide a trusted digest is
not an eligible alternative.

### Air-gapped source

The transport vehicle is a strict signed ZIP artifact bundle, not a venv or a
consumer-provided install script. It contains only:

- one canonical manifest with bundle/target/source revision and, for every
  wheel, normalized distribution, exact version, filename, size, SHA-256,
  compatible tags, and provenance;
- one detached signature/key ID accepted by host configuration; and
- wheels stored by manifest-declared names beneath a single `wheels/` prefix.

Import rejects undeclared entries, duplicate normalized names, symlinks,
absolute/parent-traversal paths, compression/size-limit violations, target
mismatches, invalid signatures, and incomplete closures. An administrator may
place the bundle in a configured read-only air-gap store or send it with the
authenticated begin/chunk/commit upload lifecycle. Upload commit imports wheels
into the same content-addressed store used by online acquisition.

### Environment construction

`toolbox/hermetic_environment.py` consumes only the verified artifact-store
paths pinned by the plan/confirmation. It installs the full exact lock with
`--no-index --no-deps`, then performs import probes. The consumer never uploads
a package, passes an index URL, or invokes pip. Downloading/caching a confirmed
wheel is acquisition; creating a new immutable environment is the confirmed
environment mutation. Neither action alters an active environment.

## Non-blocking API and progress contract

Python service methods and `engine_host_channel.py` wrappers are synchronous
functions today. `daemon/local_ipc.py::_dispatch` runs ordinary service calls in
`asyncio.to_thread`, which protects the daemon event loop but still blocks the
requesting client until that call completes. That is not an asynchronous public
API.

The final rule is:

- bounded local reads and receipt/authorization checks may return synchronously;
- any network access, artifact transfer/verification, dependency resolution,
  environment build/probe/prewarm, worker rollout/drain, recursive cleanup, or
  restart repair returns a durable `HostedOperationRef` immediately;
- duplicate/idempotent submission returns current durable status immediately;
  it never waits for the first request to finish; and
- cancellation records `cancel_requested` immediately and performs teardown in
  the operation worker. The cancel request does not wait for worker shutdown.

### Least-invasive consumer API

Use the existing generic `op-start`, `op-status`, and `op-cancel` commands as the
consumer façade instead of adding a separate `*-start`/`*-status` family for
every long hosting command. A consumer submits the existing command name and
payload once through `EngineHostChannel.start_host_operation`, then observes the
returned operation ID with `get_host_operation_status`.

Existing high-level channel methods such as `toolbox_plan_definition` and
`toolbox_apply_definition` become thin wrappers around `start_host_operation`,
so most consumers do not construct the generic envelope themselves. Their final
return type is operation status and the required result-handling change is
documented in the breaking handoff. Raw top-level dispatch of a command
classified as long must fail with `operation_start_required`; there is no second
synchronous execution path.

The server implementation must stop treating that façade as a second source of
truth. For the long commands identified below, `daemon/local_ipc.py::op-start`
must prepare/dispatch through `service/operation_repository.py` and
`service/hosted_operations.py`, and `op-status`/`op-cancel` must delegate to the
canonical hosted record. The target payload must contain a request ID and the
canonical fingerprint must make retrying `op-start` return the same operation.
The current `operations.json` snapshot path may remain only for non-toolbox
daemon operations that are independently part of the final product contract; it
is removed if no such user remains and must never mirror a hosted operation.

Human decisions are continuations, not running jobs. Planning terminates with a
result that says confirmation/approval is required. The consumer collects the
decision without an open daemon request, then starts the confirmation or apply
operation. No background worker waits for a person and no approval callback
lease is held across disconnect.

The existing workflow Python/JS and proxy streaming APIs are not reused: their
stream IDs and event buffers are tied to live execution pools and are not the
durable operation ledger. For convenient progress, add
`EngineHostChannel.watch_host_operation(operation_id, ...)`, a client-side
iterator that polls `op-status` and yields changed snapshots. Optionally add
`after_updated_at` plus a bounded `wait_timeout_ms` to `op-status` for long
polling. This is a transport optimization only; correctness and recovery remain
status-based. No SSE or new generic stream-session protocol is required.

Each long operation updates the existing strict `HostedOperationProgress`
snapshot with `phase`, stable `code`, `completed_units`, `total_units`,
`updated_at_ms`, bounded `summary`, and `cancellable`. Phase defines the units:
artifact acquisition uses verified bytes, resolution/probe/warmup use completed
items, and cleanup uses removed candidates. Status exposes the latest committed
snapshot; terminal result contains the detailed package/tool receipt. Suggested
polling is immediate, then 500 ms, backing off to 2 seconds while unchanged.

### Current async audit

| API group | Current client behavior | Required disposition |
| --- | --- | --- |
| `toolbox_get_definition`, template list/describe, references, consistency, review snapshot, hosted-operation status/result/resolve | Synchronous and client-blocking, but local/bounded | Keep synchronous and bounded. |
| `toolbox_describe` live worker inventory | Synchronous worker IPC with a ten-second default timeout | Return persisted/cached inventory synchronously; submit an explicitly requested live refresh through `op-start`. |
| `toolbox_plan_definition` | Synchronous; currently local, but the planned resolver/network metadata work would block | Submit through `op-start`; terminal canonical hosted result contains the immutable plan. |
| `toolbox_approve_definition_plan` | Synchronous bounded check/mint on the wrong authority path | Replace with synchronous bounded `toolbox_approve_confirmed_definition_plan`. |
| `toolbox_confirm_definition_plan` | Not implemented | Submit through `op-start`; acquire/verify selected wheels and return confirmation receipt. |
| `toolbox_apply_definition` | Already returns durable hosted-operation status and runs rollout on a worker thread | Retain, remove synchronous re-resolution, and add package/build progress. |
| `toolbox_template_prewarm` | Already returns durable hosted-operation status and materializes on a worker thread | Retain and add artifact-byte progress. |
| Template publish/deprecate/revoke | Synchronous catalog mutations; raw publish is superseded | Keep final bounded lifecycle mutations synchronous; remove raw publish. |
| `toolbox_gc`, `toolbox_repair`, `toolbox_reconcile` | Synchronous and client-blocking; may recover state, stop workers, traverse/delete files, or rebuild indexes | Submit mutating forms through `op-start`; keep separate read-only diagnostics bounded. |
| `toolbox_execute` | New submissions are scheduled durably, but an idempotent `attach` waits up to the request timeout | Return current status on attach; never wait in the submission API. |
| `hosted_operation_cancel` | May synchronously wait for executor teardown/respawn | Persist cancel request and return immediately; teardown reports progress asynchronously. |
| Host config apply/built-in setup, template construction, artifact-bundle commit, environment removal | Not implemented | Submit long work through `op-start`. Config validation/get and upload begin/chunk/cancel remain bounded synchronous calls. |
| Daemon startup built-in realization | Not implemented; making startup resolve inline would block control readiness | Start/recover a system-owned hosted setup operation, keep control API available, and report `toolbox_ready=false` until success. |

Therefore, the hosting/toolbox API is not uniformly asynchronous today. The
event loop is protected by threads, but plan, GC/repair/reconcile, duplicate
tool execution attach, and cancellation still block their callers. These are
corrective work, not accepted final behavior.

Extend `operation_contract.py::HostedExecutionKind` with
`toolbox_definition_plan`, `toolbox_definition_confirm`, `toolbox_setup`,
`toolbox_template_construct`, `toolbox_artifact_import`,
`toolbox_environment_remove`, and `toolbox_maintenance`; retain the existing
apply/prewarm/tool-execution kinds. Extend `HostedOperationSelector` with
`environment_digest`, `artifact_bundle_id`, and `host_scope` (whose only initial
ID is `toolbox-host`) in addition to existing toolbox/template selectors.
`service/hosted_operations.py::hosted_operation_resolve_request` must define one
canonical namespace per new kind and cancellation policy must define the last
cancellable phase. Add strict phase sets in `operation_contract.py`; do not
accept arbitrary progress phase strings.

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

## Priority and required implementation expertise

The expertise label is the minimum primary-agent level for the item, not a
license to work ahead of prerequisites. A higher tier may take any lower-tier
item. `medium` work starts only after its governing contract is frozen and must
not make new protocol, concurrency, destructive-operation, or security choices.

| Priority | Required expertise | Plan items | Why |
| --- | --- | --- | --- |
| P0: contract/concurrency decisions | average | R0-02, R3-01, R3-04, R4-01, R4-02, R4-04, R6-01, R6-02, R6-03, R6-06 | Effective-definition semantics, authority separation, immutable receipt binding, atomic publication/healing, and non-blocking idempotency have the highest wrong-implementation blast radius. |
| P0: platform/package implementation | high | R1-01..R1-03, R2-01..R2-06, R3-02, R3-03, R3-05..R3-07, R4-03, R4-05 | Requires repository-wide Python changes, resolver/artifact expertise, durable operations, daemon wiring, and native-platform diagnosis under an already decided contract. |
| P1: destructive/admin safety | average | R5-02, R6-04 | Exact environment deletion and native process/sandbox containment require adversarial reference and OS-lifecycle reasoning. |
| P1: lifecycle implementation | high | R5-01, R5-03, R5-04, R5-05, R6-05 | Dependency contraction, immutable template lifecycle, maintenance operations, and native failure testing are substantial but governed by the P0 design. |
| P2: production acceptance | high | R7-02 | The no-double suite must diagnose cross-boundary failures rather than merely assemble fixtures. |
| P0/P2: precise handoff and audit | medium | R0-03, R7-01, R7-03, R7-04 | Once replacement contracts are frozen, drafting exact migration payloads, running prescribed suites, recording evidence, and removing enumerated obsolete docs/tests are bounded tasks. A high-or-higher reviewer verifies the handoff before release. |

Execution order is P0 contract decisions, P0 platform/package implementation,
P1 lifecycle/safety, then P2 acceptance. R0-03 is prepared immediately before
each client-visible P0/P1 slice, not deferred to final closeout.

### Slice and commit discipline

Every implementation commit is one declared plan slice. The following rules are
mandatory during execution:

1. Start from a clean worktree and record the slice ID(s), required expertise
   (`average`, `medium`, or `high`), production boundary, and exact tests in
   `hosting_status.md` before changing production code.
2. One slice may contain multiple plan items only when they are tightly coupled,
   adjacent in dependency order, and have the same required expertise label.
3. A change from one identified expertise label to another always ends the
   current slice. Finish validation, update checkboxes/status, commit, and verify
   a clean worktree before the next expertise level begins. This applies even if
   the same person or agent performs both slices.
4. The slice commit contains its production code, focused tests, normative docs,
   breaking-change handoff updates, `hosting_status.md` evidence, and `[x]`
   updates for every plan item completed by that commit. Checkbox updates are
   not deferred to a later bulk documentation commit.
5. Mark an item `[x]` only after its named production boundary and required tests
   pass. If an item cannot reasonably fit in one reviewable commit, first split
   it in this plan into ordered, independently verifiable sub-checkboxes; check
   only the subitems completed by each slice and check the parent when all pass.
6. An incomplete or failing slice leaves its boxes unchecked. Diagnostic work
   may be committed only as an explicitly declared diagnostic slice with its own
   scope/evidence; it must not be mixed with the next implementation level.
7. Use a concise commit subject containing the principal work ID, for example
   `hosting: detect current target (R1-01)`. After the commit, report its hash and
   confirm the worktree is clean; a commit cannot record its own hash in its
   contents.

The already adopted consumer baseline does not authorize parent-side consumer
edits. Only newly implemented breaks create new handoff work, and the medium
handoff slice must be committed before switching to the implementation slice's
expertise level.

### Code guidance for high work

| Items | Start at these production seams | Required proof |
| --- | --- | --- |
| R1-01..R1-03 | Replace `_SUPPORTED_TARGETS` and every `os.name` x64 fallback in `toolbox/host_project_config.py`, `toolbox/dependency_policy.py`, `toolbox/catalog.py`, `toolbox/hermetic_environment.py`, `toolbox/orchestration.py`, and `service/toolbox_runtime.py` with one detector returning ABI plus ordered `packaging.tags.sys_tags()`. | Extend host-config, catalog, hermetic-builder, rollout, and hash-vector tests; native jobs must assert the detected machine and import a native wheel. |
| R2-01..R2-06 | Change `ToolboxHostProjectConfiguration.from_dict`; pass the result from `daemon/local_ipc.py::EngineHostDaemon.__init__` to `service/host_service.py::EngineHostService`; replace `initialize_configured_toolbox_templates`, `shipped_templates.py`, and shipped resource locks with intent resolution; converge acquisition in `service/toolbox_materialization.py` and `toolbox/hermetic_environment.py`. | `tests/test_hosting_toolbox_host_config.py` and `tests/test_hosted_toolbox_shipped_templates.py` must use the normal daemon and real materializer; add online/air-gap source fixtures that verify the same artifact digest and a missing-wheel not-ready result. |
| R3-02, R3-03, R3-07 | Add strict kinds/phases in `operation_contract.py`; make `daemon/local_ipc.py` `op-start`/`op-status`/`op-cancel` delegate to `AtomicJsonHostedOperationRepository`; update namespace resolution in `service/hosted_operations.py`; dispatch planning/confirmation in `service/toolbox_runtime.py`; expose start/status/watch helpers in `engine_host_channel.py` and CLI. Never write a parallel record to `operations.json`. | Extend definition transport/service/public-guarantee tests with lost-response retry, daemon restart, long-poll timeout, changed-snapshot iteration, no open request during human decision, and one canonical receipt per request ID. |
| R3-05, R3-06 | Build alternatives from `toolbox/dependency_analysis.py` import evidence and `toolbox/catalog.py` mapping rules; persist the exact source/config/artifact pins in `service/toolbox_plans.py`; sanitize projections in `service/toolbox_runtime.py`. | Extend the definition matrix with multiple tools sharing direct/transitive dependencies, three-choice truncation, declined packages, source redaction, and stable ordering through the authenticated daemon channel. |
| R4-03, R4-05 | Carry the confirmed resolved input through `toolbox/orchestration.py::spawn_resolved_assignments` into `service/toolbox_catalog.py::materialize_toolbox_environment_for_bundle` and `service/toolbox_materialization.py`; feed the exact wheel closure to `HermeticToolboxEnvironmentBuilder`. | Replace hand-built resolved inputs in hermetic-builder tests with plan/confirm output; prove every pre-publication failure leaves active state, routes, registrations, and references unchanged. |
| R5-01, R5-03..R5-05 | Use `definition_planner.py::classify_toolbox_profiles`, `service/toolbox_rollout.py`, `service/toolbox_env.py`, and catalog lifecycle methods. Route GC/repair/reconcile through the generic operation façade and preserve reference checks in the builder index. | Extend maintenance, catalog-control, and atomic-routing tests with shared-lock contraction, inactive construction, lifecycle replacement, cancellation, restart recovery, and no synchronous recursive deletion. |
| R6-05 | Exercise `service/toolbox_rollout.py::recover`, `_retire_toolbox_registration`, `service/engines.py`, and the OS launcher/worker ownership boundary. | Extend resolved-rollout, atomic-routing, and sandbox tests with native abrupt death, two healers, and checkpoints immediately before/after publication. |
| R7-02 | Create a real-daemon suite beside the current definition transport tests; do not inject catalog/materializer doubles or manufacture `ResolvedToolboxEnvironmentInput`. | One suite must traverse config load, source acquisition, plan/confirm/approve/apply, execution, removal, restart healing, maintenance, and terminal result recovery. |

### Code guidance for medium work

| Items | Bounded instructions |
| --- | --- |
| R0-03, R7-01 | Diff command names and payloads in `daemon/local_ipc.py::_call_service`, `engine_host_channel.py`, `engine_host_cli.py`, `service/auth.py`, and `service/policy.py`. Put exact removed/replacement JSON, retry/status/watch logic, and obsolete client branches/tests in `HOSTING_CLIENT_BREAKING_CHANGES.md`; never edit the dependent repository. |
| R7-03 | Run the focused files named in the high-work table, `tests/test_hosted_toolbox_contract_docs.py`, all native CI commands supplied by the high implementation slices, then the full parent suite. Record command, count, duration, and commit in `hosting_status.md`; do not reinterpret a failing contract. |
| R7-04 | Use `rg` for every removed command, contract version, old target literal, shipped realized lock, raw publish payload, `wait_for_terminal` attach, and toolbox use of `operations.json`. Reconcile `HOSTED_TOOLBOX_CONTRACT.md`, `HOSTING_ACCESS.md`, `sandbox/TOOLBOX_WORKER.md`, CLI docs, plan/status, and the breaking handoff with the surviving symbols. |

## Itemized corrective work

Every slice follows the discipline above, passes `git diff --check`, and is
committed separately. A checkbox is completed only after its production
boundary—not a double—passes.

### R0 - Corrective contract baseline

- [x] **R0-01** Replace the obsolete ledger with this code-referenced plan and
  compact current-state ledger; record that no runtime behavior changed.
- [ ] **R0-02** Update `HOSTED_TOOLBOX_CONTRACT.md` and
  `sandbox/TOOLBOX_WORKER.md` as each replacement slice becomes real. Remove
  superseded normative text rather than retain compatibility notes.
- [x] **R0-03** Before the first client-visible implementation break, replace
  the reset marker in `HOSTING_CLIENT_BREAKING_CHANGES.md` with the complete
  dependent handoff described above. Do not edit dependent repositories.

### R1 - Canonical current-host target

- [x] **R1-01** Add one detector module using the running interpreter and
  `packaging.tags.sys_tags()`. Replace target defaults in
  `host_project_config.py`, `dependency_policy.py`, `catalog.py`,
  `hermetic_environment.py`, `orchestration.py`, and `toolbox_runtime.py`.
- [x] **R1-02** Update strict target/lock/catalog/cache models for the five
  target families listed above and reject cross-target wheels before download or
  build.
- [ ] **R1-03** Add native CI jobs and production-boundary tests. A target is not
  advertised until its sandbox, worker ownership, restart, and cleanup tests
  pass natively.
  - [ ] **R1-03a** Add native Windows x64/ARM64, Linux glibc x64/ARM64, and
    macOS ARM64 jobs that assert the canonical detected machine and import a
    native CPython 3.12 wheel.
  - [ ] **R1-03b** Run the sandbox, worker-ownership, restart, and cleanup
    boundaries on every native job after R6-04 supplies POSIX parent-death
    containment; advertise a family only after its job passes.

### R2 - Revisioned hosting configuration and built-ins

- [x] **R2-01** Replace the current shipped-catalog-only schema in
  `toolbox/host_project_config.py` with the strict built-in/source/mode/retention
  schema above; remove old schema parsing and standard-config fixtures.
  - [x] **R2-01a** Land strict built-in intent, source, resolution, and retention
    models with deterministic config/source-set revisions; switch
    `EngineHostService` to the new schema and delete every old field parser.
  - [x] **R2-01b** Persist atomic configuration revisions, invalidate unused
    plans/receipts on revision changes, and leave active environments pinned.
- [x] **R2-02** Wire configuration, sources, policy, and target detection through
  `daemon/local_ipc.py::EngineHostDaemon` into `EngineHostService`; refuse
  toolbox readiness on invalid or incomplete setup.
  - [x] **R2-02a** Add strict daemon construction inputs for configuration,
    logical source bindings, dependency policy, and detected target; prove the
    normal daemon supplies them to `EngineHostService`.
  - [x] **R2-02b** Publish bounded not-ready diagnostics for missing, invalid, or
    incomplete setup without partially publishing built-ins.
- [ ] **R2-03** Replace realized shipped lock resources in
  `toolbox/shipped_templates.py` and `resources/toolbox_templates/` with built-in
  intent. Resolve exact transitive wheel closures for the current host.
  - [x] **R2-03a** Replace the shipped catalog and lock JSON resources with
    strict release-owned built-in intents and remove lock-JSON artifact bridges.
  - [ ] **R2-03b** Resolve each intent to one exact current-host transitive wheel
    closure from the configured source mode with stable missing-wheel results.
    - [x] **R2-03b1** Resolve bounded read-only air-gap wheelhouses into exact
      deterministic current-host closures with stable missing-wheel results.
    - [ ] **R2-03b2** Resolve HTTPS-backed modes through the verified artifact
      acquisition boundary delivered by R2-05a, using the same closure model.
- [ ] **R2-04** Materialize, probe, publish, and optionally prewarm built-ins via
  the real `toolbox_catalog.py` and `hermetic_environment.py` boundary. Remove
  the lock-JSON-as-wheel bridge and tests that normalize it.
  - [ ] **R2-04a** Carry resolved built-in wheel closures through catalog
    publication into the real hermetic builder and import probes.
  - [ ] **R2-04b** Make required/optional prewarm and readiness use only real
    materialization receipts through normal daemon startup.
- [ ] **R2-05** Implement revisioned source changes and both air-gap ingestion
  paths. Prove missing built-in wheels prevent readiness without partial catalog
  publication.
  - [ ] **R2-05a** Implement revisioned HTTPS index/artifact acquisition into the
    verified content-addressed artifact store with bounds and redaction.
  - [ ] **R2-05b** Implement configured read-only signed air-gap ZIP ingestion
    with strict manifest, signature, path, tag, digest, and closure checks.
  - [ ] **R2-05c** Implement authenticated begin/chunk/commit/cancel upload into
    staged storage with bounded, expiring, idempotent commit.
- [ ] **R2-06** Run/recover built-in realization as a system-owned hosted
  operation with resolution, acquisition-byte, verification, build, probe, and
  prewarm progress while the control plane remains available and toolbox
  readiness remains false.
  - [ ] **R2-06a** Add the system-owned hosted execution kind and phase/progress
    contract for built-in realization while readiness remains false.
  - [ ] **R2-06b** Recover or terminally reconcile interrupted realization after
    restart and expose its canonical status without a parallel operation record.

### R3 - Multi-tool planning and consumer confirmation

- [ ] **R3-01** Extend `bundle_models.py`, `definition_planner.py`, and
  `toolbox_plans.py` with exact complete resolutions, package diffs, bounded
  alternatives, source/config pins, and affected-tool dependency edges.
- [ ] **R3-02** Convert planning to a new hosted execution kind, route generic
  `op-start`/`op-status`/`op-cancel` to its canonical hosted record, and make
  duplicate request IDs return current status instead of waiting. Terminal
  result contains the immutable plan and bounded alternatives; no
  `operations.json` mirror is written.
- [ ] **R3-03** Add the durable confirmation/acquisition operation and receipt
  repository in `toolbox_runtime.py`, `daemon/local_ipc.py`,
  `engine_host_channel.py`, and `engine_host_cli.py`.
- [ ] **R3-04** Implement accepted, declined, skipped, preserved-active-update,
  explicit-removal, shared-environment, and namespace-conflict semantics exactly
  as specified above. Remove the old apply-original-definition behavior.
- [ ] **R3-05** Return sanitized source alternatives and exact direct/transitive
  additions, removals, and transitions. Reject arbitrary URLs, paths, locks, and
  install commands supplied by consumers.
- [ ] **R3-06** Prove multi-tool add/update/remove and idempotent plan/confirmation
  recovery through the real authenticated daemon channel.
- [ ] **R3-07** Add `EngineHostChannel.watch_host_operation` over changed
  `op-status` snapshots and optional bounded long polling. Prove human
  confirmation/approval occurs between terminal operations without an open
  request, callback lease, or in-memory workflow/proxy stream dependency.

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
  non-built-in environment removal hosted operation with active/candidate/plan/
  receipt/operation reference checks and progress.
- [ ] **R5-03** Add administrator construction of a named template from an exact
  base revision plus imports/package requirements using the same resolver,
  sources, builder, probes, and immutable publication path.
- [ ] **R5-04** Keep publication inactive until explicit activation; support
  prewarm, replace, deprecate, and revoke as final APIs. Remove superseded raw
  publication payloads and commands rather than preserve both designs.
- [ ] **R5-05** Convert mutating GC, repair, and reconcile to hosted operations.
  Keep reference/consistency/review APIs as bounded reads and prove duplicate
  requests and cancellation never block the caller.

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
- [ ] **R6-06** Remove `wait_for_terminal` from duplicate toolbox execution
  attach, make hosted cancellation acknowledge immediately while teardown is
  reflected through durable progress, and split `toolbox_describe` into a
  bounded persisted/cached read plus an explicit live-refresh operation through
  `op-start`.

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
- [ ] Online HTTPS acquisition and signed air-gap ZIP import verify into the same
  content-addressed wheel store; environments install only pinned cached wheels.
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
- [ ] Every potentially long plan/setup/confirm/apply/admin/maintenance call
  submitted through `op-start` returns canonical durable status promptly,
  exposes `op-status`/watch progress, recovers by request ID, and never waits on
  human input, duplicate attach, or cancellation teardown.
- [ ] Superseded compatibility code and documentation are removed, and every
  dependent action is recorded in the breaking-change handoff with independent
  adoption evidence.
