# Toolbox Worker Architecture

Scope: host-managed toolbox planning, environment materialization, worker
execution, routing, recovery, and resource collection.

The normative public models, limits, methods, error codes, authorization rules,
and client algorithm are defined in the
[Hosted Toolbox Definition Contract](../HOSTED_TOOLBOX_CONTRACT.md). This
document describes the internal implementation and does not redefine that
contract. Shared sandbox launch and broker behavior is described in
[Sandbox Architecture](SANDBOX_ARCHITECTURE.md).

## Runtime model

A logical toolbox is identified by `toolbox_id` and has one authoritative
active definition revision. One complete definition contains every desired
automatic tool, manual tool, and intrinsic selection. A logical toolbox may be
served by several isolated worker processes because tools with different
resolved environments or sandbox policies belong to different resolved
profiles.

The durable version-2 toolbox state is authoritative for:

- the canonical active definition and revision;
- resolved profile and bundle identities;
- the complete `tool_routes` map;
- active, candidate, and retired executor records; and
- bounded rollout and resource-reference data.

Live process discovery is diagnostic input only. Describe, gate, execute, and
cancel resolve a toolbox-scoped tool through the durable active route map.
Routing always includes `toolbox_id`, so equal advertised tool names in
different toolboxes remain independent.

## Implementation map

The primary implementation is split by responsibility:

- [definition_planner.py](../toolbox/definition_planner.py) validates complete
  definitions, resolves dependency and sandbox intent, groups tools into
  profiles, and computes the deterministic profile diff.
- [dependency_analysis.py](../toolbox/dependency_analysis.py),
  [dependency_policy.py](../toolbox/dependency_policy.py), and
  [template_resolver.py](../toolbox/template_resolver.py) analyze staged source,
  map imports to distributions, select a template or custom delta, and enforce
  host package policy.
- [catalog.py](../toolbox/catalog.py),
  [host_project_config.py](../toolbox/host_project_config.py), and
  [service/toolbox_catalog.py](../service/toolbox_catalog.py) provide built-in
  intent, immutable resolved-template catalog, and administrative lifecycle
  controls. No realized lock is shipped as a package resource.
- [hermetic_environment.py](../toolbox/hermetic_environment.py) creates and
  verifies digest-addressed Python environments.
- [bundle_models.py](../toolbox/bundle_models.py),
  [staging.py](../toolbox/staging.py), and [manifest.py](../toolbox/manifest.py)
  build and validate immutable staged worker bundles.
- [service/toolbox_plans.py](../service/toolbox_plans.py) persists immutable,
  expiring plans; [service/toolbox_approvals.py](../service/toolbox_approvals.py)
  persists exact actor- and plan-bound custom dependency approvals.
- [service/toolbox_rollout.py](../service/toolbox_rollout.py) prepares
  candidates, publishes routes, drains replaced workers, and recovers applies.
- [service/toolbox_state_v2.py](../service/toolbox_state_v2.py) provides strict,
  digest-bound, process-safe compare-and-swap state transactions.
- [service/toolbox_runtime.py](../service/toolbox_runtime.py) exposes definition,
  execution, routing, and maintenance service operations.
- [service/hosted_operations.py](../service/hosted_operations.py) and
  [service/operation_repository.py](../service/operation_repository.py) own
  durable operation dispatch, progress, request recovery, results, and
  cancellation.
- [toolbox_executor_ipc.py](../toolbox_executor_ipc.py) is the isolated worker
  process and RPC server; [execution.py](../toolbox/execution.py) is the hosted
  caller-side execution harness.
- [service/toolbox_env.py](../service/toolbox_env.py) reports references and
  performs consistency, repair, reconcile, and garbage collection against
  version-2 ownership.

## Definition planning

Planning is a pure control-plane operation. Strict frozen definition models
reject unknown fields, invalid dependency intent, duplicate stable keys,
duplicate advertised names within one toolbox, conflicting normalized bundle
paths, unresolved imports, and policy violations before any build, staging, or
worker start.

For each request, the host combines source evidence with explicit declared
imports and distribution requirements. It then selects the smallest allowed
complete template or an exact custom delta. Sandbox capability is resolved
independently of package availability; an installed package never authorizes
filesystem, network, brokered I/O, artifact, host API, or subprocess access.

Requests are grouped only after dependency and sandbox resolution. A
`ResolvedToolboxProfileSpec` binds the verified environment key, complete lock
identity, canonical sandbox policy, assigned stable tool keys, and import-probe
obligations. Profile identity is derived from environment identity plus sandbox
policy. The planner classifies profiles as reused, added, replaced, or removed
without mutating state or processes.

The plan repository binds each immutable plan to the authenticated owner,
toolbox, definition hash, expected active revision, catalog revision, package
policy revision, resolved profiles, bundle manifests, and expiry. Custom
dependency approval stores only a parent-minted opaque approval reference bound
to the same identities and actor authority.

## Template and custom package environments

The initial catalog provides independent `core` and `py-compute` templates.
Each selected template revision is pinned by its signed manifest, complete lock,
runtime artifact, Python ABI, platform, and isolation policy. A custom delta is
resolved into a new complete base-plus-delta lock; it never layers onto or
imports from another virtual environment.

Environment construction accepts only host-derived
`ResolvedToolboxEnvironmentInput`. The digest-addressed key covers the runtime
artifact and target, complete template lock, optional custom lock, and isolation
policy. Tool names and raw per-function import subsets do not affect the key,
so compatible profiles reuse one verified physical environment.

On a cache miss, the target host creates a candidate venv with
`system_site_packages=False`, installs the exact approved wheel set with
`--no-index --no-deps`, verifies every locked distribution, and probes every
resolved import root with user site and `PYTHONPATH` disabled. Publication is an
atomic rename and requires an exact verification receipt. Failed or partial
candidates are quarantined and never become selectable. Prewarm and lazy
materialization use this same path.

Toolbox workers launch only with the verified environment interpreter selected
by the resolved profile. The host interpreter and another venv are never
dependency fallbacks.

## Bundle and worker startup

Each resolved profile produces one immutable `ToolboxBundleSpec`. Its manifest
contains the toolbox and bundle identities, resolved profile projection,
dependency lock hash, exact staged files and hashes, automatic and manual tool
entries, intrinsic selections, visibility, guide, callback, concurrency, and
`non_restartable` metadata. The executor reads only the staged manifest and
bundle contents; it does not discover tools from ambient parent process state.

The host writes a `ToolboxWorkerStartupSpec` and passes its path through
`MP13_TOOLBOX_WORKER_SPEC_PATH`. The spec binds worker and sandbox identity,
toolbox revision, manifest and scratch paths, optional engine/control state
paths, the verified venv path, IPC family/address, authentication token variable,
execution and callback contracts, and effective policy. The worker validates
the spec, loads the manifest, and serves authenticated local IPC.

Supported worker RPC methods are:

- `rpc.describe`, `describe`, and `capabilities` for protocol and bounded
  inventory metadata;
- `toolbox.describe` for the staged bundle and tool inventory;
- `toolbox.execute` for one staged `ToolCall`; and
- `host.call` for an authorized host callback.

The executor rejects a tool name absent from its own immutable inventory even
when the host should have rejected it during route resolution.

## Candidate rollout and active routing

Applying a definition creates no worker for an exactly reused profile. For each
added or replaced profile, rollout acquires the exact verified environment,
stages the bundle, starts a candidate executor, waits for RPC readiness, and
checks all of the following before publication:

- exact tool inventory rather than subset membership;
- planned bundle manifest and resolved profile identity;
- exact environment identity and verification receipt; and
- enforceable sandbox policy.

Candidates are explicitly non-routable. When every candidate is ready, one
process-safe state transaction writes the canonical definition revision,
resolved profiles, complete route map, executor states, and resource references.
Readers therefore observe either the complete previous revision or the complete
new revision, never a category-by-category mixture.

After publication, new executors become active and replaced or removed
executors become retired. Retired executors accept no new routed calls. Busy
ones remain alive until in-flight calls finish; idle ones are stopped and their
bundle ownership is released. An empty definition follows the same transaction
and publishes empty profiles and routes.

Any preparation or warmup failure removes candidates and leaves the prior
active snapshot unchanged. Candidate teardown is idempotent so retry and restart
cannot expose a partially prepared revision.

## Durable apply and recovery

Definition apply returns a durable hosted-operation status immediately. The
operation fingerprint binds the toolbox, complete definition, expected
revision, plan, exact approval identity when required, and catalog/policy pins.
The stable request ID makes an identical retry resolve to the same operation;
a different fingerprint is rejected.

Progress advances through validation, environment build, staging, warmup,
publication, draining, and cleanup. Each checkpoint is persisted before its
side effects are considered complete. On daemon restart, recovery revalidates
the pinned inputs and resumes the phase idempotently.

The version-2 snapshot is the routing source during reconciliation.
Unpublished candidates are removed. Executors named by published routes are
made active; non-routed executors are retired and stopped after they become
idle. An interrupted apply is recovered as success only when publication was
persisted and its pinned definition revision is authoritative. Otherwise it
terminates before publication after candidate cleanup, preserving the prior
revision.

State parsing, digest validation, and compare-and-swap are fail-closed. A
malformed, truncated, unknown-field, wrong-version, or digest-mismatched state
cannot become an empty toolbox or trigger worker repair.

## Execution, gates, callbacks, and cancellation

The host resolves `toolbox_id + tool_name` through the active route, obtains
the routed executor, and applies native toolbox visibility/scope semantics plus
the sandbox boundary. `ToolsScope` and `ToolsView` may narrow arguments and
visibility; they cannot widen the static sandbox policy.

Tool code may receive context helpers for host calls, brokered filesystem and
HTTP operations, and named callbacks. All brokered calls traverse the local Host
Capability broker and the daemon-owned service broker. The persisted worker
sandbox policy authorizes the actual I/O. Optional per-execution approval may
narrow a brokered call but never widens sandbox capability.

The hosted callback relay binds callbacks to one execution and attaches bounded
toolbox, tool, call, signature, and callable-surface correlation data. Callback
processing is concurrent so one blocked callback does not serialize unrelated
callback responses.

Tool execution cancellation targets the routed executor and normalizes worker
loss into a canceled call result. Resubmission policy considers persisted
`non_restartable` metadata. Definition-apply cancellation is separate: it is
allowed only before the persisted publication boundary. From publication
through cleanup the operation is non-cancellable and continues to its durable
terminal result.

## Projections and authorization

Read, plan, approval, apply, execute, maintenance, template administration, and
operator-detail permissions are independently authorized against the
authenticated actor and authority. Payload data cannot assert a role.

Normal toolbox-ID describe, gate, execution, plan, progress, and terminal
responses expose stable user states, codes, summaries, and bounded safe
diagnostics. Engine IDs, resolved profile IDs, environment keys, pools, host or
package paths, raw locks, installer output, request internals, and physical
placement stay in separately authorized bounded operator projections. Direct
engine-ID diagnostics are an internal/operator surface.

## Maintenance and garbage collection

Consistency and review compare durable routes, executor inventory, bundle
manifests, environment receipts, and operation checkpoints without changing
active truth. Repair and reconcile are toolbox-scoped, serialized with apply,
and rebuild only from the authoritative definition and resolved state.

Reference reporting distinguishes candidate, active, retired, bounded retained
revision, operation, and prewarm ownership. Garbage collection never removes a
referenced bundle, environment, artifact, or busy retired executor. Unreferenced
verified environments become eligible only after the configured grace period;
failed candidates follow their quarantine retention. Collection is
deterministic and process-safe.

## Operational limits

One worker process serves one resolved profile; there is no replica pool or
percentage rollout. Parallel calls are supported by the hosted execution layer,
subject to per-tool and sandbox constraints. Sandbox policy is immutable for a
worker lifetime, and per-request scope or approval can only narrow it. Template
administration and physical materialization remain host-operator concerns;
toolbox consumers submit definitions and observe bounded readiness and durable
operation results.
