# Hosting client breaking changes

This file is the transient adoption handoff for dependent projects. The durable
supported behavior is specified by the
[Hosted Toolbox Definition Contract](HOSTED_TOOLBOX_CONTRACT.md). Delete this
file only after every dependent project listed below has adopted the
replacement.

## Change set: `HOSTED-TOOLBOX-DEFINITION`

Status: inventory and public contract frozen; release commit pending

Parent inventory baseline: `5823d87ab6095c8864ec8ed5bedd251f9772cfc8`

Inventoried dependent baseline:
`O:/repos/mp13-docs@b3493502050e4cb55a49d9f3c87d0805d4eb0b4a`

Release commit: pending

Dependent adoption commit: pending

### Required dependent-project logic change

Dependent projects must stop treating toolbox deployment as a sequence of
independent mutations and environment-management steps. One deployment must
instead submit the complete desired state of one toolbox:

1. Build one definition containing every enabled auto tool, manual tool, and
   intrinsic, including source files, metadata, dependency intent, and sandbox
   policy. Omitting a tool means removing it; an empty definition is teardown.
2. Read the host's authoritative active definition and revision with
   `get_definition()`. Persist the returned active revision as the next
   definition's `expected_revision`.
3. Plan the complete definition. Render the plan's stable user projection and,
   only when required, request the parent-minted approval reference for the
   exact planned custom dependency delta.
4. Start one durable apply with a stable request ID, the plan ID, and the
   parent-minted approval reference. Persist the returned hosted-operation ref;
   deployment is not complete merely because the start call returned.
5. Observe and recover the apply through the generic hosted-operation status,
   result, and request-recovery APIs. Treat publication as the commit boundary:
   pre-publication cancellation may succeed, while post-publication
   cancellation must not be presented as rollback.
6. On an expected-revision conflict, discard the stale plan, read the
   authoritative definition again, reconcile the user's complete desired set,
   and re-plan. Do not retry a procedural add/remove sequence.
7. Project only stable user codes, summaries, progress, and remediation
   categories to normal users. Do not persist or expose engine IDs, resolved
   profile IDs, environment keys, package paths, installer output, or raw locks
   outside an authorized operator projection.

Persisted apply progress contains a stable phase/code, bounded counts and
summary, update time, and `cancellable`. Disable cancellation in dependent UI as
soon as `cancellable` becomes false; it never becomes true again. Reconnect and
request recovery must render the recovered progress checkpoint, then fetch the
terminal result for final diagnostics.

The dependent store may retain its own authoring and UI metadata, but parent
deployment truth must be represented by the active parent revision and durable
apply operation ref. Local `state_revision` or `toolbox_definition_digest`
values are not substitutes for the parent active revision.

If a dependent needs a predictive definition digest for caching or diagnostics,
it must use the canonical identity implementation and published vectors linked
from the durable contract. The definition digest excludes
`expected_revision`; request/file/name sets use the documented canonical order.
The predictive digest never replaces `get_definition()` as the source of the
active revision used for compare-and-swap.

### Old-to-new dependent code

Delete procedural deployment shaped like this:

```python
# Remove this entire pattern. It can partially publish and lets the client
# select a host interpreter/environment.
hosted = HostedToolBoxRef(
    toolbox_id=toolbox_id,
    host=channel,
    python_executable=python_executable,
    worker_profile_class="generic",
)
pending = hosted.mutate()
for request in auto_requests:
    pending.register_auto_callable(**request)
for request in manual_requests:
    pending.register_manual_tool(**request)
pending.register_intrinsic_tools(
    intrinsic_tool_names=intrinsic_names,
    environment_name="base",
)
pending.resolve_sandbox()
```

Build and apply one complete desired definition instead. This example shows the
required control flow; dependent authoring adapters supply the full request
fields frozen by the durable contract:

```python
request_id = deployment_store.get_or_create_request_id(toolbox_id, edit_revision)
active = hosted.get_definition()
definition = {
    "contract": "hosting.toolbox.definition",
    "toolbox_id": toolbox_id,
    "expected_revision": active["active_revision"],
    "auto_requests": build_all_enabled_auto_requests(),
    "manual_requests": build_all_enabled_manual_requests(),
    "intrinsics": build_all_enabled_intrinsics(),
}

plan = hosted.plan_definition(definition)
if not plan["can_apply"] and not plan["approval_required"]:
    raise UserVisibleDeploymentError(plan["user_projection"], plan["diagnostics"])

approval_ref = None
if plan["approval_required"]:
    approval = hosted.approve_definition_plan(plan_id=plan["plan_id"])
    approval_ref = approval["dependency_approval_ref"]

started = hosted.apply_definition(
    definition=definition,
    plan_id=plan["plan_id"],
    request_id=request_id,
    dependency_approval_ref=approval_ref,
)
deployment_store.save_operation_ref(toolbox_id, request_id, started["operation"])
```

If the apply response is lost after dispatch, recover the same operation; do
not create a new request ID:

```python
status = channel.hosted_operation_resolve_request(
    execution_kind="toolbox_definition_apply",
    selector={
        "kind": "toolbox_id",
        "id": toolbox_id,
    },
    request_id=request_id,
)
operation_ref = status["operation"]
while status["lifecycle"] in {"queued", "running", "interrupted"}:
    render_progress(status.get("progress"))
    status = channel.hosted_operation_status(ref=operation_ref)
result = channel.hosted_operation_result(ref=operation_ref)
```

Use the same flow for teardown with empty request arrays. Never enumerate the
previous tool keys:

```python
active = hosted.get_definition()
empty_definition = {
    "contract": "hosting.toolbox.definition",
    "toolbox_id": toolbox_id,
    "expected_revision": active["active_revision"],
    "auto_requests": [],
    "manual_requests": [],
    "intrinsics": [],
}
plan = hosted.plan_definition(empty_definition)
started = hosted.apply_definition(
    definition=empty_definition,
    plan_id=plan["plan_id"],
    request_id=retirement_request_id,
)
```

On `revision_conflict`, re-read `get_definition()`, rebuild the complete desired
definition against the returned `active_revision`, obtain a new plan, and use a
new request ID for that changed fingerprint. On `plan_stale` or `plan_expired`,
obtain a new plan. On `dependency_approval_invalid`, obtain a new plan and a new
parent-minted approval reference if the new plan still requires it. Never edit
a plan, approval value, definition hash, resolved template, or operation ref.

### Deprecated behavior to remove from dependents

Remove, rather than wrap or emulate, all of the following behavior:

- separate auto, manual, and intrinsic registration/unregistration calls;
- per-tool add/remove deployment loops and rollback loops that unregister a
  partially deployed category;
- `mutate()` / `PendingHostedToolboxRef` batching followed by
  `resolve_sandbox()`;
- teardown by enumerating and unregistering auto/manual/intrinsic keys;
- consumer-selected `python_executable`, `worker_profile_class`,
  `profile_id`, or mutable environment name as deployment inputs;
- procedural environment description/list/upsert/clone/resolve/apply/realize/
  sync/prepare/lock/install/verify/receipt flows;
- Boolean or mapping-based dependency approval evidence, including
  `allow_resolution`, `allow_execution`, and client-fabricated approval state;
- readiness inference from lock/execution/receipt response fragments;
- retry logic that replays individual mutations after a partial failure;
- UI or persistence logic that treats environment templates or resolved
  profiles as user-saved tool categories or runtime choices;
- fallback behavior that assumes ambient host packages or a supplied bootstrap
  interpreter can satisfy undeclared dependencies; and
- dependent reads of the parent's version-1 `toolbox_sandboxes.json` or of
  candidate/live engine registrations to infer active routing.

Dependent code must also remove any package download, lock resolution,
installation, prebuilt-venv upload, interpreter-path selection, or local-host
path exchange performed on the dependent machine. Clients submit source and
dependency intent, then consume plan/build diagnostics. Template publication,
lifecycle, artifact sources, offline preseeding, and prewarm belong to the
authenticated hosting-administration channel.

Dependent projects must not add `toolbox-template-prewarm` to their deployment
sequence or call the physical builder as a substitute for definition apply.
That command is `hosting_template_admin` behavior and returns a durable
operation for host provisioning. Consumer logic must only read the bounded
template descriptor and react to `user_projection.state/code`: show
`setup_needed`, preserve the desired definition, and retry plan/apply after an
administrator restores readiness. Remove any client-side "prewarm", "install
now", environment-repair, or direct operation-polling branch that was used to
make a template ready.

Do not add compatibility shims for these behaviors. Code that still needs an
old field must be changed to construct dependency intent or consume the
authoritative definition/apply projection.

Remove all dependent environment-description and interpreter-selection code,
including persisted `environment_name` / `base_env_name` values, inheritance
walks, environment-description hashes, per-function import-list cache keys,
local venv paths, `python_executable`, and bootstrap/fallback interpreter
branches. Do not translate those values into new fields. The replacement
logic submits only source plus dependency intent to definition planning, keeps
the returned plan/operation references, and waits for the authoritative
readiness projection before applying or executing.

In particular, dependent code shaped like this must be deleted:

```python
env = resolve_environment_description(saved_environment_name)
venv_key = hash((env, tuple(function_imports)))
python = env_python if receipt_ok else sys.executable
register_tool(..., environment_name=env["name"], python_executable=python)
```

The replacement has no environment or interpreter choice:

```python
plan = toolbox_definition_plan(definition_with_dependency_intent)
operation = toolbox_definition_apply(plan_ref=plan.plan_ref, request_id=request_id)
active = wait_for_terminal_and_read_active_projection(operation)
```

Do not persist or inspect the resolved environment key to recreate the removed
cache. It is host-derived diagnostic identity, not a dependent dispatch key.
Different functions' raw import subsets must not cause dependent-side venvs or
deployment groups; the host groups by the complete resolved lock and sandbox
policy. Workflow-specific fallback logic may remain only in workflow code and
must never be reused for a toolbox executor.

Remove any dependent code that downloads wheels, invokes pip/uv/Poetry/conda,
uploads a prebuilt venv, sends a lockfile path, checks `pyvenv.cfg`, scans
`site-packages`, probes imports, or interprets a build/quarantine receipt as
permission to launch. Those are target-host responsibilities. The client-side
replacement is only to display the bounded plan/apply or setup diagnostic and
retry the authorized definition operation after the host reports readiness.
Do not add a local repair button or fall back to the application's Python when
the code is `template_artifact_lock_incomplete`,
`environment_artifact_verification_failed`, `environment_lock_receipt_failed`,
or `environment_import_probe_failed`.

Dependent teardown must likewise remove venv deletion and cache-pruning logic.
It closes/releases the toolbox through the definition API; host reference
tracking and grace-period GC decide when a physical environment can be
deleted. A failed definition mutation must not delete a shared environment or
its artifact bytes.

Remove construction of version-1 `sandbox_profile` objects from auto/manual
requests. Do not copy `environment_name`, `profile_id`, or `required_imports`
into a compatibility object. Each version-2 request instead has exactly one
`dependency` object and one `sandbox_policy`; all remaining fields must be
present with null/empty values where the contract says so. Unknown fields are
errors, not forward-compatible bags.

Also remove dependent grouping by import list, environment label, request
category, or locally computed profile ID. Submit the complete ungrouped
definition. The host resolves each request first and returns profiles grouped
by authoritative environment identity plus sandbox policy. Dependent UI may
display the bounded profile summary but must not recreate groups, persist the
resolved profile ID as desired state, or split a definition into separate
auto/manual/intrinsic mutations.

Before submission, dependent code should validate advertised-name uniqueness
only inside that toolbox across auto/manual/intrinsic/guide names. Delete any
application-global uniqueness check: identical names in different toolbox IDs
are valid. The parent remains authoritative and rejects duplicates before
package work or staging.

Remove client-side diff engines that label profiles reused/added/replaced/
removed from saved environment or worker data. Consume the classifications in
the authoritative plan; they compare parent manifest, environment, and policy
identities. Do not use a classification as permission to skip apply or mutate
one profile independently.

Do not persist a local editable copy of a definition plan or extend its expiry.
Keep only the returned opaque `plan_id` for the immediate approval/apply flow.
On `plan_expired`, `plan_stale`, catalog change, package-policy change, or
expected-revision change, discard that reference and request a new plan from
the complete desired definition. Remove branches that patch pins or resolved
profiles inside an old plan.

### Required environment-selection and readiness changes

Dependent deployment code may request only the logical template IDs `core` and
`py-compute` exposed by the initial catalog, or express reviewed package intent
for planning. It must not append version suffixes, persist a resolved revision
as a user choice, or synthesize a template from a local environment name.

Do not copy the parent's shipped `core`/`py-compute` distribution lists into a
dependent lockfile, UI model, or compatibility table. They are parent release
resources and may change under the same stable logical IDs by publishing a new
immutable revision. Dependent persistence stores desired dependency intent and
the authoritative definition/operation refs needed for recovery; it does not
store Pydantic, NumPy, SymPy, NumExpr, or transitive versions as deployment
logic. Remove any version-comparison branch that selects a template locally.

Use `core` for source limited to the standard library, staged local modules,
and the parent worker closure. Use `py-compute` only when source requires the
shipped NumPy/SymPy/NumExpr compute set. For any other import, including
Matplotlib in the inventoried starter tool, submit the import/distribution
intent and let the plan select another allowed template or produce the exact
custom delta. Do not silently promote every tool to `py-compute`, infer package
permission from importability, or treat package availability as filesystem,
network, subprocess, broker, or host-API permission.

Dependent startup and deployment UI must consume the template readiness state
and stable diagnostic code returned by the host. It must not run an import
probe through a client-selected Python, inspect a virtual-environment path, or
infer readiness from a successful lock/install subprocess. The host prewarms
both initial templates and gates standard readiness. The dependent may present
the bounded summary and retry an authorized plan/apply after readiness is
restored; it may not bypass the gate with ambient packages.

Workflow code may independently select `core` for
`workflow_python(profile=node)` modules/snippets and Python workflow helpers,
but must retain the workflow protocol, allowlist, sandbox policy, pool,
authorization, limits, and lifecycle. Do not route workflow execution through
a toolbox worker merely because both resolved to the same immutable template.

Remove dependent cache keys or dispatch shortcuts that equate a template ID or
environment digest with a worker kind. A node, snippet, helper, and toolbox may
report the same verified template/environment while retaining different
consumer binding IDs and runtime contracts. Dependent code chooses the existing
workflow/toolbox API first and submits dependency intent within that API. The
internal physical binding is not a dependent persistence or dispatch input;
clients must not dispatch by template ID, environment digest, interpreter
path, or another consumer's binding.

The model runtime is not a toolbox template or consumer-selectable Python
runtime. Remove dependent fields, UI choices, fallbacks, and dispatch branches
that expose it as an environment name, template/base, interpreter, or
arbitrary-code route. Only model-operation status UI may display its bounded
readiness/capability projection; never persist or render its activation path,
environment key, interpreter path, package path, or raw lock.

Dependent model-status logic must branch only as follows: `state == "ready"`
may enable already-authorized model operations; `degraded` or `unavailable`
shows the bounded `code`/`summary` and keeps model operations gated. It must
never copy `python_abi`, platform, artifact/lock digests, optional package set,
or materialization revision into toolbox/workflow dependency intent. Remove
any branch that turns model readiness into a generic Python readiness signal,
template candidate, custom-environment base, interpreter override, or fallback
for a missing `core`/`py-compute` receipt. The `model-runtime-status` response
is read-only; polling it must not trigger discovery, activation, installation,
or repair.

### Current parent removal inventory

The following public `HostedToolBoxRef` methods and aliases disappear:

- `mutate()`;
- `register_auto_callable()` / `add_auto_callable()`;
- `register_python_callable()` / `add_python_callable()`;
- `register_manual_tool()` / `add_manual_tool()`;
- `unregister_auto_callable()` / `remove_auto_callable()`;
- `unregister_manual_tool()` / `remove_manual_tool()`;
- `register_intrinsic_tools()` / `add_intrinsic_tools()`;
- `unregister_intrinsic_tools()` / `remove_intrinsic_tools()`;
- `environment_descriptions()` / `list_environment_descriptions()`;
- `upsert_environment_description()` and
  `clone_environment_description()`;
- `resolve_environment_requirements()`,
  `apply_environment_description()`, `realize_environment()`, and
  `sync_environment_description()`; and
- `prepare_environment_install()`, `lock_environment_install()`,
  `resolve_environment_install_lock()`,
  `verify_environment_install_lock()`,
  `execute_environment_install()`, and
  `verify_environment_install_receipt()`.

Remove the `PendingHostedToolboxRef` type and its `register_*`, `add_*`, and
`resolve_sandbox()` methods.

The matching `EngineHostControlChannel` and service methods disappear:

- `toolbox_register_auto`, `toolbox_unregister_auto`,
  `toolbox_register_manual`, `toolbox_unregister_manual`,
  `toolbox_register_intrinsics`, and `toolbox_unregister_intrinsics`;
- `toolbox_environment_description_list/get/effective_get/upsert/clone`;
- `toolbox_environment_resolve_requirements`,
  `toolbox_environment_apply`, `toolbox_environment_realize`, and
  `toolbox_environment_sync_description`; and
- `toolbox_environment_prepare_install`,
  `toolbox_environment_lock_install`,
  `toolbox_environment_resolve_install_lock`,
  `toolbox_environment_verify_install_lock`,
  `toolbox_environment_execute_install`, and
  `toolbox_environment_verify_install_receipt`.

The daemon, subprocess CLI, authorization, and policy entries removed with
those methods are:

```text
toolbox-register-auto
toolbox-unregister-auto
toolbox-register-manual
toolbox-unregister-manual
toolbox-register-intrinsics
toolbox-unregister-intrinsics
toolbox-environment-list
toolbox-environment-upsert
toolbox-environment-clone
toolbox-environment-resolve
toolbox-environment-apply
toolbox-environment-realize
toolbox-environment-sync
toolbox-environment-prepare-install
toolbox-environment-lock-install
toolbox-environment-resolve-install-lock
toolbox-environment-verify-install-lock
toolbox-environment-execute-install
toolbox-environment-verify-install-receipt
```

The replacement retains toolbox execute, describe, gate, cancel, consistency,
review, repair, reconcile, references, and GC behavior, but their internal
routing/state logic changes to active definition revisions and explicit routes.

### Current payload fields that disappear

Remove the following mutation payload shapes and fields:

- top-level mutation fields `python_executable` and
  `worker_profile_class`;
- auto/manual `sandbox_profile.profile_id`,
  `sandbox_profile.environment_name`, and
  `sandbox_profile.required_imports`;
- intrinsic `sandbox_profile.environment_name` and all consumer-selected
  intrinsic profile identity;
- unregister payloads `tool_keys` and `intrinsic_tool_names` as mutation
  operations (names still occur inside a complete desired definition);
- environment-description fields `name`, `base_env_name`, `extra_packages`,
  and `allow_online_install` when used to mutate a shared description;
- environment action selectors `environment_name`, `tool_keys`,
  `toolbox_ids`, `source_environment_name`, and
  `target_environment_name`; and
- installation authority flags `apply`, `realize`, `allow_resolution`, and
  `allow_execution`.

Preserve source files and existing tool metadata such as module/callable name,
tool definition, activation/visibility, guide content/description,
`non_restartable`, callback signature, concurrency metadata, and sandbox
policy, but move them into the complete version-2 definition models.
Dependencies become a per-request dependency object with `mode`, optional
`template_id`, `declared_imports`, and `package_requirements`. Import roots and
installable distribution requirements are different concepts.

### Parent version-1 persisted state to archive, not translate

The parent currently stores
`<hosting_root>/state/toolbox_sandboxes.json` with root fields `version: 1`,
`updated_at`, `toolboxes`, and global mutable `environment_descriptions`.
Version-1 toolbox rows contain `toolbox_id`, `requests`, `manual_requests`,
`intrinsics`, derived `profiles`, and `runtime`. Profile rows can contain
`sandbox_profile`, category-specific request lists, `engine_id`,
`bundle_revision`, `environment`, `rollout`, and `rollout_history`.
Serialized request profiles contain `profile_id`, `environment_name`,
`required_imports`, and `sandbox_policy`.

No version-1 field is active-revision truth under the replacement. Before
running replacement code against an existing hosting root, stop the daemon and
run the release command locally as the hosting-root owner:

```powershell
@'{"hosting_root":"O:\\exact\\hosting-root","expected_state_sha256":"sha256:<64-lowercase-hex>","acknowledge_version_1_archive":true}'@ |
  python -m hosting.engine_host_cli --payload-stdin toolbox-state-archive-v1
```

`hosting_root` must be an absolute, resolved, non-symlink directory selected by
the operator. The command accepts no remote target and derives the only input
state path as `<hosting_root>/state/toolbox_sandboxes.json`; a state-file path,
glob, parent directory, or alternate filename is rejected. The operator first
calculates and records the exact file SHA-256 and supplies it as
`expected_state_sha256` to prevent archiving changed state.

The command takes the process-safe toolbox-state lock, confirms no daemon owns
the hosting root, rejects malformed JSON or any root whose version is not
exactly `1`, verifies every referenced bundle path resolves below the hosting
root, and refuses an existing/incomplete archive marker. It writes and fsyncs
an inventory containing source relative paths, byte sizes, SHA-256 digests, and
the parent release commit. It then atomically moves the state plus referenced
bundle directories into
`<hosting_root>/archive/toolbox-state-v1/<UTC timestamp>-<state digest>/`,
fsyncs the archive and parent directories, writes a completion receipt, and
only then initializes an empty strict version-2 state. Any validation or move
failure leaves version 2 uninitialized and reports a stable operator error.

Store the archive inventory, receipt, parent release artifact, and command
output together. Do not hand-edit the version number, copy rows into version 2,
or run a dual-schema reader. Rollback means stopping the daemon, reinstalling
the exact parent release recorded in the archive, verifying every archived
digest, removing the empty version-2 state only through the release rollback
command, and atomically restoring the matching archived state/bundles. Do not
restore an archive under different parent code or merge it with version-2
definitions.

### Inventory-to-adoption matrix

| Inventoried path | Required dependent replacement | Behavior to delete |
| --- | --- | --- |
| `HostedToolBoxRef` construction/serialization | Persist only toolbox identity plus host connection; use the four definition calls. | `python_executable`, `worker_profile_class`, and `PendingHostedToolboxRef`. |
| Auto/manual/intrinsic deployment | Put the complete enabled set in one `ToolboxDefinitionSpec`, plan, optionally approve, then durably apply. | Every per-category register/unregister call, batching loop, and partial rollback. |
| Toolbox retirement | Plan/apply an empty complete definition against the authoritative active revision. | Tool-key enumeration and category teardown calls. |
| Environment preparation | Submit dependency mode, template ID when explicit, declared imports, and distribution requirements; consume plan/readiness diagnostics. | Description/list/clone/resolve/apply/realize/sync and prepare/lock/install/verify/receipt chains. |
| Readiness/UI | Persist active revision, request ID, operation ref, and bounded projections; recover/status/result through hosted operations. | Lock-fragment readiness, local probes, physical paths, raw locks, engine/profile/environment IDs, and installer output. |
| Conflict/retry | Re-read, rebuild the complete desired definition, re-plan, and use a new request ID when the fingerprint changes. | Replaying individual calls or reusing stale plans/approvals. |
| Package/sandbox choice | Let the plan select the smallest complete template; request sandbox capability separately through parent policy. | Ambient-package fallback and treating an installed package as capability authorization. |
| Workflow Python/helper use | Resolve `core` independently while retaining workflow contracts, pools, policies, and lifecycle. | Routing workflow work through toolbox workers. |
| Model operations | Consume only bounded model-operation readiness/status. | Model runtime as a template, base, interpreter choice, or arbitrary-code route. |
| Existing parent state | Install the replacement artifact without starting its daemon, run `toolbox-state-archive-v1` before first replacement startup, and retain the verified archive for code-matched rollback. | Direct version edits, translation, dual reads/writes, and mixed-release restoration. |

### Inventoried `mp13-docs` adoption sites

At the inventoried baseline, dependent changes are required in these concrete
areas:

- `src/backend/platform/toolboxes/hosted_store.py`: replace
  `deploy_toolbox()`'s sequential `toolbox_register_auto`,
  `toolbox_register_manual`, and `toolbox_register_intrinsics` calls with one
  complete-definition plan/apply flow. Replace
  `retire_toolbox_daemon_registration()` unregister enumeration with an empty
  definition apply. Persist the parent active revision and apply operation ref.
- The same store's local record model: remove `python_executable`,
  `worker_profile_class` as parent deployment selection, and `last_environment`
  readiness truth. Convert source/manual `sandbox_profile.environment_name` and
  `required_imports` into dependency intent. Local `tools`, `manual_tools`,
  `intrinsic_tools`, `source_function_ids`, generated-tool revision history,
  visibility, policy, and authoring metadata may remain as inputs to the
  complete definition.
- `src/tools/tool_authoring.py`: remove `_run_environment_checks()` and the
  prepare/lock/verify/execute/receipt chain; validation should plan the complete
  definition and interpret user-safe plan diagnostics. Cleanup must no longer
  call `toolbox_unregister_manual`.
- `src/tools/registry.py`, `src/tools/llm_tool_parser.py`,
  `src/tools/LLM_PROMPT.md`, and `src/tools/TOOLS_DEV_GUIDE.md`: replace
  mutable `environment` plus raw `required_imports` guidance with dependency
  mode/template selection, declared import roots, and distribution
  requirements. Dynamic or optional imports require explicit declarations.
- Tests and UI contracts that fake or assert `toolbox_register_*`,
  `toolbox_unregister_*`, `toolbox_environment_*`, `environment_name`,
  `required_imports`, `python_executable`, procedural readiness, or partial
  rollback must be rewritten around complete definitions, authoritative reads,
  plan projections, durable apply progress/results, conflict recovery, and
  empty-definition teardown.

The dependent repository had unrelated uncommitted work during inventory. This
entry records code-derived adoption requirements only; it does not claim that
those working-tree changes are part of either baseline.

### Hosted import inventory that constrains the initial catalog

This inventory distinguishes Python import roots from distributions. It is the
seed input for template/catalog work; it is not permission to guess packages
from arbitrary import strings.

| Hosted source | Actual import roots | Distribution mapping / disposition |
| --- | --- | --- |
| Parent `symbolic_algebra` intrinsic module | `numpy`, `sympy`, plus standard-library `json`, `re`, `codecs`, `dataclasses`, `typing`, `importlib` and parent `mp13_engine.mp13_config` | `numpy` -> NumPy; `sympy` -> SymPy. SymPy is present in `poetry.lock` but is not a direct `pyproject.toml` dependency and must be declared/pinned by intrinsic metadata. |
| Parent `scriptable_calculator` intrinsic module | `numpy`, optional-at-import `numexpr`, plus the same module-level imports | `numpy` -> NumPy; `numexpr` -> NumExpr. The fallback still requires NumPy; the shipped compute template must include/probe both. |
| Parent hosted chat demo | `math`, `pathlib`, `base64` | Standard library; selects `core`. Filesystem/HTTP access comes from explicit sandbox and broker policy, not from packages. |
| `mp13-docs` starter source tools `TextStats`, `AddNumbers`, `EchoWithDocstring`, `SearchMarkdownFiles`, and `BuildScopedDiagnosticsBundle` | standard library / staged local `tools` support | Select `core`. `BuildScopedDiagnosticsBundle` declares `requests` but does not import or use it; brokered `ctx.http` performs HTTP. The stale declaration must not seed a template. |
| `mp13-docs` starter `RenderLineChart` | `matplotlib` plus standard-library `base64`, `io`, and `math` | `matplotlib` -> Matplotlib. It is a dependent starter tool, not a parent intrinsic; planning must select another allowed template or an approved custom delta unless the final parent catalog intentionally covers it. |
| Generated/manual dependent tools | source-dependent; parser currently records caller/LLM `required_imports` | Analyze staged source, combine it with explicit `declared_imports`, and map through the reviewed catalog. Do not make the arbitrary generated-tool set part of a built-in template lock. |

The parent lock at the inventory baseline resolves NumPy `2.4.3`, SymPy
`1.14.0`, NumExpr `2.14.1`, Matplotlib `3.10.8`, and Requests `2.32.5`.
Those observations are reproducibility inputs, not yet the immutable shipped
template manifests. The required initial visible catalog remains `core` plus
`py-compute`; `py-compute` must cover every parent intrinsic load/execute import
and `core` must cover the installed hosting/worker artifact and standard
library-only tools without ambient site packages.

### Adoption checklist

- [ ] Parent replacement release commit recorded above.
- [ ] Durable contract link resolves and matches final public models.
- [ ] Dependent builds one complete definition for create/update/removal.
- [ ] Dependent persists authoritative active revision and apply operation ref.
- [ ] Revision conflicts re-read and re-plan the complete definition.
- [ ] Custom dependency approval uses only a parent-minted approval reference.
- [ ] UI handles user-safe plan, progress, terminal, and cancellation states.
- [ ] Procedural mutation and environment-management behavior listed above is
      removed, including tests, docs, and fallback branches.
- [ ] Existing parent version-1 state is archived with the release command.
- [ ] Parent and dependent focused/full suites pass at recorded commits.
- [ ] Dependent adoption commit recorded above.
