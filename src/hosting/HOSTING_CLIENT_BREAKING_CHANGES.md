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

Do not add compatibility shims for these behaviors. Code that still needs an
old field must be changed to construct dependency intent or consume the
authoritative definition/apply projection.

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
running replacement code against an existing hosting root, an operator must use
the release's exact-path archival command (command name pending) to validate and
archive the version-1 state and associated bundles, then initialize version 2.
Do not hand-edit the version number, copy rows into version 2, or expect a
dual-schema reader. Rollback requires matching code plus restoration of the
matching archived state.

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
