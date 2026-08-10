# Hosting client breaking changes

Status: migration required before corrective hosting rollout (2026-08-10)

This handoff supersedes the consumed `HOSTED-TOOLBOX-DEFINITION` migration.
The prior adoption remains recorded below, but it does not authorize adoption
of the replacements in this document.

Normative behavior is defined by:

- [Hosting Access §11.6](HOSTING_ACCESS.md#116-durable-hosted-operation-and-capability-contract)
- [Hosted Toolbox Definition Contract](HOSTED_TOOLBOX_CONTRACT.md)
- [Toolbox Worker](sandbox/TOOLBOX_WORKER.md)

The parent product is unreleased. These replacements remove the old paths in
the same implementation slices; there is no compatibility adapter, alias, or
deprecation period. Dependent repositories must be migrated by their own
maintainers and must not be edited from this repository.

## Adoption gate and pins

- Last adopted parent baseline: `83b35e20604c8f0c2fbe27467980b6a49385d918`
- Last adopted `mp13-docs` commit: `125d20f232bf5b755d18c1b23bc1e4b8929edf21`
- Corrective parent implementation pin: not yet available; adoption is
  prohibited until the implementing slice is committed and this field is
  replaced with that exact commit or release pin.
- Dependent adoption receipt: not yet supplied. It must identify the dependent
  commit and its migration-test command/results.

Rollout order is fixed:

1. Commit this handoff before the first client-visible parent implementation.
2. Land the parent target/configuration and operation replacements, then record
   their exact parent pin here.
3. Dependent maintainers update their code, configuration, tests, and docs
   against that pin and supply an adoption receipt.
4. Keep this file populated until every listed dependent has confirmed
   adoption. Only then may the handoff be reset.

## Removed target and host-configuration contract

The following configuration fields and behaviors are removed:

- the `toolbox_environment_catalog` setup/readiness projection name; use
  `toolbox_host_project` for sanitized effective configuration and
  `toolbox_readiness` for built-in readiness;
- `resource` and the shipped realized-catalog resource contract;
- `hosting.toolbox.shipped_templates`,
  `initialize_shipped_toolbox_templates`, and automatic runtime fallback to
  packaged realized locks;
- `required_target`, including the literals `cp312-win_amd64` and
  `cp312-manylinux_2_28_x86_64` as administrator-selected build targets;
- `required_template_ids` tied to shipped realized lock resources;
- `artifact_source_ids` and `offline_preseed_source_id` as source references
  without strict source definitions and a source-set revision;
- the assumption that a missing/non-Windows target is Linux x64;
- treating shipped lock JSON as if it were an installable wheel artifact; and
- cross-target environment construction.

The daemon now detects exactly one current-host identity from the running
CPython interpreter and ordered `packaging.tags.sys_tags()`. Internal target
identity contains the Python ABI, operating system, architecture, platform
baseline, and compatible wheel tags. Supported families are CPython 3.12 on
Windows x64/ARM64, Linux glibc x64/ARM64, and macOS ARM64. Administrators do not
select a different build target. A wheel incompatible with the detected target
is rejected before download or construction.

Remove configurations shaped like:

```json
{
  "resource": "hosting.resources.toolbox_templates.catalog.v1.json",
  "trusted_signing_key_ids": ["parent-release-toolbox-v1"],
  "required_template_ids": ["builtin-data", "builtin-web"],
  "required_target": "cp312-win_amd64",
  "prewarm_required": true,
  "artifact_source_ids": ["parent-release-resources"],
  "offline_preseed_source_id": null,
  "cache_grace_seconds": 604800,
  "build_timeout_seconds": 1800
}
```

Replace them with the strict revisioned host-owned configuration model. Its
exact top-level/nested field shape is:

```json
{
  "builtins": [
    {
      "template_id": "builtin-data",
      "imports": ["numpy"],
      "package_requirements": ["numpy"],
      "sandbox_policy": "compute-only",
      "required": true,
      "prewarm": true,
      "provenance": "parent-release"
    }
  ],
  "sources": [
    {
      "source_id": "approved-index",
      "kind": "https_index",
      "origin": "https://packages.example.invalid/simple/",
      "credential_ref": "host-secret:approved-index",
      "allowed_package_namespaces": ["*"],
      "priority": 100,
      "trust_key_ids": ["packages-2026"],
      "maximum_download_bytes": 536870912
    }
  ],
  "resolution": {
    "mode": "online",
    "timeout_seconds": 60,
    "maximum_bytes": 536870912,
    "maximum_artifacts": 256,
    "allowed_redirect_origins": ["https://packages.example.invalid"],
    "wheel_only": true
  },
  "retention": {
    "artifact_cache_grace_seconds": 604800,
    "maximum_cache_bytes": 10737418240,
    "maximum_cache_artifacts": 4096,
    "protected_digests": [],
    "remove_unreferenced_custom_revisions_on_apply": false
  }
}
```

The implementing schema is strict: dependents must use the exact field names
published by the implementing parent pin and must not copy the example as an
independent schema. Source credentials and filesystem paths remain daemon-owned.
Air-gapped packages arrive only from a configured read-only store or the
authenticated signed-bundle administrator upload lifecycle.

The accepted bundle is a canonical Ed25519-signed ZIP containing only
`manifest.json`, `signature.json`, and declared wheels under `wheels/`.
Directories, links, traversal, undeclared entries, source distributions,
incompatible tags, digest/metadata mismatches, and incomplete dependency
closures are rejected before the content-addressed store index changes.
The daemon trust-key binding is an exact map from every configured key ID to
its unpadded base64url raw 32-byte Ed25519 public key. Raw wheel files in a
read-only source root are ignored; only verified direct-child signed ZIPs feed
normal resolution.
Built-in candidate provenance must be covered by exactly one verified source
bundle. The host binds its signed manifest to configured intent and the exact
resolved closure, then builds/probes through verified CAS paths before any
catalog or public receipt mutation.
The complete configured built-in set becomes visible together: all candidate
receipts are committed in one receipt-store replacement, followed by one
catalog activation replacement. An ordinary publication failure rolls back
new receipts/references; identical retry is idempotent.

Administrator setup logic must change as follows:

- replace `EngineHostService(toolbox_environment_catalog=...,
  toolbox_sandbox_policies=...)` construction with normal
  `EngineHostDaemon(toolbox_host_project_configuration=...,
  toolbox_artifact_sources=..., toolbox_trust_public_keys=...,
  toolbox_dependency_policy=...)` construction;
- stop generating `required_target`; verify the daemon-reported detected target;
- define built-in intent and ordered sources instead of realized locks;
- treat configuration application as revision creation, not in-place mutation;
- wait for the system-owned setup operation while toolbox readiness is false;
- handle a missing compatible exact wheel as a stable not-ready result; and
- never upload a venv, source distribution, install script, arbitrary index URL,
  or consumer filesystem path.

Built-in realization is no longer a synchronous administrator action. The
canonical hosted operation has execution kind `toolbox_setup`, selector
`host_scope: toolbox-host`, receipt namespace `toolbox_setup:toolbox-host`, and
system owner `system:toolbox-setup`. Its fingerprint binds the configuration
revision, source-set revision, and detected target. Start returns immediately;
same-request retries attach to the existing operation, and recovery uses the
generic hosted-operation status/result/request-recovery surface.

Clients must render only the fixed setup phases `resolution`, `acquisition`,
`artifact_verification`, `environment_build`, `import_probe`, `prewarm`, and
`publication`. Acquisition progress counts verified artifact bytes. The setup
operation is not cancellable, and clients must not synthesize a cancellation or
a second actor-owned operation. Continue reporting toolbox readiness as false
until terminal success reports `toolbox_setup_ready` after complete atomic
receipt/catalog publication. Terminal failures carry stable bounded codes and
must not be interpreted by parsing their summaries.

Normal daemon construction now dispatches or attaches to this operation and
does not wait for source scanning, resolution, build, or probes. Hosting setup
responses expose its sanitized canonical status as `toolbox_setup_operation`.
After restart, an interrupted-before-dispatch record resumes once on the same
operation ID. An interrupted-after-dispatch record is reconciled as success
only from a durable complete-publication checkpoint plus current real receipts;
otherwise that same record fails with
`toolbox_setup_interrupted_after_dispatch`. Clients must not create a retry
record or infer success from an active catalog entry alone.

Configuration revision transitions invalidate unconsumed definition plans and
materialization receipts for non-active template revisions. Consumers must
re-plan after a change. Active catalog revisions and already published toolbox
environments remain pinned; absence of configuration after restart does not
fall back to the persisted prior revision.

An absent, partial, invalid, or incorrectly bound configuration no longer
silently uses shipped defaults and does not take down the general control
plane. Branch on `toolbox_readiness.status == "unavailable"` and the stable
codes `toolbox_configuration_missing`, `toolbox_configuration_incomplete`,
`toolbox_configuration_invalid`, or `toolbox_source_binding_invalid`; do not
parse the bounded summary.

For read-only air-gap stores, every wheel must be a direct child of the bound
source root. Built-in resolution is offline and wheel-only, verifies the exact
transitive closure against the detected target and configured bounds, and
publishes nothing if any required intent reports
`required_template_wheel_missing`, `required_template_resolution_invalid`, or
`required_template_resolution_bounds_exceeded`.

## Removed toolbox mutation commands and fields

The following current surface is removed:

- raw top-level long calls to `toolbox-plan-definition` and
  `toolbox-apply-definition`;
- `toolbox-approve-definition-plan` on the ordinary toolbox-consumer route;
- `toolbox_apply_definition(definition=...)` and the wire-level apply
  `definition` field;
- apply-time re-resolution of the submitted definition;
- approval bound only to `plan_id` and a custom-delta digest;
- synchronous client waiting during plan/apply, duplicate attachment,
  cancellation teardown, or a human decision; and
- the daemon's separate 200-snapshot `operations.json` operation mirror for
  these commands.

The replacement semantic sequence is:

1. `toolbox-get-definition` — bounded synchronous read.
2. `toolbox-plan-definition` — durable operation submitted through `op-start`.
3. `toolbox-confirm-definition-plan` — durable confirmation/acquisition
   operation submitted through `op-start`.
4. `toolbox-approve-confirmed-definition-plan` — bounded synchronous operation,
   callable only by a dependency approver when policy requires it.
5. `toolbox-apply-definition` — durable operation submitted through `op-start`.

High-level channel helpers remain the preferred entry points, but plan,
confirmation, and apply helpers return durable operation status rather than a
terminal plan/result. Raw dispatch of a command classified as long fails; it
must be wrapped by `op-start`.

## Exact replacement request sequence

First read the active definition:

```json
{
  "toolbox_id": "toolbox-demo",
  "operator_details": false
}
```

Submit planning once with a stable request ID in the command payload:

```json
{
  "command": "toolbox-plan-definition",
  "payload": {
    "request_id": "plan-2026-08-10-001",
    "definition": {
      "contract": "hosting.toolbox.definition",
      "toolbox_id": "toolbox-demo",
      "expected_revision": null,
      "auto_requests": [],
      "manual_requests": []
    },
    "operator_details": false,
    "ttl_ms": 900000
  }
}
```

The immediate response is a durable operation snapshot containing an opaque
`operation_id`. Poll `op-status`, or use the channel watch helper, until the
operation is terminal. Read the immutable plan and bounded alternatives from
the terminal result; do not keep the planning request open.

Confirm one offered alternative for every offered environment and accept or
decline its package group:

```json
{
  "command": "toolbox-confirm-definition-plan",
  "payload": {
    "request_id": "confirm-2026-08-10-001",
    "plan_id": "opaque-plan-id",
    "environment_choices": [
      {
        "environment_id": "offered-environment-id",
        "alternative_id": "offered-alternative-id",
        "accept_package_changes": true
      }
    ]
  }
}
```

Observe that operation independently. Its terminal result supplies an opaque
`confirmation_ref`, accepted tool keys, skipped tool keys and stable reasons,
explicit removals, exact package mutations, and whether privileged approval is
required. A client cannot provide a version, URL, source, lock, digest, path, or
install command in this request.

When required, a separately authenticated dependency approver submits:

```json
{
  "confirmation_ref": "opaque-confirmation-ref"
}
```

to `toolbox-approve-confirmed-definition-plan`. The returned opaque
`dependency_approval_ref` binds the confirmation, exact locks/artifacts, and all
configuration/source/policy revisions.

Finally submit apply without another definition copy:

```json
{
  "command": "toolbox-apply-definition",
  "payload": {
    "request_id": "apply-2026-08-10-001",
    "plan_id": "opaque-plan-id",
    "confirmation_ref": "opaque-confirmation-ref",
    "dependency_approval_ref": "opaque-approval-ref"
  }
}
```

Omit `dependency_approval_ref` only when the confirmation receipt states that
approval is not required. Apply publishes exactly the confirmed effective
definition and never reinterprets the original request.

## Retry, watch, confirmation, and recovery logic

Dependents must make these control-flow changes:

- generate a stable printable `request_id` before every plan, confirmation, and
  apply submission and persist it before sending;
- after a lost response, resubmit the identical command/payload or resolve the
  canonical durable operation by request ID; never create a replacement ID;
- treat duplicate submission as a status lookup, not permission to wait for or
  launch a second worker;
- poll/watch changed `op-status` snapshots and fetch the terminal result from
  the canonical hosted-operation repository;
- perform package review and human approval only between terminal operations;
- on daemon restart, recover by request ID/operation ID rather than an in-memory
  callback, workflow stream, proxy stream, or open request;
- treat cancellation as `cancel_requested` acknowledgement and continue
  observing durable teardown progress; and
- replan after any target, active definition, catalog, host-config, source-set,
  dependency-policy, artifact, or expiry pin becomes stale.

Confirmation branching must preserve the specified semantics: declining a
required package skips affected new tools; a skipped update preserves its
active version; explicit removals proceed; and an incomplete shared environment
skips all affected accepted tools with `shared_environment_incomplete`.

## Stable behavior and error changes

Remove client branches that assume planning returns a terminal plan directly,
that approval is consumer-callable, or that apply accepts the original
definition. Add handling for the implementing contract's stable codes covering:

- operation submission required for raw long commands;
- stale target/config/source/policy/artifact/expiry pins;
- missing or incompatible exact wheels;
- invalid/unoffered confirmation choices;
- `shared_environment_incomplete` and namespace conflicts;
- dependency-approver authorization failure;
- cancellation requested versus terminal cancellation; and
- toolbox setup not ready while built-ins are being realized or cannot be
  realized.

Exact spellings not already frozen above must be copied from the implementing
parent contract and pin before dependent adoption; clients must not infer codes
from exception text.

## Dependent code, configuration, tests, and documentation

Remove:

- target-selection configuration and x64/Linux fallback logic;
- shipped catalog/lock-resource fixtures and lock-JSON-as-wheel normalization;
- direct synchronous plan/apply invocations and terminal return assumptions;
- calls to `toolbox-approve-definition-plan` as a consumer;
- apply payload construction containing `definition`;
- in-memory wait/callback logic spanning human confirmation or approval;
- arbitrary package URL/path/archive/install-command inputs; and
- tests and documentation for all of those paths.

Add or change:

- detected-target reporting and all five supported target-family fixtures;
- strict built-in/source/mode/retention configuration owned by administrators;
- online and air-gap missing-wheel/not-ready handling;
- durable plan/confirm/apply submission, retry, status/watch, restart recovery,
  and immediate cancellation acknowledgement;
- a distinct dependency-approver credential/role path;
- confirmation UI/logic for alternatives, exact direct/transitive mutations,
  accept/decline choices, skips, preserved active updates, and removals;
- apply construction from `plan_id` plus receipts only; and
- migration tests pinned to the exact parent implementation commit.

Required dependent evidence must cover at least: detected native target, strict
configuration rejection, lost-response retry, daemon restart, no request held
during a human decision, partial decline/skip, separate approver authority,
apply without a definition copy, and stale-pin recovery.
