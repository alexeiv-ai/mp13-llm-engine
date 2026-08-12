# Hosted Toolbox Definition Contract

Status: normative

Contract family: `hosting.toolbox.definition`

Generic hosted-operation identity, lifecycle, retention, request recovery,
authorization, provider-session, callback, and capability-lease behavior is
normative in [Hosting Access §11.6](HOSTING_ACCESS.md#116-durable-hosted-operation-and-capability-contract).
This document specifies only toolbox definitions, dependency environments,
planning, approval, rollout, and toolbox-specific operation behavior.

## Scope and vocabulary

A tool runtime is one configured hosting target. A toolbox is one deployed tool
namespace on that runtime and is governed by one complete
`ToolboxDefinitionSpec`. A package environment is a verified dependency
environment on the runtime. An environment template is a parent-owned immutable
dependency base. A resolved profile is an internal grouping by package
environment and sandbox policy.

Definition atomicity, expected-revision comparison, and advertised-name
uniqueness are scoped to one `toolbox_id` on one tool runtime. The same
advertised name is valid in different toolboxes. Every execution route includes
both toolbox identity and tool name, so multiple toolbox references can execute
concurrently.

Environment templates and resolved profiles are deployment details. They are
not tool categories, user-saved runtime selections, or toolbox identities.

## Common encoding rules

Public payloads are UTF-8 JSON objects. Unless a model explicitly declares an
optional field, every field is required. Unknown fields are rejected at every
typed model boundary. JSON numbers must be finite. Object keys and string
values must not contain ASCII control characters other than tab, carriage
return, and line feed where file content permits them.

Identifiers use Unicode-normalized UTF-8 text and are compared after the
model-specific canonicalization described here. Hashes and revisions use
lower-case `sha256:<64 lower-case hexadecimal digits>` encoding. Timestamps are
Unix epoch milliseconds. Lists whose order is not semantically meaningful are
deduplicated and sorted during canonicalization; source-file order is replaced
by normalized path order.

## Canonical identities

The executable identity rules are in `hosting.toolbox.identity`; published
cross-process inputs and results are in
`hosting/toolbox/HOSTED_TOOLBOX_HASH_VECTORS.json`. Typed validation runs before
identity calculation.

Canonical JSON recursively normalizes strings and object keys to Unicode NFC,
requires string object keys and finite JSON numbers, preserves JSON number
types, renders negative zero as `0.0`, and serializes UTF-8 with sorted object
keys, no ASCII escaping, and separators `,` and `:`. A domain-separated digest
is SHA-256 over the canonical JSON object `{"domain": <domain>, "value":
<canonical input>}`.

| Identity | Domain | Canonical semantic input |
| --- | --- | --- |
| Definition revision | `hosting.toolbox.definition.v2` | Complete definition without `expected_revision`; auto/manual requests sorted by stable key; files sorted by normalized path; dependency import/requirement lists and intrinsic names sorted and deduplicated. |
| Resolved profile | `hosting.toolbox.resolved_profile.v2` | Resolved environment digest plus canonical sandbox policy. Assigned tool keys do not change grouping identity. |
| Package environment | `hosting.toolbox.environment.v2` | Runtime/ABI/platform/worker identity, immutable template-lock digest, nullable custom-lock digest, and isolation policy. Per-function raw import subsets are excluded. |
| Bundle manifest | `hosting.toolbox.bundle_manifest.v2` | Complete semantic manifest without its output digest/revision; files, tools, and name sets use canonical order. |
| Template lock | `hosting.toolbox.template_lock.v1` | Complete template constraints, distributions, artifacts, import roots, and provenance with set-like records canonically sorted. |
| Custom lock | `hosting.toolbox.custom_lock.v1` | Base template-lock digest plus the complete resolved distributions, artifacts, import roots, and provenance for the derived environment. |

Published vector results are:

| Vector | Digest |
| --- | --- |
| `definition_revision` | `sha256:cce56c94b2e4e8d7fc64cba803934e198df731bb6d3af6a9df48dc4c7da73795` |
| `resolved_profile_identity` | `sha256:da8dc5f609e918edca510d73e46c14a608230b6d0d3fa7870d27ec4e7b9c4a09` |
| `environment_identity` | `sha256:dea6b64250c4d630868309df55b40172a38ea68b4c67ce1f68b715edabe6370c` |
| `bundle_manifest_digest` | `sha256:44a182a5af380504695e8baee182ed3ac3aeb4aeb3eccc5085e0249b62a962d3` |
| `template_lock_digest` | `sha256:87f500104aa9645d1d558501ab23580e012ce7bf3ab90fd7b2f8836e0701110f` |
| `custom_lock_digest` | `sha256:06bf7a8786595c56a2597f2c8b75c2c6bba0a4e41b5bb7df6e6a6fce4d1757f3` |

## Validation limits

The parent enforces these limits before dependency analysis or environment
work:

| Item | Limit |
| --- | ---: |
| UTF-8 request body | 32 MiB |
| `toolbox_id` | 1-128 characters, pattern `[A-Za-z0-9][A-Za-z0-9._-]*` |
| Auto requests per definition | 512 |
| Manual requests per definition | 512 |
| Intrinsic base names per definition | 64 |
| Files per request | 256 |
| Normalized relative path | 1-512 characters |
| One file's UTF-8 content | 4 MiB |
| All file content in one definition | 24 MiB |
| Module or callable name | 1-256 characters |
| Advertised tool name | 1-128 characters |
| Declared import roots per request | 256 |
| Package requirements per request | 256 |
| One import root or package requirement | 1-256 characters |
| Guide description | 16 KiB |
| Guide content, canonical JSON | 1 MiB per request |
| Tool definition, canonical JSON | 256 KiB per manual request |
| Sandbox policy, canonical JSON | 256 KiB per request or intrinsic set |
| Callback signature, canonical JSON | 64 KiB |
| Concurrency metadata, canonical JSON | 64 KiB |
| User diagnostics returned | 64, newest/highest priority retained |
| A diagnostic summary | 512 UTF-8 bytes |
| Affected tool keys in one diagnostic | 32 |
| Rollout history entries in a read snapshot | 32 |

Relative paths use `/`, may not be absolute, may not contain an empty segment,
`.` or `..`, and are compared case-insensitively on case-insensitive target
platforms. Two files with the same normalized target path must have identical
content or the definition is rejected.

## ToolboxDefinitionSpec

`ToolboxDefinitionSpec` is the complete desired state of one toolbox:

```json
{
  "contract": "hosting.toolbox.definition",
  "toolbox_id": "workspace-tools",
  "expected_revision": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "auto_requests": [],
  "manual_requests": [],
  "intrinsics": {
    "names": ["symbolic_algebra"],
    "include_guides": true,
    "sandbox_policy": {
      "sandbox": {
        "enabled": true
      }
    }
  }
}
```

`expected_revision` is `null` only when the toolbox has no active revision. It
must equal the authoritative active revision for every replacement, including
an empty definition. The definition revision excludes `expected_revision` from
its canonical hash so the same desired content has the same identity across a
compare-and-swap retry.

An empty definition has empty auto/manual lists and an empty intrinsic-name
list. It is a valid active revision with no advertised routes.

Auto-request stable keys are `module_name:callable_name`. Manual-request stable
keys are `manual:module_name:callable_name`. Stable keys must be unique within
their respective complete definition and must not be empty.

## ToolboxAutoAssignmentRequestV2

An auto request has exactly these fields:

```json
{
  "files": [
    {
      "relative_path": "workspace_tools/weather.py",
      "content": "def CurrentWeather(city):\n    return {\"city\": city}\n"
    }
  ],
  "module_name": "workspace_tools.weather",
  "callable_name": "CurrentWeather",
  "dependency": {
    "mode": "auto",
    "template_id": null,
    "declared_imports": [],
    "package_requirements": []
  },
  "sandbox_policy": {
    "sandbox": {
      "enabled": true
    }
  },
  "activate": true,
  "hidden": false,
  "non_restartable": false,
  "guide_content": null,
  "guide_description": null,
  "callback_signature": null,
  "concurrency": null
}
```

The advertised name is the callable name produced by the parent tool loader.
The parent verifies that it agrees with loaded metadata before publication.

## ToolboxManualAssignmentRequestV2

A manual request has exactly these fields:

```json
{
  "files": [
    {
      "relative_path": "workspace_tools/lookup.py",
      "content": "def LookupRecord(record_id):\n    return {\"record_id\": record_id}\n"
    }
  ],
  "module_name": "workspace_tools.lookup",
  "callable_name": "LookupRecord",
  "tool_definition": {
    "type": "function",
    "function": {
      "name": "LookupRecord",
      "description": "Look up one record.",
      "parameters": {
        "type": "object",
        "properties": {
          "record_id": {
            "type": "string"
          }
        },
        "required": ["record_id"]
      }
    }
  },
  "dependency": {
    "mode": "template",
    "template_id": "core",
    "declared_imports": [],
    "package_requirements": []
  },
  "sandbox_policy": {
    "sandbox": {
      "enabled": true
    }
  },
  "hidden": false,
  "non_restartable": false,
  "callback_signature": null,
  "concurrency": null
}
```

The advertised name is `tool_definition.function.name`. It need not equal the
Python callable name, but it must satisfy the advertised-name limit.

## ToolboxDependencyRequest

Each auto and manual request contains one strict dependency object with exactly
`mode`, `template_id`, `declared_imports`, and `package_requirements`.

- `auto` requires `template_id: null`. The parent analyzes staged source and
  chooses the smallest allowed template. If no template covers the complete
  requirement set, the plan contains a policy-checked custom delta.
- `template` requires a non-empty immutable logical template ID. The plan fails
  if source, explicit declarations, or selected intrinsics are not covered.
- `custom` requires a non-empty allowed base template ID and at least one
  package requirement not already supplied by the base. It always requires an
  exact parent-minted dependency approval reference before apply.

`declared_imports` contains Python import roots needed for dynamic, conditional,
or optional code paths that static analysis cannot prove. Each item matches
`[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*`.
`package_requirements` contains PEP 508 distribution requirements. The parent
never copies an import root into this list or guesses a distribution solely
from an import spelling.

Sandbox policy is independent of package availability. A package request cannot
grant filesystem, network, broker, artifact, host-API, or subprocess authority.

## Intrinsic selection

`intrinsics` contains exactly `names`, `include_guides`, and `sandbox_policy`.
Names identify base intrinsics; the parent derives guide names when
`include_guides` is true. Intrinsic dependency metadata is parent-owned and is
combined with source dependencies before template selection. Names, base tools,
and derived guides participate in advertised-name uniqueness.

## Definition validation

Validation completes before any package build, bundle staging, or worker start.
The parent rejects duplicate stable keys; duplicate advertised names across
auto, manual, intrinsic, and guide tools; conflicting normalized file paths;
invalid tool definitions; unresolved imports; incompatible explicit
requirements; and sandbox-policy denial.

Name uniqueness is per toolbox. Definitions for separate toolboxes are never
compared for this rule.

The implementation parses these objects through strict frozen
`ToolboxDefinitionSpec`, `ToolboxAutoAssignmentRequestV2`,
`ToolboxManualAssignmentRequestV2`, `ToolboxDependencyRequest`, and intrinsic
selection models. Every listed field is required even when its value is null
or empty; any additional field fails. In particular, `sandbox_profile`,
`environment_name`, `required_imports`, approval booleans, runtime overrides,
and any fields outside the strict model are not aliases for definition
dependency or sandbox inputs.

Planning analyzes and resolves each request before grouping. The internal
`ResolvedToolboxProfileSpec` contains the host-derived profile digest,
environment identity, template and effective lock digests, canonical sandbox
policy, assigned stable tool keys, and the union of verification import roots.
Its identity is only environment identity plus sandbox policy. Consequently,
two functions with different raw import subsets share one profile when they
resolve to the same complete lock and policy; matching import text cannot
merge different environments or permissions.

Duplicate advertised-name validation covers auto, manual, intrinsic, and
derived intrinsic-guide names within the one submitted toolbox before any
resolution or staging. No process-global name registry participates, so the
same advertised name remains valid in two separate toolbox definitions.
Conflicting content at one normalized file path also fails before resolution.

The pure planner emits one `ToolboxBundleSpec` per resolved profile. Its
`dependency_lock_hash` is the effective immutable template/custom lock, and
the manifest includes the strict resolved-profile projection. The internal
`SandboxProfileSpec` portion of that bundle is derived exclusively by the host
and is never accepted from the public definition.

Before persistence, proposed profiles are compared with authoritative active
snapshots using manifest hash, environment key, and canonical sandbox-policy
digest. An exact triple is `reused`. A proposed profile that retains assigned
tool ownership but changes any triple field is `replaced`; unmatched proposed
and active profiles are `added` and `removed`. The comparison is deterministic
and performs no staging, environment acquisition, registration, or routing.

`toolbox-plan-tool-changes` performs a compare-and-swap merge against that same
authoritative active revision. Its strict batch contains at most 512 unique
change IDs and targets. Add, update, rename, and remove are validated together;
all targeted active keys are removed before result-key collision checks, so
rename swaps are atomic. Update must retain its stable key, rename must change
it, and request kind cannot change. Complete-definition planning derives stable
`host:sha256:<digest>` IDs from each change kind and its prior/resulting key.
Each normalized change is then bound to its bounded source import analysis,
reviewed distribution mapping, resolved environment group, preferred exact
package mutations, and dependency-approval requirement. This analysis is part
of the immutable plan record and is revalidated on restart; public projection
does not reconstruct it from mutable source or package state.

The resulting plan is stored in the process-safe atomic definition-plan
repository. Its ID binds toolbox ID, definition revision, expected active
revision, catalog revision, package-policy revision, resolved profiles, and
bundle manifest/lock identities. Each non-removal alternative also binds one
strict generic `hosting.package_lock.v1` and `hosting.environment_request.v1`;
the lock contains the exact verified CAS artifacts and dependency closure.
Records have a strict 15-minute maximum TTL, a 4 MiB encoded maximum, and a
256-record repository maximum. Expired records are pruned and cannot be
refreshed by repeating the same plan request. A restart reloads and revalidates
the complete record; corrupt, truncated, unknown-field, over-capacity, or
pin-mismatched state fails closed.

## Environment template descriptor

Consumer template list/describe responses contain bounded
`ToolboxEnvironmentTemplateDescriptor` objects:

```json
{
  "contract": "hosting.toolbox.environment_template",
  "template_id": "core",
  "template_digest": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
  "lock_digest": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
  "catalog_revision": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
  "python_abis": ["cp312"],
  "platforms": ["win_amd64"],
  "import_roots": ["hosting"],
  "lifecycle": "active",
  "materialization": "ready",
  "user_projection": {
    "state": "ready",
    "code": "template_ready",
    "summary": "The template is ready on this tool runtime."
  }
}
```

Consumer descriptors omit resolved distribution locks, artifact locations,
environment keys, interpreter paths, and installer output. Those values are
available only through an authorized bounded operator projection.

## Deployment administration policy

Package resolution, artifact acquisition, and physical environment creation run
only in the daemon on the target tool runtime. A toolbox client cannot supply a
prebuilt environment, interpreter path, arbitrary host path, package-installer
command, or unrestricted package origin. Interactive login to the target host
is not part of the management contract.

### Roles and control methods

Roles are derived from the authenticated parent authority; payload fields cannot
assert them.

| Role | Authority |
| --- | --- |
| `toolbox_consumer` | List/describe visible templates and submit definition read/plan/apply requests. |
| `toolbox_dependency_approver` | Mint an exact actor/plan/delta-bound approval reference when package policy marks the delta reviewable. |
| `hosting_template_admin` | Construct immutable template revisions from an exact base, move lifecycle state, and start prewarm/materialization. |
| `hosting_auditor` | Read bounded operator projections and audit events; no mutation authority is implied. |

The consumer control methods are `environment-template-list` and
`environment-template-describe`. The administrative methods are
`environment-template-construct`, `environment-template-activate`,
`environment-template-replace`, `environment-template-deprecate`,
`environment-template-revoke`, and `environment-template-prewarm`. They use the same
authenticated daemon control transport as other host administration.
Construction and prewarm return durable hosted-operation status. Role checks
are distinct even when one actor holds multiple roles.

`environment-template-construct` accepts exactly a stable `request_id`, a new
logical `template_id`, an exact non-revoked `base_template_digest`, bounded
imports, and bounded package requirements. The daemon retains every exact base
distribution pin, resolves the requested closure only through active
revisioned package sources, verifies daemon-computed artifact identities, builds and probes the
complete environment, commits the exact receipt, and publishes one immutable
`inactive` revision. The `environment_template_construct` operation uses the
`template_id` selector and fixed phases `validation`, `resolution`,
`artifact_verification`, `environment_build`, `import_probe`, `receipt_commit`,
`publication`, and `cleanup`. It never accepts a lock, artifact reference,
signature, URL, path, interpreter, source choice, activation flag, or force
bypass.

`environment-template-activate` activates an exact inactive revision only when the
logical template has no different active revision. `environment-template-replace`
atomically compares `expected_active_digest`, deprecates that revision, and
activates the exact `replacement_digest`. A stale expected digest fails without
changing the catalog. Deprecation removes an active pointer; revocation is
terminal. Constructing a revision never changes active selection, and the raw
publication surface does not exist.

`environment-template-prewarm` accepts exactly the logical `template_id`, optional
exact `template_digest`, target `python_abi`, target `platform`, and a stable
caller `request_id`. It accepts no path, interpreter, installer command,
artifact bytes, credential, role, or readiness assertion. If the digest is
omitted, dispatch pins the current active revision before persisting the
operation. The returned `hosting.operation_status` contains a
`environment_template_prewarm` operation selected by `template_id`; operators use
the generic hosted-operation status/result/cancel/recovery APIs thereafter.

The target-host materializer reports bounded checkpoints but only an exact,
complete verification receipt can change the descriptor from
`not_materialized`/`setup_needed` to `ready`. The receipt binds the template
revision, ABI/platform, derived environment digest, complete artifact digests,
complete exposed-import probes, verifier identity, and verification time.
Failures remain terminal diagnostics on the operation and do not create or
replace a ready receipt. The default unconfigured builder fails closed with
`template_materializer_unconfigured`; shipped builders and normal setup are
defined by P1-11/P2.

### Immutable package locks and artifacts

A published template revision pins one `hosting.package_lock.v1` identity,
runtime/builder identity, platform constraints, and policy/configuration
revision. The daemon computes SHA-256 while receiving or acquiring bytes and
stores artifacts by that identity. A caller-provided digest is only an expected
value and cannot select different bytes.

Publisher signing is not required by the baseline package contract. A source
may configure an optional verifier, but verifier metadata never replaces the
daemon-computed artifact identity and is not required for ordinary local or
authenticated-source operation. Every lock contains exact artifact sizes,
source IDs, dependency pins, and secret-free reproducible source metadata.

Artifact origin references resolve through administrator-configured sources.
Network origins are normalized HTTPS origins on an allowlist and use the
daemon's secret store for credentials. Redirects must remain on allowed origins.
Plain HTTP, VCS working trees, local client paths, mutable unpinned URLs, shell
commands, and credential material in manifests are denied.

Online index resolution is denied by default. Host project policy may allow
specific normalized HTTPS indexes, distribution allow/deny rules, and version
constraints. A custom delta that requires online resolution is always
`approval_required`; only the exact parent-minted dependency approval reference
authorizes the planned delta. Approval does not widen index, artifact, package,
or sandbox policy.

For offline hosts, administrators preseed the same digest-addressed artifacts
through the authorized package ingress commands and create a deterministic
lock. Clients receive readiness/diagnostic projections, never physical package
or credential paths. Incomplete, oversized, reordered, expired, cancelled, or
digest-mismatched uploads never become resolvable; only atomic promotion of a
complete daemon-hashed stage creates a package receipt.

### Supported targets and timeouts

The initial target families are CPython 3.12 on `win_amd64`, `win_arm64`,
`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, and
`macosx_11_0_arm64`. Their ABI tag is `cp312`. One canonical detector derives
the current daemon's interpreter ABI, operating system, architecture, platform
baseline, and ordered tag set from `packaging.tags.sys_tags()`.
Configuration cannot select a different construction target. A template
revision is advertised only when its complete lock has compatible artifacts for
the exact detected target and the configured sandbox policy is enforceable.
Other Python ABIs, 32-bit targets, Linux musl, macOS x64, and free-threaded
Python are unsupported.

Timeout ceilings are 300 seconds per artifact fetch, 600 seconds for approved
custom lock resolution, 1,800 seconds for one environment materialization, 120
seconds for the complete import-probe set, and 3,600 seconds for one prewarm or
lazy materialization durable operation. A timeout fails/quarantines the
candidate; it never publishes a partial cache entry or falls back to another
interpreter.

### Lifecycle, audit, and retention

Template lifecycle is `active`, `deprecated`, or `revoked`. Auto selection uses
only active revisions. An explicitly selected deprecated revision may be
planned with a stable warning while policy permits it. A revoked revision is
denied for new plans, builds, and worker starts. Revocation does not silently
rewrite an active toolbox definition or destroy an environment still serving an
active route; readiness reports the affected reference so an administrator can
publish/apply a replacement. Every lifecycle or active-revision-pointer change
mints a new catalog revision and invalidates older plans.

Publishing a new immutable revision under a stable logical template ID does not
mutate old revision content. The administrator separately makes the new
revision active. Plans pin the catalog revision and template/lock digests they
resolved.

Audit records cover construct, activate/replace, deprecate, revoke, prewarm,
artifact verification, build, quarantine, approval mint/validation, and GC.
They contain event time, authenticated actor/authority, logical template and
revision digests, catalog/policy revisions, target, outcome/stable code, and
operation/request IDs where applicable. They exclude signatures' secret
material, repository credentials, approval secret values, host paths, and
installer output from normal projections.

Referenced verified environments/artifacts are retained. An unreferenced
verified entry has a seven-day grace period, configurable from one to 90 days,
before deterministic least-recently-used GC may remove it. Failed or partial
builds are quarantined, never routable, and retained for bounded diagnostics for
24 hours before GC eligibility. Cache pressure may evict only unreferenced
entries whose grace period elapsed. Revoked content remains while referenced
and becomes GC-eligible only after the last active/history retention reference
and grace period expire.

Environment removal is an administrator-only durable operation,
`toolbox_environment_remove`, selected by one canonical `environment_digest`.
The digest is the exact immutable environment key; paths, globs, logical
template IDs, and force flags are invalid. The operation validates the current
configuration revision, reports progress through `validation`,
`reference_check`, `removal`, and `cleanup`, and returns `removed`,
`already_absent`, or a stable ordered list of blocking reference kinds. Active
profiles, candidates, unexpired plans and confirmations, active operations,
built-in references, protected digests, and any live
`hosting.environment_reference.v1` prevent removal. GC is owned by the shared
environment manager, uses the revisioned retention policy, and never bypasses
those checks.

Mutating `toolbox-gc`, `toolbox-repair`, and `toolbox-reconcile` are
administrator-only `toolbox_maintenance` hosted operations. Each is submitted
through `op-start` with a stable `request_id`, uses the `host_scope` selector
`toolbox-host`, and returns canonical operation status immediately. Repair and
reconcile may also carry bounded `toolbox_ids`, `only_inconsistent`, and
`details`; no command accepts a path or force bypass. Fixed progress phases are
`validation`, `recovery`, `repair`, `gc`, and `cleanup`. Cancellation may commit
only before recovery/mutation starts and always returns immediately; once a
non-cancellable recovery or mutation checkpoint is durable, the caller keeps
watching the same operation. An identical retry after a lost response or daemon
restart attaches to and safely resumes the same idempotent operation. Read-only
reference, consistency, and review commands remain bounded synchronous calls.

## Initial environment catalog

The release-owned configuration declares exactly two stable logical built-in
intent IDs: `core` and `py-compute`. Logical IDs have no version suffix. Intent
contains imports, package requirements, a sandbox-policy reference, readiness
flags, and provenance only. It contains no resolved distribution lock, wheel
filename, artifact digest, or target-selected manifest.

The package ships no realized built-in catalog or lock JSON. Normal host setup
must resolve each intent from the configured source mode for the one detected
current-host target. Until every required intent has one complete exact wheel
closure, no entry from that configuration revision is published and toolbox
readiness remains false with a stable bounded diagnostic.

Read-only air-gap resolution invokes the bundled installer only as a bounded
offline dependency solver: dry-run report, no indexes, no installed-state
reuse, and wheels only. The host then independently verifies every reported
artifact is a direct child of a configured source root, allowed by that
source's package namespaces, compatible with the detected target, consistent
with its wheel name/version metadata, and within per-source and aggregate
artifact bounds. It hashes each artifact itself and derives the lock identity
from the exact distributions, artifact identities, and target. A timeout,
missing transitive wheel, or incompatible wheel produces
`required_template_wheel_missing`; malformed or escaped report data produces
`required_template_resolution_invalid`. No filesystem path or installer output
appears in either diagnostic.

After resolution and verification, each complete closure becomes a signed
immutable template revision. Its identity includes the logical template ID,
template manifest digest, complete lock digest, catalog revision, Python ABI,
platform tag, parent worker artifact digest, and isolation policy version.
Changing any member creates a different revision. Plans pin the selected
revision and never resolve an unqualified logical ID again during apply.

`core` contains the installed hosting/worker artifact and only its complete
protocol, serialization, validation, and sandbox-harness dependency closure.
It contains no optional mathematics, data, document, network-client, or model
packages. Standard-library modules need no distribution entry.

`py-compute` resolves to its own complete independently materialized lock. It
includes the same hosting/worker closure plus exact versions of NumPy, SymPy,
NumExpr, and every third-party distribution imported by a release-owned parent
compute intrinsic.
`py-compute` is not constructed by copying, layering, or inheriting the
`core` site-packages directory. Sharing digest-addressed artifact bytes is
allowed; sharing a mutable installation is not.

### Hermetic environment input and identity

Toolbox environment construction consumes one strict
`ResolvedToolboxEnvironmentInput` minted by the host after template and custom
dependency resolution. It contains the exact runtime version and artifact
digest, Python ABI, platform, complete immutable distribution lock and digest,
nullable custom resolved-lock digest, isolation-policy version, and the full
set of import roots that the final interpreter must probe. Unknown or missing
fields fail closed. It contains no environment name, base-environment name,
client path, interpreter override, or online-install switch.

The toolbox environment key uses the published
`hosting.toolbox.environment.v2` identity. Its runtime portion is exactly the
runtime version, runtime artifact digest, ABI, and platform; its remaining
inputs are the complete template-lock digest, nullable custom resolved-lock
digest, and isolation-policy version. Logical template labels, manifest
labels, profile IDs, function names, and each function's raw import subset do
not participate. Therefore two functions with different import subsets share
one physical environment when their complete resolved lock, runtime target,
and isolation policy are identical.

The resolved import roots are verification obligations, not cache-key input.
Every root still has to pass a probe under the final environment interpreter
before publication. Toolbox and workflow selection are separate: toolbox
workers can use only that verified environment interpreter, while workflow
helper interpreter selection is governed by its independent workflow contract.

### Physical materialization and publication

The target host maps every locked distribution to exactly one immutable wheel
in an administrator-configured artifact source. Artifact references contain a
source ID, basename-only filename, canonical SHA-256 digest, and byte length;
the builder rejects missing, extra, ambiguous, non-wheel, path-escaping, or
digest/size-mismatched artifacts. Network resolution is never an installation
fallback. Built-in prewarm and lazy acquisition use the same builder and the
same resolved input. A dependent cannot supply an artifact source, wheel,
lockfile, venv, or filesystem path.

For definition apply, confirmation persists one `ResolvedToolboxEnvironmentInput`
per effective profile. It is constructed from the selected offered alternative
and verified CAS objects, then carried unchanged through rollout orchestration
to the builder. The selected generic package lock and `EnvironmentRequest` are
loaded from the same immutable plan and used unchanged for generic environment
adoption; apply does not import package bytes or recreate the lock. Active
custom profiles persist that exact resolved input so a later plan compares
against the real active closure rather than an installed environment or a newly
resolved approximation.

Removal planning resolves each remaining profile again from only its complete
remaining tool and package requirements. Packages needed solely by removed
tools are not retained implicitly: a custom closure may contract to the exact
built-in closure. Apply reuses an active immutable environment only when the
complete profile identity is unchanged, preserving its generic environment
reference. Replaced and removed `hosting.environment_reference.v1` records
remain live through the atomic definition publication and are released only
afterward. Physical
deletion remains a separate grace-period, reference-checked operation.

Each cache miss creates a new candidate with `venv` configured as
`with_pip=True` and `system_site_packages=False`. The candidate's own Python
runs pip with `--no-index --no-deps` against the approved exact wheel set. The
same final interpreter then verifies every distribution/version in the
complete lock and imports every resolved root with user site and `PYTHONPATH`
disabled. A custom environment repeats the complete base-plus-delta lock into
a new venv; it never copies or references the base venv's `site-packages`.

Only a candidate with a byte-for-byte matching strict verification receipt,
non-inheriting `pyvenv.cfg`, complete installed lock, and successful probes is
atomically renamed to its digest-addressed published path. Failed candidates
are moved under the quarantine namespace with a bounded code and are never
returned as ready. Per-environment OS file locks plus in-process locks
deduplicate concurrent builders. The builder owns no reference index. Strict
generic receipts, locks, and references are persisted by the shared environment
manager; release changes only a generic reference, and deletion occurs only in
its reference-aware GC.

All environment builds, staging, spawn, and readiness checks occur before the
single active-definition publication. A failure before publication retires
candidate registrations and releases candidate environment references while
leaving active definition state and routes unchanged. The durable success
result reports the confirmed accepted, skipped, preserved, and removed tools
plus logical package mutations; physical engine/profile/environment identities
remain operator-only.

Catalog prewarm commits the public materialization receipt only after this
physical publication succeeds. A service configured with artifact sources
routes toolbox worker launch through `materialize_toolbox_environment_for_bundle`
and uses only the published environment Python. There is no bootstrap
interpreter branch on that path. The setup summary continues to report the
required `core` and `py-compute` target receipts; absence or mismatch is a
degraded setup state and prevents resolved acquisition.

No distribution version is hard-coded in package resources or runtime code.
Release-owned intent supplies package constraints; configured source metadata
and the detected target produce the exact immutable closure. Consumers cannot
select or override those versions.

Toolbox built-in intent is retained as toolbox-owned policy, while sources and
resolution authority come from `hosting.configuration.v3` package management:

| Key | Required value and meaning |
| --- | --- |
| `builtins` | Non-empty ordered built-in intents. Each has exactly `template_id`, `imports`, `package_requirements`, `sandbox_policy`, `required`, `prewarm`, and `provenance`. A prewarmed intent must be required. Requirements cannot contain direct URLs. |
| `sources` | Priority-ordered `hosting.package_source.v1` records with logical source ID, kind, sanitized locator, optional daemon-owned credential reference, enabled state, and priority. |
| `resolution` | Exact `mode`, `timeout_seconds`, `maximum_bytes`, `maximum_artifacts`, `allowed_redirect_origins`, and required `wheel_only: true`. Modes are `online`, `prefer_airgap`, or `air_gapped`. |
| `retention` | Exact `artifact_cache_grace_seconds`, `maximum_cache_bytes`, `maximum_cache_artifacts`, `protected_digests`, and `remove_unreferenced_custom_revisions_on_apply`. |

The daemon derives the target; configuration contains no target selector. An
`airgap_store` origin is the logical `airgap://<source_id>`, never a filesystem
path. HTTPS origins reject embedded credentials, query strings, and fragments.
`air_gapped` mode rejects HTTPS sources; online mode requires an HTTPS source;
air-gap modes require an air-gap store. Source order is descending priority.

Canonical JSON of the complete configuration produces an immutable
`config_revision`. Canonical sources plus resolution policy produce a distinct
`source_set_revision`. Normal readiness projections include those revisions,
the detected target, built-in intent, sanitized origins, and bounds, but omit
every `credential_ref` and daemon path.
An enabled source set that cannot satisfy resolution reports
`package_source_unavailable` without preventing authenticated control diagnosis.

HTTPS acquisition accepts PEP 503 HTML or PEP 691 JSON only from the configured source origin
or an exact origin in `allowed_redirect_origins`, follows at most five explicit
redirects, and never forwards credentials to an unapproved origin. A source
with `credential_ref` requires one exact daemon-owned Authorization binding;
bindings are not status, progress, receipt, or error data. The baseline trusts
only daemon-computed SHA-256 identities. Optional source verifiers may impose
additional policy without changing the generic package contract.

Only current-target wheel entries with a source-provided SHA-256 and exact byte
size are eligible; signed PEP 503 anchors carry these as the `sha256` URL
fragment and exact `data-size`. Metadata and artifact responses are streamed with configured
timeouts, redirect limits, per-source and aggregate byte bounds. The downloaded
filename, size, digest, wheel tags, distribution/version metadata, and allowed
namespace are verified before one atomic shared-CAS index replacement. The CAS
contract is `hosting.toolbox.artifact_store.v2`; its `https_manifests` evidence
is separate from signed air-gap `bundles`, while both reference the same
immutable `objects` collection. Failure changes neither evidence nor the object
index.

For each configured built-in root, online/prefer-air-gap setup discovers a
bounded transitive candidate wheelhouse by reading verified wheel metadata and
following applicable `Requires-Dist` entries. Candidate count and total bytes
never exceed resolution bounds. The exact resolver then runs offline against
verified CAS object paths only; it does not give pip an index URL, credential,
or network path. `prefer_airgap` first accepts a complete exact air-gap closure
and contacts HTTPS only when that closure is unavailable.

The host deterministically binds the selected artifact identities and source
metadata into one immutable package lock. Online and offline sources containing
identical bytes produce identical artifact identities; source and policy IDs
remain explicit in the lock.

The host records every applied configuration revision in one process-locked,
atomically replaced state file and marks exactly one revision current. Applying
the same revision is idempotent. A transition to a different revision
invalidates unconsumed definition plans and materialization receipts not tied
to an active catalog revision. It does not mutate the active catalog map,
published toolbox definition state, or generic environment references. An
otherwise unconfigured restart does not resurrect the last persisted revision;
explicit daemon configuration remains required.

Normal daemon construction loads `hosting.configuration.v3` from the top-level
MP13 configuration and passes that one immutable object to `EngineHostService`.
Package sources, credential references, dependency policy, and environment
roots come from that authority. Credential values are resolved locally and are
never launcher parameters.

Foreground, background, service, local-channel bootstrap, and relay-equivalent
startup accept only `mp13_config_file`. The configuration is validated before
the listener is bound. No package mapping, credential, policy payload, or
temporary launcher configuration is accepted.

Detached startup never places a credential, configuration payload, policy
payload, or artifact-source map in process arguments. Launcher diagnostics,
results, and logs do not project configuration contents or credential values.

Normal daemon construction starts or attaches to the canonical system setup
operation without waiting for source I/O, resolution, installation, or probes.
That worker scans only direct `*.zip` children of each bound read-only air-gap
root, imports them through the verified artifact store, and passes only rehashed
CAS object paths to resolution. Raw wheel files beside a bundle are never
eligible. Invalid bundle ingestion produces a bounded degraded
toolbox-readiness diagnostic while the general control plane remains available;
no physical source path or public-key value is projected.

Administrator bundle upload begins in a process-locked untrusted staging
repository, never in the verified artifact store. Begin binds the authenticated
owner and idempotency request ID to exactly one air-gap source ID, current
config/source-set revisions, detected target, declared archive byte size, and
SHA-256. Identical begin retries return the same `upload_id`; a changed binding
is `artifact_upload_conflict`. At most 64 uploads are retained, an archive is
bounded by both its source and resolution byte limits, and an open upload
expires after 15 minutes.

Chunks are unpadded base64url, at most 1 MiB decoded, and carry exact zero-based
index and byte offset. Only the next contiguous chunk is accepted. Retrying an
identical committed chunk is idempotent; changed content/order is rejected.
Stage-file append is fsynced before one atomic metadata replacement, and restart
continues from the last committed offset. Expiry or synchronous cancel removes
only the untrusted stage file and retains bounded terminal metadata. Status and
errors expose no stage path or chunk content. Begin/chunk/cancel alone cannot
create a CAS object, evidence record, catalog entry, or materialization receipt.

Only an authenticated administrator may call
`package-artifact-upload-begin`, `package-artifact-upload-chunk`,
`package-artifact-upload-status`, `package-artifact-upload-cancel`, or
`package-artifact-upload-commit`. Commit requires a complete stage and binds
one commit request ID to one content-addressed package receipt. Repeating the
identical commit returns the same receipt; a changed binding is
`package_upload_conflict`.

Artifact-import progress is non-cancellable and uses only `validation`,
`artifact_verification`, `publication`, and `cleanup`. The worker rehashes the
complete staged archive against its declared byte size and SHA-256, verifies
the complete staged bytes against size, digest, source, target, and dependency
policy, and only then atomically indexes the artifact in the shared CAS. Terminal success
is `artifact_upload_committed`; terminal cleanup removes the untrusted stage
file. A verification failure publishes no new CAS entry and returns a stable
bounded terminal code rather than exception or path text.

After restart, an import interrupted before dispatch is redispatched on its
existing operation ID. An import interrupted after dispatch is reconciled as
success only from the durable committed upload result; otherwise that same
operation becomes terminal failure with
`artifact_upload_interrupted_after_dispatch`. Recovery never creates a
parallel operation record.

For a release-owned built-in, candidate construction requires one exact package
lock whose artifact set covers the entire resolved closure. The immutable
template provenance retains that lock and its secret-free source metadata. The
host binds those inputs to the configured intent, exact closure, and detected
target. The `mp13-engine` wheel
digest in the closure is the parent-worker artifact digest. Absence of that
runtime artifact fails before build.

The hermetic builder receives an exact `(source_id, filename) -> verified CAS
path` map. Once this map is configured it cannot fall back to a raw source
root. Candidate preparation computes the final template digest, constructs the
same strict resolved input used by catalog prewarm, installs the exact lock, and
runs every declared import probe. This pre-publication boundary returns strict
candidate receipts but writes neither the catalog nor the public receipt store.
Any candidate failure releases all references created by that batch.

Publication accepts only a prepared batch whose config/source revisions and
target still match and whose template IDs exactly cover configured built-ins.
It validates all candidate identities before mutation, writes the complete
receipt set with one atomic receipt-store replacement, then publishes and
activates the complete template set with one atomic catalog replacement.
Ordinary catalog failure removes only receipts inserted by that attempt and
releases candidate references. Retrying the identical prepared batch is
idempotent and creates no duplicate revision.

Physical environment identity deliberately permits templates with the same
runtime artifact, complete lock, custom lock and isolation policy to share one
immutable environment. Its physical verification receipt therefore compares
only those physical fields, not logical template ID/digest. Every reuse still
reruns the requesting template's complete import-root probes before a
template-specific public receipt can be committed.

Built-in realization is represented by one system-owned durable hosted
operation. Its execution kind is `toolbox_setup`, its only selector is
`host_scope: toolbox-host`, its receipt namespace is
`toolbox_setup:toolbox-host`, and its owner is `system:toolbox-setup`. The
request fingerprint binds the complete `config_revision`,
`source_set_revision`, and detected target. Start returns the queued or running
generic `hosting.operation_status` immediately; a duplicate request ID with the
same fingerprint attaches to the same canonical operation.

Setup progress uses only `resolution`, `acquisition`,
`artifact_verification`, `environment_build`, `import_probe`, `prewarm`, and
`publication`. Acquisition units are verified artifact bytes; candidate and
publication phases use bounded item counts. The operation is never
cancellable. Its terminal success code is `toolbox_setup_ready`; resolution or
execution failure is terminal with a stable bounded diagnostic. Toolbox
readiness remains false until the complete receipt and catalog publication has
succeeded. Clients recover it through the generic hosted-operation
status/result/request-recovery APIs and never create an actor-owned parallel
record.

On restart, a queued operation that had not claimed dispatch is redispatched
once on the same operation ID. An operation interrupted after dispatch is never
blindly replayed: it is reconciled as success only when its durable
`builtin_publication_committed` checkpoint and current real materialization
receipts prove complete publication. Otherwise the same record becomes terminal
failure with `toolbox_setup_interrupted_after_dispatch`, readiness stays false,
and no replacement or parallel setup record is created. The sanitized canonical
status is projected as `toolbox_setup_operation` in the hosting setup summary.

The daemon control plane remains available when toolbox setup is absent,
partial, or invalid. `toolbox_readiness` is then `unavailable`, contains no
template entries, and uses exactly one of `toolbox_configuration_missing`,
`toolbox_configuration_incomplete`, `toolbox_configuration_invalid`, or
`toolbox_source_binding_invalid`. The normal projection contains a bounded
summary and detected target but no parser exception, credential, signed query,
origin path, or daemon path. No built-in catalog entry is published from an
invalid or incomplete setup.

The required built-in `sandbox_policy` reference is `compute-only`. Its exact
effective policy is:

```json
{
  "policy_id": "compute-only",
  "sandbox_required": true,
  "filesystem_read_roots": [],
  "filesystem_write_roots": [],
  "artifact_roots": [],
  "network": false,
  "subprocess": false,
  "brokered_io": {
    "filesystem": false,
    "http": false,
    "subprocess": false
  },
  "host_api_permissions": []
}
```

An omitted request policy means this compute-only policy. A request may narrow
it. Widening any capability requires an authorized parent sandbox-policy choice
and remains independent of dependency approval. A package being importable
never grants filesystem, network, subprocess, artifact, broker, or host API
capability.

In the startup worker, the host validates template records, complete package
locks, artifact availability, target tags, worker artifact digest, and
the ability to enforce compute-only isolation. It then materializes and import
probes both required templates before standard readiness succeeds. Readiness is
derived from the active catalog plus real materialization receipts, never from
intent or a queued/running setup record. If the platform cannot enforce the
policy, the host neither advertises nor launches the affected revision. A
required intent with `prewarm: false` is an explicit non-standard deployment;
it reports degraded readiness until that built-in has passed the same checks.

Readiness diagnostics use the stable codes `environment_template_unavailable`,
`required_template_lock_invalid`,
`required_template_artifact_unavailable`,
`required_template_materialization_failed`, `required_template_probe_failed`,
and `compute_only_policy_unenforceable`. Normal projections contain only the
template ID, target, state, stable code, bounded summary, catalog revision, and
manifest/lock digests. Authorized operator projections may include bounded
artifact and probe diagnostics but never credentials, approval values, host
paths, interpreter paths, or installer output.

Selection always chooses the smallest allowed complete template. Source using
only the standard library, staged local modules, and the parent worker closure
selects `core`. Source requiring only reviewed shipped compute distributions
selects `py-compute`. Other reviewed requirements select another active
template when one exists; otherwise they form an exact custom delta subject to
package policy and, when required, dependency approval. A caller cannot force a
larger template merely because it is installed.

Toolbox, workflow Python node, workflow Python snippet, and workflow Python
helper resolution all call the same pure dependency/template resolver and read
the same exact target materialization receipts. The resulting binding contains
the logical template/lock, verified physical environment digest, target, and
effective sandbox-policy digest. It also contains a consumer-specific binding
identity. Thus consumers may share immutable artifact bytes and one verified
physical environment while toolbox, node, snippet, and helper identities,
worker pools, protocols, lifecycle, authorization, and public methods remain
separate. Resolution is read-only and never starts or discovers a worker.

## Cross-worker use of core

The catalog resolver may select the same immutable `core` revision for:

- standard-library-only toolbox functions;
- Python workflow modules and snippets whose workflow contract selects
  `workflow_python(profile=node)`; and
- Python workflow helper workers whose source and declared dependencies fit
  `core`.

This is environment reuse, not worker or protocol unification. Each consumer
retains its own execution contract, source/import allowlist, effective sandbox
policy, worker-pool identity, resource limits, authorization checks, operation
kind, routing, readiness, cancellation rules, and lifecycle. A worker accepts
only its own request protocol. Pools do not exchange live interpreters,
processes, globals, module caches, credentials, or routes. Failure, draining,
replacement, and GC of one pool do not implicitly mutate another pool.

Catalog resolution and verified artifact caching may be shared parent services.
They expose immutable revision identity and bounded readiness to each worker
owner; they do not expose a generic Python execution endpoint.

## Model runtime boundary

The model runtime is a separate parent-owned execution domain. Its complete
lock is derived from the root `pyproject.toml`, the committed lock file, and the
administrator-configured optional model package set. Its identity pins the
Python ABI/platform, engine artifact digest, complete distribution lock digest,
optional package-set digest, model isolation-policy version, and materialization
revision.

The exact host configuration namespace is `model_runtime`:

| Key | Meaning |
| --- | --- |
| `project_resource` | Required package-resource reference to the root `pyproject.toml`. |
| `lock_resource` | Required package-resource reference to the committed complete lock. |
| `optional_package_set` | Name of one administrator-reviewed optional model package set. |
| `required_target` | Exact supported Python ABI/platform target. |
| `engine_artifact_digest` | Required digest of the model-engine artifact used by workers. |
| `readiness_required` | Boolean controlling whether model readiness gates model-operation readiness only. |

Only authenticated model operations may activate this runtime, and they remain
subject to their own model authorization, resource, network, data-access, and
secret policies. Toolbox definition planning, custom environment building,
toolbox workers, workflow Python nodes, workflow helpers, template
administration, and consumer payloads cannot select it, derive from it, or use
it as a template/base. It is not a generic interpreter or arbitrary-code route.

The bounded `ModelRuntimeStatus` projection contains exactly `state`,
`code`, `summary`, `python_abi`, `platform`, `engine_artifact_digest`,
`complete_lock_digest`, `optional_package_set`, `materialization_revision`,
and `updated_at_ms`. The normal projection never contains an environment name,
environment key, virtual-environment path, interpreter path, activation
command, package path, raw lock, credential, or installer output. An authorized
operator may receive bounded package/probe diagnostics without those values.

A preinstalled model environment is allowed only when it verifies to the same
complete identity. Its discovery and activation remain internal to the model
worker owner. Model-runtime failure affects model-operation readiness and does
not cause toolbox catalog fallback, template substitution, or ambient-package
use.

The read command is `model-runtime-status`. It is side-effect-free and returns
exactly the ten `ModelRuntimeStatus` fields above. Generic template,
dependency/custom-build, workflow-Python environment, and template-control
inputs are checked by one fail-closed selection guard. Explicit model-runtime
keys and model aliases in template, runtime, profile, environment, worker, or
interpreter selectors fail with `model_runtime_selection_denied` before build
or launch. This remains true when the exclusive model materialization is
installed, verified, and healthy. Legitimate authenticated model commands and
their `model_path` inputs do not pass through the generic guard.

## Planning

`plan_definition(definition, request_id=...)` is actor-authorized and starts a
canonical `toolbox_definition_plan` hosted operation. Planning is side-effect-
free with respect to logical toolbox state, package installation, bundle
staging, and workers. Its terminal result is the complete immutable
`hosting.toolbox.definition_plan.v2` record described below. Identical retries
return current canonical status; they never wait or create a second receipt.

Each bounded `imports` entry contains exactly `import_root`, `classification`
(`standard_library`, `local_staged`, `parent_runtime`, `known_third_party`,
`declared_dynamic`, or `unresolved`), `distribution` (nullable), and `evidence`.
Each evidence item contains exactly `relative_path`, `line`, and `kind`; at most
16 evidence items are returned per import root. Each bounded `environments`
entry contains exactly `request_keys`, `mode`, `template_id`,
`package_requirements`, `approval_required`, and `diagnostics`. Normal results
contain logical template and reviewed distribution data but no physical
materialization identity.

Plans are immutable, retained for 15 minutes by default, and may be configured
from 5 to 60 minutes. At most 64 unexpired plans are retained per actor and
toolbox; older plans are evicted first. Apply rejects an expired or evicted plan
rather than recomputing it implicitly.

## Complete immutable definition-plan record

The durable mutation protocol persists
`hosting.toolbox.definition_plan.v2`, never the former partial v1 plan. Its
identity binds the exact active and proposed definitions, active-definition
revision, detected target, catalog revision, host-configuration revision,
dependency-policy revision, source-set revision, every environment offer,
every offered exact artifact and logical package mutation, dependency edges,
owner, authority, and resolved rollout draft. Expiry is fixed by the first
successful create and an identical retry returns the same record without
refreshing it.

Each environment offer has unique affected tool keys and their exact
added/updated/unchanged/removed classification, immutable base template and
revision, one to three deterministic alternatives, the preferred alternative,
an alternatives-truncated flag, confirmation/approval requirements, and a
complete dependency edge for every affected tool. Exact artifacts carry import
roots, mapped distribution, direct/transitive/template-runtime reason, version,
wheel filename, digest, compatible wheel tags, bounded provenance, and logical
source ID. Source origins are sanitized `https` or logical `airgap` URLs with
no userinfo, query, or fragment. Duplicate tools, incomplete edges, changed
wheel identity/tags, unbounded alternatives, missing pins, and persisted state
corruption fail closed.

Offer construction accepts only preverified exact candidates. It compares each
candidate closure with the exact active environment closure, emits explicit
addition/removal/version-transition mutations with direct or transitive reason,
groups the complete affected tool set, and creates removal-only offers even
when there is no proposed environment. Candidate order is source priority,
logical source ID, then lock digest; the first is policy preferred and only the
first three are exposed, with `alternatives_truncated: true` when more were
verified. Every candidate, including one omitted by truncation, must have a
sanitized configured-source identity and internally consistent artifact source.

## Consumer confirmation reduction

Confirmation is a pure reduction over the immutable active definition,
proposed definition, complete environment offers, and exactly one choice per
offered environment. A choice contains only the offered `environment_id`, an
offered `alternative_id`, and `accept_package_changes`; versions, sources,
URLs, locks, artifacts, paths, and install commands are not accepted.

Declining an addition or transition skips every new affected tool with
`package_changes_declined`. A skipped update preserves its exact active
request. Explicit removals proceed. Dependency edges are then reduced to a
fixed point: an otherwise accepted tool depending on a skipped affected tool
is skipped with `shared_environment_incomplete`, preserving its active request
when it was an update. The effective definition is reconstructed from accepted
proposed requests plus preserved active requests and is revalidated for file
and advertised-name conflicts before any confirmation receipt or apply worker
can exist. A namespace conflict is terminal
`toolbox_confirmation_namespace_conflict`; tool ordering never resolves it.

The reduction result contains the exact effective definition and revision,
selected alternative IDs, accepted tools, skipped tools and stable reasons,
preserved active updates, explicit removals, effective logical package
mutations, and whether dependency approval remains required. Apply consumes
that pinned result and never reinterprets the original proposed definition.

## Dependency approval references

`approve_confirmed_definition_plan(confirmation_ref)` is available only to the
distinct authenticated dependency-approver role (and the administrator
superset), never the ordinary toolbox consumer. Policy denial returns a stable diagnostic and no reference.
Approval returns:

```json
{
  "contract": "hosting.toolbox.dependency_approval",
  "approval_ref": "approval_opaque_parent_value",
  "plan_id": "plan_01JABCDEF0123456789",
  "expires_at_ms": 1786233600000,
  "user_projection": {
    "state": "ready",
    "code": "custom_dependency_approved",
    "summary": "The planned additional packages were approved."
  }
}
```

The reference is opaque and minted only by the parent. Its stored record is
bound to the confirmation owner and authority, approving actor, toolbox and
plan IDs, confirmation-ref digest, effective-definition revision, the exact
selected locks/artifacts digest, the complete target/config/catalog/source/
policy pins digest, decision, mint time, expiry, and consumption identity.

Approval lifetime cannot exceed the plan lifetime or 60 minutes. Apply
atomically binds first use to one stable `request_id`. Idempotent retries by the
same actor with the same request ID and identical apply fingerprint may reuse
it. Another request ID is denied as consumed. Revocation is effective until
route publication begins. Expired, revoked, consumed, cross-actor,
wrong-authority, wrong-plan, wrong-confirmation, changed exact resolution, or
changed pinned configuration cases all return `dependency_approval_invalid` to an
unauthorized caller so another actor's approval state is not disclosed.

Approval decisions and validation attempts are audited with actor, authority,
toolbox, plan, definition/delta/policy/catalog digests, decision/code,
request ID when consumed, and event time. Secret reference values and package
credentials are never written to audit projections.

## Authoritative read

`get_definition()` is an actor-authorized, side-effect-free read. Authorization
requires toolbox read plus source-read authority; the API returns no partial
snapshot when source content is not authorized. It does not discover, start,
repair, reconcile, or otherwise contact workers.

```json
{
  "contract": "hosting.toolbox.definition_snapshot",
  "tool_runtime_id": "runtime-local",
  "toolbox_id": "workspace-tools",
  "active_revision": "sha256:8888888888888888888888888888888888888888888888888888888888888888",
  "definition": {
    "contract": "hosting.toolbox.definition",
    "toolbox_id": "workspace-tools",
    "expected_revision": null,
    "auto_requests": [],
    "manual_requests": [],
    "intrinsics": {
      "names": [],
      "include_guides": false,
      "sandbox_policy": {}
    }
  },
  "active_tools": [],
  "rollout": [],
  "diagnostics": [],
  "user_projection": {
    "state": "ready",
    "code": "toolbox_definition_active",
    "summary": "The active toolbox definition is available."
  }
}
```

The nested canonical definition always sets `expected_revision` to `null`;
clients copy top-level `active_revision` into the next proposed definition.
Source is subject to the definition size limits. Rollout and diagnostics are
bounded by the limits above. A revision-conflict response contains the current
revision and instructs the client to read again; it does not embed another
definition.

## Apply and durable operation behavior

`apply_definition(definition, plan_id, request_id,
dependency_approval_ref)` revalidates the definition hash, expected active
revision, plan expiry, catalog revision, package-policy revision, and approval
binding before dispatch. `request_id` is 1-128 printable ASCII characters and
must be stable across retries.

The call returns immediately in the generic `hosting.operation_status` shape
with an actor-owned `HostedOperationRef`. The execution kind is
`toolbox_definition_apply` and its selector contains `toolbox_id`. The
fingerprint includes toolbox ID, definition hash, expected revision, plan ID,
custom-delta digest, approval identity digest, catalog revision, and policy
revision. Reusing a request ID with a different fingerprint fails with
`request_id_conflict`.

The immediate response uses this generic shape (generic fields not applicable
to a queued start are `null` or empty according to the hosted-operation
contract):

```json
{
  "contract": "hosting.operation_status",
  "api_status": "ok",
  "operation": {
    "contract": "hosting.operation_ref",
    "operation_id": "op_01JABCDEF0123456789",
    "request_id": "deploy-workspace-tools-42",
    "execution_kind": "toolbox_definition_apply",
    "selector": {
      "kind": "toolbox_id",
      "id": "workspace-tools"
    },
    "fingerprint": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "receipt_namespace": "actor-session"
  },
  "lifecycle": "queued",
  "request_id": "deploy-workspace-tools-42",
  "created_at_ms": 1786230000000,
  "updated_at_ms": 1786230000000,
  "dispatch_claimed_at_ms": null,
  "terminal_at_ms": null,
  "reason": null,
  "progress": {
    "phase": "validation",
    "code": "definition_apply_queued",
    "completed_units": 0,
    "total_units": null,
    "updated_at_ms": 1786230000000,
    "summary": "The toolbox definition is queued for validation.",
    "cancellable": true
  },
  "result": null,
  "result_ref": null,
  "result_omission": null
}
```

Clients observe status, retrieve results, and recover a request through the
generic hosted-operation APIs. A status may include bounded progress with
`phase`, `code`, optional non-negative `completed_units` and `total_units`,
`updated_at_ms`, and a user-safe `summary`. Stable phases are `validation`,
`environment_build`, `staging`, `warmup`, `publication`, `draining`, and
`cleanup`. The strict progress object also contains `cancellable`; it is always
false from `publication` through `cleanup` and can never return to true for one
operation.

Cancellation is allowed while queued and during validation, environment build,
staging, and warmup when candidate cleanup is safe. The persisted start of
publication is the non-cancellable commit boundary. A cancellation request at
or after that boundary returns `apply_publication_committed`; it does not roll
back the active definition. Draining and cleanup then complete idempotently.

Terminal success returns the active revision, active tool inventory, bounded
rollout summary, and user projection through the durable operation result.
Terminal failure before publication leaves the prior active revision unchanged.
Operation receipts and tombstones follow the generic hosted-operation retention
contract (default seven-day receipts and fourteen-day tombstones).

A successful result payload is:

```json
{
  "contract": "hosting.toolbox.definition_apply_result",
  "tool_runtime_id": "runtime-local",
  "toolbox_id": "workspace-tools",
  "active_revision": "sha256:9999999999999999999999999999999999999999999999999999999999999999",
  "active_tools": ["CurrentWeather", "LookupRecord"],
  "rollout": {
    "reused_profiles": 0,
    "started_profiles": 1,
    "retired_profiles": 0,
    "cleanup_pending": false
  },
  "diagnostics": [],
  "user_projection": {
    "state": "ready",
    "code": "toolbox_definition_published",
    "summary": "The toolbox definition is active."
  }
}
```

The normal result never contains worker or physical package-environment
identities. A failed result uses the generic operation error plus bounded
diagnostics and `user_projection`; it does not return a partial active
definition.

## Public client surface

The supported client calls are:

```python
get_definition(*, operator_details: bool = False) -> dict
plan_definition(definition: dict, *, request_id: str, operator_details: bool = False) -> dict
confirm_definition_plan(*, plan_id: str, environment_choices: list[dict], request_id: str) -> dict
approve_confirmed_definition_plan(*, confirmation_ref: str) -> dict
apply_definition(
    *,
    plan_id: str,
    confirmation_ref: str,
    request_id: str,
    dependency_approval_ref: str | None = None,
) -> dict
list_environment_templates() -> dict
describe_environment_template(*, template_id: str) -> dict
```

`toolbox_describe` is a bounded persisted/registration read and never contacts
a worker. A client that needs a live worker inventory submits
`toolbox-describe-refresh` through `op-start` with a stable `request_id`, then
observes the durable `toolbox_describe_refresh` operation and reads its
terminal description result.

`operator_details=True` requests but does not grant the separate operator
projection; authorization still decides whether that object is present.
Planning, confirmation, and apply observation, result retrieval, cancellation,
request recovery, and changed-snapshot watching use generic hosted-operation
client calls and the returned operation ref. Environment template lifecycle and physical materialization controls are
administrative surfaces, not toolbox-client calls.

## Actor authorization

The parent authorizes every action against the authenticated actor, authority,
tool runtime, and toolbox scope. Read, plan, approve, apply, execute, and
operator-detail permissions are distinct. An actor may not read or reuse
another actor's plan, approval reference, request recovery record, operation
ref, or result merely by knowing its identifier.

Authorization failure uses stable `access_denied` or the non-disclosing
approval code described above. Normal errors do not reveal whether a resource
owned by another actor exists.

## User and operator projections

Normal plan, read, progress, and terminal responses contain a
`user_projection` with stable `state`, `code`, and bounded `summary`, plus
bounded remediation and affected-tool data where safe. Supported states are
`ready`, `setup_needed`, `approval_required`, `deploying`, and
`deployment_failed`.

Engine IDs, resolved profile IDs, environment keys, package/filesystem paths,
interpreter paths, raw locks, artifact credentials, and installer output are
excluded from all normal projections. They may appear only in a separately
authorized, bounded `operator_details` object. Clients must branch on stable
codes, not human-readable summaries.

## Stable error codes

The public error code set is:

| Code | Meaning |
| --- | --- |
| `access_denied` | Actor lacks the requested scoped authority. |
| `toolbox_not_found` | No active toolbox exists in the actor-visible scope. |
| `definition_invalid` | A typed field, value, or cross-field rule is invalid. |
| `definition_too_large` | A request exceeds a frozen validation limit. |
| `duplicate_stable_key` | Two requests use the same stable key. |
| `duplicate_tool_name` | Two entries advertise the same name in one toolbox. |
| `bundle_path_conflict` | Normalized staged paths have conflicting content. |
| `revision_conflict` | `expected_revision` is not the active revision. |
| `dependency_unresolved` | Required source imports cannot be resolved safely. |
| `dependency_declaration_conflict` | Source evidence and declarations conflict. |
| `template_not_found` | The selected logical template is unavailable. |
| `template_incompatible` | Runtime, ABI, platform, or imports do not match. |
| `package_policy_denied` | Package/index/artifact policy rejects a dependency. |
| `sandbox_policy_denied` | Requested sandbox capability is not authorized. |
| `plan_not_found` | The actor-visible plan does not exist. |
| `plan_expired` | The immutable plan is outside retention. |
| `plan_stale` | Catalog, package policy, or active revision changed. |
| `dependency_approval_required` | The exact custom delta needs parent approval. |
| `dependency_approval_invalid` | Approval authority cannot be accepted. |
| `request_id_conflict` | A stable request ID was reused for different work. |
| `state_corrupt` | Authoritative toolbox state failed strict validation. |
| `apply_failed` | Preparation failed before publication; prior routes remain. |
| `apply_publication_committed` | Cancellation is denied after the commit boundary. |
| `operation_not_found` | No actor-visible operation exists inside retention. |

Additional diagnostics may refine remediation, but clients must always handle
unknown future codes by presenting the bounded state and summary without
assuming success.

## Client algorithm

For every desired change, a client reads the authoritative revision, constructs
the complete desired definition, plans it, obtains an exact parent approval
reference when required, starts one durable apply with a stable request ID, and
persists the operation ref. It recovers status/result through the generic
operation APIs. On `revision_conflict` or `plan_stale`, it reads again,
reconciles the complete desired set, and creates a new plan. Applying an empty
complete definition is the supported toolbox teardown operation.
