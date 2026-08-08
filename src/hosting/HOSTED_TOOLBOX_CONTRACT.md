# Hosted Toolbox Definition Contract

Status: normative

Contract family: `hosting.toolbox.definition`

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
| `hosting_template_admin` | Publish immutable template revisions, move lifecycle state, and start prewarm/materialization. |
| `hosting_auditor` | Read bounded operator projections and audit events; no mutation authority is implied. |

The consumer control methods are `toolbox-template-list` and
`toolbox-template-describe`. The administrative methods are
`toolbox-template-publish`, `toolbox-template-deprecate`,
`toolbox-template-revoke`, and `toolbox-template-prewarm`. They use the same
authenticated daemon control transport as other host administration. Prewarm
returns a durable hosted-operation ref. Role checks are distinct even when one
actor holds multiple roles.

### Immutable manifests and artifacts

A published revision contains a stable logical template ID, complete resolved
distribution lock, import-root set, Python ABI/platform constraints, parent
worker artifact digest, isolation-policy version, artifact records, provenance,
manifest digest, signing-key ID, signature algorithm, and signature.

The signature algorithm is `ed25519`. The signature is base64url over canonical
manifest bytes with the signature field absent. `signing_key_id` must resolve to
an active trusted public key in host project configuration. Every artifact
record contains immutable filename/distribution identity, exact byte size,
SHA-256 digest, and an approved logical origin reference. The daemon verifies
signature, manifest digest, artifact digest, artifact size, lock consistency,
and target tags before installation. A logical template ID plus identical
manifest digest is idempotent; the same immutable revision identity with
different content is rejected.

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
through the approved deployment/artifact channel. Preseeding verifies manifest
signature and artifact digest before the artifact becomes discoverable. Clients
receive readiness/diagnostic projections, never the physical preseed path.

### Supported targets and timeouts

The initial supported targets are CPython 3.12 on `win_amd64` and CPython 3.12
on `manylinux_2_28_x86_64`. Their ABI tag is `cp312`. A template revision is
advertised only when its complete lock has compatible artifacts for the exact
target and the configured sandbox policy is enforceable. Other Python ABIs,
32-bit targets, macOS, musl, ARM, and free-threaded Python are unsupported until
a later signed catalog revision explicitly adds them.

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

Audit records cover publish, active-pointer change, deprecate, revoke, prewarm,
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

## Planning

`plan_definition(definition)` is actor-authorized and side-effect-free with
respect to logical toolbox state, package installation, bundle staging, and
workers. It returns a `ToolboxDefinitionPlan`:

```json
{
  "contract": "hosting.toolbox.definition_plan",
  "plan_id": "plan_01JABCDEF0123456789",
  "toolbox_id": "workspace-tools",
  "definition_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
  "expected_revision": null,
  "catalog_revision": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
  "package_policy_revision": "sha256:6666666666666666666666666666666666666666666666666666666666666666",
  "expires_at_ms": 1786233600000,
  "can_apply": false,
  "approval_required": true,
  "custom_delta_digest": "sha256:7777777777777777777777777777777777777777777777777777777777777777",
  "imports": [],
  "environments": [],
  "profile_diff": {
    "reused": 0,
    "added": 1,
    "replaced": 0,
    "removed": 0
  },
  "diagnostics": [],
  "user_projection": {
    "state": "approval_required",
    "code": "custom_dependency_approval_required",
    "summary": "Review is required for additional packages."
  }
}
```

The bounded `imports` and `environments` projections carry detected import
roots, source evidence, reviewed distribution mappings, selected template IDs,
custom deltas, unresolved imports, and stable policy diagnostics. Internal
environment/profile identities and physical paths are not part of the normal
plan result.

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

## Dependency approval references

`approve_definition_plan(plan_id)` is an authenticated parent decision, never a
client assertion. Policy denial returns a stable diagnostic and no reference.
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
bound to the authenticated actor and authority, toolbox ID, plan ID, definition
hash, exact custom-delta digest, catalog revision, package-policy revision,
decision, mint time, expiry, and audit event.

Approval lifetime cannot exceed the plan lifetime or 60 minutes. Apply
atomically binds first use to one stable `request_id`. Idempotent retries by the
same actor with the same request ID and identical apply fingerprint may reuse
it. Another request ID is denied as consumed. Revocation is effective until
route publication begins. Expired, revoked, consumed, cross-actor,
wrong-authority, wrong-plan, wrong-definition, changed-delta, changed-policy,
and changed-catalog cases all return `dependency_approval_invalid` to an
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
plan_definition(definition: dict, *, operator_details: bool = False) -> dict
approve_definition_plan(*, plan_id: str) -> dict
apply_definition(
    *,
    definition: dict,
    plan_id: str,
    request_id: str,
    dependency_approval_ref: str | None = None,
) -> dict
list_environment_templates() -> dict
describe_environment_template(*, template_id: str) -> dict
```

`operator_details=True` requests but does not grant the separate operator
projection; authorization still decides whether that object is present.
Long-running apply observation, result retrieval, cancellation, and request
recovery use generic hosted-operation client calls and the returned operation
ref. Environment template lifecycle and physical materialization controls are
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
