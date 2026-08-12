# Unified hosting client breaking changes

Status: final control-v3 client contract frozen; parent implementation pin and
dependent adoption pending

This is the active handoff for the breaking plan in
[`hosting_access_plan.md`](hosting_access_plan.md). The previous toolbox rollout
handoff and adoption receipts are historical and remain available in Git. They
do not demonstrate adoption of this cut.

The product is unreleased. Parent implementation removes the old surface in the
same slices that add its replacement. There is no compatibility adapter,
deprecated alias, dual-read period, or automatic legacy environment migration.

## 1. Handoff gate

Consumer implementation must target the exact request, response, error,
capability, and version contract below and the recorded parent implementation
pin. Directional names are not permission to infer missing payloads.

- [x] Record the new daemon/control contract major version.
- [x] Publish a complete retained/renamed/removed command manifest.
- [x] Publish exact request and response schemas for every renamed command.
- [x] Publish generic readiness codes and the old-to-new disposition table.
- [x] Publish startup/configuration argument signatures.
- [x] Publish operation kinds, selectors, progress phases, and receipts.
- [x] Publish state/receipt contract versions and old-version rejection errors.
- [x] Publish atomic tool-change, selective-revision, per-tool analysis,
  and candidate-validation contracts before dependent adoption.
- [ ] Record the parent implementation pin.
- [ ] Record each dependent owner, revision, test receipt, and adoption status.

### 1.1 Frozen control contract

The following is the exact client-visible contract. Later slices may add fields
only through the versioned rules below. Unknown fields are rejected. No old
command, field, file, or state record is translated or used as a fallback.

#### Control version and envelope

- The daemon/control major is **3**, named `hosting.control.v3`.
- Every control response includes `contract: "hosting.control.v3"` and
  `contract_major: 3`; capability negotiation advertises the same values.
- A request sent with another major, or without a supported contract marker,
  fails with:

```json
{
  "ok": false,
  "error": "contract_mismatch",
  "error_code": "hosting_contract_major_unsupported",
  "error_details": {
    "expected_contract": "hosting.control.v3",
    "expected_major": 3,
    "received_contract": "hosting.control.v2"
  }
}
```

- A request envelope is `{contract, request_id, command, payload}`. `request_id`
  is an opaque non-empty ASCII identifier, at most 256 bytes, and is stable
  across retries. Responses are `{ok, result}` or the error object above;
  durable operations additionally return the versioned operation status below.
- The daemon version begins at `3.0.0` for this cut. The existing worker IPC
  protocol versions are independent and are not control-channel versions.

#### Configuration authority, roots, and records

`<config root>/hosting/hosting_config.json` is the only static hosting
authority and has contract `hosting.configuration.v3`. The top-level MP13
configuration owns these logical roots:

```json
{
  "category_dirs": {
    "hosting_root_dir": "@home/.mp13-llm/hosting",
    "packages_root_dir": "@home/.mp13-llm/packages",
    "environments_root_dir": "@home/.mp13-llm/environments"
  }
}
```

The only root-definition anchors are `@home`, `@config`, and `@temp`.
Persistent root definitions cannot use `@project`, another persistent label,
an absolute path, `..`, or a traversal escape. After host-local resolution the
three persistent roots must be existing-or-creatable directories and must not
overlap one another. Normal hosting configuration values may reference
`@hosting`, `@packages`, and `@environments` with forward-slash logical paths;
logical values are preserved on save and resolved only at a host boundary.

Root examples are frozen: `@home/.mp13-llm/packages` and
`@config/hosting` are valid root values; `@project/packages`,
`@home/../outside`, `C:\\outside`, and a root value of `@packages` are
rejected. `@packages/artifacts` is valid in a normal hosting value, while a
cycle such as `hosting_root_dir: "@packages"` plus
`packages_root_dir: "@hosting"` is rejected before resolution.

The layout and writer authority are fixed:

| Location | Class | Writer |
|---|---|---|
| `hosting/hosting_config.json` | static configuration | local hosting setup/config library only |
| `hosting/keyring/` | key and credential material | local setup/keyring writer; daemon reads |
| `hosting/audit/` | append-only audit records | daemon only |
| `hosting/state/` | mutable daemon and operation state | daemon only |
| `hosting/scratch/` | incomplete uploads/builds | daemon only; policy-based reap |
| `packages/artifacts/`, `packages/locks/` | immutable package content/locks | daemon only |
| `environments/templates/`, `environments/receipts/`, `environments/content/` | immutable templates, receipts, and built environments | daemon only |

Setup may create directories but may not write daemon records. The daemon
reads static configuration at startup and never rewrites it. Remote responses
contain logical roots only; local setup inspection may additionally return
resolved paths. Credential values, tokens, key material, and unrestricted host
paths are never returned.

#### Authorization and artifact identity

The effective role and scope are evaluated from the server-side authenticated
session. Password and public-key sessions with the same role and scope are
equivalent.

| Operation | Minimum authority |
|---|---|
| source or credential policy | `admin` |
| package upload for an assigned consumer | `worker_user` for that consumer, or `admin` |
| package lock approval | `dependency_approver` or `admin` |
| template create/replace/activate/deprecate/revoke | `admin` |
| environment request/reference acquire/release for an assigned consumer | `worker_user` for that consumer, or `admin` |
| toolbox tool-change plan/revision/confirmation | `worker_user` for that assigned toolbox, or `admin` |
| candidate prepare/read/renew/execute/publish/discard | `worker_user` for that assigned toolbox with the corresponding scoped permission, or `admin` |
| environment removal and garbage collection | `admin` |
| toolbox or worker administration outside the caller's assignment | `admin` |
| read-only status/list/diagnostics | `diagnostic_user`, `worker_user`, or `admin` as scoped |

Every received artifact is streamed into daemon-owned scratch space, hashed by
the daemon with SHA-256, and promoted atomically under the resulting canonical
identity `sha256:<64 lowercase hex digits>`. A caller-supplied digest is an
optional expectation; it never proves stored bytes. Audit records contain an
event ID, actor ID, effective role/scope, request/operation ID, result, and
configuration revision, but never secrets or unrestricted paths.

The authorization table applies identically to password and public-key
sessions: a matching role/scope pair receives the same allow or deny result;
the authentication method is recorded for audit only.

#### Neutral package and environment records

The strict record contracts are:

`hosting.package_source.v1`, `hosting.package_policy.v1`,
`hosting.package_lock.v1`, `hosting.environment_template.v1`,
`hosting.environment_request.v1`, `hosting.environment_lock.v1`,
`hosting.environment_receipt.v1`, and `hosting.environment_reference.v1`.

All reject unknown fields. IDs are 1–128 byte opaque ASCII values; revisions
are positive integers; digests use the canonical `sha256:` form. References
always contain `consumer_kind`, `consumer_id`, and `revision`. The minimum
schemas are:

```json
{
  "contract": "hosting.package_source.v1",
  "source_id": "internal-wheelhouse",
  "kind": "local",
  "locator": "@packages/artifacts",
  "credential_ref": null,
  "enabled": true,
  "priority": 100
}
```

```json
{
  "contract": "hosting.package_policy.v1",
  "policy_id": "default",
  "revision": 1,
  "allowed_source_ids": ["internal-wheelhouse"],
  "allowed_platforms": ["win_amd64"],
  "allowed_runtimes": ["python"],
  "max_artifact_bytes": 67108864,
  "require_sha256": true,
  "optional_verifier": null
}
```

```json
{
  "contract": "hosting.package_lock.v1",
  "lock_id": "lock-01",
  "revision": 1,
  "policy_id": "default",
  "policy_revision": 1,
  "artifacts": [{"artifact_id": "sha256:0000000000000000000000000000000000000000000000000000000000000000", "size_bytes": 1234, "source_id": "internal-wheelhouse"}],
  "dependencies": [{"name": "numpy", "version": "1.26.4", "artifact_id": "sha256:0000000000000000000000000000000000000000000000000000000000000000"}],
  "lock_digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
}
```

```json
{
  "contract": "hosting.environment_template.v1",
  "template_id": "py-compute",
  "revision": 3,
  "runtime_kind": "python",
  "builder_id": "python-venv-v1",
  "package_lock_id": "lock-01",
  "platforms": ["win_amd64"],
  "state": "active"
}
```

```json
{
  "contract": "hosting.environment_request.v1",
  "request_id": "req-01",
  "consumer_kind": "toolbox",
  "consumer_id": "toolbox-01",
  "revision": 7,
  "template_id": "py-compute",
  "template_revision": 3,
  "package_lock_digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "runtime_kind": "python",
  "platform": "win_amd64",
  "configuration_revision": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
}
```

`hosting.environment_lock.v1` records the immutable content key and the exact
runtime, platform, template revision, package-lock digest, and policy inputs.
`hosting.environment_receipt.v1` adds the verified environment ID, content key,
receipt revision, and logical root reference. `hosting.environment_reference.v1`
contains `{reference_id, environment_id, consumer_kind, consumer_id, revision,
acquired_at_ms, released_at_ms|null}`. None of these records contains a
credential, token, or resolved absolute path.

The same request shape is used by non-toolbox workers. For example, a Python
workflow helper uses `consumer_kind: "workflow_python_helper"`, its stable
`consumer_id`, and its own definition revision while resolving the same package
lock and template as a toolbox; it does not use a toolbox-owned record.

#### Command cutover and readiness

The renamed command payloads are strict and use the common envelope. Successful
upload responses return an artifact identity; asynchronous commands return a
`hosting.operation_status.v3` operation reference.

| Removed command | Replacement | Request payload | Success result |
|---|---|---|---|
| `toolbox-artifact-upload-begin` | `package-artifact-upload-begin` | `source_id`, `total_size`, optional `expected_digest`, `request_id` | `upload_id`, `chunk_size`, `expires_at_ms`, `configuration_revision` |
| `toolbox-artifact-upload-chunk` | `package-artifact-upload-chunk` | `upload_id`, `chunk_index`, `offset`, `chunk_base64url` | `upload_id`, `received_bytes`, `next_chunk_index` |
| `toolbox-artifact-upload-status` | `package-artifact-upload-status` | `upload_id` | `upload_id`, `state`, `received_bytes`, optional `computed_digest` |
| `toolbox-artifact-upload-cancel` | `package-artifact-upload-cancel` | `upload_id`, `request_id` | `upload_id`, `state: cancelled` |
| `toolbox-artifact-upload-commit` | `package-artifact-upload-commit` | `upload_id`, `request_id` | `artifact_id`, `digest`, `size_bytes`, `receipt` |
| `toolbox-template-list` | `environment-template-list` | `include_revoked` (boolean, default false) | `templates[]`, `configuration_revision` |
| `toolbox-template-describe` | `environment-template-describe` | `template_id`, optional `revision` | one `hosting.environment_template.v1` record |
| `toolbox-template-construct` | `environment-template-construct` | `template_id`, `base_revision`, `imports[]`, `package_requirements[]`, `request_id` | operation status |
| `toolbox-template-activate` | `environment-template-activate` | `template_id`, `revision`, `request_id` | template record |
| `toolbox-template-replace` | `environment-template-replace` | `template_id`, `expected_active_revision`, `replacement`, `request_id` | template record |
| `toolbox-template-deprecate` | `environment-template-deprecate` | `template_id`, `revision`, `request_id` | template record |
| `toolbox-template-revoke` | `environment-template-revoke` | `template_id`, `revision`, `request_id` | template record |
| `toolbox-template-prewarm` | `environment-template-prewarm` | `template_id`, optional `revision`, `runtime_kind`, `platform`, `request_id` | operation status |
| `toolbox-environment-remove` | `environment-remove` | `environment_id`, `request_id` | operation status, or a frozen removal-denial code |

`toolbox-get-definition`, `toolbox-plan-definition`,
`toolbox-plan-tool-changes`, `toolbox-revise-definition-plan`,
`toolbox-confirm-definition-plan`, `toolbox-approve-confirmed-definition-plan`,
`toolbox-prepare-definition-candidate`,
`toolbox-execute-definition-candidate`,
`toolbox-get-definition-candidate`,
`toolbox-renew-definition-candidate`,
`toolbox-publish-definition-candidate`,
`toolbox-discard-definition-candidate`, `toolbox-apply-definition`,
`toolbox-execute`, and toolbox describe, consistency, repair, reconcile,
review, references, and archive commands remain toolbox-specific. `toolbox-gc`
is replaced by generic `hosting-gc`; its selector is
`{kind: "host_scope", id: "hosting"}` and it marks from every worker reference
before sweeping.

The complete toolbox command disposition is:

- retained/new in the v3 toolbox family: `toolbox-get-definition`,
  `toolbox-plan-definition`, `toolbox-plan-tool-changes`,
  `toolbox-revise-definition-plan`, `toolbox-confirm-definition-plan`,
  `toolbox-approve-confirmed-definition-plan`,
  `toolbox-prepare-definition-candidate`,
  `toolbox-execute-definition-candidate`,
  `toolbox-get-definition-candidate`,
  `toolbox-renew-definition-candidate`,
  `toolbox-publish-definition-candidate`,
  `toolbox-discard-definition-candidate`, `toolbox-apply-definition`,
  `toolbox-execute`, `toolbox-describe`, `toolbox-describe-refresh`,
  `toolbox-consistency`, `toolbox-gate`, `toolbox-reconcile`,
  `toolbox-references`, `toolbox-repair`, and `toolbox-review-snapshot`;
- renamed: every upload/template/environment command enumerated in the table
  above, plus `toolbox-gc` to `hosting-gc`; and
- removed without an alias: `toolbox-state-archive-v1`. The v3 daemon rejects
  old state directly and provides only the separately authorized local cleanup
  procedure; it does not expose a remote legacy-state archive command. The
  procedure archives only legacy toolbox-owned state, never generic package or
  environment state, and refuses to run after any newer toolbox lifecycle
  repository has been initialized.

Strings such as `toolbox-host`, `toolbox-ready`, `toolbox-executor-v1`, and
`toolbox-host-capability-dispatch` are internal selectors, readiness states, or
worker contracts rather than control commands. Their owning slices either
retain them for toolbox execution semantics or version them independently;
they are not command aliases.

The exact generic readiness codes are:

| Old code | New code |
|---|---|
| `toolbox_configuration_missing` | `hosting_configuration_missing` |
| `toolbox_configuration_incomplete` | `hosting_configuration_incomplete` |
| `toolbox_configuration_invalid` | `hosting_configuration_invalid` |
| `toolbox_source_binding_invalid` | `package_source_invalid` |
| — | `hosting_configuration_unsupported` |
| — | `package_source_unavailable` |
| — | `package_credential_unavailable` |
| — | `package_policy_rejected` |
| — | `package_artifact_hash_mismatch` |
| — | `environment_template_unavailable` |
| — | `environment_build_failed` |
| — | `environment_referenced`, `environment_busy`, `environment_retained` |

Readiness uses `hosting.readiness.v1` with exactly `status` (`ready`,
`degraded`, or `unavailable`), `code`, `summary`, `subsystem`, and
`configuration_revision`. It never includes credentials or resolved paths.

#### Durable operations

The operation contracts are `hosting.operation_ref.v3`,
`hosting.operation_status.v3`, `hosting.result_ref.v3`, and
`hosting.result_omission.v3`. Operation kinds and selectors are:

| Kind | Selector | Phases |
|---|---|---|
| `package_artifact_upload` | `upload_id` | `validation`, `ingress`, `hashing`, `promotion`, `receipt` |
| `environment_template` | `template_id` | `validation`, `resolution`, `artifact_verification`, `environment_build`, `receipt_commit`, `publication`, `cleanup` |
| `environment_remove` | `environment_id` | `validation`, `reference_check`, `removal`, `cleanup` |
| `hosting_gc` | `host_scope=hosting` | `validation`, `mark`, `sweep`, `cleanup` |
| `toolbox_definition_plan` | `toolbox_id` | `validation`, `import_analysis`, `resolution`, `plan_commit` |
| `toolbox_definition_plan_revision` | `toolbox_id` | `validation`, `reduction`, `import_analysis`, `resolution`, `plan_commit` |
| `toolbox_definition_candidate_prepare` | `toolbox_id` | `validation`, `environment_build`, `staging`, `warmup`, `candidate_ready` |
| `toolbox_definition_candidate_publish` | `toolbox_id` | `validation`, `publication`, `draining`, `cleanup` |
| `toolbox_definition_candidate_discard` | `toolbox_id` | `validation`, `draining`, `cleanup` |

The status shape is `{contract, api_status, operation, lifecycle, request_id,
created_at_ms, updated_at_ms, reason, result, progress}`. A retry with the same
`(owner_actor_id, receipt_namespace, request_id)` and identical fingerprint
returns the original operation; a different fingerprint returns
`operation_idempotency_conflict` and performs no mutation.

#### Clean-cut state and rejection

The new repositories use the record contracts above plus
`hosting.upload.v1`, `hosting.operation.v3`, and `hosting.configuration.v3`.
The following are rejected as unsupported, never read as fallback data:

- `access_control.json`, toolbox launcher JSON, and the five toolbox startup
  mapping fields;
- `hosting.toolbox.artifact_store.v2`, `hosting.toolbox.artifact_uploads.v1`,
  `hosting.toolbox.environment.v2`, `hosting.toolbox.environment_references.v1`,
  `hosting.toolbox.template_catalog_state.v1`, and toolbox-owned receipt/state
  identifiers;
- unversioned or v1 operation/result envelopes and all old toolbox command
  names; and
- `toolbox_venvs`, `runtime_envs`, and `toolbox_environment_cache`.

The stable rejection is `{ok:false, error:"unsupported_state_contract",
error_code:"state_contract_unsupported", error_details:{received_contract,
supported_contracts}}`. A host containing only `access_control.json` fails with
`hosting_configuration_missing`; it is not imported. Legacy environments are
rebuilt under `@environments` and are never discovered, reused, referenced, or
garbage-collected by the new daemon. Operator cleanup is a separate local
action.

#### Host-local root customization

The host-local setup API uses `hosting.setup.v1` requests:

```json
{
  "contract": "hosting.setup.v1",
  "operation": "plan|apply|inspect|status|reset",
  "config_file": "C:/host/mp13.json",
  "roots": {
    "hosting_root_dir": "@home/.mp13-llm/hosting",
    "packages_root_dir": "@home/.mp13-llm/packages",
    "environments_root_dir": "@home/.mp13-llm/environments"
  },
  "expected_config_revision": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
  "expected_hosting_revision": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
  "allow_nonempty_destinations": false,
  "confirm": false
}
```

`plan` returns logical roots, local resolved roots, revisions, and preflight
results for permissions, collisions, free space, non-empty destinations,
daemon activity, and cross-volume moves. `apply` requires matching expected
revisions and `confirm:true`; it returns `hosting.setup.result.v1` with the new
revisions and journal state. `inspect` and `status` are local-only and may show
resolved paths. `reset` requires `confirm:true`, preserves audit history, and
does not delete packages or environments.

When both top-level configuration and `hosting_config.json` change, the setup
library writes a local journal with phases `prepared`, `top_level_written`,
`hosting_written`, and `committed`. Each file uses locked temporary write,
fsync, and replace semantics. Recovery is idempotent; active-daemon relocation,
unsafe destinations, and unapproved cross-volume moves are refused. No remote
control command can relocate roots, and no cross-filesystem atomic rename is
claimed.

If interrupted in `prepared`, recovery discards the temporary files; in
`top_level_written`, it restores the last complete pair; in
`hosting_written`, it completes or rolls back both files according to the
recorded target revisions; and in `committed`, it only removes journal scratch.
Every recovery path is idempotent and leaves one declared revision pair.

## 2. Ownership and rollout

The parent repository owns daemon, channel, CLI, setup, configuration, package,
and environment contracts. Dependent maintainers own their repository changes.
Parent work may inspect dependent code but must not modify it.

Rollout order:

1. Complete R0 contract freeze and R1 handoff/inventory in the parent.
2. Land a coherent parent breaking slice and record its exact pin here.
3. Dependent maintainers update code, configuration, tests, and documentation
   against that pin.
4. The dependent supplies a revision and test receipt.
5. Parent acceptance verifies the receipt and records it here.
6. Keep this file populated until all named consumers adopt the new major
   contract.

## 3. Configuration and startup cut

### Removed configuration authorities

- `access_control.json`
- the short-lived toolbox launcher JSON
- direct startup/control settings named:
  - `toolbox_host_project_configuration`
  - `toolbox_artifact_sources`
  - `toolbox_trust_public_keys`
  - `toolbox_source_credentials`
  - `toolbox_dependency_policy`
- `--toolbox-config-file`
- `engine_host_toolbox_config_file`
- mandatory `trust_key_ids`, signed manifests, and publisher public-key maps in
  the baseline package path

### Replacement direction

The one authoritative static hosting configuration is:

```text
<config root>/hosting/hosting_config.json
```

The top-level MP13 configuration owns `hosting_root_dir`, `packages_root_dir`,
and `environments_root_dir`, exposed as `@hosting`, `@packages`, and
`@environments`. Hosting configuration uses logical references; clients must
not manufacture daemon-local absolute paths.

Daemon startup receives only the top-level MP13 configuration location needed
to resolve the hosting configuration locally. The frozen local parameter is
`mp13_config_file`; the CLI flag is `--mp13-config-file`; and the locally owned
channel setting is `engine_host_mp13_config_file`. None may contain credentials
or a hosting-specific launcher map.

Static authentication, source, credential, dependency, retention, and root
policy is host-local and becomes active after deliberate daemon restart.
Toolboxes, packages, templates, and environments remain dynamic control-channel
operations after startup and do not rewrite `hosting_config.json`.

### Dependent removals

- [ ] Stop constructing or forwarding any of the five toolbox mappings.
- [ ] Stop creating launcher JSON files.
- [ ] Remove the old CLI/control-setting names and conflict branching.
- [ ] Stop reading `access_control.json` as readiness or configured-state truth.
- [ ] Do not expose credentials or host paths in arguments, logs, status, or UI.
- [ ] If the dependent owns local daemon bootstrap, adopt
  `--mp13-config-file`/`engine_host_mp13_config_file`.

## 4. Authorization and artifact identity

Authenticated server-side role and effective scope authorize package,
environment, toolbox, and worker administration. Password and public-key
sessions with the same effective role/scope have the same authority.

The daemon computes SHA-256 from bytes received into host-controlled scratch
storage. A client-provided hash is an optional expected value only. Clients
must not treat their own hash, a URL, a filename, or a publisher signature as
proof that the daemon stored those bytes.

Dependent behavior:

- [ ] Use daemon-returned package identities and immutable locks.
- [ ] Treat role denial, policy denial, and expected-hash mismatch as distinct
  bounded errors.
- [ ] Remove mandatory publisher-signing-key provisioning from the mainstream
  setup flow.
- [ ] Never send a private publisher key, environment directory, install
  command, or credential value through the control channel.

## 5. Authentication result propagation

Fresh and cached public-key authentication paths return an equivalent complete
session result. Do not narrow this mapping to a token string:

```json
{
  "status": "ok",
  "token": "opaque-session-token",
  "role": "admin",
  "auth_method": "public_key",
  "scope": "control",
  "key_id": "registered-key-id",
  "reused": false
}
```

`reused` changes on cached reuse; token expiry/binding fields may also be
present. The authorization metadata contract is otherwise equivalent without a
second handshake. Tokens remain secret even though this structured result is
returned to the trusted local caller.

Parent navigation:

- `src/hosting/client_realm_api.py::authenticate_client_with_key`
- `src/hosting/engine_host_channel.py::ensure_public_key_session`

Dependent navigation:

- `src/backend/platform/hosting/hosting_admin.py::ensure_public_key_session`

Dependent adoption must prove that an admin-only call succeeds from the
returned/cached role-bearing session without an extra handshake or a separate
role-probing workaround.

## 6. Package and environment API cut

### Renamed command families

| Removed | Replacement | Exact payload status |
|---|---|---|
| `toolbox-artifact-upload-begin` | `package-artifact-upload-begin` | Frozen in §1.1 |
| `toolbox-artifact-upload-chunk` | `package-artifact-upload-chunk` | Frozen in §1.1 |
| `toolbox-artifact-upload-status` | `package-artifact-upload-status` | Frozen in §1.1 |
| `toolbox-artifact-upload-cancel` | `package-artifact-upload-cancel` | Frozen in §1.1 |
| `toolbox-artifact-upload-commit` | `package-artifact-upload-commit` | Frozen in §1.1 |
| `toolbox-template-*` | `environment-template-*` | Frozen in §1.1 |
| `toolbox-environment-remove` | `environment-remove` | Frozen in §1.1 |

There are no old-name aliases. Generic package/environment requests and
references carry `consumer_kind`, `consumer_id`, and `revision` where frozen by
R0.4/R0.5.

### Retained toolbox command family

These names remain toolbox-specific because their payloads describe toolbox
semantics:

- `toolbox-get-definition`
- `toolbox-plan-definition`
- `toolbox-confirm-definition-plan`
- `toolbox-approve-confirmed-definition-plan`
- `toolbox-apply-definition`
- `toolbox-execute`
- toolbox describe, consistency, gate, reconcile, repair, review, references,
  GC, and archive commands; `toolbox-gc` is replaced by `hosting-gc`.

Retaining a name does not guarantee that every nested package/environment
field remains unchanged. Nested identities, receipts, operations, and readiness
use the frozen generic contracts in §1.1.

### Tool-change and candidate-validation contract

The control major is `hosting.control.v3`. Toolbox planning uses
`hosting.toolbox.definition_plan.v2` and successful confirmation uses
`hosting.toolbox.confirmation_receipt.v1`.

`toolbox-plan-tool-changes` accepts exactly `toolbox_id`,
`expected_revision`, `changes`, `request_id`, and `operator_details`. There are
at most 512 changes; each has a unique 1-128 printable-ASCII `change_id` and one
strict shape:

```json
{
  "change_id": "edit-weather",
  "kind": "add|update|rename|remove",
  "target_tool_key": "workspace_tools.weather:CurrentWeather",
  "request_kind": "auto|manual",
  "request": {}
}
```

For add, `target_tool_key` is null. For update it identifies the active request
and the resulting stable key is unchanged. For rename it identifies the active
request and the resulting stable key changes. For remove, `request_kind` and
`request` are null. The result is a `toolbox_definition_plan` operation whose
terminal v2 record contains `proposal_kind`, normalized `changes`, and
`tool_analysis`. Every analysis entry contains exactly `change_id`, `tool_key`,
`prior_tool_key`, `change`, `imports`, `environment_id`, `package_mutations`,
and `approval_required`; each import contains `import_root`, `classification`,
nullable `distribution`, and bounded `{relative_path,line,kind}` evidence.
For `toolbox-plan-definition`, which has no caller change IDs, the host assigns
each changed tool a deterministic `host:sha256:<64 lowercase hex>` change ID
from its change kind, prior stable key, and proposed stable key. Selective
child-plan revision is therefore available to both planning entry points.

The public terminal plan projection has exactly `contract`, `plan_id`,
`parent_plan_id`, `toolbox_id`, `proposal_kind`, `definition_hash`,
`expected_revision`, `pins`, `expires_at_ms`, `can_apply`,
`confirmation_required`, `approval_required`, `changes`, `tool_analysis`,
`environment_mutations`, `profile_diff`, `reduction`, `diagnostics`, and
`user_projection`. `parent_plan_id` and `reduction` are null on a root plan.
On a revised plan, `reduction` has exactly `excluded_changes`,
`preserved_active_tool_keys`, and `cascade_exclusions`.
Each normalized change has exactly `change_id`, `kind`, `prior_tool_key`,
`tool_key`, and `request_kind`. The existing strict pins, environment mutation,
alternative, exact artifact, package mutation, dependency-edge, diagnostic, and
projection shapes remain nested unchanged except that every affected tool also
carries its `change_id`.

`toolbox-revise-definition-plan` accepts exactly `plan_id`, `decisions`,
`request_id`, and `operator_details`. It requires one decision for
every changed tool:

```json
{
  "change_id": "edit-weather",
  "decision": "accept|exclude",
  "denied_import_roots": ["requests"]
}
```

Accepted decisions have an empty denied-import list. Excluded import roots must
be evidenced for that tool and are explanatory only: the host excludes the add,
preserves the active update/rename/removal, reduces dependent changes, and
recomputes a complete definition and exact package closure. Success is a
`toolbox_definition_plan_revision` operation whose terminal result is a new v2
plan with `parent_plan_id`, `excluded_changes`,
`preserved_active_tool_keys`, and `cascade_exclusions`. The parent plan and its
locks are never edited or reused for the reduced closure.

Final `toolbox-confirm-definition-plan` keeps the strict per-environment shape
`{environment_id, alternative_id, accept_package_changes}`. False for an
addition or transition returns `tool_change_revision_required`, bounded
affected change IDs, and no confirmation receipt. The client revises and then
confirms the child plan. Successful confirmation produces only
`hosting.toolbox.confirmation_receipt.v1`.
Its public terminal projection has exactly `contract`, `confirmation_ref`,
`plan_id`, `definition_revision`, `accepted_change_ids`,
`selected_alternatives`, `package_mutations`,
`dependency_approval_required`, `expires_at_ms`, and `user_projection`.

`toolbox-prepare-definition-candidate` accepts exactly `plan_id`,
`confirmation_ref`, `request_id`, nullable `dependency_approval_ref`, and
nullable `requested_lifetime_ms`. It consumes a required approval exactly once
and returns a
`toolbox_definition_candidate_prepare` operation. Terminal
success is `hosting.toolbox.definition_candidate.v1` with exactly
`candidate_ref`, `toolbox_id`, `definition_revision`, `changed_tool_keys`,
`created_at_ms`, `expires_at_ms`, `state`, and `user_projection`.
`toolbox-get-definition-candidate` accepts exactly `candidate_ref` and returns
that same bounded record with state `ready`, `published`, `discarded`, or
`expired`; it is side-effect-free and returns no worker or environment identity.
`toolbox-renew-definition-candidate` accepts exactly `candidate_ref`,
`requested_lifetime_ms`, and stable `request_id`. It returns the same bounded
candidate record with a new `expires_at_ms`. Each requested preparation or
renewal window must be 300000 through 14400000 milliseconds. Renewal can repeat
while the same actor/scope remains authorized and all active and policy pins
remain current.

`toolbox-execute-definition-candidate` accepts exactly `candidate_ref`,
`tool_call`, stable `execution_request_id`, `timeout_seconds`, `tools_view`,
`callback_binding`, and `host_api_approval`, using the same field contracts as
`toolbox-execute`. It can address only changed candidate tools and uses the same
durable execution result. It does not publish routes. It applies all normal
tool gates and sandbox/host-API/data/network approvals; candidate execution is
not a dry run and may perform authorized external effects.
Dispatch atomically acquires an in-flight candidate lease. Candidate expiry does
not retire workers, release references, or cancel that execution before its
ordinary terminal result and cleanup grace. If no execution lease exists when
expiry is reached, new execution/publication is denied and cleanup begins.

`toolbox-publish-definition-candidate` accepts exactly `candidate_ref` and
`request_id`; it revalidates all active and policy pins and publishes the exact
warmed candidate without rebuilding or reresolving. Its operation kind is
`toolbox_definition_candidate_publish`, and terminal success uses the existing
`hosting.toolbox.definition_apply_result` projection.
`toolbox-discard-definition-candidate`
accepts the same two fields and idempotently retires candidate workers and
references through `toolbox_definition_candidate_discard`; terminal success is
`hosting.toolbox.definition_candidate_discard_result.v1` with exactly
`contract`, `toolbox_id`, `definition_revision`, `state: "discarded"`, and
`user_projection`. Only the same authenticated actor in an authorized toolbox
scope can recover, execute, publish, or discard a candidate; reconnecting under
that same identity and scope is supported. The host returns exact
`expires_at_ms`; consumers may request or renew a window within advertised
bounds. Expired or stale candidates never publish.

`toolbox-apply-definition` remains the one-shot alternative and accepts
`plan_id`, `confirmation_ref`, `request_id`, and nullable
`dependency_approval_ref`. A dependency approval is consumed by either one-shot
apply or candidate preparation. Candidate publication uses the retained
approval identity and never receives the raw approval again.

The new stable codes are `tool_change_invalid`, `tool_change_conflict`,
`tool_change_revision_required`, `candidate_not_found`, `candidate_expired`,
`candidate_stale`, `candidate_renewal_denied`, and
`candidate_execution_denied`. Candidate user projection state may be
`candidate_ready`; selective rejection uses `review_required`.

### Removed environment ownership

- `ToolboxEnvironmentManager`
- the compatibility `RuntimeEnvironmentManager` subclass
- toolbox-specific environment receipt/reference contracts selected by a
  directory kind
- `toolbox_venvs`
- `runtime_envs`
- `toolbox_environment_cache`

The replacement is the worker-neutral `EnvironmentManager` with
`EnvironmentRequest`, `EnvironmentLock`, `EnvironmentReceipt`, and
`EnvironmentReference`. Old environments are rebuilt; they are not discovered
or reused by the new daemon.

## 7. Readiness and capability cut

The following current readiness family is removed:

- `toolbox_configuration_missing`
- `toolbox_configuration_incomplete`
- `toolbox_configuration_invalid`
- `toolbox_source_binding_invalid`

The exact mapping is:

- unsupported/missing/invalid hosting configuration;
- unavailable package source or credential;
- dependency-policy rejection or artifact hash mismatch;
- unavailable template or failed environment build; and
- referenced/busy/retained environment removal denial.

Clients branch only on frozen codes and capability/contract versions. They must
not parse summaries or translate old readiness codes locally.

Known dependent navigation:

- `src/backend/platform/capabilities/parent_truth.py`
- `src/backend/platform/toolboxes/definition_coordinator.py`
- `src/ui/web/static/js/features/chat/CapabilityToolsPanel.js`
- `tests/backend_infra/test_parent_toolbox_truth.py`

## 8. Dynamic lifecycle behavior

After daemon startup, an authorized client can create or revise toolbox
definitions, upload packages, manage environment templates, and construct/reuse
environments without restarting the daemon. These operations use versioned
state and immutable content under host-configured roots.

Ordinary consumers batch one user action into `toolbox-plan-tool-changes` and
do not send every editor keystroke. Source-only iterations whose exact package
closure is unchanged reuse the generic environment. A consumer that needs to
try changed tools prepares an expiring candidate, executes only through its
opaque ref with ordinary effect approvals, and explicitly publishes or discards
it. Long review sessions renew the returned expiry; a long-running execution
holds an in-flight lease and is governed by its execution timeout rather than
candidate retention. This reuses the normal toolbox worker protocol and does
not add a generic code execution surface.

Python consumers use `HostedToolBoxRef.plan_tool_changes` and
`prepare_definition_candidate`; after resolving the prepare operation, they
reconnect with `candidate_session(candidate_ref=...)` and call its typed
`get`, `renew`, `execute`, `publish`, or `discard` methods. The session retains
only the opaque ref and toolbox identity and does not expose worker,
environment, or package identities.

Plans and operations pin the active hosting configuration revision. A restart
under changed static policy makes incompatible pending work stale; clients
must re-plan rather than silently continuing under new policy.

The durable operation kinds, selectors, phases, retry rules, cancellation rules,
and receipts are frozen in §1.1. Static policy changes invalidate a pending
operation whose pinned configuration revision is no longer compatible.

## 9. State and local data cut

- Old configuration and serialized package/environment contracts fail fast.
- Old environment directories are ignored by resolution, reuse, references,
  removal, and GC.
- Operator cleanup is explicit and local; daemon startup does not broadly
  delete legacy data.
- Keyring, audit, mutable state, scratch, immutable packages, and environments
  remain separate records/data even though static configuration has one
  authority.

The exact old/new identifiers and stable rejection error are frozen in §1.1;
implementations must fail closed with `state_contract_unsupported` and must not
attempt migration or legacy discovery.

## 10. Adoption receipt

### 10.1 Dependent implementation map (R1.1/R1.4)

The inspected dependent is `mp13-docs`, branch `redesign/cards_workflows`, at
revision `a36400e8af908f702a4db84e4fdb1894ac28da36`. The inspection was read-only.
Its existing untracked `parent_project_feature.md` is user-owned and was not
read, edited, or included as evidence.

The dependent team owns these changes:

| Boundary | Stable files and symbols | Required adoption |
|---|---|---|
| Contract gate | `src/backend/app/factory.py::MIN_HOST_DAEMON_VERSION`, `REQUIRED_HOST_DAEMON_CAPABILITIES`; `src/backend/platform/hosting/daemon_contract.py::ensure_min_daemon_contract` | Require `hosting.control.v3`, major 3, daemon `3.0.0+`, and the generic package/environment capabilities. Reject v2 rather than branching around it. |
| Authentication | `src/backend/platform/hosting/hosting_admin.py::_require_authentication_result_mapping`, `public_key_session_payload`; `src/backend/platform/hosting/daemon_sessions.py::ensure_daemon_session_for_backend_client` | Preserve `token`, `role`, `scope`, `auth_method`, `key_id`, and `reused` for both fresh and cached sessions. Keep token redaction only in external projections. |
| Readiness projection | `src/backend/platform/capabilities/parent_truth.py::sanitize_parent_toolbox_summary`; `src/backend/platform/capabilities/runtimes.py`; `src/backend/platform/toolboxes/definition_coordinator.py::_PARENT_RUNTIME_FAILURE_CODES` | Replace the four removed toolbox configuration codes with the §1.1 generic readiness codes and retain `subsystem` plus `configuration_revision`. |
| UI remediation | `src/ui/web/static/js/features/chat/CapabilityToolsPanel.js::normalizeRuntime`, `readinessRemediation` | Branch on the frozen generic codes, distinguish configuration/package/environment failures, and remove mandatory-signed-package guidance. |
| Toolbox plan/apply shapes | `src/backend/platform/toolboxes/definition_coordinator.py::_safe_environments`, `_safe_plan`, `_safe_confirmation`, `_safe_operation_status`, `ToolboxDefinitionCoordinator`; `src/backend/platform/toolboxes/hosted_store.py` | Carry package-lock and environment identities, consumer kind/ID/revision, configuration revision, and v3 operation records without translating old state. |
| Tool-change review | `src/backend/platform/toolboxes/definition_coordinator.py`; `src/ui/web/static/js/features/chat/CapabilityToolsController.js`, `CapabilityToolsPanel.js` | Submit atomic change sets; render per-tool import/package evidence; revise rejected changes through child plans; never locally strip packages or reconstruct locks. |
| Candidate validation | `src/backend/platform/toolboxes/definition_coordinator.py`; hosted operation storage; tool execution UI | Persist candidate refs, warn that execution may have external effects, apply ordinary tool gates, renew long reviews, and explicitly publish/discard or handle expiry/staleness. |
| Capability routes | `src/backend/app/routers/capabilities.py`; `src/backend/platform/capabilities/runtimes.py`; `src/ui/web/static/js/features/chat/CapabilityToolsController.js` | Adopt generic environment/template/package responses while keeping toolbox definition semantics toolbox-specific. |
| Startup/configuration | `src/backend/platform/hosting/hosting_admin.py::plan_local_hosting_config_payload`, `apply_local_hosting_config_payload`; `src/backend/app/routers/hosting_config.py` | Use `engine_host_mp13_config_file`/`--mp13-config-file`, logical roots, and `hosting.setup.v1`; remove `access_control.json` readiness assumptions. |

No direct generic command strings are currently present in dependent production
code; the dependent primarily calls typed parent-channel helpers. The only
removed command-family match is the intentional negative assertion in
`tests/backend_infra/test_toolbox_replacement_residuals.py`; it must be updated
to assert all command names in §1.1 are absent rather than treated as a runtime
caller.

Minimum dependent proof is:

- `tests/backend_infra/test_backend_client_auth.py`: fresh and cached
  role-bearing public-key results plus an admin-only call without re-handshake;
- `tests/backend_infra/test_parent_toolbox_truth.py` and
  `test_capability_runtimes.py`: exact readiness mapping and sanitized
  configuration revision;
- `tests/backend_infra/test_toolbox_definition_coordinator.py` and
  `test_hosting_definition_adoption.py`: generic locks, references, v3
  operations, atomic change sets, selective child replan, candidate
  prepare/execute/publish/discard, retry, restart, and stale-policy rejection;
- UI contract tests for `CapabilityToolsPanel.js` and
  `CapabilityToolsController.js`: generic remediation and no host-path/secret
  projection; and
- `tests/backend_infra/test_toolbox_replacement_residuals.py`: removed command,
  field, code, and compatibility vocabulary has no production matches.

The dependent owner is the `mp13-docs` maintainer team; no individual owner is
declared in-repository. Completion therefore requires a named maintainer in the
receipt below, a full dependent commit, the commands/results above, and the
parent implementation pin against which they ran.

| Field | Required value | Current value |
|---|---|---|
| Parent contract major | Exact version | `hosting.control.v3` / `3` |
| Parent implementation pin | Full commit | Pending |
| Dependent owner | Team/person | Pending |
| Dependent revision | Full commit | Pending |
| Configuration/startup tests | Commands and result | Pending |
| Authentication metadata tests | Fresh, cached, admin-only result | Pending |
| Package command tests | New commands; old commands rejected | Pending |
| Environment command tests | New commands; old commands rejected | Pending |
| Readiness/capability tests | Exact new codes/version | Pending |
| Toolbox lifecycle tests | Plan through execute/retry/restart | Pending |
| Tool-change review tests | Per-tool imports, rejection/replan, rename, conflicts | Pending |
| Candidate validation tests | Gates, warning, renewal, long execution lease, publish/discard/expiry/stale | Pending |
| Non-toolbox worker tests | Python helper and Node adoption | Pending |
| Secret/path redaction tests | Arguments/logs/status/errors/receipts | Pending |

The receipt is complete only when all rows identify reproducible evidence.
Prior toolbox-rollout receipts do not fill these rows.
