# Dependent consumer migration guide: hosting control v3

Status: host contract and implementation are complete; dependent adoption and
receipt are pending

This document is the implementation guide for the dependent consumer project.
It describes what the consumer must change from its previous hosting client
integration to adopt the completed control-v3 host. It intentionally omits host
implementation internals and history, storage algorithms, and operator-only
maintenance details.

Adopt and test against this exact host implementation:

```text
migration_contract: hosting.control.v3
minimum_daemon_version: 3.0.0
host_implementation_commit: 4d01307f664366c3149bef539aaa1b4e3f98a82f
```

This is a clean break. Do not add v2 fallback branches, old command aliases,
dual reads or writes, client-side translation of old state, or automatic reuse
of old environments.

## 1. Required consumer changes at a glance

| Previous consumer behavior | Required control-v3 behavior |
|---|---|
| Accept daemon 2.x and infer capability from older toolbox responses | Require daemon `3.0.0+` and every capability listed below |
| Start or configure hosting with toolbox-specific maps, launcher JSON, `access_control.json`, or `--toolbox-config-file` | Pass only the top-level MP13 config through `--mp13-config-file` / `engine_host_mp13_config_file`; use `hosting.setup.v1` for local setup |
| Treat authentication as a token-only result or probe role separately | Preserve the complete fresh or cached session mapping: token, role, scope, auth method, key ID, and reuse state |
| Use toolbox-owned artifact, template, environment, or GC commands | Use generic package, environment-template, environment, and hosting-GC commands |
| Parse toolbox-specific package/environment records and readiness codes | Parse strict generic package/environment records and `hosting.readiness.v1` |
| Locally remove rejected packages/imports or reconstruct a reduced definition | Submit atomic tool changes, review per-tool evidence, and request an immutable child plan for exclusions |
| Apply a changed toolbox immediately to validate it | Optionally prepare an expiring candidate, execute it with normal effect approvals, then publish or discard it |
| Persist unversioned/v1 operation results or old toolbox environment state | Persist v3 operation/result projections and opaque generic references only; reject old state |
| Display host paths, tokens, credentials, worker IDs, or environment internals | Retain secrets only in trusted local session custody and expose only bounded public projections |

## 2. Gate the daemon and capabilities

Update the consumer's daemon compatibility gate before changing downstream
features. `hosting.control.v3` is the name of this migration contract; the
typed client-visible compatibility signal is the daemon version plus advertised
capabilities.

Require all of the following:

- daemon version `>=3.0.0`;
- these advertised capabilities:
  - `package_artifact_ingress_v1`;
  - `package_locks_v1`;
  - `environment_management_v1`;
  - `environment_references_v1`;
  - `environment_execution_leases_v1`;
  - `toolbox_tool_changes_v1`; and
  - `toolbox_definition_candidates_v1`.

Reject daemon 2.x or a missing capability. Do not hide the mismatch by
disabling only the affected UI or by translating old responses. Use the typed
channel helpers rather than duplicating the host transport envelope. Where a
method accepts `request_id`, supply an opaque stable ID and reuse it only for an
identical retry.

### Dependent code to change

- `src/backend/app/factory.py`
  - raise `MIN_HOST_DAEMON_VERSION` to `3.0.0`;
  - update `REQUIRED_HOST_DAEMON_CAPABILITIES` with the seven keys above.
- `src/backend/platform/hosting/daemon_contract.py`
  - make `ensure_min_daemon_contract` require the daemon version and every
    advertised capability above;
  - remove v2-compatible feature branches.

## 3. Replace startup and configuration integration

### Remove

- `access_control.json` as configured-state or readiness truth;
- toolbox launcher JSON;
- `toolbox_host_project_configuration`;
- `toolbox_artifact_sources`;
- `toolbox_trust_public_keys`;
- `toolbox_source_credentials`;
- `toolbox_dependency_policy`;
- `--toolbox-config-file`;
- `engine_host_toolbox_config_file`; and
- mandatory signing-key or publisher-key setup from the normal package path.

### Use

The consumer passes the top-level MP13 configuration file as `mp13_config_file`.
The process flag is `--mp13-config-file`, and the locally owned channel setting
is `engine_host_mp13_config_file`.

The top-level configuration may define these logical roots:

```json
{
  "category_dirs": {
    "hosting_root_dir": "@home/.mp13-llm/hosting",
    "packages_root_dir": "@home/.mp13-llm/packages",
    "environments_root_dir": "@home/.mp13-llm/environments"
  }
}
```

The consumer must preserve logical references such as `@hosting`, `@packages`,
and `@environments`. Do not construct, persist, or return daemon-local absolute
paths. Root definitions accept only `@home`, `@config`, or `@temp`; reject
absolute paths, `@project`, traversal, overlap, and root-to-root cycles.

Local setup uses a strict `hosting.setup.v1` payload:

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

For apply/reset, require explicit confirmation and the expected revisions.
Remote status and UI projections must contain logical roots only; local-only
setup inspection may show resolved paths.

### Dependent code to change

- `src/backend/platform/hosting/hosting_admin.py`
  - revise `plan_local_hosting_config_payload` and
    `apply_local_hosting_config_payload` for `hosting.setup.v1`;
  - pass only `engine_host_mp13_config_file` at startup.
- `src/backend/app/routers/hosting_config.py`
  - accept and return logical roots;
  - remove launcher-map and `access_control.json` assumptions.

## 4. Preserve the complete authentication result

Fresh and cached public-key authentication return the same authorization
metadata. Keep the complete trusted-local result instead of narrowing it to a
token:

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

`reused` changes for cached sessions; expiry and binding fields may also be
present. Password and public-key sessions with the same effective role and
scope have the same authority. Do not infer authorization from authentication
method, and do not perform a separate role-probing handshake.

Keep the token in trusted local custody. External API, UI, log, error, status,
audit, and receipt projections must redact it while preserving non-secret role,
scope, method, key ID, and reuse information when needed.

### Dependent code to change

- `src/backend/platform/hosting/hosting_admin.py`
  - update `_require_authentication_result_mapping` and
    `public_key_session_payload`.
- `src/backend/platform/hosting/daemon_sessions.py`
  - make `ensure_daemon_session_for_backend_client` retain the structured
    result for both fresh and cached sessions.

Prove that an admin-only call succeeds from a cached role-bearing session
without another handshake.

## 5. Adopt generic package and environment contracts

### Rename commands

There are no old-name aliases.

| Remove | Use instead | Strict payload | Success result |
|---|---|---|---|
| `toolbox-artifact-upload-begin` | `package-artifact-upload-begin` | `source_id`, `total_size`, optional `expected_digest`, `request_id` | `upload_id`, `chunk_size`, `expires_at_ms`, `configuration_revision` |
| `toolbox-artifact-upload-chunk` | `package-artifact-upload-chunk` | `upload_id`, `chunk_index`, `offset`, `chunk_base64url` | `upload_id`, `received_bytes`, `next_chunk_index` |
| `toolbox-artifact-upload-status` | `package-artifact-upload-status` | `upload_id` | `upload_id`, `state`, `received_bytes`, optional `computed_digest` |
| `toolbox-artifact-upload-cancel` | `package-artifact-upload-cancel` | `upload_id`, `request_id` | `upload_id`, `state: cancelled` |
| `toolbox-artifact-upload-commit` | `package-artifact-upload-commit` | `upload_id`, `request_id` | `artifact_id`, `digest`, `size_bytes`, `receipt` |
| `toolbox-template-list` | `environment-template-list` | `include_revoked` (default `false`) | `templates[]`, `configuration_revision` |
| `toolbox-template-describe` | `environment-template-describe` | `template_id`, optional `revision` | one template record |
| `toolbox-template-construct` | `environment-template-construct` | `template`: one complete `hosting.environment_template.v1` record | template record |
| `toolbox-template-activate` | `environment-template-activate` | `template_id`, `revision` | template record |
| `toolbox-template-replace` | `environment-template-replace` | `template`: one complete replacement `hosting.environment_template.v1` record | active template record |
| `toolbox-template-deprecate` | `environment-template-deprecate` | `template_id`, `revision` | template record |
| `toolbox-template-revoke` | `environment-template-revoke` | `template_id`, `revision` | template record |
| `toolbox-template-prewarm` | `environment-template-prewarm` | `request`: one complete `hosting.environment_request.v1` record | environment receipt/result |
| `toolbox-environment-remove` | `environment-remove` | `environment_id`, `request_id` | operation status or a removal-denial code |
| `toolbox-gc` | `hosting-gc` | `request_id` | operation status with selector `{kind: "host_scope", id: "hosting"}` |

Remove `toolbox-state-archive-v1` without replacement. A consumer must never
invoke it remotely.

These toolbox commands remain toolbox-specific and must not be renamed:

- `toolbox-get-definition`;
- `toolbox-plan-definition`;
- `toolbox-plan-tool-changes`;
- `toolbox-revise-definition-plan`;
- `toolbox-confirm-definition-plan`;
- `toolbox-approve-confirmed-definition-plan`;
- `toolbox-prepare-definition-candidate`;
- `toolbox-get-definition-candidate`;
- `toolbox-renew-definition-candidate`;
- `toolbox-execute-definition-candidate`;
- `toolbox-publish-definition-candidate`;
- `toolbox-discard-definition-candidate`;
- `toolbox-apply-definition`;
- `toolbox-execute`;
- `toolbox-describe` and `toolbox-describe-refresh`;
- `toolbox-consistency`, `toolbox-gate`, and `toolbox-reconcile`;
- `toolbox-references` and `toolbox-repair`; and
- `toolbox-review-snapshot`.

Although these names remain, their nested operation, package, environment,
readiness, and receipt projections use the new contracts in this guide.

### Parse the new records

The relevant strict record contracts are:

- `hosting.package_source.v1`;
- `hosting.package_policy.v1`;
- `hosting.package_lock.v1`;
- `hosting.environment_template.v1`;
- `hosting.environment_request.v1`;
- `hosting.environment_lock.v1`;
- `hosting.environment_receipt.v1`; and
- `hosting.environment_reference.v1`.

IDs are opaque ASCII values; revisions are positive integers; digests use
`sha256:<64 lowercase hex digits>`. Treat the daemon-returned artifact identity
and package lock as authoritative. A caller hash is only an expectation.

Every environment request/reference must retain `consumer_kind`,
`consumer_id`, and `revision`. Do not reinterpret a workflow helper or Node
worker as a toolbox consumer. Non-toolbox workers use their own stable kind,
ID, and definition revision while sharing the generic package/environment
system.

A representative environment request is:

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

Never expose credentials, install commands, resolved environment paths, or
private keys in requests or projections. Publisher signatures may be optional
policy but are not required by the baseline flow.

### Dependent code to change

- `src/backend/platform/toolboxes/definition_coordinator.py`
  - update `_safe_environments`, `_safe_plan`, `_safe_confirmation`, and
    `_safe_operation_status` for generic identities and v3 records.
- `src/backend/platform/toolboxes/hosted_store.py`
  - persist only versioned bounded projections and opaque references.
- `src/backend/app/routers/capabilities.py` and
  `src/backend/platform/capabilities/runtimes.py`
  - return generic package/template/environment results while retaining
    toolbox-specific definition semantics.

## 6. Replace readiness handling

Readiness is strict `hosting.readiness.v1` with exactly:

```text
status: ready | degraded | unavailable
code
summary
subsystem
configuration_revision
```

Branch on `code`; do not parse `summary`. Preserve `subsystem` and
`configuration_revision` through backend sanitization and UI state.

| Previous code | Control-v3 code |
|---|---|
| `toolbox_configuration_missing` | `hosting_configuration_missing` |
| `toolbox_configuration_incomplete` | `hosting_configuration_incomplete` |
| `toolbox_configuration_invalid` | `hosting_configuration_invalid` |
| `toolbox_source_binding_invalid` | `package_source_invalid` |

Also support these new codes:

- `hosting_configuration_unsupported`;
- `package_source_unavailable`;
- `package_credential_unavailable`;
- `package_policy_rejected`;
- `package_artifact_hash_mismatch`;
- `environment_template_unavailable`;
- `environment_build_failed`;
- `environment_referenced`;
- `environment_busy`; and
- `environment_retained`.

The UI must distinguish configuration, package-source/credential/policy, and
environment failures. Remove remediation text that requires signed packages.

### Dependent code to change

- `src/backend/platform/capabilities/parent_truth.py`
  - update `sanitize_parent_toolbox_summary`.
- `src/backend/platform/capabilities/runtimes.py`
  - carry the new code, subsystem, and configuration revision.
- `src/backend/platform/toolboxes/definition_coordinator.py`
  - replace `_PARENT_RUNTIME_FAILURE_CODES`.
- `src/ui/web/static/js/features/chat/CapabilityToolsPanel.js`
  - update `normalizeRuntime` and `readinessRemediation`.

## 7. Persist and resolve v3 durable operations

The consumer must accept:

- `hosting.operation_ref.v3`;
- `hosting.operation_status.v3`;
- `hosting.result_ref.v3`; and
- `hosting.result_omission.v3`.

The status projection is exactly `{contract, api_status, operation, lifecycle,
request_id, created_at_ms, updated_at_ms, reason, result, progress}`.

| Operation kind | Selector | Progress phases |
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

Retry with the same stable request ID and unchanged payload fingerprint.
An identical retry resolves to the original operation. A changed fingerprint
returns `operation_idempotency_conflict` and must not be retried as if it were
the same mutation.

Persist enough bounded state to resolve a lost response and reconnect after a
daemon restart. If the pinned `configuration_revision` becomes stale, re-plan;
do not silently continue under changed policy.

## 8. Implement atomic tool-change review

Keep the toolbox-specific definition commands, but add the new atomic change
and revision flow. Toolbox plans use `hosting.toolbox.definition_plan.v2` and
successful confirmation uses `hosting.toolbox.confirmation_receipt.v1`.

### Plan changes

Call `toolbox-plan-tool-changes` with exactly `toolbox_id`,
`expected_revision`, `changes`, `request_id`, and `operator_details`. Batch one
user action rather than sending editor keystrokes. There may be at most 512
changes. Every change has a unique 1–128 printable-ASCII `change_id`:

```json
{
  "change_id": "edit-weather",
  "kind": "add|update|rename|remove",
  "target_tool_key": "workspace_tools.weather:CurrentWeather",
  "request_kind": "auto|manual",
  "request": {}
}
```

- add: `target_tool_key` is null;
- update: target the active tool and preserve its stable key;
- rename: target the active tool and produce a new stable key; and
- remove: `request_kind` and `request` are null.

The terminal plan has exactly `contract`, `plan_id`, `parent_plan_id`,
`toolbox_id`, `proposal_kind`, `definition_hash`, `expected_revision`, `pins`,
`expires_at_ms`, `can_apply`, `confirmation_required`, `approval_required`,
`changes`, `tool_analysis`, `environment_mutations`, `profile_diff`,
`reduction`, `diagnostics`, and `user_projection`.

Render each tool's imports, bounded evidence, environment, package mutations,
and approval requirement. Preserve `change_id` through every projection.

### Revise exclusions

Do not locally delete imports, packages, or tools from a returned plan. Call
`toolbox-revise-definition-plan` with exactly `plan_id`, `decisions`,
`request_id`, and `operator_details`, with one decision per changed tool:

```json
{
  "change_id": "edit-weather",
  "decision": "accept|exclude",
  "denied_import_roots": ["requests"]
}
```

Accepted decisions have no denied imports. Denied imports must be evidenced for
that tool. The returned immutable child plan has `parent_plan_id` and a
`reduction` containing `excluded_changes`, `preserved_active_tool_keys`, and
`cascade_exclusions`. Display cascade outcomes; do not mutate or reuse the
parent plan's lock.

### Confirm or re-plan

`toolbox-confirm-definition-plan` retains per-environment choices shaped as
`{environment_id, alternative_id, accept_package_changes}`. If package changes
are declined, handle `tool_change_revision_required`, show the affected change
IDs, request a child plan, and then confirm that child. Do not expect a receipt
from the rejected confirmation.

A successful confirmation returns only
`hosting.toolbox.confirmation_receipt.v1`, with exactly `contract`,
`confirmation_ref`, `plan_id`, `definition_revision`, `accepted_change_ids`,
`selected_alternatives`, `package_mutations`,
`dependency_approval_required`, `expires_at_ms`, and `user_projection`.

Handle the stable tool-change errors `tool_change_invalid`,
`tool_change_conflict`, and `tool_change_revision_required`.

### Dependent code to change

- `src/backend/platform/toolboxes/definition_coordinator.py`
  - add atomic plan/revise/confirm orchestration and bounded projections.
- `src/ui/web/static/js/features/chat/CapabilityToolsController.js`
  - submit batches, persist change IDs, and invoke child revision.
- `src/ui/web/static/js/features/chat/CapabilityToolsPanel.js`
  - display per-tool import/package evidence and cascade results.

## 9. Implement candidate validation

Candidate validation is optional as a user path but, when exposed, must use the
complete lifecycle below. It is not a dry run: candidate execution can perform
authorized external effects and must show the same warning and approval gates
as normal toolbox execution.

### Prepare

`toolbox-prepare-definition-candidate` accepts exactly `plan_id`,
`confirmation_ref`, `request_id`, nullable `dependency_approval_ref`, and
nullable `requested_lifetime_ms`. A required dependency approval is consumed
once by preparation.

Terminal success is `hosting.toolbox.definition_candidate.v1` with exactly
`candidate_ref`, `toolbox_id`, `definition_revision`, `changed_tool_keys`,
`created_at_ms`, `expires_at_ms`, `state`, and `user_projection`.

### Recover and renew

- `toolbox-get-definition-candidate`: exactly `candidate_ref`;
- `toolbox-renew-definition-candidate`: exactly `candidate_ref`,
  `requested_lifetime_ms`, and stable `request_id`.

Persist only the opaque candidate ref and toolbox identity. Candidate states are
`ready`, `published`, `discarded`, or `expired`. Each requested lifetime must be
between 300000 and 14400000 milliseconds. Use the returned `expires_at_ms` and
renew long review sessions.

### Execute

`toolbox-execute-definition-candidate` accepts exactly `candidate_ref`,
`tool_call`, stable `execution_request_id`, `timeout_seconds`, `tools_view`,
`callback_binding`, and `host_api_approval`.

Only changed candidate tools may execute. Apply normal tool gates and
filesystem, data, network, subprocess, host-API, callback, timeout, and
cancellation policy. A candidate expiry must not be interpreted as cancellation
of an already leased execution; wait for its ordinary terminal result.

### Publish or discard

- `toolbox-publish-definition-candidate`: exactly `candidate_ref`, `request_id`;
- `toolbox-discard-definition-candidate`: exactly `candidate_ref`, `request_id`.

Publishing activates the prepared candidate. Discard is idempotent. Never send
the dependency approval again at publication. Only the same authenticated actor
and authorized toolbox scope can recover, renew, execute, publish, or discard.

Handle `candidate_not_found`, `candidate_expired`, `candidate_stale`,
`candidate_renewal_denied`, and `candidate_execution_denied`. On expiry or stale
pins, offer re-plan rather than publish.

`toolbox-apply-definition` remains the one-shot alternative and accepts
`plan_id`, `confirmation_ref`, `request_id`, and nullable
`dependency_approval_ref`.

### Dependent code to change

- `src/backend/platform/toolboxes/definition_coordinator.py`
  - add prepare/get/renew/execute/publish/discard orchestration.
- hosted operation persistence
  - store the opaque candidate ref and restart-safe operation identity.
- toolbox execution UI
  - show real-effect warnings, ordinary approvals, expiry, renewal, explicit
    publish/discard, and stale recovery.

## 10. Remove old state and compatibility behavior

Delete consumer parsing, persistence, fixtures, and fallback logic for:

- `access_control.json` and launcher JSON;
- the five toolbox startup mappings listed in section 3;
- `hosting.toolbox.artifact_store.v2`;
- `hosting.toolbox.artifact_uploads.v1`;
- `hosting.toolbox.environment.v2`;
- `hosting.toolbox.environment_references.v1`;
- `hosting.toolbox.template_catalog_state.v1`;
- unversioned or v1 operation/result envelopes;
- `toolbox_venvs`;
- `runtime_envs`;
- `toolbox_environment_cache`;
- every removed command and readiness code in this guide; and
- mandatory-signed-package language in mainstream setup and remediation.

Do not discover, import, reuse, reference, remove, or garbage-collect old
environment directories through the v3 client. The host rejects old state with
`state_contract_unsupported`; project that error without attempting client-side
migration:

```json
{
  "ok": false,
  "error": "unsupported_state_contract",
  "error_code": "state_contract_unsupported",
  "error_details": {
    "received_contract": "old.contract",
    "supported_contracts": ["new.contract"]
  }
}
```

## 11. Consumer implementation checklist

- [ ] Gate on daemon `3.0.0+` and all seven required capabilities.
- [ ] Replace startup/configuration fields with the top-level MP13 config and
  `hosting.setup.v1` logical-root flow.
- [ ] Preserve complete fresh/cached authentication metadata while redacting
  secrets externally.
- [ ] Replace all renamed package, template, environment, and GC calls; prove
  old names are absent from production.
- [ ] Parse generic package/environment records and preserve consumer identity
  and configuration revision.
- [ ] Replace readiness codes and remediation UI.
- [ ] Persist and resolve v3 durable operations with stable request IDs.
- [ ] Implement atomic change planning, per-tool evidence, selective immutable
  child plans, and confirmation retry through revision.
- [ ] Implement candidate prepare/recover/renew/execute/publish/discard if the
  UI offers pre-publication validation.
- [ ] Adopt the generic environment system for Python helper and Node workers
  without labeling them as toolbox consumers.
- [ ] Remove old state, aliases, fallback logic, old fixtures, and old docs.
- [ ] Verify no secret, host path, worker identity, or environment internals
  escape bounded projections.
- [ ] Record the adoption receipt in section 13.

## 12. Required verification

At minimum, add or update these dependent tests:

- `tests/backend_infra/test_backend_client_auth.py`
  - fresh and cached role-bearing public-key results;
  - admin-only call from cached session without re-handshake.
- `tests/backend_infra/test_parent_toolbox_truth.py` and
  `tests/backend_infra/test_capability_runtimes.py`
  - exact daemon-version/capability gate;
  - exact generic readiness mappings;
  - sanitized `subsystem` and `configuration_revision`.
- `tests/backend_infra/test_toolbox_definition_coordinator.py` and
  `tests/backend_infra/test_hosting_definition_adoption.py`
  - generic package locks, environment references, and v3 operations;
  - atomic add/update/rename/remove conflicts;
  - per-tool import/package evidence;
  - selective child revision and package-decline re-plan;
  - candidate prepare/get/renew/execute/publish/discard;
  - lost-response retry, daemon restart, expiry, stale policy, and long
    execution lease behavior.
- UI contract tests for `CapabilityToolsPanel.js` and
  `CapabilityToolsController.js`
  - generic remediation;
  - per-tool/cascade review;
  - real-effect candidate warnings and approval gates;
  - no secret, host-path, worker, or environment-internal projection.
- `tests/backend_infra/test_toolbox_replacement_residuals.py`
  - every removed command, field, code, state contract, and compatibility term
    has no production match.
- Python-helper and Node-worker integration tests
  - stable non-toolbox consumer identity;
  - generic environment reference acquisition/release;
  - retry/restart behavior.

Run the focused suites above plus the dependent project's complete backend and
UI regression suites. A narrow mocked test is not an adoption receipt: at least
one Windows run must use the pinned host implementation and exercise the real
daemon-version/capability gate, authentication result, planning operation, and
candidate or one-shot apply lifecycle.

## 13. Adoption receipt

Complete this table after all checklist items and tests pass. Prior receipts
against earlier host commits do not satisfy this migration.

| Field | Required value | Current value |
|---|---|---|
| Host migration contract | Exact target | `hosting.control.v3` / daemon `3.0.0` |
| Host implementation | Full commit | `4d01307f664366c3149bef539aaa1b4e3f98a82f` |
| Dependent owner | Named team/person | Pending |
| Dependent revision | Full commit | Pending |
| Configuration/startup tests | Commands and results | Pending |
| Authentication tests | Fresh, cached, and admin-only results | Pending |
| Package command tests | New commands and old-name rejection | Pending |
| Environment command tests | New commands and old-name rejection | Pending |
| Readiness/capability tests | Exact codes, daemon version, and capabilities | Pending |
| Toolbox lifecycle tests | Plan through apply/execute, retry, and restart | Pending |
| Tool-change review tests | Evidence, exclusion/replan, rename, and conflicts | Pending |
| Candidate validation tests | Gates, warning, renewal, long execution, publish/discard, expiry, and stale state | Pending |
| Non-toolbox worker tests | Python helper and Node adoption | Pending |
| Redaction tests | Arguments, logs, status, errors, and receipts | Pending |
| Full dependent regressions | Backend and UI commands/results | Pending |

The receipt is complete only when every row contains reproducible evidence.
