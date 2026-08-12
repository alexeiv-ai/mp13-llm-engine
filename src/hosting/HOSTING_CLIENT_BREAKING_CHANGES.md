# Unified hosting client breaking changes

Status: reset for the unified hosting configuration and package/environment cut

This is the active handoff for the breaking plan in
[`hosting_access_plan.md`](hosting_access_plan.md). The previous toolbox rollout
handoff and adoption receipts are historical and remain available in Git. They
do not demonstrate adoption of this cut.

The product is unreleased. Parent implementation removes the old surface in the
same slices that add its replacement. There is no compatibility adapter,
deprecated alias, dual-read period, or automatic legacy environment migration.

## 1. Handoff gate

Do not begin a client-visible parent implementation slice until its exact
request, response, error, capability, and version contract is frozen here.
Directional names in this reset are not permission to infer missing payloads.

- [ ] Record the new daemon/control contract major version.
- [ ] Publish a complete retained/renamed/removed command manifest.
- [ ] Publish exact request and response schemas for every renamed command.
- [ ] Publish generic readiness codes and the old-to-new disposition table.
- [ ] Publish startup/configuration argument signatures.
- [ ] Publish operation kinds, selectors, progress phases, and receipts.
- [ ] Publish state/receipt contract versions and old-version rejection errors.
- [ ] Record the parent implementation pin.
- [ ] Record each dependent owner, revision, test receipt, and adoption status.

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
to resolve the hosting configuration locally. The exact Python parameter, CLI
flag, and control-setting names remain gated by R0.5 and must be inserted here
before implementation.

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
- [ ] If the dependent owns local daemon bootstrap, adopt the exact new
  top-level configuration argument after R0.5 freezes it.

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
| `toolbox-artifact-upload-begin` | `package-artifact-upload-begin` | Pending R0.5 |
| `toolbox-artifact-upload-chunk` | `package-artifact-upload-chunk` | Pending R0.5 |
| `toolbox-artifact-upload-status` | `package-artifact-upload-status` | Pending R0.5 |
| `toolbox-artifact-upload-cancel` | `package-artifact-upload-cancel` | Pending R0.5 |
| `toolbox-artifact-upload-commit` | `package-artifact-upload-commit` | Pending R0.5 |
| `toolbox-template-*` | `environment-template-*` | Pending R0.5 |
| `toolbox-environment-remove` | `environment-remove` | Pending R0.5 |

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
  GC, and archive commands whose exact disposition is recorded by R0.5

Retaining a name does not guarantee that every nested package/environment
field remains unchanged. R0.5 must publish any nested identity, receipt,
operation, or readiness changes before the owning parent slice begins.

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

R0.5 must replace this section with an exact mapping covering:

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

Plans and operations pin the active hosting configuration revision. A restart
under changed static policy makes incompatible pending work stale; clients
must re-plan rather than silently continuing under new policy.

The exact durable operation kinds, selectors, phases, retry rules, cancellation
rules, and receipts for renamed generic commands are pending R0.4/R0.5.

## 9. State and local data cut

- Old configuration and serialized package/environment contracts fail fast.
- Old environment directories are ignored by resolution, reuse, references,
  removal, and GC.
- Operator cleanup is explicit and local; daemon startup does not broadly
  delete legacy data.
- Keyring, audit, mutable state, scratch, immutable packages, and environments
  remain separate records/data even though static configuration has one
  authority.

R0.6 must add the exact old contract identifiers, new contract identifiers,
and stable rejection errors here before their implementation slice begins.

## 10. Adoption receipt

| Field | Required value | Current value |
|---|---|---|
| Parent contract major | Exact version | Pending R0.5 |
| Parent implementation pin | Full commit | Pending |
| Dependent owner | Team/person | Pending |
| Dependent revision | Full commit | Pending |
| Configuration/startup tests | Commands and result | Pending |
| Authentication metadata tests | Fresh, cached, admin-only result | Pending |
| Package command tests | New commands; old commands rejected | Pending |
| Environment command tests | New commands; old commands rejected | Pending |
| Readiness/capability tests | Exact new codes/version | Pending |
| Toolbox lifecycle tests | Plan through execute/retry/restart | Pending |
| Non-toolbox worker tests | Python helper and Node adoption | Pending |
| Secret/path redaction tests | Arguments/logs/status/errors/receipts | Pending |

The receipt is complete only when all rows identify reproducible evidence.
Prior toolbox-rollout receipts do not fill these rows.
