# Unified hosting configuration and package/environment cutover plan

Status: active breaking-change plan

This plan supersedes the completed toolbox-specific access plan. It defines the
clean cut to one host-owned configuration file and one worker-neutral package
and environment subsystem. Compatibility readers, legacy fallbacks, dual command
names, and automatic migration of old environments are intentionally out of
scope.

## 1. Outcome

After this plan is complete:

- the top-level MP13 configuration owns the physical root map and the logical
  `@hosting`, `@packages`, and `@environments` labels;
- `<config root>/hosting/hosting_config.json` is the only authoritative hosting
  configuration file;
- key material, audit logs, runtime state, caches, and built environments remain
  separate records or data, not additional configuration authorities;
- daemon startup receives the top-level configuration location and resolves the
  hosting configuration locally;
- authenticated control-channel roles authorize package, environment, toolbox,
  and worker administration;
- the daemon computes SHA-256 identities from bytes received at package ingress;
- package acquisition, locking, environment construction, reuse, verification,
  references, and garbage collection are shared by toolboxes and other worker
  kinds;
- toolbox definition and execution APIs remain toolbox-specific, while package
  upload and environment lifecycle APIs become generic; and
- removed configuration fields, commands, state formats, and directory layouts
  fail fast under a new daemon contract version.

## 2. Ownership and repository boundary

This repository owns the daemon, hosting client/channel, CLI, setup library,
configuration contracts, and package/environment implementation.

The dependent project is inspection-only during this work. Do not edit it from
this plan. Record its exact breaking adoption work in
[`HOSTING_CLIENT_BREAKING_CHANGES.md`](HOSTING_CLIENT_BREAKING_CHANGES.md),
including command names, request and response shapes, readiness codes, contract
versions, and removal dates. Acceptance requires a dependent-team receipt that
the recorded cutover was adopted.

Use [`hosting_status.md`](hosting_status.md) as the execution ledger. Every
completed slice records its plan IDs, minimum expertise, production boundary,
tests, negative-path proof, and any dependent handoff change.

This plan, `hosting_status.md`, and `HOSTING_CLIENT_BREAKING_CHANGES.md` are
transient delivery records. Permanent contracts, operator guides, architecture
documents, source comments, and tests must not link to them or treat their text
as normative. A production slice copies the finalized behavior into its owning
permanent document and tests that permanent contract directly.

## 3. Locked design decisions

### 3.1 One configuration authority

The sole authoritative hosting configuration is:

```text
<config root>/hosting/hosting_config.json
```

It replaces `access_control.json`, the short-lived toolbox launcher JSON, and
the five toolbox startup mappings. It contains static control, package, and
environment policy. It does not contain mutable runtime records or individual
toolbox definitions.

The phrase "one configuration file" does not mean "one hosting file." The
following remain separate because they are secrets, logs, or mutable records:

```text
<hosting root>/
  keyring/                 # access-restricted authentication material
  audit/                   # append-oriented audit records
  state/                   # mutable control and operation state
  scratch/                 # incomplete uploads/builds; safe to reap by policy

<packages root>/           # immutable content-addressed package artifacts/locks
<environments root>/       # immutable/reusable built environments and receipts
```

### 3.2 Logical paths first

Extend the existing top-level `category_dirs` path model rather than creating a
second physical-roots system. Add these root keys and anchors:

```json
{
  "category_dirs": {
    "hosting_root_dir": "@home/.mp13-llm/hosting",
    "packages_root_dir": "@home/.mp13-llm/packages",
    "environments_root_dir": "@home/.mp13-llm/environments"
  }
}
```

The corresponding labels are `@hosting`, `@packages`, and `@environments`.
Normal values in `hosting_config.json` use logical references such as
`@packages/artifacts` and `@environments/python`; the daemon resolves them on
the host. Do not make absolute paths the mainstream wire or file contract.

Root definitions may use stable top-level anchors such as `@home`, `@config`,
or `@temp`. Do not base persistent daemon roots on `@project`, because the
working directory differs between direct, service, and SSH-relay launches.

`hosting_config.py` and `hosting_setup_api.py` own the hosting-specific local
operator experience for viewing and customizing these three roots. They must
update the top-level configuration through the shared MP13 configuration/path
library rather than directly inventing another root file. Root relocation is a
host-local administrative operation and is not exposed through the remote
control channel. The project-root `hosting_config.py` remains a thin executable
entry point; configuration behavior belongs in the importable hosting library.

### 3.3 Configuration shape

Freeze a strict `hosting.configuration.v3` contract before implementation. Its
normative structure is:

```json
{
  "contract": "hosting.configuration.v3",
  "control": {
    "authentication": {},
    "roles": {},
    "session_policy": {},
    "audit": {}
  },
  "package_management": {
    "artifact_root": "@packages/artifacts",
    "lock_root": "@packages/locks",
    "sources": {},
    "credentials": {},
    "dependency_policy": {},
    "verification": {
      "hash_algorithm": "sha256"
    }
  },
  "environment_management": {
    "environment_root": "@environments",
    "scratch_root": "@hosting/scratch",
    "retention": {},
    "cache": {}
  }
}
```

The final schema may add explicitly frozen fields, but must not restore the five
startup maps or embed per-toolbox definitions. Credentials are host-local,
access-restricted configuration values or references to the host keyring. Never
place credential values in process arguments, logs, remote status, or receipts.

### 3.4 Authority and integrity

An authenticated control-channel role is the authority to create or change
packages, environments, tools, and toolboxes. A public-key login and a password
login granting the same role have the same administrative authority.

Publisher signatures are not mandatory in the baseline. Remove mandatory
`toolbox_trust_public_keys`, `trust_key_ids`, and signed-manifest requirements.
An external publisher verifier may be introduced later as an optional policy
plugin, but it must not be required by the core package/environment contract.

For every uploaded or acquired artifact, the daemon must:

1. authorize the requested operation from the server-side session role;
2. stream bytes into host-controlled scratch storage with size and policy limits;
3. compute SHA-256 itself from those bytes;
4. atomically promote the completed artifact under its content identity;
5. lock exact artifact identities and dependency decisions; and
6. write an audit event and immutable receipt without secrets.

A caller-supplied hash is only an expectation checked against the daemon result;
it is never accepted as proof of the bytes actually stored.

### 3.5 Shared package and environment model

Use worker-neutral contracts and implementation names:

- `PackageSource`
- `PackagePolicy`
- `EnvironmentTemplate`
- `EnvironmentRequest`
- `EnvironmentLock`
- `EnvironmentReceipt`
- `EnvironmentReference`
- `EnvironmentManager`

Every reference has `consumer_kind`, `consumer_id`, and `revision`. Toolboxes
are one consumer kind, not the owner of the subsystem. Toolbox-specific code
continues to own toolbox definition planning, confirmation, approval,
application, materialization, and execution.

### 3.6 Static policy versus dynamic consumers

Daemon startup reads and validates host-local authentication, package-source,
credential, dependency, retention, and root policy. In this clean cut, changes
to that static policy become effective after a deliberate daemon restart; do
not add an implicit file watcher or a remote configuration editor.

After startup, authorized consumers create and revise tools/toolboxes, upload
packages, manage environment templates, and request environments through the
control channel without restarting the daemon. Those operations write versioned
state and immutable content under the configured roots; they do not rewrite
`hosting_config.json`. Pin the active configuration revision into plans and
operations so a restart onto changed policy makes stale work explicit.

### 3.7 Breaking-cut rules

- Bump the daemon/control contract major version.
- Reject old configuration contracts and old state records with a precise
  operator error.
- Remove old command names instead of translating them.
- Remove old constructor parameters, CLI flags, channel settings, environment
  manager aliases, and directory fallbacks.
- Rebuild environments under the new content-addressed roots. Do not search or
  reuse `toolbox_venvs`, `runtime_envs`, or `toolbox_environment_cache`.
- Keep no `access_control.json` fallback.
- Do not implement dual-read, dual-write, or legacy import code.

## 4. Public API cutover

Freeze the complete request/response schemas in R0. The minimum command mapping
is:

| Removed command family | Replacement | Required generic identity |
|---|---|---|
| `toolbox-artifact-upload-*` | `package-artifact-upload-*` | artifact/upload ID plus authorized session |
| `toolbox-template-*` | `environment-template-*` | template ID plus supported worker/runtime kinds |
| `toolbox-environment-remove` | `environment-remove` | environment identity and reference checks |

The following stay toolbox-specific because they operate on toolbox semantics:

- `toolbox-get-definition`
- `toolbox-plan-definition`
- `toolbox-confirm-definition-plan`
- `toolbox-approve-confirmed-definition-plan`
- `toolbox-apply-definition`
- `toolbox-execute`
- toolbox describe, consistency, gate, reconcile, repair, review, and archive
  operations whose payloads are explicitly toolbox state

Any generic package or environment request originating from a toolbox includes
`consumer_kind="toolbox"`, its stable `consumer_id`, and its definition
`revision`. Future worker adapters use the same requests with their own kind.

Replace toolbox-specific configuration readiness codes with frozen generic
package/environment codes. R0 must specify the exact mapping for at least:

- configuration missing, invalid, or unsupported contract;
- package source or credential unavailable;
- package policy rejection or artifact hash mismatch;
- environment template unavailable or environment build failed; and
- referenced environment busy, retained, or not removable.

## 5. Code navigation map

Use these seams as starting points; search call sites before changing a contract.

| Concern | Start here | Navigation instruction |
|---|---|---|
| Top-level roots and labels | `src/mp13_engine/mp13_config_paths.py` | Extend `DEFAULT_CATEGORY_DIRS`, `CATEGORY_ROOT_KEYS`, `_split_anchor`, `PathResolver._anchor_base`, and `resolve_config_paths` together. |
| Top-level config UI | `src/app/config.py` | Add fields and displayed anchors through the existing section/field model; do not create a second path editor. |
| Other config UI | `src/app/mp13chat.py` | Remove or redirect any duplicate path-editing flow so one shared library owns validation and persistence. |
| Path documentation | `CONFIG.md` | Document label resolution, allowed root anchors, containment rules, and host-local relocation. |
| Hosting config entry point | `hosting_config.py` | Keep this as a thin proxy to the importable implementation; do not place configuration logic in the repository-root script. |
| Hosting setup API | `src/hosting/hosting_setup_api.py` | Evolve `LocalHostingSetupRequest`, plan/apply/inspect/status/reset, and return both logical and local resolved paths where authorized. |
| Hosting setup implementation | `src/hosting/hosting_config_cli.py` | Replace `_hosting_root`, `_default_paths`, `_resolve_paths`, `_write_json`, and directory setup with the shared resolver and strict repository. |
| Setup contract docs | `src/hosting/HOSTING_CONFIG_SCRIPT.md` | Replace the `access_control.json` output layout and document the one-file authority plus record/data directories. |
| Static service configuration | `src/hosting/service/host_service.py` | Replace the five toolbox constructor maps with one normalized configuration object assembled before service construction. |
| Default control path | `src/hosting/service/constants.py` | Remove the hardcoded `access_control.json` authority and derive paths from resolved configuration. |
| Launcher configuration | `src/hosting/daemon/toolbox_launch_config.py` | Delete this module after callers use the top-level config location. Do not retain a compatibility wrapper. |
| Foreground/background startup | `src/hosting/daemon/foreground.py`, `src/hosting/daemon/background.py` | Remove the five mappings, ephemeral launcher file, and `--toolbox-config-file`; accept the top-level config location only. |
| Transport/demo bootstrap | `src/hosting/transport_bootstrap_api.py`, `src/app/hosted_chat_demo.py` | Replace derived `access_control.json` paths and forward the top-level config location through the same bootstrap contract. |
| Local daemon and dispatch | `src/hosting/daemon/local_ipc.py` | Load normalized configuration once, rename generic commands, authorize server-side, and expose sanitized status. |
| Client startup forwarding | `src/hosting/engine_host_channel.py` | Remove `_daemon_toolbox_launch_kwargs` and old settings; pass only the top-level config location to locally owned bootstrap. |
| Client CLI forwarding | `src/hosting/engine_host_cli.py` | Remove `engine_host_toolbox_config_file` and old flags/help. Add only the agreed top-level config option if needed. |
| Package source model | `src/hosting/toolbox/host_project_config.py` | Extract generic source/config types and remove toolbox ownership and mandatory signing fields. |
| Dependency policy | `src/hosting/toolbox/dependency_policy.py` | Move policy into the generic package subsystem without changing enforcement semantics accidentally. |
| Artifact stores and upload services | `src/hosting/service/toolbox_artifact_store.py`, `src/hosting/service/toolbox_artifact_uploads.py`, `src/hosting/service/toolbox_artifact_upload_service.py` | Extract generic ingress/storage services, daemon hashing, operation kinds, and secret-free receipts before renaming public commands. |
| Catalog/config resolution | `src/hosting/service/toolbox_catalog.py`, `src/hosting/service/toolbox_definition_resolution.py`, `src/hosting/service/toolbox_host_config_state.py` | Keep toolbox definition semantics but replace host-project configuration and trust-key assumptions with package locks and the active configuration revision. |
| Environment manager | `src/hosting/toolbox/environment.py` | Replace `ToolboxEnvironmentManager` and the `RuntimeEnvironmentManager` compatibility subclass with one neutral manager. |
| Hermetic builder | `src/hosting/toolbox/hermetic_environment.py` | Rename toolbox contracts, use resolved generic roots, and emit generic locks/receipts/references. |
| Runtime consumers | `src/hosting/sandbox/runtime_base.py`, `src/hosting/sandbox/python_runtime.py`, `src/hosting/sandbox/js_runtime.py` | Preserve neutral runtime identity concepts and adapt each worker/runtime through the generic manager. |
| Serialized environment identity | `src/hosting/toolbox/bundle_models.py`, `src/hosting/sandbox/toolbox_runtime.py` | Remove legacy `environment_root_kind` defaults and carry generic environment identities/references instead of directory-kind routing. |
| Toolbox orchestration | `src/hosting/toolbox/orchestration.py`, `src/hosting/toolbox/staging.py` | Translate toolbox plans into exact package locks and environment requests. |
| Service materialization/runtime | `src/hosting/service/toolbox_materialization.py`, `src/hosting/service/toolbox_runtime.py`, `src/hosting/service/proxy.py` | Keep toolbox behavior while replacing environment ownership and old command/config assumptions. |
| Hosting state/health | `src/hosting/service/state.py` | Replace `access_control.json` existence checks with unified configuration health and keep mutable state separate from configuration. |
| Existing contract/docs | `src/hosting/HOSTED_TOOLBOX_CONTRACT.md`, `src/hosting/HOSTING_ACCESS.md`, `src/hosting/ENGINE_HOST_CLI.md`, `src/hosting/sandbox/SANDBOX_ARCHITECTURE.md` | Remove launcher maps, mandatory signing, old roots, and compatibility statements when their owning implementation slice lands. |

Before editing, use `rg` to find every import, constructor call, command string,
readiness code, legacy directory, and serialized contract named by the work item.
The table identifies entry points, not the complete change radius.

### 5.1 Permanent documentation cutover

Update permanent documentation with the production slice that makes the
behavior true. Do not update it early to describe planned behavior, and do not
add backlinks to the transient delivery records.

| Permanent document | Owning work | Required final content |
|---|---|---|
| `CONFIG.md` | R2 | Top-level root keys/labels, resolution, containment, and host-local customization. |
| `src/hosting/HOSTING_CONFIG_SCRIPT.md` | R2–R3 | One configuration authority, logical/resolved status, safe local apply/recovery, and record/data layout. |
| `src/hosting/HOSTING_ACCESS.md` | R3, R5–R6, R9 | Authentication/role authority, credential handling, package ingress hashing, generic operations, audit, and redaction. |
| `src/hosting/HOSTED_TOOLBOX_CONTRACT.md` | R5–R7 | Toolbox planning/execution over generic package locks, environment requests/references, and retained toolbox APIs. |
| `src/hosting/ENGINE_HOST_CLI.md` | R4, R9 | Single startup configuration input and new generic package/environment commands; no removed flags. |
| `src/hosting/sandbox/SANDBOX_ARCHITECTURE.md` | R6–R8 | Worker-neutral environment ownership, content keys, runtime adapters, references, retention, and cleanup. |

R1.5 must discover additional permanent READMEs, examples, generated help, or
API references and assign each to an owning production slice.

## 6. Priority and expertise sequence

The sequence is deliberately monotonic. Do not interleave a lower-expertise
slice after a higher-expertise block has begun. A slice touching mixed work is
classified at the highest expertise it requires.

| Continuous block | Priority | Minimum expertise | Work items | Exit condition |
|---|---:|---|---|---|
| A — contract freeze | P0 | average | R0.1–R0.7 | All breaking contracts and removals are explicit and internally consistent. |
| B — handoff and inventory | P0 | medium | R1.1–R1.5 | Implementers and dependent consumers have exact navigation and adoption instructions. |
| C — implementation and acceptance | P0–P2 | high | R2.1–R9.8 | Clean-cut code, tests, docs, receipts, and consumer adoption are complete. |

Priority is handled inside the continuous high block: complete all P0 items,
then P1, then P2. Do not move a convenient P2 cleanup ahead of a P0 contract or
security dependency merely to make smaller commits.

## 7. Slice and evidence discipline

Each implementation commit is a declared slice.

1. Select one or more tightly coupled, consecutive plan IDs with the same
   minimum expertise.
2. Record the slice ID, priority, expertise, production boundary, expected
   removals, tests, and negative paths in `hosting_status.md`.
3. Inspect all callers and serialized forms before changing the boundary.
4. Change production code, tests, documentation, handoff, status evidence, and
   this plan's checkboxes in the same slice when they are part of one contract.
5. Mark an item complete only after its proof commands pass. Leave failed or
   partially implemented items unchecked.
6. End a slice when the required expertise changes.
7. Use a concise commit subject describing the completed behavior, not the
   files touched.

For removals, evidence must include an `rg` zero-result check over production,
tests, and docs. For security boundaries, include at least one denial test and
one proof that secrets or host-only paths are absent from remote output.

## 8. Work items

### Block A — contract freeze (`average`, P0)

#### R0.1 Freeze file ownership and directory layout

- [x] Specify `hosting.configuration.v3` as the only static hosting authority.
- [x] Classify each path as configuration, secret material, audit, mutable
  state, scratch, immutable package content, or built environment content.
- [x] State which process may write each class. The daemon reads static config;
  only the local hosting setup/config library writes it.
- [x] Record the exact old files and directories that become invalid.

Proof: the frozen contract and layout are stated consistently in this plan and
`HOSTING_CLIENT_BREAKING_CHANGES.md`; permanent setup documentation remains
owned by R2–R3 and is not updated ahead of shipped behavior.

#### R0.2 Freeze root-label semantics

- [x] Define `hosting_root_dir`, `packages_root_dir`, and
  `environments_root_dir` in the existing top-level `category_dirs` contract.
- [x] Define `@hosting`, `@packages`, and `@environments`, including allowed
  nesting, normalization, containment, and cycle rejection.
- [x] Define which stable anchors may appear in root definitions and prohibit
  persistent roots based on `@project`.
- [x] Define the authorized local status shape containing logical and resolved
  paths and the sanitized remote status shape.

Proof: table-driven examples cover Windows and POSIX syntax, traversal,
unknown labels, cycles, and roots outside an allowed containment policy.

#### R0.3 Freeze authority and artifact identity

- [x] Define the minimum role for source management, credential management,
  package upload, environment template changes, environment removal, and GC.
- [x] Define daemon-computed SHA-256 as the canonical artifact identity.
- [x] Define caller-supplied hashes as optional expectations only.
- [x] Remove publisher-key and signed-manifest requirements from the baseline
  contract; name the extension point for a future optional verifier.
- [x] Define audit fields without tokens, passwords, credential values, or
  unrestricted local paths.

Proof: the authorization matrix includes positive and negative cases for both
password and public-key sessions that grant the same role.

#### R0.4 Freeze generic package/environment contracts

- [x] Specify every neutral type listed in section 3.5 and its serialized
  contract/version.
- [x] Require `consumer_kind`, `consumer_id`, and `revision` on references.
- [x] Define immutable package locks, environment locks, build receipts,
  reference lifecycle, and content-address keys.
- [x] Define restart, retry, cancellation, incomplete-build cleanup, and
  concurrent-build behavior.

Proof: examples show one toolbox consumer and at least one non-toolbox worker
consumer resolving the same package and environment contracts.

#### R0.5 Freeze command and readiness cutover

- [x] Enumerate every retained, renamed, and removed control command.
- [x] Specify exact request, success, error, cached-result, and status payloads.
- [x] Specify the replacement for every `toolbox_configuration_*` readiness
  code and any dependent UI state derived from it.
- [x] State the new daemon contract major and rejection response for an old
  client or command.

Proof: a machine-readable or table-driven command manifest has no aliases and
every removed command has exactly one disposition.

#### R0.6 Freeze clean-cut state behavior

- [x] Inventory old configuration, operation, upload, template, environment,
  reference, and receipt contract identifiers.
- [x] Specify fail-fast behavior for each old format.
- [x] Require environment rebuild under the generic roots and prohibit legacy
  discovery or reuse.
- [x] Define operator cleanup instructions separately from daemon behavior.

Proof: no item says "fallback," "try old," "migrate automatically," or
"translate" as an implementation requirement.

#### R0.7 Freeze host-local root customization

- [x] Define plan/apply/inspect/status/reset payloads for changing the three
  top-level roots through the hosting setup library.
- [x] Define preflight checks for permissions, collisions, free space,
  non-empty destinations, daemon activity, and cross-volume moves.
- [x] Use a local journal and idempotent recovery for a change spanning the
  top-level configuration and hosting configuration. Do not claim a
  cross-filesystem atomic rename.
- [x] Prohibit remote control-channel root relocation.

Proof: interruption points and recovery outcomes are enumerated before code is
changed.

### Block B — handoff and inventory (`medium`, P0)

#### R1.1 Produce the exact dependent-client handoff

- [x] Update `HOSTING_CLIENT_BREAKING_CHANGES.md` with the command manifest,
  payloads, readiness mapping, version negotiation, and old-name removals.
- [x] Include client session metadata requirements: cached authentication must
  preserve token, role, auth method, scope, and key ID without repeating the
  handshake.
- [x] Give dependent implementers stable navigation/search terms rather than
  relying only on current line numbers.

Proof: another engineer can implement the client cut without reading daemon
internals or inferring a payload.

#### R1.2 Inventory the production change radius

- [x] Search all production callers of `toolbox_host_project_configuration`,
  `toolbox_artifact_sources`, `toolbox_trust_public_keys`,
  `toolbox_source_credentials`, `toolbox_dependency_policy`,
  `--toolbox-config-file`, and `engine_host_toolbox_config_file`.
- [x] Search imports and instantiations of toolbox-specific package policy,
  environment managers, builders, receipts, references, and legacy paths.
- [x] Search command strings, dispatch tables, capability declarations,
  readiness codes, audit actions, and operation kinds.
- [x] Record the file list and ownership per R2–R9 in `hosting_status.md`.

Proof: the inventory includes indirect re-exports and documentation examples,
not only class definitions.

#### R1.3 Inventory tests and fixtures

- [x] Assign existing unit, integration, daemon, channel, CLI, setup, toolbox,
  sandbox, workflow, and native-platform tests to R2–R9.
- [x] List fixtures serialized under removed contracts and decide whether each
  becomes a new fixture or a deliberate rejection fixture.
- [x] Identify tests that accidentally pass because of default home-directory
  state and make their roots explicit.

Proof: every production boundary in R2–R9 has a named proof location and at
least one negative-path test target.

#### R1.4 Inventory the dependent repository without editing it

- [x] Locate direct command strings, readiness codes, authentication result
  narrowing, and package/environment response assumptions.
- [x] Record exact files and stable symbols in the breaking-change handoff.
- [x] Identify owner and adoption evidence required for final acceptance.

Proof: the inspection records all known affected dependent surfaces while the
dependent worktree remains unchanged.

#### R1.5 Prepare the documentation cutover map

- [x] Assign all hosting startup, configuration, security, toolbox, and worker
  documentation to an implementation work item.
- [x] List examples that expose credentials, absolute mainstream paths, old
  flags, mandatory signing, or toolbox-owned environments.
- [x] Define the final `rg` removal patterns used by R9.
- [x] Remove permanent-document backlinks and tests that treat this plan,
  `hosting_status.md`, or `HOSTING_CLIENT_BREAKING_CHANGES.md` as normative.

Proof: no known user-facing document is left without an owning work item.

### Block C — implementation and acceptance (`high`)

#### R2 — Extend the shared path/configuration foundation (`high`, P0)

##### R2.1 Add and resolve the three top-level roots

- [x] Extend `DEFAULT_CATEGORY_DIRS`, `CATEGORY_ROOT_KEYS`, anchor parsing, and
  `PathResolver` in `src/mp13_engine/mp13_config_paths.py` as one contract.
- [x] Reject unknown labels, cycles, traversal escapes, invalid root types, and
  ambiguous self-reference.
- [x] Preserve logical values during load/save; resolve only at an explicit
  host boundary.

Proof: table-driven resolver tests cover default/custom roots, nested labels,
Windows/POSIX forms, normalization, cycles, and traversal.

##### R2.2 Give hosting setup ownership of root customization

- [x] Add the three root fields to the shared config model/UI in
  `src/app/config.py`.
- [x] Make `hosting_setup_api.py` expose hosting-focused plan/apply/inspect/
  status/reset operations that call the shared writer.
- [x] Remove or redirect duplicate editing in `src/app/mp13chat.py`.
- [x] Return logical refs by default; return resolved paths only to a local
  authorized operator surface.

Proof: round-trip tests show both the general config UI and hosting setup use
one validator and preserve unrelated top-level configuration.

##### R2.3 Implement safe multi-file local apply

- [x] Add locked, atomic-per-file writes with restrictive permissions and
  fsync/replace semantics appropriate to the platform.
- [x] Journal the two-authority update when both top-level roots and
  `hosting_config.json` change; make retry/recovery idempotent.
- [x] Refuse active-daemon relocation and unsafe/non-empty destinations unless
  the frozen local plan explicitly permits them.

Proof: fault-injection tests interrupt every journal phase and recover to one
declared state without losing unrelated configuration.

#### R3 — Implement the unified hosting configuration (`high`, P0)

##### R3.1 Add strict models and repository

- [x] Implement `hosting.configuration.v3` models for `control`,
  `package_management`, and `environment_management`.
- [x] Reject unknown security-sensitive keys, wrong types, unresolved labels,
  unsupported versions, and credential-policy conflicts.
- [x] Make one repository exclusively responsible for locked reads and local
  setup writes of `hosting_config.json`.

Proof: schema/model tests cover valid minimal/full files and each rejection
class without leaking rejected secret values.

##### R3.2 Replace `access_control.json`

- [ ] Move static authentication, role, session, and audit policy under
  `control` while keeping keyring/audit/state records separate.
- [ ] Change service constants and setup defaults to resolve the new file.
- [ ] Remove every reader, writer, fixture, and document that treats
  `access_control.json` as an authority.

Proof: clean startup succeeds with only `hosting_config.json`; startup fails
precisely when only the old file exists; `rg` finds no production fallback.

##### R3.3 Add sanitized inspection and health

- [ ] Report contract version, logical root refs, configuration health, source
  availability, and environment subsystem health.
- [ ] Restrict local resolved paths to local administrative inspection.
- [ ] Redact credential values, token material, key material, sensitive query
  strings, and unrestricted host paths from remote status and errors.

Proof: snapshot tests use sentinel secrets and local paths and assert they do
not appear in remote status, logs, audit messages, or exception strings.

#### R4 — Cut daemon startup to the single configuration path (`high`, P0)

##### R4.1 Replace startup inputs

- [ ] Make foreground, background, service, CLI, and locally owned channel
  bootstrap accept only the top-level MP13 configuration location needed to
  resolve `@hosting/hosting_config.json`.
- [ ] Load and validate configuration before binding externally reachable
  listeners or accepting control requests.
- [ ] Pass one normalized immutable configuration object into `EngineHostService`.
- [ ] Capture a stable configuration revision in long-running plans/operations;
  reject their continuation after a restart under incompatible changed policy.

Proof: direct, background, service, and SSH-relay-equivalent launch tests all
resolve the same logical configuration without credential-bearing arguments.

##### R4.2 Delete launcher configuration and five mappings

- [ ] Remove the five `toolbox_*` constructor/settings mappings.
- [ ] Delete `src/hosting/daemon/toolbox_launch_config.py`.
- [ ] Remove `--toolbox-config-file`, `engine_host_toolbox_config_file`,
  ephemeral launch JSON creation, conflict handling, help text, and fixtures.
- [ ] Do not leave deprecated parameters or permissive `**kwargs` sinks.

Proof: signature tests reject old arguments and `rg` returns zero relevant
results in production, tests, and docs.

##### R4.3 Generalize startup readiness

- [ ] Separate control readiness from package/environment configuration health.
- [ ] Emit the exact R0 readiness codes and contract version.
- [ ] Keep authentication usable for authorized diagnosis when a non-control
  package/environment subsystem is unhealthy, if the frozen policy permits it.

Proof: readiness tests distinguish missing/invalid control, package, source,
credential, template, and environment states.

#### R5 — Build the generic package subsystem (`high`, P0)

##### R5.1 Extract neutral package contracts

- [x] Move source and dependency policy models out of toolbox ownership into a
  generic package module.
- [x] Rename public and serialized types to `PackageSource` and `PackagePolicy`.
- [x] Remove mandatory trust-key IDs and signed-manifest validation from the
  base path while preserving an optional verifier interface.

Proof: package model tests have no toolbox dependency and demonstrate the
optional verifier is absent/disabled by default.

##### R5.2 Implement authorized content-addressed ingress

- [x] Authorize begin/chunk/status/commit/cancel from the server-side session.
- [x] Bound size, chunk order, concurrency, timeout, and scratch allocation.
- [x] Compute SHA-256 during daemon-owned ingress, compare any caller
  expectation, and atomically promote only a complete artifact.
- [x] Make retries idempotent and quarantine/remove incomplete or mismatched
  uploads without making them resolvable.

Proof: tests cover permission denial, disconnect, reordered/duplicate chunks,
oversize input, expected-hash mismatch, concurrent commit, restart, and retry.

##### R5.3 Implement sources, credentials, policy, and locks

- [x] Resolve source and credential configuration locally without serializing
  secrets into operations or receipts.
- [x] Enforce source allowlists, dependency policy, platform/runtime targets,
  and exact artifact selection before an environment build.
- [x] Persist immutable package locks containing daemon-computed identities and
  reproducible source metadata stripped of secrets.

Proof: resolution tests cover allowed/denied sources, missing credentials,
dependency conflicts, deterministic lock output, and offline reuse.

##### R5.4 Cut the public package commands

- [x] Replace `toolbox-artifact-upload-*` dispatch/channel/CLI/API surfaces with
  `package-artifact-upload-*` and the frozen payloads.
- [x] Advertise only the new commands in capabilities/version negotiation.
- [x] Use generic operation/audit kinds and reject the old command family.

Proof: no-double tests prove exactly one authorized ingress effect and old
commands fail with the frozen version/unknown-command response.

#### R6 — Build the generic environment subsystem (`high`, P0)

##### R6.1 Replace toolbox-owned manager and builder types

- [x] Implement the neutral environment contracts from R0.
- [ ] Replace `ToolboxEnvironmentManager`, the compatibility-only
  `RuntimeEnvironmentManager` subclass, and hermetic toolbox names with one
  `EnvironmentManager` and generic builders.
- [x] Keep runtime-specific mechanics behind adapters rather than in the shared
  contract.

Proof: imports and type tests show toolbox and non-toolbox consumers depend on
the neutral interface; old aliases are absent.

##### R6.2 Use generic roots and content keys

- [x] Store reusable environments under resolved `@environments` roots and
  transient builds under `@hosting/scratch`.
- [x] Derive keys from runtime/builder identity, platform, exact package lock,
  environment template revision, and relevant policy inputs.
- [x] Publish a receipt only after validation and atomic promotion.

Proof: same inputs reuse one valid environment; changed package/template/
runtime/platform inputs produce distinct identities; incomplete builds do not.

##### R6.3 Add generic references and concurrency control

- [x] Store references with `consumer_kind`, `consumer_id`, and `revision`.
- [x] Serialize builders per environment key while allowing unrelated builds.
- [x] Prevent removal/GC of referenced or active environments and make release
  idempotent.

Proof: concurrent toolbox and non-toolbox tests cover build coalescing,
reference acquisition/release, active execution, removal denial, and restart.

##### R6.4 Cut generic template and environment commands

- [x] Replace `toolbox-template-*` with `environment-template-*`.
- [x] Replace `toolbox-environment-remove` with `environment-remove`.
- [ ] Generalize operation kinds, audit events, receipts, dispatch, channel,
  CLI, status, and capability declarations.

Proof: old commands and serialized toolbox environment kinds are rejected and
new commands preserve role checks and no-double semantics.

##### R6.5 Remove legacy roots and fallback behavior

- [ ] Remove discovery/use of `toolbox_venvs`, `runtime_envs`, and
  `toolbox_environment_cache`.
- [ ] Remove legacy receipt/reference readers and compatibility aliases.
- [ ] Provide explicit local operator cleanup instructions; do not delete old
  data automatically as a side effect of daemon startup.

Proof: a fixture containing only legacy directories cannot affect resolution,
reuse, references, GC, or execution.

#### R7 — Adopt the shared subsystem in toolbox flows (`high`, P1)

##### R7.1 Translate toolbox plans into generic requests

- [ ] Make toolbox definition planning resolve exact package locks and an
  `EnvironmentRequest` through the generic subsystem.
- [ ] Carry package/environment identities through confirmation, approval, and
  apply so approved bytes cannot change between stages.
- [ ] Attach/release the toolbox reference transactionally with definition
  revision activation.

Proof: mutation-between-plan-and-apply, stale approval, restart, retry, and
concurrent revision tests cannot execute unapproved bytes or leak references.

##### R7.2 Adapt materialization and execution

- [ ] Make materialization consume an immutable environment receipt/reference
  rather than constructing a toolbox-owned venv.
- [ ] Preserve sandbox, proxy, tool exposure, runtime selection, and execution
  constraints while changing environment ownership.
- [ ] Keep toolbox logical commands and response semantics frozen in R0.

Proof: end-to-end toolbox tests cover definition through execution for cached
and newly built environments with no duplicate build or execution effect.

##### R7.3 Adapt maintenance operations

- [ ] Update toolbox consistency, gate, reconcile, repair, review snapshot,
  references, GC, and archive behavior to call generic package/environment
  operations where appropriate.
- [ ] Keep toolbox-only state and generic shared state in separately versioned
  repositories with explicit cross-references.

Proof: maintenance tests cannot remove shared content still referenced by a
toolbox or another worker kind.

#### R8 — Complete worker-neutral state and operations (`high`, P1)

##### R8.1 Version repositories and reject old state

- [x] Version generic package, environment template, lock, receipt, reference,
  upload, and operation repositories.
- [x] Reject old toolbox-owned serialized records with the R0 operator message.
- [x] Ensure repository writes are locked, atomic, bounded, and crash-recoverable.

Proof: corruption, truncation, unsupported-version, interrupted-write, and
concurrent-writer tests preserve the last valid state or fail closed.

##### R8.2 Adopt the shared manager in existing worker runtimes

- [ ] Integrate the existing Python workflow helper and JavaScript/Node workflow
  runtime with `EnvironmentManager` without adding their mechanics to the
  shared manager.
- [ ] Use stable `workflow_python_helper` and `workflow_js_node` consumer kinds
  (or the exact R0 replacements) with consumer ID and revision.
- [ ] Exercise source resolution, package lock, build/reuse, references,
  execution handoff, and release for both worker kinds.

Proof: cross-consumer tests demonstrate shared reuse where inputs match and
independent retention where toolbox, Python helper, and Node references differ.

##### R8.3 Harden maintenance and resource controls

- [x] Apply quotas and bounded listing/pagination to package/environment state.
- [x] Make GC mark from all consumer references before sweeping and re-check
  activity under lock before deletion.
- [ ] Make repair observational by default and explicitly authorized for any
  mutation.

Proof: stress and adversarial tests cover large state, concurrent create/GC,
active executions, stale scratch, and partial artifacts without broad deletion.

#### R9 — Finish public surfaces, acceptance, and handoff (`high`, P2)

##### R9.1 Complete channel, CLI, and capability surfaces

- [ ] Align every public method signature, CLI option, help example, capability,
  request, response, and error with the R0 manifest.
- [ ] Preserve complete authentication results through fresh and cached paths:
  token, role, auth method, scope, and key ID.
- [ ] Remove annotations or conversions that narrow structured results to a
  token string.

Proof: channel/API contract tests compare fresh and cached session metadata and
exercise an admin-only call without another handshake.

##### R9.2 Run cross-boundary security acceptance

- [ ] Prove lower roles cannot manage sources, credentials, packages,
  templates, environments, references, or GC.
- [ ] Prove equal roles granted by password and public key receive equal policy.
- [ ] Prove caller hashes cannot substitute for daemon-computed hashes.
- [ ] Prove secrets and restricted local paths are absent from arguments,
  process inspection fixtures, logs, audit, receipts, errors, and remote status.

Proof: record the exact security suite and negative-path results in
`hosting_status.md`.

##### R9.3 Run lifecycle and no-double acceptance

- [ ] Exercise direct, background, service, and relay-equivalent startup.
- [ ] Exercise package upload, source resolution, environment build/reuse,
  template replacement, reference release, removal, repair, and GC across
  disconnect/retry/restart boundaries.
- [ ] Exercise toolbox and second-worker flows concurrently.
- [ ] Prove one authorized logical request causes at most one durable mutation
  or execution effect.

Proof: integration tests include stable operation IDs and receipt identities
that demonstrate idempotence rather than relying only on response counts.

##### R9.4 Remove all legacy surfaces

- [ ] Run the R1 removal searches over production, tests, docs, and examples.
- [ ] Remove old fields, commands, aliases, readiness codes, contract IDs,
  filenames, directory fallbacks, and mandatory-signing language.
- [ ] Inspect remaining matches individually and document any historical-only
  occurrence that is intentionally retained.

Proof: `rg` zero-result evidence is recorded for all prohibited runtime and API
terms.

##### R9.5 Update operator and developer documentation

- [ ] Update `CONFIG.md`, `HOSTING_CONFIG_SCRIPT.md`, startup/CLI docs, security
  model, package/environment lifecycle, toolbox docs, and worker integration
  guidance.
- [ ] Show logical refs as the normal configuration form and clearly separate
  configuration, secrets, audit, state, scratch, packages, and environments.
- [ ] Document explicit cleanup/rebuild of legacy environments without adding
  fallback code.
- [ ] Verify permanent documents contain no links to or normative dependency on
  the three transient delivery records.

Proof: all executable examples are run or covered by doc tests and contain no
credential values in command arguments.

##### R9.6 Complete dependent adoption

- [ ] Deliver the final `HOSTING_CLIENT_BREAKING_CHANGES.md` to the dependent
  team and record owner/receipt.
- [ ] Validate the dependent client preserves structured authentication
  metadata and adopts the new commands/readiness codes/version.
- [ ] Do not close this item based only on a daemon-side compatibility shim.

Proof: record the dependent revision/test evidence in `hosting_status.md` while
keeping this repository's ownership boundary intact.

##### R9.7 Run the full repository matrix

- [ ] Run focused unit and integration suites for each R2–R8 boundary.
- [ ] Run the repository's required aggregate test, lint, type, and platform
  lanes, including native lanes where the affected runtime requires them.
- [ ] Record skipped lanes with an owner and blocking reason; a skipped required
  lane is not acceptance.

Proof: commands, environment, results, and relevant artifact paths are recorded
in `hosting_status.md`.

##### R9.8 Close the plan

- [ ] Confirm every checkbox is backed by evidence and every failed item remains
  open.
- [ ] Confirm public docs, handoff, capability version, and configuration schema
  agree exactly.
- [ ] Confirm no unplanned compatibility or fallback code was introduced.
- [ ] Change this plan status to complete only after R9.6 and R9.7 pass.

## 9. Final acceptance criteria

- [ ] One top-level path map owns `@hosting`, `@packages`, and `@environments`.
- [ ] One `hosting_config.json` owns all static hosting/control/package/
  environment configuration.
- [ ] The hosting setup library safely customizes roots and exclusively writes
  hosting configuration; the daemon only reads it.
- [ ] Daemon startup accepts no toolbox mappings or credential-bearing launcher
  file/arguments.
- [ ] Server-side roles authorize all mutations and daemon-computed SHA-256
  identifies stored artifacts.
- [ ] Mandatory publisher signing is absent from the baseline.
- [ ] Toolboxes and at least one other worker kind use the same neutral package
  and environment contracts.
- [ ] Generic commands replace toolbox-owned upload/template/environment
  lifecycle commands without aliases.
- [ ] Old configuration, state, environment roots, and clients fail fast under
  the new major contract.
- [ ] Remote responses, logs, audit, and receipts expose neither secrets nor
  unrestricted host-local paths.
- [ ] Retry, restart, and concurrency tests prove no duplicate durable or
  execution effects.
- [ ] The dependent consumer has adopted the exact breaking handoff.
- [ ] Full required test/platform evidence is recorded in `hosting_status.md`.
- [ ] Permanent docs describe shipped behavior and do not reference transient
  plan, status, or breaking-change ledgers.

## 10. Explicitly out of scope

- compatibility readers, shims, aliases, dual commands, or automatic legacy
  environment reuse/migration;
- editing the dependent repository from this plan;
- distributing publisher private keys or requiring publisher signatures in the
  baseline;
- remote control-channel relocation of host roots;
- embedding individual toolbox definitions in `hosting_config.json`;
- accepting caller-supplied hashes without independently hashing received bytes;
- exposing credential values through CLI arguments or remote inspection; and
- claiming transactionality across filesystems without a journaled recovery
  protocol.
