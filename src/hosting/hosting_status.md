# Unified hosting cutover status

Last updated: 2026-08-11

Status: active; contract freeze not started

This is the fresh execution ledger for
[`hosting_access_plan.md`](hosting_access_plan.md). The prior toolbox corrective
ledger and its test transcript remain available in Git history. Their completed
work may be useful implementation foundation, but it is not evidence that the
new unified configuration, generic APIs, state cut, or dependent adoption is
complete.

## 1. Current gate

Current continuous block: A — contract freeze (`average`, P0)

Active slice: none

Implementation is gated on R0.1–R0.7 and the exact client-visible portions of
[`HOSTING_CLIENT_BREAKING_CHANGES.md`](HOSTING_CLIENT_BREAKING_CHANGES.md).
This is a planned sequencing gate, not a technical blocker.

## 2. Progress ledger

| Work | Priority | Expertise | Status | Evidence |
|---|---:|---|---|---|
| R0.1 File ownership/layout | P0 | average | Not started | — |
| R0.2 Root-label semantics | P0 | average | Not started | — |
| R0.3 Authority/artifact identity | P0 | average | Not started | — |
| R0.4 Generic contracts | P0 | average | Not started | — |
| R0.5 Commands/readiness/version | P0 | average | Not started | — |
| R0.6 Clean-cut state behavior | P0 | average | Not started | — |
| R0.7 Host-local root customization | P0 | average | Not started | — |
| R1.1 Exact dependent handoff | P0 | medium | Not started | — |
| R1.2 Production inventory | P0 | medium | Not started | — |
| R1.3 Tests/fixtures inventory | P0 | medium | Not started | — |
| R1.4 Dependent read-only inventory | P0 | medium | Not started | — |
| R1.5 Documentation cutover map | P0 | medium | Not started | — |
| R2 Shared paths/config foundation | P0 | high | Not started | — |
| R3 Unified hosting configuration | P0 | high | Not started | — |
| R4 Single-path daemon startup | P0 | high | Not started | — |
| R5 Generic package subsystem | P0 | high | Not started | — |
| R6 Generic environment subsystem | P0 | high | Not started | — |
| R7 Toolbox adoption | P1 | high | Not started | — |
| R8 Worker-neutral state/operations | P1 | high | Not started | — |
| R9 Public surfaces/acceptance/handoff | P2 | high | Not started | — |

Do not mark a row complete merely because an older toolbox-specific
implementation exists. Record proof against the new contract and exact plan
checkboxes.

## 3. Locked direction

The current plan has locked these architectural boundaries; R0 turns them into
exact implementable contracts:

- one static authority at `<config root>/hosting/hosting_config.json`;
- top-level `category_dirs` roots and `@hosting`, `@packages`, and
  `@environments` labels;
- host-local setup/config library writes static configuration; daemon reads it;
- static policy is activated by deliberate restart;
- dynamic package, template, environment, tool, and toolbox management remains
  available through the control channel after startup;
- authenticated server-side role/effective scope authorizes mutations;
- daemon-computed SHA-256 identifies received artifacts;
- mandatory publisher signing is absent from the baseline;
- one worker-neutral package/environment subsystem serves toolbox, Python
  workflow helper, and JavaScript/Node worker consumers; and
- clean major-version cut with no compatibility readers, aliases, or legacy
  environment reuse.

## 4. Provisional inventory

This is navigation input only. R1.2–R1.5 must replace it with an exhaustive,
owned inventory before implementation.

### Configuration and startup seams

- `src/mp13_engine/mp13_config_paths.py`
- `src/app/config.py`
- `src/app/mp13chat.py`
- `hosting_config.py`
- `src/hosting/hosting_config_cli.py`
- `src/hosting/hosting_setup_api.py`
- `src/hosting/transport_bootstrap_api.py`
- `src/hosting/daemon/foreground.py`
- `src/hosting/daemon/background.py`
- `src/hosting/daemon/local_ipc.py`
- `src/hosting/daemon/toolbox_launch_config.py`
- `src/hosting/engine_host_channel.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/service/host_service.py`
- `src/hosting/service/constants.py`
- `src/hosting/service/state.py`

### Package/environment/toolbox seams

- `src/hosting/toolbox/host_project_config.py`
- `src/hosting/toolbox/dependency_policy.py`
- `src/hosting/toolbox/environment.py`
- `src/hosting/toolbox/hermetic_environment.py`
- `src/hosting/toolbox/bundle_models.py`
- `src/hosting/service/toolbox_artifact_store.py`
- `src/hosting/service/toolbox_artifact_uploads.py`
- `src/hosting/service/toolbox_artifact_upload_service.py`
- `src/hosting/service/toolbox_catalog.py`
- `src/hosting/service/toolbox_definition_resolution.py`
- `src/hosting/service/toolbox_host_config_state.py`
- `src/hosting/service/toolbox_materialization.py`
- `src/hosting/service/toolbox_runtime.py`
- `src/hosting/sandbox/runtime_base.py`
- `src/hosting/sandbox/python_runtime.py`
- `src/hosting/sandbox/js_runtime.py`
- `src/hosting/sandbox/toolbox_runtime.py`

### Known dependent seams (inspection only)

- `src/backend/platform/hosting/hosting_admin.py`
- `src/backend/platform/capabilities/parent_truth.py`
- `src/backend/platform/toolboxes/definition_coordinator.py`
- `src/ui/web/static/js/features/chat/CapabilityToolsPanel.js`
- `tests/backend_infra/test_parent_toolbox_truth.py`

## 5. Required removal searches

Record zero-result evidence in the owning completion slice for runtime/API
occurrences of:

```text
toolbox_host_project_configuration
toolbox_artifact_sources
toolbox_trust_public_keys
toolbox_source_credentials
toolbox_dependency_policy
--toolbox-config-file
engine_host_toolbox_config_file
access_control.json
ToolboxEnvironmentManager
RuntimeEnvironmentManager
toolbox_venvs
runtime_envs
toolbox_environment_cache
toolbox-artifact-upload-
toolbox-template-
toolbox-environment-remove
toolbox_configuration_missing
toolbox_configuration_incomplete
toolbox_configuration_invalid
toolbox_source_binding_invalid
```

Historical migration documents may retain a match only when the final R9 audit
identifies it explicitly as non-runtime history.

## 6. Active slice record

Populate this section before changing production code.

```text
Slice ID:
Plan IDs:
Priority:
Minimum expertise:
Owner:
Production boundary:
Expected removals:
Dependent handoff impact:
Positive tests:
Negative/security tests:
Native/platform lanes:
Status: Active | Blocked | Complete
```

Only one slice may be active. Consecutive plan IDs may share a slice when they
are tightly coupled and require the same expertise. A change in expertise ends
the slice.

## 7. Evidence log

### DOC-RESET — planning ledger reset

- Date: 2026-08-11
- Scope: documentation only; no production behavior changed
- Files:
  - `src/hosting/hosting_access_plan.md`
  - `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`
  - `src/hosting/hosting_status.md`
  - `tests/test_hosted_toolbox_contract_docs.py`
- Outcome: superseded completed ledgers replaced with the unified cutover plan,
  active dependent handoff, and fresh all-open status ledger; permanent contract
  tests no longer read the transient handoff
- Validation: permanent-to-transient inbound-reference audit returned zero;
  Markdown structure checks and `git diff --check` passed;
  `poetry run pytest -q tests/test_hosted_toolbox_contract_docs.py` passed 9/9
- Plan completion credited: none

Append one concise entry per completed slice. Include exact commands, counts,
durations where useful, negative-path results, commit pin, and dependent receipt
impact. Do not paste an unstructured full test transcript.

## 8. Blockers and external evidence

Current blockers: none recorded

External evidence required before final completion:

- dependent revision and adoption receipt for the new major contract;
- required native/platform lane results identified by R1.3/R9.7; and
- proof that remote output and process arguments contain no sentinel secrets or
  unrestricted host-local paths.

An unavailable external lane is recorded as not run with an owner and reason;
it is not reported as a pass.

## 9. Status update rules

1. Record an active slice before changing production code.
2. Keep one coherent slice per commit and use the highest expertise touched.
3. Update production code, tests, docs, handoff, status, and plan checkboxes in
   the same slice when they form one contract.
4. Mark a plan item complete only after its named proof passes.
5. Keep failures and partial work visible with checkboxes open.
6. Record removal `rg` results for every clean-cut boundary.
7. Include denial and redaction tests for authorization/security boundaries.
8. Never edit the dependent repository; record its team-supplied evidence.
9. Do not infer adoption from a parent compatibility shim.
10. Do not restore old fallbacks, aliases, dual commands, or migration readers.
11. Do not link permanent docs to transient plan/status/handoff records or test
    those records as normative contracts.
