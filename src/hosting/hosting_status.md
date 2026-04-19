# Toolbox Harness Refactoring Status

Last updated: 2026-04-18

## Scope

This document is the working plan and status tracker for refactoring
`src/hosting/toolbox_harness.py`. The file has grown into a multi-domain module
that owns bundle data models, staging, environment realization, sandbox
orchestration, hosted-tool callback approval, execution harness behavior, hosted
toolbox references, and manifest loading.

The desired destination is a new `hosting/toolbox/` package while preserving the
existing public import path `hosting.toolbox_harness`.

The refactor should be incremental and compatibility-preserving. Do not move all
hosting toolbox files at once, and do not change public runtime contracts while
moving code.

## Current Findings

- `src/hosting/toolbox_harness.py` is approximately 3.8k lines.
- The file contains both public API objects and private infrastructure helpers.
- Tests and runtime callers import from `hosting.toolbox_harness` directly.
- `src/hosting/__init__.py` re-exports several names from `toolbox_harness.py`.
- `src/hosting/toolbox_executor_ipc.py` imports `ToolboxWorkerStartupSpec` and
  `load_toolbox_from_manifest` from `toolbox_harness.py`.
- `src/hosting/service/toolbox_env.py` and
  `src/hosting/service/toolbox_runtime.py` dynamically import several harness
  classes from `toolbox_harness.py`.
- Tests monkeypatch `hosting.toolbox_harness.subprocess.run`; preserve this
  monkeypatch compatibility until tests and callers are deliberately migrated.

## Public Surface To Preserve

Keep these imports working from `hosting.toolbox_harness`:

- `HostedToolCallbackContext`
- `ToolboxBundleFile`
- `ToolboxBundleTool`
- `ToolboxBundleAutoTool`
- `SandboxProfileSpec`
- `ToolboxAutoAssignmentRequest`
- `ToolboxManualAssignmentRequest`
- `ToolboxSandboxAssignment`
- `ToolboxBundleSpec`
- `ToolboxWorkerStartupSpec`
- `ToolboxEnvironmentSpec`
- `ToolboxEnvironmentManager`
- `StagedToolboxBundle`
- `ToolboxBundleStager`
- `ToolboxSandboxOrchestrator`
- `ToolboxHarnessConfig`
- `ToolboxExecutionHarness`
- `HostedToolBoxRef`
- `PendingHostedToolboxRef`
- `SandboxedToolboxFacade`
- `serialize_tools_view`
- `is_canceled_tool_error`
- `should_resubmit_canceled_tool_call`
- `load_toolbox_from_manifest`

Private helpers can move, but callers that monkeypatch module globals must keep
working during the first pass. In particular, retain or bridge:

- `hosting.toolbox_harness.subprocess`
- `hosting.toolbox_harness.Client`
- `hosting.toolbox_harness.Listener`
- `hosting.toolbox_harness.os`
- `hosting.toolbox_harness.tempfile`
- Windows named-pipe helper globals used by the low-integrity pipe helper.

## Target Package Layout

Keep the existing top-level module as a compatibility shim:

```text
src/hosting/toolbox_harness.py
```

Add a toolbox package:

```text
src/hosting/toolbox/
  __init__.py
  common.py
  cancellation.py
  tools_view.py
  callbacks.py
  bundle_models.py
  environment.py
  staging.py
  orchestration.py
  execution.py
  hosted_ref.py
  manifest.py
  windows_ipc.py
```

Suggested ownership:

- `common.py`: `_stable_json`, `_sha256_text`, shared JSON/path helpers.
- `cancellation.py`: canceled-tool detection and coarse cancel error helpers.
- `tools_view.py`: `serialize_tools_view`, tools-view cloning, approval and
  scope constraint helpers.
- `callbacks.py`: `HostedToolCallbackContext`, `_HostedToolCallbackRelay`, and
  approval request helpers.
- `bundle_models.py`: dataclasses/specs for bundle files, tools, sandbox
  profiles, assignment requests, sandbox assignments, bundle specs, worker
  startup specs, environment specs, and harness config.
- `environment.py`: `ToolboxEnvironmentManager`.
- `staging.py`: `StagedToolboxBundle` and `ToolboxBundleStager`.
- `orchestration.py`: `ToolboxSandboxOrchestrator`.
- `execution.py`: `ToolboxExecutionHarness`.
- `hosted_ref.py`: `HostedToolBoxRef`, `PendingHostedToolboxRef`, and
  `SandboxedToolboxFacade`.
- `manifest.py`: `load_toolbox_from_manifest`.
- `windows_ipc.py`: low-integrity Windows named-pipe creation helper.

The compatibility module should re-export all public names:

```python
from .toolbox import *
```

If tests still monkeypatch module globals on `hosting.toolbox_harness`, either
keep compatibility aliases in the shim or have moved modules resolve legacy
globals through `sys.modules["hosting.toolbox_harness"]` where needed.

## What Belongs In `hosting/toolbox/`

Move toolbox harness-owned logic into `hosting/toolbox/`:

- toolbox bundle and assignment dataclasses
- toolbox environment description normalization, virtualenv path planning,
  installation lock/receipt helpers, and install verification
- staging bundle files and manifest generation
- sandbox orchestration that maps assignment requests to service registrations
- native toolbox execution harness and concurrent tool-call execution
- hosted toolbox reference facade and pending mutation builder
- hosted-tool callback relay and approval timeout helpers
- manifest loading into an executable `Toolbox`
- tools-view serialization and request-scope constraint helpers
- Windows low-integrity pipe helper used by callback relay IPC

## What Should Stay Outside `hosting/toolbox/`

These files represent different boundaries and should not be folded into the
toolbox package during this refactor:

- `engine_host_service.py` and `hosting/service/`: service control plane and
  process lifecycle orchestration.
- `toolbox_executor_ipc.py`: toolbox executor process entrypoint. It can import
  from the new package, but it should stay a standalone worker entrypoint.
- `toolbox_admin.py`: admin convenience API over the service.
- `engine_worker_ipc.py`: engine worker entrypoint.
- `sandbox/`: broker and worker sandbox primitives.
- `mp13_engine/`: engine/toolbox model abstractions outside the hosting package.

## Refactoring Phases

### Phase 0 - Baseline And Safety

Status: Not started

Goals:

- Capture current test baseline before moving code.
- Identify module-level monkeypatch targets in tests.
- Keep public import paths stable.

Tasks:

- Run targeted toolbox harness tests with Poetry:
  - `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`
  - `poetry run pytest tests/test_mp13chat_hosted_toolbox_api.py -q`
  - `poetry run pytest tests/test_toolbox_admin.py -q`
- Run the broader suite after a green targeted baseline:
  - `poetry run pytest tests -q`
- Search tests for `hosting.toolbox_harness` monkeypatches and imports.
- Record unrelated failures before changing files.
- Preserve dataclass field names, `to_dict` / `from_dict` payloads, manifest
  keys, worker environment variables, callback message shapes, error strings,
  and public class names.

### Phase 1 - Create `hosting/toolbox/` Skeleton

Status: Not started

Goals:

- Introduce the package without behavior changes.
- Establish public re-export boundaries.

Tasks:

- Add `src/hosting/toolbox/__init__.py`.
- Move only tiny stateless helpers first:
  - `_stable_json`
  - `_sha256_text`
  - cancellation helpers
  - `serialize_tools_view`
- Keep `src/hosting/toolbox_harness.py` importing and re-exporting these names.
- Keep compatibility aliases for monkeypatched globals.

Verification:

- `poetry run pytest tests/test_hosting_toolbox_sandbox.py::test_load_toolbox_from_manifest_supports_intrinsic_only_revision -q`
- `poetry run pytest tests/test_hosting_toolbox_sandbox.py::test_native_toolbox_harness_executes_calls_in_parallel -q`
- `poetry run pytest tests/test_mp13chat_hosted_toolbox_api.py -q`

### Phase 2 - Move Data Models

Status: Not started

Goals:

- Move dataclasses/spec objects that have limited runtime side effects.
- Reduce the main file before moving behavior-heavy classes.

Target modules:

- `bundle_models.py`

Move:

- `HostedToolCallbackContext`
- `ToolboxBundleFile`
- `ToolboxBundleTool`
- `ToolboxBundleAutoTool`
- `SandboxProfileSpec`
- `ToolboxAutoAssignmentRequest`
- `ToolboxManualAssignmentRequest`
- `ToolboxSandboxAssignment`
- `ToolboxBundleSpec`
- `ToolboxWorkerStartupSpec`
- `ToolboxEnvironmentSpec`
- `ToolboxHarnessConfig`

Tasks:

- Preserve `to_dict`, `from_dict`, `normalized_profile_id`, stable-key, and
  worker startup serialization behavior exactly.
- Update imports in remaining harness modules and service modules.
- Re-export all moved names from `hosting.toolbox_harness` and
  `hosting.toolbox`.

Verification:

- `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`
- `poetry run pytest tests/test_mp13chat_hosted_toolbox_api.py -q`

### Phase 3 - Move Environment Management

Status: Not started

Goals:

- Isolate environment description, virtualenv, lock, receipt, and verification
  behavior.
- Preserve service imports from `hosting.toolbox_harness` during the first pass.

Target modules:

- `environment.py`

Move:

- `ToolboxEnvironmentManager`

Tasks:

- Keep `subprocess.run` monkeypatch compatibility for environment install tests.
  Options:
  - preserve `subprocess` as a compatibility alias in `toolbox_harness.py`, and
    have `environment.py` resolve legacy `subprocess` from
    `sys.modules["hosting.toolbox_harness"]`; or
  - update tests and callers together in a later compatibility cleanup phase.
- Preserve environment metadata filenames, lock payload fields, receipt fields,
  and status strings.
- Preserve path layout under `toolbox_envs/`.

Verification:

- `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`
- Focus especially on environment apply/realize/sync/prepare/lock/verify tests.

### Phase 4 - Move Staging And Manifest Loading

Status: Not started

Goals:

- Separate bundle staging and manifest loading from execution and hosted refs.

Target modules:

- `staging.py`
- `manifest.py`

Move:

- `StagedToolboxBundle`
- `ToolboxBundleStager`
- `load_toolbox_from_manifest`

Tasks:

- Preserve manifest JSON shape, revision hashing, bundle root layout, intrinsic
  tool metadata, hidden/advertised tools-view behavior, and worker command/env
  output.
- Update `toolbox_executor_ipc.py` imports after the compatibility shim is
  confirmed.
- Keep `ToolboxBundleStager` importable from `hosting.toolbox_harness`.

Verification:

- `poetry run pytest tests/test_hosting_toolbox_sandbox.py::test_load_toolbox_from_manifest_supports_intrinsic_only_revision -q`
- `poetry run pytest tests/test_hosting_toolbox_sandbox.py::test_load_toolbox_from_manifest_restores_hidden_user_tool_names -q`
- `poetry run pytest tests/test_hosting_toolbox_sandbox.py::test_load_toolbox_from_manifest_supports_auto_callable_discovery -q`
- `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`

### Phase 5 - Move Sandbox Orchestration

Status: Not started

Goals:

- Isolate assignment-to-registration orchestration used by the service runtime.

Target modules:

- `orchestration.py`

Move:

- `ToolboxSandboxOrchestrator`

Tasks:

- Preserve service calls, engine id naming, sandbox profile id normalization,
  staged bundle metadata, registration bundle/environment/tool access payloads,
  and replaced-engine shutdown behavior.
- Update dynamic imports in `hosting/service/toolbox_env.py` and
  `hosting/service/toolbox_runtime.py` only after re-export compatibility is
  confirmed.

Verification:

- `poetry run pytest tests/test_hosting_toolbox_sandbox.py::test_toolbox_sandbox_orchestrator_spawns_and_routes_multi_profile_toolbox -q`
- `poetry run pytest tests/test_hosting_toolbox_sandbox.py::test_toolbox_register_auto_persists_membership_and_replaces_profile_executor -q`
- `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`

### Phase 6 - Move Callback Relay And Tools-View Approval Helpers

Status: Not started

Goals:

- Isolate hosted-tool callback IPC, approval state, and tools-view mutation
  helpers.

Target modules:

- `windows_ipc.py`
- `callbacks.py`
- `tools_view.py`

Move:

- `_create_windows_low_integrity_pipe`
- `_HostedToolCallbackRelay`
- `_request_hosted_tool_approval`
- `_request_hosted_tool_approval_with_timeout`
- `_clone_tools_view`
- `_approve_tool_in_view`
- `_extract_scope_constraints`
- `_merge_scope_ref_into_callback_context`
- `_apply_tool_constraints_in_view`
- `_resolve_scope_ref_from_callback_context`
- `_persist_approved_tool`
- `_persist_scope_constraints`
- `_coerce_approval_decision`
- `_approval_timeout_seconds`

Tasks:

- Preserve callback payload keys and response payloads.
- Preserve approval timeout behavior and default timeout.
- Preserve Windows named-pipe low-integrity behavior.
- Preserve `Client`/`Listener` monkeypatch compatibility if tests or callers
  patch them through `hosting.toolbox_harness`.

Verification:

- `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`
- `poetry run pytest tests/test_mp13chat_hosted_toolbox_api.py -q`

### Phase 7 - Move Execution Harness

Status: Not started

Goals:

- Move native toolbox execution behavior into a focused module.

Target modules:

- `execution.py`

Move:

- `ToolboxExecutionHarness`

Tasks:

- Preserve concurrent execution behavior.
- Preserve parser/tool-call conversion, callback relay lifecycle, tools-view
  gating, cancel retry/resubmit behavior, and error payload formatting.
- Keep `ToolboxExecutionHarness` importable from both `hosting.toolbox_harness`
  and `hosting`.

Verification:

- `poetry run pytest tests/test_hosting_toolbox_sandbox.py::test_native_toolbox_harness_executes_calls_in_parallel -q`
- `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`
- `poetry run pytest tests/test_mp13chat_hosted_toolbox_api.py -q`

### Phase 8 - Move Hosted Toolbox References

Status: Not started

Goals:

- Move app-facing hosted toolbox facade and pending mutation builder into a
  focused module.

Target modules:

- `hosted_ref.py`

Move:

- `HostedToolBoxRef`
- `PendingHostedToolboxRef`
- `SandboxedToolboxFacade = HostedToolBoxRef`

Tasks:

- Preserve `to_dict` / `from_dict` payloads.
- Preserve `as_toolbox`, `as_executor`, register/unregister methods, mutation
  builder methods, and sandbox resolution behavior.
- Keep `HostedToolBoxRef` importable from both `hosting.toolbox_harness` and
  `hosting`.

Verification:

- `poetry run pytest tests/test_mp13chat_hosted_toolbox_api.py -q`
- `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`

### Phase 9 - Convert `toolbox_harness.py` To Shim

Status: Not started

Goals:

- Leave `toolbox_harness.py` as a thin compatibility module.
- Make `hosting/toolbox/` the implementation home.

Tasks:

- Re-export public names from `hosting.toolbox`.
- Keep compatibility globals for monkeypatches if any tests still require them.
- Optionally add `__all__` to both `hosting.toolbox` and
  `hosting.toolbox_harness`.
- Update internal imports to prefer `hosting.toolbox` only after the shim is
  verified.

Verification:

- `poetry run pytest tests/test_hosting_toolbox_sandbox.py -q`
- `poetry run pytest tests/test_mp13chat_hosted_toolbox_api.py -q`
- `poetry run pytest tests/test_toolbox_admin.py -q`
- `poetry run pytest tests -q`

## Compatibility Rules

- Preserve `hosting.toolbox_harness` as a public import path.
- Preserve `hosting` package re-exports.
- Preserve `SandboxedToolboxFacade` alias behavior.
- Preserve dataclass payloads and manifest schemas.
- Preserve worker startup environment variables and command construction.
- Preserve callback relay message fields and response fields.
- Preserve tools-view serialization and scope constraint behavior.
- Preserve error strings that tests assert or callers may inspect.
- Preserve environment path layout and install lock/receipt metadata.
- Avoid broad formatting-only changes while moving code.

## Current Status

- Toolbox harness split: Not started.
- `toolbox_harness.py` remains the implementation file.
- Proposed package: `src/hosting/toolbox/`.
- Compatibility shim strategy: Required.
- Daemon/service code movement: Out of scope.
- Verification baseline: Not yet captured for this refactor.

## Open Risks

- `subprocess.run` monkeypatch compatibility for environment install tests is the
  largest compatibility risk once `ToolboxEnvironmentManager` moves.
- Callback relay behavior touches IPC, threads, futures, and Windows pipe
  security; move it only after lower-risk data/staging/environment modules are
  stable.
- `HostedToolBoxRef` is app-facing and re-exported from `hosting`; move it late.
- `load_toolbox_from_manifest` is imported by the executor process entrypoint;
  keep that import path stable until the shim is verified in the full suite.

