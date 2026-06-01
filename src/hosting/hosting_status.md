# Hosting Refactor Status

Date: 2026-06-01

This file tracks progress on the hosted sandbox runtime refactoring plan in `src/hosting/hosting_access_plan.md`.

## Current Status

- [x] Initial planning context captured.
- [x] Existing architecture reviewed at a high level.
- [x] Existing `workflow_python_helper` and `workflow_js_helper` placement decided: migrate to compatibility aliases over new workflow runtime APIs.
- [x] Existing `toolbox_executor` placement decided: migrate later onto shared base while preserving toolbox semantics.
- [x] Existing generic/model worker placement decided: remain separate; borrow IPC/streaming ideas only.
- [x] Non-Python worker scope clarified: out of this epic except for external implementation of the selected wire contract.
- [ ] Implementation not started.
- [ ] Tests not updated.
- [ ] CLI not updated.
- [ ] Docs not updated beyond planning/tracking files.

## Active Phase

- [x] Phase 0: Discovery And Tests Baseline.
- [ ] Phase 1: Shared Base Contracts And Models.

## Progress Log

### 2026-06-01

- Added comprehensive refactoring plan to `hosting_access_plan.md`.
- Added client migration checklist to `HOSTING_CLIENT_BREAKING_CHANGES.md`.
- Seeded this status file.
- Completed Phase 0 test inventory. Existing focused helper coverage is in `tests/test_workflow_python_helper_ipc.py`, `tests/test_workflow_js_helper_ipc.py`, `tests/test_workflow_helper_service.py`, and `tests/test_engine_host_channel.py`.
- Existing sandbox navigation remains in `src/hosting/sandbox/sandbox_test_status.md`; new runtime-base tests should be added beside the helper/service tests rather than replacing the current sandbox suite.
- Started Phase 1 with `hosting.sandbox.runtime_base`: deterministic sandbox policy hashes, runtime/environment key specs, pool keys, worker slot snapshots, request lifecycle records, stream event envelopes, and pool metrics.

## Key Design Decisions So Far

- [x] The shared base should be internal/abstract, not a public sandbox kind.
- [x] Use two main internal layers:
  - `HostedProcessSandboxBase` for process/lifecycle/IPC/pool/metrics/cancel.
  - `HostedPythonRuntimeBase` for Python runtime environments and dependency identity.

- [x] `workflow_python` should be a concrete public kind.
- [x] `workflow_python_helper` should become a compatibility alias for `workflow_python(profile=helper, environment_name=workflow-python-helper)`.
- [x] `workflow_js_helper` should migrate to the shared process/pool base as `workflow_js(profile=helper)`.
- [x] Toolbox should migrate after workflow Python proves the base.
- [x] Generic/model workers remain separate concrete workers.
- [x] Reworked sandboxes should support streaming responses where needed.
- [x] Reworked sandboxes should report latency and concurrency metrics.
- [x] Host should control capacity/concurrency by `environment_key`.
- [x] Host should track request lifetime and cancellation.

## Known Gaps Before Implementation

- [ ] No formal hosted process pool abstraction exists yet.
- [x] First deterministic `environment_key` model exists in `hosting.sandbox.runtime_base`.
- [ ] No first-class `environment_key` routing exists for workflow Python/JS.
- [ ] Existing helper pools are tied to helper engine IDs and internal child pools.
- [ ] Existing Python helper only separates hot child checkout by Python executable, not full dependency/policy identity.
- [ ] Existing workflow environment management is present mostly through toolbox-shaped APIs.
- [ ] Existing helper response shape is narrower than planned workflow node responses.
- [ ] Existing helper streaming support is absent.
- [ ] CLI and interactive CLI are helper-command oriented.

## Next Implementation Steps

- [x] Add internal data models for environment keys, pool keys, request lifetime, stream events, and metrics.
- [x] Implement stable environment-key derivation tests before changing worker routing.
- [ ] Draft the new workflow Python API surface in service/channel/CLI.
- [ ] Keep `HOSTING_CLIENT_BREAKING_CHANGES.md` updated as compatibility shims land.
