# Hosting Refactoring Progress

Date: 2026-06-21

Scope: implementation progress for the legacy cleanup items following the completed Sandbox Event Stream Protocol pillar.

## Completed In Current Slice

### Slice 1: Workflow Event Subscribe Cleanup

- [x] Added `HostedProcessSandboxBase.event_subscribe(...)` as the canonical workflow subscription read API.
- [x] Changed workflow Python and workflow JS service subscription methods to use `event_subscribe(...)` instead of delegating to legacy `stream_recv(...)`.
- [x] Removed public `workflow-python-stream-recv` and `workflow-js-stream-recv` command dispatch from:
  - control channel wrappers
  - daemon local IPC dispatch
  - CLI command tables
  - auth role command sets
  - daemon policy allowlists
- [x] Updated interactive CLI workflow event viewing to call `workflow-*-event-subscribe`.
- [x] Updated workflow stream tests to consume `normalized_events` and `batch.loss`.
- [x] Replaced active stream-path `HostedStreamEvent(...)` construction with direct `HostedStreamFrame`/`HostedStreamBatch` row building.
- [x] Updated public docs to describe `workflow-*-event-subscribe` as the workflow event read command.
- [x] Added client breaking-change instructions to `HOSTING_CLIENT_BREAKING_CHANGES.md`.

### Slice 2: Internal Legacy Shape Cleanup

- [x] Removed the legacy `HostedProcessSandboxBase.stream_recv(...)` response shape from the shared workflow process base.
- [x] Updated process-base tests to assert `event_subscribe(...)`, `batch`, and `normalized_events` only.
- [x] Kept `proxy-stream-recv` as a lower-level generic worker/proxy primitive rather than a workflow compatibility route.
- [x] Audited terminal output handling and removed duplicate JS terminal stdout emission when console output was already streamed live.
- [x] Marked completed cleanup items in `hosting_access_plan.md`.

## Still Pending

- [ ] Run the broader hosting test suite after the next cleanup slice.

## Verification

- [x] `pytest tests/test_engine_host_channel.py tests/test_engine_host_cli_interactive.py tests/test_engine_host_cli_remote_args.py tests/test_hosting_auth_roles.py -q`
- [x] `pytest tests/test_workflow_helper_service.py -q -k "daemon_dispatches_workflow or stream"`
- [x] `pytest tests/test_hosting_sandbox_process_base.py tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_runtime_pool.py -q`
- [x] `pytest tests/test_hosting_sandbox_process_base.py -q`
