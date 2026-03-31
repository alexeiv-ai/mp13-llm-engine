## Hosting Sandbox Status

Date: 2026-03-29
Scope: Windows-first worker sandbox starter slice

### Implemented

1. Added `hosting/sandbox/` package:
   - [__init__.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/__init__.py)
   - [policy.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/policy.py)
   - [launcher.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/launcher.py)
   - [windows.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/windows.py)
   - [broker_fs.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/broker_fs.py)
   - [broker_http.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/broker_http.py)
   - [worker_http.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/worker_http.py)
2. Introduced `WorkerSandboxPolicy` normalization and persistence in worker registrations.
3. Moved worker launch orchestration out of [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) into sandbox launcher helpers.
4. Wired daemon/CLI `spawn` payloads to accept optional `sandbox_policy`.
5. Default spawn hygiene now uses `close_fds=True` when `inherit_parent_handles=false`.
6. Added Windows sandbox launcher path intended to use:
   - restricted token
   - Low Integrity Level
   - Job Object
7. Added brokered filesystem enforcement on the host side:
   - root-scoped read/write/list/stat/mkdir helpers
   - traversal denial
   - registration-bound sandbox policy lookup
8. Added daemon/CLI command wiring for:
   - `sandbox-fs-list`
   - `sandbox-fs-read-text`
   - `sandbox-fs-write-text`
   - `sandbox-fs-mkdir`
   - `sandbox-fs-stat`
   - `sandbox-http-fetch`
9. Added brokered HTTP enforcement on the host side:
   - broker-only network mode check
   - host allowlist and URL-prefix allowlist enforcement
   - response size cap and header sanitization

### Current Effective Outcome

1. Policy schema exists and is persisted in managed worker registration metadata.
2. Spawn/respawn paths are routed through a dedicated launcher abstraction.
3. Parent-handle inheritance control is implemented at the subprocess launcher level.
4. Windows restricted-token / Low IL / Job Object path is live and validated by integration tests for:
   - write denial against a medium-integrity file
   - named-pipe RPC continuity for a sandboxed helper worker
5. Existing worker RPC-over-pipe/socket contract is preserved by design, with a gated environment-sensitive test for the full `hosting.engine_worker_ipc` path that also requires a real model or engine config input.
6. Brokered filesystem allowlist enforcement is live on the host side for declared `root_id` entries.
7. Added worker-side brokered filesystem client adapter module:
   - [worker_fs.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/worker_fs.py)
8. Added brokered HTTP host-side enforcement and worker-side adapter:
   - [broker_http.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/broker_http.py)
   - [worker_http.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/worker_http.py)
9. Live Windows `engine_worker_ipc` validation no longer fails just because a post-shutdown cleanup path is slow; the test now kills stuck workers after proving the pipe RPC path.

### Not Yet Complete

1. Worker-side adapter hookup from actual sandboxed worker code to brokered filesystem/HTTP methods
2. Un-gated live Windows validation for the full `hosting.engine_worker_ipc` path under Low IL without requiring an external model/config fixture
3. Child-process restriction enforcement beyond Job Object starter wiring
4. Linux `bwrap` backend

### Test Evidence

Commands run:

1. `python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_daemon_pidfile.py tests/test_hosting_service_security.py tests/test_hosting_auth_roles.py -q`
   - result: `67 passed`
2. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_daemon_startup.py tests/test_hosting_daemon_pidfile.py -q`
   - result: `29 passed`
3. `python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_worker_sandbox_windows_live.py -q`
   - result: `11 passed, 1 skipped`
4. `python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_worker_sandbox_windows_live.py tests/test_hosting_daemon_pidfile.py tests/test_hosting_service_security.py tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py tests/test_hosting_daemon_startup.py -q`
   - result: `87 passed, 1 skipped`

Covered by tests:

1. sandbox policy normalization
2. spawn persistence of sandbox policy and runtime metadata
3. plain launcher `close_fds` behavior
4. live Windows Low-IL denial of write to a medium-integrity file
5. live Windows named-pipe RPC continuity for a sandboxed helper worker
6. brokered filesystem root-scoped read/write/list and traversal denial
7. brokered HTTP allowlist enforcement and response shaping
8. worker-side brokered filesystem and HTTP client payload construction
9. existing daemon/channel regression slices around startup and hosting auth/security

### Assessment

This slice now includes the first host-side brokered filesystem implementation, in addition to the live Windows enforcement slice. The first minimal acceptance milestone from [hosting_access_plan.md](/o:/repos/mp13-llm-engine/src/hosting/hosting_access_plan.md) is implemented, and the next phase has started:

1. `inherit_parent_handles=false`: implemented and tested
2. restricted token + Low IL + Job Object: implemented and exercised
3. write-up denial against medium-integrity files: proven by live integration test
4. named-pipe RPC continuity: proven by live integration test for sandboxed helper worker
5. full `hosting.engine_worker_ipc` under Low IL: still gated as environment-sensitive, but cleanup no longer treats a stuck post-shutdown exit as a sandbox failure
6. brokered filesystem host-side enforcement: implemented and tested
7. worker-side brokered filesystem adapter: implemented and unit tested
8. brokered HTTP host-side enforcement: implemented and unit tested
9. worker-side brokered HTTP adapter: implemented and unit tested

So the current status is:

1. architecture in place
2. spawn hygiene in place
3. Windows sandbox launcher minimally working
4. brokered filesystem host-side enforcement in place
5. brokered HTTP host-side enforcement in place
6. worker-side broker adapters in place
7. transport hookup for worker code to call broker still pending
