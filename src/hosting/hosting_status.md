# Hosting Refactoring Status

Last updated: 2026-04-18

## Scope

This document is the working plan and status tracker for refactoring the hosting
package service boundary. The immediate target is `engine_host_service.py`, which
has grown into a multi-domain implementation file. The desired destination is a
new `hosting/service/` package, while preserving the existing public import path
`hosting.engine_host_service`.

The refactor should be incremental and compatibility-preserving. Do not move all
hosting files at once.

## Current Findings

- `src/hosting/engine_host_service.py` is approximately 9.5k lines.
- `EngineHostService` spans approximately 9.4k lines and 166 methods.
- The class currently owns multiple responsibilities:
  - metrics and service-level runtime counters
  - JSON state file persistence
  - access control state and daemon ownership tracking
  - auth keys, sessions, SSH challenges, issued tokens, and audit lists
  - command authorization and daemon claim policy enforcement
  - engine config listing/creation/model resolution
  - engine process spawn/discovery/registration
  - worker IPC, proxy HTTP/RPC/stream calls
  - toolbox environment descriptions, install locks, repair, reconcile, and GC
  - toolbox registration, gate, execute, cancel, and rollout history
  - sandbox filesystem and HTTP callback APIs
  - logs, shutdown, and liveness helpers
- Tests and runtime callers import `EngineHostService` from
  `hosting.engine_host_service` directly.
- Some tests monkeypatch module-level names on `hosting.engine_host_service`,
  including `MPClient`, `os.name`, and `tempfile.gettempdir`. Preserve this
  compatibility during the first refactor pass.
- `pyproject.toml` currently packages `app`, `mp13_engine`, and `hosting`; adding
  a new top-level package is unnecessary churn.

## Target Package Layout

Keep the existing top-level module as a shim:

```text
src/hosting/engine_host_service.py
```

Add the new service package:

```text
src/hosting/service/
  __init__.py
  host_service.py
  constants.py
  core.py
  errors.py
  logs.py
  state.py
  metrics.py
  auth.py
  claims.py
  control.py
  policy.py
  configs.py
  engines.py
  proxy.py
  sandbox_api.py
  toolbox_env.py
  toolbox_runtime.py
```

Initial implementation can use mixins to minimize behavior changes:

```python
class EngineHostService(
    MetricsMixin,
    StateMixin,
    AuthMixin,
    ClaimMixin,
    PolicyMixin,
    ConfigMixin,
    EngineMixin,
    ProxyMixin,
    SandboxApiMixin,
    ToolboxEnvironmentMixin,
    ToolboxRuntimeMixin,
):
    ...
```

The old module should continue to re-export:

```python
from .service import EngineHostService, ToolboxRolloutError
```

If tests still monkeypatch module globals on `hosting.engine_host_service`, keep
compatibility aliases there until tests and callers are migrated deliberately.

## What Belongs In `hosting/service/`

Move only service-owned logic into `hosting/service/`:

- `EngineHostService`
- `ToolboxRolloutError`
- service constants such as role names, lifecycle profile names, valid override
  reasons, default service paths, and daemon capability version metadata
- JSON state helpers and control state layout helpers
- auth/session/challenge/token/audit logic
- command authorization and daemon claim policy logic
- claim/token APIs
- engine config store methods
- engine spawn/discovery/registration methods owned by `EngineHostService`
- worker IPC helpers used by the service
- proxy request/RPC/stream methods
- sandbox callback methods exposed by the service
- toolbox environment management, repair, reconcile, references, and GC
- toolbox gate/describe/execute/cancel/register/unregister orchestration

## What Should Stay Outside `hosting/service/`

These files represent different boundaries and should not be folded into
`hosting/service/` during the service refactor:

- `engine_host_cli.py`: CLI entrypoint and command parsing.
- `engine_host_channel.py`: client-side control channel.
- `engine_host_connection.py`: local/SSH client connection transports.
- `engine_process_supervisor.py`: standalone persisted process supervisor.
- `engine_worker_ipc.py`: engine worker process entrypoint.
- `toolbox_executor_ipc.py`: toolbox executor process entrypoint.
- `client_realm.py`: client-side secret/profile realm.
- `transport_bootstrap.py`: client transport bootstrap bundle provisioning.
- `toolbox_admin.py`: admin convenience API over the service.
- `toolbox_harness.py`: toolbox bundling, staging, sandbox orchestration, and
  app-facing toolbox references. It is large and may deserve its own later split,
  but it should not be moved wholesale into `hosting/service/`.
- `sandbox/`: broker and worker sandbox primitives remain their own package.

## Daemon Split Guidance

`engine_host_daemon.py` should not move into `hosting/service/` wholesale. It is
daemon infrastructure, not service domain logic. It owns:

- PID file management
- local IPC listener setup
- HTTP ingress daemon setup
- foreground/background process startup
- daemon ACL/path hardening
- request dispatch into `EngineHostService`

If it is split later, use a separate package:

```text
src/hosting/daemon/
  __init__.py
  pidfile.py
  security.py
  local_ipc.py
  http_ingress.py
  runtime.py
  launcher.py
```

Keep `src/hosting/engine_host_daemon.py` as a compatibility shim after that
split, re-exporting `EngineHostDaemon`, `EngineHostHttpIngressDaemon`,
`DaemonPidFile`, `DEFAULT_DAEMON_PORT`, `DEFAULT_HTTP_INGRESS_PORT`,
`run_daemon_foreground`, `run_http_ingress_foreground`,
`start_daemon_background`, and `start_http_ingress_background`.

Do the service split first. Split daemon code only after the service package is
stable.

## Refactoring Phases

### Phase 0 - Baseline And Safety

Status: Completed

Goals:

- Capture current test baseline before moving code.
- Identify module-level monkeypatch targets in tests.
- Keep public import paths stable.

Tasks:

- Run targeted hosting tests:
  - `pytest tests/test_hosting_auth_roles.py`
  - `pytest tests/test_hosting_service_security.py`
  - `pytest tests/test_hosting_service_list_configs.py`
  - `pytest tests/test_hosting_toolbox_sandbox.py`
  - `pytest tests/test_hosting_worker_sandbox.py`
  - `pytest tests/test_hosting_http_ingress.py`
  - `pytest tests/test_engine_host_channel.py`
- Record failures that are unrelated to the refactor before changing files.
- Search tests for `hosting.engine_host_service` monkeypatches and keep a list.
- Avoid changing persisted JSON schemas, command names, error strings, or token
  formats in this refactor.

### Phase 1 - Create `hosting/service/` Skeleton

Status: Completed

Goals:

- Introduce the package without changing behavior.
- Move constants and errors first.

Tasks:

- Add `src/hosting/service/__init__.py`.
- Add `src/hosting/service/constants.py`.
- Add `src/hosting/service/errors.py`.
- Add `src/hosting/service/host_service.py` containing the current
  `EngineHostService` implementation or an initial subclass/mixin shell.
- Change `src/hosting/engine_host_service.py` into a compatibility shim only
  after import compatibility is confirmed.
- Re-export `EngineHostService` and `ToolboxRolloutError` from both
  `hosting.service` and `hosting.engine_host_service`.

Verification:

- `python -c "from hosting.engine_host_service import EngineHostService"`
- `python -c "from hosting.service import EngineHostService"`
- Run the Phase 0 targeted tests that import the service directly.

### Phase 2 - Move Low-Risk Shared Service Internals

Status: Completed

Goals:

- Reduce file size with low behavioral risk.
- Avoid splitting high-coupling toolbox/auth flows first.

Candidate modules:

- `metrics.py` (completed)
- `state.py` (basic JSON state, toolbox locks, and control-state read/write completed)
- `configs.py` (completed)

Tasks:

- Move metrics helpers:
  - `_ensure_metrics_initialized`
  - `_metrics_proxy_start`
  - `_metrics_proxy_finish`
  - `_metrics_auth_denied`
  - `_metrics_challenge_event`
  - `get_host_metrics`
- Move JSON/control path helpers:
  - `_read_json`
  - `_write_json`
  - `_read_engines`
  - `_write_engines`
  - `_read_control`
  - `_write_control`
  - `_default_control_payload`
  - `_control_layout`
- Move config-store helpers:
  - `_merge_default_and_selected_config`
  - `_resolve_path_token`
  - `list_engine_configs`
  - `create_engine_config`
  - `models_from_config`

Progress notes:

- `hosting/service/state.py` currently contains the basic JSON state helpers,
  engine/toolbox state read/write helpers, toolbox lock helpers, and
  control-state default/read/write helpers.

Verification:

- Run `tests/test_hosting_service_list_configs.py`.
- Run `tests/test_hosting_service_security.py`.
- Confirm no persisted state output changes except ordering where tests already
  allow it.

### Phase 3 - Move Auth, Claims, And Policy

Status: Completed

Goals:

- Separate access-control behavior from engine/toolbox runtime behavior.
- Keep security-sensitive behavior byte-for-byte equivalent where practical.

Candidate modules:

- `auth.py` (completed)
- `claims.py` (completed)
- `policy.py` (completed)
- `control.py` (completed)

Tasks:

- Move auth key/session/challenge methods:
  - `_hash_secret`
  - `_token_preview`
  - `_prune_expired_sessions`
  - `_prune_expired_challenges`
  - `_verify_ssh_signature`
  - `_extract_session_token`
  - `_role_allowed_scopes`
  - `_commands_allowed_for_role`
  - `_authorize_role_for_command`
  - `_validate_session`
  - `auth_status`
  - `auth_list_keys`
  - `auth_list_sessions`
  - `auth_list_issued_tokens`
  - `auth_list_audit_events`
  - `auth_upsert_key`
  - `auth_revoke_key`
  - `auth_issue_session`
  - `_issue_session_for_key`
  - `auth_begin_challenge`
  - `auth_complete_challenge`
  - `auth_revoke_session`
  - `reset_hosting_access`
- Move policy methods:
  - `authorize_command`
  - `enforce_daemon_claim_policy`
- Move claim/token methods:
  - `_claim_scope_key`
  - `_claim_acl_policy`
  - owner keepalive and ownership notice helpers
  - `_append_claim_audit_event`
  - `_append_auth_audit_event`
  - `_actor_id_from_payload`
  - override reason helpers
  - `claim_engine`
  - `claim_endpoint`
  - `claim_resource`
  - status/token issue/validate methods

Verification:

- Run `tests/test_hosting_auth_roles.py`.
- Run `tests/test_hosting_service_security.py`.
- Run `tests/test_hosting_daemon_acl.py`.
- Run `tests/test_hosting_http_ingress.py`.

Progress notes:

- `hosting/service/auth.py` currently contains the low-level auth helpers:
  `_hash_secret`, `_token_preview`, expiry pruning, SSH signature verification,
  session token extraction, role-to-scope mapping, role command tables, session
  validation, and public auth APIs.
- `hosting/service/claims.py` currently contains private ownership, keepalive,
  ownership-change notice, actor, audit, public claim, and token helpers.
- `hosting/service/policy.py` currently contains command authorization and
  daemon claim policy enforcement.

### Phase 4 - Move Engine, IPC, Proxy, Logs, And Sandbox APIs

Status: Completed

Goals:

- Isolate worker/process communication from control-state logic.

Candidate modules:

- `engines.py`
- `engines.py` (completed)
- `proxy.py` (completed, including service-owned IPC address/call helpers)
- `sandbox_api.py` (completed)
- `logs.py` (completed)

Tasks:

- Move engine lifecycle methods:
  - `_next_engine_id`
  - `_check_module_available`
  - `_check_module_discoverable`
  - `_engine_python_executable`
  - `_build_engine_spawn_spec`
  - `_build_generic_spawn_spec`
  - `connect_from_config`
  - `discover_running`
  - `get_registration`
  - `register_spawned`
  - `spawn`
  - `remove_registration`
  - `shutdown`
  - `ensure_running`
  - log tail/follow helpers
- Move IPC helpers:
  - `_allocate_ipc_address`
  - `_parse_worker_authkey_token`
  - `_proxy_request_via_ipc`
  - `_ipc_call`
  - `_require_ipc_registration`
- Move proxy methods:
  - `proxy_request`
  - `proxy_rpc_call`
  - `proxy_rpc_open`
  - `proxy_rpc_send`
  - `proxy_rpc_recv`
  - `proxy_rpc_close`
  - `proxy_stream_open`
  - `proxy_stream_send`
  - `proxy_stream_recv`
  - `proxy_stream_close`
- Move sandbox callback methods:
  - `_sandbox_callback_result`
  - `sandbox_fs_read_text`
  - `sandbox_fs_write_text`
  - `sandbox_fs_mkdir`
  - `sandbox_fs_list`
  - `sandbox_fs_stat`
  - `sandbox_http_fetch`

Progress notes:

- `hosting/service/sandbox_api.py` contains sandbox callback APIs via
  `SandboxApiMixin`.
- `hosting/service/logs.py` contains log tail/follow helpers via `LogsMixin`.
- Worker IPC proxy helpers remain in `host_service.py` for now because tests and
  callers monkeypatch `hosting.engine_host_service.MPClient`; `proxy.py` now
  includes a compatibility bridge for that module-level binding as well as
  `os` and `tempfile`.

Compatibility note:

- Tests currently monkeypatch `hosting.engine_host_service.MPClient`,
  `hosting.engine_host_service.os.name`, and
  `hosting.engine_host_service.tempfile.gettempdir`. Either preserve aliases in
  the shim or migrate tests after the first green refactor.

Verification:

- Run `tests/test_hosting_worker_sandbox.py`.
- Run `tests/test_hosting_toolbox_sandbox.py` IPC-address tests.
- Run `tests/test_engine_host_channel.py`.

### Phase 5 - Move Toolbox Service Runtime

Status: Completed

Goals:

- Split the largest service domain after lower-risk boundaries are stable.

Candidate modules:

- `toolbox_env.py`
- `toolbox_runtime.py`

Tasks for `toolbox_env.py`:

- Move toolbox locks and persisted toolbox state helpers.
- Move environment description/list/get/upsert/clone methods.
- Move requirements resolution and install lock methods.
- Move environment apply/realize/sync/prepare/execute/verify methods.
- Move consistency/review/repair/reconcile/reference/gc methods.

Tasks for `toolbox_runtime.py`:

- Move toolbox metadata and registration helpers.
- Move `toolbox_describe`, `toolbox_gate`, `toolbox_execute`, and
  `toolbox_cancel`.
- Move auto/manual/intrinsic register and unregister flows.
- Move rollout history helpers.
- Move executor readiness and assignment readiness checks.

Verification:

- Run full `tests/test_hosting_toolbox_sandbox.py`.
- Run `tests/test_toolbox_admin.py`.

Result:

- `toolbox_env.py` completed for environment descriptions, environment
  resolution/install workflows, consistency, repair, reconcile, references, and
  GC.
- `toolbox_runtime.py` completed for toolbox metadata, routing, describe, gate,
  execute, cancel, wait-for-ready, and register/unregister orchestration.
- Run `tests/test_mp13chat_hosted_toolbox_api.py`.

### Phase 6 - Optional Daemon Package Split

Status: Deferred

Do this only after the service split is stable.

Target package:

```text
src/hosting/daemon/
  __init__.py
  pidfile.py
  security.py
  local_ipc.py
  http_ingress.py
  runtime.py
  launcher.py
```

Tasks:

- Move `DaemonPidFile` to `pidfile.py`.
- Move ACL and secure path helpers to `security.py`.
- Move `EngineHostHttpIngressDaemon` to `http_ingress.py`.
- Move `EngineHostDaemon` to `runtime.py`.
- Move foreground/background start helpers to `launcher.py`.
- Keep `engine_host_daemon.py` as a compatibility shim.

Verification:

- Run `tests/test_hosting_daemon_pidfile.py`.
- Run `tests/test_hosting_daemon_startup.py`.
- Run `tests/test_hosting_daemon_acl.py`.
- Run `tests/test_hosting_http_ingress.py`.
- Run `tests/test_engine_host_channel.py`.

## Compatibility Rules

- Preserve these import paths until a separate breaking-change migration is
  planned:
  - `hosting.engine_host_service.EngineHostService`
  - `hosting.engine_host_service.ToolboxRolloutError`
  - `hosting.EngineHostService`
  - `hosting.engine_host_daemon.EngineHostDaemon`
  - `hosting.engine_host_daemon.DaemonPidFile`
- Preserve command names accepted by daemon/channel dispatch.
- Preserve persisted state filenames and JSON schemas.
- Preserve auth token formats, claim keys, audit event shapes, and denial error
  payloads.
- Preserve worker IPC contracts and environment variables.
- Avoid broad formatting-only changes while moving code.

## Current Status

- Service split: Phases 0 through 5 completed; Phase 6 daemon package split is
  deferred.
- `engine_host_service.py` has been moved to `hosting/service/host_service.py`;
  the old import path remains as a module alias shim.
- `hosting/service/constants.py` contains service constants and default paths.
- `hosting/service/core.py` contains small generic service helpers via
  `CoreMixin`.
- `hosting/service/errors.py` contains `ToolboxRolloutError`.
- `hosting/service/logs.py` contains log tail/follow helpers via `LogsMixin`.
- `hosting/service/metrics.py` contains metrics helpers via `MetricsMixin`.
- `hosting/service/state.py` contains basic JSON state helpers and control-state
  default/read/write helpers via `StateMixin`.
- `hosting/service/configs.py` contains engine config-store helpers via
  `ConfigMixin`, including profile/path/config-selector helpers and traffic
  policy normalization.
- `hosting/service/auth.py` contains low-level auth helpers via `AuthMixin`.
- `hosting/service/auth.py` also contains role command tables, session
  validation, and public auth APIs via `AuthMixin`.
- `hosting/service/claims.py` contains private ownership, keepalive, notice,
  actor, audit, public claim, and token helpers via `ClaimsMixin`.
- `hosting/service/policy.py` contains command authorization and daemon claim
  policy enforcement via `PolicyMixin`.
- `hosting/service/engines.py` contains engine lifecycle, registration,
  discovery, spawn, shutdown, and sandbox adapter helpers via `EnginesMixin`.
- `hosting/service/proxy.py` contains IPC allocation/calls, HTTP proxy, RPC,
  stream proxy, and token revocation helpers via `ProxyMixin`.
- `hosting/service/sandbox_api.py` contains sandbox callback APIs via
  `SandboxApiMixin`.
- `hosting/service/logs.py` contains log tail/follow helpers via `LogsMixin`.
- `hosting/service/control.py` contains control config, lifecycle policy, actor
  resolution, and runtime safety profile helpers via `ControlMixin`.
- `hosting/service/toolbox_env.py` contains toolbox environment descriptions,
  environment apply/realize/sync/install workflows, consistency, review,
  repair, reconcile, references, and GC helpers via `ToolboxEnvironmentMixin`.
- `hosting/service/toolbox_runtime.py` contains toolbox routing, describe, gate,
  execute, cancel, ready checks, and register/unregister orchestration via
  `ToolboxRuntimeMixin`.
- `hosting/service/host_service.py` now contains the `EngineHostService`
  constructor and mixin composition only.
- Verified:
  - legacy and new import paths both resolve
  - `pytest tests/test_hosting_service_list_configs.py -q` passed
  - `pytest tests/test_hosting_auth_roles.py -q` passed
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_list_configs.py -q` passed after metrics split
  - `pytest tests/test_hosting_service_security.py -q` passed after metrics split
  - `pytest tests/test_hosting_service_list_configs.py tests/test_hosting_service_security.py -q` passed after state split
  - `pytest tests/test_hosting_worker_sandbox.py -q` passed after state split
  - `pytest tests/test_hosting_service_list_configs.py -q` passed after config split
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_hosting_worker_sandbox.py -q` passed after config split
  - `pytest tests/test_hosting_toolbox_sandbox.py -q` passed after Phase 2 partial split
  - `pytest tests/test_toolbox_admin.py tests/test_hosting_http_ingress.py -q` passed after Phase 2 partial split
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py -q` passed after auth helper split
  - `pytest tests/test_hosting_http_ingress.py tests/test_engine_host_channel.py -q` passed after auth helper split
  - `pytest tests/test_hosting_daemon_pidfile.py -q` passed after restoring daemon-local PID liveness monkeypatch behavior
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_engine_host_channel.py -q` passed after daemon PID helper fix
  - `pytest tests -q` passed: 314 passed, 1 skipped
  - `pytest tests/test_hosting_service_list_configs.py tests/test_hosting_service_security.py tests/test_hosting_http_ingress.py tests/test_engine_host_channel.py -q` passed after config/profile helper split
  - `pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_toolbox_sandbox.py -q` passed after sandbox API split
  - `pytest tests/test_hosting_service_security.py tests/test_hosting_http_ingress.py tests/test_engine_host_channel.py -q` passed after sandbox API split
  - `pytest tests/test_engine_host_channel.py tests/test_hosting_service_security.py tests/test_hosting_worker_sandbox.py -q` passed after logs split
  - `pytest tests -q` passed after logs split: 314 passed, 1 skipped
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_hosting_http_ingress.py tests/test_engine_host_channel.py -q` passed after control-state split
  - `pytest tests/test_hosting_daemon_acl.py tests/test_hosting_daemon_pidfile.py -q` passed after control-state split
  - `pytest tests -q` passed after control-state split: 314 passed, 1 skipped
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_hosting_daemon_acl.py tests/test_hosting_http_ingress.py -q` passed after claim helper split
  - `pytest tests/test_engine_host_channel.py tests/test_hosting_toolbox_sandbox.py -q` passed after claim helper split
  - `pytest tests -q` passed after claim helper split: 314 passed, 1 skipped
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_hosting_http_ingress.py tests/test_engine_host_channel.py -q` passed after public auth API split
  - `pytest tests -q` passed after public auth API split: 314 passed, 1 skipped
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_hosting_daemon_acl.py tests/test_hosting_http_ingress.py tests/test_engine_host_channel.py -q` passed after policy split
  - `pytest tests -q` passed after policy split: 314 passed, 1 skipped
  - `pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_toolbox_admin.py -q` passed after engine split
  - `pytest tests/test_hosting_service_list_configs.py tests/test_hosting_service_security.py tests/test_hosting_http_ingress.py -q` passed after engine split
  - `pytest tests -q` passed after engine split: 314 passed, 1 skipped
  - `pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_http_ingress.py -q` passed after proxy split
  - `pytest tests/test_hosting_service_security.py tests/test_hosting_auth_roles.py -q` passed after proxy split
  - `pytest tests/test_hosting_toolbox_sandbox.py::test_engine_host_service_allocate_ipc_address_uses_unix_socket_on_posix tests/test_hosting_toolbox_sandbox.py::test_engine_host_service_allocate_ipc_address_uses_named_pipe_on_windows -q` passed after proxy split
  - `pytest tests -q` passed after proxy split: 314 passed, 1 skipped
  - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_hosting_daemon_acl.py tests/test_engine_host_channel.py -q` passed after public claim/token API split
  - `pytest tests -q` passed after public claim/token API split: 314 passed, 1 skipped
  - `poetry run pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_hosting_service_list_configs.py tests/test_hosting_http_ingress.py tests/test_engine_host_channel.py -q` passed after control split: 71 passed
  - `poetry run pytest tests/test_hosting_toolbox_sandbox.py tests/test_toolbox_admin.py -q` passed after toolbox environment split: 122 passed
  - `poetry run pytest tests/test_hosting_toolbox_sandbox.py tests/test_toolbox_admin.py -q` passed after toolbox runtime split: 122 passed
  - `poetry run pytest tests/test_hosting_auth_roles.py tests/test_hosting_service_security.py tests/test_hosting_service_list_configs.py tests/test_hosting_worker_sandbox.py tests/test_hosting_http_ingress.py tests/test_engine_host_channel.py tests/test_hosting_toolbox_sandbox.py tests/test_toolbox_admin.py -q` passed after toolbox runtime split: 202 passed
  - `poetry run pytest tests -q` passed after service split completion: 314 passed, 1 skipped
- Test collection note:
  - Full unscoped `pytest -q` still collects non-test-tree paths in this
    checkout and failed before running tests because `logs/pytest_tmp` is not
    readable and `misc/test_DISK_perf.py` imports missing optional dependency
    `aiofiles`. Use `pytest tests -q` for the normal repository suite until
    collection is narrowed or those environment issues are fixed.
- Daemon split: Deferred.
- Compatibility shim strategy: Required.
- Top-level `server/` package: Rejected in favor of `hosting/service/`.
