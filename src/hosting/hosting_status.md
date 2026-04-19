# Engine Host Daemon Refactoring Status

Last updated: 2026-04-18

## Scope

This document is the working plan and status tracker for refactoring
`src/hosting/engine_host_daemon.py` into a new `src/hosting/daemon/` package.

The desired destination is a focused daemon package while preserving the
existing public import path `hosting.engine_host_daemon`. This refactor should
move daemon infrastructure only. Service control-plane logic already lives in
`hosting/service/`, and toolbox harness logic already lives in
`hosting/toolbox/`.

The refactor should be incremental and compatibility-preserving. Do not change
daemon command names, PID file payloads, local IPC payloads, HTTP ingress
contracts, startup behavior, or lifecycle policy semantics while moving code.

## Current Findings

- `src/hosting/engine_host_daemon.py` is approximately 2.3k lines.
- The file owns several distinct responsibilities:
  - default daemon state path and PID path helpers
  - local IPC endpoint derivation
  - Windows ACL hardening and secure JSON writes
  - PID file read/write/remove/stale-process handling
  - HTTP ingress server and request dispatch
  - local daemon listener setup
  - daemon request dispatch into `EngineHostService`
  - foreground daemon run loops
  - lifecycle policy handling on foreground terminal disconnect
  - detached/background process launch helpers
- Tests and runtime callers import from `hosting.engine_host_daemon` directly.
- `engine_host_cli.py` starts foreground/background daemons through this module.
- `engine_host_channel.py` and daemon-related tests depend on PID file payload
  shape and local IPC endpoint fields.
- Some daemon tests monkeypatch module-level names on
  `hosting.engine_host_daemon`, including `os`, `subprocess`, and process
  liveness behavior. Preserve compatibility during the first refactor pass.

## Public Surface To Preserve

Keep these names importable from `hosting.engine_host_daemon`:

- `DEFAULT_DAEMON_PORT`
- `DEFAULT_HTTP_INGRESS_PORT`
- `DaemonPidFile`
- `EngineHostDaemon`
- `EngineHostHttpIngressDaemon`
- `run_daemon_foreground`
- `run_http_ingress_foreground`
- `start_daemon_background`
- `start_http_ingress_background`

Private helpers can move, but keep them available from the compatibility module
if tests or callers currently import or monkeypatch them:

- `_default_state_dir`
- `_default_pid_file`
- `_default_http_pid_file`
- `_daemon_local_ipc_endpoint`
- `_current_windows_account_name`
- `_tighten_windows_acl`
- `_secure_state_parent_dir`
- `_secure_path`
- `_atomic_write_secure_json`
- `_apply_foreground_terminal_disconnect_policy`

## Target Package Layout

Keep the existing module as a compatibility shim:

```text
src/hosting/engine_host_daemon.py
```

Add the daemon package:

```text
src/hosting/daemon/
  __init__.py
  constants.py
  paths.py
  security.py
  pidfile.py
  dispatch.py
  local_ipc.py
  http_ingress.py
  lifecycle.py
  foreground.py
  background.py
```

Suggested ownership:

- `constants.py`: daemon port defaults.
- `paths.py`: default state/PID paths and local IPC endpoint derivation.
- `security.py`: Windows account detection, ACL hardening, secure path helpers,
  and atomic secure JSON writes.
- `pidfile.py`: `DaemonPidFile`.
- `dispatch.py`: command dispatch helpers that adapt daemon requests to
  `EngineHostService`.
- `local_ipc.py`: `EngineHostDaemon` local IPC listener/runtime.
- `http_ingress.py`: `EngineHostHttpIngressDaemon` and HTTP request handler
  plumbing.
- `lifecycle.py`: foreground terminal disconnect policy handling.
- `foreground.py`: `run_daemon_foreground` and
  `run_http_ingress_foreground`.
- `background.py`: `start_daemon_background` and
  `start_http_ingress_background`.

The compatibility module should re-export the public API:

```python
from .daemon import *
```

If tests still monkeypatch module globals on `hosting.engine_host_daemon`, either
keep compatibility aliases in the shim or have moved modules resolve legacy
globals through `sys.modules["hosting.engine_host_daemon"]` where needed.

## What Belongs In `hosting/daemon/`

Move daemon-owned infrastructure into `hosting/daemon/`:

- PID file lifecycle and stale PID checks
- local IPC endpoint derivation
- local IPC daemon listener and request loop
- daemon HTTP ingress listener and request handling
- daemon command dispatch to `EngineHostService`
- daemon foreground run loops
- detached/background subprocess launch helpers
- Windows ACL/path hardening for daemon state files
- foreground terminal disconnect policy handling

## What Should Stay Outside `hosting/daemon/`

These files represent different boundaries and should not be folded into the
daemon package during this refactor:

- `hosting/service/`: service domain and process lifecycle implementation.
- `engine_host_service.py`: service compatibility shim.
- `engine_host_cli.py`: CLI command parsing and user-facing command entrypoint.
- `engine_host_channel.py`: client-side daemon/control channel.
- `engine_host_connection.py`: local/SSH connection transports.
- `engine_worker_ipc.py`: engine worker process entrypoint.
- `toolbox_executor_ipc.py`: toolbox executor process entrypoint.
- `hosting/toolbox/`: toolbox harness implementation package.
- `sandbox/`: sandbox/broker primitives.

## Refactoring Phases

### Phase 0 - Baseline And Safety

Status: Completed

Goals:

- Capture current daemon test baseline before moving code.
- Identify module-level monkeypatch targets in tests.
- Preserve daemon import paths and payload contracts.

Tasks:

- Run targeted daemon tests with Poetry:
  - `poetry run pytest tests/test_hosting_daemon_acl.py -q`
  - `poetry run pytest tests/test_hosting_daemon_pidfile.py -q`
  - `poetry run pytest tests/test_hosting_daemon_startup.py -q`
  - `poetry run pytest tests/test_engine_host_channel.py -q`
  - `poetry run pytest tests/test_hosting_http_ingress.py -q`
- Run the broader suite after targeted tests pass:
  - `poetry run pytest tests -q`
- Search tests for `hosting.engine_host_daemon` monkeypatches/imports.
- Record unrelated failures before moving code.
- Avoid changing PID file fields, local IPC endpoint fields, HTTP request/response
  JSON shapes, shutdown token semantics, or daemon command names.

### Phase 1 - Create `hosting/daemon/` Skeleton

Status: Completed

Goals:

- Introduce the daemon package without behavior changes.
- Move constants and stateless path helpers first.

Target modules:

- `constants.py`
- `paths.py`

Move:

- `DEFAULT_DAEMON_PORT`
- `DEFAULT_HTTP_INGRESS_PORT`
- `_default_state_dir`
- `_default_pid_file`
- `_default_http_pid_file`
- `_daemon_local_ipc_endpoint`

Tasks:

- Add `src/hosting/daemon/__init__.py`.
- Re-export constants and helpers from `hosting.engine_host_daemon`.
- Preserve local IPC address hashing and pipe/socket naming exactly.

Verification:

- `poetry run pytest tests/test_hosting_daemon_pidfile.py -q`
- `poetry run pytest tests/test_engine_host_channel.py -q`

### Phase 2 - Move Security And Atomic State Writes

Status: Completed

Goals:

- Isolate platform-specific path hardening.
- Preserve Windows ACL behavior and POSIX permissions.

Target modules:

- `security.py`

Move:

- `_current_windows_account_name`
- `_tighten_windows_acl`
- `_secure_state_parent_dir`
- `_secure_path`
- `_atomic_write_secure_json`

Tasks:

- Preserve `icacls` command arguments and warning behavior.
- Preserve POSIX permissions: directory `0700`, file `0600`.
- Preserve atomic temporary-file behavior on Windows and POSIX.
- Keep `subprocess.run` monkeypatch compatibility if daemon ACL tests patch it
  through `hosting.engine_host_daemon`.

Verification:

- `poetry run pytest tests/test_hosting_daemon_acl.py -q`
- `poetry run pytest tests/test_hosting_daemon_pidfile.py -q`

### Phase 3 - Move PID File Handling

Status: Completed

Goals:

- Move PID file read/write/remove and liveness checks into a focused module.

Target modules:

- `pidfile.py`

Move:

- `DaemonPidFile`

Tasks:

- Preserve PID file payload keys:
  - `pid`
  - `port`
  - `shutdown_token`
  - daemon local IPC endpoint fields
  - HTTP ingress metadata fields
- Preserve stale PID cleanup behavior.
- Preserve local `os.kill` based liveness semantics:
  - `ProcessLookupError` means not alive
  - `PermissionError` and platform `SystemError` style responses mean alive
- Keep `DaemonPidFile` importable from `hosting.engine_host_daemon`.

Verification:

- `poetry run pytest tests/test_hosting_daemon_pidfile.py -q`
- `poetry run pytest tests/test_engine_host_channel.py -q`
- `poetry run pytest tests/test_hosting_http_ingress.py -q`

### Phase 4 - Move HTTP Ingress Daemon

Status: Completed

Goals:

- Isolate HTTP ingress server setup and request handling.

Target modules:

- `http_ingress.py`

Move:

- `EngineHostHttpIngressDaemon`

Tasks:

- Preserve bind host/port behavior.
- Preserve HTTP routes, method handling, auth handling, response JSON shapes,
  status codes, and shutdown semantics.
- Preserve integration with `EngineHostService`.
- Keep `EngineHostHttpIngressDaemon` importable from
  `hosting.engine_host_daemon`.

Verification:

- `poetry run pytest tests/test_hosting_http_ingress.py -q`
- `poetry run pytest tests/test_toolbox_admin.py -q`

### Phase 5 - Move Local IPC Daemon Runtime And Dispatch

Status: Completed

Goals:

- Isolate the local IPC daemon runtime and command dispatch.

Target modules:

- `dispatch.py`
- `local_ipc.py`

Move:

- `EngineHostDaemon`
- daemon request command dispatch helpers extracted from `EngineHostDaemon` if
  useful.

Tasks:

- Preserve built-in commands:
  - `__ping__`
  - `__shutdown__`
- Preserve dispatch payloads to `EngineHostService`.
- Preserve error handling and JSON response shape:
  - `seq`
  - `ok`
  - `result`
  - `error`
- Preserve shutdown token validation.
- Preserve listener cleanup and PID file cleanup behavior.
- Keep `MPClient`/`MPListener` monkeypatch compatibility if tests patch through
  `hosting.engine_host_daemon`.

Verification:

- `poetry run pytest tests/test_engine_host_channel.py -q`
- `poetry run pytest tests/test_hosting_service_security.py -q`
- `poetry run pytest tests/test_hosting_auth_roles.py -q`

### Phase 6 - Move Foreground Lifecycle Helpers

Status: Completed

Goals:

- Isolate foreground daemon run loops and terminal disconnect policy.

Target modules:

- `lifecycle.py`
- `foreground.py`

Move:

- `_apply_foreground_terminal_disconnect_policy`
- `run_daemon_foreground`
- `run_http_ingress_foreground`

Tasks:

- Preserve lifecycle profile policy semantics.
- Preserve foreground daemon logging and shutdown behavior.
- Preserve service/root path argument handling.
- Preserve signal handling and terminal disconnect behavior.

Verification:

- `poetry run pytest tests/test_hosting_daemon_startup.py -q`
- `poetry run pytest tests/test_hosting_daemon_pidfile.py -q`
- `poetry run pytest tests/test_engine_host_channel.py -q`

### Phase 7 - Move Background Launchers

Status: Completed

Goals:

- Isolate detached process startup helpers.

Target modules:

- `background.py`

Move:

- `start_daemon_background`
- `start_http_ingress_background`

Tasks:

- Preserve command construction, Python executable selection, working directory,
  environment variables, stdout/stderr redirection, Windows creation flags, and
  polling behavior.
- Preserve startup timeout and error text.
- Keep `subprocess.Popen` monkeypatch compatibility if tests patch through
  `hosting.engine_host_daemon`.

Verification:

- `poetry run pytest tests/test_hosting_daemon_startup.py -q`
- `poetry run pytest tests/test_hosting_daemon_pidfile.py -q`

### Phase 8 - Convert `engine_host_daemon.py` To Shim

Status: Completed

Goals:

- Leave `engine_host_daemon.py` as a thin compatibility module.
- Make `hosting/daemon/` the implementation home.

Tasks:

- Re-export public names from `hosting.daemon`.
- Keep compatibility aliases for monkeypatched globals if needed.
- Add `__all__` to both `hosting.daemon` and `hosting.engine_host_daemon`.
- Update internal imports to prefer `hosting.daemon` only after compatibility is
  verified.

Verification:

- `poetry run pytest tests/test_hosting_daemon_acl.py tests/test_hosting_daemon_pidfile.py tests/test_hosting_daemon_startup.py -q`
- `poetry run pytest tests/test_engine_host_channel.py tests/test_hosting_http_ingress.py -q`
- `poetry run pytest tests -q`

## Compatibility Rules

- Preserve `hosting.engine_host_daemon` as a public import path.
- Preserve `hosting` package re-exports if any are added later.
- Preserve PID file JSON schemas.
- Preserve daemon local IPC endpoint derivation.
- Preserve HTTP ingress API routes and response shapes.
- Preserve daemon command names and dispatch payloads.
- Preserve shutdown token semantics.
- Preserve foreground lifecycle profile behavior.
- Preserve background subprocess command construction.
- Preserve monkeypatch compatibility for daemon tests where practical.
- Avoid broad formatting-only changes while moving code.

## Current Status

- Daemon split: Completed.
- `src/hosting/engine_host_daemon.py` is now a compatibility shim that re-exports
  the public API from `hosting.daemon` and preserves selected legacy globals for
  monkeypatch compatibility.
- `src/hosting/daemon/` is now the implementation package.
- Implemented modules:
  - `constants.py`: daemon port defaults.
  - `paths.py`: default state/PID paths and local IPC endpoint derivation.
  - `security.py`: Windows account lookup, ACL hardening, secure path helpers,
    and atomic secure JSON writes.
  - `pidfile.py`: `DaemonPidFile`.
  - `http_ingress.py`: `EngineHostHttpIngressDaemon`.
  - `local_ipc.py`: `EngineHostDaemon`.
  - `lifecycle.py`: foreground terminal disconnect policy handling.
  - `foreground.py`: foreground daemon entrypoints.
  - `background.py`: detached/background daemon launchers.
- Service and toolbox movement: Out of scope and unchanged by this refactor.
- Verified:
  - `poetry run pytest tests/test_hosting_daemon_acl.py tests/test_hosting_daemon_pidfile.py tests/test_hosting_daemon_startup.py tests/test_engine_host_channel.py tests/test_hosting_http_ingress.py -q` passed: 56 passed.
  - `poetry run pytest tests -q` passed: 314 passed, 1 skipped.

## Residual Notes

- `hosting.engine_host_daemon.os`, `subprocess`, `time`, `signal`, `http`,
  `MPClient`, and `MPListener` remain available on the shim for legacy callers
  and monkeypatch-based tests.
- Background launch helpers resolve `DaemonPidFile` through the legacy shim when
  patched, preserving existing test and caller behavior.
- Security ACL hardening resolves `_current_windows_account_name` through the
  legacy shim when patched.

