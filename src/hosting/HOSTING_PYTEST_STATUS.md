# Hosting Pytest Status (IPC/RPC Migration)

Date: 2026-03-14

This file lists pytest commands relevant to the IPC-only + RPC lifecycle migration.

## 1) Environment

Run from repo root.

Use one of these setups:

- Preferred: install package in editable mode, then run pytest without extra env vars.
- Alternative: if you are not installing the package, set `PYTHONPATH=src` so imports like `from hosting...` resolve.

Windows PowerShell (alternative mode):

```powershell
$env:PYTHONPATH = "src"
```

Linux/macOS bash (alternative mode):

```bash
export PYTHONPATH=src
```

No other environment variables are required for these tests.

Temp-root policy for this repo:
- `tests/conftest.py` sets `PYTEST_DEBUG_TEMPROOT` automatically with this fallback order:
  1. repo parent `.mp13_pytest` (preferred, outside project root)
  2. system temp `mp13_pytest`
  3. repo-local `.tmp_pytest` (last resort when sandbox ACLs block (1) and (2))
- This removes the need to pass `--basetemp` for normal runs.

## 2) Feature-to-Test Matrix

1. Claim ACL + ownership conflict/force-override denials
   - `tests/test_hosting_daemon_acl.py`
   - `pytest tests/test_hosting_daemon_acl.py -q`
2. Channel command/auth wiring
   - `tests/test_engine_host_channel.py`
   - `pytest tests/test_engine_host_channel.py -q`
3. HTTP ingress host proxy path
   - `tests/test_hosting_http_ingress.py`
   - `pytest tests/test_hosting_http_ingress.py -q`
4. Host service security primitives
   - `tests/test_hosting_service_security.py`
   - `pytest tests/test_hosting_service_security.py -q`
5. Detached daemon PID/readiness regressions
   - `tests/test_hosting_daemon_pidfile.py` (+ channel smoke parity)
   - `pytest tests/test_hosting_daemon_pidfile.py tests/test_engine_host_channel.py -q`
6. Clean-slate auth/authz roles + safe unauth gate + endpoint mode default/override
   - `tests/test_hosting_auth_roles.py`
   - `pytest tests/test_hosting_auth_roles.py -q`
   - includes model override and generic worker profile role gating for `connect-from-config`
7. Setup script / keyring migration / doctor diagnostics
   - `tests/test_hosting_config.py`
   - `pytest tests/test_hosting_config.py -q`

## 3) Focused ACL Regression (access denied)

```bash
pytest tests/test_hosting_daemon_acl.py -q
```

Environment note:
- In this sandbox, `tests/test_hosting_daemon_acl.py` can hit pytest tmpdir teardown ACL issues; use workspace-based harness execution if needed.

Expected denial codes include:
- `session_token_required`
- `engine_shared_claim_not_member`
- `exclusive_owner_conflict`
- `localhost_force_override_confirmation_required`
- `force_override_reason_required`
- `force_override_emergency_reason_invalid`
- `non_localhost_shared_claim_denied`

Emergency override coverage includes:
- reason-required denial for force override
- localhost emergency override grant without confirmation for stale/malicious/security reasons
- high-severity claim audit tagging on emergency grant
- displaced-owner denial until reclaim path (`ownership_changed_reclaim_required`)

## 4) Channel/Auth Path


```bash
pytest tests/test_engine_host_channel.py -q
```

## 5) HTTP Ingress


```bash
pytest tests/test_hosting_http_ingress.py -q
```

## 6) Security Suite


```bash
pytest tests/test_hosting_service_security.py -q
```

## 7) Combined Relevant Run


```bash
pytest tests/test_hosting_daemon_acl.py tests/test_engine_host_channel.py tests/test_hosting_http_ingress.py tests/test_hosting_service_security.py -q
```

## 8) Windows Detached Daemon RCA Notes (2026-03-12)

Scope investigated:
- Reported behavior: daemon appears to exit after readiness poll does bare TCP connect+close.
- Environment focus: Windows + Python 3.12 + detached process flags.

What was tested:
- Isolated detached `asyncio.start_server` reproduction with `ProactorEventLoop`, client bare connect+close, handler path `readline() -> empty -> writer.close() -> await writer.wait_closed()`.
- Real daemon process launched detached via `python -m hosting.engine_host_cli --daemon ...`, then bare connect+close probe.
- Repeated runs checking post-probe liveness and protocol responsiveness.
- PID comparison between `Popen.pid` and PID written by daemon (`os.getpid()` in pid file).

Observed results on this host:
- Isolated reproduction: no loop/process termination after bare connect+close.
- Real daemon repeated runs: no daemon self-exit after bare connect+close; daemon remained pingable.
- `Popen.pid` vs pid-file PID: matched in all sampled runs.

RCA conclusion:
- Primary confirmed root cause is liveness misclassification in `DaemonPidFile._pid_alive`:
  - `os.kill(pid, 0)` can raise `SystemError` on Windows detached paths.
  - Previous code treated generic exceptions as dead, causing false "daemon not alive" status.
- Readiness probe was hardened from bare TCP connect/close to protocol `__ping__` to avoid fragile teardown-only probes and align readiness with actual daemon protocol handling.

Current status:
- No reproducible evidence (on this machine) of intrinsic daemon self-exit caused by `writer.wait_closed()` after empty client read.
- If this is still observed elsewhere, it is likely environment-specific (Python build/OS patch level/security tooling) and should be captured with per-process logs and faulthandler output from that host.

## 9) Added Targeted Regression Tests

```bash
pytest tests/test_hosting_daemon_pidfile.py tests/test_engine_host_channel.py -q
```

Includes checks for:
- `_pid_alive` handling of `SystemError` / `ProcessLookupError` / `PermissionError`.
- `start_daemon_background()` readiness using protocol ping (`__ping__`) rather than bare socket connect/close.

Additional ACL hardening checks now include:
- force override reason requirement
- emergency override (no confirmation) path for stale/malicious ownership scenarios
- high-severity claim audit event tagging on emergency override grant

## 10) Clean-Slate Auth/AuthZ Role-Gating Slice

```bash
pytest tests/test_hosting_auth_roles.py -q
```

Windows ACL-safe variant:

```bash
pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role7
```

Coverage includes:
- `diagnostic_user` command-role denial (`insufficient_role`) for spawn.
- safe-profile hard-gate rejection for `require_auth=false` under non-local connectivity.
- runtime command-path rejection of unsafe manual unauth configuration edits.
- rejection of legacy role names during key upsert (`management` no longer accepted).
- direct runtime policy assertion rejection for unsafe unauth profiles.
- claim behavior honoring `endpoint_mode_default` when `exclusive` is omitted.
- daemon runtime endpoint mode override applying to claim commands.
- `model_user` denied model override in `connect-from-config`.
- `model_user_with_model_control` allowed model override in `connect-from-config`.
- `model_user` denied generic worker profile in `connect-from-config`.
- `worker_user` allowed generic worker profile in `connect-from-config`.
- `model_user` denied proxy traffic to registered generic engine profile.
- `worker_user` allowed proxy traffic to registered generic engine profile.
- transport role rejects shared-secret onboarding and cannot issue sessions/challenges.
- non-local connectivity requires `ssh_binding` for session/challenge bootstrap paths.
- non-local connectivity requires SSH binding on session-backed command path (including legacy unbound-session denial).
- admin-only key/session invalidation control authorization is enforced (`auth-revoke-key`, `auth-revoke-session`).
- admin-only auth audit query authorization is enforced (`auth-audit-list`).
- lifecycle profile control-config contract:
  - `lifecycle_profile` defaults/persistence
  - invalid lifecycle profile rejection
  - lifecycle policy override persistence
- channel payload plumbing for lifecycle config fields (`set-control-config`).

Note:
- Default test policy now avoids explicit `--basetemp` for this suite.
- Latest run in this repo: `28 passed, 2 warnings` (warnings are pytest cache ACL warnings).
- Command:
  - `pytest tests/test_hosting_auth_roles.py -q`

## 11) Setup Script / Migration / Doctor Coverage

```bash
pytest tests/test_hosting_config.py -q
```

Coverage includes:
- setup import path writes expected Hosting artifacts (`client_key_map.json`, `bootstrap_state.json`) and admin key metadata.
- safe-profile enforcement rejects unsafe `require_auth=false` setup intent.
- legacy key rename-to-`.migrated` flow writes migration metadata.
- `--doctor` returns healthy status for valid setup.
- `--doctor` flags unsafe runtime policy drift.
- lifecycle profile setup contract:
  - requested lifecycle profile is persisted in control config
  - lifecycle profile is mirrored in setup/access artifacts

Latest run in this repo:
- `6 passed, 2 warnings` (warnings are pytest cache ACL warnings).
- command:
  - `pytest tests/test_hosting_config.py -q`

## 12) Config-Driven Connect Runtime Coverage

```bash
pytest tests/test_hosting_service_list_configs.py -q
```

Coverage includes:
- connect progress event contract for model-engine connect path.
- generic worker connect path succeeds without model selection.
- generic worker connect path fails with `generic_worker_command_missing` when command is not configured.
8. Config-driven connect runtime (model + generic worker profiles)
   - `tests/test_hosting_service_list_configs.py`
   - `pytest tests/test_hosting_service_list_configs.py -q`
   - Environment note: in this sandbox, run via workspace harness if pytest tmpdir teardown ACL errors occur.

## 13) Lifecycle Enforcement Regression Commands (Manual Outside Sandbox)

Added coverage includes:
- foreground terminal-disconnect policy hook behavior
- detached runtime profile bootstrap flag wiring
- exclusive-owner disconnect shutdown policy behavior

Commands attempted in this sandbox:
1. `pytest tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg19`
2. `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl19`
3. `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid20`
4. `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid21`
5. `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl22`
6. `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid22`
7. `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl23`
8. `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid23`
9. `pytest tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg20`
10. Manual outside-sandbox rerun:
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid23`
11. Manual outside-sandbox rerun:
   - `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl23`

Sandbox failure reason:
- commands (1)-(3) fail in this sandbox during pytest session teardown with:
  - `PermissionError: [WinError 5] Access is denied`
  - failing path is the specified `--basetemp` directory during pytest `cleanup_dead_symlinks(...)`

Latest note:
- command (3) also executes tests, then fails during the same pytest session teardown path with identical basetemp ACL error.
- command (4) also executes tests, then fails during the same pytest session teardown path with identical basetemp ACL error.
- command (5) also executes tests, then fails during the same pytest session teardown path with identical basetemp ACL error.
- command (6) also executes tests, then fails during the same pytest session teardown path with identical basetemp ACL error.
- command (7) also fails during pytest session teardown with identical basetemp ACL error.
- command (8) executes tests, then fails during pytest session teardown with identical basetemp ACL error.
- command (9) passes in sandbox: `34 passed, 2 warnings` (warnings are pytest cache ACL warnings).
- command (10) passes outside sandbox: `14 passed in 0.24s`.
- command (11) passes outside sandbox: `15 passed in 1.17s`.

Action:
- manual outside-sandbox validation for Phase 6 daemon suites is complete.

## 14) Phase 8 Legacy-Role Cutover Regression Commands (Manual Outside Sandbox)

Updated scope includes:
- app-level host-auth helper role cutover to clean-slate role names
- hosting security/ingress tests migrated from legacy roles to clean-slate roles

Commands attempted in this sandbox:
1. `pytest tests/test_app_config_host_auth.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_appcfg`
2. `pytest tests/test_hosting_service_security.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_security`
3. `pytest tests/test_hosting_http_ingress.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_ingress`
4. Manual outside-sandbox rerun:
   - `pytest tests/test_app_config_host_auth.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_appcfg`
5. Manual outside-sandbox rerun:
   - `pytest tests/test_hosting_service_security.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_security`
6. Manual outside-sandbox rerun:
   - `pytest tests/test_hosting_http_ingress.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_ingress`

Sandbox failure reason:
- commands (1)-(3) fail in this sandbox during pytest session teardown with:
  - `PermissionError: [WinError 5] Access is denied`
  - failing path is the specified `--basetemp` directory during pytest `cleanup_dead_symlinks(...)`

Action:
- manual outside-sandbox validation for this Phase 8 cutover regression slice is complete.

Latest note:
- command (4) passes outside sandbox: `1 passed in 0.10s`.
- command (5) passes outside sandbox: `14 passed in 0.86s`.
- command (6) passes outside sandbox: `3 passed in 2.53s`.

## 15) Temp-Root Investigation (2026-03-15)

Root-cause findings:
1. Repo-root `.tmp_*` directories were created by test-local helpers using `Path.cwd() / ".tmp_*"` in:
   - `tests/test_hosting_auth_roles.py`
   - `tests/test_hosting_config.py`
2. Those helpers now use `PYTEST_DEBUG_TEMPROOT`-based workspace directories.
3. Pytest temp root is now auto-configured in `tests/conftest.py` (outside-root preferred, repo-local fallback only when ACL-restricted).

Commands attempted during investigation:
1. `pytest tests/test_hosting_auth_roles.py -q`
   - initial attempt failure (before fallback fix): `PermissionError: [Errno 13]` writing to `C:\Users\me\AppData\Local\Temp\...`
2. `pytest tests/test_hosting_config.py -q`
   - initial attempt failure (before fallback fix): `PermissionError: [WinError 5]` creating directories under `C:\Users\me\AppData\Local\Temp\...`
3. `pytest tests/test_hosting_auth_roles.py -q`
   - intermediate conftest failure: `PermissionError: [WinError 5]` creating `O:\repos\.mp13_pytest`
4. `pytest tests/test_hosting_config.py -q`
   - intermediate conftest failure: `PermissionError: [WinError 5]` creating `O:\repos\.mp13_pytest`

Post-fix reruns (no `--basetemp`):
1. `pytest tests/test_hosting_auth_roles.py -q` -> `26 passed, 2 warnings`
2. `pytest tests/test_hosting_config.py -q` -> `6 passed, 2 warnings`

Warnings note:
- remaining warnings are pytest cache ACL warnings on `.pytest_cache`; they do not affect test result status.

## 16) Phase 5 Enforcement Tightening (2026-03-15)

Scope:
1. `set-control-config` now revalidates no-auth safe profile even when `require_auth` is omitted in update payload.
2. Session/challenge issuance commands are denied when `require_auth=false`:
   - `auth-issue-session`
   - `auth-begin-challenge`
   - `auth-complete-challenge`
   - denial code: `require_auth_disabled_disallows_session_commands`

New regression tests:
1. `test_require_auth_false_rejected_when_profile_drifts_without_require_auth_field`
2. `test_require_auth_false_rejects_session_and_challenge_issue_paths`

Validation commands (no `--basetemp`):
1. `pytest tests/test_hosting_auth_roles.py -q` -> `28 passed, 2 warnings`
2. `pytest tests/test_hosting_config.py -q` -> `6 passed, 2 warnings`

Warnings note:
- warnings are pytest cache ACL warnings in this sandbox and do not change pass/fail status.

## 17) Phase 2 Ownership Enforcement Consistency (2026-03-15)

Scope:
1. Daemon endpoint-mode special command handlers now enforce displaced-owner claim policy:
   - `set-endpoint-mode-override`
   - `get-endpoint-mode-effective`
2. Displaced owner remains denied on these non-claim commands until reclaim:
   - `ownership_changed_reclaim_required`

Regression coverage updates:
1. Extended `test_displaced_owner_is_denied_until_reclaim_then_cleared` to include endpoint-mode command denials.

Sandbox command attempted:
1. `pytest tests/test_hosting_daemon_acl.py -q`
   - failed in fixture setup (before test execution) with:
     - `PermissionError: [WinError 5] Access is denied`
     - failing path: `C:\Users\me\AppData\Local\Temp\mp13_pytest\pytest-of-me`

Action:
- command must be rerun manually outside this sandbox.
- syntax validation in sandbox passed:
  - `python -m py_compile src/hosting/engine_host_daemon.py tests/test_hosting_daemon_acl.py`
- manual outside-sandbox rerun result:
  - `pytest tests/test_hosting_daemon_acl.py -q`
  - `15 passed in 1.10s`
