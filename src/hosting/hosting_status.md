# Hosting Security Refactor Status

Date: 2026-03-08

## Goal

Harden hosting for remote-first operation with minimal external components:
- built-in auth for hosting commands
- separate authorization boundary for engine-config operations
- restrict protocol-visible configs to hosted config store

## Implemented

## 1) Service auth infrastructure

File: `src/hosting/engine_host_service.py`

Implemented:
- control config auth fields:
  - `require_auth`
  - `auth.keys`
  - `auth.sessions`
- key/session primitives:
  - `auth_status`
  - `auth_list_keys`
  - `auth_upsert_key`
  - `auth_revoke_key`
  - `auth_issue_session`
  - `auth_revoke_session`
- hashed secret storage (SHA-256)
- session TTL enforcement and expired session pruning
- command authorization policy:
  - `control` scope for control-plane operations
  - `config` scope for config operations
  - `traffic` scope for worker traffic forwarding
  - traffic session can be restricted to specific engine IDs

## 2) Config-path hardening

File: `src/hosting/engine_host_service.py`

Implemented:
- restricted `config_path` selector policy:
  - allowed: `default`, or hosted config name
  - denied: absolute paths, relative traversal, path separators
- resolution constrained to hosted config store:
  - `<default_config_dir>/backend/configs/*.json`

## 3) Daemon enforcement and command surface

File: `src/hosting/engine_host_daemon.py`

Implemented:
- authorization check before service execution for RPC commands
- `auth_failed` error path on denied requests
- support for new auth command dispatch
- `set-control-config` now accepts `require_auth`
- added `proxy-request` command dispatch for data-plane forwarding

## 4) CLI enforcement and command surface

File: `src/hosting/engine_host_cli.py`

Implemented:
- auth policy enforced in direct-fallback execution path as well
- added CLI subcommands:
  - `auth-status`, `auth-list-keys`, `auth-upsert-key`, `auth-revoke-key`
  - `auth-issue-session`, `auth-revoke-session`, `proxy-request`, `host-metrics`
- updated examples for auth bootstrap/session issuance

## 5) Data-plane bridge command

File: `src/hosting/engine_host_service.py`

Implemented:
- `proxy_request(...)` forwards HTTP(S) request to a registered worker endpoint
- supports method/path/query/headers/body (base64)
- bounded response size with truncation flag
- returns status code + headers + body (base64)
- enforces traffic auth scope and engine allowlist on `engine_id`
- enforces traffic policy:
  - allowed HTTP methods
  - allowed path prefixes
  - request header allowlist
  - response header allowlist
  - request/response size caps
- runtime diagnostics metrics exposed via `host-metrics`:
  - current in-flight proxy requests (total + per engine)
  - proxy success/error/failure counters
  - auth denial counters and last reason
  - request/response byte counters
  - recent proxy request ring buffer (default 100)

This enables single-port remote traffic flow through hosting protocol.

## 6) Config-path helper extensions

File: `src/mp13_engine/mp13_config_paths.py`

Implemented:
- `get_hosting_config_store_dir()`
- `normalize_hosting_config_selector()`
- `resolve_hosting_config_path()`

These helpers align the hosting store-only config model with shared config-path utilities.

## 7) Dedicated daemon HTTP ingress mode

Files:
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`

Implemented:
- dedicated HTTP ingress daemon mode:
  - `python -m hosting.engine_host_cli --daemon-http`
  - `python -m hosting.engine_host_cli --daemon-http --background`
- ingress endpoints:
  - `GET /health`
  - `POST /__shutdown__` (token-guarded)
  - `* /proxy/<engine_id>/<path...>`
  - `* /api/engine-host/proxy/<engine_id>/<path...>`
- proxy auth/session enforcement uses the same hosted session model as `proxy-request`:
  - session token via `Authorization: Bearer <token>` or `X-Session-Token`
  - `EngineHostService.authorize_command("proxy-request", payload)` enforces traffic scope and engine allowlist
- ingress path forwards to `EngineHostService.proxy_request(...)` so traffic policy constraints remain centralized.

## 8) Token introspection/audit endpoints

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- new control-scope auth audit commands:
  - `auth-list-sessions`
  - `auth-list-issued-tokens`
- outputs use redacted `token_preview` values (no full token material).
- command surface wired through:
  - service methods
  - daemon dispatch
  - CLI subcommands
  - control channel helper methods

## 9) Per-engine traffic policy overrides

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- `set-control-config` now accepts `engine_traffic_policies` map (`engine_id -> traffic_policy`).
- proxy path enforcement resolves policy per engine:
  - global `traffic_policy` as base
  - engine-specific override merged and normalized per request
- applies to:
  - command-level `proxy-request`
  - HTTP ingress proxy routes

## 10) Session binding to SSH identity/fingerprint

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_channel.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`

Implemented:
- `auth-issue-session` supports optional `ssh_binding`:
  - `target`
  - `key_fingerprint`
- bound sessions require `_ssh_session_binding` in subsequent command payloads.
- binding mismatch rejects session usage (`ssh_binding_required` / `ssh_binding_mismatch`).
- SSH mode control channel auto-populates binding metadata:
  - auto-issued sessions include `ssh_binding`
  - subsequent commands include `_ssh_session_binding`.

## 11) Audit listing filtering/pagination

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- `auth-list-sessions` supports:
  - filters: `key_id`, `scope`, `role`, `token_preview_contains`
  - pagination: `limit`, `offset`
- `auth-list-issued-tokens` supports:
  - filters: `engine_id`, `resource_kind`, `resource_id`, `backend_id`, `token_preview_contains`
  - pagination: `limit`, `offset`
- responses now include pagination metadata:
  - `offset`, `limit`, `count`, `has_more`, `next_offset`

## 12) IPC RPC command surface

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- synchronous RPC command:
  - `proxy-rpc-call`
- async RPC lifecycle commands:
  - `proxy-rpc-open`
  - `proxy-rpc-send`
  - `proxy-rpc-recv`
  - `proxy-rpc-close`
- traffic-scope authorization and engine allowlist enforcement for RPC commands.
- channel wrappers added for external consumers.

## 13) Asymmetric key challenge-response authentication

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- key auth methods now support:
  - `shared_secret` (existing)
  - `public_key` (new)
- new challenge commands:
  - `auth-begin-challenge`
  - `auth-complete-challenge`
- public-key keys cannot use direct `auth-issue-session`; they must use challenge flow.
- status/key listing now expose challenge/key auth metadata (`challenges_count`, `auth_method`).

## 14) IPC-only worker transport hardening

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- host-managed workers are now IPC-only (no host-managed HTTP/WSS worker transport).
- websocket command-level pass-through removed from daemon/CLI/channel.
- control config no longer exposes websocket session policy knobs.

## 15) Challenge auth telemetry hardening

Files:
- `src/hosting/engine_host_service.py`

Implemented:
- challenge lifecycle telemetry in host metrics auth block:
  - `challenge_begin_total`
  - `challenge_complete_ok`
  - `challenge_complete_failed`
  - `challenge_replay_suspected`
  - `challenge_recent_events` ring buffer
- replay-suspected tracking when challenge completion attempts reference missing/expired challenge IDs
  or invalid challenge signatures.

## 16) Challenge transport-binding assurance

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`

Implemented:
- challenge payload now embeds SSH binding claims when present:
  - `ssh_binding_target`
  - `ssh_binding_key_fingerprint`
- challenge completion enforces matching presented SSH binding when challenge was bound.

Security hole mitigated:
- Prevents cross-transport relay of captured signed challenges within TTL.
  Previously, an attacker who obtained a valid challenge signature might attempt completion
  from a different SSH transport context. With binding enforcement, completion must originate
  from the same bound SSH identity context (target/fingerprint), reducing replay/relay risk.

## 17) Daemon-native claim ACL enforcement and denial contract

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_channel.py`
- `src/hosting/engine_host_cli.py`
- `tests/test_hosting_daemon_acl.py`

Implemented:
- daemon-native durable claim actor identity:
  - claim actor is derived from authenticated session (`key:<key_id>`)
  - daemon command path no longer trusts caller-supplied `backend_id` for claim identity
- daemon-enforced claim checks for sensitive daemon command handlers:
  - `spawn`, `get-registration`, `shutdown`, `ensure-running`, `remove-registration`
  - `logs-tail`, `logs-follow`, `inspect-capabilities`
  - `claim-engine`, `claim-endpoint`, `claim-resource`
  - `issue-token`, `issue-resource-token`
- daemon-side keepalive + orphan takeover policy:
  - owner keepalive state persisted in control state (`claim_owner_keepalive`)
  - TTL policy in `control_config.claim_acl_policy.owner_ttl_seconds`
  - explicit claim transition values: `joined_shared`, `refreshed`, `orphan_takeover`, `force_override`
- daemon-side localhost force override confirmation:
  - `force_override=true` requires `force_override_confirmation="CONFIRM_LOCALHOST_FORCE_OVERRIDE"` on localhost command path
- daemon-side non-localhost shared-claim denial:
  - non-localhost callers are denied if claim command requests shared mode (`exclusive=false`)
- structured daemon denial contract:
  - daemon error response now includes `error_code` and `error_details`
  - denied command results are surfaced with stable machine codes
- stable claim audit events in control state:
  - `claim_audit_events` with schema version `1`
  - grant/deny/takeover/override events are appended with bounded retention (`audit_event_limit`)

Regression coverage added (daemon command path):
- unauthorized command denied
- non-member denied on shared claim
- exclusive owner conflict denied
- orphan takeover allowed only per policy
- localhost force override requires explicit confirmation token
- non-localhost shared claim denied

Compatibility note for consumers:
- minimum behavior requirement published as `Hosting ACL Contract v2`
- downstream consumers (`mp13-docs`) should require v2 fields/codes for claim-sensitive UX mapping

## 18) Daemon version pinning and capability contract fields

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `tests/test_engine_host_channel.py`
- `tests/test_hosting_http_ingress.py`

Implemented:
- stable SemVer daemon version field:
  - `daemon_version` now included in `auth-status`
  - `daemon_version` now included in `get-control-config`
- capability flags for machine gating:
  - `capabilities.claim_acl_v2`
  - `capabilities.structured_denials_v1`
  - `capabilities.force_override_confirmation_v1`
  - `capabilities.ipc_rpc_v1`
- structured denial envelope stability improvements:
  - daemon RPC responses now consistently include `error_code` and `error_details` on parse/auth/access/internal failures
- transport path consistency:
  - HTTP ingress `/health` now includes `daemon_version` and `capabilities` from the same service contract
- contract regression coverage:
  - SemVer validation for `auth-status.daemon_version`
  - cross-path equality check between daemon RPC `auth-status` and HTTP ingress `/health`

## 19) ACL Denial Smoke Validation (runtime)

Validation date: 2026-03-08

Executed:
- direct daemon `_dispatch(...)` smoke script covering ACL regression scenarios equivalent to `tests/test_hosting_daemon_acl.py`

Observed:
- unauthorized command denial: `session_token_required`
- shared-claim non-member denial: `engine_shared_claim_not_member`
- exclusive owner conflict denial: `exclusive_owner_conflict`
- orphan takeover transition allowed: `orphan_takeover`
- localhost force-override confirmation gating:
  - denied without confirmation: `localhost_force_override_confirmation_required`
  - allowed with confirmation: `force_override`
- non-localhost shared claim denial: `non_localhost_shared_claim_denied`

Notes:
- pytest invocation in this environment is currently blocked by filesystem ACL issues around pytest temp/cache dirs, so this run used direct daemon dispatch smoke validation instead.

## Not Implemented Yet

None (for the currently tracked hosting_status scope).

## Operational Notes

1. Bootstrap:
   - create first admin key
   - enable `require_auth=true`
   - issue short-lived sessions for client operations
2. Prefer SSH transport to reduce replay/timing exposure at network layer.
3. Rotate keys regularly and keep session TTL short.
4. For external GUI/backend consumers:
   - fetch endpoint metrics via the selected endpoint channel (not always local)
   - include host auth material in the endpoint/profile:
     `engine_host_session_token` or `engine_host_key_id` + `engine_host_key_secret`

## Suggested Next Step

Monitor production behavior and tune challenge/IPC-RPC defaults (stream concurrency/queue limits/cancel behavior) based on operational telemetry.

## 20) Clean-Slate Auth/AuthZ Rewrite Slice (2026-03-14)

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`
- `src/hosting/hosting_config.py`
- `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`

Implemented:
1. New role catalog support in auth model:
   - `admin`, `config_editor`, `worker_user`, `model_user_with_model_control`, `model_user`, `diagnostic_user`, `transport`
2. Role-command authorization checks in command policy path (in addition to scope checks).
3. `require_auth=false` safe-profile enforcement:
   - local-only connectivity profile only
   - admin-only key profile constraints
   - reject unsafe unauth runtime configurations
4. Runtime policy validation on daemon startup to fail fast on unsafe unauth configurations.
5. New `hosting_config` setup/reconfig utility:
   - intent-driven setup (local/tunnel/remote)
   - key import/generation support
   - client key mapping output
   - legacy key-file rename migration to `.migrated`
6. Explicit client-breaking migration guide added and linked from hosting docs.
7. Endpoint mode default and runtime override support:
   - persisted `endpoint_mode_default` in control config
   - daemon runtime override commands (`set-endpoint-mode-override`, `get-endpoint-mode-effective`)
   - omitted `exclusive` claim payload now follows effective endpoint mode
8. `hosting_config --doctor` diagnostics command:
   - SSH dependency check
   - config path existence/writability checks
   - control-config readability + runtime policy safety validation
9. Key generation robustness improvement in setup tool:
   - generate in temp path first
   - fallback to direct target generation when temp-path generation fails
10. Added setup-script test suite:
   - `tests/test_hosting_config.py`
   - validates setup outputs, safe-profile guardrails, legacy `.migrated` migration flow, and `--doctor` diagnostics behavior
11. Added setup bootstrap snapshot artifact:
   - `hosting/bootstrap/bootstrap_state.json`
   - captures effective setup policy and key/file references for reconfigure and troubleshooting flows
12. Added model override role semantics:
   - `connect-from-config` with `model_path` override now requires `model_user_with_model_control` or higher
13. Added emergency force-override contract and policy checks:
   - `force_override_reason` required for override claims
   - localhost non-emergency override requires confirmation token
   - localhost emergency override can bypass confirmation for stale/malicious/security reasons
14. Added high-severity claim audit tagging for force/emergency overrides:
   - claim audit events now include `severity`
15. Extended claim command payload surface across daemon/channel:
   - `force_override_reason`
   - `force_override_emergency`
16. Added deterministic displaced-owner lifecycle:
   - override displacement creates persisted ownership-change notice
   - non-claim commands for displaced actor denied with `ownership_changed_reclaim_required`
   - successful reclaim clears notice
17. Added generic worker profile role gating for `connect-from-config`:
   - worker profile classification (`model` vs `generic`) from config metadata
   - generic profile usage requires `worker_user` (or higher)
   - `model_user` and `model_user_with_model_control` denied generic profile usage
18. Added generic worker runtime support in `connect-from-config`:
   - generic profile skips model selection flow
   - generic spawn command from `worker_command` or `spawn.command`
   - explicit failure reason `generic_worker_command_missing` when command is absent
19. Added runtime generic-engine communication enforcement:
   - engine registration now includes `worker_profile_class`
   - model roles are denied proxy/rpc authorization on generic engine registrations
20. Hardened orthogonal transport role onboarding:
   - `transport` keys must be `public_key` auth method
   - transport shared-secret upsert is rejected
   - transport role cannot issue sessions/challenges for command auth
21. Hardened remote auth bootstrap SSH binding requirement:
   - in non-local connectivity profiles, shared-secret session issuance is denied (`shared_secret_bootstrap_not_supported_for_remote_connectivity`)
   - in non-local connectivity profiles, public-key challenge begin requires `ssh_binding`
   - missing binding denied with `ssh_binding_required_for_remote_connectivity`
22. Hardened non-local command-path SSH binding requirement:
   - session-backed command authorization requires presented `_ssh_session_binding`
   - session must be SSH-bound in non-local connectivity modes
   - legacy unbound sessions denied with `ssh_binding_required_for_remote_connectivity`
23. Added auth lifecycle audit events in control state:
   - `auth_upsert_key`
   - `auth_revoke_key`
   - `auth_revoke_session`
24. Added explicit role coverage for admin-only invalidation controls:
   - `config_editor` denied `auth-revoke-key` / `auth-revoke-session` (`insufficient_role`)
   - `admin` allowed
25. Added admin query command for auth lifecycle audit:
   - `auth-audit-list` with paging + filters
   - wired in service/daemon/channel/CLI surfaces
26. Added role coverage for `auth-audit-list` authorization:
   - `config_editor` denied (`insufficient_role`)
   - admin allowed and filter path validated
27. Added lifecycle profile + policy baseline in control config:
   - `lifecycle_profile` (`foreground_terminal_bound|detached_user_process|service_managed`)
   - `lifecycle_policy` (`on_terminal_disconnect`, `terminal_control_enabled`, `owner_disconnect_shutdown`)
28. Added lifecycle policy effective inspection command across service/daemon/CLI/channel:
   - `get-lifecycle-policy-effective`
29. Added lifecycle profile regression coverage:
   - service-managed defaults
   - invalid profile rejection
   - lifecycle policy override persistence
30. Added daemon lifecycle enforcement hooks:
   - owner-disconnect shutdown path for exclusive endpoint owner when `lifecycle_policy.owner_disconnect_shutdown=true`
   - foreground terminal-disconnect policy handling (SIGHUP ignore for keep-running mode where supported)
31. Added detached runtime profile hint in background bootstrap path:
   - daemon background start includes `--runtime-profile detached_user_process`
32. Added lifecycle enforcement regression tests:
   - `tests/test_hosting_daemon_pidfile.py` (runtime-profile + foreground policy hook)
   - `tests/test_hosting_daemon_acl.py` (owner-disconnect shutdown enabled/disabled)
33. Extended `hosting_config` setup flow with lifecycle profile input:
   - setup accepts `--lifecycle-profile` (`foreground_terminal_bound|detached_user_process|service_managed`)
   - lifecycle profile is persisted into control state and setup/access artifacts
34. Added setup-script lifecycle profile regression test coverage:
   - `tests/test_hosting_config.py` includes lifecycle profile persistence assertions
35. Added daemon shutdown-order checkpoints:
   - daemon stop path now enumerates managed registrations and attempts orderly worker shutdown
   - checkpoint report tracks attempted/stopped/failed outcomes and registration counts before/after stop
36. Added shutdown-checkpoint regression tests:
   - `tests/test_hosting_daemon_pidfile.py` now covers checkpoint ordering and discovery-failure handling
37. Strengthened daemon shutdown sequencing with operation drain:
   - in-flight async operation tasks are drained before managed worker shutdown checkpoints run
   - shutdown stage events are captured for sequencing diagnostics
38. Added operation-drain regression coverage:
   - `tests/test_hosting_daemon_pidfile.py` validates pending operation drain behavior
39. Enforced terminal control gating in daemon runtime:
   - when `terminal_control_enabled=false`, daemon denies:
     - `__shutdown__`
     - `set-endpoint-mode-override`
40. Hardened runtime endpoint-mode override auth path:
   - daemon now enforces auth for `set-endpoint-mode-override` and `get-endpoint-mode-effective`
41. Added daemon ACL regression tests for terminal-control policy and override auth:
   - shutdown denial under disabled terminal control
   - override denial under disabled terminal control
   - override requires session token when auth is enabled
42. Phase 8 cutover slice: removed legacy role bridge from app host-auth helper path.
   - `src/app/config.py` now accepts only clean-slate role names for `--host-auth-role`:
     - `admin`, `config_editor`, `worker_user`, `model_user_with_model_control`, `model_user`, `diagnostic_user`
43. Updated security/ingress/app-config tests to clean-slate role names:
   - `tests/test_hosting_service_security.py`
   - `tests/test_hosting_http_ingress.py`
   - `tests/test_app_config_host_auth.py`
44. Updated user-facing hosting docs to remove remaining legacy role examples:
   - `src/hosting/HOSTING.md` bootstrap/traffic/mp13config examples now use clean-slate roles only.
45. Phase 8 closure status:
   - legacy-role bridge removal + regression validation is complete for this tracked cutover scope.
46. Phase 7 planning doc drafted with risk-gate contract (kept as planned):
   - `src/hosting/hosting_phase7_hardening.md`
47. Temp-root behavior investigation completed for hosting suites:
   - root `.tmp_*` folders were test-helper-created (`Path.cwd() / ".tmp_*"`), not pytest defaults
   - updated tests now allocate workspaces from `PYTEST_DEBUG_TEMPROOT` instead
48. Pytest no-`--basetemp` default improved via `tests/conftest.py`:
   - fallback order: repo parent `.mp13_pytest` -> system temp -> repo-local `.tmp_pytest`
49. Validation without explicit `--basetemp` now passes:
   - `pytest tests/test_hosting_auth_roles.py -q` -> `26 passed, 2 warnings`
   - `pytest tests/test_hosting_config.py -q` -> `6 passed, 2 warnings`

Notes:
1. Lifecycle Phase 6 implementation scope is closed in code/docs and validated by outside-sandbox daemon-suite reruns.
2. In this environment, pytest remains blocked by filesystem ACL issues for tmp/cache handling; validation used py-compile and direct smoke runs for `hosting_config`.
3. Targeted role-gating tests pass when using repo-local basetemp:
   - `pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role6`
   - latest result: 7 passed (cache warnings only)
4. `hosting_config` setup+doctor smoke passed for import-key flow in `.tmp_hosting_config_doctor_smoke3` with `issues_count=0`.
5. Generated-key setup flow cannot be fully validated in this mapped-drive sandbox because OpenSSH keygen write behavior is filesystem-restricted here (`Bad file descriptor`/permission errors); fallback logic is implemented and documented.
6. New setup-script test run:
   - `pytest tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_hosting_config`
   - result: 5 passed (cache warnings only)
7. Updated role-gating test run:
   - `pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role8`
   - result: 9 passed (cache warnings only)
8. Combined role + setup suite run:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg4`
   - result: 14 passed (cache warnings only)
9. Daemon ACL emergency-override scenarios validated via in-workspace harness:
   - all `tests/test_hosting_daemon_acl.py` test functions pass under workspace temp directories
   - pytest execution of this module remains blocked by environment tmpdir ACL teardown behavior
10. Updated daemon ACL harness validation now includes displaced-owner reclaim lifecycle test:
   - `test_displaced_owner_is_denied_until_reclaim_then_cleared`
11. Latest role + setup suite run:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg7`
   - result: 16 passed (cache warnings only)
12. Updated role + setup suite run:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg10`
   - result: 16 passed (cache warnings only)
13. `tests/test_hosting_service_list_configs.py` (including new generic runtime tests) validated via workspace harness:
   - all test functions pass
   - direct pytest for this module still affected by environment tmpdir ACL teardown behavior
14. Latest role + setup suite run:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg12`
   - result: 18 passed (cache warnings only)
15. Latest role-only suite run after transport hardening:
   - `pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role12`
   - result: 15 passed (cache warnings only)
16. Latest role + setup suite run after remote binding hardening:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg14`
   - result: 22 passed (cache warnings only)
17. Latest role + setup suite run after command-path binding hardening:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg15`
   - result: 24 passed (cache warnings only)
18. Latest role + setup suite run after auth audit + admin control coverage:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg16`
   - result: 26 passed (cache warnings only)
19. Latest role + setup suite run after `auth-audit-list` command integration:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg17`
   - result: 28 passed (cache warnings only)
20. Latest role + channel suite run after lifecycle profile/policy changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg18`
   - result: 34 passed (cache warnings only)
21. Lifecycle enforcement pytest commands currently blocked in this sandbox by teardown ACL restrictions:
   - `pytest tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg19`
   - `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl19`
   - failure reason: `PermissionError: [WinError 5] Access is denied` on basetemp cleanup (`cleanup_dead_symlinks`)
22. Syntax validation run (sandbox): pass
   - `python -m py_compile src/hosting/engine_host_service.py src/hosting/engine_host_daemon.py src/hosting/engine_host_cli.py tests/test_hosting_daemon_pidfile.py tests/test_hosting_daemon_acl.py`
23. Setup-script suite run after lifecycle profile setup changes:
   - `pytest tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_hosting_config2`
   - result: 6 passed (cache warnings only)
24. Shutdown-checkpoint pytest command in sandbox:
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid20`
   - tests execute, then pytest teardown fails with `PermissionError: [WinError 5] Access is denied` on basetemp cleanup
25. Syntax validation run (shutdown checkpoint slice): pass
   - `python -m py_compile src/hosting/engine_host_daemon.py tests/test_hosting_daemon_pidfile.py`
26. Updated shutdown-sequencing pytest command in sandbox:
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid21`
   - tests execute, then pytest teardown fails with `PermissionError: [WinError 5] Access is denied` on basetemp cleanup
27. Updated daemon ACL pytest command in sandbox (terminal-control policy tests included):
   - `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl22`
   - tests execute, then pytest teardown fails with `PermissionError: [WinError 5] Access is denied` on basetemp cleanup
28. Updated daemon pidfile pytest command in sandbox (operation-drain sequencing tests included):
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid22`
   - tests execute, then pytest teardown fails with `PermissionError: [WinError 5] Access is denied` on basetemp cleanup
29. Syntax validation run after terminal-control/override-auth hardening: pass
   - `python -m py_compile src/hosting/engine_host_daemon.py tests/test_hosting_daemon_acl.py`
30. Latest role + channel lifecycle-control rerun:
   - `pytest tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg20`
   - result: 34 passed, 2 warnings (pytest cache ACL warnings only)
31. Latest daemon lifecycle suite reruns remain teardown-blocked in sandbox:
   - `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl23`
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid23`
   - failure reason: `PermissionError: [WinError 5] Access is denied` during pytest session teardown (`cleanup_dead_symlinks`) on `--basetemp`.
32. Manual outside-sandbox lifecycle daemon-suite reruns completed:
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid23`
   - result: 14 passed in 0.24s
   - `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl23`
   - result: 15 passed in 1.17s
33. Phase 8 targeted pytest commands in sandbox:
   - `pytest tests/test_app_config_host_auth.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_appcfg`
   - `pytest tests/test_hosting_service_security.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_security`
   - `pytest tests/test_hosting_http_ingress.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_ingress`
   - all three fail during pytest session teardown with `PermissionError: [WinError 5] Access is denied` in pytest `cleanup_dead_symlinks(...)`.
34. Manual outside-sandbox Phase 8 regression reruns completed:
   - `pytest tests/test_app_config_host_auth.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_appcfg`
   - result: 1 passed in 0.10s
   - `pytest tests/test_hosting_service_security.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_security`
   - result: 14 passed in 0.86s
   - `pytest tests/test_hosting_http_ingress.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_ingress`
   - result: 3 passed in 2.53s
35. Phase 5 hardening: no-auth config drift is now blocked in `set-control-config` even when `require_auth` is omitted from payload.
36. Phase 5 hardening: no-auth mode now rejects session/challenge issuance bootstrap paths:
   - `auth-issue-session`
   - `auth-begin-challenge`
   - `auth-complete-challenge`
   - denial: `require_auth_disabled_disallows_session_commands`
37. Added role regression coverage for the above hardening:
   - `test_require_auth_false_rejected_when_profile_drifts_without_require_auth_field`
   - `test_require_auth_false_rejects_session_and_challenge_issue_paths`
38. Validation reruns (no explicit `--basetemp`):
   - `pytest tests/test_hosting_auth_roles.py -q` -> `28 passed, 2 warnings`
   - `pytest tests/test_hosting_config.py -q` -> `6 passed, 2 warnings`
39. Phase 2 ownership-enforcement consistency update:
   - daemon special endpoint-mode handlers now apply displaced-owner claim policy checks.
   - displaced owners are denied until reclaim for:
     - `set-endpoint-mode-override`
     - `get-endpoint-mode-effective`
   - denial code: `ownership_changed_reclaim_required`
40. Updated daemon ACL test coverage:
   - `test_displaced_owner_is_denied_until_reclaim_then_cleared` now asserts endpoint-mode command denial for displaced owner.
41. Sandbox daemon ACL pytest rerun status:
   - `pytest tests/test_hosting_daemon_acl.py -q`
   - blocked in fixture setup by temp-root ACL error:
     - `PermissionError: [WinError 5] Access is denied`
     - `C:\Users\me\AppData\Local\Temp\mp13_pytest\pytest-of-me`
   - command recorded for manual outside-sandbox rerun.
42. Syntax validation (sandbox): pass
   - `python -m py_compile src/hosting/engine_host_daemon.py tests/test_hosting_daemon_acl.py`
43. Manual outside-sandbox rerun for updated daemon ACL suite:
   - `pytest tests/test_hosting_daemon_acl.py -q`
   - result: `15 passed in 1.10s`
44. Pre-Phase-7 baseline freeze:
   - Phases 0-5 are now formally closed in `hosting_access_plan.md` with exit-criteria evidence mapping.
   - Phase 6 and Phase 8 remain closed.
   - Phase 7 remains planned/risk-gated via `hosting_phase7_hardening.md`.
