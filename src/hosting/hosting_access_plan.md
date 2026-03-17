# Hosting Access Hardening Plan

Date: 2026-03-16
Status: Living implementation plan (functional-first, security staged)

References:
- `src/hosting/hosting_access.md`
- `src/hosting/HOSTING.md`
- `src/hosting/HOSTING_PYTEST_STATUS.md`
- `src/hosting/hosting_status.md`
- `src/hosting/hosting_phase7_hardening.md`

## 1. Guiding priorities

1. Deliver functional and usable access flows first.
2. Keep baseline no-admin operation for Windows and Linux.
3. Treat SSH as required dependency for remote-capable operation.
4. Gate advanced hardening by documented threat/risk and scope impact.
5. Execute as a breaking clean-slate auth/authz redesign (no backward compatibility constraints).

## 1.1 Progress update (2026-03-14)

Completed in this implementation slice:
1. Introduced clean-slate role catalog in service auth layer.
2. Added command-level role authorization checks in daemon/service command path.
3. Added safe-profile enforcement for `require_auth=false` with `access_profile.connectivity_mode`.
4. Plumbed `access_profile` through service/daemon/CLI/channel `set-control-config`.
5. Added initial `hosting_config` implementation (`python -m hosting.hosting_config` and `--hosting-config` CLI mode).
6. Added `hosting_config` script specification document:
   - `src/hosting/hosting_config_script.md`
7. Added explicit client-breaking migration document:
   - `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`
8. Enforced runtime safe-profile policy when `require_auth=false`:
   - daemon startup now validates policy invariants
   - command authorization path rejects unsafe unauth configurations
9. Implemented legacy key-file rename migration in `hosting_config`:
   - rename to `.migrated`/`.migrated.N`
   - emit audit + migrations metadata
10. Added clean-slate auth role/safe-profile regression tests:
   - `tests/test_hosting_auth_roles.py`
11. Updated pytest status docs with new role-gating test command and environment note.
12. Added `HOSTING.md` setup wizard usage examples for `hosting_config`.
13. Validated clean-slate role/safe-profile tests:
   - `pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role4`
   - result: 6 passed (cache warnings only due workspace ACLs)
14. Added endpoint default mode contract plumbing:
   - `endpoint_mode_default` in control config set/get
   - claim commands use daemon default when `exclusive` is omitted
15. Added daemon runtime endpoint-mode override commands:
   - `set-endpoint-mode-override` (temporary until daemon shutdown)
   - `get-endpoint-mode-effective`
16. Extended targeted tests to validate runtime override claim behavior:
   - `pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role6`
   - result: 7 passed (cache warnings only due workspace ACLs)
17. Added initial setup diagnostics command:
   - `python -m hosting.hosting_config --doctor`
   - checks SSH dependency, config/keyring paths, write probe, control-config readability, runtime policy safety
18. Validated setup + diagnostics smoke flow (import-key path):
   - setup: `python -m hosting.hosting_config --mode local_only --endpoint-mode exclusive --require-auth --key-source import ...`
   - doctor: `python -m hosting.hosting_config --doctor --default-config-dir ... --control-state-file ...`
   - result: `status=ok`, `issues_count=0`
19. Improved generated-key path resilience:
   - key generation now attempts temp-path generation then direct-path fallback
   - this reduces filesystem-specific keygen failures in mixed Windows/Linux environments
20. Revalidated auth role suite after setup-tool changes:
   - `pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role7`
   - result: 7 passed (cache warnings only due workspace ACLs)
21. Added setup-script coverage tests:
   - `tests/test_hosting_config.py`
   - covers setup artifact creation, safe-profile rejection, `.migrated` migration metadata, and `--doctor` healthy/unsafe cases
22. Validated setup-script tests in sandbox:
   - `pytest tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_hosting_config`
   - result: 5 passed (cache warnings only due workspace ACLs)
23. Added setup bootstrap state artifact:
   - `Hosting/state/bootstrap_state.json`
   - includes effective setup profile, key source, and managed file references for lifecycle/debug automation
24. Added role gate for model override semantics:
   - `connect-from-config` with explicit `model_path` now requires `model_user_with_model_control` or higher
25. Extended role-gating tests for model override behavior:
   - `tests/test_hosting_auth_roles.py` now includes model override deny/allow coverage
   - `pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role8`
   - result: 9 passed (cache warnings only due workspace ACLs)
26. Implemented force-override reason/emergency contract in claim policy path:
   - new payload fields: `force_override_reason`, `force_override_emergency`
   - new denial codes: `force_override_reason_required`, `force_override_emergency_reason_invalid`
27. Implemented emergency localhost override path for stale/malicious ownership scenarios:
   - confirmation can be bypassed only for emergency reasons
   - non-emergency localhost overrides still require confirmation token
28. Added high-severity claim audit tagging for force/emergency overrides:
   - claim audit events now include `severity`
29. Added/updated daemon ACL tests for emergency override semantics:
   - reason required test
   - emergency no-confirmation grant test
   - high-severity audit verification for emergency grant
30. Validated daemon ACL tests via in-workspace harness (pytest tmpdir ACL constraints in this environment):
   - all `tests/test_hosting_daemon_acl.py` scenarios executed with per-test workspace directories
   - result: all pass in harness execution
31. Revalidated role + setup suites after emergency override changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg4`
   - result: 14 passed (cache warnings only due workspace ACLs)
32. Implemented deterministic post-takeover owner lifecycle:
   - displaced owner notice persisted in control state (`ownership_change_notices`)
   - non-claim commands denied with `ownership_changed_reclaim_required` until reclaim
33. Added daemon ACL test coverage for displaced-owner denial and reclaim-clear flow:
   - `test_displaced_owner_is_denied_until_reclaim_then_cleared`
34. Revalidated role + setup suites after ownership-notice changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg5`
   - result: 14 passed (cache warnings only due workspace ACLs)
35. Revalidated full daemon ACL scenarios via workspace harness (tmpdir ACL-safe):
   - all `tests/test_hosting_daemon_acl.py` functions pass, including new displaced-owner reclaim flow
36. Implemented generic worker profile role gating for `connect-from-config`:
   - worker profile classified from config metadata (`model` vs `generic`)
   - generic profile requires `worker_user` or higher
   - model roles are denied generic profile usage (`insufficient_role`)
37. Added role tests for generic worker profile policy:
   - `model_user` denied generic profile
   - `worker_user` allowed generic profile
38. Revalidated role + setup suites after generic profile policy changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg7`
   - result: 16 passed (cache warnings only due workspace ACLs)
39. Implemented generic worker runtime support in `connect-from-config`:
   - model selection is skipped for generic worker profiles
   - generic command sourced from `worker_command` or `spawn.command`
   - explicit failure path when generic command is missing (`generic_worker_command_missing`)
40. Added runtime tests for generic `connect-from-config` flow:
   - generic spawn success without `model_path`
   - generic missing-command failure contract
41. Validated `test_hosting_service_list_configs.py` scenarios via workspace harness (pytest tmpdir ACL constraint in this environment):
   - all tests pass, including new generic runtime tests
42. Revalidated role + setup suites after runtime generic flow changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg10`
   - result: 16 passed (cache warnings only due workspace ACLs)
43. Added runtime generic-engine communication enforcement:
   - engine registrations carry `worker_profile_class`
   - model roles are denied proxy/rpc authorization for generic engine registrations
44. Added role tests for generic engine proxy enforcement:
   - model role denied proxy to generic registered engine
   - worker role allowed proxy to generic registered engine
45. Revalidated role + setup suites after proxy enforcement changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg12`
   - result: 18 passed (cache warnings only due workspace ACLs)
46. Hardened orthogonal `transport` role onboarding constraints:
   - `transport` keys now require `auth_method=public_key`
   - shared-secret transport key creation is rejected
47. Added transport role regression tests:
   - reject shared-secret upsert for `transport`
   - reject session/challenge issuance for public-key transport key
48. Revalidated role test slice after transport hardening:
   - `pytest tests/test_hosting_auth_roles.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role12`
   - result: 15 passed (cache warnings only due workspace ACLs)
49. Hardened remote auth bootstrap paths to require SSH binding:
   - non-local `auth_issue_session` is denied (`shared_secret_bootstrap_not_supported_for_remote_connectivity`)
   - non-local `auth_begin_challenge` requires `ssh_binding`
   - missing binding denied with `ssh_binding_required_for_remote_connectivity`
50. Added role tests for remote SSH-binding bootstrap requirement:
   - shared-secret session issue denied with or without binding in non-local profiles
   - public-key challenge begin denied without binding, allowed with binding
51. Revalidated role + setup suites after remote binding hardening:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg14`
   - result: 22 passed (cache warnings only due workspace ACLs)
52. Hardened non-local command authorization to require SSH binding context:
   - session-backed command path now requires presented `_ssh_session_binding` in non-local modes
   - session must contain persisted SSH binding metadata
53. Added tests for command-path SSH-binding enforcement:
   - remote mode command denial when `_ssh_session_binding` is missing
   - legacy unbound session denied after connectivity profile flips to non-local
54. Revalidated role + setup suites after command-path SSH-binding hardening:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg15`
   - result: 24 passed (cache warnings only due workspace ACLs)
55. Added explicit auth audit trail events:
   - `auth_upsert_key`
   - `auth_revoke_key`
   - `auth_revoke_session`
56. Added role tests for admin-only invalidation controls:
   - non-admin (`config_editor`) denied `auth-revoke-key` / `auth-revoke-session`
   - admin authorized for those commands
57. Revalidated role + setup suites after admin-control/audit changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg16`
   - result: 26 passed (cache warnings only due workspace ACLs)
58. Added admin audit query command across service/daemon/channel/CLI:
   - `auth-audit-list` with filters (`event_type`, `actor_key_id`, `target_key_id`, `result`) and paging (`limit`, `offset`)
59. Added role tests for `auth-audit-list`:
   - `config_editor` denied (`insufficient_role`)
   - admin can list and filter auth audit events
60. Revalidated role + setup suites after audit-list command changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg17`
   - result: 28 passed (cache warnings only due workspace ACLs)
61. Added lifecycle profile baseline in control config:
   - new persisted fields: `lifecycle_profile`, `lifecycle_policy`
   - supported profiles: `foreground_terminal_bound`, `detached_user_process`, `service_managed`
62. Added lifecycle policy effective inspection command:
   - `get-lifecycle-policy-effective` across service/daemon/CLI/channel
63. Added lifecycle profile regression coverage:
   - profile defaulting for `service_managed`
   - invalid profile rejection
   - lifecycle policy override persistence
64. Revalidated role + channel suites after lifecycle profile/policy changes:
   - `pytest tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg18`
   - result: 34 passed (cache warnings only due workspace ACLs)
65. Added daemon lifecycle enforcement hooks:
   - owner-disconnect shutdown enforcement for exclusive endpoint ownership when `lifecycle_policy.owner_disconnect_shutdown=true`
   - foreground terminal disconnect policy hook honoring `on_terminal_disconnect` (SIGHUP ignore in keep-running mode where supported)
66. Wired detached runtime profile hint for background daemon bootstrap:
   - `start_daemon_background` now passes `--runtime-profile detached_user_process`
67. Added lifecycle enforcement regression tests:
   - daemon pidfile/startup tests for runtime-profile argv + foreground disconnect policy hook
   - daemon ACL tests for owner-disconnect shutdown enabled/disabled behavior
68. Sandbox execution note for lifecycle enforcement test commands:
   - both relevant pytest commands hit `PermissionError: [WinError 5] Access is denied` during pytest session teardown on basetemp directories in this environment
   - commands and reasons are recorded in `HOSTING_PYTEST_STATUS.md` for manual rerun outside sandbox.
69. Extended `hosting_config` setup contract for lifecycle profile selection:
   - new setup input: `--lifecycle-profile foreground_terminal_bound|detached_user_process|service_managed`
   - lifecycle profile persisted to control config and setup artifacts
70. Added setup-script lifecycle profile tests:
   - `tests/test_hosting_config.py` validates lifecycle profile persistence in control/access artifacts
71. Validated setup-script suite after lifecycle setup updates:
   - `pytest tests/test_hosting_config.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_hosting_config2`
   - result: 6 passed (cache warnings only due workspace ACLs)
72. Implemented daemon shutdown-order checkpoints for clean resource release:
   - on daemon stop, managed engine registrations are enumerated and orderly shutdown is attempted
   - checkpoint summary tracks attempted/stopped/failed counts and before/after registration counts
73. Added shutdown-checkpoint regression tests:
   - `tests/test_hosting_daemon_pidfile.py` coverage for checkpoint ordering and failure handling
74. Sandbox execution note for shutdown-checkpoint pytest command:
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid20`
   - tests executed, but pytest teardown failed with `PermissionError: [WinError 5] Access is denied` on basetemp cleanup
   - command and reason recorded in `HOSTING_PYTEST_STATUS.md` for manual rerun outside sandbox
75. Strengthened shutdown sequencing with operation drain stage:
   - daemon shutdown now drains in-flight async `op-start` tasks before managed worker shutdown checkpoints
   - shutdown stage events are captured for sequence diagnostics (`shutdown.begin`, `shutdown.operations_drain`, `shutdown.managed_workers`)
76. Added operation-drain regression coverage:
   - `tests/test_hosting_daemon_pidfile.py` validates drain of pending operation tasks before shutdown continuation
77. Sandbox execution note for updated shutdown-sequencing tests:
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid21`
   - tests executed, but pytest teardown failed with `PermissionError: [WinError 5] Access is denied` on basetemp cleanup
78. Enforced terminal control gating in daemon runtime paths:
   - when `terminal_control_enabled=false`, daemon denies terminal control operations:
     - `__shutdown__`
     - `set-endpoint-mode-override`
79. Hardened runtime endpoint-mode override dispatch auth path:
   - daemon now requires command authorization for `set-endpoint-mode-override` and `get-endpoint-mode-effective`
80. Added daemon ACL regression coverage for terminal-control policy:
   - deny shutdown token path when terminal control is disabled
   - deny runtime endpoint-mode override when terminal control is disabled
   - require auth token for runtime endpoint-mode override
81. Sandbox execution note for terminal-control ACL test command:
   - `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl22`
   - tests executed, but pytest teardown failed with `PermissionError: [WinError 5] Access is denied` on basetemp cleanup
82. Sandbox execution note for updated daemon pidfile sequence command:
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid22`
   - tests executed, but pytest teardown failed with `PermissionError: [WinError 5] Access is denied` on basetemp cleanup
83. Phase 6 implementation scope is functionally complete in code/docs:
   - lifecycle profiles + survival rules + terminal control gating + shutdown sequencing/checkpoints are implemented
   - final full-pass evidence requires manual rerun of recorded pytest commands outside this sandbox.
84. Final Phase 6 in-sandbox validation rerun:
   - `pytest tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_role_cfg20`
   - result: 34 passed, 2 warnings (pytest cache ACL warnings only)
85. Final Phase 6 daemon-suite reruns remain teardown-blocked in this sandbox:
   - `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl23`
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid23`
   - both fail during pytest session teardown with `PermissionError: [WinError 5] Access is denied` in `cleanup_dead_symlinks(...)` for `--basetemp`.
86. Manual outside-sandbox daemon-suite reruns completed:
   - `pytest tests/test_hosting_daemon_pidfile.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_pid23`
   - result: 14 passed in 0.24s
   - `pytest tests/test_hosting_daemon_acl.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_daemon_acl23`
   - result: 15 passed in 1.17s
87. Phase 8 cutover started: removed remaining legacy role bridge from `mp13config` host-auth helper path.
   - `src/app/config.py` host auth role validation now accepts only clean-slate roles:
     - `admin|config_editor|worker_user|model_user_with_model_control|model_user|diagnostic_user`
   - legacy role names are no longer accepted by this helper path.
88. Updated regression tests/examples to clean-slate role names:
   - `tests/test_hosting_service_security.py` role updates (`admin` / `model_user`)
   - `tests/test_hosting_http_ingress.py` role updates (`model_user`)
   - `tests/test_app_config_host_auth.py` role update (`admin`)
   - `src/hosting/HOSTING.md` bootstrap examples updated (`admin` / `model_user`)
89. Sandbox execution note for Phase 8 targeted test commands:
   - `pytest tests/test_app_config_host_auth.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_appcfg`
   - `pytest tests/test_hosting_service_security.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_security`
   - `pytest tests/test_hosting_http_ingress.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_ingress`
   - all three fail during pytest session teardown with `PermissionError: [WinError 5] Access is denied` at pytest `cleanup_dead_symlinks(...)`.
90. Manual outside-sandbox validation for Phase 8 cutover regression slice completed:
   - `pytest tests/test_app_config_host_auth.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_appcfg` -> `1 passed in 0.10s`
   - `pytest tests/test_hosting_service_security.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_security` -> `14 passed in 0.86s`
   - `pytest tests/test_hosting_http_ingress.py -q --basetemp o:\repos\mp13-llm-engine\.tmp_pytest_phase8_ingress` -> `3 passed in 2.53s`
91. Phase 8 closure check completed:
   - runtime legacy role bridge removed from `mp13config` host-auth helper path
   - clean-slate role-only contract documented in client-breaking + user docs
   - regression coverage for app host-auth helper + hosting security/ingress paths validated outside sandbox
92. Phase 7 planning doc drafted and aligned to existing plan scope:
   - `src/hosting/hosting_phase7_hardening.md`
   - keeps Phase 7 risk-gated and non-default with explicit threat/impact gates
93. Test temp-root investigation completed:
   - root `.tmp_*` creation cause identified in:
     - `tests/test_hosting_auth_roles.py`
     - `tests/test_hosting_config.py`
   - both now use `PYTEST_DEBUG_TEMPROOT`-based workspace directories (not `Path.cwd()/.tmp_*`)
94. Pytest default temp-root policy added for no-`--basetemp` runs:
   - `tests/conftest.py` now sets `PYTEST_DEBUG_TEMPROOT` with fallback chain:
     - repo parent `.mp13_pytest` (preferred, outside repo)
     - system temp `mp13_pytest`
     - repo-local `.tmp_pytest` (last-resort fallback)
   - validation without `--basetemp`:
     - `pytest tests/test_hosting_auth_roles.py -q` -> `26 passed, 2 warnings`
     - `pytest tests/test_hosting_config.py -q` -> `6 passed, 2 warnings`
95. Phase 5 hardening: blocked no-auth profile drift on partial `set-control-config` updates.
   - if effective config remains `require_auth=false`, safe-profile validation now runs even when `require_auth` is omitted from payload.
   - this closes unsafe transition path where connectivity could be flipped to remote without explicitly toggling `require_auth`.
96. Phase 5 hardening: blocked session/challenge issuance while auth is disabled.
   - denied commands under `require_auth=false`:
     - `auth-issue-session`
     - `auth-begin-challenge`
     - `auth-complete-challenge`
   - denial code: `require_auth_disabled_disallows_session_commands`
97. Added targeted regression coverage for new Phase 5 hardening:
   - `test_require_auth_false_rejected_when_profile_drifts_without_require_auth_field`
   - `test_require_auth_false_rejects_session_and_challenge_issue_paths`
98. Validation reruns (no `--basetemp`):
   - `pytest tests/test_hosting_auth_roles.py -q` -> `28 passed, 2 warnings`
   - `pytest tests/test_hosting_config.py -q` -> `6 passed, 2 warnings`
99. Phase 2 ownership enforcement consistency hardening:
   - daemon special-case command handlers now apply claim-policy displaced-owner checks for:
     - `set-endpoint-mode-override`
     - `get-endpoint-mode-effective`
   - displaced owners now receive deterministic denial until reclaim: `ownership_changed_reclaim_required`
100. Added daemon ACL regression coverage for endpoint-mode ownership gating:
   - extended `test_displaced_owner_is_denied_until_reclaim_then_cleared` to assert denial on:
     - `set-endpoint-mode-override`
     - `get-endpoint-mode-effective`
101. Sandbox validation note for daemon ACL rerun:
   - command attempted: `pytest tests/test_hosting_daemon_acl.py -q`
   - failure reason: setup-time tmp fixture error in sandbox temp root discovery:
     - `PermissionError: [WinError 5] Access is denied`
     - path: `C:\Users\me\AppData\Local\Temp\mp13_pytest\pytest-of-me`
   - command recorded in `HOSTING_PYTEST_STATUS.md` for manual outside-sandbox rerun.
102. Manual outside-sandbox rerun for updated daemon ACL suite completed:
   - `pytest tests/test_hosting_daemon_acl.py -q`
   - result: `15 passed in 1.10s`
103. Shared-secret bootstrap hardening applied for scenario alignment:
   - non-local connectivity now denies `auth_issue_session` regardless of `ssh_binding`
   - denial code: `shared_secret_bootstrap_not_supported_for_remote_connectivity`
   - non-local bootstrap path is public-key challenge only (`auth_begin_challenge` + `auth_complete_challenge`)
104. Updated role regression coverage for remote bootstrap policy:
   - remote shared-secret session issuance now denied with/without `ssh_binding`
   - remote command-path SSH-binding denial coverage preserved via profile-flip session flow
105. Updated user/client docs to reflect local-only shared-secret bootstrap policy:
   - `src/hosting/hosting_access.md`
   - `src/hosting/HOSTING.md`
   - `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`
106. Enforced emergency takeover eligibility predicates in daemon claim policy:
   - new denial code: `force_override_emergency_predicate_not_met`
   - reason-specific predicate checks:
     - `stale_owner_unreachable` requires orphan conflicting owner
     - `owner_malicious` and `security_incident` require active conflicting owner
107. Added daemon ACL regression tests for emergency predicate enforcement:
   - deny stale-owner emergency override while conflicting owner remains active
   - allow stale-owner emergency override when conflicting owner is orphaned
108. Updated SSH-target channel helper behavior to avoid local-only shared-secret auto bootstrap:
   - SSH mode now skips auto `auth-issue-session` from shared-secret credentials
   - SSH mode still injects `_ssh_session_binding` context on commands
109. Added host-path keygen readiness probe in `hosting_config --doctor`:
   - non-blocking check: `ssh_keygen_host_path_probe`
   - validates `ssh-keygen` write behavior under `Hosting/keyring/private`
   - probe output remains visible for pre-Phase-7 readiness tracking without blocking import-key baseline setup
110. Updated regression coverage for SSH helper and doctor probe:
   - `tests/test_engine_host_channel.py` SSH bootstrap expectation aligned to no-auto-shared-secret policy
   - `tests/test_hosting_config.py` asserts doctor reports `ssh_keygen_host_path_probe`

Notes:
1. This is an initial vertical slice; full role semantics and lifecycle policy coverage continue in later phases.
2. In this mapped-drive sandbox, `ssh-keygen` still fails key generation at runtime (`Bad file descriptor` / permission constraints). Current fallback logic is in place, but fully green generated-key validation requires a host path where OpenSSH can write keys successfully.

## 1.2 Pre-Phase-7 readiness fixes (must-close list)

1. Align status language across design and plan docs so implemented baseline does not appear as "missing."
2. Emergency takeover eligibility predicates explicit + test-mapped:
   - code/test complete; keep docs/error-catalog wording synchronized.
3. Keep scenario runbooks explicit on:
   - mitigated vs unmitigated vectors
   - minimum controls to remain in scenario
   - switch-to-next-scenario triggers
4. Keep residual-risk boundary explicit:
   - baseline does not claim local-compromise prevention
5. Maintain a reproducible validation path for suites that may fail sandbox teardown due to ACL constraints.
6. Generated-key validation on constrained Windows host paths:
   - doctor probe now records host-path keygen readiness evidence (`ssh_keygen_host_path_probe`)
   - outside-sandbox host validation reruns remain required for baseline evidence.
7. Normalize terminology in docs for connectivity intents:
   - `local_only`
   - `ssh_tunnel_only`
   - `truly_remote`

## 2. Delivery phases

### Phase 0: Contract + status baseline

Status: Closed (implementation + validation complete)

Scope:
1. Freeze target clean-slate contract in docs.
2. Explicitly mark legacy auth/authz as reference-only input for rewrite.
3. Add endpoint definition consistency (daemon+resources as one endpoint).
4. Define legacy removal/deprecation list.

Exit criteria:
1. Documentation reflects target breaking-contract accurately.
2. Legacy-to-target delta is explicit and implementation-ready.
3. Evidence:
   - clean-slate contract + status docs in place:
     - `src/hosting/hosting_access.md`
     - `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`
     - `src/hosting/hosting_status.md`
   - progress update items: 1-12, 87-91

### Phase 1: Role hierarchy implementation

Status: Closed (implementation + validation complete)

Scope:
1. Implement hierarchy:
   - `admin` > `config_editor` > `worker_user` > `model_user_with_model_control` > `model_user` > `diagnostic_user`
2. Keep `transport` orthogonal and optional.
3. Extend command authz matrix:
   - `config_editor` can spawn engine workers and save modified custom configs under new name
   - `model_user_with_model_control` can override/select model when config default differs or is absent
   - `worker_user` can spawn/interact with generic non-model worker process configs
   - `diagnostic_user` read-only logs/status
4. Remove dependency on legacy external scope semantics as compatibility contract.

Exit criteria:
1. Role-command matrix tests pass.
2. New role model is authoritative and self-contained.
3. Evidence:
   - role hierarchy + command authz implemented across service/daemon/channel
   - role regression coverage in `tests/test_hosting_auth_roles.py`
   - progress update items: 1-3, 10, 24-25, 36-38, 43-45, 46-60

### Phase 2: Endpoint mode and ownership semantics

Status: Closed (implementation + validation complete)

Scope:
1. Persist endpoint default mode (`exclusive`/`shared`) in config.
2. Add admin runtime override:
   - temporary (until daemon shutdown)
   - permanent (persisted default)
3. Implement stale/malicious owner emergency force-override path:
   - explicit reason
   - policy checks
   - high-severity audit trail
4. Ensure former owner gets deterministic ownership-changed behavior.

Exit criteria:
1. Exclusive/shared behavior is deterministic and test-covered.
2. Force-override logic covers unavailable-confirmation scenarios safely.
3. Evidence:
   - endpoint default + runtime override path implemented
   - force/emergency override reason codes and high-severity audit implemented
   - displaced-owner deterministic denial/reclaim flow implemented
   - endpoint-mode runtime commands now honor displaced-owner denial contract
   - progress update items: 14-16, 26-35, 99-102

### Phase 3: Keyring storage and migration

Status: Closed (implementation + validation complete)

Scope:
1. Reserve and enforce `Hosting/` subfolder in default engine config directory.
2. Introduce keyring layout (`access_control`, `keyring`, `audit`, `state`).
3. Implement legacy key migration by rename to `.migrated` and import.
4. Add audit records for migration and key lifecycle events.

Exit criteria:
1. Migration is idempotent and non-destructive.
2. No silent key loss; `.migrated` artifacts remain traceable.
3. Evidence:
   - `Hosting/` layout + migration metadata/audit implemented
   - setup tests validate `.migrated` behavior
   - progress update items: 9, 21-23

### Phase 4: Setup script and scenario-specific bootstrap

Status: Closed (implementation + validation complete)

Scope:
1. Add user-friendly pre-daemon setup script.
   - First cut implemented: `src/hosting/hosting_config.py`
2. Script supports three intents:
   - local-only
   - SSH tunnel-only
   - truly remote
3. Script writes effective config, initializes keyring, registers first admin key, and emits external setup instructions.
4. Script sets default endpoint mode and validates SSH prerequisites.
5. Diagnostics (`--doctor`) validates effective setup health and runtime policy compatibility.

Exit criteria:
1. Fresh install can be fully configured before daemon start.
2. Users get clear external steps for each connectivity scenario.
3. Script behavior matches `hosting_config_script.md`.
4. Evidence:
   - setup/doctor implementation + script spec + docs:
     - `src/hosting/hosting_config.py`
     - `src/hosting/hosting_config_script.md`
   - regression coverage in `tests/test_hosting_config.py`
   - progress update items: 5-6, 17-23, 69-71

### Phase 5: `require_auth=false` hard gate

Status: Closed (implementation + validation complete)

Scope:
1. Restrict unauth mode to safe-only profile:
   - local-only bind
   - exclusive mode
   - single-user admin-only keys
   - no remote relay/tunnel/public ingress
2. Startup must fail fast when `require_auth=false` is configured outside safe profile.
3. Add clear diagnostics and remediation guidance.

Exit criteria:
1. Unsafe unauth combinations are rejected deterministically.
2. Local bootstrap usability remains simple for safe profile.
3. Evidence:
   - runtime safe-profile assertions + command-path enforcement in service/daemon
   - no-auth drift hardening in partial `set-control-config` updates
   - no-auth mode rejects session/challenge bootstrap commands
   - regression coverage:
     - `test_require_auth_false_rejected_for_non_local_profile`
     - `test_authorize_command_rejects_unsafe_no_auth_runtime_config`
     - `test_runtime_policy_assertion_rejects_unsafe_unauth_profile`
     - `test_require_auth_false_rejected_when_profile_drifts_without_require_auth_field`
     - `test_require_auth_false_rejects_session_and_challenge_issue_paths`
   - progress update items: 3, 8, 95-98

### Phase 6: Lifecycle scenarios and terminal-disconnect behavior

Status: Closed (implementation + validation complete)

Scope:
1. Implement/document lifecycle profiles:
   - foreground terminal-bound
   - detached user-process
   - system auto-start service (optional add-on)
2. Define survival rules by effective access config:
   - when terminal/local auth must remain present
   - when daemon survives disconnect
   - when terminal control can be disabled while daemon remains alive
3. Add shutdown ordering checkpoints for clean resource release.

Exit criteria:
1. Bootstrap-to-shutdown paths are predictable and test-covered.
2. Behavior after terminal disconnect is explicit per profile.

### Phase 7: Advanced hardening (risk-gated)

Status: Planned

Scope:
1. Key rotation automation and replay-resistance enhancements.
2. Optional hardware-backed key storage.
3. Advanced anomaly detection/lockout.

Gate condition:
1. Each feature requires documented threat reduction and impact scope analysis.

Exit criteria:
1. No advanced feature enabled by default without explicit risk justification.

### Phase 8: Legacy auth removal and cutover

Status: Closed (implementation + validation complete)

Scope:
1. Remove legacy auth paths not part of clean-slate target.
2. Remove compatibility flags/bridges and obsolete scope-based public docs.
3. Finalize single authoritative auth/authz API surface.

Exit criteria:
1. No runtime dependency on deprecated legacy auth behavior.
2. Docs and tests reflect only the new model.

## 3. Work backlog (ordered)

1. Build role-command authz matrix for the new model.
2. Add endpoint mode persistence and admin override paths.
3. Implement stale-owner emergency force-override with audit reason codes.
4. Implement `Hosting/` keyring structure and legacy `.migrated` flow.
5. Add setup script for local/tunnel/remote intents.
6. Enforce `require_auth=false` safe-only startup policy.
7. Add lifecycle profile controls and disconnect-survival policy hooks.
8. Remove legacy auth paths and deprecated compatibility behavior.
9. Extend tests and docs per phase completion.
10. Expand troubleshooting deliverables (error catalog, playbooks) on top of initial `hosting_config --doctor`.
11. Add testing deliverables (unit/integration/migration/e2e setup coverage).

## 4. Validation matrix

1. Unit tests:
   - role hierarchy and inheritance
   - endpoint mode transitions and ownership checks
   - safe-only unauth gating logic
2. Integration tests:
   - daemon RPC/CLI/channel/HTTP ingress parity
   - local-only vs tunnel-only vs truly remote flows
3. Migration tests:
   - key file rename to `.migrated`
   - keyring import and audit continuity
4. Lifecycle tests:
   - bootstrap and shutdown sequencing
   - terminal disconnect survival behavior by profile
5. Security regressions:
   - stale-owner takeover paths
   - misuse of low-privilege roles
   - unsafe unauth startup attempts
6. Cutover tests:
   - legacy commands/paths fail with explicit deprecation errors (until removed)
   - final build has no active legacy auth code paths

### 4.1 Sandbox validation policy and outside-sandbox reruns

1. In this workspace sandbox, `PermissionError: [WinError 5] Access is denied` during pytest temp cleanup is acceptable when:
   - attempted command is recorded
   - failure reason is recorded
   - outside-sandbox rerun command is documented in `HOSTING_PYTEST_STATUS.md`
2. Required outside-sandbox reruns for ACL/lifecycle-sensitive coverage:
   - `pytest tests/test_hosting_daemon_acl.py -q`
   - `pytest tests/test_hosting_daemon_pidfile.py -q`
   - `pytest tests/test_hosting_service_security.py -q`
   - `pytest tests/test_hosting_http_ingress.py -q`
   - `pytest tests/test_app_config_host_auth.py -q`
3. Minimum evidence format in status docs:
   - exact command
   - pass/fail result summary
   - execution environment note (`sandbox` vs `outside-sandbox`)

## 5. Documentation maintenance

After each phase:
1. Update this plan status and evidence.
2. Update `hosting_access.md` implementation status section.
3. Update `HOSTING_PYTEST_STATUS.md` with new relevant test commands.
4. Update `HOSTING.md` user-facing behavior only when implemented.
