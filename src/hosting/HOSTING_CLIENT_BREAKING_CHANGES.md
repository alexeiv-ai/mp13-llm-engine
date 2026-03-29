# Hosting Client Breaking Changes (Clean-Slate Auth/AuthZ)

Date: 2026-03-14
Applies to: clean-slate hosting auth/authz redesign in `src/hosting`

This change set intentionally breaks legacy client auth assumptions.

## 1. Role names changed

Old role names are no longer accepted:
1. `management`
2. `config`
3. `traffic`

New role names:
1. `admin`
2. `config_editor`
3. `worker_user`
4. `model_user_with_model_control`
5. `model_user`
6. `diagnostic_user`
7. `transport` (orthogonal transport identity)

Client impact:
1. Update all `auth-upsert-key` payloads to new `role` values.
2. `mp13config --host-auth-upsert-key --host-auth-role` now accepts only clean-slate roles:
   - `admin|config_editor|worker_user|model_user_with_model_control|model_user|diagnostic_user`
   - legacy role strings are rejected.

## 2. Role-based command authorization is now enforced

Authorization is no longer “scope-only”. A valid session token can still be denied if role does not allow command.

Client impact:
1. Handle `insufficient_role` denial code.
2. Request/use key+session for the correct role for the command path.
3. `connect-from-config` now distinguishes model vs generic worker profiles:
   - generic worker profiles require `worker_user` (or higher)
   - `model_user` and `model_user_with_model_control` are denied generic worker profile usage
4. Generic worker profile execution contract:
   - set `worker_kind: "generic"` (or `worker_type: "generic"`)
   - provide command via `worker_command` or `spawn.command`
   - generic profile does not require model selection (`model_path` may be omitted)
5. Runtime traffic/rpc enforcement for generic engines:
   - engines registered with `worker_profile_class=generic` reject model-role proxy/rpc usage
   - `model_user` and `model_user_with_model_control` receive `insufficient_role` on those engine paths
6. `transport` role key constraints:
   - `transport` keys must use `auth_method=public_key`
   - `transport` keys cannot issue sessions/challenges for command authorization
7. Shared-secret bootstrap is now local-only:
   - when `access_profile.connectivity_mode != local_only`:
     - `auth-issue-session` (shared-secret path) is denied
     - denial code:
       - `shared_secret_bootstrap_not_supported_for_remote_connectivity`
   - non-local bootstrap must use public-key challenge flow.
   - SSH-target helper channels do not auto-issue shared-secret sessions; use pre-issued session token or explicit challenge flow.
8. Remote connectivity public-key bootstrap and command path require SSH binding:
   - for `auth-begin-challenge` in non-local modes, `ssh_binding` is required
   - missing binding is denied with:
     - `ssh_binding_required_for_remote_connectivity`
9. Remote connectivity command path now enforces SSH binding presence:
   - for session-backed commands under non-local connectivity:
     - payload must include `_ssh_session_binding`
     - session must have been issued with matching binding
   - legacy unbound sessions are denied after profile flip to non-local mode with:
     - `ssh_binding_required_for_remote_connectivity`
10. Admin-only invalidation controls are now explicitly role-enforced:
   - `auth-revoke-key` and `auth-revoke-session` require `admin` role
   - lower control roles (for example `config_editor`) are denied with `insufficient_role`
11. New admin audit query command:
   - `auth-audit-list` exposes paged/filterable auth lifecycle audit events
   - requires `admin` role; non-admin control roles are denied with `insufficient_role`
12. No-auth (`require_auth=false`) bootstrap command-path tightening:
   - session/challenge issuance commands are now denied when auth is disabled:
     - `auth-issue-session`
     - `auth-begin-challenge`
     - `auth-complete-challenge`
   - denial code:
     - `require_auth_disabled_disallows_session_commands`
13. No-auth profile drift hardening in `set-control-config`:
   - even when `require_auth` is not present in payload, updates are rejected if resulting profile violates no-auth safe connectivity rule.
   - denial code:
     - `require_auth_false_only_supported_for_local_only_connectivity`
14. Displaced-owner denial now consistently applies to endpoint-mode runtime control commands until reclaim:
   - denied for displaced owner:
     - `set-endpoint-mode-override`
     - `get-endpoint-mode-effective`
   - denial code:
     - `ownership_changed_reclaim_required`

## 3. `require_auth=false` now hard-gated

Unauth mode is allowed only for safe profile:
1. `access_profile.connectivity_mode == local_only`
2. `endpoint_mode_default == exclusive`
3. single admin-only key profile
4. no active sessions/challenges

Client impact:
1. `set-control-config` may fail when disabling auth in unsafe profile.
2. Handle new denial codes:
   - `require_auth_false_only_supported_for_local_only_connectivity`
   - `require_auth_false_requires_exclusive_endpoint_mode`
   - `require_auth_false_requires_no_active_sessions_or_challenges`
   - `require_auth_false_requires_single_admin_key_profile`
   - `require_auth_false_requires_admin_only_keys`
3. whenever effective `require_auth=false`, `set-control-config` force-coerces `endpoint_mode_default` to `exclusive`.
4. in daemon mode, claim commands are force-coerced to `exclusive=true` whenever effective `require_auth=false` (caller-provided `exclusive=false` is ignored).
5. local daemon bootstrap behavior changed for unconfigured hosting:
   - if local hosting is unconfigured (`keys_count == 0`) and the backend/channel auto-bootstrap path starts the daemon,
     bootstrap now forces temporary local-only safe defaults before daemon start:
     - `require_auth=false`
     - `endpoint_mode_default=exclusive`
   - clients must treat this as a temporary recovery/bootstrap state, not as steady-state desired configuration.
6. when that temporary local bootstrap state is detected, clients should warn the operator to configure `hosting_access` as soon as possible.
7. do not assume persisted `require_auth=true` survives local unconfigured auto-bootstrap unchanged; re-read daemon/control status after bootstrap.
8. new local-only recovery helper:
   - `reset-hosting-access`
   - stops local daemon and clears only auth state from local control state
   - this helper is intentionally not accessible over daemon RPC/auth command paths
   - treat it as a local operator recovery shortcut, not as a normal remote/client control API

## 4. `set-control-config` payload/response extended

New field:
1. `access_profile` (currently includes `connectivity_mode`)
2. `endpoint_mode_default` (`exclusive` or `shared`)
3. `lifecycle_profile`:
   - `foreground_terminal_bound`
   - `detached_user_process`
   - `service_managed`
4. `lifecycle_policy`:
   - `on_terminal_disconnect` (`stop_daemon` or `keep_daemon_running`)
   - `terminal_control_enabled` (`bool`)
   - `owner_disconnect_shutdown` (`bool`)
   - compatibility note: current daemon owner-disconnect shutdown for exclusive endpoint ownership is enforced regardless of this flag value.

Client impact:
1. If you own control config management, include/understand `access_profile`.
2. If you parse `get-control-config`, include/understand `endpoint_mode_default`.
3. If you parse `get-control-config`, include/understand `lifecycle_profile` and `lifecycle_policy`.
4. If you parse `get-control-config`, allow and preserve unknown top-level fields.

## 4.5 Local daemon status contract expanded

`get_daemon_status()` now carries more than PID/process liveness when used by local backend/UI clients.

Returned fields now include:
1. `require_auth`
2. `keys_count`
3. `endpoint_mode_default`
4. `control_config`
   - local persisted control snapshot fallback, used even when daemon RPC is unreachable
5. `warnings`
   - warning list for operator-visible notices
   - includes temporary local unconfigured bootstrap warning when daemon was started in forced local-only no-auth exclusive mode
6. `status_event`
   - change event emitted by polling snapshots when daemon status materially changes
   - includes PID-file creation/removal/update and reachability transitions

Client impact:
1. Treat `get_daemon_status()` as the authoritative local status surface for daemon lifecycle plus local hosting bootstrap state.
2. If daemon RPC is unreachable, continue polling `get_daemon_status()`; do not freeze prior daemon state in UI.
3. When `status_event` is non-null, refresh daemon/hosting UI state immediately.
4. If `status_event.reason` indicates PID-file removal/change, treat any cached "daemon running" state as stale.
5. If `warnings` contains the temporary no-auth bootstrap warning, surface that warning prominently and direct the user to configure `hosting_access`.
6. Prefer top-level `require_auth` / `keys_count` / `endpoint_mode_default` from `get_daemon_status()` for local daemon cards/status panes, because those fields can still be populated from local control-state fallback when RPC auth/reachability paths fail.
7. If your product exposes `reset-hosting-access`, label it as local recovery only and do not route it through generic daemon command execution code.

## 4.1 Endpoint mode behavior update

Claim commands now support omitted `exclusive` by using daemon effective endpoint mode:
1. persistent default: `endpoint_mode_default` from control config
2. optional daemon runtime override: `set-endpoint-mode-override`

Runtime override behavior:
1. daemon-only feature
2. temporary until daemon shutdown
3. inspect via `get-endpoint-mode-effective`

## 4.3 Lifecycle policy inspection command

New command:
1. `get-lifecycle-policy-effective`
   - returns normalized profile/policy and effective disconnect-survival booleans

Client impact:
1. Use this command to drive lifecycle-sensitive UX/runbook decisions.

## 4.4 Lifecycle enforcement behavior update

Lifecycle policy is now used by daemon runtime enforcement:
1. exclusive endpoint ownership:
   - when endpoint is exclusively owned, owner disconnect triggers daemon shutdown (independent of `owner_disconnect_shutdown` value).
2. foreground terminal profile with terminal-disconnect action:
   - `on_terminal_disconnect=keep_daemon_running` enables SIGHUP-ignore behavior where supported.
3. daemon stop ordering:
   - daemon shutdown path attempts orderly managed-engine shutdown before final stop.
4. daemon stop sequencing now drains in-flight host operations before worker checkpoint shutdown.
5. terminal control gating:
   - when `terminal_control_enabled=false`, terminal control paths are denied:
     - `__shutdown__`
     - `set-endpoint-mode-override`

Client/operator impact:
1. For exclusive flows that require daemon continuity after owner disconnect, do not rely on lifecycle policy toggles; use shared endpoint mode or reconnect/transfer ownership explicitly.

## 4.2 Force-override payload contract update

Claim override payload now requires explicit reason metadata:
1. `force_override_reason` is required when `force_override=true`
2. localhost non-emergency override still requires:
   - `force_override_confirmation="CONFIRM_LOCALHOST_FORCE_OVERRIDE"`
3. emergency override path:
   - set `force_override_emergency=true`
   - allowed emergency reasons:
     - `stale_owner_unreachable`
     - `owner_malicious`
     - `security_incident`
4. emergency takeover eligibility predicates are now explicitly enforced:
   - denial code:
     - `force_override_emergency_predicate_not_met`
   - `stale_owner_unreachable`:
     - denied unless conflicting owner is orphaned (not active)
   - `owner_malicious` and `security_incident`:
     - denied unless there is at least one active conflicting owner

Client impact:
1. Include `force_override_reason` in any override/takeover claim request.
2. Handle new denial codes:
   - `force_override_reason_required`
   - `force_override_emergency_reason_invalid`
   - `force_override_emergency_predicate_not_met`
   - `ownership_changed_reclaim_required`
3. After being displaced by override, non-claim commands are denied until client reclaims ownership (or drops session).

## 5. New setup/reconfig path

New utility:
1. `python -m hosting.hosting_config ...`
2. `python -m hosting.engine_host_cli --hosting-config ...`

Client/operator impact:
1. Bootstrap/reconfigure should move to `hosting_config` flow.
2. Consume generated client mapping file:
   - `<default_engine_config_dir>/hosting/bootstrap/client_key_map.json`
3. Use bootstrap snapshot for automation/debug:
   - `<default_engine_config_dir>/hosting/bootstrap/bootstrap_state.json`
4. Setup/reconfigure now accepts lifecycle profile selection:
   - `--lifecycle-profile foreground_terminal_bound|detached_user_process|service_managed`

## 6. Minimal migration checklist

1. Replace old role strings in all key provisioning calls.
2. Update error handling to process `insufficient_role` and safe-profile denial codes.
3. Update control-config logic to include `access_profile`.
4. Update operational runbooks to use `hosting_config` for first-time setup and reconfig.
5. Validate each client path against intended role (admin/config_editor/worker/model/diagnostic).
6. Contract-probe client behavior:
   - do not classify daemon as "too old" only because `daemon_version` is absent/empty in a failed `auth-status` path.
   - when `require_auth=true` and keys exist, call `auth-status` with a valid session token.
   - treat `daemon_version=None` as "contract metadata unavailable on this auth/reachability path" until auth/reachability is validated.
7. For local daemon bootstrap/start UX:
   - if hosting is unconfigured (`keys_count == 0`), expect temporary forced bootstrap to `require_auth=false` + `endpoint_mode_default=exclusive`
   - show an explicit warning that `hosting_access` must be configured as soon as possible
   - after bootstrap/start, refresh state from `get_daemon_status()` instead of trusting only the initial start result or old PID/process state
8. For local daemon diagnostics UX:
   - keep polling `get_daemon_status()` even when RPC is unreachable
   - react to `status_event` so PID-file disappearance or replacement clears stale "daemon running" displays promptly
9. For local operator recovery UX:
   - `reset-hosting-access` is available only as a local helper
   - do not expect it to succeed through daemon RPC, relay mode, or normal authenticated command transport
   - explain that it clears only auth state, not the rest of hosting control config

## 7. Phase 7 planning note (no new breaking changes yet)

1. Advanced hardening remains planned/risk-gated and is not enabled by default.
2. No additional client-breaking contract is introduced by the current Phase 7 draft.
3. Design reference: `src/hosting/hosting_phase7_hardening.md`.
4. Pre-Phase-7 baseline freeze note:
   - Phases 0-5 closure is documentation/status formalization only and does not introduce additional client-breaking changes.
