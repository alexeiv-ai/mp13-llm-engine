# Hosting Access Design (Security, Roles, Keys, Lifecycle)

Date: 2026-03-16
Status: Design + implementation-aligned baseline (functional-first, Phase 7 planned)
Scope: `src/hosting` daemon/channel/auth/claim/lifecycle on Windows and Linux

## 0. Design policy (breaking change)

1. This is a clean-slate auth/authz redesign for hosting.
2. Backward compatibility with legacy auth model is not a requirement.
3. No mixed old/new compatibility layer is part of target architecture.
4. Legacy auth/session/key paths are considered deprecated and removable.

Client migration checklist and breaking payload/role changes are documented in:
- `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`

## 1. Goals

1. Harden hosting for local, SSH relay, and truly remote scenarios.
2. Keep design usable first; add advanced hardening as opt-in with clear risk tradeoffs.
3. Define deterministic daemon/resource lifetime for exclusive and shared endpoint modes.
4. Use SSH-key identities as the primary long-lived identity model; use tokens for bounded access time.
5. Keep admin/root operations optional and only as explicit add-ons.

## 2. Preconditions and Assumptions

1. Engine package is installed from a terminal (local console or remote shell).
2. SSH is a hard dependency (OS built-in or OpenSSH) for remote-capable deployments.
3. SSH keys are long-lived identities; short-lived tokens are issued when access duration must be bounded.
4. Endpoint means the whole hosting daemon plus all hosted resources, not a single worker.
5. Exclusive/shared is a daemon-level effective mode:
   - default persistent mode from config
   - admin may override temporarily (runtime) or permanently (config)
6. Root/admin privileges may be unavailable; baseline flow must work in user context.
7. If local terminal access is fully compromised, local files may be rewritten; design targets containment, auditability, and explicit ownership transitions.

### 2.1 Residual risk boundary (explicit)

1. Local host compromise is out of scope for full prevention in baseline architecture.
2. Baseline controls primarily provide:
   - role separation
   - bounded session/token validity
   - deterministic ownership transition behavior
   - auditability for key/session/claim lifecycle actions
3. Baseline controls do not guarantee:
   - protection against local credential/key theft after host compromise
   - protection against local audit/state tampering by a privileged attacker
4. Phase 7 candidates must be evaluated against this boundary and must not claim full local-compromise prevention.

## 3. Roles, Hierarchy, and Capabilities

### 3.1 Role hierarchy

`admin` > `config_editor` > `worker_user` > `model_user_with_model_control` > `model_user` > `diagnostic_user`

`transport` is orthogonal (optional security layer) and does not replace user roles.

### 3.2 Role semantics

1. `admin`
   - full auth/authz administration
   - create/revoke keys by role
   - invalidate keys or disconnect sessions
   - dispose individual workers
   - temporary/permanent override of daemon effective mode
2. `config_editor`
   - can spawn engine worker(s)
   - can modify and save custom configs under new names
   - cannot manage keys/sessions unless separately granted
3. `worker_user`
   - includes `model_user` permissions
   - can spawn and communicate with model engines
   - can spawn/communicate with generic non-model worker processes (generic worker configs)
4. `model_user_with_model_control`
   - includes `model_user`
   - may choose model different from config default or when model is absent in config
5. `model_user`
   - can use existing model-engine sessions within granted scope
   - no config editing or worker-type expansion
6. `diagnostic_user`
   - read-only status and logs
   - no spawn/config/claim/key actions
7. `transport` (orthogonal)
   - tied to remote public `key_id`
   - private key never stored locally by hosting
   - provides transport/channel trust context only

### 3.3 Authorization model contract

The target model is role-hierarchy-first with explicit command authorization.
Implementation may internally use scope primitives, but this is not an external compatibility promise.

## 4. Key, Keyring, and Storage Model

### 4.1 Identity model

1. Primary: SSH public key identities (long-lived).
2. Shared-secret strategy decision:
   - keep as local-only bootstrap fallback for early/simple local scenarios
   - do not use for non-local bootstrap or remote-capable connectivity modes
   - SSH-target helper auto-session issuance via shared-secret is disabled by policy
3. Access duration: session/token TTL controls, not key expiration.
4. SSH private-key passphrases are external user/agent controls; hosting does not store or verify those passphrases directly.

### 4.2 Keyring paradigm

Use a dedicated keyring structure under default config root:

`<default_engine_config_dir>/hosting/`

Suggested contents:
1. `access_control.json` (role/policy config)
2. `keyring/` (active keys and metadata)
3. `audit/` (bounded append-only audit records)
4. `state/` (claims/sessions/runtime state checkpoints)

Migration rule:
1. If legacy key file is detected, move it to `<name>.migrated` before importing into keyring metadata.
2. Record migration event in audit log.
3. Never auto-delete `.migrated` files.

### 4.3 Baseline integrity posture

1. File permissions restricted to daemon user account.
2. Audit every key create/update/revoke/migrate event.
3. Optional tamper-warning checksum chain (best effort, not a local-compromise prevention guarantee).

## 5. Endpoint Access Modes and Ownership

### 5.1 Effective endpoint mode

1. `exclusive`
   - one owner identity/session at a time for endpoint-sensitive actions
   - exclusive owner disconnect triggers daemon/resource shutdown
2. `shared`
   - multiple clients by role permissions
   - daemon remains alive until explicit shutdown/policy stop

Mode source precedence:
1. runtime admin override (highest, temporary)
2. persistent configured default

Current implementation note:
1. persistent default is implemented via `endpoint_mode_default` in control config
2. temporary runtime override is implemented as daemon commands:
   - `set-endpoint-mode-override`
   - `get-endpoint-mode-effective`
3. lifecycle profile baseline is now persisted in control config:
   - `lifecycle_profile` (`foreground_terminal_bound|detached_user_process|service_managed`)
   - `lifecycle_policy` (`on_terminal_disconnect`, `terminal_control_enabled`, `owner_disconnect_shutdown`)
   - note: current daemon behavior treats exclusive-owner disconnect shutdown as unconditional; `owner_disconnect_shutdown` is retained for compatibility/forward control-surface stability.
4. effective lifecycle inspection is available via:
   - `get-lifecycle-policy-effective`

### 5.2 Force-override and stale/malicious owner handling

1. Normal force-override uses explicit confirmation token.
2. If owner is stale/malicious and confirmation is impossible:
   - allow admin emergency override path with stronger checks:
     - owner heartbeat timeout exceeded or verified policy breach
     - explicit reason code required
     - high-severity audit event
     - former owner is notified on next contact and receives deterministic denial until reclaim/reauth

Current implementation note:
1. `force_override_reason` is required when `force_override=true`.
2. localhost non-emergency override still requires confirmation token.
3. localhost emergency override is allowed without confirmation only with reason:
   - `stale_owner_unreachable`
   - `owner_malicious`
   - `security_incident`
4. claim audit events include `severity`, and force/emergency overrides are tagged `high`.
5. displaced owners are flagged by ownership-change notice and receive deterministic denial (`ownership_changed_reclaim_required`) on non-claim commands until reclaim.
6. daemon endpoint-mode runtime control commands now participate in this displaced-owner denial contract:
   - `set-endpoint-mode-override`
   - `get-endpoint-mode-effective`
7. Emergency takeover confirmation bypass is currently localhost-only and reason-code constrained.
8. Emergency eligibility predicates are enforced and denial is explicit:
   - denial code: `force_override_emergency_predicate_not_met`
   - `stale_owner_unreachable` requires conflicting owner to be orphaned (not active)
   - `owner_malicious` and `security_incident` require at least one active conflicting owner
   - denial details include `predicate`, `active_conflicting_owners`, `orphan_conflicting_owners`

## 6. Auth/AuthZ Current Status (Implementation-Aligned)

Implementation baseline status:
1. Phase 1/2/3/4/5/6/8 access-control scope is implemented and validated per `hosting_access_plan.md`.
2. Clean-slate role model is active across service/daemon/channel/client helper paths.
3. Endpoint default mode + runtime override + displaced-owner deterministic denial/reclaim behavior are implemented.
4. Setup + diagnostics flows (`hosting_config`, `--doctor`) and key migration (`.migrated`) are implemented.
5. Safe-only no-auth policy is enforced at startup and on runtime drift (`set-control-config` partial updates).
6. Non-local auth bootstrap uses public-key challenge with SSH session binding context.
7. Shared-secret session issuance (`auth-issue-session`) is local-only and denied for non-local connectivity profiles.
8. Auth lifecycle audit events and admin audit query command (`auth-audit-list`) are implemented.
9. Lifecycle profiles/policies, terminal-control gating, and shutdown sequencing/checkpoints are implemented.
10. Legacy runtime role bridge is removed for cutover scope.
11. Contract metadata (`daemon_version`, `capabilities`) is exposed by `auth-status`, but retrieval depends on a valid command auth path when auth is enabled.

Remaining baseline gaps before Phase 7 feasibility discussion:
1. Emergency takeover predicates are now code-enforced and test-mapped; keep operator docs aligned to denial taxonomy.
2. Troubleshooting artifacts should expand beyond initial `--doctor` into explicit error catalog + playbooks.
3. Generated-key validation across constrained Windows filesystem variants needs host-path confirmation coverage.
4. Scenario runbooks must retain explicit minimum controls and escalation triggers (captured in Section 10).
5. Client-side contract probes must treat missing `daemon_version` as auth/reachability-path failure first (not automatically as daemon-version incompatibility).

## 7. `require_auth=false` Safe-Only Policy

`require_auth=false` must be valid only in explicitly safe configurations:

Allowed only when all are true:
1. Endpoint binding is local-only (`127.0.0.1`/local IPC).
2. Effective mode is `exclusive`.
3. Single-user profile (`admin` only, no secondary role keys).
4. No remote relay/tunnel/public ingress enabled.
5. No persisted shared sessions/tokens.

When any condition is false, daemon must require auth (`require_auth=true`) or fail startup with explicit error.

Current implementation note:
1. `set-control-config` now revalidates no-auth safe-profile constraints even when `require_auth` is omitted from update payload.
2. `set-control-config` force-coerces `endpoint_mode_default` to `exclusive` whenever effective `require_auth=false`.
3. no-auth runtime/profile validation explicitly rejects non-exclusive endpoint defaults with:
   - `require_auth_false_requires_exclusive_endpoint_mode`
4. daemon claim-policy path force-coerces claim requests to `exclusive=true` whenever effective `require_auth=false`.
5. no-auth mode rejects session/challenge issuance bootstrap paths with:
   - `require_auth_disabled_disallows_session_commands`

Rationale:
1. Prevent accidental unauthenticated multi-user or remote exposure.
2. Preserve simple local bootstrap/dev use case.

## 8. Minimal Configuration Flow (Daemon Not Started)

Provide one user-facing setup script (for example `hosting_setup`) that asks intent and writes config before daemon starts.
Detailed script contract: `src/hosting/hosting_config_script.md`.

### 8.1 Script input intents

1. `local_only` (local-only clients)
2. `ssh_tunnel_only` (SSH-mediated remote clients; current implementation uses SSH relay for daemon control)
3. `truly_remote` (non-loopback direct or proxied remote access)

### 8.2 Common script outputs

1. Create/verify `<default_engine_config_dir>/hosting/` structure.
2. Initialize access control config and keyring metadata.
3. Register first admin SSH public key.
4. Choose persistent default endpoint mode (`exclusive` or `shared`).
5. Choose lifecycle profile (`foreground_terminal_bound|detached_user_process|service_managed`).
6. Set `require_auth` based on safe-only policy.
7. Emit platform-specific start instructions.

### 8.3 External steps by intent

1. Local-only
   - bind daemon control to local IPC only
   - optional `require_auth=false` if safe-only gate passes
2. SSH tunnel-only
   - bind daemon control to local IPC only on host
   - use SSH relay for daemon control
   - enforce auth and SSH-bound session usage
3. Truly remote
   - explicit non-loopback bind (or reverse proxy) by admin choice
   - enforce auth, role separation, short token TTL
   - require firewall/network policy setup outside daemon

### 8.4 Taking Ownership Of An Unconfigured Daemon

Preconditions:
1. The daemon is local to the operator and the caller has local filesystem/process access as the same OS user or an equivalently privileged user.
2. Access control is considered "not configured" when no admin key has been provisioned yet (`keys_count=0`).
3. The backend/channel may observe this either:
   - before daemon start via local control-state inspection
   - after daemon start via `auth-status`
4. This state is a bootstrap/recovery state, not a general-purpose bypass state.

Detection guidance for a local client:
1. If daemon RPC is reachable, call `auth-status`.
2. If daemon RPC is unreachable, inspect local control status/state through the local backend/channel status surface.
3. Treat `keys_count=0` as "hosting access is unconfigured".
4. Clients must not treat arbitrary configured/authenticated daemons as eligible for auth downgrade.

Supported ownership path:
1. Provision the first admin credential through bootstrap-safe auth setup.
2. Then issue a control session and continue through normal authenticated control/config flows.
3. This is the intended recovery/bootstrap path for an unconfigured daemon.
4. Current local backend/channel implementation also supports a temporary local recovery bootstrap:
   - if local hosting is unconfigured (`keys_count=0`) and local daemon auto-bootstrap is requested,
   - bootstrap first forces safe local-only defaults in persisted control config:
     - `require_auth=false`
     - `endpoint_mode_default=exclusive`
   - daemon is then started under that temporary local-only no-auth exclusive profile
   - clients must warn the operator to configure `hosting_access` as soon as possible after startup
5. Current local backend/channel implementation also provides a local recovery helper:
   - `reset_hosting_access`
   - local-helper only; not available over daemon RPC
   - stops local daemon, then clears only auth state from local control state
   - preserves unrelated control config and runtime metadata outside `control_config.auth`

Unsupported client takeover path:
1. There is no supported daemon command that lets a normal client take full control without first configuring access control.
2. There is no supported daemon command that lets a normal client flip an already configured/authenticated daemon to `require_auth=false`.
3. `set-control-config` remains an authenticated admin control operation once auth is active.

Local operator caveat:
1. If the caller can execute arbitrary local Python or otherwise read/write local state files as the daemon user, the caller is outside the daemon's client trust boundary and effectively has local-operator powers.
2. In that case the caller may be able to inspect or manipulate daemon state through OS/file/process access, but that is a consequence of local host access, not a supported auth/authz bypass.
3. This remains within the documented residual-risk boundary for local host compromise.

Local shutdown/restart notes relevant to ownership recovery:
1. The daemon stop path is `__shutdown__` guarded by `shutdown_token`.
2. `shutdown_token` is persisted in the daemon PID file (`hosting/state/daemon.pid`) alongside `pid`, `port`, and local IPC transport metadata.
3. `terminal_control_enabled` is persisted in control state under `control_config.lifecycle_policy.terminal_control_enabled`.
4. Even with the correct `shutdown_token`, daemon shutdown is denied when `terminal_control_enabled=false`.
5. The effective terminal-control state should be read via `get-lifecycle-policy-effective`; raw persisted config can also be inspected via `get-control-config`.
6. `reset_hosting_access` may use local process termination fallback when graceful daemon shutdown is unavailable.
7. `reset_hosting_access` is intentionally a local-helper shortcut to editing local auth state; it does not traverse daemon auth/RPC command paths.

## 9. Daemon Lifetime Cycle Scenarios

### 9.1 Foreground terminal-bound cycle

1. bootstrap via terminal
2. daemon runs attached
3. terminal disconnect stops daemon (unless switched to detached/service mode)

Use when:
1. temporary admin operations
2. local-only single-user sessions

### 9.2 Detached user-process cycle

1. bootstrap from terminal
2. daemon detaches and survives terminal disconnect
3. remains alive per endpoint mode/policy until explicit shutdown or fatal policy condition

Use when:
1. shared multi-client operation
2. SSH relay workflows requiring continuity

### 9.3 Service-managed cycle (optional admin add-on)

1. bootstrap config once
2. daemon auto-start by system policy
3. terminal may be unnecessary or disabled for routine operation

Use when:
1. managed hosts requiring persistent availability
2. controlled operational environments

### 9.4 Access-config-dependent survival rules

1. Exclusive + owner disconnect: daemon auto-shutdown is enforced.
   - current implementation: exclusive owner disconnect triggers daemon shutdown regardless of `owner_disconnect_shutdown` value.
2. Shared + valid sessions: daemon stays alive independent of one client terminal.
3. Service-managed configuration may keep daemon alive across user logouts/reboots.
4. Policy may require local auth presence in stricter profiles; in service profiles this can be relaxed.

Current implementation note:
1. lifecycle profile/policy persistence and normalized effective-policy inspection are implemented.
2. owner-disconnect enforcement hook is implemented:
   - exclusive owner disconnect triggers daemon shutdown (independent of `owner_disconnect_shutdown` value).
3. foreground terminal-disconnect behavior now honors lifecycle policy:
   - `on_terminal_disconnect=keep_daemon_running` applies SIGHUP-ignore where supported.
4. daemon shutdown-order checkpoints are implemented:
   - daemon stop path attempts orderly managed-engine shutdown and registration cleanup.
5. daemon shutdown sequencing now drains in-flight async host operations before managed-worker checkpoint shutdown.
6. terminal control gating is enforced:
   - when `terminal_control_enabled=false`, terminal control paths are denied (`__shutdown__`, runtime endpoint-mode override).
7. additional profile-hardening remains staged for follow-up.

## 10. Scenario Comparison (Functional First, Operator-Oriented)

### 10.1 Scenario A: Local-only single-user bootstrap (`local_only`, optional no-auth)

Intended usage:
1. single developer/operator on one host
2. lowest-friction bootstrap and local development

Mitigated attack vectors:
1. accidental remote exposure via safe-only `require_auth=false` gate
2. unsafe no-auth profile drift through partial config updates
3. session/challenge abuse when auth is disabled

Unmitigated or weakly mitigated vectors:
1. local host compromise (key/state/audit file tampering)
2. misuse by any process with equivalent local user privileges

Minimum controls to remain in this scenario:
1. daemon control remains local IPC only
2. endpoint mode remains `exclusive`
3. single-user admin-only key profile remains enforced
4. no tunnel/relay/public ingress is enabled
5. if shared-secret bootstrap is used, keep it local-only and avoid persistent plaintext secret storage
6. if daemon was auto-started from an unconfigured local state using temporary `require_auth=false`, operator must complete `hosting_access` configuration as soon as possible and should not treat the temporary no-auth state as steady-state policy

Escalate to next scenario when:
1. a second user/process identity needs access
2. remote access is needed

### 10.2 Scenario B: Local-only authenticated multi-role operation

Intended usage:
1. same host, multiple trusted users/process identities
2. local least-privilege separation without network exposure

Mitigated attack vectors:
1. role misuse for config/model/generic-worker actions
2. unauthorized key/session revocation by non-admin roles
3. unauthorized model override and generic-worker proxy/rpc usage

Unmitigated or weakly mitigated vectors:
1. local credential/key theft after host compromise
2. local denial-of-service by privileged local attacker

Minimum controls to remain in this scenario:
1. `require_auth=true`
2. role assignments limited to least privilege (`diagnostic_user`/`model_user`/`worker_user`/`config_editor`/`admin`)
3. audit review using `auth-audit-list`
4. keyring filesystem permissions remain restricted to daemon user account
5. shared-secret usage (if any) remains bootstrap-only and local-only; prefer public-key challenge for durable operator flows

Escalate to next scenario when:
1. off-host operators need access

### 10.3 Scenario C: SSH relay remote operation (`ssh_tunnel_only`)

Intended usage:
1. remote operator access while keeping daemon control local-IPC-bound on host
2. continuity workflows via detached lifecycle profile

Mitigated attack vectors:
1. direct non-loopback daemon exposure
2. non-local auth bootstrap without SSH binding
3. remote shared-secret bootstrap misuse (non-local `auth-issue-session` is denied)
4. non-local command use without persisted + presented SSH binding context

Unmitigated or weakly mitigated vectors:
1. stolen SSH private keys
2. host compromise on SSH relay endpoint
3. SSH relay endpoint operational misconfiguration

Minimum controls to remain in this scenario:
1. host daemon control stays local-IPC-only
2. SSH relay endpoint hardening is applied outside daemon (host/network policy)
3. short session/token TTL and strict role separation
4. regular claim/auth audit review for suspicious takeover/auth patterns

Escalate to next scenario when:
1. direct non-loopback clients or broader remote ingress is required

### 10.4 Scenario D: Truly remote multi-client operation (`truly_remote`)

Intended usage:
1. persistent remote-serving environment
2. multi-client/multi-role access with explicit external network controls

Mitigated attack vectors:
1. role-based privilege escalation attempts within command surfaces
2. stale-owner lock-in via emergency takeover + deterministic displaced-owner handling
3. terminal-control abuse when disabled by lifecycle policy

Unmitigated or weakly mitigated vectors:
1. internet-facing brute-force and credential replay pressure
2. large-scale auth abuse requiring adaptive controls
3. sophisticated key compromise scenarios needing rotation/stronger storage

Minimum controls to remain in this scenario:
1. `require_auth=true` always
2. strict role separation + admin-only key/session invalidation operations
3. external firewall/reverse-proxy policy and ingress minimization
4. operational lifecycle policy review (`service_managed` vs detached profile)
5. alerting/playbook ownership for auth/claim audit events

Escalate beyond baseline (Phase 7 candidates) when:
1. threat model includes key replay/rapid credential churn
2. compliance or assurance requires hardware-backed keys
3. attack volume requires adaptive lockout/anomaly controls

## 11. Advanced Hardening and Risk Assessment (Later Stages)

These are intentionally secondary to functional/usability baseline:
1. key rotation automation
2. replay-protection deepening beyond current challenge/session controls
3. hardware-backed key storage
4. advanced anomaly detection and adaptive lockouts

Phase 7 planning detail:
1. `src/hosting/hosting_phase7_hardening.md`

Each must be documented with:
1. threat introduced/expanded
2. compensating controls
3. scope of impact (for example local-only admin-only deployment may not need it)

## 12. Phase 7 Readiness Prerequisites (Before Feasibility Decision)

1. Docs are status-aligned across:
   - `hosting_access.md`
   - `hosting_access_plan.md`
   - `HOSTING_PYTEST_STATUS.md`
2. Emergency takeover contract includes explicit eligibility predicates and denial semantics.
3. Validation evidence includes required outside-sandbox reruns for ACL- and lifecycle-sensitive suites when sandbox teardown ACL issues occur.
4. Generated-key path behavior is validated on host paths where OpenSSH write semantics are supported.
5. Scenario-specific minimum controls and escalation triggers remain current with implementation.
