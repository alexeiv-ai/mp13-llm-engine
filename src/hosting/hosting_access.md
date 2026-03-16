# Hosting Access Design (Security, Roles, Keys, Lifecycle)

Date: 2026-03-14
Status: Proposed design update (functional-first, security hardening staged)
Scope: `src/hosting` daemon/channel/auth/claim/lifecycle on Windows and Linux

## 0. Design policy (breaking change)

1. This is a clean-slate auth/authz redesign for hosting.
2. Backward compatibility with legacy auth model is not a requirement.
3. No mixed old/new compatibility layer is part of target architecture.
4. Legacy auth/session/key paths are considered deprecated and removable.

Client migration checklist and breaking payload/role changes are documented in:
- `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`

## 1. Goals

1. Harden hosting for local, SSH tunnel, and truly remote scenarios.
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
2. Secondary: shared secret keys (optional fallback path, not compatibility requirement).
3. Access duration: session/token TTL controls, not key expiration.

### 4.2 Keyring paradigm

Use a dedicated keyring structure under default config root:

`<default_engine_config_dir>/Hosting/`

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
   - owner disconnect normally triggers daemon/resource shutdown
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

## 6. Auth/AuthZ Current Status (Legacy Snapshot for Rewrite Planning)

Based on current `src/hosting` docs and status files, for rewrite planning only:

Implemented:
1. `require_auth` config and session/key primitives.
2. Scoped authorization (`control/config/traffic`) with session TTL.
3. Public-key challenge auth support.
4. SSH binding fields and binding checks for bound sessions/challenges.
5. Daemon-derived claim actor identity (`key:<key_id>`).
6. Claim ACL with owner TTL, takeover transitions, and structured denial codes.
7. Non-localhost shared claim deny behavior in daemon path.
8. Localhost force-override confirmation token path.
9. Audit-style claim events and metrics telemetry.

Missing for clean-slate target design:
1. Full hierarchical role model listed above (beyond current scope roles).
2. `diagnostic_user` explicit command surface.
3. Endpoint-level mode defaults + temporary override persistence model.
   - default + runtime override commands implemented
   - remaining work: override audit/ownership-notification integration and persistence policy refinements
4. Full setup-wizard UX and reconfigure diff/apply workflow polish for local/tunnel/remote intent.
5. Extended troubleshooting artifacts (error catalog/playbooks) beyond initial `--doctor`.
6. Comprehensive generated-key path validation across constrained Windows filesystem variants.

Implemented note:
1. `connect-from-config` now classifies worker profile (`model` vs `generic`) and enforces generic worker usage only for `worker_user` (or higher).
2. Generic worker profile runtime path is now supported in `connect-from-config`:
   - `worker_kind/worker_type = generic`
   - spawn command from `worker_command` or `spawn.command`
   - model selection step is skipped for generic profiles
3. Traffic/rpc runtime enforcement now applies to generic engine registrations:
   - `worker_profile_class=generic` engines deny model-role proxy/rpc usage with `insufficient_role`
4. `transport` role is now hard-gated to `public_key` onboarding only and cannot issue auth sessions/challenges.
5. In non-local connectivity profiles, auth bootstrap requires SSH binding:
   - shared-secret `auth_issue_session` requires `ssh_binding`
   - public-key `auth_begin_challenge` requires `ssh_binding`
6. In non-local connectivity profiles, command authorization also requires SSH binding context:
   - presented `_ssh_session_binding` is mandatory
   - session must contain persisted binding metadata
   - unbound legacy sessions are denied in non-local mode
7. Auth lifecycle operations now include explicit audit trail records:
   - key upsert/revoke and session revoke events are written to `auth_audit_events` in control state.
8. Admin query surface now includes `auth-audit-list` for paged/filterable access to auth audit events.
9. Legacy role bridge removal is complete for this cutover scope:
   - app-level host auth helper (`mp13config`) now accepts only clean-slate role names.

## 7. `require_auth=false` Safe-Only Policy

`require_auth=false` must be valid only in explicitly safe configurations:

Allowed only when all are true:
1. Endpoint binding is local-only (`127.0.0.1`/local IPC).
2. Effective mode is `exclusive`.
3. Single-user profile (`admin` only, no secondary role keys).
4. No remote relay/tunnel/public ingress enabled.
5. No persisted shared sessions/tokens.

When any condition is false, daemon must require auth (`require_auth=true`) or fail startup with explicit error.

Rationale:
1. Prevent accidental unauthenticated multi-user or remote exposure.
2. Preserve simple local bootstrap/dev use case.

## 8. Minimal Configuration Flow (Daemon Not Started)

Provide one user-facing setup script (for example `hosting_setup`) that asks intent and writes config before daemon starts.
Detailed script contract: `src/hosting/hosting_config_script.md`.

### 8.1 Script input intents

1. Local-only clients
2. SSH tunnel-only remote clients
3. Truly remote access

### 8.2 Common script outputs

1. Create/verify `<default_engine_config_dir>/Hosting/` structure.
2. Initialize access control config and keyring metadata.
3. Register first admin SSH public key.
4. Choose persistent default endpoint mode (`exclusive` or `shared`).
5. Choose lifecycle profile (`foreground_terminal_bound|detached_user_process|service_managed`).
6. Set `require_auth` based on safe-only policy.
7. Emit platform-specific start instructions.

### 8.3 External steps by intent

1. Local-only
   - bind daemon to loopback/local IPC only
   - optional `require_auth=false` if safe-only gate passes
2. SSH tunnel-only
   - bind daemon loopback only on host
   - establish SSH `-L`/relay path externally
   - enforce auth and SSH-bound session usage
3. Truly remote
   - explicit non-loopback bind (or reverse proxy) by admin choice
   - enforce auth, role separation, short token TTL
   - require firewall/network policy setup outside daemon

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
2. SSH tunnel workflows requiring continuity

### 9.3 Service-managed cycle (optional admin add-on)

1. bootstrap config once
2. daemon auto-start by system policy
3. terminal may be unnecessary or disabled for routine operation

Use when:
1. managed hosts requiring persistent availability
2. controlled operational environments

### 9.4 Access-config-dependent survival rules

1. Exclusive + owner-disconnect policy: daemon may auto-shutdown.
2. Shared + valid sessions: daemon stays alive independent of one client terminal.
3. Service-managed configuration may keep daemon alive across user logouts/reboots.
4. Policy may require local auth presence in stricter profiles; in service profiles this can be relaxed.

Current implementation note:
1. lifecycle profile/policy persistence and normalized effective-policy inspection are implemented.
2. owner-disconnect enforcement hook is implemented:
   - when `owner_disconnect_shutdown=true`, exclusive owner disconnect can trigger daemon shutdown.
3. foreground terminal-disconnect behavior now honors lifecycle policy:
   - `on_terminal_disconnect=keep_daemon_running` applies SIGHUP-ignore where supported.
4. daemon shutdown-order checkpoints are implemented:
   - daemon stop path attempts orderly managed-engine shutdown and registration cleanup.
5. daemon shutdown sequencing now drains in-flight async host operations before managed-worker checkpoint shutdown.
6. terminal control gating is enforced:
   - when `terminal_control_enabled=false`, terminal control paths are denied (`__shutdown__`, runtime endpoint-mode override).
7. additional profile-hardening remains staged for follow-up.

## 10. Scenario Comparison (Functional First)

### 10.1 Local-only (limited impact by design)

1. simplest ops model
2. smallest network attack surface
3. can allow constrained unauth mode under strict safe-only gate

### 10.2 SSH tunnel-only

1. remote usability with loopback-bound daemon
2. transport exposure minimized
3. auth still required for multi-role operation

### 10.3 Truly remote

1. highest operational flexibility
2. highest attack surface
3. requires strict role separation, enforced auth, and external network controls

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
