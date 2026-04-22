# Hosting Access Design (Security, Roles, Keys, Lifecycle)

Date: 2026-03-16
Status: Implementation-aligned architecture document
Scope: `src/hosting` daemon/channel/auth/claim/lifecycle on Windows and Linux

## 0. Design policy

1. This document is the implementation-aligned architecture contract for hosting access.
2. The supported runtime auth model is role-hierarchy-first with explicit command authorization.
3. Local, SSH relay, and remote-capable scenarios must be configured through explicit access and transport policy.
4. Security-sensitive defaults should favor pinned host keys, public-key identities, and least-privilege role assignment.

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
4. Non-local SSH-capable operation requires explicit SSH host-key pinning on the hosting consumer side; opportunistic first-connect host trust is not a supported baseline mode.
5. Endpoint means the whole hosting daemon plus all hosted resources, not a single worker.
6. Exclusive/shared is a daemon-level effective mode:
   - default persistent mode from config
   - admin may override temporarily (runtime) or permanently (config)
7. Root/admin privileges may be unavailable; baseline flow must work in user context.
8. If local terminal access is fully compromised, local files may be rewritten; design targets containment, auditability, and explicit ownership transitions.

### 2.1 Terminology: hosting consumer vs UI client

This document uses "client" in older sections because that was the original implementation term. The more precise term is "hosting consumer".

1. A hosting consumer is normally a long-running backend process that talks to the hosting daemon.
2. A UI may configure or observe hosting through that backend, but the UI is not usually the direct hosting protocol peer.
3. Consumer-side setup can therefore include backend-owned files, SSH profiles, private-key import, and reconnect behavior that would be inappropriate to treat as transient UI state.
4. When this document says "client side" in key/transport sections, read it as "hosting consumer side" unless the text explicitly discusses UI.

### 2.2 Residual risk boundary (explicit)

1. Local user-account compromise is out of scope for full prevention in baseline architecture.
2. Baseline controls primarily provide:
   - role separation
   - bounded session/token validity
   - deterministic ownership transition behavior
   - auditability for key/session/claim lifecycle actions
3. Baseline controls do not guarantee:
   - protection against local credential/key theft after local user-account compromise
   - protection against local audit/state tampering by a privileged attacker
4. Advanced hardening candidates must be evaluated against this boundary and must not claim full local-account-compromise prevention.

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
   - if an admin shared-secret key already exists, it can issue/control a session in `local_only`
   - the same shared-secret key cannot issue a session in `ssh_tunnel_only` or `truly_remote`; use public-key challenge instead
3. Access duration: session/token TTL controls, not key expiration.
4. SSH private-key passphrases are external user/agent controls; hosting does not store or verify those passphrases directly.

Bootstrap, as used in this document, means the first step that gets hosting from "no trusted identity exists yet" to "at least one trusted admin identity exists and can start normal authenticated session flows."

There are two different bootstrap situations:
1. First-key bootstrap
   - the daemon has `require_auth=true` but no keys yet
   - the service allows only minimal first-key provisioning commands such as `auth-upsert-key`
   - this is local-only; remote-capable connectivity must not expose zero-key bootstrap
   - after the first admin key is added, normal authenticated flows take over
2. Local recovery bootstrap
   - a local-only helper may temporarily force a safe no-auth profile with `require_auth=false`
   - this is for local recovery or initial local startup convenience
   - it is not the normal remote-capable bootstrap model and should not be treated as steady-state policy

These are different on purpose:
1. `require_auth=true` with zero keys means auth is conceptually on, but the system still needs one initial trusted key.
2. `require_auth=false` means the daemon is temporarily allowing unauthenticated local access under a tightly restricted safe profile.
3. Remote-capable bootstrap should use the first model, not the second.

### 4.1.1 Private-key handling

The short version:
1. Hosting needs the public key to recognize an identity.
2. The user who installed the hosting component is still responsible for the private key.
3. Clients should assume they must already have the private key through some normal local SSH/key-management workflow.
4. Hosting should make it obvious whether a key was imported or generated, and where the private key is expected to be.

What this means in practice:
1. If you import an existing key, hosting stores the public key only.
2. If hosting generates a new keypair for convenience, the public key is registered with hosting and the private key must be accounted for clearly.
3. Hosting-generated private keys can be handled in either of two explicit ways:
   - export to a private-key file immediately
   - store in the setup machine's default client realm and print a later export/import handoff command
4. Clients must not expect hosting to hand them private key material later through normal RPC/API calls.

Imported-key example:
1. A user already has `C:\Users\me\.ssh\id_ed25519` and `C:\Users\me\.ssh\id_ed25519.pub`.
2. The setup flow imports `id_ed25519.pub`.
3. Hosting records that the admin key is `imported`.
4. The private key remains where the user already keeps it.
5. In this case hosting may know the public key came from a file, but it does not manage the private key.

Generated-key example:
1. A user asks the setup flow to generate a new admin keypair.
2. Hosting registers the generated public key.
3. The private key is stored in the setup machine's client realm secret store.
4. The key can be copied to the real hosting consumer by printing structured handoff text with `--client-show-key-handoff` or `Manage RBAC keys` -> `Show local admin handoff text`, then importing it with the client-realm API.
5. If hosting-generated private key material is still embedded in local hosting metadata, that is a legacy/repair state and should be treated as follow-up work.

What the user who installed the hosting component should do:
1. If the setup output says `imported`, use the private key you already had before running setup.
2. If the setup output says `generated`, migrate the client-realm secret into the hosting consumer's own client-realm storage during consumer configuration.
3. If the consumer is configured separately, print the handoff text and paste it into the consumer's client-realm import path. Treat that text as private-key material.
4. If the setup output says `generated` but also shows a warning such as "private key still embedded in hosting metadata", fix that before treating the setup as complete.
5. For `transport` keys, keep the private key on the hosting consumer side; hosting should only track the public key reference.

What hosting consumer code or UX should assume:
1. The hosting consumer already has the private key or knows where to find it.
2. Hosting will report useful metadata about the key:
   - whether it was `imported` or `generated`
   - how the public key was supplied
   - whether the private key is externally managed, exported to a file, stored as a client-realm secret, handed off into a consumer realm, or still requires user follow-up
3. Consumer tooling can discover exported private-key file references from keyring metadata, migrate private keys between client realms, and hand off a local exported file into the consumer client realm.
4. The consumer-facing UX should show that information plainly to the user instead of making them infer it.

### 4.2 Keyring paradigm

Use a dedicated keyring structure under default config root:

`<default_engine_config_dir>/hosting/`

Suggested contents:
1. `access_control.json` (role/policy config)
2. `keyring/` (active keys and metadata)
3. `audit/` (bounded append-only audit records)
4. `state/` (claims/sessions/runtime state checkpoints)

Key metadata expectations:
1. Persisted key metadata should record whether a public key was `imported` or `generated`.
2. When known, persisted metadata should also record:
   - how the public key was supplied (`file`, `inline`, generated, existing keyring)
   - whether private key material is not managed, exported to a file, stored in a client-realm secret, handed off into a consumer realm, or still embedded locally
   - whether an exported private-key file still exists, was purged after hand-off, or was purged without recorded hand-off
   - any operator warning that requires follow-up

Migration rule:
1. If a previous-format key file is detected, move it to `<name>.migrated` before importing into keyring metadata.
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
   - intended for local-only single-consumer operation
   - when the consumer dies or disconnects, hosting terminates all child processes it created
2. `shared`
   - multiple hosting consumers by role permissions
   - daemon runs detached/independently of any one consumer
   - consumers can disconnect and reconnect without forcing daemon/child-process restart

Hosting-created child processes include local LLM engine worker processes and sandboxed helper/tooling processes.

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

## 6. Auth/AuthZ Design Notes (Implementation-Aligned)

The following points are intended as stable design notes rather than phased status updates:
1. The role-hierarchy model is the supported runtime auth model for hosting.
2. Endpoint default mode, runtime override, and deterministic displaced-owner denial/reclaim behavior are part of the baseline command contract.
3. Setup, diagnostics, and migration helpers are expected operator surfaces:
   - `hosting_config`
   - `hosting_config --doctor`
   - previous key-file migration to `.migrated`
4. Safe-only no-auth policy is enforced both at startup and on runtime control-config changes.
5. Non-local auth bootstrap uses public-key challenge plus SSH session binding context.
6. Shared-secret session issuance (`auth-issue-session`) is local-only and must be denied for non-local connectivity profiles.
7. Auth lifecycle audit events and admin audit query surfaces are baseline operator controls, not optional add-ons.
8. Lifecycle profiles/policies, terminal-control gating, and shutdown sequencing are part of the supported lifecycle contract.
9. `auth-status` exposes contract metadata such as `daemon_version` and `capabilities`, but clients must treat retrieval failure under `require_auth=true` primarily as an auth/reachability-path problem.
10. SSH-targeted control paths require explicit pinned SSH host-key input from the client side; `accept-new` host-key onboarding is not a supported baseline mode.

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
3. Keep temporary no-auth recovery distinct from first-key bootstrap under `require_auth=true`.

## 8. Minimal Configuration Flow (Daemon Not Started)

Provide one user-facing setup script (for example `hosting_setup`) that asks hosting-consumer context first, suggests an auto-configuration, and writes config only after the operator chooses to apply it.
Detailed script contract: `src/hosting/HOSTING_CONFIG_SCRIPT.md`.

### 8.1 Up-front context collection

The setup script should collect enough context before showing low-level options:

1. Who consumes hosting?
   - local experiment only: leave hosting unconfigured; any same-user local consumer is implicitly admin
   - same-box backend consumer: local long-running backend under the same user account
   - SSH relay/tunnel consumer: remote backend reaches local hosting through SSH
   - remote backend consumer: direct/proxied remote access
2. What lifecycle does the consumer need?
   - single exclusive consumer: consumer death/disconnect stops the hosting daemon and all hosting-created children
   - reconnectable/shared daemon: daemon remains detached so consumers can reconnect
3. What access shape is expected?
   - single user, same as admin: one operator/admin identity
   - many roles: separate admin and user access keys
   - multi-user: more users and granular roles, managed after bootstrap
4. What credential style is preferred?
   - SSH keys: more secure baseline
   - local password/shared-secret convenience: easier but less secure, and session issuance is only valid in `local_only`
   - no auth local-only: only valid for local single-user exclusive safe profile
5. For SSH relay or remote backend consumers, can setup perform administrator/root changes on the target host?
   - no admin/root access: use user-scoped SSH setup only
   - admin/root available interactively: offer explicit elevated steps without storing the password
   - admin/root managed externally: emit instructions and diagnostics for an administrator or infrastructure tool
   - elevated execution, when requested, uses platform-native prompts such as Windows UAC, macOS authorization, Linux/Unix `pkexec`, or terminal `sudo`; hosting config does not collect the password
   - after the suggested remote configuration, interactive setup offers to generate the admin script, run elevated setup now, or skip it

The setup script should then show a suggested auto-configuration and follow-up actions before asking whether to apply, customize, or leave hosting unconfigured.

### 8.2 Script input intents

1. `local_only` (local-only hosting consumers)
2. `ssh_tunnel_only` (SSH-mediated remote hosting consumers; current implementation uses SSH relay for daemon control)
3. `truly_remote` (non-loopback direct or proxied remote hosting consumer access)

### 8.3 Auto-configuration projections

Recommended projections:

1. Local experiment only
   - do not write hosting access files
   - any same-user local consumer is treated as implicit admin because no hosting auth boundary has been configured
   - leave hosting unconfigured until a real long-running hosting consumer needs stable access
2. Single user, same as admin
   - default to `local_only`
   - default to `exclusive`
   - allow `require_auth=false` only when the safe-only gate passes
   - passwords/no-auth may be convenient, but SSH keys are the more secure option
3. Many roles
   - require auth
   - provision/bootstrap admin first
   - add separate user/operator keys later from hosting consumer admin UI or RBAC tooling
4. Multi-user
   - require auth
   - default to `shared`
   - expect more keys and more passwords/passphrases to manage
   - add/edit users and granular roles later from hosting consumer admin UI or RBAC tooling
5. SSH relay/remote access
   - require auth
   - require explicit SSH host-key pinning
   - require SSH relay/transport setup
   - at least one `transport` role SSH key must exist for the relay/transport trust layer
   - shared-secret keys cannot issue remote sessions in these modes

### 8.4 Common script outputs

1. Create/verify `<default_engine_config_dir>/hosting/` structure.
2. Initialize access control config and keyring metadata.
3. Register first admin SSH public key.
4. Choose persistent default endpoint mode (`exclusive` or `shared`).
5. Choose lifecycle profile (`foreground_terminal_bound|detached_user_process|service_managed`).
6. Set `require_auth` based on safe-only policy.
7. Emit platform-specific start instructions.
8. Emit hosting-consumer follow-up actions, such as importing a private key, configuring SSH relay/transport, or opening the consumer admin UI to add user keys.

### 8.5 External steps by intent

1. Local-only
   - bind daemon control to local IPC only
   - optional `require_auth=false` if safe-only gate passes
2. SSH tunnel-only
   - bind daemon control to local IPC only on host
   - use SSH relay for daemon control
   - enforce auth and SSH-bound session usage
   - use `hosting_config_cli --transport-harden-ssh` to compose client profile creation, forced-command authorized-key installation, hosting `transport` RBAC registration, and strict SSH validation
   - use `hosting_config_cli --transport-admin-setup` only for explicit elevated SSH service/firewall/user-linger follow-up actions
   - hosting config tooling can provision realm-local SSH client artifacts from a transport bootstrap profile:
     - materialized private key
     - pinned known_hosts file
     - SSH config snippet using `StrictHostKeyChecking yes`
   - hosting config tooling can install the transport public key into a user-scoped server-side `authorized_keys` file
     - the default installed entry must be forced-command hardened to the hosting relay entrypoint
     - the default installed entry must disable interactive PTY and common forwarding features unless an operator explicitly chooses a broader SSH access mode
     - the same public key must be registered in hosting auth state with role `transport` so SSH access and hosting transport enforcement cannot drift silently
   - private key material must remain on the hosting consumer side, never in server-side `authorized_keys`
3. Truly remote
   - explicit non-loopback bind (or reverse proxy) by admin choice
   - enforce auth, role separation, short token TTL
   - require firewall/network policy setup outside daemon

### 8.6 Taking Ownership Of An Unconfigured Daemon

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
   - hosting consumers must warn the operator to configure hosting access as soon as possible after startup
5. Current local backend/channel implementation also provides a local recovery helper:
   - `reset_hosting_access`
   - local-helper only; not available over daemon RPC
   - stops local daemon, then clears only auth state from local control state
   - preserves unrelated control config and runtime metadata outside `control_config.auth`

Unsupported client takeover path:
1. There is no supported daemon command that lets a normal client take full control without first configuring access control.
2. There is no supported daemon command that lets a normal client flip an already configured/authenticated daemon to `require_auth=false`.
3. `set-control-config` remains an authenticated admin control operation once auth is active.

Current implementation note:
1. `require_auth=true` with `keys_count == 0` is still supported in code as the first-key bootstrap state.
2. It is not marked deprecated in the daemon code path.
3. In that state, `authorize_command()` allows `auth-upsert-key` and `auth-status` without a session token so the first key can be provisioned, but only for `local_only` connectivity.
4. Remote-capable profiles reject zero-key bootstrap with:
   - `zero_key_bootstrap_local_only`
5. This is narrower than `require_auth=false` because the daemon is not generally open; only the first-key provisioning path is open.

Local operator caveat:
1. If the caller can execute arbitrary local Python or otherwise read/write local state files as the daemon user, the caller is outside the daemon's client trust boundary and effectively has local-operator powers.
2. In that case the caller may be able to inspect or manipulate daemon state through OS/file/process access, but that is a consequence of local user-account access, not a supported auth/authz bypass.
3. This remains within the documented residual-risk boundary for local user-account compromise.

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
1. local user-account compromise (key/state/audit file tampering)
2. misuse by any process with equivalent local user privileges

Minimum controls to remain in this scenario:
1. daemon control remains local IPC only
2. endpoint mode remains `exclusive`
3. single-user admin-only key profile remains enforced
4. no tunnel/relay/public ingress is enabled
5. if shared-secret bootstrap is used, keep it local-only and avoid persistent plaintext secret storage
6. if daemon was auto-started from an unconfigured local state using temporary `require_auth=false`, operator must complete hosting access configuration as soon as possible and should not treat the temporary no-auth state as steady-state policy

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
1. local credential/key theft after local user-account compromise
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
2. local user-account compromise on SSH relay endpoint
3. SSH relay endpoint operational misconfiguration

Minimum controls to remain in this scenario:
1. host daemon control stays local-IPC-only
2. clients pin and verify the SSH host key explicitly; opportunistic `accept-new` onboarding is not supported
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

Escalate beyond baseline when:
1. threat model includes key replay/rapid credential churn
2. compliance or assurance requires hardware-backed keys
3. attack volume requires adaptive lockout/anomaly controls

## 11. Hosting Consumer Integration Contract

This section is the authoritative integration contract for hosting consumers.

### 11.1 Transport choices

1. Local control access uses local IPC discovered from the daemon PID file under `hosting/state/daemon.pid`.
2. The PID file carries local connection metadata such as:
   - `ipc_family`
   - `ipc_address`
   - `shutdown_token`
3. Remote control access uses the SSH relay pattern:
   - open SSH to the target host
   - execute `python -m hosting.engine_host_cli --relay-wrapper` as a forced command
   - deny PTY allocation for the transport key; this key must not provide an interactive shell
   - bridge JSON-RPC traffic over SSH stdio
4. Remote control always requires SSH to execute the relay wrapper. A running daemon alone is not remotely controllable because daemon control is local IPC only.
5. When the daemon is already running, the relay wrapper attaches to it through PID-file local IPC metadata.
6. When the daemon is not running, relay wrapper auto-start is only attempted if saved hosting config is remote-enabled, `require_auth=true`, at least one auth key is registered, and lifecycle is `detached_user_process`.
7. Remote hosting consumers must not rely on opportunistic first-connect SSH trust; pinned host-key material is required.
8. Consumers that cannot execute any remote SSH command do not currently have a full remote control-plane transport.
9. Straight SSH port forwarding to daemon TCP control is TBD and is blocked server-side today.
10. Standard HTTP ingress, when needed, is handled by the separate `--daemon-http` process or by an external reverse proxy in front of loopback-only listeners. This ingress is for worker HTTP traffic and health, not full daemon control-plane commands.

### 11.2 Consumer-local realm and key custody

1. The hosting consumer side may maintain its own hosting realm under `<default_engine_config_dir>/hosting_client/<realm>/`.
2. The consumer realm may contain:
   - `client_access.json`
   - `keyring/keys.json`
   - `secrets/`
   - `managed_keys/`
   - `known_hosts/`
   - `ssh_config/`
   - `profiles/`
   - `audit/`
3. Long-lived consumer private keys should live in consumer-local custody:
   - imported existing file
   - exported managed file
   - client-realm secret record
4. Consumer secret records store OpenSSH private-key text. Password protection, when used, is OpenSSH private-key passphrase protection and is reported with `private_key_protection: "openssh_passphrase"`.
5. Hosting must not use normal runtime daemon RPC as the mechanism that returns private keys back to consumers.
6. Client-realm helpers support generated-key handoff:
   - discover exported private-key file references from a hosting or client keyring
   - import a private key from a file or sanitized inline paste argument
   - hand off a previously exported private-key file into the local consumer realm
   - migrate private-key secret records between client realms
   - optionally delete the exported source file after hand-off
   - explicitly purge a tracked exported file when the operator accepts possible key-material loss
7. `--client-import-key` remains an operator/script bridge for manual import and tests; consumer projects should prefer the client-realm API helpers directly.
8. The interactive RBAC/key-management menu exposes custody operations for operators: list exported private-key files, export stored client-realm keys for remote handoff, hand off local exported files, purge exported files with warning, and revoke RBAC keys.
9. A consumer project should not treat the setup machine's exported file path as its durable vault. It should copy/import/hand off the private key into its own realm or vault, then mark/purge the loose exported file.
10. Setup and doctor output use these custody states:
   - `exported_file`: a loose private-key file was created and is still tracked
   - `client_realm_secret`: private key is stored in a client-realm secret record
   - `private_key_export_purged_at`: exported file was intentionally deleted after client-realm hand-off
   - `private_key_export_purged_without_adoption_at`: exported file was deleted without recorded hand-off and may require rotation/recovery
   - `private_key_adopted_client_realm_root`: realm root that received the exported file

### 11.3 Auth flows consumers must support

1. Local-only bootstrap may use `auth-issue-session` with shared secret, but only for `local_only` connectivity.
2. Remote-capable consumers must support the public-key challenge flow:
   - `auth-begin-challenge`
   - local signature generation
   - `auth-complete-challenge`
3. Remote-capable consumers must include `_ssh_session_binding` metadata so issued sessions remain tied to the expected SSH route.
4. Consumers must treat missing or rejected SSH binding as a hard security failure, not as a retry-without-binding hint.

### 11.4 Transport bootstrap and profile handling

1. MITM-resistant first remote connection requires both:
   - transport private key material
   - pinned SSH host key (`ssh_known_hosts_line`)
2. Supported consumer bootstrap flow is:
   - import out-of-band bootstrap bundle
   - store transport private key in consumer-local custody
   - persist pinned host-key material
   - create/update a named consumer profile
   - validate the profile with strict SSH options before normal use
3. Imported consumer profiles may be consumed directly by `EngineHostControlChannel` through consumer-realm profile resolution and managed-key materialization.

### 11.5 Consumer behavior checklist

1. Read local PID/control metadata for local IPC instead of guessing transport details.
2. Use SSH relay or explicitly configured HTTP ingress according to deployment mode.
3. Implement public-key challenge auth for remote-capable consumers.
4. Inject `_ssh_session_binding` for SSH-mediated sessions.
5. Parse structured denials (`error_code`, `error_details`) and preserve them in UX/logs.
6. Distinguish auth-path failure from daemon-version incompatibility when `auth-status` metadata is unavailable.
7. Surface key provenance and custody state clearly:
   - `imported` vs `generated`
   - exported file vs client-realm secret
   - OpenSSH passphrase-protected vs unprotected private key
8. Validate imported transport profiles with strict host-key checking before treating them as ready.

## 12. Advanced Hardening and Risk Assessment

These are intentionally secondary to functional/usability baseline:
1. key rotation automation
2. replay-protection deepening beyond current challenge/session controls
3. hardware-backed key storage
4. advanced anomaly detection and adaptive lockouts

Each must be documented with:
1. threat introduced/expanded
2. compensating controls
3. scope of impact (for example local-only admin-only deployment may not need it)

## 13. Documentation Maintenance Requirements

1. `HOSTING_ACCESS.md` is the implementation-aligned architecture reference for hosting access.
2. `hosting_access_plan.md` remains the forward-looking plan document and should not duplicate the architecture contract.
3. Scenario-specific minimum controls and escalation triggers in Section 10 must stay current with the implemented command and policy behavior.
4. Client-facing guidance should be maintained in Section 11 instead of being split across drifting duplicate documents.
5. Any change to first-key bootstrap, no-auth safety rules, SSH host-key requirements, or client-realm custody rules must be reflected here at the same time as the code change.
