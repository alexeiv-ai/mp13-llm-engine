# Hosting Access Forward Plan: SSH Mutual Authentication Provisioning

Date: 2026-04-18
Status: Partially implemented plan
Scope: `src/hosting` setup/configuration tooling for SSH relay and remote-capable hosting access

## 1. Problem Statement

The current hosting access architecture already distinguishes hosting-layer RBAC identities from the SSH transport layer, and the client-realm helpers already support private-key custody, pinned `known_hosts` material, and strict SSH profile generation.

The remaining product/security gap is operator setup quality:

1. Remote hosting access depends on SSH mutual authentication.
2. SSH client authentication is only strong if the transport public key is registered on the SSH server side and bound to the hosting transport identity.
3. SSH server authentication is only strong if the hosting consumer pins the target server host key and connects with strict host-key checking.
4. Normal users are unlikely to configure `sshd`, `authorized_keys`, host-key pinning, and daemon lifecycle correctly without tooling.
5. Setup should therefore provide a first-class SSH transport hardening workflow instead of leaving the operator to infer several separate commands.

This plan builds on the existing client-side helpers instead of replacing them.

## 2. Security Goal

For `ssh_tunnel_only` and SSH-mediated remote daemon access, the setup flow should be able to produce and diagnose this state:

1. The hosting consumer has a private SSH transport key in consumer-local custody.
2. The hosting consumer has pinned SSH host-key material for the target.
3. The generated SSH client profile uses strict host-key checking.
4. The server-side user account has the transport public key in `authorized_keys`.
5. The server-side `authorized_keys` entry is constrained to the hosting relay command by default.
6. The same public key is registered in hosting auth state with role `transport`.
7. Hosting-layer remote session issuance requires public-key challenge auth and expected SSH session binding metadata.
8. Diagnostics can prove the whole chain or report the exact missing piece.

This closes the intended mutual-auth gap:

1. Client-to-server authentication is enforced by the server-side `authorized_keys` entry plus hosting `transport` key registration.
2. Server-to-client authentication is enforced by pinned `known_hosts` material and `StrictHostKeyChecking yes`.
3. Hosting-layer command authorization still comes from RBAC roles and sessions; `transport` remains an orthogonal trust context, not a user role that can issue hosting sessions.

## 3. Non-Goals and Boundaries

1. Do not store or replay admin/root passwords.
2. Do not silently edit machine-wide SSH server configuration.
3. Do not silently open firewall rules or install/start system services.
4. Do not promise protection after local user-account compromise.
5. Do not treat opportunistic `accept-new` host-key onboarding as a secure remote baseline.
6. Do not grant the transport key an unrestricted OS shell by default.

Admin/root operations may be supported only as explicit elevated add-ons using platform-native elevation or generated operator instructions.

## 4. Current Baseline

Implemented pieces already available:

1. Client-realm private-key storage and materialization.
2. Transport bootstrap bundle export/import.
3. Realm-local SSH config generation with `IdentityFile`, `UserKnownHostsFile`, `StrictHostKeyChecking yes`, and `IdentitiesOnly yes`.
4. Transport profile validation using strict SSH options.
5. User-scoped server-side `authorized_keys` installation.
6. Hosting `transport` role exists and cannot issue sessions directly.

Implemented hardening and first workflow slice:

1. `--transport-install-authorized-key` writes a forced-command entry by default.
2. The default forced command is `python -m hosting.engine_host_cli --relay`.
3. The default entry disables PTY, agent forwarding, X11 forwarding, and port forwarding.
4. The CLI registers the same public key into hosting auth state as role `transport`.
5. `--transport-harden-ssh` composes bootstrap import, client SSH artifact provisioning, forced-command authorized-key installation, hosting `transport` RBAC registration, and strict SSH profile validation.
6. The interactive usage questionnaire asks remote/SSH users whether administrator/root changes are available, then includes that answer in recommendations.
7. `--doctor` can verify transport authorized-key presence/hardening and hosting transport RBAC registration/public-key match when the relevant transport key inputs are provided.
8. `--transport-admin-setup` generates elevated SSH service/firewall/user-linger setup scripts and can execute them through Windows UAC, macOS authorization, Linux/Unix `pkexec`, or terminal `sudo` when `--admin-setup-execute` is explicitly passed.

## 5. Proposed User Workflow

### 5.1 Intent Questionnaire Additions

Add one setup vector after the remote/SSH intent is known:

Question: "Can this setup perform administrator/root changes on the target host?"

Supported answers:

1. `no_admin_available`
   - The operator does not know the admin/root password or cannot elevate.
   - Recommend user-scoped SSH setup only.
   - Do not attempt service, firewall, or machine-wide SSH changes.
2. `admin_available_interactive`
   - The operator can approve UAC/sudo/polkit prompts interactively.
   - Recommend the best managed option for the platform.
   - Never capture or persist the password.
3. `admin_managed_externally`
   - The operator can hand instructions to an admin or infrastructure tool.
   - Generate exact commands/config snippets and a diagnostic checklist.

The question should be phrased as an environment capability, not as a request for the password.

### 5.2 Recommended Paths

For `ssh_tunnel_only`:

1. If `no_admin_available`:
   - install user-scoped forced-command `authorized_keys`
   - keep daemon lifecycle as `detached_user_process` if reconnect/auto-start is needed
   - use relay wrapper auto-start under the target user account only
2. If `admin_available_interactive`:
   - verify or enable system SSH server only through explicit elevated action
   - optionally configure daemon auto-start using systemd user linger, Windows Task Scheduler, launchd, or service-managed profile as appropriate
   - still prefer user-scoped forced-command key for least privilege unless machine policy requires central `sshd_config`
3. If `admin_managed_externally`:
   - write an operator bundle containing exact target-side instructions
   - include expected post-change diagnostics

For `truly_remote`:

1. Require explicit non-loopback daemon ingress or reverse proxy choice.
2. Keep SSH transport hardening available for administrative bootstrap and relay operation.
3. Require external firewall/proxy policy confirmation.

## 6. CLI Design

Add a first-class subcommand group under `hosting_config_cli` using flags consistent with the existing argparse style.

### 6.1 New Primary Action

Add:

`--transport-harden-ssh`

This action should compose existing helpers and new checks into one idempotent workflow.

Inputs:

1. `--transport-target`
2. `--transport-key-id`
3. one of:
   - `--transport-public-key-file`
   - `--transport-public-key-inline`
   - generated client transport key from existing client-key helper
4. one of:
   - `--ssh-known-hosts-file`
   - `--ssh-known-hosts-line`
   - explicit future host-key scan mode guarded by fingerprint confirmation
5. `--ssh-authorized-keys-file`
6. `--ssh-authorized-key-command`
7. `--ssh-authorized-key-unrestricted` for exceptional operator override
8. `--client-realm`
9. `--transport-profile-name`
10. `--admin-capability` with values from Section 5.1

Outputs:

1. server-side authorized key path
2. forced command and restriction status
3. hosting `transport` key id and public-key fingerprint
4. client realm profile path
5. known_hosts file path
6. SSH config snippet path
7. validation command summary
8. follow-up actions that still require admin/root or firewall changes

### 6.2 Existing Action Compatibility

Keep these existing actions:

1. `--transport-export-bootstrap`
2. `--transport-import-bootstrap`
3. `--transport-provision-ssh-artifacts`
4. `--transport-install-authorized-key`
5. `--transport-validate-profile`

The new hardening action should orchestrate them where possible instead of duplicating logic.

## 7. Server-Side Authorized Key Requirements

Default managed key line:

```text
command="python -m hosting.engine_host_cli --relay",no-pty,no-agent-forwarding,no-X11-forwarding,no-port-forwarding ssh-ed25519 <key> <comment>
```

Rules:

1. Default to forced relay command.
2. Default to no PTY.
3. Default to no agent forwarding.
4. Default to no X11 forwarding.
5. Default to no port forwarding.
6. Use a managed marker block for idempotent replacement.
7. Preserve unrelated `authorized_keys` lines.
8. Reject malformed SSH public keys.
9. Surface an explicit warning if `--ssh-authorized-key-unrestricted` is used.

Future optional command forms:

1. `python -m hosting.engine_host_cli --relay`
2. `python -m hosting.engine_host_cli --relay --start-daemon-if-needed`
3. platform-specific relay wrapper script path generated by setup

The wrapper command must not provide a general shell.

## 8. Hosting Auth Coupling

Whenever a transport key is installed into server-side SSH for hosting, the CLI should also register or verify:

1. `key_id = transport_key_id`
2. `role = transport`
3. `auth_method = public_key`
4. `public_key = same public key as authorized_keys`
5. `disabled = false`

Doctor should detect:

1. SSH key installed but missing from hosting auth state.
2. Hosting transport key exists but differs from authorized_keys.
3. Transport role key uses non-public-key auth method.
4. Transport role key is disabled.

## 9. Client-Side Host-Key Pinning

The setup flow should require pinned host-key material before remote profile activation.

Accepted inputs:

1. full `known_hosts` line
2. file containing a full `known_hosts` line
3. future fingerprint-verified host-key scan mode

Generated SSH config must include:

```text
UserKnownHostsFile <client-realm>/known_hosts/<profile>.known_hosts
StrictHostKeyChecking yes
IdentitiesOnly yes
IdentityFile <materialized-key>
```

Do not use `accept-new` in generated profiles.

## 10. Admin-Capability Handling

### 10.1 No Admin Available

Supported actions:

1. write user-scoped `authorized_keys` if current user owns the target file
2. generate client realm artifacts
3. register hosting `transport` key
4. validate strict SSH connection
5. configure detached user-process lifecycle

Unsupported actions:

1. install or start system SSH server
2. edit machine-wide `sshd_config`
3. open firewall rules
4. install service-managed daemon

### 10.2 Admin Available Interactively

Supported actions:

1. detect platform and SSH server availability
2. offer explicit elevated setup step
3. use platform-native elevation, not password capture
4. write a preflight summary before elevation
5. rerun diagnostics after elevation

Platform candidates:

1. Windows: OpenSSH Server optional feature, `sshd` service, firewall rule, Task Scheduler or service-managed daemon.
2. Linux systemd: OpenSSH server package/service, firewall policy notes, user lingering for user services, optional system service.
3. macOS: Remote Login/launchd guidance, user LaunchAgent for daemon lifecycle.

### 10.3 Admin Managed Externally

Generate instructions:

1. exact `authorized_keys` managed block
2. expected server SSH host-key fingerprint/known_hosts line
3. service/firewall commands when applicable
4. diagnostic command the non-admin user can run afterward

## 11. Auto Daemon Start Over SSH

Remote auto-start is acceptable only if it is implemented as a constrained relay behavior.

Recommended design:

1. The forced command invokes a hosting relay wrapper.
2. The wrapper checks local hosting config.
3. If lifecycle allows remote auto-start, it starts or connects to the daemon under the target user account.
4. The wrapper then bridges JSON-RPC over stdio.
5. The wrapper refuses if the configured access profile is unsafe, no pinned/transport context is present, or lifecycle policy forbids terminal control.

Constraints:

1. It must not spawn a general shell.
2. It must not elevate privileges.
3. It must not downgrade `require_auth`.
4. It must preserve audit events for auto-start attempts.

## 12. Diagnostics Plan

Extend `--doctor` with an SSH transport hardening section.

Checks:

1. `ssh_dependency`: SSH client tools are available.
2. `transport_profile_exists`: selected client profile exists.
3. `transport_private_key_present`: client private key material can be materialized.
4. `transport_known_hosts_present`: pinned known_hosts file exists.
5. `transport_known_hosts_strict`: profile uses strict host-key checking.
6. `transport_authorized_key_present`: server-side key block exists when local filesystem path is available.
7. `transport_authorized_key_hardened`: key block contains forced command and restriction options.
8. `transport_rbac_registered`: hosting transport key exists.
9. `transport_rbac_matches_ssh`: hosting transport public key matches authorized_keys key.
10. `transport_ssh_probe`: strict SSH probe succeeds.
11. `transport_relay_probe`: relay command reaches daemon or reports expected daemon-not-started state.
12. `transport_ssh_binding_enforced`: remote session path includes expected SSH binding metadata.

Doctor output should include:

1. root cause
2. impact
3. exact remediation command
4. whether admin/root is required

## 13. Test Plan

Unit tests:

1. authorized_keys install writes forced-command restricted line by default.
2. authorized_keys install preserves unrelated entries.
3. authorized_keys install replaces managed block idempotently.
4. unrestricted override omits forced command and restrictions.
5. CLI install action registers matching hosting `transport` role key.
6. CLI hardening action composes client profile, known_hosts, authorized_keys, and transport keyring state.
7. doctor flags raw/unrestricted transport key entries.
8. doctor flags missing or mismatched hosting `transport` role key.
9. doctor flags missing pinned known_hosts material.
10. strict SSH validation command includes expected options.

Integration tests with mocked subprocess/elevation:

1. strict SSH probe passes.
2. strict SSH probe failure is reported without fallback to non-strict host checking.
3. admin-capability choices produce correct recommendations and do not request passwords.
4. remote auto-start wrapper refuses unsafe lifecycle policy.

Manual/live tests:

1. Windows built-in OpenSSH Server with standard user target account.
2. Linux OpenSSH with systemd user service and linger enabled.
3. macOS Remote Login with user LaunchAgent.

## 14. Implementation Sequence

1. Add admin-capability question to setup questionnaire and projection output. Implemented.
2. Add `--transport-harden-ssh` action that orchestrates existing transport helpers. Implemented.
3. Add server-side authorized_keys hardening diagnostics. Implemented for explicit transport key/path inputs.
4. Add hosting transport key consistency diagnostics. Implemented for explicit transport key inputs.
5. Add client profile strictness diagnostics.
6. Add relay probe that validates daemon reachability and SSH binding context.
7. Add optional auto-start relay wrapper with lifecycle gates.
8. Add platform-specific elevated instruction generation. Implemented for SSH service/firewall/user-linger setup.
9. Add interactive elevated setup only after the generated-instruction path is stable. Implemented as explicit `--admin-setup-execute`, not automatic wizard execution.

## 15. Acceptance Criteria

For `ssh_tunnel_only`, a successful hardened setup should prove:

1. client profile has pinned host key and strict checking
2. server accepts only the configured transport key for relay command use
3. transport key cannot open a normal shell by default
4. hosting auth state has matching role `transport`
5. relay can reach the daemon or trigger allowed user-scoped auto-start
6. remote hosting session issuance fails without public-key challenge and SSH binding
7. doctor returns `ok` for all SSH transport hardening checks
