# `hosting_config_cli` Script Specification

Date: 2026-04-18
Status: Implementation-aligned operator contract

## 1. Purpose

`hosting_config_cli` is the setup, diagnostics, RBAC, client-realm key, and transport-bootstrap utility for hosting access control.

Entrypoints:
1. `python -m hosting.hosting_config_cli`
2. `python -m hosting.engine_host_cli --hosting-config`
3. repo-root compatibility wrappers when present

The script is designed for both first-time setup before daemon startup and reconfiguration/inspection of an existing hosting installation.

Terminology:
1. "Hosting consumer" means the long-running backend process that talks to the hosting daemon.
2. A UI may guide or display these actions, but is usually not the direct hosting protocol peer.
3. Older flags/docs may still say "client"; read that as hosting consumer unless UI is explicitly mentioned.

## 2. Modes

Supported top-level modes:
1. Interactive wizard: default when no read-only/operator action is selected.
2. Non-interactive setup: `--no-interactive` plus setup flags.
3. Status: `--status`.
4. Diagnostics: `--doctor`.
5. RBAC/session/audit operations:
   - `--list-keys`
   - `--upsert-key`
   - `--revoke-key-id`
   - `--list-sessions`
   - `--revoke-session`
   - `--list-issued-tokens`
   - `--list-auth-audit`
6. Client-realm private-key operations:
   - `--client-list-keys`
   - `--client-generate-key`
   - `--client-import-key`
   - `--client-export-key`
7. Transport bootstrap operations:
   - `--transport-export-bootstrap`
   - `--transport-import-bootstrap`
   - `--transport-validate-profile`
   - `--transport-provision-ssh-artifacts`
   - `--transport-install-authorized-key`

Human-readable terminal output is the default. `--json-output` is available for automation.

## 3. Interactive Wizard Contract

### 3.1 Operator menu

Before mutation, the wizard shows:
1. configure hosting now
2. review status details
3. run doctor diagnostics

Controls:
1. Enter accepts the default/current value.
2. `b` goes back where the current prompt supports back navigation.
3. `c` prints staged setup changes.
4. `q`, `quit`, or `exit` quits.
5. Ctrl+C quits immediately and drops staged setup changes.

On `q`, if setup changes are staged, the script prints them and asks whether to save/apply them. On Ctrl+C it does not ask.

### 3.2 Context-first setup

The wizard collects hosting-consumer context before exposing low-level fields:

1. Who consumes hosting?
   - local experiment only
   - same-box backend consumer
   - SSH relay/tunnel consumer
   - remote backend consumer
2. What lifecycle is needed?
   - single exclusive consumer
   - reconnectable/shared daemon
3. How many access roles/users are expected?
   - single user, same as admin
   - many roles
   - multi-user
4. Preferred credential style?
   - SSH keys
   - local password/shared-secret convenience
   - no auth, local only

The script then prints a suggested auto-configuration and follow-up actions. The operator can apply it, customize from it, go back, or leave hosting unconfigured.

### 3.3 Local experiment/no-write path

If the operator chooses local experiment only, the suggested action is to leave hosting unconfigured.

Semantics:
1. No hosting access files are written.
2. Any same-user local consumer is effectively an implicit admin because no daemon auth boundary has been configured yet.
3. The operator should rerun setup when a real long-running hosting consumer needs stable daemon access.

Interactive setup delays directory creation until the operator confirms final apply.

### 3.4 Auto-configuration projections

The wizard projects context into setup fields:

1. Single user, same as admin:
   - usually `local_only`
   - usually `exclusive`
   - `require_auth=false` only when safe-profile checks pass
2. Many roles:
   - authenticated access
   - bootstrap/admin key first
   - additional user/operator keys added later through admin UI or RBAC tooling
3. Multi-user:
   - authenticated shared access
   - more keys/passphrases to manage
   - additional users and granular roles managed after bootstrap
4. SSH relay/tunnel or truly remote:
   - auth required
   - SSH keys/public-key challenge required
   - explicit SSH host-key pinning required
   - at least one `transport` role SSH key required for relay/transport trust

### 3.5 Low-level setup review

After suggestion/customization, the wizard reviews:
1. endpoint mode: `exclusive` or `shared`
2. lifecycle profile:
   - `foreground_terminal_bound`
   - `detached_user_process`
   - `service_managed`
3. `require_auth`
4. key handling action: keep existing or replace
5. key source: generate or import
6. admin key id
7. permission action: none or best-effort tighten

The final apply prompt is the mutation boundary.

## 4. Connectivity and Credential Rules

Connectivity modes:
1. `local_only`: same box/user account; no off-host consumers.
2. `ssh_tunnel_only`: remote consumer reaches hosting through SSH relay/tunnel.
3. `truly_remote`: direct/proxied non-loopback remote access.

Endpoint modes:
1. `exclusive`: intended for local-only single-consumer use. Consumer death/disconnect stops the hosting daemon and all child processes it created.
2. `shared`: daemon runs detached/independently of consumers so they can reconnect.

Credential rules:
1. SSH public-key identities are the primary durable model.
2. Shared-secret/password session issuance is local-only.
3. An existing admin shared-secret key can issue/control a session in `local_only`.
4. The same shared-secret key cannot issue sessions in `ssh_tunnel_only` or `truly_remote`; use public-key challenge instead.
5. `transport` role requires `public_key`.
6. `require_auth=false` is valid only for the local-only exclusive safe profile.

## 5. Output Layout

Target hosting directory:

`<default_engine_config_dir>/hosting/`

Managed files:
1. `access_control.json`
2. `keyring/keys.json`
3. `keyring/migrations.json`
4. `audit/setup_audit.jsonl`
5. `bootstrap/bootstrap_state.json`
6. `bootstrap/client_key_map.json`
7. `state/` runtime/local state files

Client realm directory:

`<default_engine_config_dir>/hosting_client/<realm>/`

Client realm may contain private-key secret records and transport profiles used by the hosting consumer side.

## 6. Key Handling

Setup registers the first admin public key.

Key source options:
1. import an existing public key
2. generate a new SSH keypair
3. keep existing key when reconfiguring an existing setup

Generated private-key behavior:
1. The public key is registered with hosting.
2. The private key can be exported to a file.
3. If export is requested interactively and no path is provided, the wizard prompts with a default under `hosting/keyring/<key-id>.private`.
4. If not exported, generated private key material is stored in the client realm secret store.
5. Client realm secret records store OpenSSH private-key text. When `--client-secret-password` is supplied, the private key is protected with OpenSSH private-key passphrase protection.

Imported-key behavior:
1. Hosting stores the public key and metadata.
2. The private key remains under external custody.

## 7. RBAC and Runtime Operations

Implemented RBAC/operator surfaces:
1. list keys
2. create/update key
3. revoke key and sessions
4. list sessions
5. revoke session
6. list issued runtime tokens
7. list auth audit events

`--upsert-key` supports:
1. `--auth-method public_key`
2. `--auth-method shared_secret`
3. role selection through `--key-role`
4. optional `--allowed-configs`
5. optional `--allowed-engines`
6. disabled-key creation/update

`shared_secret` is intentionally documented as local-only for session issuance.

## 8. Client-Realm and Transport Bootstrap

Client-realm commands manage private-key custody for the hosting consumer side:
1. generate a client private key
2. import a client private key
3. export a client private key
4. list client-realm key metadata

Transport bootstrap commands support out-of-band transfer of remote SSH transport material:
1. harden SSH transport end to end
2. export bootstrap bundle
3. import bootstrap bundle into client realm
4. provision realm-local SSH artifacts
5. install the transport public key into a user-scoped server-side `authorized_keys`
6. validate imported transport profile

Consumer-side provisioning writes user-scoped/client-realm artifacts, not global SSH files:
1. materialized private key under the client realm `managed_keys/`
2. pinned known_hosts file under the client realm `known_hosts/`
3. SSH config snippet under the client realm `ssh_config/`
4. ready command form: `ssh -F <realm ssh config> <alias>`

The generated SSH config uses:
1. `IdentityFile`
2. `UserKnownHostsFile`
3. `StrictHostKeyChecking yes`
4. `IdentitiesOnly yes`

Server-side authorized-key installation writes only public key material:
1. default path is `~/.ssh/authorized_keys`
2. `--ssh-authorized-keys-file` can override the path
3. a managed marker block is used so reruns update instead of duplicating
4. private key material is never installed server-side
5. the default entry is forced-command hardened to `python -m hosting.engine_host_cli --relay-wrapper`
6. the default entry disables PTY, agent forwarding, X11 forwarding, and port forwarding
7. the same public key is registered in hosting auth state with role `transport`

Straight SSH port forwarding to daemon TCP control is TBD and blocked server-side today. The supported transport-key installation path is the forced-command relay wrapper; port-forward-only SSH keys are not a full control-plane transport in this release.

Relay wrapper runtime behavior:
1. SSH must be able to execute the wrapper; a running daemon alone is not remotely controllable because daemon control is local IPC only
2. if the daemon is already running, the wrapper attaches through PID-file local IPC metadata
3. if the daemon is not running, wrapper auto-start is only attempted when saved hosting config is remote-enabled, `require_auth=true`, has registered keys, and uses `detached_user_process` lifecycle
4. wrapper execution itself does not prompt for or store an administrator/root password
5. control operations sent through the relay still require the normal hosting auth/session required by that command

Transport keys must not be granted PTY or shell access in the supported relay posture. If an operator intentionally grants broader SSH rights to the same key, that deployment takes on additional local-account compromise risks outside the hosting transport contract.

Remote-capable setup still requires pinned SSH host-key material. Opportunistic `accept-new` host-key trust is not a supported baseline.

When the usage questionnaire selects an SSH relay or remote backend consumer, setup asks whether administrator/root changes are available on the target host:
1. no admin/root access: recommend user-scoped forced-command SSH setup only
2. admin/root available interactively: recommend explicit elevated setup steps without storing the password
3. admin/root managed externally: recommend generated administrator instructions and post-change diagnostics

`--transport-admin-setup` covers the explicit elevated follow-up path:
1. default behavior is dry-run script generation
2. `--admin-setup-execute` invokes platform-native elevation
3. Windows uses UAC through elevated PowerShell
4. macOS uses the system authorization dialog through `osascript`
5. Linux/Unix uses `pkexec` when a GUI session is available, otherwise `sudo`
6. the setup tool never prompts for or stores the administrator/root password itself

Interactive setup offers the same admin setup path after the remote recommendation is accepted:
1. generate admin setup script
2. run elevated admin setup now
3. skip admin setup

The elevated option remains an explicit menu choice.

## 9. Diagnostics

`--doctor` is implemented and non-mutating.

Checks include:
1. SSH dependency/readiness.
2. default config and hosting root existence.
3. control state and keyring readability.
4. hosting root writability.
5. zero-key remote bootstrap policy.
6. runtime policy safety.
7. client-realm access readability.
8. client transport profile integrity.
9. generated admin private-key secret encryption posture when applicable.
10. transport authorized-key presence/hardening when a transport key id or authorized_keys path is provided.
11. hosting transport RBAC registration and public-key match when a transport key id is provided.

Doctor output includes:
1. all checks
2. summary
3. must-fix/warning detail sections with root cause and impact
4. recommendations
5. guided setup follow-up prompt for fixable missing setup state

## 10. Migration and Permissions

Legacy key files are renamed to `.migrated`; they are not auto-deleted.

Migration events are recorded in audit/migration metadata.

Permission action:
1. `none`: do not change filesystem permissions.
2. `tighten`: best-effort private permissions on hosting files/directories.

## 11. Current Limits

1. The setup wizard bootstraps the first admin identity; it does not create the full multi-user key matrix.
2. Additional users/roles are added later through RBAC tooling or hosting consumer admin UI.
3. Firewall, proxy, and machine-wide SSH server policy are outside this script today.
4. Realm-local SSH client artifacts are supported through transport provisioning and `--transport-harden-ssh`, but the script does not edit global `~/.ssh/config`.
5. User-scoped `authorized_keys` installation is supported with forced-command hardening.
6. Machine-wide SSH service/firewall setup is available only through explicit `--transport-admin-setup`; it does not edit arbitrary `sshd_config` policy.
7. Hardware-backed key storage is not implemented.
8. Local user-account compromise remains outside baseline prevention guarantees.
