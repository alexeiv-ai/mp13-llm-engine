# `hosting_config_cli` Script Specification

Date: 2026-03-14
Status: Draft implementation contract (functional-first)

## 1. Purpose

`hosting_config_cli` is a user-facing setup/reconfiguration script for hosting access control.

Entrypoints:
1. `python -m hosting.hosting_config_cli`
2. `py hosting_config.py` from the repo root

It must:
1. Work before daemon startup.
2. Guide setup by user intent (`local_only`, `ssh_tunnel_only`, `truly_remote`).
3. Manage key onboarding (generate/import) and optional key export decisions.
4. Produce explicit `key_id` mapping for client configuration.

## 2. Inputs and modes

Supported modes:
1. Interactive wizard (default).
2. Non-interactive flags (CI/provisioning).
3. Reconfigure existing installation.
4. Read-only operator status/reporting.
5. Operator-oriented RBAC/session inspection and revocation.

Primary intent input:
1. `local_only`
2. `ssh_tunnel_only`
3. `truly_remote`

Secondary inputs:
1. Endpoint default mode (`exclusive` or `shared`).
2. Lifecycle profile:
   - `foreground_terminal_bound`
   - `detached_user_process`
   - `service_managed`
3. Auth mode (`require_auth=true` unless safe-profile exception is valid).
4. Role/key setup plan (who gets which role).

Operator command surfaces:
1. `--status`
2. `--doctor`
3. `--list-keys`
4. `--upsert-key`
5. `--revoke-key-id`
6. `--list-sessions`
7. `--revoke-session`
8. `--list-issued-tokens`
9. `--list-auth-audit`

## 3. Output layout and files

Target base directory:

`<default_engine_config_dir>/hosting/`

Managed files:
1. `access_control.json`
2. `keyring/keys.json`
3. `keyring/migrations.json`
4. `audit/setup_audit.jsonl`
5. `bootstrap/bootstrap_state.json`
6. `bootstrap/client_key_map.json`

Non-destructive migration:
1. Legacy key files are renamed to `.migrated`.
2. Imported metadata records original path and migration timestamp.

## 4. Wizard flow

## 4.1 Detect existing state

1. Detect legacy auth/key files.
2. Detect existing `hosting/` config.
3. Show summary: new install vs reconfigure.
4. Provide an operator menu before mutation:
   - review current config/file probes
   - run `--doctor`-style diagnostics
   - continue into guided configure/reconfigure
   - exit without changes

## 4.2 Connectivity intent selection

1. Explain security/operational implications of each intent.
2. Write `access_profile.connectivity_mode`.
3. Validate SSH dependency for non-local intents.

## 4.3 Endpoint mode and auth baseline

1. Select persistent endpoint mode (`exclusive`/`shared`).
2. Select lifecycle profile (foreground/detached/service-managed).
3. Set `require_auth`:
   - default `true`
   - allow `false` only when safe-profile checks pass
4. Persist config and show effective policy summary.

## 4.4 Key onboarding

For each planned identity:
1. Select role (`admin`, `config_editor`, `worker_user`, `model_user_with_model_control`, `model_user`, `diagnostic_user`, optional `transport`).
2. Choose key source:
   - generate new keypair (script-assisted)
   - import existing public key
3. Assign `key_id`.
4. If importing interactively, prompt for either:
   - public-key file path
   - pasted inline public key
4. Apply role-specific constraints (allowed engines/config selectors).

## 4.5 Key export policy guidance

Per key, ask:
1. Export private key for client use?
2. Keep private key unmanaged by host?

Guidance:
1. `transport` keys: private key should never be stored by hosting.
2. User role keys: export optional based on deployment model.
3. Long-lived SSH identities + short-lived session tokens for duration gating.

## 4.6 Client configuration output

Generate client-ready mapping table:
1. `client_name`
2. `key_id`
3. `role`
4. required client settings fields
5. connectivity mode assumptions (local/tunnel/remote)

Example output fields:
1. `engine_host_key_id`
2. `engine_host_session_scope`
3. `engine_host_session_ttl_seconds`
4. endpoint host/port or SSH relay settings

## 4.7 Reconfiguration path

1. Load and show current settings.
2. Preview changes before apply.
3. Allow staged apply:
   - connectivity/profile changes
   - role/key changes
   - endpoint mode changes
4. Emit restart guidance when changes require daemon restart.

## 4.8 Operator-oriented runtime actions

Read-only commands:
1. `--status` prints a short human-readable hosting-access summary.
2. `--doctor` prints diagnostics without mutating configuration.
3. `--list-keys` shows persisted RBAC keys.
4. `--list-sessions` shows active auth sessions with role, scope, TTL, and SSH binding.
5. `--list-issued-tokens` shows issued runtime engine/resource tokens.
6. `--list-auth-audit` shows auth audit events.

Mutating operator commands:
1. `--upsert-key` creates or updates one persisted RBAC key.
2. `--revoke-key-id` revokes one key and any sessions issued from it.
3. `--revoke-session` revokes one active session token.

Output contract:
1. Human-readable terminal output is the default.
2. `--json-output` is optional for automation/integration scenarios.

## 5. First implementation scope

1. Configuration authoring and validation.
2. Key import/register flow.
3. Optional key generation helper integration.
4. Client mapping report generation.
5. Audit log for setup actions.

Out of scope for first cut:
1. Full auto-firewall management.
2. Hardware-backed key management.
3. Advanced anomaly/risk scoring.

## 6. Troubleshooting deliverables (future)

Planned deliverables:
1. `hosting_config --doctor` diagnostics command.
2. Structured error catalog with remediation suggestions.
3. Common failure playbooks:
   - invalid key format/import
   - role/scope mismatch
   - unsafe `require_auth=false` rejection
   - tunnel/remote reachability issues
4. “What changed” audit diff after reconfigure.

## 7. Testing deliverables (future)

Planned test suites:
1. Unit tests for intent/profile validation.
2. Migration tests for `.migrated` behavior.
3. Role/key onboarding matrix tests.
4. End-to-end smoke tests for local/tunnel/remote setup outputs.
5. Reconfiguration idempotency and rollback safety tests.

Current implementation note:
1. Interactive mode is the default entry path; use `--no-interactive` for flag-only operation.
2. The main implementation lives in `src/hosting/hosting_config_cli.py`.
3. Repo-root `hosting_config.py` proxies to that CLI entrypoint.
4. Human-readable output is the default; `--json-output` is available when machine-readable output is needed.
5. Initial `--doctor` command is implemented with SSH dependency, path readability/writability, and runtime policy checks.
6. Interactive import-key setup now prompts for public-key source material instead of requiring separate CLI flags.
7. Generated-key setup uses `--generated-key-passphrase` as the primary option name; `--key-passphrase` remains as a compatibility alias.
8. Operator-oriented RBAC essentials are implemented for persisted keys and runtime inspection:
   - key list/create-update/revoke
   - session list/revoke
   - issued-token listing
   - auth-audit listing
9. Lifecycle profile selection is supported via setup flags and persisted into control/setup artifacts.
10. Zero-key bootstrap under `require_auth=true` is local-only; remote-capable modes must have a pre-provisioned key before normal remote auth flows can begin.
