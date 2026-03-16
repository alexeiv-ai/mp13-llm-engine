# MP13 Hosting

This file documents the current hosting implementation in `src/hosting` and practical usage workflows.

For the hardened security/roles/key-management design (including local vs remote SSH/password scenarios, exclusive/shared ownership semantics, and daemon lifecycle guarantees), see:
- `src/hosting/hosting_access.md`

For rollout sequencing and implementation tracking, see:
- `src/hosting/hosting_access_plan.md`

For client migration requirements due to intentional auth/authz breaking changes, see:
- `src/hosting/HOSTING_CLIENT_BREAKING_CHANGES.md`

Breaking notice:
- Legacy role names (`management`, `config`, `traffic`) are removed from clean-slate runtime auth paths.
- Use new roles (`admin`, `config_editor`, `worker_user`, `model_user_with_model_control`, `model_user`, `diagnostic_user`, optional `transport`).
- Prefer bootstrap/reconfiguration via `python -m hosting.engine_host_cli --hosting-config ...` or `python -m hosting.hosting_config ...`.
- Legacy role payloads are no longer accepted by runtime auth surfaces.

## 1. Architecture

Hosting is a control-plane plus guarded traffic bridge.

- Control-plane:
  - worker lifecycle (`spawn`, `shutdown`, `ensure-running`, `discover-running`)
  - config-driven worker startup
  - claims/tokens/resource ownership state
- Traffic bridge:
  - `proxy-request` forwards HTTP-like engine API requests over local IPC
  - `proxy-rpc-*` provides generic sync/async RPC over local IPC
  - traffic authorization and policy are enforced before forwarding

Workers are still separate processes. Hosting does not expose worker private keys and does not require worker ports to be publicly forwarded.

## 2. Security Model

Auth roles:
- `admin`
- `config_editor`
- `worker_user`
- `model_user_with_model_control`
- `model_user`
- `diagnostic_user`
- `transport` (orthogonal transport identity)

Transport role constraints:
- key onboarding must use `auth_method=public_key`
- transport identity cannot issue command authorization sessions/challenges

Remote bootstrap SSH-binding constraints:
- if `access_profile.connectivity_mode` is `ssh_tunnel_only` or `truly_remote`:
  - shared-secret `auth-issue-session` requires `ssh_binding`
  - public-key `auth-begin-challenge` requires `ssh_binding`
  - missing binding is denied with `ssh_binding_required_for_remote_connectivity`

Remote command-path SSH-binding constraints:
- when connectivity mode is non-local, session-backed commands require `_ssh_session_binding`
- session must include persisted SSH binding metadata
- unbound legacy sessions are denied in non-local mode (`ssh_binding_required_for_remote_connectivity`)

Admin-only invalidation controls:
- `auth-revoke-key` and `auth-revoke-session` are admin-only
- non-admin control roles are denied with `insufficient_role`
- `auth-audit-list` is admin-only

Auth audit trail:
- control state now records `auth_audit_events` for:
  - `auth_upsert_key`
  - `auth_revoke_key`
  - `auth_revoke_session`
- `auth-audit-list` provides paged/filterable query access to these events

Session scopes:
- `control`
- `config`
- `traffic`

When `require_auth=true`, hosting commands require `session_token` in payload (except bootstrap-safe flows and key-based session issuance).

Config path hardening:
- exposed selectors are limited to:
  - `default`
  - hosted config names in `<default_config_dir>/backend/configs`
- direct absolute/relative traversal selectors are rejected

Traffic policy hardening (`control_config.traffic_policy`):
- method allowlist
- path-prefix allowlist
- request header allowlist
- response header allowlist
- request/response size caps
- optional blocking of forwarded `Authorization` header

Daemon-side claim ACL hardening:
- claim actor identity is daemon-derived from authenticated session:
  - actor id format: `key:<key_id>`
  - client-supplied `backend_id` is ignored for daemon-enforced claim identity
- owner keepalive and orphan policy are daemon-enforced (`control_config.claim_acl_policy`):
  - `owner_ttl_seconds` (default `120`)
  - `audit_event_limit` (default `200`)
- non-localhost management connection restriction:
  - non-localhost callers cannot create shared claims (`exclusive=false`) for claim commands
- localhost force override safety:
  - `force_override=true` requires `force_override_reason`
  - localhost non-emergency overrides require:
    - `force_override_confirmation="CONFIRM_LOCALHOST_FORCE_OVERRIDE"`
  - localhost emergency overrides (`force_override_emergency=true`) skip confirmation only for:
    - `stale_owner_unreachable`
    - `owner_malicious`
    - `security_incident`
  - emergency and force-override events are audit-tagged with high severity

### 2.1 Required Claim/Auth Fields (Daemon Command Path)

Required auth material when `require_auth=true`:
- `session_token` (or `auth_token`) in payload
- optional `_ssh_session_binding` when session was SSH-bound

Claim/ownership payload fields:
- `exclusive` (`bool`): request exclusive claim mode
- `force_override` (`bool`, optional): request override against active owner
- `force_override_reason` (`string`, required when `force_override=true`)
- `force_override_emergency` (`bool`, optional):
  - when `true`, localhost confirmation is not required
  - valid only with emergency reason set (`stale_owner_unreachable|owner_malicious|security_incident`)
- `force_override_confirmation` (`string`, required on localhost non-emergency force override):
  - exact token: `CONFIRM_LOCALHOST_FORCE_OVERRIDE`

Daemon-injected identity/context fields (not caller-controlled in daemon mode):
- `backend_id` -> normalized actor id (`key:<key_id>`)
- `_claim_actor_id`
- `_daemon_peer_host`

### 2.2 Sensitive Command Enforcement Matrix

`R=requires claim membership when claim exists`, `X=requires exclusive/shared rules`, `N=non-localhost shared denied`, `F=localhost force-confirm required`.

| Command | Resource scope | Enforcement |
|---|---|---|
| `claim-engine` | engine | `X`, `N`, `F` |
| `claim-endpoint` | endpoint | `X`, `N`, `F` |
| `claim-resource` | resource | `X`, `N`, `F` |
| `issue-token` | engine | `R`, exclusive/shared conflict denial |
| `issue-resource-token` | resource/engine | `R`, exclusive/shared conflict denial |
| `spawn` | engine | `R` (when existing claim present) |
| `get-registration` | engine | `R` (when existing claim present) |
| `shutdown` | engine | `R` (when existing claim present) |
| `ensure-running` | engine | `R` (when existing claim present) |
| `remove-registration` | engine | `R` (when existing claim present) |
| `logs-tail` | engine | `R` (when existing claim present) |
| `logs-follow` | engine | `R` (when existing claim present) |
| `inspect-capabilities` | engine | `R` (when existing claim present) |

### 2.3 Takeover And Override Rules

- active owner: owner keepalive within `owner_ttl_seconds`
- orphan owner: owner keepalive expired
- takeover transitions returned in claim results:
  - `joined_shared`
  - `refreshed`
  - `orphan_takeover`
  - `force_override`
- deny conditions:
  - active conflicting owner + no force override
  - non-localhost shared claim attempt
  - force override missing/invalid reason
  - localhost non-emergency force override without confirmation token
  - displaced owner non-claim operation before reclaim (`ownership_changed_reclaim_required`)

### 2.4 Daemon Denial Contract (Stable Error Taxonomy)

Daemon denial response shape:

```json
{
  "seq": 1,
  "ok": false,
  "error": "access_denied|auth_failed",
  "error_code": "stable_machine_code",
  "error_details": { "optional": "details" },
  "result": { "status": "denied", "...": "optional command result payload" }
}
```

Common `error_code` values:
- `session_token_required`
- `missing_or_invalid_session_token`
- `session_revoked`
- `insufficient_scope`
- `engine_access_denied`
- `config_access_denied`
- `non_localhost_shared_claim_denied`
- `localhost_force_override_confirmation_required`
- `force_override_reason_required`
- `force_override_emergency_reason_invalid`
- `ownership_changed_reclaim_required`
- `engine_shared_claim_not_member`
- `engine_exclusive_conflict`
- `resource_shared_claim_not_member`
- `resource_exclusive_conflict`
- `endpoint_exclusive_conflict`
- `exclusive_owner_conflict`

### 2.5 Claim Audit Event Schema

Claim ACL audit events are written to control state (`claim_audit_events`) with schema version `1`.

Event fields:
- `schema_version`
- `event_id`
- `timestamp`
- `event_type` (`claim_grant`, `claim_deny`)
- `command`
- `scope` (`engine|endpoint|resource`)
- `resource_kind`, `resource_id`, `resource_key`
- `actor_id`
- `peer_host`
- `decision` (`grant|deny`)
- `code`
- `transition`
- `mode` (`shared|exclusive`)
- `severity` (`normal|high`)
- `owners_before`
- `owners_after`
- `details`

### 2.6 Compatibility Note (For mp13-docs)

Minimum required daemon behavior: **Hosting ACL Contract v2**.

Consumers should require daemon support for:
- daemon-derived claim actor identity (`key:<key_id>`)
- structured denial response (`error_code`, `error_details`)
- owner keepalive/orphan takeover transitions
- localhost force-override confirmation token
- non-localhost shared-claim denial

If these fields/codes are missing, treat daemon as pre-v2 and disable claim-sensitive UX automation.

## 3. Diagnostics

`host-metrics` provides process-runtime diagnostics:
- current in-flight proxy requests:
  - `proxy.inflight_total`
  - `proxy.inflight_by_engine`
  - `proxy.inflight_peak`
- proxy counters:
  - `total`, `ok`, `http_error`, `failed`
  - request/response byte totals
- auth denial counters and last denial reason
- challenge auth telemetry:
  - `auth.challenge_begin_total`
  - `auth.challenge_complete_ok`
  - `auth.challenge_complete_failed`
  - `auth.challenge_replay_suspected`
  - `auth.challenge_recent_events`
- recent request ring buffer:
  - `proxy.recent_requests` (default max `100`)
  - each entry includes engine/method/path/status/outcome/duration/bytes/truncation/error

Important:
- Metrics are process-scoped. For stable metrics use daemon mode, not one-shot CLI mode.
- In multi-endpoint GUIs, fetch metrics through each endpoint's own host channel.
  There is no built-in cross-endpoint aggregation endpoint in this repo.

## 4. Core Commands

Auth and sessions:
- `auth-status`
- `auth-list-keys`
- `auth-list-sessions`
- `auth-list-issued-tokens`
- `auth-audit-list`
- `auth-upsert-key`
- `auth-revoke-key`
- `auth-issue-session`
- `auth-begin-challenge`
- `auth-complete-challenge`
- `auth-revoke-session`

Optional SSH-bound session issuance payload fields:
- `ssh_binding.target`
- `ssh_binding.key_fingerprint`

When a session is SSH-bound, subsequent commands must include `_ssh_session_binding`
with matching values.

Control config:
- `get-control-config`
- `set-control-config`
- `get-endpoint-mode-effective` (daemon runtime view)
- `set-endpoint-mode-override` (daemon runtime override, until daemon shutdown)
- `get-lifecycle-policy-effective` (effective lifecycle/disconnect-survival policy)

`set-control-config` lifecycle fields:
- `lifecycle_profile`:
  - `foreground_terminal_bound`
  - `detached_user_process`
  - `service_managed`
- `lifecycle_policy`:
  - `on_terminal_disconnect`: `stop_daemon|keep_daemon_running`
  - `terminal_control_enabled`: `bool`
  - `owner_disconnect_shutdown`: `bool`

Lifecycle enforcement notes:
- if `owner_disconnect_shutdown=true` and endpoint is exclusively owned, daemon may stop when owner disconnects.
- foreground profile honors `on_terminal_disconnect` policy; keep-running mode ignores SIGHUP where supported.
- daemon stop path now runs shutdown-order checkpoints to stop managed engines and release registrations.
- daemon stop sequence now drains in-flight async operations before managed worker shutdown checkpoints.
- when `terminal_control_enabled=false`, terminal control paths are denied (`__shutdown__`, runtime endpoint-mode override).

Config lifecycle and connect:
- `list-configs`
- `create-config`
- `models-from-config`
- `connect-from-config`

Worker traffic bridge:
- `proxy-request`
- `proxy-rpc-call`
- `proxy-rpc-open`
- `proxy-rpc-send`
- `proxy-rpc-recv`
- `proxy-rpc-close`
- `proxy-stream-open`
- `proxy-stream-send`
- `proxy-stream-recv`
- `proxy-stream-close`

Diagnostics:
- `host-metrics`

## 5. Workflow Examples

## 5.0 Hosting Setup/Reconfigure Wizard

Use the dedicated setup tool before first daemon start (or for reconfiguration):

```powershell
$env:PYTHONPATH='src'
python -m hosting.engine_host_cli --hosting-config --interactive
```

Non-interactive example:

```powershell
$env:PYTHONPATH='src'
python -m hosting.hosting_config --mode local_only --endpoint-mode exclusive --lifecycle-profile detached_user_process --key-source import --admin-key-id admin-main --admin-public-key-file "$HOME\\.ssh\\id_ed25519.pub" --require-auth
```

Diagnostics example:

```powershell
$env:PYTHONPATH='src'
python -m hosting.hosting_config --doctor
```

Generated artifacts:
- `<default_engine_config_dir>/Hosting/access_control.json`
- `<default_engine_config_dir>/Hosting/keyring/keys.json`
- `<default_engine_config_dir>/Hosting/state/client_key_map.json`
- `<default_engine_config_dir>/Hosting/state/bootstrap_state.json`

## 5.1 Start daemon

```powershell
$env:PYTHONPATH='src'
python -m hosting.engine_host_cli --daemon --background
```

Optional diagnostics log file for detached mode:

```powershell
python -m hosting.engine_host_cli --daemon --background --log-file "$HOME/.mp13-llm/backend/host_daemon.log"
```

Start dedicated HTTP ingress daemon (for worker API proxy ingress):

```powershell
$env:PYTHONPATH='src'
python -m hosting.engine_host_cli --daemon-http --background
```

Optional diagnostics log file for detached ingress mode:

```powershell
python -m hosting.engine_host_cli --daemon-http --background --log-file "$HOME/.mp13-llm/backend/host_daemon_http.log"
```

Ingress endpoints:
- `GET /health`
- `POST /__shutdown__`
- `* /proxy/<engine_id>/<path...>`
- `* /api/engine-host/proxy/<engine_id>/<path...>`

HTTP ingress is HTTP-only for host API proxy routes.

## 5.2 Bootstrap auth (first admin key)

Check current auth state first (recommended for bootstrap automation):

```powershell
python -m hosting.engine_host_cli auth-status
```

`auth-status` returns:

```json
{
  "daemon_version": "2.1.0",
  "capabilities": {
    "claim_acl_v2": true,
    "structured_denials_v1": true,
    "force_override_confirmation_v1": true
  },
  "require_auth": false,
  "config_store_mode": "store_only",
  "keys_count": 0,
  "sessions_count": 0,
  "roles": []
}
```

Version pinning guidance:
- pin to a SemVer `daemon_version` (for example `2.1.0`)
- prefer capability checks first (`capabilities.*`) and use version as fallback gate

Bootstrap only when `keys_count == 0`.

Public-key bootstrap (asymmetric challenge-response):

```powershell
@'{
  "key_id":"admin-pub",
  "auth_method":"public_key",
  "public_key":"ssh-ed25519 AAAA... user@host",
  "role":"admin"
}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key

@'{"key_id":"admin-pub","scope":"control"}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-begin-challenge

# sign returned `challenge` with your private key externally, then:
@'{"challenge_id":"<id>","signature_ssh":"-----BEGIN SSH SIGNATURE-----..."}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-complete-challenge
```

Transport binding assurance:
- If `ssh_binding` is supplied at `auth-begin-challenge`, completion requires matching
  `_ssh_session_binding` (same target/fingerprint) and the signed challenge payload
  includes those binding claims.

```powershell
@'{"key_id":"admin-key","key_secret":"CHANGE_ME","role":"admin"}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-upsert-key

@'{"require_auth":true}'@ |
python -m hosting.engine_host_cli --payload-stdin set-control-config
```

Issue control session:

```powershell
@'{"key_id":"admin-key","key_secret":"CHANGE_ME","scope":"control","ttl_seconds":900}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-issue-session
```

Use returned token:

```powershell
@'{"session_token":"<control_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin discover-running
```

## 5.3 Create traffic-scoped key and session

```powershell
@'{"key_id":"traffic-key","key_secret":"CHANGE_ME","role":"model_user","allowed_engines":["worker1"]}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-upsert-key

@'{"key_id":"traffic-key","key_secret":"CHANGE_ME","scope":"traffic","ttl_seconds":600,"engine_ids":["worker1"]}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-issue-session
```

## 5.4 Proxy worker request through hosting

```powershell
@'{"engine_id":"worker1","method":"GET","path":"/health","session_token":"<traffic_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin proxy-request
```

## 5.4.1 Connect engine from config (host-controlled spawn)

`connect-from-config` launches engine instances with host-owned deterministic behavior.
It does not use client-provided `spawn_command`/`worker_profile` fields.

- transport: cross-platform local IPC (fixed)
  - Linux: Unix domain socket (`AF_UNIX`)
  - Windows: named pipe (`AF_PIPE`)
- IPC worker command: `python -m hosting.engine_worker_ipc --ipc-family <AF_UNIX|AF_PIPE> --ipc-address <path_or_pipe>`
- endpoint registration: `ipc://local`
- model: from payload `model_path` or config model fields
- host->worker anti-spoofing token:
  - host generates per-engine token
  - token is passed to worker env: `MP13_ENGINE_HOST_TOKEN`
- IPC channel auth uses this token as connection authkey

Generic worker profile mode:
- classify config as generic via `worker_kind`/`worker_type = generic`
- spawn command from config:
  - `worker_command` (preferred), or
  - `spawn.command`
- generic profile does not require model selection (`model_path` optional/ignored)
- if command is missing, connect fails with reason `generic_worker_command_missing`
- runtime policy: engines registered as generic (`worker_profile_class=generic`) deny model-role proxy/rpc traffic (`insufficient_role`)

IPC RPC mode (default transport):
- sync RPC: `proxy-rpc-call`
- async RPC lifecycle: `proxy-rpc-open` / `proxy-rpc-recv` / `proxy-rpc-send` / `proxy-rpc-close`
- `request_id` is required for async RPC and for cancel control messages
- multiple concurrent `stream_id` sessions are supported per worker
- `proxy-stream-*` remains as compatibility aliases for engine inference flows

```powershell
# optional: pin the Python environment used for engine launch
$env:MP13_ENGINE_PYTHON='C:\path\to\python.exe'

@'{"config_path":"default","engine_id":"worker_cfg","model_path":"C:\\models\\granite-3.3-2b-instruct"}'@ |
python -m hosting.engine_host_cli --payload-stdin connect-from-config
```

Generic RPC command-level flow (IPC):

```powershell
# sync call
@'{
  "engine_id":"worker1",
  "method":"rpc.describe",
  "params":{},
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-call

# async open
@'{
  "engine_id":"worker1",
  "method":"run-inference",
  "params":{"messages_list":[[{"role":"user","content":"hello"}]],"stream":true},
  "request_id":"req-1",
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-open

# recv loop
@'{
  "engine_id":"worker1",
  "stream_id":"<stream_id>",
  "timeout_seconds":2.0,
  "max_items":64,
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-recv

# optional cancel
@'{
  "engine_id":"worker1",
  "stream_id":"<stream_id>",
  "message":{"action":"cancel","request_id":"req-1"},
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-send

# close
@'{
  "engine_id":"worker1",
  "stream_id":"<stream_id>",
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-close
```

IPC stream command-level flow (default transport):

```powershell
# open inference stream
@'{
  "engine_id":"worker1",
  "tool":"run-inference",
  "arguments":{
    "messages_list":[[{"role":"user","content":"hello"}]],
    "stream":true
  },
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-stream-open

# recv events (repeat until done=true)
@'{
  "engine_id":"worker1",
  "stream_id":"<stream_id>",
  "timeout_seconds":2.0,
  "max_items":64,
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-stream-recv

# optional cancel
@'{
  "engine_id":"worker1",
  "stream_id":"<stream_id>",
  "message":{"action":"cancel","request_id":"<request_id>"},
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-stream-send

# close
@'{
  "engine_id":"worker1",
  "stream_id":"<stream_id>",
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-stream-close
```

## 5.5 Set restrictive traffic policy

```powershell
@'{
  "traffic_policy":{
    "allowed_methods":["GET","POST"],
    "allowed_path_prefixes":["/health","/inference"],
    "request_header_allowlist":["content-type","x-request-id"],
    "response_header_allowlist":["content-type","content-length","x-request-id"],
    "allow_authorization_header":false,
    "max_request_bytes":262144,
    "max_response_bytes":524288
  },
  "session_token":"<control_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin set-control-config
```

Per-engine traffic policy override (HTTP ingress and proxy-request path evaluation):

```powershell
@'{
  "engine_traffic_policies":{
    "worker2":{
      "allowed_methods":["GET"],
      "allowed_path_prefixes":["/other"]
    }
  },
  "session_token":"<control_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin set-control-config
```

No websocket session policy is exposed. Worker transport is IPC-only.

## 5.6 Read diagnostics

```powershell
@'{"session_token":"<control_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin host-metrics
```

## 5.7 Host auth management via `mp13config`

`src/app/config.py` now exposes host-auth management helpers (separate from normal engine config editing):

```powershell
# generate secret
mp13config --host-auth-generate-secret 32

# upsert key using env var to avoid shell history leakage
$env:MP13_HOST_SECRET="CHANGE_ME"
mp13config --host-auth-upsert-key admin-key --host-auth-role admin --host-auth-secret-env MP13_HOST_SECRET

# issue session
mp13config --host-auth-issue-session admin-key --host-auth-scope control --host-auth-secret-env MP13_HOST_SECRET

# status
mp13config --host-auth-status
```

Prefer `--host-auth-secret-env` or `--host-auth-secret-stdin` over `--host-auth-secret`.

## 5.8 Audit sessions and issued tokens

```powershell
@'{"session_token":"<control_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-list-sessions

@'{"session_token":"<control_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-list-issued-tokens
```

Both endpoints return metadata with redacted `token_preview` values (not full tokens).

Auth audit event query example (admin-only):

```powershell
@'{"session_token":"<control_token>","event_type":"auth_revoke_key","limit":50,"offset":0}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-audit-list
```

Filtering and pagination examples:

```powershell
# sessions: traffic-only, first page
@'{"session_token":"<control_token>","scope":"traffic","limit":50,"offset":0}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-list-sessions

# issued tokens: worker1 only, page 2
@'{"session_token":"<control_token>","engine_id":"worker1","limit":50,"offset":50}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-list-issued-tokens
```

## 6. Remote Access Patterns

Two supported SSH patterns:

1. SSH relay (GUI default):
   - `SSHRelayConnection` starts `python -m hosting.engine_host_cli --relay` on the remote host.
   - The relay process connects to `127.0.0.1:<daemon_port>` on the remote host.
   - Benefits: no persistent tunnel process, no extra exposed listener, simple on-demand flow.

2. SSH tunnel (advanced option):
   - Example: `ssh -L 19876:127.0.0.1:19876 user@host`
   - Then use local-style daemon connection (`LocalSocketConnection`) to forwarded `127.0.0.1:19876`.
   - Benefits: persistent channel and lower per-operation latency.

If daemon is directly exposed on public network:
- enforce `require_auth=true`
- use short session TTL
- rotate secrets
- restrict source IPs/firewall where possible

## 7. Current Limitations

- Worker transport is local IPC only (no host-managed remote worker transport).
- `proxy-request` is a compatibility bridge for HTTP-like engine routes over IPC.
- Generic worker integrations should prefer `proxy-rpc-*` for sync/async RPC.
- Metrics are per-process runtime (not persisted across daemon restarts).
- Host channel credential bootstrap requires wiring `engine_host_key_id` + `engine_host_key_secret`
  (or a pre-issued `engine_host_session_token`) in control settings/profile construction.

## 8. Consumer Contract (GUI/Backend in Other Projects)

When another project consumes hosting APIs/channels, use this contract:

1. Daemon status:
   - `get_daemon_status()` should be treated as both process and auth readiness.
   - Returned shape includes:
     - daemon fields: `pid_file`, `pid`, `port`, `started_at`, `alive`
     - auth fields: `auth_status` (same shape as `auth-status`), `auth_status_error`

2. Auth bootstrap check:
   - Always call `auth-status` first.
   - First-time bootstrap is `keys_count == 0`.
   - Avoid blind `auth-upsert-key` in automated startup.

3. Endpoint-scoped metrics:
   - Metrics are daemon-process scoped, not global across endpoints.
   - Consumer APIs should route metrics through the selected endpoint channel
     (for example, backend endpoint `GET /api/engine-host/metrics?endpoint_id=<id>`).
   - Do not always route metrics to local supervisor if remote endpoints are present.

4. Host profile/session wiring:
   - Include one of:
     - `engine_host_session_token` (pre-issued), or
     - `engine_host_key_id` + `engine_host_key_secret` (issue session on demand).
   - Optional tuning fields:
     - `engine_host_session_scope` (default `control`)
     - `engine_host_session_ttl_seconds` (default `900`)
