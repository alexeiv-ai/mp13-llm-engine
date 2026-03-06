# MP13 Hosting

This file documents the current hosting implementation in `src/hosting` and practical usage workflows.

## 1. Architecture

Hosting is a control-plane plus guarded traffic bridge.

- Control-plane:
  - worker lifecycle (`spawn`, `shutdown`, `ensure-running`, `discover-running`)
  - config-driven worker startup
  - claims/tokens/resource ownership state
- Traffic bridge:
  - `proxy-request` forwards HTTP(S) requests to registered worker endpoint
  - traffic authorization and policy are enforced before forwarding

Workers are still separate processes. Hosting does not expose worker private keys and does not require worker ports to be publicly forwarded.

## 2. Security Model

Auth roles:
- `management`: full control scope
- `config`: config scope only
- `traffic`: traffic proxy scope only

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
  - `force_override=true` requires `force_override_confirmation="CONFIRM_LOCALHOST_FORCE_OVERRIDE"`
  - applies to localhost claim overrides in daemon command path

### 2.1 Required Claim/Auth Fields (Daemon Command Path)

Required auth material when `require_auth=true`:
- `session_token` (or `auth_token`) in payload
- optional `_ssh_session_binding` when session was SSH-bound

Claim/ownership payload fields:
- `exclusive` (`bool`): request exclusive claim mode
- `force_override` (`bool`, optional): request override against active owner
- `force_override_confirmation` (`string`, required on localhost force override):
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
  - localhost force override without confirmation token

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

Worker traffic bridge:
- `proxy-request`
- `proxy-ws-open`
- `proxy-ws-send`
- `proxy-ws-recv`
- `proxy-ws-close`

Diagnostics:
- `host-metrics`

## 5. Workflow Examples

## 5.1 Start daemon

```powershell
$env:PYTHONPATH='src'
python -m hosting.engine_host_cli --daemon --background
```

Start dedicated HTTP ingress daemon (for worker API proxy ingress):

```powershell
$env:PYTHONPATH='src'
python -m hosting.engine_host_cli --daemon-http --background
```

Ingress endpoints:
- `GET /health`
- `POST /__shutdown__`
- `* /proxy/<engine_id>/<path...>`
- `* /api/engine-host/proxy/<engine_id>/<path...>`

Websocket ingress:
- Use websocket upgrade on the same proxy routes.
- Auth/session uses the same traffic token model (`Authorization: Bearer <token>` or `X-Session-Token`).
- Engine allowlist and traffic path policy are enforced before tunnel creation.

## 5.2 Bootstrap auth (first management key)

Check current auth state first (recommended for bootstrap automation):

```powershell
python -m hosting.engine_host_cli auth-status
```

`auth-status` returns:

```json
{
  "require_auth": false,
  "config_store_mode": "store_only",
  "keys_count": 0,
  "sessions_count": 0,
  "roles": []
}
```

Bootstrap only when `keys_count == 0`.

Public-key bootstrap (asymmetric challenge-response):

```powershell
@'{
  "key_id":"admin-pub",
  "auth_method":"public_key",
  "public_key":"ssh-ed25519 AAAA... user@host",
  "role":"management"
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
@'{"key_id":"admin-key","key_secret":"CHANGE_ME","role":"management"}'@ |
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

## 5.3 Create traffic key and scoped session

```powershell
@'{"key_id":"traffic-key","key_secret":"CHANGE_ME","role":"traffic","allowed_engines":["worker1"]}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-upsert-key

@'{"key_id":"traffic-key","key_secret":"CHANGE_ME","scope":"traffic","ttl_seconds":600,"engine_ids":["worker1"]}'@ |
python -m hosting.engine_host_cli --payload-stdin auth-issue-session
```

## 5.4 Proxy worker request through hosting

```powershell
@'{"engine_id":"worker1","method":"GET","path":"/health","session_token":"<traffic_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin proxy-request
```

Websocket command-level flow:

```powershell
# open
@'{"engine_id":"worker1","path":"/ws","session_token":"<traffic_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin proxy-ws-open

# send text
@'{"ws_id":"<ws_id>","text":"hello","session_token":"<traffic_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin proxy-ws-send

# receive
@'{"ws_id":"<ws_id>","timeout_seconds":2.0,"session_token":"<traffic_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin proxy-ws-recv

# close
@'{"ws_id":"<ws_id>","session_token":"<traffic_token>"}'@ |
python -m hosting.engine_host_cli --payload-stdin proxy-ws-close
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

Websocket session policy (command-level `proxy-ws-*` lifecycle):

```powershell
@'{
  "websocket_session_policy":{
    "max_sessions":128,
    "idle_timeout_seconds":300,
    "max_lifetime_seconds":3600
  },
  "session_token":"<control_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin set-control-config
```

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
mp13config --host-auth-upsert-key admin-key --host-auth-role management --host-auth-secret-env MP13_HOST_SECRET

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

- `proxy-request` currently supports HTTP(S) only.
- Native websocket pass-through is currently available in HTTP ingress mode only (not in `proxy-request` command path).
- Command-level websocket pass-through uses session lifecycle commands (`proxy-ws-open/send/recv/close`), not a single streaming `proxy-request` response.
- Command-level websocket sessions are in-memory process state and subject to configured GC policy (`websocket_session_policy`).
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
