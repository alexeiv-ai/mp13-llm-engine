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
- `auth-upsert-key`
- `auth-revoke-key`
- `auth-issue-session`
- `auth-revoke-session`

Control config:
- `get-control-config`
- `set-control-config`

Worker traffic bridge:
- `proxy-request`

Diagnostics:
- `host-metrics`

## 5. Workflow Examples

## 5.1 Start daemon

```powershell
$env:PYTHONPATH='src'
python -m hosting.engine_host_cli --daemon --background
```

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
- Native websocket pass-through is not yet implemented.
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
