# Hosting Security Refactor Status

Date: 2026-03-03

## Goal

Harden hosting for remote-first operation with minimal external components:
- built-in auth for hosting commands
- separate authorization boundary for engine-config operations
- restrict protocol-visible configs to hosted config store

## Implemented

## 1) Service auth infrastructure

File: `src/hosting/engine_host_service.py`

Implemented:
- control config auth fields:
  - `require_auth`
  - `auth.keys`
  - `auth.sessions`
- key/session primitives:
  - `auth_status`
  - `auth_list_keys`
  - `auth_upsert_key`
  - `auth_revoke_key`
  - `auth_issue_session`
  - `auth_revoke_session`
- hashed secret storage (SHA-256)
- session TTL enforcement and expired session pruning
- command authorization policy:
  - `control` scope for management operations
  - `config` scope for config operations
  - `traffic` scope for worker traffic forwarding
  - traffic session can be restricted to specific engine IDs

## 2) Config-path hardening

File: `src/hosting/engine_host_service.py`

Implemented:
- restricted `config_path` selector policy:
  - allowed: `default`, or hosted config name
  - denied: absolute paths, relative traversal, path separators
- resolution constrained to hosted config store:
  - `<default_config_dir>/backend/configs/*.json`

## 3) Daemon enforcement and command surface

File: `src/hosting/engine_host_daemon.py`

Implemented:
- authorization check before service execution for RPC commands
- `auth_failed` error path on denied requests
- support for new auth command dispatch
- `set-control-config` now accepts `require_auth`
- added `proxy-request` command dispatch for data-plane forwarding

## 4) CLI enforcement and command surface

File: `src/hosting/engine_host_cli.py`

Implemented:
- auth policy enforced in direct-fallback execution path as well
- added CLI subcommands:
  - `auth-status`, `auth-list-keys`, `auth-upsert-key`, `auth-revoke-key`
  - `auth-issue-session`, `auth-revoke-session`, `proxy-request`, `host-metrics`
- updated examples for auth bootstrap/session issuance

## 5) Data-plane bridge command

File: `src/hosting/engine_host_service.py`

Implemented:
- `proxy_request(...)` forwards HTTP(S) request to a registered worker endpoint
- supports method/path/query/headers/body (base64)
- bounded response size with truncation flag
- returns status code + headers + body (base64)
- enforces traffic auth scope and engine allowlist on `engine_id`
- enforces traffic policy:
  - allowed HTTP methods
  - allowed path prefixes
  - request header allowlist
  - response header allowlist
  - request/response size caps
- runtime diagnostics metrics exposed via `host-metrics`:
  - current in-flight proxy requests (total + per engine)
  - proxy success/error/failure counters
  - auth denial counters and last reason
  - request/response byte counters
  - recent proxy request ring buffer (default 100)

This enables single-port remote traffic flow through hosting protocol.

## 6) Config-path helper extensions

File: `src/mp13_engine/mp13_config_paths.py`

Implemented:
- `get_hosting_config_store_dir()`
- `normalize_hosting_config_selector()`
- `resolve_hosting_config_path()`

These helpers align the hosting store-only config model with shared config-path utilities.

## 7) Dedicated daemon HTTP ingress mode

Files:
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`

Implemented:
- dedicated HTTP ingress daemon mode:
  - `python -m hosting.engine_host_cli --daemon-http`
  - `python -m hosting.engine_host_cli --daemon-http --background`
- ingress endpoints:
  - `GET /health`
  - `POST /__shutdown__` (token-guarded)
  - `* /proxy/<engine_id>/<path...>`
  - `* /api/engine-host/proxy/<engine_id>/<path...>`
- proxy auth/session enforcement uses the same hosted session model as `proxy-request`:
  - session token via `Authorization: Bearer <token>` or `X-Session-Token`
  - `EngineHostService.authorize_command("proxy-request", payload)` enforces traffic scope and engine allowlist
- ingress path forwards to `EngineHostService.proxy_request(...)` so traffic policy constraints remain centralized.
- websocket pass-through in ingress mode:
  - HTTP Upgrade on proxy routes is authenticated with the same traffic session model
  - engine allowlist and traffic path policy are enforced before tunnel creation
  - backend websocket handshake response and frame stream are tunneled bidirectionally

## 8) Token introspection/audit endpoints

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- new control-scope auth audit commands:
  - `auth-list-sessions`
  - `auth-list-issued-tokens`
- outputs use redacted `token_preview` values (no full token material).
- command surface wired through:
  - service methods
  - daemon dispatch
  - CLI subcommands
  - control channel helper methods

## 9) Per-engine traffic policy overrides

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- `set-control-config` now accepts `engine_traffic_policies` map (`engine_id -> traffic_policy`).
- proxy path enforcement resolves policy per engine:
  - global `traffic_policy` as base
  - engine-specific override merged and normalized per request
- applies to:
  - command-level `proxy-request`
  - HTTP ingress proxy routes
  - HTTP ingress websocket upgrade path checks

## 10) Session binding to SSH identity/fingerprint

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_channel.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`

Implemented:
- `auth-issue-session` supports optional `ssh_binding`:
  - `target`
  - `key_fingerprint`
- bound sessions require `_ssh_session_binding` in subsequent command payloads.
- binding mismatch rejects session usage (`ssh_binding_required` / `ssh_binding_mismatch`).
- SSH mode control channel auto-populates binding metadata:
  - auto-issued sessions include `ssh_binding`
  - subsequent commands include `_ssh_session_binding`.

## 11) Audit listing filtering/pagination

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- `auth-list-sessions` supports:
  - filters: `key_id`, `scope`, `role`, `token_preview_contains`
  - pagination: `limit`, `offset`
- `auth-list-issued-tokens` supports:
  - filters: `engine_id`, `resource_kind`, `resource_id`, `backend_id`, `token_preview_contains`
  - pagination: `limit`, `offset`
- responses now include pagination metadata:
  - `offset`, `limit`, `count`, `has_more`, `next_offset`

## 12) Command-level websocket pass-through

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- websocket lifecycle commands:
  - `proxy-ws-open`
  - `proxy-ws-send`
  - `proxy-ws-recv`
  - `proxy-ws-close`
- traffic-scope authorization and engine allowlist enforcement for ws commands.
- policy enforcement uses engine traffic policy resolution for websocket open path.
- channel wrappers added for external consumers.

## 13) Asymmetric key challenge-response authentication

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- key auth methods now support:
  - `shared_secret` (existing)
  - `public_key` (new)
- new challenge commands:
  - `auth-begin-challenge`
  - `auth-complete-challenge`
- public-key keys cannot use direct `auth-issue-session`; they must use challenge flow.
- status/key listing now expose challenge/key auth metadata (`challenges_count`, `auth_method`).

## 14) Websocket session GC hardening

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`
- `src/hosting/engine_host_channel.py`

Implemented:
- configurable websocket session policy in control config:
  - `websocket_session_policy.max_sessions`
  - `websocket_session_policy.idle_timeout_seconds`
  - `websocket_session_policy.max_lifetime_seconds`
- policy is applied to command-level websocket sessions (`proxy-ws-*`):
  - idle timeout GC
  - absolute lifetime GC
  - bounded active session count with oldest-session eviction
- policy is exposed in `get-control-config` / `set-control-config`.

## 15) Challenge auth telemetry hardening

Files:
- `src/hosting/engine_host_service.py`

Implemented:
- challenge lifecycle telemetry in host metrics auth block:
  - `challenge_begin_total`
  - `challenge_complete_ok`
  - `challenge_complete_failed`
  - `challenge_replay_suspected`
  - `challenge_recent_events` ring buffer
- replay-suspected tracking when challenge completion attempts reference missing/expired challenge IDs
  or invalid challenge signatures.

## 16) Challenge transport-binding assurance

Files:
- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`
- `src/hosting/engine_host_cli.py`

Implemented:
- challenge payload now embeds SSH binding claims when present:
  - `ssh_binding_target`
  - `ssh_binding_key_fingerprint`
- challenge completion enforces matching presented SSH binding when challenge was bound.

Security hole mitigated:
- Prevents cross-transport relay of captured signed challenges within TTL.
  Previously, an attacker who obtained a valid challenge signature might attempt completion
  from a different SSH transport context. With binding enforcement, completion must originate
  from the same bound SSH identity context (target/fingerprint), reducing replay/relay risk.

## Not Implemented Yet

None (for the currently tracked hosting_status scope).

## Operational Notes

1. Bootstrap:
   - create first management key
   - enable `require_auth=true`
   - issue short-lived sessions for client operations
2. Prefer SSH transport to reduce replay/timing exposure at network layer.
3. Rotate keys regularly and keep session TTL short.
4. For external GUI/backend consumers:
   - fetch endpoint metrics via the selected endpoint channel (not always local)
   - include host auth material in the endpoint/profile:
     `engine_host_session_token` or `engine_host_key_id` + `engine_host_key_secret`

## Suggested Next Step

Monitor production behavior and tune challenge/WS policy defaults based on operational telemetry.
