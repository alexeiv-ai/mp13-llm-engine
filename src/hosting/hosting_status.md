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

## Not Implemented Yet

1. Asymmetric key challenge-response (current keys are shared-secret based).
2. Session binding to SSH identity/fingerprint.
3. Native websocket streaming pass-through in `proxy-request`.
4. Fine-grained per-engine traffic policy in daemon HTTP ingress mode.
5. Token introspection/audit endpoints beyond current status commands.

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

Implement a dedicated daemon HTTP ingress mode that proxies worker API calls and enforces engine-level traffic auth on the same hosted session model.
