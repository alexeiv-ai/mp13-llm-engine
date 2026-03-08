# Hosting Pytest Status (IPC/RPC Migration)

Date: 2026-03-08

This file lists pytest commands relevant to the IPC-only + RPC lifecycle migration.
Run these outside sandboxed environments when possible.

## 1) Prerequisites

- Run from repo root.
- Ensure `PYTHONPATH=src`.
- Use a writable temp base (`--basetemp`) outside restricted dirs.

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
$BASE = "C:\\temp\\mp13-pytest"
New-Item -ItemType Directory -Force $BASE | Out-Null
```

Linux/macOS bash:

```bash
export PYTHONPATH=src
BASE=/tmp/mp13-pytest
mkdir -p "$BASE"
```

## 2) Focused ACL Regression (access denied)

```powershell
pytest tests/test_hosting_daemon_acl.py -q --basetemp "$BASE\\acl" -p no:cacheprovider
```

```bash
pytest tests/test_hosting_daemon_acl.py -q --basetemp "$BASE/acl" -p no:cacheprovider
```

Expected high-level behavior:
- Denials return stable codes like:
  - `session_token_required`
  - `engine_shared_claim_not_member`
  - `exclusive_owner_conflict`
  - `localhost_force_override_confirmation_required`
  - `non_localhost_shared_claim_denied`

## 3) Channel/Auth Path

```powershell
pytest tests/test_engine_host_channel.py -q --basetemp "$BASE\\channel" -p no:cacheprovider
```

```bash
pytest tests/test_engine_host_channel.py -q --basetemp "$BASE/channel" -p no:cacheprovider
```

## 4) HTTP Ingress (legacy WS test excluded)

The websocket passthrough test is legacy for this round and should be excluded.

```powershell
pytest tests/test_hosting_http_ingress.py -q --basetemp "$BASE\\ingress" -p no:cacheprovider -k "not websocket"
```

```bash
pytest tests/test_hosting_http_ingress.py -q --basetemp "$BASE/ingress" -p no:cacheprovider -k "not websocket"
```

## 5) Security Suite (legacy WS lifecycle tests excluded)

```powershell
pytest tests/test_hosting_service_security.py -q --basetemp "$BASE\\security" -p no:cacheprovider -k "not proxy_ws and not websocket"
```

```bash
pytest tests/test_hosting_service_security.py -q --basetemp "$BASE/security" -p no:cacheprovider -k "not proxy_ws and not websocket"
```

## 6) Combined Relevant Run

```powershell
pytest tests/test_hosting_daemon_acl.py tests/test_engine_host_channel.py tests/test_hosting_http_ingress.py tests/test_hosting_service_security.py -q --basetemp "$BASE\\all" -p no:cacheprovider -k "not proxy_ws and not websocket"
```

```bash
pytest tests/test_hosting_daemon_acl.py tests/test_engine_host_channel.py tests/test_hosting_http_ingress.py tests/test_hosting_service_security.py -q --basetemp "$BASE/all" -p no:cacheprovider -k "not proxy_ws and not websocket"
```

## 7) If Temp/Cache Permissions Still Fail

- Keep `-p no:cacheprovider`.
- Move `--basetemp` to another writable location.
- Avoid repository directories mounted with restrictive ACLs.
