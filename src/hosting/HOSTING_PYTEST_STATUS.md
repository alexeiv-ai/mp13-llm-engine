# Hosting Pytest Status (IPC/RPC Migration)

Date: 2026-03-09

This file lists pytest commands relevant to the IPC-only + RPC lifecycle migration.

## 1) Environment

Run from repo root.

Use one of these setups:

- Preferred: install package in editable mode, then run pytest without extra env vars.
- Alternative: if you are not installing the package, set `PYTHONPATH=src` so imports like `from hosting...` resolve.

Windows PowerShell (alternative mode):

```powershell
$env:PYTHONPATH = "src"
```

Linux/macOS bash (alternative mode):

```bash
export PYTHONPATH=src
```

No other environment variables are required for these tests.

## 2) Focused ACL Regression (access denied)

```bash
pytest tests/test_hosting_daemon_acl.py -q
```

Expected denial codes include:
- `session_token_required`
- `engine_shared_claim_not_member`
- `exclusive_owner_conflict`
- `localhost_force_override_confirmation_required`
- `non_localhost_shared_claim_denied`

## 3) Channel/Auth Path


```bash
pytest tests/test_engine_host_channel.py -q
```

## 4) HTTP Ingress


```bash
pytest tests/test_hosting_http_ingress.py -q
```

## 5) Security Suite


```bash
pytest tests/test_hosting_service_security.py -q
```

## 6) Combined Relevant Run


```bash
pytest tests/test_hosting_daemon_acl.py tests/test_engine_host_channel.py tests/test_hosting_http_ingress.py tests/test_hosting_service_security.py -q
```

