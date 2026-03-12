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

## 7) Windows Detached Daemon RCA Notes (2026-03-12)

Scope investigated:
- Reported behavior: daemon appears to exit after readiness poll does bare TCP connect+close.
- Environment focus: Windows + Python 3.12 + detached process flags.

What was tested:
- Isolated detached `asyncio.start_server` reproduction with `ProactorEventLoop`, client bare connect+close, handler path `readline() -> empty -> writer.close() -> await writer.wait_closed()`.
- Real daemon process launched detached via `python -m hosting.engine_host_cli --daemon ...`, then bare connect+close probe.
- Repeated runs checking post-probe liveness and protocol responsiveness.
- PID comparison between `Popen.pid` and PID written by daemon (`os.getpid()` in pid file).

Observed results on this host:
- Isolated reproduction: no loop/process termination after bare connect+close.
- Real daemon repeated runs: no daemon self-exit after bare connect+close; daemon remained pingable.
- `Popen.pid` vs pid-file PID: matched in all sampled runs.

RCA conclusion:
- Primary confirmed root cause is liveness misclassification in `DaemonPidFile._pid_alive`:
  - `os.kill(pid, 0)` can raise `SystemError` on Windows detached paths.
  - Previous code treated generic exceptions as dead, causing false "daemon not alive" status.
- Readiness probe was hardened from bare TCP connect/close to protocol `__ping__` to avoid fragile teardown-only probes and align readiness with actual daemon protocol handling.

Current status:
- No reproducible evidence (on this machine) of intrinsic daemon self-exit caused by `writer.wait_closed()` after empty client read.
- If this is still observed elsewhere, it is likely environment-specific (Python build/OS patch level/security tooling) and should be captured with per-process logs and faulthandler output from that host.

## 8) Added Targeted Regression Tests

```bash
pytest tests/test_hosting_daemon_pidfile.py tests/test_engine_host_channel.py -q
```

Includes checks for:
- `_pid_alive` handling of `SystemError` / `ProcessLookupError` / `PermissionError`.
- `start_daemon_background()` readiness using protocol ping (`__ping__`) rather than bare socket connect/close.
