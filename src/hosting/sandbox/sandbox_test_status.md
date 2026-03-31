# Sandbox Test Status

Date: 2026-03-29
Scope: `src/hosting/sandbox` Windows-first worker sandbox feature

## 1. Environment

Run from repo root.

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

## 2. Primary Commands

### 2.1 Core sandbox policy + live Windows sandbox slice

```powershell
python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_worker_sandbox_windows_live.py -q
```

Latest result in this sandbox:

1. `11 passed, 1 skipped`

What this covers:

1. sandbox policy normalization
2. spawn persistence of sandbox metadata
3. plain launcher `close_fds` behavior
4. live Windows Low-IL denial of write to a medium-integrity file
5. live Windows named-pipe RPC continuity for a sandboxed minimal helper worker
6. brokered filesystem root-scoped read/write/list behavior
7. traversal denial for brokered filesystem roots
8. worker-side brokered filesystem client payload construction
9. brokered HTTP allowlist enforcement and response shaping
10. worker-side brokered HTTP client payload construction
11. optional full `hosting.engine_worker_ipc` Low-IL test is skipped unless explicitly enabled and the unsandboxed preflight also works

### 2.2 Focused regression slice including existing hosting tests

```powershell
python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_worker_sandbox_windows_live.py tests/test_hosting_daemon_pidfile.py tests/test_hosting_service_security.py tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py tests/test_hosting_daemon_startup.py -q
```

Latest result in this sandbox:

1. `87 passed, 1 skipped`

## 3. Environment-Sensitive / Manual Reruns

### 3.1 Full `engine_worker_ipc` under Low Integrity

This test is intentionally gated because it requires a real engine startup input, not just sandbox support.

Important:

1. `hosting.engine_worker_ipc` always calls `_init_engine()` on startup.
2. That means the test needs either:
   - `MP13_SANDBOX_ENGINE_MODEL_PATH`, or
   - `MP13_SANDBOX_ENGINE_CONFIG_PATH`
3. Even with those inputs, the test first checks whether the same real worker can start unsandboxed in the current environment.
4. If the unsandboxed preflight cannot start the worker, the test skips rather than reporting a sandbox failure.

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:MP13_RUN_HOSTING_SANDBOX_SENSITIVE = "1"
$env:MP13_SANDBOX_ENGINE_MODEL_PATH = "C:\path\to\model"
# or: $env:MP13_SANDBOX_ENGINE_CONFIG_PATH = "C:\path\to\engine_config.json"
python -m pytest tests/test_hosting_worker_sandbox_windows_live.py -q -k engine_worker_ipc
```

Current status in this sandbox:

1. not run as an enforced pass criterion
2. test exists and is skipped by default unless:
   - `MP13_RUN_HOSTING_SANDBOX_SENSITIVE=1`
   - and a real model or config path is supplied

What it is intended to prove:

1. a sandboxed Low-IL worker can launch the real `hosting.engine_worker_ipc`
2. the worker still answers the existing `hello` RPC over `AF_PIPE`
3. the worker can be shut down through the same pipe transport
4. if the environment proves the worker can answer `hello` but process exit remains stuck after shutdown, the test now cleans it up instead of treating that as a sandbox regression
5. the sandboxed result is compared only after an unsandboxed preflight succeeds

## 4. Interpretation

Current enforced minimal implementation status:

1. supported and tested:
   - `inherit_parent_handles=false`
   - Low-IL write denial against medium-integrity file
   - named-pipe RPC continuity for sandboxed worker process
   - brokered HTTP allowlist enforcement
2. implemented but environment-sensitive:
   - full `hosting.engine_worker_ipc` validation under Low IL

## 5. Recommended Manual Follow-Up

If you want to validate the full worker path outside this sandbox, run:

```powershell
$env:PYTHONPATH = "src"
$env:MP13_RUN_HOSTING_SANDBOX_SENSITIVE = "1"
$env:MP13_SANDBOX_ENGINE_MODEL_PATH = "C:\path\to\model"
# or: $env:MP13_SANDBOX_ENGINE_CONFIG_PATH = "C:\path\to\engine_config.json"
python -m pytest tests/test_hosting_worker_sandbox_windows_live.py -q
```

If that test fails, report back:

1. failing test name
2. full traceback
3. whether the minimal helper named-pipe test still passes
