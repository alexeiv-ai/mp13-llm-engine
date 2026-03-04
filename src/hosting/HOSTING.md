# MP13 Hosting Guide

This document describes how `src/hosting` works, what must be prepared on the host machine, and how remote clients can control and use hosted MP13 engine worker processes.

Scope:
- Control-plane and process-hosting behavior in `src/hosting`
- Linux and Windows operational setup
- Local and remote control flows

Out of scope:
- How to use MP13 engine library APIs inside a worker process (assumed already solved)

## 1. What Hosting Is

`src/hosting` is a control plane for managed engine worker processes.

It provides:
- Worker process lifecycle: spawn, discover, ensure-running, shutdown
- Persisted registration/state: engine IDs, PIDs, endpoints, logs
- Claim and token state for access coordination
- Local daemon and CLI transport options
- SSH-based remote command transport

It does **not** provide:
- Inference data-plane proxying
- Reverse proxy/load balancer for worker HTTP endpoints

Practical model:
1. Host control plane starts/manages worker processes.
2. Each worker exposes its own endpoint (for example `http://127.0.0.1:9001`).
3. Clients use hosting APIs/CLI to discover registrations and coordinate ownership.
4. Clients talk to worker endpoints directly (or through infrastructure you add separately).

## 2. Components

- `engine_host_service.py`
  - Core file-backed logic (`EngineHostService`)
  - Owns managed engine and control state JSON files

- `engine_host_daemon.py`
  - Long-lived local TCP daemon (`127.0.0.1:19876` by default)
  - Routes JSON RPC commands to `EngineHostService`

- `engine_host_cli.py`
  - CLI entry point
  - Modes:
    - `--daemon` / `--daemon --background`
    - `--relay` (stdin/stdout bridge for SSH relay)
    - one-shot command mode

- `engine_host_connection.py`
  - Persistent connection strategies:
    - `LocalSocketConnection` (local daemon TCP)
    - `SSHRelayConnection` (persistent SSH command running `--relay`)

- `engine_host_channel.py`
  - Backend-facing wrapper (`EngineHostControlChannel`)
  - Tries persistent connection first
  - Falls back to per-command subprocess CLI path

## 3. Execution Paths

### Typical local path
`EngineHostControlChannel -> LocalSocketConnection -> EngineHostDaemon -> EngineHostService`

### Degraded path
`EngineHostControlChannel -> subprocess (engine_host_cli one-shot) -> EngineHostService`

### Remote SSH path (persistent)
`EngineHostControlChannel -> SSHRelayConnection -> remote engine_host_cli --relay -> remote daemon -> remote service`

### Remote SSH path (degraded)
`EngineHostControlChannel -> ssh remote one-shot engine_host_cli command -> remote service`

Notes:
- Remote persistent relay works best when remote daemon is already running.
- Per-command remote path does not require daemon pre-start, but is slower.
- PTY is not required; SSH exec command capability is required.

## 4. Host State and Files

Default state location:
- Derived from MP13 config dir when available
- Fallback: `~/.mp13-llm/backend`

Important files:
- `managed_engines.json`
- `engine_host_control.json`
- `host_daemon.pid`
- `logs/<engine_id>.log`

## 5. What Remote Clients Can Do

Remote clients (through SSH-backed control channel) can:
- Discover running managed workers
- Spawn and shut down worker processes
- Ensure worker process is running
- Register/read endpoint metadata
- Inspect endpoint capabilities (`/health`, `/capabilities`, `/inference`, `/ws`)
- Tail/follow worker logs
- Apply claim/token/resource coordination semantics
- Use config-driven worker connect/spawn flow (`connect-from-config`)

Remote clients do **not** automatically get traffic proxying to workers. They must:
- Reach worker endpoints directly, or
- Use separate networking/proxying infrastructure

## 6. Linux Setup

## 6.1 Admin prerequisites (one-time)

Example (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y openssh-server python3 python3-venv git
sudo systemctl enable --now ssh
```

If GPU is required, admin also installs and validates NVIDIA driver/CUDA stack.

SSH policy requirements for remote hosting control:
- User can authenticate by key
- User can execute non-interactive remote commands
- Account is not restricted to SFTP-only
- PTY is optional (not required)

Optional admin hardening:
- Restrict users with `Match User`
- Use forced commands and dedicated service accounts if desired
- Use firewall policy aligned with your endpoint exposure model

## 6.2 Non-admin user setup

```bash
mkdir -p ~/mp13
cd ~/mp13
# clone / sync repo and install runtime as your environment requires
```

Start host daemon (foreground):

```bash
python -m hosting.engine_host_cli --daemon
```

Start host daemon (background):

```bash
python -m hosting.engine_host_cli --daemon --background
```

Check status by command:

```bash
python -m hosting.engine_host_cli discover-running
```

Spawn a managed worker (example):

```bash
cat <<'JSON' | python -m hosting.engine_host_cli --payload-stdin spawn
{"engine_id":"worker1","command":["python","-m","http.server","9001"],"endpoint":"http://127.0.0.1:9001"}
JSON
```

## 7. Windows Setup

## 7.1 Admin prerequisites (one-time)

1. Install Python.
2. Install and enable OpenSSH Server feature.
3. Ensure `sshd` service is running.
4. Grant account access and key-based auth as needed.
5. If GPU is required, install NVIDIA driver/CUDA stack.

PowerShell examples:

```powershell
# Install OpenSSH server capability (if missing)
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start and persist service
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

## 7.2 Non-admin user setup

In PowerShell:

```powershell
cd C:\path\to\repo
python -m hosting.engine_host_cli --daemon --background
python -m hosting.engine_host_cli discover-running
```

Spawn worker example:

```powershell
@'{"engine_id":"worker1","command":["python","-m","http.server","9001"],"endpoint":"http://127.0.0.1:9001"}'@ |
python -m hosting.engine_host_cli --payload-stdin spawn
```

## 8. Remote Client Usage Patterns

Assume client can SSH to host user account and repo/runtime is available on host.

## 8.1 Persistent remote control channel (preferred)

Requirements:
- Remote daemon already running (recommended)
- SSH exec command access
- Correct remote command path/environment

Control settings should include:
- `engine_host_ssh_target`
- `control_ssh_key`
- optional `engine_host_remote_cmd` (defaults to `python -m hosting.engine_host_cli`)

In SSH mode, channel will use relay command:
- `python -m hosting.engine_host_cli --relay`

## 8.2 Remote per-command fallback

If persistent relay fails, control channel can execute one-shot SSH CLI commands.

Pros:
- Works without persistent remote daemon in many cases

Cons:
- Higher latency and process overhead per command
- Less efficient for high-frequency control operations

## 9. Security and Exposure Guidance

- Keep daemon bound to localhost (`127.0.0.1`) unless you intentionally redesign exposure.
- Prefer SSH transport over opening daemon TCP externally.
- Treat issued control tokens as sensitive.
- If worker endpoints must be remotely reachable, place them behind controlled network policy (VPN/firewall/reverse proxy).
- Use dedicated service accounts for hosting where possible.

## 10. Operational Checklist

Before declaring host ready:

1. SSH login and remote command exec works:
   - `ssh user@host "echo ok"`
2. Python command works remotely:
   - `ssh user@host "python -V"`
3. Daemon can start:
   - `python -m hosting.engine_host_cli --daemon --background`
4. Control command returns:
   - `python -m hosting.engine_host_cli discover-running`
5. Worker spawn + endpoint probe works:
   - spawn command succeeds
   - `inspect-capabilities` shows expected endpoints
6. Logs can be tailed:
   - `logs-tail` and `logs-follow` return output

## 11. Troubleshooting

- `engine host command returned no output`
  - Check Python/module path on host
  - Check SSH command restrictions

- relay connection fails quickly
  - Remote daemon likely not running or not reachable by remote `--relay`
  - Start daemon and retry

- spawn works but endpoint unavailable
  - Worker process started but app inside worker did not bind expected port/path
  - Inspect worker log file and command/env templates

- permission denied on state files
  - Ensure runtime user owns writable state directory

