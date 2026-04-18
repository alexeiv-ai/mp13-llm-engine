# MP13 Hosting - Developer Orientation & Guide

This file serves as the primary developer orientation and guide for the `src/hosting` feature within the `mp13-llm-engine` project. 

The `hosting` module provides the control-plane and guarded traffic bridge for managing local and remote engine workers.

## Important Documentation Links

- **[Hosting Access Design](hosting_access.md)**: If you are building a GUI, client, or backend service that consumes the hosting APIs, start with Section 11, `Client Integration Contract`. The legacy `HOSTING_ACCESS_GUIDE.md` file now redirects there.
- **[Client Breaking Changes](HOSTING_CLIENT_BREAKING_CHANGES.md)**: Details on intentional auth/authz breaking changes (e.g., legacy role removal).
- **[Hosting Configuration Script](hosting_config_script.md)**: Specification for the user-facing setup and reconfiguration script (`hosting_config`).

---

## 1. Architecture Overview

Hosting acts as a control-plane plus a guarded traffic bridge.

- **Control-plane**:
  - Worker lifecycle management (`spawn`, `shutdown`, `ensure-running`, `discover-running`).
  - Config-driven worker startup.
  - Claims, tokens, and resource ownership state management.
- **Traffic bridge**:
  - `proxy-request`: Forwards HTTP-like engine API requests over local IPC.
  - `proxy-rpc-*`: Provides generic sync/async RPC over local IPC.
  - Traffic authorization and policy enforcement prior to forwarding.

### Process Model
Workers run as separate processes. The host does not expose worker private keys and does not require worker ports to be publicly forwarded. Worker traffic is proxied through hosting control/traffic commands and the worker IPC layer; the daemon itself is reached through local IPC by default and can also expose a guarded loopback TCP compatibility listener when policy permits.

### The Daemons
The architecture relies on separate daemons to isolate concerns:
1. **Primary Control Daemon (`--daemon`)**: Hosts the control JSON-RPC protocol for lifecycle, auth, claims, config, and proxy commands. It starts a local IPC listener and may also start a loopback TCP listener when the configured auth policy allows it.
2. **HTTP Ingress Daemon (`--daemon-http`)**: A separate HTTP ingress mode implemented in `engine_host_daemon.py`, handling standard HTTP ingress such as `GET /health` and worker API proxy routes.

### Network Support Boundaries & Trade-offs
The hosting layer has strict boundaries regarding network transport:

#### 1. Local IPC & SSH Relay (Primary Transport)
The preferred local transport for the primary control daemon is cross-platform local IPC: Unix sockets on Linux/macOS and Windows named pipes on Windows. The primary daemon is the only daemon that accepts control commands such as `spawn`, `shutdown`, and `proxy-rpc-call`.

*   **SSH Relay**: Remote clients can open an SSH session and run `python -m hosting.engine_host_cli --relay`. The relay process uses the daemon PID file to reach the local control channel, preferring local IPC and falling back to loopback TCP when needed, then bridges JSON-RPC traffic over SSH stdio.
*   **Pros**: 
    *   Inherently secure against external network probing.
    *   Fast, low-overhead communication between local processes.
    *   No persistent background tunnel processes to manage; the relay lives and dies with the SSH session.
*   **Cons**:
    *   Requires the remote SSH account to execute the relay command and read the daemon PID file.
    *   Not accessible via standard load balancers or HTTP-based reverse proxies.

#### 2. Dedicated HTTP Ingress (`--daemon-http`)
Because IPC cannot serve standard network requests, a separate HTTP ingress daemon is available. It binds to local TCP by default (usually port 19877) to serve HTTP.
*   **Boundary**: This daemon **DOES NOT** accept daemon control commands (like `spawn` or `auth-issue-session`). It is strictly a traffic bridge.
*   **Pros**:
    *   Provides standard `GET /health` routes necessary for infrastructure monitoring (e.g., Kubernetes liveness probes).
    *   Allows reverse proxies (like Nginx) or load balancers to route REST/HTTP traffic directly to managed engines via the `* /proxy/<engine_id>/<path...>` route.
*   **Cons**:
    *   Opens a TCP port, requiring firewall or network-policy management.
    *   Limited to HTTP verbs/traffic (cannot serve full host-control RPC or async streams).
    *   Requires managing a second daemon process.

#### 3. TCP Port Forwarding (Conditional Remote Access)
*   **Boundary**: The primary control daemon can bind to a local TCP port (`127.0.0.1:19876`) to support secure SSH port forwarding (`ssh -L 19876:127.0.0.1:19876`).
*   **Security Guardrail**: The TCP listener is **disabled by default**. It is only enabled when the daemon meets all of the following conditions:
    1.  `require_auth` is set to `true`.
    2.  At least one `admin` key is configured in the keyring.
    3.  At least one `transport` key is configured in the keyring.

##### Optional "Temp Admin Access" Lock-down Workflow
If a remote client relies solely on the `--relay` subprocess via IPC, the host's `~/.ssh/authorized_keys` must allow that SSH identity to execute commands. However, leaving terminal/command execution open is a significant security hole: if the key is compromised, an attacker can spawn a shell and read the physical `.json` keyrings.

The conditional loopback TCP listener allows operators to avoid long-term SSH command execution for a transport key while still retaining daemon control through SSH port forwarding. The workflow relies on **temporary admin access**:
1.  **Temp Admin Access:** The operator connects to the host using an unrestricted terminal session (e.g., via a standard SSH login). This makes life easier because you do not need complex system services (like systemd) to manage the daemon.
2.  **Start the Daemon:** The operator manually starts the daemon in the background (`python -m hosting.engine_host_cli --daemon --background`).
3.  **Lock Down the Session:** The operator edits their host's `~/.ssh/authorized_keys` to completely disable terminal and command execution for the `transport` key, restricting it *only* to port forwarding:
    ```text
    command="/bin/false",no-pty,permitopen="127.0.0.1:19876" ssh-ed25519 AAAA... transport_user@machine
    ```
4.  **Steady State:** The transport SSH key can use standard SSH tunnels to control the daemon over TCP, while terminal and command execution for that key remain disabled.
5.  **Recovery (When the Daemon Dies):** If the daemon crashes or the host reboots, the TCP port will no longer answer. Because the terminal is locked down with `command="/bin/false"`, standard remote restart scripts (like `EngineHostControlChannel.restart_remote_daemon` which uses SSH exec) will be denied by the SSH server. 
6.  **Service-managed recovery:** To solve the locked-down recovery problem, operators can wrap the daemon in an OS-level service manager such as `systemd` or Windows Services. A service policy can restart the daemon after exit, allowing remote restart through daemon control commands without re-enabling SSH command execution for the transport key.

### Daemon Lifecycle Profiles: Foreground vs. Detached
The daemon **does not** automatically detach by default. You must explicitly choose its lifecycle behavior depending on your current configuration phase:

*   **Before Auth is Configured (Bootstrap Phase):** We recommend running the daemon in the foreground (`python -m hosting.engine_host_cli --daemon`). Because the daemon is temporarily unauthenticated, running it attached to your terminal ensures that if you walk away or close the SSH session, the daemon dies immediately, securely closing the temporary access hole.
*   **After Auth is Configured (Steady State):** Once keys are provisioned and `require_auth=true` is set, you should transition to a durable lifecycle. 
    *   *If you have root/admin rights:* Wrap the daemon in an OS-level service manager (`systemd` or Windows Services). This provides the **Service Managed** remote restart pattern described above, allowing auto-recovery and remote restarts even when your terminal is locked down.
    *   *If you do not have root/admin rights:* You must manually start the daemon as a **Detached User Process** (`python -m hosting.engine_host_cli --daemon --background`). The daemon will survive when you close your terminal, allowing you to control it remotely via SSH Relay. However, be aware that if the daemon crashes, you *cannot* restart it remotely if you have locked down your terminal with `command="/bin/false"`. You will need to regain temporary terminal access to manually restart it.

### Can the Daemon be Exposed Directly to the Internet?
**No, natively it should not be treated as an internet-facing daemon.** The conditional control TCP listener (`127.0.0.1:19876`) and the HTTP ingress daemon (`127.0.0.1:19877` by default) bind to loopback.

If you want to expose HTTP proxy routes to a broader network, run a reverse proxy such as NGINX, HAProxy, or Traefik in front of the loopback HTTP ingress and apply external network policy. Do not expose the control plane directly unless you have a reviewed deployment-specific wrapper and auth/network policy.

## 2. Developer Workflows & CLI Examples

As a developer, you will often interact with the hosting layer using the `engine_host_cli`.

### 2.1 Setup and Reconfiguration

Use the setup tool to initialize or reconfigure access. See the [Hosting Configuration Script](hosting_config_script.md) for more details.

```powershell
$env:PYTHONPATH='src'
# Interactive wizard
python -m hosting.hosting_config_cli --interactive

# Non-interactive example
python -m hosting.hosting_config_cli --no-interactive --mode local_only --endpoint-mode exclusive --lifecycle-profile detached_user_process --key-source import --admin-key-id admin-main --admin-public-key-file "$HOME\.ssh\id_ed25519.pub" --require-auth
```

### 2.2 Starting the Daemons

```powershell
$env:PYTHONPATH='src'
# Start primary IPC daemon
python -m hosting.engine_host_cli --daemon --background

# Start dedicated HTTP ingress daemon
python -m hosting.engine_host_cli --daemon-http --background
```

### 2.3 Bootstrapping Authentication

Check current auth state (useful before bootstrap):
```powershell
python -m hosting.engine_host_cli auth-status
```

Prefer `hosting_config_cli` for normal setup. The raw commands below are low-level local-only examples for development or recovery. Shared-secret sessions are not supported for non-local connectivity modes.

Provision a first local admin shared-secret key when local bootstrap policy allows it:
```powershell
@'{"key_id":"admin-key","key_secret":"CHANGE_ME","role":"admin"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key
```

Issue a local control session token:
```powershell
@'{"key_id":"admin-key","key_secret":"CHANGE_ME","scope":"control","ttl_seconds":900}'@ | python -m hosting.engine_host_cli --payload-stdin auth-issue-session
```

Then use the control token for authenticated config changes:
```powershell
@'{"require_auth":true,"session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin set-control-config
```

### 2.4 Managing Engine Workers

Use your control token to manage engines:

```powershell
# Discover running engines
@'{"session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin discover-running

# Launch an engine from config
@'{"config_path":"default","engine_id":"worker_cfg","model_path":"C:\\models\\granite-3.3-2b-instruct"}'@ | python -m hosting.engine_host_cli --payload-stdin connect-from-config
```

### 2.5 Proxying Worker Requests (RPC & Streams)

Once a traffic session is issued, developers can test proxy commands.

```powershell
# Sync RPC call
@'{
  "engine_id":"worker1",
  "method":"rpc.describe",
  "params":{},
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-call

# Async open & receive
@'{
  "engine_id":"worker1",
  "method":"run-inference",
  "params":{"messages_list":[[{"role":"user","content":"hello"}]],"stream":true},
  "request_id":"req-1",
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-open

@'{
  "engine_id":"worker1",
  "stream_id":"<stream_id>",
  "timeout_seconds":2.0,
  "max_items":64,
  "session_token":"<traffic_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin proxy-rpc-recv
```

## 3. Diagnostics and Auditing

Diagnostics and auditing are heavily utilized during development and operations.

### 3.1 Hosting Access Doctor
Run the hosting access doctor for file/config/key custody checks:

```powershell
python -m hosting.hosting_config_cli --no-interactive --doctor
```

### 3.2 Host Metrics
`host-metrics` provides process-runtime diagnostics, including in-flight proxy requests, proxy byte totals, auth denial counters, and a recent request ring buffer.

```powershell
@'{"session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin host-metrics
```
*Note: Metrics are process-scoped. For stable metrics, query the running daemon, not a one-shot CLI invocation. In multi-endpoint GUIs, fetch metrics through each endpoint's own host channel.*

### 3.3 Audit Logs
Hosting maintains an audit trail of key and session lifecycle events.

```powershell
# Audit sessions (e.g., traffic scope)
@'{"session_token":"<control_token>","scope":"traffic","limit":50,"offset":0}'@ | python -m hosting.engine_host_cli --payload-stdin auth-list-sessions

# Query specific audit events (Admin-only)
@'{"session_token":"<control_token>","event_type":"auth_revoke_key","limit":50,"offset":0}'@ | python -m hosting.engine_host_cli --payload-stdin auth-audit-list
```

## 4. Current Limitations & Implementation Notes

- **IPC Transport Only**: Worker transport is local IPC only. There is no host-managed remote worker transport.
- **HTTP Bridge**: `proxy-request` serves as a compatibility bridge for HTTP-like engine routes over IPC. New generic worker integrations should prefer `proxy-rpc-*` for sync/async RPC.
- **Metrics Scope**: Metrics are per-process runtime and are not persisted across daemon restarts.
- **Bootstrap Credentials**: Host channel credential bootstrap requires wiring `engine_host_key_id` + `engine_host_key_secret` (or a pre-issued `engine_host_session_token`) in control settings/profile construction.
- **Legacy Components**: Legacy role payloads are no longer accepted by runtime auth surfaces. Legacy roles (`management`, `config`, `traffic`) are removed from clean-slate runtime auth paths.
