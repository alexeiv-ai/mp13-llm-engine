# MP13 Hosting - Developer Orientation & Guide

This file serves as the primary developer orientation and guide for the `src/hosting` feature within the `mp13-llm-engine` project. 

The `hosting` module provides the control-plane and guarded traffic bridge for managing local and remote engine workers.

## Important Documentation Links

- **[Hosting Access Guide for API Clients](HOSTING_ACCESS_GUIDE.md)**: If you are building a GUI, client, or backend service that consumes the hosting APIs, **start here**. It covers connection protocols (IPC/SSH Relay), authentication handshakes, the Role-Based Access Control (RBAC) model, and lifecycle management contracts.
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
Workers run as separate processes. The host does not expose worker private keys and does not require worker ports to be publicly forwarded. Transport between the host daemon and worker processes is securely handled via cross-platform local IPC (`AF_UNIX` or `AF_PIPE`).

### The Daemons
The architecture relies on separate daemons to isolate concerns:
1. **Primary Control Daemon (`--daemon`)**: IPC-only daemon handling custom JSON-RPC for control and proxying.
2. **HTTP Ingress Daemon (`--daemon-http`)**: Dedicated daemon handling standard HTTP ingress (e.g., `/health`, API proxy routes).

### Network Support Boundaries & Trade-offs
The hosting layer has strict boundaries regarding network transport:

#### 1. Local IPC & SSH Relay (Primary Transport)
The primary control daemon (`--daemon`) is explicitly restricted to cross-platform **Local IPC** (Unix Sockets or Windows Named Pipes) and does not listen on a TCP port. It is the **only** daemon that accepts control commands (e.g., `spawn`, `shutdown`, `proxy-rpc-call`).

*   **How Remote Access Works (SSH Relay vs. SSH Tunnel)**: Because there is no TCP port, traditional "SSH Tunneling" (e.g., `ssh -L 19876:localhost:19876`) is impossible. Instead, remote clients must use the **SSH Relay** pattern. The client opens an SSH session and runs `python -m hosting.engine_host_cli --relay`. This creates a subprocess on the remote machine that reads the IPC socket and bridges the JSON-RPC traffic back to the client over the standard SSH standard I/O streams.
*   **Pros**: 
    *   Inherently secure against external network probing.
    *   Fast, low-overhead communication between local processes.
    *   No persistent background tunnel processes to manage; the relay lives and dies with the SSH session.
*   **Cons**:
    *   Requires local filesystem access to read the PID file and connect.
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

#### 3. TCP Port Forwarding (Deprecated)
*   **Boundary**: Legacy daemon setups used to expose a local TCP port (19876) that accepted all control commands. This is explicitly deprecated and removed from the v2 architecture in favor of IPC.
*   **Cons**: Vulnerable to port scanning, lacks inherent OS-level access controls (which IPC provides via file permissions), and requires complex SSH port forwarding instead of a simple subprocess relay.

## 2. Developer Workflows & CLI Examples

As a developer, you will often interact with the hosting layer using the `engine_host_cli`.

### 2.1 Setup and Reconfiguration

Use the setup tool to initialize or reconfigure access. See the [Hosting Configuration Script](hosting_config_script.md) for more details.

```powershell
$env:PYTHONPATH='src'
# Interactive wizard
python -m hosting.engine_host_cli --hosting-config --interactive

# Non-interactive example
python -m hosting.hosting_config --mode local_only --endpoint-mode exclusive --lifecycle-profile detached_user_process --key-source import --admin-key-id admin-main --admin-public-key-file "$HOME\.ssh\id_ed25519.pub" --require-auth
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

Bootstrap the first admin key (when `keys_count == 0`):
```powershell
@'{"key_id":"admin-key","key_secret":"CHANGE_ME","role":"admin"}'@ | python -m hosting.engine_host_cli --payload-stdin auth-upsert-key

@'{"require_auth":true}'@ | python -m hosting.engine_host_cli --payload-stdin set-control-config
```

Issue a control session token:
```powershell
@'{"key_id":"admin-key","key_secret":"CHANGE_ME","scope":"control","ttl_seconds":900}'@ | python -m hosting.engine_host_cli --payload-stdin auth-issue-session
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

### 3.1 Host Metrics
`host-metrics` provides process-runtime diagnostics, including in-flight proxy requests, proxy byte totals, auth denial counters, and a recent request ring buffer.

```powershell
@'{"session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin host-metrics
```
*Note: Metrics are process-scoped. For stable metrics, query the running daemon, not a one-shot CLI invocation. In multi-endpoint GUIs, fetch metrics through each endpoint's own host channel.*

### 3.2 Audit Logs
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