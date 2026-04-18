# MP13 Hosting Access Guide for API Clients

This guide is designed for clients, GUIs, and backend services interacting with the MP13 LLM Engine Hosting control plane and proxy services. It details the required connection protocols, authentication handshakes, and lifecycle management contracts.

## 1. Architectural Overview

The hosting layer operates as a control plane for managing engine worker lifecycles and a traffic bridge for guarded API access. The architecture isolates concerns into separate processes:

1.  **Primary Control Daemon (`--daemon`)**: 
    - The core control plane, which natively listens **only** on cross-platform local IPC (Unix Domain Sockets on Linux/macOS or Named Pipes on Windows).
    - Uses a custom JSON-RPC protocol.
    - Used for worker lifecycle (`spawn`, `shutdown`), config management, and stream/RPC proxying.
    - **Conditional TCP Support:** The daemon will conditionally open a local TCP listener strictly on `127.0.0.1:19876` if and only if `require_auth=true` and both an `admin` key and a `transport` key are configured. This is exclusively to support secure SSH port-forwarding. It never binds to `0.0.0.0`.
2.  **Dedicated HTTP Ingress Daemon (`--daemon-http`)**: 
    - An optional, separate process for handling standard HTTP requests.
    - Binds strictly to `127.0.0.1` and provides `GET /health` and `* /proxy/<engine_id>/<path...>` for HTTP-like engine API routes.
3.  **Stateless CLI (`engine_host_cli`)**:
    - Usable as a fallback for one-off commands (e.g., passing JSON to `--payload-stdin` and reading stdout).

## 2. Connecting to the Daemon

The connection mechanism depends entirely on your deployment scenario:

### 2.1. Local Connection (Local IPC)

When the client runs on the same machine as the daemon:
1.  **Locate the PID File**: Read `<default_engine_config_dir>/hosting/state/daemon.pid`.
2.  **Extract Connection Metadata**: The PID file contains critical connection details:
    -   `transport`: Set to `local_ipc`.
    -   `ipc_family`: Either `AF_UNIX` (Linux/macOS) or `AF_PIPE` (Windows).
    -   `ipc_address`: The socket path or named pipe address.
    -   `shutdown_token`: A secret token required to authenticate the IPC connection itself (verifying you are the same OS user).
3.  **Connect**: Establish a direct IPC connection using the provided `ipc_family` and `ipc_address` along with the `shutdown_token`.

### 2.2. Remote Connection (SSH Relay)

For remote clients, direct TCP connections to a public port are not natively supported. Instead, remote clients must use the **SSH Relay pattern**:
1.  **Open an SSH Session** to the target host.
2.  **Spawn the Relay Process**: Execute `python -m hosting.engine_host_cli --relay` within the SSH session.
3.  **Communicate via Stdio**: The `--relay` process automatically connects to the local IPC socket on the remote machine and bridges the JSON-RPC traffic over the SSH standard I/O streams.

### 2.3. External Ingress (Reverse Proxy)

If you require standard external network access (e.g., HTTPS/WSS over the internet), you must deploy an external Reverse Proxy (like NGINX or Traefik) on the host to route traffic to the daemon's `127.0.0.1` endpoints. 

## 3. Authentication and Authorization

The system enforces a strict Role-Based Access Control (RBAC) model governed by a configured `connectivity_mode`.

### 3.1. Connectivity Modes and Security Policies

The daemon operates in one of three connectivity modes, which dictate the acceptable authentication methods. You can determine the current mode by inspecting the daemon's runtime configuration file.

1.  **`local_only`**: For single-host, loopback-only usage.
    - Allows the use of `shared_secret` authentication for rapid session bootstrapping.
    - Can optionally operate with `require_auth=false` (No-Auth Policy) if strictly configured for single-user, `exclusive` endpoint mode.
2.  **`ssh_tunnel_only`** and **`truly_remote`**: For any scenario involving off-host clients or reverse proxies.
    - These modes **enforce strict authentication** (`require_auth=true` is mandatory).
    - **`shared_secret` authentication is unconditionally denied.**
    - All API requests must be authenticated using the **Asymmetric Public-Key Challenge Flow**.
    - Clients must supply `_ssh_session_binding` metadata in their payloads to securely lock their API session token to their specific connection/routing path.

*(Note: `ssh_tunnel_only` and `truly_remote` share the exact same strict runtime enforcement logic, but are distinguished during setup for auditing and future adaptive hardening policies.)*

### 3.2. Roles
Clients must be provisioned with one of the following standard roles. A given role can have multiple keys assigned to it.
-   `admin`: Full administration, override capabilities, key/session management.
-   `config_editor`: Can spawn workers and modify custom configs.
-   `worker_user`: Can spawn and communicate with engines.
-   `model_user_with_model_control`: Can use engines and override default models.
-   `model_user`: Can use existing model-engine sessions.
-   `diagnostic_user`: Read-only status and logs.

**The Special `transport` Role:**
The `transport` role is an orthogonal, highly specialized security layer. It possesses **zero API privileges** (it cannot issue sessions or spawn workers). Its sole purpose is to act as a flag to enable the local TCP listener (`127.0.0.1:19876`). Operators assign this public key to a restricted user in the host OS's `~/.ssh/authorized_keys` specifically for locking down SSH port-forwarding tunnels. It cannot be spoofed to the daemon, as its authentication is handled natively by the SSH server during tunnel creation.

### 3.3. Authentication Flows

*   **Shared Secret Flow (Local Only):**
    Clients use the `auth-issue-session` command, passing the `key_secret` to obtain a session token. The daemon securely hashes the secret for verification and never stores it in plaintext.
*   **Public-Key Challenge Flow (Required for Remote):**
    1.  Call `auth-begin-challenge` (injecting `_ssh_session_binding` metadata).
    2.  Sign the returned challenge cryptographically on the client side.
    3.  Call `auth-complete-challenge` with the signed payload to obtain a session token.

## 4. Daemon Lifecycle, Claims, and Ownership

The daemon manages resources (like engines) using a "Claim" system based on your session token.

### 4.1. Effective Endpoint Modes
-   **`exclusive`**: Only one owner/session can control the endpoint at a time. **When the exclusive owner disconnects, the daemon automatically shuts down.**
-   **`shared`**: Multiple clients can interact based on their role permissions. The daemon remains alive until explicitly shut down.

### 4.2. Takeovers and Force Overrides
Clients may need to reclaim an endpoint or engine. The daemon enforces strict rules for `force_override=true`:
-   **Reason Required**: A `force_override_reason` string must always be provided.
-   **Localhost Confirmation**: Non-emergency force overrides on localhost require passing `force_override_confirmation="CONFIRM_LOCALHOST_FORCE_OVERRIDE"`.
-   **Emergency Overrides**: `force_override_emergency=true` bypasses confirmation but strictly requires a valid emergency reason (e.g., `stale_owner_unreachable`, `owner_malicious`, `security_incident`) and specific state predicates to be met.

## 5. Client Integration Checklist

When building or updating a client to interact with the MP13 Hosting APIs, ensure the following:

1.  [ ] **Use IPC or Relay**: Implement PID file parsing for local IPC connections, and use `--relay` for SSH-based remote connections.
2.  [ ] **Support Challenge Flow**: If you support remote connectivity, you must implement the asymmetric `auth-begin-challenge` / `auth-complete-challenge` flow. Shared secrets will be rejected.
3.  [ ] **Inject Session bindings**: Ensure your client dynamically reads its current SSH/connection state and injects `_ssh_session_binding` (with `target` and `key_fingerprint`) into API requests.
4.  [ ] **Manage HTTP Ingress**: If your client relies on standard HTTP proxy routes, ensure the `--daemon-http` process is started or a reverse proxy is configured.
5.  [ ] **Handle Structured Denials**: Parse `error_code` and `error_details` from failed responses. Be prepared to handle `session_token_required`, `ssh_binding_mismatch`, `shared_secret_bootstrap_not_supported_for_remote_connectivity`, and `engine_exclusive_conflict`.
6.  [ ] **Handle Keepalives**: Implement keepalives (`__ping__` or API polls) to maintain active claims and avoid being marked as an "orphan" owner.
