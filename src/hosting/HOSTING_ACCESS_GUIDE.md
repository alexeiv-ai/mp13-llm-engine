# MP13 Hosting Access Guide for API Clients

This guide is designed for clients, GUIs, and backend services interacting with the MP13 LLM Engine Hosting control plane and proxy services. It details the required connection protocols, authentication handshakes, and lifecycle management contracts.

> **Note:** The hosting architecture has undergone a complete redesign, replacing legacy TCP control and legacy auth roles with a hardened local-IPC, SSH-relay, and role-based access model. 

## 1. Architectural Overview

The hosting layer operates as a control plane for managing engine worker lifecycles and a traffic bridge for guarded API access. The previous unified TCP daemon has been split to isolate concerns:

1.  **Primary Control Daemon (`--daemon`)**: 
    - Listens *only* on cross-platform local IPC (Unix Domain Sockets or Windows Named Pipes).
    - Speaks a custom JSON-RPC protocol over IPC.
    - Used for worker lifecycle (`spawn`, `shutdown`), config management, and stream/RPC proxying.
2.  **Dedicated HTTP Ingress Daemon (`--daemon-http`)**: 
    - An optional, separate process for handling standard HTTP requests.
    - Provides `GET /health` and `* /proxy/<engine_id>/<path...>` for HTTP-like engine API routes.
3.  **Stateless CLI (`engine_host_cli`)**:
    - Still usable as a fallback for one-off commands (e.g., passing JSON to `--payload-stdin` and reading stdout).

*(Reference: `src/hosting/HOSTING.md`, `src/hosting/engine_host_cli.py`)*

## 2. Connecting to the Daemon

Clients must no longer assume a default TCP port for daemon control. The connection mechanism depends on the deployment scenario:

### 2.1. Local Connection (Local IPC)

When the client runs on the same host as the daemon:
1.  **Locate the PID File**: Read `<default_engine_config_dir>/hosting/state/daemon.pid`.
2.  **Extract Connection Metadata**: The PID file contains critical connection details:
    -   `pid`: Process ID of the daemon.
    -   `transport`: Set to `ipc` (legacy `tcp` may be present in older versions, but `ipc` is standard).
    -   `ipc_family`: Either `AF_UNIX` (Linux/macOS) or `AF_PIPE` (Windows).
    -   `ipc_address`: The socket path or named pipe address.
    -   `shutdown_token`: A secret token required to issue a graceful `__shutdown__` command.
3.  **Connect**: Establish a direct IPC connection using the provided `ipc_family` and `ipc_address`.

### 2.2. Remote Connection (SSH Relay)

**SSH forwarded-port control is no longer supported.** Instead, remote clients must use the SSH Relay pattern:
1.  **Open an SSH Session** to the target host.
2.  **Spawn the Relay Process**: Execute `python -m hosting.engine_host_cli --relay` within the SSH session.
3.  **Communicate via Stdio**: The `--relay` process automatically reads the remote `daemon.pid`, connects to the remote local IPC socket, and bridges stdin/stdout to the client over the SSH channel.

## 3. Authentication and Authorization (Contract v2)

The system enforces a strict, clean-slate Role-Based Access Control (RBAC) model. 

### 3.1. Roles
Legacy roles (`management`, `config`, `traffic`) are removed. Clients must request and assert the new roles:
-   `admin`: Full administration, override capabilities, key/session management.
-   `config_editor`: Can spawn workers and modify custom configs.
-   `worker_user`: Can spawn and communicate with engines.
-   `model_user_with_model_control`: Can use engines and override default models.
-   `model_user`: Can use existing model-engine sessions.
-   `diagnostic_user`: Read-only status and logs.

*(Reference: `HOSTING_CLIENT_BREAKING_CHANGES.md`)*

### 3.2. Checking Daemon Status & Bootstrap
Always call the `auth-status` command before attempting operations.

*   If `keys_count == 0`: The daemon is unconfigured. The client should prompt the operator to run the setup wizard or perform a local-only bootstrap to provision the first `admin` key.
*   Treat missing `daemon_version` as an auth or reachability failure, not necessarily an outdated daemon.

### 3.3. Remote vs. Local Session Issuance
-   **Local Clients**: Can use `auth-issue-session` with a shared secret to get a session token, provided the daemon is configured to allow it.
-   **Remote Clients (SSH/Truly Remote)**: Shared-secret bootstrap is **denied** (`shared_secret_bootstrap_not_supported_for_remote_connectivity`). Remote clients must use asymmetric key challenges:
    1.  `auth-begin-challenge` (requires `ssh_binding` metadata for remote connections).
    2.  Sign the challenge externally.
    3.  `auth-complete-challenge` to obtain the session token.

### 3.4. Safe-Only No-Auth Policy (`require_auth=false`)
`require_auth=false` is strictly restricted to safe, local-only, single-user profiles. If any remote ingress, shared access, or multi-user keys are detected, the daemon will reject the configuration or force `require_auth=true`.

## 4. Daemon Lifecycle, Claims, and Ownership

The daemon manages resources (like engines) using a "Claim" system.

### 4.1. Effective Endpoint Modes
-   **`exclusive`**: Only one owner/session can control the endpoint at a time. **Crucially, when the exclusive owner disconnects, the daemon automatically shuts down.**
-   **`shared`**: Multiple clients can interact based on their role permissions. The daemon remains alive until explicitly shut down.

### 4.2. Takeovers and Force Overrides
Clients may need to reclaim an endpoint or engine. The daemon enforces strict rules for `force_override=true`:
-   **Reason Required**: A `force_override_reason` string must always be provided.
-   **Localhost Confirmation**: Non-emergency force overrides on localhost require passing `force_override_confirmation="CONFIRM_LOCALHOST_FORCE_OVERRIDE"`.
-   **Emergency Overrides**: `force_override_emergency=true` bypasses confirmation but strictly requires one of three reasons (`stale_owner_unreachable`, `owner_malicious`, `security_incident`) and specific active/orphan state predicates to be met.

## 5. Client Integration Checklist

When building or updating a client to interact with the MP13 Hosting APIs, ensure the following:

1.  [ ] **Switch to IPC / Relay**: Remove TCP socket connection logic. Use PID file parsing for local IPC, and `--relay` for SSH.
2.  [ ] **Manage HTTP Ingress**: If your client relies on `GET /health` or standard HTTP proxy routes, ensure the `--daemon-http` process is started and managed correctly.
3.  [ ] **Implement Contract v2 Structured Denials**: Parse `error_code` and `error_details` from failed responses. Handle codes like `session_token_required`, `insufficient_scope`, `ownership_changed_reclaim_required`, and `engine_exclusive_conflict`.
4.  [ ] **Handle Actor Identities**: Expect actor IDs in the format `key:<key_id>`. The client-supplied `backend_id` is ignored by the daemon's internal ACLs.
5.  [ ] **Handle Keepalives**: Implement keepalives to maintain active claims and avoid being marked as an "orphan" owner (default TTL is 120 seconds).
6.  [ ] **Route Metrics Properly**: Diagnostics from `host-metrics` are process-scoped. Fetch metrics individually per endpoint daemon, not globally across the entire host.