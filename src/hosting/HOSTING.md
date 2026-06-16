# MP13 Hosting - Developer Orientation & Guide

This file serves as the primary developer orientation and guide for the `src/hosting` feature within the `mp13-llm-engine` project. 

The `hosting` module provides the control-plane and guarded traffic bridge for managing local and remote engine workers.

## Important Documentation Links

- **[Hosting Access Design](HOSTING_ACCESS.md)**: If you are building a GUI, hosting consumer backend, or integration that consumes the hosting APIs, start with Section 11, `Hosting Consumer Integration Contract`.
- **[Hosting Configuration Script](HOSTING_CONFIG_SCRIPT.md)**: Specification for the user-facing setup and reconfiguration script (`hosting_config`).
- **[Sandbox Architecture](sandbox/SANDBOX_ARCHITECTURE.md)**: Shared sandbox policy, launch, and broker foundation for hosted workers.
- **[Toolbox Worker](sandbox/TOOLBOX_WORKER.md)** and **[Generic Worker](sandbox/GENERIC_WORKER.md)**: Worker-specific sandbox and IPC contracts.

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
Workers run as separate processes. The host does not expose worker private keys and does not require worker ports to be publicly forwarded. Worker traffic is proxied through hosting control/traffic commands and the worker IPC layer; the daemon control plane is reached through local IPC metadata discovered from the daemon PID file.

Worker sandboxing is layered under this process model. The shared sandbox layer normalizes policy, launches workers, persists sandbox runtime metadata, and brokers filesystem/HTTP access where configured. Generic/model workers and sandboxed toolbox executors share that foundation but expose different worker contracts; see the sandbox docs linked above for the split.

### The Daemons
The architecture relies on separate daemons to isolate concerns:
1. **Primary Control Daemon (`--daemon`)**: Hosts the control JSON-RPC protocol for lifecycle, auth, claims, config, and proxy commands. It starts a local IPC listener and writes the connection metadata to the daemon PID file.
2. **HTTP Ingress Daemon (`--daemon-http`)**: A separate HTTP ingress mode implemented in the `hosting.daemon` package, handling standard HTTP ingress such as `GET /health` and worker API proxy routes.

### Network Support Boundaries & Trade-offs
The hosting layer has strict boundaries regarding network transport:

#### 1. Local IPC & SSH Relay (Primary Transport)
The preferred local transport for the primary control daemon is cross-platform local IPC: Unix sockets on Linux/macOS and Windows named pipes on Windows. The primary daemon is the only daemon that accepts control commands such as `spawn`, `shutdown`, and `proxy-rpc-call`.

*   **SSH Relay**: Remote hosting consumers use an SSH transport key constrained to the forced command `python -m hosting.engine_host_cli --relay-wrapper`. The wrapper connects to an existing daemon or starts the detached user daemon when the saved remote/auth policy allows it, then bridges JSON-RPC traffic over SSH stdio.
*   **Pros**: 
    *   Inherently secure against external network probing.
    *   Fast, low-overhead communication between local processes.
    *   No persistent background tunnel processes to manage; the relay lives and dies with the SSH session.
*   **Cons**:
    *   Requires the remote SSH account to execute the relay command and read the daemon PID file.
    *   Not accessible via standard load balancers or HTTP-based reverse proxies.
*   **Runtime behavior**:
    *   SSH must be able to execute the wrapper. A running daemon alone is not remotely controllable because daemon control is local IPC only.
    *   If the daemon is already running, the wrapper attaches through PID-file local IPC metadata.
    *   If the daemon is not running, wrapper auto-start is only attempted when saved hosting config is remote-enabled, `require_auth=true`, has registered keys, and uses `detached_user_process` lifecycle.
    *   Control operations sent through the relay still need the normal hosting auth/session required by the command.

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

#### 3. No Full Control Plane Over Port-Only TCP
The primary control daemon does not support a loopback TCP control listener today. The daemon blocks it server-side, and local clients must read PID-file local IPC metadata. Remote clients must use SSH relay command execution unless a future, reviewed remote control transport is added.

*   **Boundary**: `DEFAULT_DAEMON_PORT` is metadata compatibility only; consumers must not assume `127.0.0.1:19876` is connectable.
*   **SSH requirement**: The transport key must be restricted to the relay wrapper command (`python -m hosting.engine_host_cli --relay-wrapper`). Do not grant PTY or shell access to this key; the supported posture is a forced command with `no-pty` and the other relay hardening options.
*   **Setup requirement**: Installing a forced-command wrapper into user-scoped `authorized_keys` can be done by the target account. Machine-wide SSH service, firewall, or service-manager changes require explicit administrator/root elevation; the setup tooling does not store that password.
*   **HTTP ingress boundary**: `--daemon-http` serves health and worker HTTP proxy routes. It is not a full daemon control-plane API and does not replace SSH relay for commands such as `spawn`, auth/key management, or control-config changes.
*   **TBD**: Straight SSH port forwarding to daemon TCP control is not supported in this release. It needs an explicit TCP client mode, authorized-key mode, tests, and security review before becoming a consumer contract.

### Daemon Lifecycle Profiles: Foreground vs. Detached
The daemon **does not** automatically detach by default. You must explicitly choose its lifecycle behavior depending on your current configuration phase:

*   **Before Auth is Configured (Bootstrap Phase):** We recommend running the daemon in the foreground (`python -m hosting.engine_host_cli --daemon`). Because the daemon is temporarily unauthenticated, running it attached to your terminal ensures that if you walk away or close the SSH session, the daemon dies immediately, securely closing the temporary access hole.
*   **After Auth is Configured (Steady State):** Once keys are provisioned and `require_auth=true` is set, you should transition to a durable lifecycle. 
    *   *If you have root/admin rights:* Wrap the daemon in an OS-level service manager (`systemd` or Windows Services). This provides the **Service Managed** remote restart pattern described above, allowing auto-recovery and remote restarts even when your terminal is locked down.
    *   *If you do not have root/admin rights:* You must manually start the daemon as a **Detached User Process** (`python -m hosting.engine_host_cli --daemon --background`). The daemon will survive when you close your terminal, allowing you to control it remotely via SSH Relay. However, if the daemon crashes, remote restart still requires SSH command execution or a separate service manager.

### Can the Daemon be Exposed Directly to the Internet?
**No, natively it should not be treated as an internet-facing daemon.** Daemon control is local IPC only. The HTTP ingress daemon (`127.0.0.1:19877` by default) binds to loopback.

If you want to expose HTTP proxy routes to a broader network, run a reverse proxy such as NGINX, HAProxy, or Traefik in front of the loopback HTTP ingress and apply external network policy. Do not expose the control plane directly unless you have a reviewed deployment-specific wrapper and auth/network policy.

## 2. Developer Workflows & CLI Examples

As a developer, you will often interact with the hosting layer using the `engine_host_cli`.

### 2.1 Setup and Reconfiguration

Use the setup tool to initialize or reconfigure access. See the [Hosting Configuration Script](HOSTING_CONFIG_SCRIPT.md) for more details.

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

Validate an existing token before reusing it:
```powershell
@'{"token":"<control_token>","scope":"control","expected_key_id":"admin-key","check_ssh_binding":true}'@ | python -m hosting.engine_host_cli --payload-stdin auth-validate-session
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

For long-running lifecycle work, especially model startup, prefer the async
operation wrapper. It returns an `operation_id` immediately and lets a UI or
backend poll progress without blocking its control connection:

```powershell
# Start config-driven launch as an operation
@'{
  "command":"connect-from-config",
  "payload":{
    "config_path":"default",
    "engine_id":"worker_cfg",
    "model_path":"C:\\models\\granite-3.3-2b-instruct",
    "session_token":"<control_token>"
  }
}'@ | python -m hosting.engine_host_cli --payload-stdin op-start

# Poll status
@'{"operation_id":"<operation_id>","session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin op-status

# Request cancellation when supported by the operation
@'{"operation_id":"<operation_id>","reason":"user_requested","session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin op-cancel
```

`op-status` returns a public operation snapshot with `status`, `done`,
`result`/`error`, and `progress_events`. For `connect-from-config`, the daemon
also makes a best-effort attempt to correlate the operation with the spawned
worker registration and parse model-loading progress from the worker log. When
available, callers may see `progress_percent`, `progress_text`, and
`diagnostics.worker_log`. Operation snapshots are persisted best-effort under
the hosting state directory so recent operation status can survive daemon
object recreation; consumers should still treat operation status as operational
telemetry, not as a durable job queue contract.

### 2.5 Workflow Runtime APIs

Workflow runtime APIs are the migration path for existing workflow helper
lanes. New integrations should prefer workflow-named commands and treat old
helper command names as compatibility aliases.

Workflow Python helper profile:

```powershell
# Derive/inspect the host-owned environment identity.
@'{
  "profile":"helper",
  "environment_name":"workflow-python-helper",
  "python":{"import_allowlist":["json"],"package_pins":{}},
  "sandbox_policy":{"sandbox":{"enabled":true,"profile":"workflow_python_helper_v1"}}
}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-environment-spec

# Ensure a worker/pool for that environment key.
@'{
  "profile":"helper",
  "environment_key":"<environment_key>",
  "capacity":2
}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-ensure

# Execute helper-profile source-in/JSON-out workflow code.
@'{
  "profile":"helper",
  "environment_key":"<environment_key>",
  "request":{
    "request_id":"req-123",
    "module_source":"def condition(input):\n    return {\"accepted\": True}\n",
    "module_sha256":"<sha256>",
    "package_id":"pkg",
    "workflow_id":"workflow",
    "package_source_digest":"<digest>",
    "operation":"condition",
    "payload":{},
    "limits":{"timeout_ms":5000,"output_limit_bytes":65536}
  }
}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-execute
```

Dependency-bearing Python environments are host-managed. Use
`workflow-python-prepare-environment`, `workflow-python-lock-environment`,
`workflow-python-verify-environment`, `workflow-python-install-environment`
when explicitly allowed, and `workflow-python-verify-install-receipt` before
depending on installed packages. The stable `install_status` field summarizes
that lifecycle.

Resource and request operations are keyed by `environment_key`:

```powershell
@'{"profile":"helper","environment_key":"<environment_key>"}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-resources
@'{"profile":"helper","environment_key":"<environment_key>","capacity":4}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-set-capacity
@'{"profile":"helper","environment_key":"<environment_key>","request_id":"req-123"}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-request-status
@'{"profile":"helper","environment_key":"<environment_key>","request_id":"req-123"}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-cancel-request
```

Workflow Python node profile uses the stable node response envelope. The sync
path is `workflow-python-execute` with `profile=node`; the streaming path is
`workflow-python-stream-open`, `workflow-python-stream-recv`,
`workflow-python-stream-send`, and `workflow-python-stream-close`.
Stream-open returns immediately and background execution emits `started`,
`log`, optional `progress`, `result` or structured `error`, and `done`.

Workflow JS node profile exposes the same environment-keyed management shape:
`workflow-js-environment-spec`, `workflow-js-ensure`, `workflow-js-execute`,
`workflow-js-stream-open`, `workflow-js-stream-recv`, `workflow-js-stream-send`,
`workflow-js-stream-close`, `workflow-js-resources`, `workflow-js-set-capacity`,
`workflow-js-request-status`, and `workflow-js-cancel-request`.

### 2.6 Proxying Worker Requests (RPC & Streams)

Once a traffic session is issued, developers can test proxy commands.
Use `proxy-rpc-open`/`proxy-rpc-recv`/`proxy-rpc-close` for streamed
`run-inference`. Do not retry an empty streamed completion through
`proxy-rpc-call`; an empty terminal stream without chunk text, `response`, or
`final_response` is a relay/runtime error that should be surfaced.

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

### 2.6 Workflow Runtime Workers

Workflow helpers are hosted `generic` workers behind workflow runtime facades. Use these instead of backend-owned local helper subprocesses:

```powershell
@'{"profile":"helper","environment_name":"workflow-python-helper","capacity":2,"session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-ensure
@'{"profile":"helper","environment_key":"<environment_key>","session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-resources
@'{"profile":"helper","environment_key":"<environment_key>","capacity":4,"session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-set-capacity
@'{"profile":"helper","environment_key":"<environment_key>","request_id":"req-1","session_token":"<control_token>"}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-cancel-request
```

Execute Python helper-profile code through the workflow facade:

```powershell
@'{
  "profile":"helper",
  "environment_key":"<environment_key>",
  "request":{
    "request_id":"req-1",
    "module_source":"def condition(input):\n    return {\"accepted\": input[\"value\"] == 7}\n",
    "module_sha256":"<sha256-of-module_source>",
    "export_name":"condition",
    "operation":"condition",
    "payload":{"value":7},
    "limits":{"timeout_ms":5000,"output_limit_bytes":65536,"memory_limit_mb":128},
    "python":{"import_allowlist":[],"package_pins":{},"environment_name":"workflow-python-helper"}
  },
  "session_token":"<control_token>"
}'@ | python -m hosting.engine_host_cli --payload-stdin workflow-python-execute
```

Workflow JavaScript uses the QuickJS-backed node facade: `workflow-js-ensure`, `workflow-js-execute`, `workflow-js-stream-open`, `workflow-js-stream-recv`, `workflow-js-stream-send`, `workflow-js-stream-close`, `workflow-js-resources`, `workflow-js-set-capacity`, and `workflow-js-cancel-request`. JS requests use `profile:"node"` and a single-script contract such as `exports.run = function(input, api) { return {output: input}; };`. The host verifies `module_sha256` before execution and exposes filesystem, HTTP, codec, crypto, console, and progress behavior only through explicit host APIs. Clients that author with imports can use `hosting.sandbox.build_workflow_js_bundle(...)` to patch enabled `@host/...` bridge imports and inspect disabled or unresolved imports before submitting the single script. `host_call_id` values are scoped to a worker/request IPC conversation and are not global daemon-channel routes.

## 3. Diagnostics and Auditing

Diagnostics and auditing are heavily utilized during development and operations.

### 3.1 Hosting Access Doctor
Run the hosting access doctor for file/config/key custody checks:

```powershell
python -m hosting.hosting_config_cli --no-interactive --doctor
```

### 3.2 Host Metrics
`host-metrics` provides process-runtime diagnostics, including in-flight proxy requests, proxy byte totals, auth denial counters, caller auth status for the supplied session token, and a recent request ring buffer.

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

`auth-list-sessions` returns session metadata and token previews only; it does
not disclose bearer tokens. Use `auth-validate-session` to check a token already
held by the client/channel before reusing or adopting it.

## 4. Current Limitations & Implementation Notes

- **IPC Transport Only**: Worker transport is local IPC only. There is no host-managed remote worker transport.
- **HTTP Bridge**: `proxy-request` serves as a compatibility bridge for HTTP-like engine routes over IPC. New generic worker integrations should prefer `proxy-rpc-*` for sync/async RPC.
- **Sandbox Boundary**: The implemented sandbox foundation is strongest through host-brokered filesystem/HTTP APIs and Windows restricted-token/low-integrity launch. POSIX workers currently launch as plain subprocesses; direct network policy is not an OS-level filter.
- **Metrics Scope**: Metrics are per-process runtime and are not persisted across daemon restarts.
- **Bootstrap Credentials**: Local-only shared-secret bootstrap requires `engine_host_key_id` + `engine_host_key_secret` or a pre-issued `engine_host_session_token`. SSH relay and remote-capable profiles require public-key challenge/transport setup instead of shared-secret session issuance.
