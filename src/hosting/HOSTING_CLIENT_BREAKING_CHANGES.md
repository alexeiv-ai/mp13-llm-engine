# Hosting Client Breaking Changes

Date: 2026-03-29
Scope: local daemon control transport

## 1. Supported clients that should keep working

If your client uses the supported hosting entry points, no client-side code change should be needed:

1. `EngineHostControlChannel`
2. `LocalSocketConnection`
3. `python -m hosting.engine_host_cli`
4. `python -m hosting.engine_host_cli --relay`

These APIs now discover and use the correct local control transport internally.

## 2. What hosting API clients should use

To continue working across this change:

1. Use `EngineHostControlChannel` for normal daemon control from Python code.
2. Use `LocalSocketConnection` only through the hosting modules that already construct it from daemon state.
3. Use `python -m hosting.engine_host_cli` for local CLI control.
4. Use `python -m hosting.engine_host_cli --relay` for remote control over SSH.

Clients should not open their own raw control sockets or assume the daemon is reachable at `127.0.0.1:<daemon_port>`.

## 3. What changed under the hood

1. Local daemon control no longer depends on a TCP listener at `127.0.0.1:<daemon_port>`.
2. `LocalSocketConnection` now discovers local control transport from the daemon PID file and connects through local IPC:
   - Windows: named pipe (`AF_PIPE`)
   - Linux/macOS: Unix domain socket (`AF_UNIX`)
3. SSH relay remains the supported remote control path and now connects to the remote daemon through local IPC discovered from the remote PID file.

## 4. Who must update client code

You need client changes only if your code bypasses the hosting APIs and does one of these:

1. reads `hosting/state/daemon.pid` and assumes local control is always a TCP port
2. opens raw TCP connections to `127.0.0.1:<daemon_port>` for local daemon control
3. uses SSH forwarded-port control (`ssh -L ...`) to reach the daemon TCP control listener directly

## 5. PID file contract extension

1. Daemon PID files now include local control transport metadata:
   - `transport`
   - `ipc_family`
   - `ipc_address`
2. `shutdown_token` remains present and unchanged for `__shutdown__`.
3. `port` may still be present for compatibility and status surfaces, but clients must not assume it represents an active local daemon TCP control listener.

## 6. Migration summary for custom clients

1. For local control, switch to hosting APIs that resolve daemon transport from the PID file instead of dialing `127.0.0.1:<daemon_port>` yourself.
2. For remote control, switch to SSH relay (`python -m hosting.engine_host_cli --relay`) instead of forwarding a daemon TCP port.
3. If you parse `daemon.pid` directly, treat `transport`, `ipc_family`, and `ipc_address` as the control-endpoint discovery contract.
