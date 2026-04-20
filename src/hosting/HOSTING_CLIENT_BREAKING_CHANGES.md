# Hosting client breaking changes

The hosting API is being finalized before the next consumer project ships. The following pre-release compatibility paths and custom formats are being removed from the client contract.

## Removed Python import paths

Do not import these modules:

- `hosting.engine_host_service`
- `hosting.engine_host_daemon`

Use the documented hosting CLI/config flows, public `hosting` package exports, or canonical internal modules such as `hosting.service.host_service` and `hosting.daemon`.

## Removed monkeypatch targets

Tests and integrations must not patch through:

- `hosting.engine_host_service.*`
- `hosting.engine_host_daemon.*`

Patch the canonical module that owns the symbol instead.

## Removed legacy key-file migration

Hosting setup no longer auto-migrates:

- `backend/host_auth_keys.json`
- `backend/engine_host_auth_keys.json`

Run the current setup flow and use the current access-control, keyring, client mapping, and bootstrap artifacts.

## Removed daemon TCP fallback

Local daemon control uses PID-file local IPC metadata. Port-only local TCP fallback on `127.0.0.1` is not supported. The daemon also blocks the loopback TCP control listener server-side for now.

This does not remove SSH relay remote control. Remote consumers should use a forced-command transport key that runs `python -m hosting.engine_host_cli --relay-wrapper`. That key must not be granted PTY or shell access; configure it with `no-pty` and the other generated SSH restrictions. The relay runs on the target host and connects to the daemon through local IPC metadata.

Remote control always requires SSH to execute the relay wrapper. A daemon that is already running is not remotely controllable by itself, because daemon control is local IPC only.

If the daemon is already running, the wrapper attaches through PID-file local IPC metadata. If the daemon is not running, wrapper auto-start is only attempted when saved hosting config is remote-enabled, `require_auth=true`, at least one auth key is registered, and lifecycle is `detached_user_process`; otherwise remote control operations require the daemon to be started by some other approved path first.

Supported today: SSH relay via the forced-command wrapper, local IPC on the daemon host, and HTTP worker ingress through `--daemon-http`.

TBD: straight SSH port forwarding to a daemon TCP control listener. Consumers must not assume `DEFAULT_DAEMON_PORT` is connectable. Consumers that cannot execute any remote SSH command do not currently have a full remote control-plane transport. The `--daemon-http` process is HTTP worker ingress, not a replacement for the daemon control API.

## Removed custom `password_v1` encryption

The custom `password_v1` JSON encryption envelope is removed.

Unsupported after cleanup:

- Client secret records with `encryption: "password_v1"`.
- Transport bootstrap bundles with `transport_private_key_encryption: "password_v1"`.
- Transport bootstrap fields named `transport_private_key_password_v1`.

Private keys must be stored and exchanged as OpenSSH private keys. If password protection is required, use OpenSSH passphrase-protected private-key formatting.

Regenerate or re-import any client realm secrets and transport bootstrap bundles that used `password_v1`.

## Shared-secret verifier format is unchanged for now

Local-only shared-secret verifier hardening is deferred and is not part of this cleanup.

Consumers should still prefer public-key challenge auth for remote-capable profiles. If the verifier format changes before release, shared-secret records will need to be recreated then.

## Toolbox executor kind rename

`toolbox_executor_v1` is not guaranteed as the release executor kind.

Unless executor contracts are explicitly versioned for the first release, consumers should expect the release name to be `toolbox_executor` and must not hard-code `toolbox_executor_v1`.
