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

## Hosting-generated private key custody is explicit

Generated admin private keys are no longer an implicit setup-side artifact.

Current setup behavior:

- If the operator exports the generated key, setup records `private_key_storage: "exported_file"` and `private_key_export_path`.
- If the operator does not export immediately, setup stores the private key in the setup machine's default client realm and prints a `--client-export-key` handoff command.
- Doctor reports a loose exported generated private-key file as a non-blocking custody warning with a recommendation to hand it off into a local consumer realm or export it for remote transfer, then purge the loose file.
- After hand-off with source-file deletion, key metadata records `private_key_export_purged_at` and `private_key_adopted_client_realm_root`; doctor no longer treats the missing exported file as an error.
- If the operator purges an exported file without recorded hand-off, metadata records `private_key_export_purged_without_adoption_at`; doctor keeps a warning because the private key may be lost unless another copy exists.

Consumer-facing adjustment:

- Do not depend on the setup machine's exported private-key file as durable storage.
- Discover exported key references with `--client-list-exported-keys`.
- For a local consumer on the same filesystem, move a generated exported key into the consumer realm with `--client-handoff-exported-key --client-key-id <id> --client-delete-exported-key-file`.
- For a remote consumer, export or transfer the private-key file out-of-band, then import it into the remote consumer's vault/client realm there.
- Purge a tracked exported file without importing it only with `--client-purge-exported-key`; this can lose the only private-key copy.
- The interactive `Manage RBAC keys` menu exposes list/export/hand-off/purge flows for generated private-key custody alongside key revocation and auth audit views.
- `--client-import-key` remains available as an operator/manual bridge, but consumer projects should prefer the client-realm API helpers for import and realm migration. The CLI normalizes quoted/literal-newline paste input and clears the inline private-key argument after reading it.

## Shared-secret verifier format is unchanged for now

Local-only shared-secret verifier hardening is deferred and is not part of this cleanup.

Consumers should still prefer public-key challenge auth for remote-capable profiles. If the verifier format changes before release, shared-secret records will need to be recreated then.

## Toolbox executor kind rename

`toolbox_executor_v1` is not guaranteed as the release executor kind.

Unless executor contracts are explicitly versioned for the first release, consumers should expect the release name to be `toolbox_executor` and must not hard-code `toolbox_executor_v1`.
