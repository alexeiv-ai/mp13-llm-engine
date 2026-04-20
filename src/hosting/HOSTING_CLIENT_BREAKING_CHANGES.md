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

Local daemon control uses PID-file local IPC metadata. Port-only local TCP fallback on `127.0.0.1` is not supported.

Consumers should use the current control channel/bootstrap configuration instead of assuming `DEFAULT_DAEMON_PORT` is connectable.

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
