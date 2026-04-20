# Hosting cleanup status

This file records the hosting cleanup decisions that should be completed before handing the hosting client contract to the next project. The next consumer project has not shipped yet, so compatibility shims and custom pre-release formats should be removed instead of preserved as public legacy behavior.

## Implemented removals before release

### Compatibility import shims

Removed these compatibility modules:

- `src/hosting/engine_host_service.py`
- `src/hosting/engine_host_daemon.py`

The implementations already live under:

- `hosting.service.host_service`
- `hosting.daemon`

Completed work:

- Move production imports to canonical modules.
- Move tests and monkeypatch targets to canonical modules.
- Deleted the shim files once no source or test imports depended on them.

Consumer-facing break:

- Consumers must not import `hosting.engine_host_service` or `hosting.engine_host_daemon`.

### Dynamic legacy monkeypatch lookups

Removed dynamic lookup hooks that read from the compatibility modules:

- `src/hosting/service/engines.py`: call `hosting.sandbox.launcher.launch_worker_process` directly.
- `src/hosting/service/proxy.py`: removed `_legacy_attr`; use local `os`, `tempfile`, and `MPClient`.
- `src/hosting/daemon/background.py`: removed `_legacy_daemon_attr`; use `DaemonPidFile` directly.
- `src/hosting/daemon/security.py`: removed `_legacy_daemon_attr`; call `_current_windows_account_name` directly.

Consumer-facing break:

- Tests or integrations must patch canonical modules, not `hosting.engine_host_service.*` or `hosting.engine_host_daemon.*`.

### Legacy key-file migration

Removed `_migrate_legacy_key_files` and setup/reporting references for old key filenames:

- `backend/host_auth_keys.json`
- `backend/engine_host_auth_keys.json`

Consumer-facing break:

- These old files are not auto-migrated. Consumers must run the current setup flow and use the current access-control, keyring, client mapping, and bootstrap artifacts.

### Daemon TCP control listener and fallback

Removed the local TCP fallback from `LocalSocketConnection` and blocked the daemon loopback TCP control listener server-side:

- `src/hosting/engine_host_connection.py`: removed `_connect_legacy_tcp`.
- Make missing PID-file local IPC metadata a hard connection error.
- `src/hosting/daemon/local_ipc.py`: `_should_enable_tcp()` returns `False`; straight SSH port forwarding to daemon control is TBD.

Consumer-facing break:

- Port-only local daemon connections are not supported. Local daemon control must use PID-file local IPC metadata.
- SSH relay remote control remains supported through a forced-command transport key that executes `python -m hosting.engine_host_cli --relay-wrapper`.
- A running daemon alone is not remotely controllable; SSH must be able to execute the wrapper because daemon control is local IPC only.
- Transport keys must not grant PTY or shell access; use `no-pty` and the generated forced-command hardening.
- Supported today: local IPC, SSH relay-wrapper remote control, and `--daemon-http` worker ingress.
- TBD: straight SSH port forwarding to a daemon TCP control listener.
- Consumers that cannot execute any remote SSH command do not currently have a full remote control-plane transport. The `--daemon-http` process is worker HTTP ingress, not daemon control-plane HTTP.

### Custom `password_v1` encryption scheme

Removed the custom `password_v1` envelope entirely.

Previous use:

- Client-realm private-key secret records could wrap private key text in `password_v1`.
- Transport bootstrap bundles could wrap transport private key text in `transport_private_key_password_v1`.

Decision:

- Do not keep the custom `scrypt` + HMAC-SHA256 XOR-stream envelope.
- Private keys are already OpenSSH private keys, and hosting already uses OpenSSH tooling as an external dependency.
- Password protection for private keys should use OpenSSH private-key encryption, not an app-specific JSON encryption envelope.

Completed work:

- Generate passphrase-protected keys through `ssh-keygen` when key protection is requested.
- For imported plaintext private keys, re-protect them through OpenSSH tooling before storing when a protection passphrase is requested.
- Store the OpenSSH private key text directly as the secret payload.
- Replaced secret record fields such as `encryption: "password_v1"` with metadata such as `private_key_format: "openssh"` and `private_key_protection: "openssh_passphrase"` or `"none"`.
- Replaced transport bootstrap fields `transport_private_key_encryption: "password_v1"` and `transport_private_key_password_v1` with an OpenSSH-encrypted `transport_private_key_openssh` payload plus protection metadata.
- Removed `_password_v1_encrypt`, `_password_v1_decrypt`, `_password_v1_keystream`, and the related allowed encryption value from `client_realm.py`.

Consumer-facing break:

- `password_v1` records and `transport_private_key_password_v1` bundle fields are not supported. Consumers must regenerate or re-import client realm secrets and transport bootstrap bundles using OpenSSH-formatted private keys.

### Shared-secret verifier hashing

Deferred for a later release decision. Keep this separate from private-key encryption. The shared-secret path stores a verifier for local-only session issuance; it is not an encryption envelope.

Current implementation:

- `AuthMixin._hash_secret` stores `sha256(secret)`.
- Verification uses `hmac.compare_digest`, but the stored verifier is not an HMAC and is not salted.
- Shared-secret session issuance is already blocked for remote-capable profiles.

Decision:

- Do not confuse this with OpenSSH private-key protection.
- Punt for this cleanup. Before release, either remove shared-secret auth entirely or replace the verifier with a salted password hashing scheme.

Preferred hardening if shared-secret auth remains:

- Store a versioned verifier record, not a raw SHA-256 digest.
- Use a salted KDF from the standard library, such as `hashlib.scrypt`, or an approved dependency if one is chosen for password hashing.
- Example metadata shape: `secret_verifier: {scheme, salt, params, digest}`.
- Keep `hmac.compare_digest` for constant-time digest comparison.

Future consumer-facing break if this changes:

- Existing shared-secret key records must be recreated after verifier format changes. Since the next consumer has not shipped, no compatibility migration is required.

### `toolbox_executor_v1`

Renamed `toolbox_executor_v1` before release because there is no concrete need for versioned executor protocol identifiers from day one.

Current assessment:

- There is only one active toolbox executor contract.
- No concrete `toolbox_executor_v2` requirement exists today.
- The suffix is only future-proofing.

Completed work:

- Chose the release name `toolbox_executor`.
- Updated writers, runtime checks, manifests, bundle models, tests, and docs in one change.

Consumer-facing break:

- Consumers must not send or expect `toolbox_executor_v1`; use the release executor kind.

## Cleanup order used

1. Move imports and tests from shim modules to canonical modules.
2. Remove dynamic legacy monkeypatch helpers.
3. Delete `engine_host_service.py` and `engine_host_daemon.py`.
4. Remove legacy key-file migration logic and docs.
5. Block daemon TCP control and remove TCP fallback.
6. Replace `password_v1` private-key protection with OpenSSH private-key encryption.
7. Deferred shared-secret verifier storage; it remains a separate release decision.
8. Renamed `toolbox_executor_v1` because versioned executor contracts are not needed for the first release.
9. Made generated private-key custody explicit in setup, doctor, RBAC key management, and client-realm helpers:
   - exported generated private-key files can be discovered from keyring metadata
   - exported files can be handed off into a local consumer client realm
   - client realm private-key secrets can be migrated between realms through API helpers
   - hand-off can delete the loose exported file and mark the source keyring with purge/hand-off metadata
   - exported files can be purged without hand-off only as an explicit warning-bearing action
   - inline private-key import accepts sanitized pasted OpenSSH key text and clears the sensitive argument after reading
10. Re-run the hosting test suite and update `HOSTING_CLIENT_BREAKING_CHANGES.md` with implementation-specific details.
