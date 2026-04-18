# Hosting Access Plan

Status: forward plan for remaining hosting access work after the current auth/authz baseline.

Use this file for the remaining implementation plan.
Do not use it as the implementation-aligned architecture document; that role belongs to `hosting_access.md`.

## 1. Goal

1. Add a guided non-local bootstrap flow that closes first-connect SSH MITM risk through explicit out-of-band provisioning.
2. Add a client-side hosting realm so hosting can manage client-local transport and RBAC identity artifacts without treating normal runtime RPC as a private-key retrieval channel.
3. Add file-based secret storage with tagged records and optional password encryption, while keeping the design swappable to future OS/platform secret backends.
4. Keep the current role model and SSH-bound session model intact; this plan is additive around bootstrap, storage, and client tooling.

## 2. Non-Goals

1. No daemon-specific cryptographic mutual-auth protocol in this phase.
2. No hardware-backed key custody in this phase.
3. No claim that local compromise of the client realm is fully prevented.
4. No requirement in the first cut to migrate daemon-owned runtime auth state or currently client-held runtime session tokens into the new client-realm backend.

## 3. Current Starting Point

1. SSH host-key pinning is now mandatory for SSH relay and restart paths.
2. Hosting already has a structured host-side file layout:
   - `access_control.json`
   - `keyring/keys.json`
   - `state/*.json`
   - `audit/*.json`
3. Client-realm foundation now exists:
   - dedicated client realm root/layout helpers
   - `client_access.json`
   - file-backed tagged secret records
   - client profile persistence
   - client-local audit event records
4. Transport bootstrap bundle helpers now exist at module level:
   - bundle creation/validation
   - file export/import helpers
   - client-realm import with strict `ssh_known_hosts_line` requirement
5. `hosting_config_cli` now exposes local-helper transport bootstrap export/import commands for file-based out-of-band transfer.
6. `hosting_config_cli` already supports:
   - public-key import
   - generated keypair creation
   - optional file export of generated private keys
   - metadata about key origin and private-key storage state
7. New generated-key steady state can use client-realm secret storage instead of embedding raw private keys in `keys.json`.
8. Password-encrypted secret records and password-encrypted transport bootstrap private-key payloads now exist through the `password_v1` file/envelope path.
9. Imported client profiles can now automatically feed `EngineHostControlChannel` construction paths through client-realm profile resolution and managed SSH-key materialization.
10. Current client control settings are plain values such as:
   - `control_ssh_key`
   - `ssh_known_hosts_line`
   - `control_ssh_fingerprint`
   - `engine_host_session_token`
11. Imported transport profiles can now be validated through a strict SSH probe path or a local consistency-only validation path.

## 4. Target Outcome

### 4.1 Client Realm

1. Hosting can maintain a client-local realm under a dedicated client root, separate from the remote host realm.
2. The client realm stores:
   - transport private-key references or protected blobs
   - RBAC private-key references or protected blobs
   - pinned SSH host-key material
   - client-local connection profiles and mapping metadata
   - bootstrap/import audit records
3. The client realm uses the same broad layout concepts as the host realm, but with client-local semantics.

### 4.2 Guided Transport Bootstrap

1. Remote hosting setup can generate a dedicated `transport` keypair.
2. Remote hosting setup can emit a bootstrap artifact for out-of-band transfer to the client.
3. The bootstrap artifact includes:
   - transport public key
   - transport private key or encrypted private-key payload
   - pinned SSH host key as a `known_hosts` line
   - target/host metadata required by the client profile
4. Client hosting tooling can import that artifact and configure SSH-related client settings without using `accept-new`.

### 4.3 File-Based Secret Storage

1. Secret records can be tagged by purpose:
   - `transport_private_key`
   - `rbac_private_key`
   - `session_token`
   - future tags as needed
2. Secret records support:
   - unencrypted storage with explicit warning/state
   - password-encrypted storage
   - future migration to OS secret backends without changing higher-level workflows
3. The storage contract is file based first, with clean backend seams.

## 5. Threat Model Focus For This Plan

1. Primary threat addressed:
   - first-connect MITM during initial non-local SSH client bootstrap
2. Secondary threats reduced:
   - accidental storage sprawl of transport/RBAC private keys
   - unclear provenance of imported/generated client-side private keys
   - accidental use of unpinned SSH host trust during setup
3. Explicitly not eliminated:
   - local theft of client-side secrets after client-host compromise
   - misuse of decrypted private keys during an active compromised session

## 6. Design Direction

### 6.1 Client Realm Layout

Use a client-local hosting root with a layout parallel to the host realm:

1. `client_access.json`
   - non-secret client realm config and profile metadata
2. `keyring/keys.json`
   - public-key metadata
   - provenance metadata
   - references to secret records, not raw plaintext private keys in the preferred steady state
3. `secrets/`
   - file-based secret records
4. `known_hosts/`
   - managed host-key pin material when stored as files
5. `audit/`
   - client import/export/bootstrap events
6. `profiles/`
   - named client connection profiles

Notes:
1. Reuse as much of the existing host metadata vocabulary as possible:
   - `key_origin`
   - `public_key_source`
   - `private_key_storage`
   - `private_key_export_path`
   - warning fields
2. Add realm-local identifiers so multiple remote hosts can coexist safely.

### 6.2 Secret Record Format

Each secret record should carry:

1. `version`
2. `secret_id`
3. `tag`
4. `realm`
5. `created_at`
6. `updated_at`
7. `encryption`
   - `none`
   - `password_v1`
8. `payload`
   - plaintext or encrypted bytes encoded for JSON/file transport
9. `metadata`
   - advisory fields only; no secret material

Password encryption requirements:

1. The current `password_v1` path uses a stdlib-only envelope built from `hashlib.scrypt` for KDF plus an authenticated ciphertext wrapper, so it can ship without adding a new dependency.
2. Store KDF parameters with the record.
3. Never persist the supplied password.
4. Permit future rewrapping into OS-native secret backends.
5. Keep the `password_v1` envelope versioned so it can later be replaced or migrated onto OS-native secret backends or a dedicated crypto library.

### 6.3 Transport Bootstrap Artifact

The bootstrap artifact should be exportable as:

1. a JSON bundle file for file transfer
2. a copy/paste terminal block for manual transfer

The artifact should contain:

1. `bundle_version`
2. `kind`
   - `hosting_transport_bootstrap`
3. `created_at`
4. `target`
5. `ssh_known_hosts_line`
6. `transport_key_id`
7. `transport_public_key`
8. one of:
   - `transport_private_key_openssh`
   - encrypted `transport_private_key_secret_ref` or `transport_private_key_ciphertext`
9. optional:
   - `control_ssh_fingerprint`
   - suggested client profile name
   - notes/warnings

Rules:

1. The bundle is explicitly an out-of-band bootstrap artifact, not a normal daemon RPC result for routine runtime use.
2. If the bundle contains private key material, it should default to password encryption for file export.
3. A copy/paste flow may allow plaintext only with an explicit operator confirmation.

### 6.4 SSH Provisioning Coupling

Remote-side setup responsibilities:

1. Generate or import the `transport` keypair.
2. Register the transport public key in hosting auth metadata.
3. Guide the operator to provision the public key into SSH server policy:
   - `authorized_keys`
   - restricted tunnel/relay account
4. Capture or request the exact SSH host-key line for the target host.
5. Refuse to produce a “MITM-safe” bootstrap artifact if the host key is not explicitly available.

Client-side import responsibilities:

1. Import and validate the bootstrap artifact.
2. Persist the pinned host key.
3. Persist the transport private key into the client realm secret storage or file export path.
4. Configure/update client profile fields:
   - `engine_host_ssh_target`
   - `control_ssh_key`
   - `ssh_known_hosts_line`
   - `control_ssh_fingerprint` when available
5. Run a strict SSH validation check before marking the profile ready.

## 7. Planned Phases

### Phase A: Client Realm And Storage Foundation

Scope:

1. Introduce a client-realm root resolver and layout helpers.
2. Introduce secret-record storage abstraction with file backend.
3. Add tagged secret record CRUD for local use.
4. Add audit records for client import/export actions.

Deliverables:

1. New client realm path helpers.
2. Secret storage module with interface shaped for future backend replacement.
3. Unit tests for record creation, lookup, tagging, updates, and deletion.

Exit criteria:

1. Client realm can persist tagged secrets and metadata without affecting current host realm behavior.
2. Existing host-side setup flows continue to work unchanged.

### Phase B: Key Metadata Refactor

Scope:

1. Refactor current generated-key handling so preferred steady state uses secret references instead of embedded private key strings.
2. Preserve compatibility with existing `private_key_openssh` entries during migration.
3. Add migration warnings and read-compatibility for old embedded-key rows.

Deliverables:

1. Extended key metadata schema.
2. Migration routine for legacy embedded key records.
3. Status/doctor output updates showing:
   - embedded legacy secret
   - external file
   - client-realm secret ref

Exit criteria:

1. Existing generated-key records remain readable.
2. New writes prefer file/secret-ref based storage.

### Phase C: Remote Transport Provisioning Export

Scope:

1. Add setup flow for dedicated `transport` key generation.
2. Add operator flow to produce a transport bootstrap artifact.
3. Require explicit SSH host-key capture as part of the export path intended to mitigate first-connect MITM.

Deliverables:

1. `hosting_config` or companion command to generate/export a transport bootstrap bundle.
2. Validation that the bundle includes a pinned `known_hosts` line.
3. Documentation for SSH server-side public-key installation and restrictions.

Exit criteria:

1. Operator can create a transport bootstrap artifact without using ad hoc file copying.
2. Export path clearly distinguishes:
   - MITM-safe bootstrap bundle
   - incomplete bundle missing host-key pinning

### Phase D: Client Transport Import And Profile Wiring

Scope:

1. Add client-side import command/API for transport bootstrap artifacts.
2. Store imported secrets in client realm.
3. Materialize/update client connection profile settings.
4. Validate strict SSH connectivity after import.

Deliverables:

1. Import command/API.
2. Profile update logic for `EngineHostControlChannel` settings, including managed SSH-key materialization from client-realm secret records.
3. Validation command that proves strict host-key verification is active.

Exit criteria:

1. Client can connect to the target host without any `accept-new` behavior.
2. Imported profile works for relay and restart flows.

### Phase E: RBAC Client-Key Lifecycle

Scope:

1. Extend client realm support to RBAC private keys.
2. Add generate/import/export helpers for client-side RBAC identities.
3. Keep remote hosting auth model public-key centered.

Deliverables:

1. Client-local RBAC key generation/import flow.
2. Public-key registration helper for remote host realm.
3. Status/doctor reporting for RBAC client-key storage state.

Exit criteria:

1. Client tooling can manage RBAC private-key custody locally without requiring normal runtime RPC to return private keys.
2. Transport and RBAC key lifecycles share one storage model with different tags.

## 8. API And CLI Surface

Planned additions:

1. Client realm helpers:
   - resolve client realm root
   - list profiles
   - show profile status
2. Secret storage helpers:
   - put secret
   - get secret
   - re-encrypt secret
   - delete secret
3. Transport bootstrap:
   - export bootstrap bundle
   - import bootstrap bundle
   - validate imported profile
4. RBAC client key management:
   - generate client key
   - import client key
   - export client key
   - register public key with remote hosting

API design rules:

1. Prefer local-helper and setup APIs over normal daemon command paths for private-key handling.
2. Normal runtime daemon RPC must not become the path that hands private keys back to clients.
3. Any password input is process-local and ephemeral only.

## 9. Security Rules

1. The default safe path for exported bootstrap bundles that contain private key material is password-encrypted file export.
2. Copy/paste plaintext bootstrap output must require explicit confirmation and should be marked operator-action-required.
3. Host-key pinning is mandatory for any bootstrap flow that claims first-connect MITM resistance.
4. `transport` keys remain transport-only in RBAC semantics.
5. The client realm should use restrictive filesystem permissions by default.
6. Client realm diagnostics should flag:
   - plaintext secret records
   - legacy embedded keys
   - missing expected exported files
   - bootstrap bundles imported without host-key pinning

## 10. Testing Plan

Unit coverage:

1. Client realm path/layout helpers.
2. Secret record encode/decode and password-encryption round-trips.
3. Legacy embedded-key migration.
4. Bootstrap bundle export/import validation.

Integration coverage:

1. Remote transport key generation and export.
2. Client import into local realm.
3. Strict SSH relay connection using imported profile.
4. Restart helper path using imported profile.

Negative-path coverage:

1. bundle missing `ssh_known_hosts_line`
2. wrong password for encrypted bundle
3. tampered encrypted bundle
4. tampered host-key pin
5. mismatched target host metadata
6. import into existing profile with conflicting pinned host key

Manual validation:

1. copy/paste bootstrap path
2. password-encrypted file transfer path
3. Windows host-path behavior
4. Linux/macOS host-path behavior

## 11. Migration Notes

1. Existing host-side `keys.json` rows with `private_key_openssh` must remain readable.
2. New client-realm work should not require immediate migration of every existing installation.
3. Migration should be opportunistic:
   - read old embedded record
   - offer export or import into client realm
   - rewrite metadata to secret reference when completed

## 12. Open Decisions

1. Whether the current stdlib `password_v1` envelope should later be replaced by a dedicated crypto library or OS-native protection layer.
2. Whether bootstrap bundle encryption should remain aligned with the same `password_v1` envelope or split into a transfer-specific format later.
3. Whether client profiles live inside hosting realm files or remain external config objects with hosting-managed references.
4. Whether transport bootstrap export also writes SSH config snippets or leaves that strictly optional.
5. Whether plaintext copy/paste export is allowed by default in interactive mode or must always be explicitly enabled.

## 13. Recommendation Order

1. Phase A first: client realm + storage abstraction.
2. Phase B second: stop treating embedded private keys as preferred steady state.
3. Phase C and D next: out-of-band MITM-safe transport bootstrap.
4. Phase E last: generalize the same machinery to RBAC client keys.

This ordering keeps the first-connect MITM fix aligned with a usable custody model instead of adding another ad hoc private-key export path.
