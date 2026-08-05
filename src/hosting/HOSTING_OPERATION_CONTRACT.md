# Hosted Operation and Capability Contract

Status: authoritative implementation contract

This document freezes the decisions required by
`src/hosting/hosting_access_plan.md`. The replacement API is deliberately
breaking and unversioned. Superseded signatures and persisted schemas are not
accepted.

## Hosted operation identity

- `operation_id` is a parent-minted URL-safe opaque identifier with at least
  144 bits of randomness. It is globally unique within a hosting root and is
  persisted before dispatch permission is returned.
- `(owner_actor_id, receipt_namespace, request_id)` is the idempotency key.
  `operation_id` and that tuple are unique repository keys. Including the owner
  prevents one actor from probing or conflicting with another actor's request
  identity.
- The repository stores execution kind, resolved selector, fingerprint, and
  authenticated owner actor. The selector in a client-supplied ref is
  descriptive and must exactly match the stored selector; it is never used as
  unverified routing input.
- A tombstone retains the complete operation ref, fingerprint, owner, terminal
  digest, and forgotten time until tombstone expiry.

`HostedOperationRef.from_dict` is the only accepted mapping parser at public
boundaries. It rejects unknown fields, invalid contracts, non-canonical
digests, unsupported execution/selector kinds, control characters, and values
over the limits exported by `hosting.operation_contract`.

## Lifecycle and terminal payload

`api_status` reports whether the status/cancel call itself succeeded.
`lifecycle` reports durable operation truth. The lifecycle enum is defined by
`HostedOperationLifecycle` and no family-specific aliases are returned.

Terminal result fields are mutually exclusive:

- `result` contains canonical JSON no larger than 64 KiB;
- `result_ref` contains an authorized, retrievable result artifact; or
- `result_omission` contains only canonical digest, `size_bytes`, and reason.

All digests use `sha256:<lowercase-hex>`. All sizes use `size_bytes`. Reasons
are limited to 512 UTF-8 bytes. Timestamps are non-negative Unix milliseconds.

## Result retention policy

The default policy is:

- redaction happens before digesting or storing;
- inline canonical JSON is retained up to 64 KiB;
- larger redacted JSON may be retained as an artifact up to 16 MiB when
  `retain_terminal_result` is explicitly true in effective hosting policy;
- absent/false policy, non-JSON values, values over 16 MiB, or artifact-store
  failure produce a digest-only omission;
- result artifacts are multi-read by the owning actor until their receipt is
  forgotten; tombstone transition deletes the artifact;
- orphan cleanup may delete an artifact only when no live receipt references
  it; and
- callback bindings, credentials, raw worker memory, host paths, and unbounded
  arguments are never stored.

Artifact retrieval verifies authenticated ownership, operation/ref linkage,
expiry, maximum returned size, and digest. Public refs never expose host paths.

## Canonical fingerprints

All families hash canonical JSON with sorted keys and compact separators and
return `sha256:<hex>`. Callback/approval bindings, credentials, timestamps,
transport metadata, and generated operation IDs are excluded.

Toolbox fingerprint input:

```text
execution_kind
resolved selector
tool name
tool arguments
effective tools view
effective host API approval policy
registration sandbox policy and sandbox profile ID
effective concurrency policy
```

Workflow Python fingerprint input:

```text
execution_kind
resolved engine selector
profile and derived environment specification/key
normalized request, including action and pinned instance identity
effective Python/runtime configuration
effective sandbox policy
capacity where it changes dispatch behavior
```

Workflow JavaScript fingerprint input:

```text
execution_kind
resolved engine selector
profile and derived environment specification/key
normalized request, including action and pinned instance identity
effective node and JavaScript configuration
effective sandbox policy
capacity where it changes dispatch behavior
```

Test vectors live in `tests/test_hosting_operation_contract.py` and are the
authority for canonical serialization changes.

## Repository interface and JSON persistence

The storage-neutral repository exposes these operations:

- `prepare(identity, fingerprint, metadata)` returns exactly one of dispatch,
  attach, replay, conflict, forgotten, or capacity;
- `mark_dispatch_claimed(operation_id)` performs queued-to-running compare and
  set;
- `finish(operation_id, lifecycle, terminal)` performs one terminal transition;
- `cancel_before_dispatch(operation_id, reason)` performs a queued/interrupted
  compare and set;
- `get_by_operation_id(operation_id)` and
  `get_by_request(owner_actor_id, namespace, request_id)` perform indexed reads;
- `wait_for_terminal(operation_id, timeout)` waits without dispatching; and
- `prune()` deterministically retains bounded receipts and tombstones.

The production implementation remains a process-locked JSON checkpoint. Every
mutation is serialized under the repository lock and committed by writing a
temporary file, flushing and fsyncing it, then atomically replacing the target.
In-memory indexes are rebuilt and validated at load. Load, status, attach, and
replay must not discover or start a worker or sandbox.

An unknown schema, malformed row, duplicate key, interrupted cutover marker, or
unreadable checkpoint fails closed. The parent does not translate legacy
schemas.

## Authorization matrix

| Operation | Owning authenticated actor | Different actor | Administrative force |
| --- | --- | --- | --- |
| Execute/attach/replay | Allowed when idempotency identity matches | Conflict/not found without disclosure | Allowed only through existing admin claim policy |
| Status | Allowed after stored-ref equality check | Not found without disclosure | Read allowed and audited |
| Cancel | Allowed after stored-ref equality check | Not found without disclosure | Force cancel allowed and audited |
| Result retrieval | Allowed while retained | Not found without disclosure | Read allowed and audited |
| Session list/close | Own sessions only | Denied | Allowed and audited |
| Authority renew/revoke | Same actor plus protected lease token | Denied | Force revoke allowed and audited |
| Administrative inspection | Not implied by ownership | Denied | Existing admin claim required and audited |

Unknown IDs and unauthorized IDs intentionally have the same public response.
Authorization is resolved from the stored actor, never a caller-supplied owner
field.

If the synchronous execute response is lost after the parent persisted an
operation, `hosted_operation_resolve_request` performs an authenticated indexed
lookup by execution kind, resolved selector, and request ID and returns the
stored canonical ref/status. It never probes or starts workers. Once recovered,
all status, result, and cancel calls remain ref-only.

## Provider and session identity

`provider_id` identifies the logical provider. `session_id` identifies one
registration instance. Both are required and must remain distinct through
registration, discovery, dispatch, approval, audit, filtering, and close.
Duplicate provider/session combinations or contradictory identities fail
registration. No fallback derives one from the other.

## Approval callback identity and lifetime

Within one logical hosted request, retries of the same provider call reuse its
`provider_call_id`; retries of the associated approval reuse `approval_id`.
New logical calls mint new IDs. An allow-once decision is keyed by the stable
IDs and cannot authorize a second logical call.

A direct callable is bound before daemon invocation. A synchronous lease closes
once on return or error. A stream owns its lease until terminal event, explicit
close/cancel, failed open, or channel shutdown. Close is thread-safe and
idempotent. Exactly one of callable, pre-bound binding, or lease is accepted.

## Capability authority lease

- `expires_at_ms: null` means no expiry; `0` is invalid.
- `on_transport_loss` is `close` or `retain_until_expiry`.
- `on_authority_revoked` is `close`.
- `on_request_terminal` is `close` or `retain`.
- Retain-on-loss requires a finite expiry or successful explicit renewal.
- `owner_authority_id` is bound to the registering authenticated actor.
- Registration returns a protected lease token once. Only a digest is retained
  by the parent; clients keep the token process-local and never place it in
  descriptors, receipts, logs, or artifacts.
- Renew/revoke requires the same actor and token. Administrative force revoke
  uses the existing admin claim path and is audited.

Request terminal means the durable operation reaches success, failure,
cancellation, or an interrupted terminal policy state. Sessions are in-memory
and do not survive daemon restart; no persistence capability is promised.

## Breaking cutover and rollback

The change is prepared against parent base
`9895a98b8b7af7e4b248951d61b622c0c9c1caa3`. The first breaking commit and the
implementation release commit are recorded in
`HOSTING_CLIENT_BREAKING_CHANGES.md`.

Before starting the replacement parent against an existing hosting root:

1. Stop the daemon and verify no protected operation remains within the replay
   window.
2. Run the explicit ledger cutover command with the required acknowledgement.
3. Confirm the legacy ledger was archived and the new ledger path is absent.
4. Start the new parent and repin the dependent project to its release commit.
5. Apply every change listed in `HOSTING_CLIENT_BREAKING_CHANGES.md` and run the
   dependent recovery/approval/session suites.

Rollback stops the daemon, restores the previous parent commit, archives the
new-format ledger, and restores the matching archived legacy ledger. There is
no runtime API fallback, dual-format reader, or automatic data migration.
