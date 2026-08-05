# Hosting Client Breaking Changes

This file is the authoritative handoff log for dependent projects. The hosting
client does not provide compatibility adapters, fallback signatures, or
parallel versioned APIs. Each dependent project must repin and apply every entry
whose parent commit is newer than its current pin.

## Unified hosted operations and capability leases

Change set: `HOSTING-OPERATION-CONTRACT`

Prepared against parent: `9895a98b8b7af7e4b248951d61b622c0c9c1caa3`

First parent commit containing break: `f4e4ec021e0e62485415ca376953fae9388f6e73`

Implementation release commit: `31a5b123fe4a7e554b1cf55cbb1f4ad8956bb85b`

Dependent project: `O:/repos/mp13-docs`

### HC-001 - Hosted execute/status/cancel contract replaced

Affected methods:

- toolbox, workflow Python, and workflow JavaScript execute methods;
- removed family-specific request-status methods; and
- removed family-specific cancel methods.

All execute methods now return `hosting.operation_status` with a typed
`hosting.operation_ref`. Status and cancel accept only that ref:

```python
started = channel.execute_workflow_python(request=request)
ref = started["operation"]
status = channel.hosted_operation_status(ref=ref)
canceled = channel.hosted_operation_cancel(ref=ref, reason="workspace_unload")
```

If transport loss hides the initial execute response, recover the canonical
ref/status through the authenticated request index, then persist the returned
ref and resume the normal ref-only APIs:

```python
status = channel.hosted_operation_resolve_request(
    execution_kind="workflow_python",
    selector={"kind": "engine_id", "id": resolved_engine_id},
    request_id=request_id,
)
ref = status["operation"]
```

This recovery lookup never probes or starts a worker. It is only for losing the
parent-minted ref; status and cancel themselves do not accept reconstructed
selectors.

Old selector reconstruction is unsupported:

```python
# Removed
channel.workflow_python_request_status(
    profile="node",
    environment_key=environment_key,
    request_id=request_id,
)
```

The canonical lifecycle field is `lifecycle`. Digests are
`sha256:<lowercase-hex>`, sizes are `size_bytes`, and terminal payloads use
exactly one of `result`, `result_ref`, or `result_omission`.

Required dependent changes:

- persist the returned operation ref rather than selector dictionaries;
- delete lifecycle normalization and method-name switching;
- delete signature inspection and `TypeError` fallback behavior; and
- use generic ref-only status/cancel for all three families.

### HC-002 - Receipt checkpoint schema replaced

The new parent rejects the legacy
`state/toolbox_execution_receipts.json` schema. It does not import or translate
legacy rows.

Before repinning:

1. Stop the old daemon.
2. Confirm no protected hosted operation remains inside the configured replay
   window.
3. Run `hosting-receipt-ledger-cutover` with the explicit acknowledgement flag
   documented by `src/hosting/HOSTING_OPERATION_CONTRACT.md`.
4. Verify the legacy file was archived, not deleted.
5. Start the replacement parent and run recovery smoke tests.

Rollback requires the previous parent commit and its matching archived ledger.
Never open a new-format ledger with the previous parent or vice versa.

### HC-003 - Host Capability provider identity is explicit

`provider_id` is required during registration and identifies the logical
provider. `session_id` identifies one registration instance; callers may
supply it or persist the parent-minted value returned by registration. The two
identities must differ, and the parent never derives `provider_id` from
`session_id`.

```python
channel.host_capability_session_register(
    provider_id="workspace.tools",
    session_id="workspace.tools.session-17",
    methods=methods,
    scope=scope,
    authority_lease={
        "expires_at_ms": expires_at_ms,
        "on_transport_loss": "retain_until_expiry",
        "on_authority_revoked": "close",
        "on_request_terminal": "close",
    },
)
```

Update list/close filters, audit projections, toolbox-provider registration,
and service-broker helpers to use the correct identity.

### HC-004 - Approval callback ownership moved to the parent client

Workflow execute, action-describe, action-execute, pinned-instance execute, and
stream-open accept exactly one of:

- `approval_requester=<callable>`;
- `approval_requester_binding=<binding>`; or
- `approval_callback_lease=<lease>`.

Delete dependent callback-relay creation/release code. For streams, retain the
returned stream handle and close it; the handle owns callback lease lifetime.
Contradictory callback arguments fail validation.

### HC-005 - Capability session lifetime descriptor replaced the disconnect boolean

`close_on_client_disconnect` is removed. Registration requires an explicit
authority lease descriptor:

```python
authority_lease = {
    "expires_at_ms": expires_at_ms,  # null means no expiry; zero is invalid
    "on_transport_loss": "close",  # or retain_until_expiry
    "on_authority_revoked": "close",
    "on_request_terminal": "close",  # or retain
}
```

The parent binds `owner_authority_id` from the authenticated actor. Use
`host_capability_session_renew(...)` and
`host_capability_session_revoke(...)`. Keep the protected
lease token returned at registration process-local; never persist it in a
workspace journal or send it to browser/UI state.

### Adoption verification

The dependent project must run its complete:

- hosted-operation recovery/replay/cancel suite;
- approval and allow-once deduplication suite;
- Host Capability registration/dispatch/audit/close suite;
- authority lease disconnect/expiry/revocation suite; and
- workflow Python and JavaScript execution suites.

Repin to the implementation release commit above or a descendant containing
it. Do not adopt only a subset of these breaking changes.
