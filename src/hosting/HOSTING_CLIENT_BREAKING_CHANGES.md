# Hosting Client Breaking Changes

Date: 2026-06-22

## Host Capability Approval Requester Relay

Workflow Python/JS Host Capability approvals can now be handled through public
execution APIs.

In-process service clients may pass `approval_requester=callback` to:

- `execute_workflow_python(...)`
- `workflow_python_instance_execute(...)`
- `workflow_python_stream_open(...)`
- `execute_workflow_js(...)`
- `workflow_js_instance_execute(...)`
- `workflow_js_stream_open(...)`

Daemon/control-channel clients must use a relay binding instead of passing a raw
callable:

```python
from hosting import HostCapabilityApprovalCallbackRelay, host_capability_approval_decision

relay = HostCapabilityApprovalCallbackRelay()
binding = relay.bind_callback(
    lambda request: host_capability_approval_decision(
        "allow_once",
        approval_id=request["approval_id"],
    )
)

channel.execute_workflow_python(
    profile="node",
    request=request,
    approval_requester_binding=binding,
)
```

The approval callback receives a normalized
`hosting.sandbox.host_capability_approval.v1` payload. It includes method,
provider, approval policy, context, correlation ids, and `argument_keys`; it
does not include raw `arguments`.

Return `host_capability_approval_decision("deny" | "allow_once" |
"add_to_scope", ...)`. Stream-open starts execution immediately, so pass the
approval requester binding on `workflow_*_stream_open(...)` when gated calls may
happen before event subscription.

Audit and approval stream events are still emitted whether the decision is
approved or denied.

## Persistent Module Instance State

Pinned Python and JavaScript module instances now support explicit in-process
module state:

```json
{
  "instance_state_mode": "persistent_module"
}
```

Client requirements:

1. Use `persistent_module` only with explicit `workflow-python-instance-*` or
   `workflow-js-instance-*` routing.
2. Use it only for Python `execution_mode="module"` or JavaScript script/module
   requests. It is not valid for snippets or project mode.
3. Treat module globals, JS globals, singletons, caches, and open handles as
   process-local state. They survive sequential calls on the same live instance,
   but are lost on close, crash, cancellation that kills the process, or
   replacement.
4. Submit edited code with `workflow-*-instance-create(..., replace=True)`.
   Executing changed code directly against a persistent module instance returns
   `workflow_python_instance_code_replacement_required` or
   `workflow_js_instance_code_replacement_required`.
5. Treat `runtime_key` and `code_key` as opaque diagnostics. Do not parse their
   internal pipe-delimited shape.

Default behavior remains request-global ephemeral: a warm worker process may be
reused, but Python module globals or JavaScript QuickJS globals are not
preserved unless `persistent_module` is explicitly requested.

## Recovery Pattern Clarification

The recommended edit+continue recovery model is instance-scoped artifact refs, not old-path remapping.

Clients should:

1. Keep using the same logical `instance_id`.
2. Claim recovered artifacts without `target_id` when an `instance_id` is available.
3. Continue with refs under `@artifacts/instances/<instance_id>/...`.
4. Treat `old_path_to_new_path` and `old_path_to_new_ref` in raw claim responses as low-level diagnostics or migration aids only.

Use old-path mappings only when client-owned metadata already persisted absolute worker-local paths and must be patched. New client flows should persist host artifact refs instead.
